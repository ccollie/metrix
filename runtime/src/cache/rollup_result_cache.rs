use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use ahash::AHashSet;
/// import commonly used items from the prelude:
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use tracing::span::EnteredSpan;
use tracing::{field, info, span_enabled, trace_span, Level, Span};
use xxhash_rust::xxh3::Xxh3;
use metricsql_common::hash::{FastHasher, Signature};
use metricsql_common::prelude::{get_pooled_buffer, AtomicCounter, RelaxedU64Counter};
use metricsql_common::types::Label;
use metricsql_parser::ast::Expr;
use metricsql_parser::prelude::Matchers;

use crate::cache::default_result_cache_storage::DefaultResultCacheStorage;
use crate::cache::serialization::{compress_series_slice, deserialize_series_between};
use crate::cache::traits::RollupResultCacheStorage;
use crate::common::encoding::{marshal_var_int, marshal_var_usize, read_i64, read_u64, read_usize};
use crate::common::memory::memory_limit;
use crate::common::memory_limiter::MemoryLimiter;
use crate::execution::EvalConfig;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::{assert_identical_timestamps, SeriesSlice, Timestamp, Timeseries, TimestampTrait, MetricName};

/// The maximum duration since the current time for response data, which is always queried from the
/// original raw data, without using the response cache. Increase this value if you see gaps in responses
/// due to time synchronization issues between this library and data sources. See also
/// -provider.disableAutoCacheReset
/// TODO: move to EvalConfig
static CACHE_TIMESTAMP_OFFSET: Duration = Duration::from_secs(5);
static ROLLUP_RESULT_CACHE_KEY_PREFIX: OnceLock<u64> = OnceLock::new();

fn get_rollup_result_cache_key_prefix() -> u64 {
    *ROLLUP_RESULT_CACHE_KEY_PREFIX.get_or_init(|| {
        // todo: some sort of uid
        let mut rng = rand::rng();
        rng.random()
    })
}

fn get_default_cache_size() -> u64 {
    // todo: tune this
    let mut n = memory_limit().unwrap() / 16;
    if n < 1024 * 1024 {
        n = 1024 * 1024;
    }
    n
}

#[derive(Clone, Default)]
pub struct RollupCacheStats {
    pub full_hits: u64,
    pub partial_hits: u64,
    pub misses: u64,
}

struct Inner {
    cache: Box<dyn RollupResultCacheStorage + Send + Sync>,
    stats: RollupCacheStats,
    hasher: Xxh3,
}

pub struct RollupResultCache {
    inner: Mutex<Inner>,
    memory_limiter: MemoryLimiter,
    max_marshaled_size: u64,
    cache_key_suffix: RelaxedU64Counter,
    pub full_hits: RelaxedU64Counter,
    pub partial_hits: RelaxedU64Counter,
    pub misses: RelaxedU64Counter,
}

impl Default for RollupResultCache {
    fn default() -> Self {
        let size = get_default_cache_size();
        Self::with_size(size as usize)
    }
}

impl RollupResultCache {
    // todo: pass in cache

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_size(max_size: usize) -> Self {
        let mut rng = rand::rng();

        let cache = Box::new(DefaultResultCacheStorage::new(max_size));
        let hasher = Xxh3::default();
        let suffix: u64 = rng.random_range((1 << 16)..(1 << 31));
        let memory_limiter = MemoryLimiter::new(max_size);

        let inner = Inner {
            cache,
            stats: Default::default(),
            hasher,
        };

        Self {
            inner: Mutex::new(inner),
            memory_limiter,
            max_marshaled_size: max_size as u64 / 4_u64,
            cache_key_suffix: RelaxedU64Counter::new(suffix),
            full_hits: Default::default(),
            partial_hits: Default::default(),
            misses: Default::default(),
        }
    }

    pub fn reserve_memory(&self, size: usize) -> bool {
        self.memory_limiter.get(size)
    }

    pub fn release_memory(&self, size: usize) -> RuntimeResult<()> {
        self.memory_limiter.put(size)
    }

    pub fn memory_limit(&self) -> usize {
        self.memory_limiter.max_size
    }

    pub fn get_series(
        &self,
        ec: &EvalConfig,
        expr: &Expr,
        window: Duration,
    ) -> RuntimeResult<(Option<Vec<Timeseries>>, i64)> {
        let is_tracing = span_enabled!(Level::TRACE);

        let span = if is_tracing {
            let mut query = expr.to_string();
            query.truncate(300);
            let window = window.as_millis() as u64;
            let step = ec.step.as_millis() as u64;
            trace_span!(
                "rollup_cache::get_series",
                query,
                start = ec.start,
                end = ec.end,
                step,
                series = field::Empty,
                window
            )
        } else {
            Span::none()
        }
        .entered();

        if !ec.may_cache() {
            info!("did not fetch series from cache, since it is disabled in the current context");
            return Ok((None, ec.start));
        }

        // Obtain tss from the cache.
        let mut meta_info_buf = get_pooled_buffer(1024);

        let mut inner = self.inner.lock().unwrap();

        let res = self.get_cache_metadata(&mut inner, ec, expr, window)?;
        if res.is_none() {
            info!("not matching metadata found in the cache");
            return Ok((None, ec.start));
        }
        let (mut mi, hash) = res.unwrap();
        let key = mi.get_best_key(ec.start, ec.end)?;
        if key.prefix == 0 && key.suffix == 0 {
            // todo: add start, end properties ?
            info!("nothing found in the timeRange");
            return Ok((None, ec.start));
        }

        let mut bb = get_pooled_buffer(64);
        key.marshal(&mut bb);

        let mut compressed_result_buf = get_pooled_buffer(2048);

        if !inner
            .cache
            .get_big(bb.as_slice(), &mut compressed_result_buf)
            || compressed_result_buf.is_empty()
        {
            mi.remove_key(key);
            mi.marshal(&mut meta_info_buf);

            let hash_key = hash.to_ne_bytes();

            inner.cache.set(&hash_key, meta_info_buf.as_slice());

            info!("missing cache entry");
            return Ok((None, ec.start));
        }
        // we don't need the cache past this point
        drop(inner);

        // Decompress into newly allocated byte slice
        info!(
            "load compressed entry from cache with size {} bytes",
            compressed_result_buf.len()
        );

        // Extract values for the matching timestamps
        let tss = deserialize_series_between(
            &compressed_result_buf,
            ec.start,
            ec.end,
        ).map_err(|err| {
            let msg = format!("BUG: cannot deserialize from RollupResultCache: {:?}; it looks like it was improperly saved", err);
            RuntimeError::SerializationError(msg)
        })?;

        info!("unmarshal {} series", tss.len());

        if tss.is_empty() {
            info!("no timeseries found in the cached series on the given timeRange");
            return Ok((None, ec.start));
        }

        let timestamps = tss[0].timestamps.as_slice();
        if timestamps.is_empty() {
            // no matches.
            info!("no data-points found in the cached series on the given timeRange");
            return Ok((None, ec.start));
        }

        // is this right ??  - cc
        if timestamps[0] != ec.start {
            // The cached range doesn't cover the requested range.
            info!("cached series don't cover the given timeRange");
            return Ok((None, ec.start));
        }
        let last = timestamps[timestamps.len() - 1];

        let new_start = last.add(ec.step);

        if is_tracing {
            let start_string = ec.start.to_rfc3339();
            let end_string = new_start.sub(ec.step).to_rfc3339();
            span.record("series", tss.len());

            // todo: store as properties
            info!(
                "return {} series on a timeRange=[{}..{}]",
                tss.len(),
                start_string,
                end_string
            );
        }

        Ok((Some(tss), new_start))
    }

    pub fn put_series(
        &self,
        ec: &EvalConfig,
        expr: &Expr,
        window: Duration,
        tss: &[Timeseries],
    ) -> RuntimeResult<()> {
        let is_tracing = span_enabled!(Level::TRACE);
        let span = if is_tracing {
            let mut query = expr.to_string();
            query.truncate(300);
            
            let window = window.as_millis() as u64;
            let step = ec.step.as_millis() as u64;
            
            trace_span!(
                "rollup_cache::put_series",
                query,
                start = ec.start,
                end = ec.end,
                step,
                series = field::Empty,
                window
            )
        } else {
            Span::none()
        }
        .entered();

        if tss.is_empty() {
            info!("nothing to store in the cache");
            return Ok(());
        }

        if !ec.may_cache() {
            info!("do not store series to cache, since it is disabled in the current context");
            return Ok(());
        }

        if tss.len() > 1 {
            // Verify whether tss contains series with duplicate naming.
            // There is little sense in storing such series in the cache, since they cannot be merged in mergeSeries() later.
            let mut map: AHashSet<u64> = AHashSet::with_capacity(tss.len());
            for ts in tss {
                let hash = metric_name_hash_sorted(&ts.metric_name);
                if !map.insert(hash) {
                    let msg = format!("BUG: cannot store series in the cache, since they contain duplicate metric names: {}", ts.metric_name);
                    return Err(RuntimeError::DuplicateMetricLabels(msg));
                }
            }
        }

        // Remove values up to currentTime - step - CACHE_TIMESTAMP_OFFSET,
        // since these values may be added later.
        let timestamps = tss[0].timestamps.as_slice();
        let deadline = Timestamp::now() - (ec.step.as_millis() as i64) - (CACHE_TIMESTAMP_OFFSET.as_millis() as i64);
        
        let i = timestamps.partition_point(|&t| t <= deadline);
        if i == 0 {
            // Nothing to store in the cache.
            info!("nothing to store in the cache, since all the points have timestamps bigger than {}", deadline);
            return Ok(());
        }

        // timestamps are stored only once for all the tss, since they are identical.
        assert_identical_timestamps(tss, ec.step)?;

        if i < timestamps.len() {
            let rvs = tss
                .iter()
                .map(|ts| SeriesSlice::from_timeseries(ts, Some((0, i))))
                .collect::<Vec<SeriesSlice>>();

            self.put_internal(&rvs, ec, expr, window, &span)
        } else {
            let rvs = tss
                .iter()
                .map(|ts| {
                    SeriesSlice::from_timeseries(ts, None)
                })
                .collect::<Vec<SeriesSlice>>();

            self.put_internal(&rvs, ec, expr, window, &span)
        }
    }

    fn put_internal(
        &self,
        tss: &[SeriesSlice],
        ec: &EvalConfig,
        expr: &Expr,
        window: Duration,
        span: &EnteredSpan,
    ) -> RuntimeResult<()> {
        let is_tracing = span_enabled!(Level::TRACE);

        let size = estimate_size(tss);
        if self.max_marshaled_size > 0 && size > self.max_marshaled_size as usize {
            // do not marshal tss, since it would occupy too much space
            info!(
                "cannot store series in the cache, since they would occupy more than {} bytes",
                self.max_marshaled_size
            );
            return Ok(());
        }

        let mut inner = self.inner.lock().unwrap();

        let res = self.get_cache_metadata(&mut inner, ec, expr, window)?;
        let mut mi = if let Some((mi, _)) = res {
            mi
        } else {
            RollupResultCacheMetaInfo::new()
        };

        let timestamps = &tss[0].timestamps;
        let start = timestamps[0];
        let end = timestamps[timestamps.len() - 1];

        if mi.covers_time_range(start, end) {
            if is_tracing {
                let start_string = start.to_rfc3339();
                let end_string = end.to_rfc3339();

                info!(
                    "series on the given timeRange=[{}..{}] already exist in the cache",
                    start_string, end_string
                );
            }
            return Ok(());
        }

        let mut result_buf = get_pooled_buffer(size);
        // todo: should we handle error here and consider it a cache miss ?

        compress_series_slice(tss, &mut result_buf)?;

        if is_tracing {
            let start_string = start.to_rfc3339();
            let end_string = end.to_rfc3339();
            span.record("series", tss.len());

            info!(
                "marshal {} series on a timeRange=[{}..{}] into {} bytes",
                tss.len(),
                start_string,
                end_string,
                result_buf.len()
            )
        }

        let suffix = self.cache_key_suffix.inc();
        let key = RollupResultCacheKey::new(suffix);

        // Store tss in the cache.
        let mut meta_info_key = get_pooled_buffer(32);
        let mut meta_info_buf = get_pooled_buffer(32);

        key.marshal(&mut meta_info_key);

        inner.cache.set_big(&meta_info_key, result_buf.as_slice());

        info!("store {} bytes in the cache", result_buf.len());

        mi.add_key(key, start, end)?;
        mi.marshal(&mut meta_info_buf);
        inner
            .cache
            .set(meta_info_key.as_slice(), meta_info_buf.as_slice());

        Ok(())
    }

    fn get_cache_metadata(
        &self,
        inner: &mut Inner,
        ec: &EvalConfig,
        expr: &Expr,
        window: Duration,
    ) -> RuntimeResult<Option<(RollupResultCacheMetaInfo, u64)>> {
        let hash = marshal_rollup_result_cache_key(
            &mut inner.hasher,
            expr,
            window,
            ec.step,
            &ec.enforced_tag_filters,
        );
        let mut meta_info_buf = get_pooled_buffer(512);
        let found = inner.cache.get(&hash.to_ne_bytes(), &mut meta_info_buf);
        if found && !meta_info_buf.is_empty() {
            match RollupResultCacheMetaInfo::from_buf(&meta_info_buf) {
                Err(_) => {
                    let msg = "BUG: cannot unmarshal RollupResultCacheMetaInfo; it looks like it was improperly saved";
                    Err(RuntimeError::SerializationError(msg.to_string()))
                }
                Ok(mi) => Ok(Some((mi, hash))),
            }
        } else {
            Ok(None)
        }
    }

    pub fn get_stats(&self) -> RollupCacheStats {
        let inner = self.inner.lock().unwrap();
        inner.stats.clone()
    }

    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.cache.clear();
        inner.stats = Default::default();
    }
}

// let resultBufPool = ByteBufferPool

/// Increment this value every time the format of the cache changes.
const ROLLUP_RESULT_CACHE_VERSION: u8 = 8;

const ROLLUP_TYPE_TIMESERIES: u8 = 0;

fn marshal_rollup_result_cache_key_internal(
    hasher: &mut Xxh3,
    expr: &Expr,
    window: Duration,
    step: Duration,
    etfs: &Option<Matchers>,
    cache_type: u8,
) -> u64 {
    hasher.reset();

    let prefix: u64 = get_rollup_result_cache_key_prefix();
    hasher.write_u8(ROLLUP_RESULT_CACHE_VERSION);
    hasher.write_u64(prefix);
    hasher.write_u8(cache_type);
    hasher.write_u128(window.as_millis());
    hasher.write_u128(step.as_millis());
    hasher.write(format!("{}", expr).as_bytes());

    if let Some(etfs) = etfs {
        for etf in etfs.iter() {
            for f in etf.iter() {
                hasher.write_str(&f.label);
                hasher.write_str(f.op.as_str());
                hasher.write_str(&f.value);
            }
        }
    }

    hasher.digest()
}

fn marshal_rollup_result_cache_key(
    hasher: &mut Xxh3,
    expr: &Expr,
    window: Duration,
    step: Duration,
    etfs: &Option<Matchers>,
) -> u64 {
    marshal_rollup_result_cache_key_internal(
        hasher,
        expr,
        window,
        step,
        etfs,
        ROLLUP_TYPE_TIMESERIES,
    )
}

/// Merges two sets of timeseries `a` and `b` and returns the result.
///
/// ### Arguments
/// * `a` - The first set of timeseries, covering the range [ec.Start .. bStart - ec.Step].
/// * `b` - The second set of timeseries, covering the range [bStart .. ec.End].
/// * `b_start` - The start timestamp for the `b` timeseries.
/// * `ec` - The evaluation configuration containing the shared timestamps and step.
///
/// ### Returns
/// A tuple containing the merged timeseries. If the merge fails (e.g., due to duplicate metric names),
/// `None` is returned.
pub(crate) fn merge_timeseries(
    a: Vec<Timeseries>,
    b: Vec<Timeseries>,
    b_start: i64,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<Timeseries>> {
    let shared_timestamps = ec.get_timestamps()?;

    // Find the index where `b_start` begins in the shared timestamps.
    let i = shared_timestamps
        .iter()
        .position(|&ts| ts >= b_start)
        .unwrap_or(shared_timestamps.len());

    let a_timestamps = &shared_timestamps[..i];
    let b_timestamps = &shared_timestamps[i..];

    // If `b` covers the entire range, return `b` directly.
    if b_timestamps.len() == shared_timestamps.len() {
        for ts_b in &b {
            if ts_b.timestamps.as_slice() != b_timestamps {
                panic!(
                    "BUG: invalid timestamps in b series {}; got {:?}; want {:?}",
                    ts_b.metric_name, ts_b.timestamps, b_timestamps
                );
            }
        }
        return Ok(b);
    }

    // Create a map of metric names to timeseries for `a`.
    let mut a_map: HashMap<Signature, Timeseries> = HashMap::with_capacity(a.len());

    for ts_a in a.into_iter() {
        if ts_a.timestamps.as_slice() != a_timestamps {
            panic!(
                "BUG: invalid timestamps in a series {}; got {:?}; want {:?}",
                ts_a.metric_name, ts_a.timestamps, a_timestamps
            );
        }
        let mut ts_a = ts_a;
        ts_a.metric_name.sort_labels();
        let signature = ts_a.signature();

        match a_map.entry(signature) {
            Occupied(_) => {
                return Err(RuntimeError::DuplicateMetricLabels(ts_a.metric_name.to_string()));
            }, // Duplicate metric names in `a`.
            Vacant(entry) => {
                entry.insert(ts_a);
            }
        }
    }

    // Create a map of metric names to timeseries for `b`.
    let mut b_map: AHashSet<Signature> = AHashSet::with_capacity(b.len());
    let mut merged_series = Vec::new();
    let mut a_nans = Vec::new();

    let sample_count = shared_timestamps.len();
    for ts_b in b.into_iter() {
        if ts_b.timestamps.as_slice() != b_timestamps {
            panic!(
                "BUG: invalid timestamps in b series {}; got {:?}; want {:?}",
                ts_b.metric_name, ts_b.timestamps, b_timestamps
            );
        }

        let signature = ts_b.signature();

        if !b_map.insert(signature) {
            return Err(RuntimeError::DuplicateMetricLabels(ts_b.metric_name.to_string()));
        }

        // Create a new timeseries for the merged result.
        let mut merged_ts = Timeseries {
            metric_name: ts_b.metric_name,
            timestamps: Arc::clone(&shared_timestamps),
            values: Vec::with_capacity(sample_count),
        };

        // Append values from `a` or NaNs if `a` doesn't have the series.
        if let Some(ts_a) = a_map.remove(&signature) {
            merged_ts.values.extend_from_slice(&ts_a.values);
        } else {
            if a_nans.is_empty() {
                a_nans = vec![f64::NAN; a_timestamps.len()];
            }
            merged_ts.values.extend_from_slice(&a_nans);
        }

        // Append values from `b`.
        merged_ts.values.extend_from_slice(&ts_b.values);
        merged_series.push(merged_ts);
    }

    // Handle remaining series in `a` that weren't in `b`.
    let mut b_nans = Vec::new();
    for (_, ts_a) in a_map {
        let mut merged_ts = Timeseries {
            metric_name: ts_a.metric_name.clone(),
            timestamps: Arc::clone(&shared_timestamps),
            values: Vec::with_capacity(sample_count),
        };

        // Append values from `a`.
        merged_ts.values.extend_from_slice(&ts_a.values);

        // Append NaNs for the `b` range.
        if b_nans.is_empty() {
            b_nans = vec![f64::NAN; b_timestamps.len()];
        }
        merged_ts.values.extend_from_slice(&b_nans);

        merged_series.push(merged_ts);
    }

    Ok(merged_series)
}

#[derive(Clone, Default)]
struct RollupResultCacheMetaInfo {
    entries: Vec<RollupResultCacheMetaInfoEntry>,
}

impl RollupResultCacheMetaInfo {
    fn new() -> Self {
        Self { entries: vec![] }
    }

    fn from_buf(buf: &[u8]) -> RuntimeResult<Self> {
        let (res, _) = Self::unmarshal(buf)?;
        Ok(res)
    }

    fn marshal(&self, dst: &mut Vec<u8>) {
        marshal_var_usize(dst, self.entries.len());
        for entry in &self.entries {
            entry.marshal(dst);
        }
    }

    fn unmarshal(buf: &[u8]) -> RuntimeResult<(RollupResultCacheMetaInfo, &[u8])> {
        let mut src = buf;

        let (_tail, entries_len) = read_usize(src, "entries count")?;

        let mut entries: Vec<RollupResultCacheMetaInfoEntry> = Vec::with_capacity(entries_len);
        let mut i = 0;
        while i < entries_len {
            let (v, tail) = RollupResultCacheMetaInfoEntry::read(src).map_err(|err| {
                RuntimeError::from(format!("cannot unmarshal entry #{}: {:?}", i, err))
            })?;
            src = tail;
            entries.push(v);
            i += 1;
        }

        if i < entries_len {
            return Err(RuntimeError::from(format!(
                "expected {} cache entries: got {}",
                entries_len,
                entries.len()
            )));
        }

        if !src.is_empty() {
            return Err(RuntimeError::from(format!(
                "unexpected non-empty tail left; len(tail)={}",
                src.len()
            )));
        }

        Ok((Self { entries }, src))
    }

    fn covers_time_range(&self, start: i64, end: i64) -> bool {
        if start > end {
            // todo: remove panic. return Result instead
            panic!("BUG: start cannot exceed end; got {} vs {}", start, end)
        }
        self.entries
            .iter()
            .any(|entry| start >= entry.start && end <= entry.end)
    }

    fn get_best_key(&self, start: i64, end: i64) -> RuntimeResult<RollupResultCacheKey> {
        if start > end {
            return Err(RuntimeError::ArgumentError(format!(
                "BUG: start cannot exceed end; got {} vs {}",
                start, end
            )));
        }
        let mut best_key: RollupResultCacheKey = RollupResultCacheKey::default();
        let mut d_max: i64 = 0;
        for e in self.entries.iter() {
            if start < e.start {
                continue;
            }
            let mut d = e.end - start;
            if end <= e.end {
                d = end - start
            }
            if d >= d_max {
                d_max = d;
                best_key = e.key;
            }
        }
        Ok(best_key)
    }

    fn add_key(&mut self, key: RollupResultCacheKey, start: i64, end: i64) -> RuntimeResult<()> {
        if start > end {
            return Err(RuntimeError::ArgumentError(format!(
                "BUG: start cannot exceed end; got {} vs {}",
                start, end
            )));
        }

        self.entries
            .push(RollupResultCacheMetaInfoEntry { start, end, key });

        if self.entries.len() > 30 {
            // Remove old entries.
            self.entries.drain(0..9);
        }

        Ok(())
    }

    fn remove_key(&mut self, key: RollupResultCacheKey) {
        self.entries.retain(|x| x.key != key)
    }
}

#[derive(Default, Clone, PartialEq, Hash, Serialize, Deserialize)]
struct RollupResultCacheMetaInfoEntry {
    start: Timestamp,
    end: Timestamp,
    key: RollupResultCacheKey,
}

impl RollupResultCacheMetaInfoEntry {
    fn read(src: &[u8]) -> RuntimeResult<(RollupResultCacheMetaInfoEntry, &[u8])> {
        Self::unmarshal(src)
    }

    fn marshal(&self, dst: &mut Vec<u8>) {
        marshal_var_int(dst, self.start);
        marshal_var_int(dst, self.end);
        self.key.marshal(dst);
    }

fn unmarshal(src: &[u8]) -> RuntimeResult<(Self, &[u8])> {
    if src.len() < 8 {
        return Err(RuntimeError::SerializationError(format!(
            "cannot unmarshal start from {} bytes; need at least {} bytes",
            src.len(),
            8
        )));
    }

    let (src, start) = read_i64(src, "result cache index start")?;
    let (src, end) = read_i64(src, "result cache index end")?;
    let (key, src) = RollupResultCacheKey::unmarshal(src)?;

    Ok((Self { start, end, key }, src))
}
}

/// RollupResultCacheKey must be globally unique across nodes,
/// so it has prefix and suffix.
#[derive(Hash, Copy, Eq, PartialEq, Clone, Serialize, Deserialize)]
pub struct RollupResultCacheKey {
    prefix: u64,
    suffix: u64,
}

impl Default for RollupResultCacheKey {
    fn default() -> Self {
        Self::new(0)
    }
}

impl RollupResultCacheKey {
    fn new(suffix: u64) -> Self {
        // not sure if this is safe
        RollupResultCacheKey {
            prefix: get_rollup_result_cache_key_prefix(),
            suffix,
        }
    }

    // todo: replace this code with serde ?
    fn marshal(&self, dst: &mut Vec<u8>) {
        marshal_var_int(dst, ROLLUP_RESULT_CACHE_VERSION);
        marshal_var_int(dst, self.prefix);
        marshal_var_int(dst, self.suffix);
    }

    pub(self) fn unmarshal(src: &[u8]) -> RuntimeResult<(RollupResultCacheKey, &[u8])> {
        let (mut src, version) = read_u64(src, "result cache version")?;
        if version != ROLLUP_RESULT_CACHE_VERSION as u64 {
            return Err(RuntimeError::SerializationError(format!(
                "invalid result cache version: {}",
                version
            )));
        }

        let (tail, prefix) = read_u64(src, "prefix")?;
        src = tail;

        let (tail, suffix) = read_u64(src, "suffix")?;

        Ok((RollupResultCacheKey { prefix, suffix }, tail))
    }
}

fn estimate_size(tss: &[SeriesSlice]) -> usize {
    if tss.is_empty() {
        return 0;
    }
    // estimate size of labels
    let labels_size = tss
        .iter()
        .fold(0, |acc, ts| acc + ts.metric_name.serialized_size());
    let value_size = tss.iter().fold(0, |acc, ts| acc + ts.values.len() * 8);
    let timestamp_size = 8 * tss[0].timestamps.len();

    // Calculate the required size for marshaled tss.
    labels_size + value_size + timestamp_size
}

fn metric_name_hash_sorted(metric_name: &MetricName) -> u64 {
    let mut hasher = FastHasher::default();
    let mut labels: SmallVec<&Label, 8> = SmallVec::with_capacity(metric_name.labels.len());

    for label in metric_name.labels.iter() {
        labels.push(label);
    }

    labels.sort_unstable();

    hasher.write_str(&metric_name.measurement);
    for label in labels.iter() {
        hasher.write_str(&label.name);
        hasher.write_str(&label.value);
    }

    hasher.finish()
}