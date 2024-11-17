use std::hash::Hasher;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use ahash::AHashMap;
/// import commonly used items from the prelude:
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::span::EnteredSpan;
use tracing::{field, info, span_enabled, trace_span, Level, Span};
use xxhash_rust::xxh3::Xxh3;

use metricsql_common::prelude::{get_pooled_buffer, AtomicCounter, RelaxedU64Counter};
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
use crate::types::{assert_identical_timestamps, SeriesSlice, Timestamp, Timeseries, TimestampTrait};

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
        let mut rng = thread_rng();
        rng.gen::<u64>()
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
        let mut rng = thread_rng();

        let cache = Box::new(DefaultResultCacheStorage::new(max_size));
        let hasher = Xxh3::default();
        let suffix: u64 = rng.gen_range((1 << 16)..(1 << 31));
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

    pub fn get(
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
                "rollup_cache::get",
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
            || compressed_result_buf.len() == 0
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

    pub fn put(
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
                "rollup_cache::put",
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

        // Remove values up to currentTime - step - CACHE_TIMESTAMP_OFFSET,
        // since these values may be added later.
        let timestamps = tss[0].timestamps.as_slice();
        let deadline = Timestamp::now() - (ec.step - CACHE_TIMESTAMP_OFFSET).as_millis() as i64;
        let mut i = timestamps.len() - 1;
        // todo: use binary search
        while i > 0 && timestamps[i] > deadline {
            i -= 1;
        }
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
                .map(|ts| SeriesSlice::from_timeseries(ts, None))
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
        if found && meta_info_buf.len() > 0 {
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
                hasher.write(f.label.as_bytes());
                hasher.write(f.op.as_str().as_bytes());
                hasher.write(f.value.as_bytes());
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

/// `merge_timeseries` concatenates b with a and returns the result.
///
/// Preconditions:
/// - a mustn't intersect with b.
/// - a timestamps must be smaller than b timestamps.
///
/// Post conditions:
/// - a and b cannot be used after returning from the call.
pub fn merge_timeseries(
    a: Vec<Timeseries>,
    b: Vec<Timeseries>,
    b_start: i64,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<Timeseries>> {
    let shared_timestamps = ec.get_timestamps()?;
    if b_start == ec.start {
        // Nothing to merge - b covers all the time range.
        // Verify b is correct.
        let mut second = b;
        for ts_second in second.iter_mut() {
            ts_second.timestamps = Arc::clone(&shared_timestamps);
            validate_timeseries_length(ts_second)?;
        }
        // todo(perf): is this clone the most efficient
        return Ok(second);
    }

    // todo: use Signature
    let mut map: AHashMap<String, Timeseries> = AHashMap::with_capacity(a.len());

    for ts in a.into_iter() {
        let key = ts.metric_name.to_string();
        map.insert(key, ts);
    }

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(map.len());

    for mut ts_second in b.into_iter() {
        let key = ts_second.metric_name.to_string();

        let mut tmp: Timeseries = Timeseries {
            metric_name: std::mem::take(&mut ts_second.metric_name), // todo(perf): how to avoid clone() (use into)?
            timestamps: Arc::clone(&shared_timestamps),
            values: Vec::with_capacity(shared_timestamps.len()),
        };

        match map.get_mut(&key) {
            None => {
                let mut t_start = ec.start;
                let step = ec.step.as_millis() as i64;
                while t_start < b_start {
                    tmp.values.push(f64::NAN);
                    t_start = t_start + step;
                }
            }
            Some(ts_a) => {
                tmp.values.extend_from_slice(&ts_a.values);
                map.remove(&key);
            }
        }

        tmp.values.append(&mut ts_second.values);
        validate_timeseries_length(&ts_second)?;

        rvs.push(tmp);
    }

    // todo: collect() then rvs.extend()
    // Copy the remaining timeseries from m.
    for ts_a in map.values_mut() {
        let t_start = b_start;
        while t_start <= ec.end {
            ts_a.values.push(f64::NAN);
            t_start.add(ec.step);
        }

        validate_timeseries_length(ts_a)?;

        rvs.push(std::mem::take(ts_a));
    }

    Ok(rvs)
}

fn validate_timeseries_length(ts: &Timeseries) -> RuntimeResult<()> {
    if ts.values.len() != ts.timestamps.len() {
        let msg = format!(
            "mismatched timestamp/value length in timeseries; got {}; want {}",
            ts.values.len(),
            ts.timestamps.len()
        );
        return Err(RuntimeError::InvalidState(msg));
    }
    Ok(())
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
