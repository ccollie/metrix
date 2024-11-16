use std::sync::Arc;
use std::sync::RwLock;

use metricsql_parser::label::Matchers;

use crate::execution::context::Context;
use crate::provider::Deadline;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::{Timeseries, Timestamp, TimestampTrait};

/// validate_max_points_per_timeseries checks the maximum number of points that
/// may be returned per each time series.
///
/// The number mustn't exceed max_points_per_timeseries.
pub(crate) fn validate_max_points_per_timeseries(
    start: Timestamp,
    end: Timestamp,
    step: i64,
    max_points_per_timeseries: usize,
) -> RuntimeResult<()> {
    let points = (end - start).saturating_div(step + 1);
    if (max_points_per_timeseries > 0) && points > max_points_per_timeseries as i64 {
        let msg = format!(
            "too many points for the given step={}, start={} and end={}: {}; cannot exceed {}",
            step, start, end, points, max_points_per_timeseries
        );
        Err(RuntimeError::from(msg))
    } else {
        Ok(())
    }
}

/// The minimum number of points per timeseries for enabling time rounding.
/// This improves cache hit ratio for frequently requested queries over
/// big time ranges.
const MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: i64 = 50;

pub fn adjust_start_end(start: Timestamp, end: Timestamp, step: i64) -> (Timestamp, Timestamp) {
    // if disableCache {
    //     // do not adjust start and end values when cache is disabled.
    //     // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/563
    //     return (start, end);
    // }
    let points = (end - start) / step + 1;
    if points < MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING {
        // Too small number of points for rounding.
        return (start, end);
    }

    // Round start and end to values divisible by step in order
    // to enable response caching (see EvalConfig.mayCache).
    let (start, end) = align_start_end(start, end, step);

    // Make sure that the new number of points is the same as the initial number of points.
    let mut new_points = (end - start) / step + 1;
    let mut _end = end;
    while new_points > points {
        _end = end - step;
        new_points -= 1;
    }

    (start, _end)
}

pub fn align_start_end(start: Timestamp, end: Timestamp, step: i64) -> (Timestamp, Timestamp) {
    // Round start to the nearest smaller value divisible by step.
    let new_start = start - start % step;
    // Round end to the nearest bigger value divisible by step.
    let adjust = end % step;
    let mut new_end = end;
    if adjust > 0 {
        new_end += step - adjust
    }
    (new_start, new_end)
}

pub struct EvalConfig {
    pub start: Timestamp,
    pub end: Timestamp,
    pub step: i64, // todo: Duration

    /// `max_series` is the maximum number of time series which can be scanned by the query.
    /// Zero means 'no limit'
    pub max_series: usize,

    /// quoted remote address.
    pub quoted_remote_addr: Option<String>,

    pub deadline: Deadline,

    /// lookback_delta is analog to `-query.lookback-delta` from Prometheus.
    /// todo: change type to Duration
    pub lookback_delta: i64,

    /// How many decimal digits after the point to leave in response.
    pub round_digits: u8,

    /// `enforced_tag_filters` may contain additional label filters to use in the query.
    pub enforced_tag_filters: Option<Matchers>,

    /// Set this flag to true if the data doesn't contain Prometheus stale markers, so there is
    /// no need in spending additional CPU time on its handling. Staleness markers may exist only in
    /// data obtained from Prometheus scrape targets
    pub no_stale_markers: bool,

    /// The limit on the number of points which can be generated per each returned time series.
    pub max_points_per_series: usize,

    /// Whether to disable response caching. This may be useful during data back-filling
    pub disable_cache: bool,

    /// Whether to return an error for queries that rely on implicit subquery conversions,
    /// see https://docs.victoriametrics.com/metricsql/#subqueries for details.
    /// 		"See also -search.logImplicitConversion.")
    pub disable_implicit_conversion: bool,

    /// The timestamps for the query.
    /// Note: investigate using https://docs.rs/arc-swap/latest/arc_swap/
    _timestamps: RwLock<Arc<Vec<Timestamp>>>,

    /// Whether the response can be cached.
    _may_cache: bool,
}

impl EvalConfig {
    pub fn new(start: Timestamp, end: Timestamp, step: i64) -> Self {
        EvalConfig {
            start,
            end,
            step,
            no_stale_markers: true,
            ..Default::default()
        }
    }

    pub fn copy_no_timestamps(&self) -> EvalConfig {
        EvalConfig {
            start: self.start,
            end: self.end,
            step: self.step,
            deadline: self.deadline,
            max_series: self.max_series,
            quoted_remote_addr: self.quoted_remote_addr.clone(),
            _may_cache: self._may_cache,
            lookback_delta: self.lookback_delta,
            round_digits: self.round_digits,
            enforced_tag_filters: self.enforced_tag_filters.clone(),
            // do not copy src.timestamps - they must be generated again.
            _timestamps: RwLock::new(Arc::new(vec![])),
            no_stale_markers: self.no_stale_markers,
            max_points_per_series: self.max_points_per_series,
            disable_cache: self.disable_cache,
            disable_implicit_conversion: self.disable_implicit_conversion,
        }
    }

    pub fn validate(&self) -> RuntimeResult<()> {
        if self.start > self.end {
            let msg = format!(
                "BUG: start cannot exceed end; got {} vs {}",
                self.start, self.end
            );
            return Err(RuntimeError::from(msg));
        }
        if self.step <= 0 {
            let msg = format!("BUG: step must be greater than 0; got {}", self.step);
            return Err(RuntimeError::from(msg));
        }
        Ok(())
    }

    pub fn may_cache(&self) -> bool {
        if self.disable_cache {
            return false;
        }
        if self._may_cache {
            return true;
        }
        if self.start % self.step != 0 {
            return false;
        }
        if self.end % self.step != 0 {
            return false;
        }

        true
    }

    pub fn no_cache(&mut self) {
        self._may_cache = false
    }
    pub fn set_caching(&mut self, may_cache: bool) {
        self._may_cache = may_cache;
    }

    pub fn update_from_context(&mut self, ctx: &Context) {
        let state_config = &ctx.config;
        self.disable_cache = state_config.disable_cache;
        self.max_points_per_series = state_config.max_points_subquery_per_timeseries;
        self.no_stale_markers = state_config.no_stale_markers;
        self.lookback_delta = state_config.max_lookback.num_milliseconds();
        self.max_series = state_config.max_unique_timeseries;
    }

    pub fn get_timestamps(&self) -> RuntimeResult<Arc<Vec<Timestamp>>> {
        let locked = self._timestamps.read().unwrap();
        if locked.len() > 0 {
            return Ok(Arc::clone(&locked));
        }
        drop(locked);
        let mut write_locked = self._timestamps.write().unwrap();
        if write_locked.len() > 0 {
            return Ok(Arc::clone(&write_locked));
        }
        let ts = get_timestamps(self.start, self.end, self.step, self.max_points_per_series)?;

        let timestamps = Arc::new(ts);
        let res = timestamps.clone();
        *write_locked = timestamps;

        Ok(res)
    }

    pub fn timerange_string(&self) -> String {
        format!("[{}..{}]", self.start.to_rfc3339(), self.end.to_rfc3339())
    }

    pub fn data_points(&self) -> usize {
        let n: usize = (1 + (self.end - self.start) / self.step) as usize;
        n
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            step: 0,
            max_series: 0,
            quoted_remote_addr: None,
            deadline: Deadline::default(),
            _may_cache: false,
            lookback_delta: 0,
            round_digits: 100,
            enforced_tag_filters: None,
            no_stale_markers: true,
            max_points_per_series: 0,
            disable_cache: false,
            disable_implicit_conversion: false,
            _timestamps: RwLock::new(Arc::new(vec![])),
        }
    }
}

impl Clone for EvalConfig {
    fn clone(&self) -> Self {
        let timestamps = self.get_timestamps().unwrap_or_else(|_| Arc::new(vec![]));
        Self {
            start: self.start,
            end: self.end,
            step: self.step,
            deadline: self.deadline,
            max_series: self.max_series,
            quoted_remote_addr: self.quoted_remote_addr.clone(),
            _may_cache: self._may_cache,
            lookback_delta: self.lookback_delta,
            round_digits: self.round_digits,
            enforced_tag_filters: self.enforced_tag_filters.clone(),
            _timestamps: RwLock::new(timestamps),
            no_stale_markers: self.no_stale_markers,
            max_points_per_series: self.max_points_per_series,
            disable_cache: self.disable_cache,
            disable_implicit_conversion: self.disable_implicit_conversion,
        }
    }
}

impl From<&Context> for EvalConfig {
    fn from(ctx: &Context) -> Self {
        let mut config = EvalConfig::default();
        config.update_from_context(ctx);
        config
    }
}

pub fn get_timestamps(
    start: Timestamp,
    end: Timestamp,
    step: i64,
    max_timestamps_per_timeseries: usize,
) -> RuntimeResult<Vec<i64>> {
    // Sanity checks.
    if step <= 0 {
        let msg = format!("Step must be bigger than 0; got {step}");
        return Err(RuntimeError::from(msg));
    }

    if start > end {
        let msg = format!("Start cannot exceed End; got {start} vs {end}");
        return Err(RuntimeError::from(msg));
    }

    if let Err(err) =
        validate_max_points_per_timeseries(start, end, step, max_timestamps_per_timeseries)
    {
        let msg = format!(
            "BUG: {:?}; this must be validated before the call to get_timestamps",
            err
        );
        return Err(RuntimeError::from(msg));
    }

    // Prepare timestamps.
    let n: usize = (1 + (end - start) / step) as usize;
    // todo: use a pool
    let mut timestamps: Vec<i64> = Vec::with_capacity(n);
    for ts in (start..=end).step_by(step as usize) {
        timestamps.push(ts);
    }

    Ok(timestamps)
}

pub(crate) fn eval_number(ec: &EvalConfig, n: f64) -> RuntimeResult<Vec<Timeseries>> {
    let timestamps = ec.get_timestamps()?;
    let ts = Timeseries {
        metric_name: Default::default(),
        timestamps: timestamps.clone(),
        values: vec![n; timestamps.len()],
    };
    Ok(vec![ts])
}

pub(crate) fn eval_time(ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
    let mut rv = eval_number(ec, f64::NAN)?;
    let timestamps = rv[0].timestamps.clone(); // this is an Arc, so it's cheap to clone
    for (ts, val) in timestamps.iter().zip(rv[0].values.iter_mut()) {
        *val = (*ts as f64) / 1e3_f64;
    }
    Ok(rv)
}
