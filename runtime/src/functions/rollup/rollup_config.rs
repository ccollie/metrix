use metricsql_common::prelude::humanize_duration;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::Duration;
use metricsql_parser::ast::Expr;
use metricsql_parser::functions::{RollupFunction};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;

use crate::common::math::quantile;
use crate::execution::{get_timestamps, validate_max_points_per_timeseries};
use crate::functions::rollup::candlestick::{rollup_close, rollup_high, rollup_low, rollup_open};
use crate::functions::rollup::delta::delta_values;
use crate::functions::rollup::deriv::deriv_values;
use crate::functions::rollup::rollup_fns::{
    get_rollup_fn, remove_counter_resets, rollup_avg, rollup_max, rollup_min,
};
use crate::functions::rollup::{
    get_rollup_func_by_name, RollupFuncArg, RollupHandler, TimeSeriesMap,
};
use crate::types::{get_timeseries, MetricName, Timeseries, Timestamp};
use crate::{RuntimeError, RuntimeResult};

const EMPTY_STRING: &str = "";

/// The maximum interval without previous rows.
pub const MAX_SILENCE_INTERVAL: Duration = Duration::from_secs(5);

pub(crate) type PreFunction = fn(&mut [f64], &[Timestamp]) -> ();

#[inline]
pub(crate) fn eval_pre_funcs(fns: &PreFunctionVec, values: &mut [f64], timestamps: &[Timestamp]) {
    for f in fns {
        f(values, timestamps)
    }
}

#[inline]
fn remove_counter_resets_pre_func(values: &mut [f64], _: &[Timestamp]) {
    remove_counter_resets(values);
}

#[inline]
fn delta_values_pre_func(values: &mut [f64], _: &[Timestamp]) {
    delta_values(values);
}

/// Calculate intervals in seconds between samples.
fn calc_sample_intervals_pre_fn(values: &mut [f64], timestamps: &[Timestamp]) {
    // Calculate intervals in seconds between samples.
    let mut ts_secs_prev = f64::NAN;
    for (value, ts) in values.iter_mut().zip(timestamps.iter()) {
        let ts_secs = (ts / 1000) as f64;
        *value = ts_secs - ts_secs_prev;
        ts_secs_prev = ts_secs;
    }

    if values.len() > 1 {
        // Overwrite the first NaN interval with the second interval,
        // So min, max and avg rollup could be calculated properly,
        // since they don't expect to receive NaNs.
        values[0] = values[1]
    }
}

// Pre-allocated handlers for closure to save allocations at runtime
const FN_OPEN: RollupHandler = RollupHandler::Wrapped(rollup_open);
const FN_CLOSE: RollupHandler = RollupHandler::Wrapped(rollup_close);
const FN_MIN: RollupHandler = RollupHandler::Wrapped(rollup_min);
const FN_MAX: RollupHandler = RollupHandler::Wrapped(rollup_max);
const FN_AVG: RollupHandler = RollupHandler::Wrapped(rollup_avg);
const FN_LOW: RollupHandler = RollupHandler::Wrapped(rollup_low);
const FN_HIGH: RollupHandler = RollupHandler::Wrapped(rollup_high);

static MAX: &str = "max";
static MIN: &str = "min";
static AVG: &str = "avg";
static OPEN: &str = "open";
static CLOSE: &str = "close";
static LOW: &str = "low";
static HIGH: &str = "high";

fn get_tag_fn_from_str(name: &str) -> Option<(&'static str, &RollupHandler)> {
    match name {
        op if op.eq_ignore_ascii_case(MIN) => Some((MIN, &FN_MIN)),
        op if op.eq_ignore_ascii_case(MAX) => Some((MAX, &FN_MAX)),
        op if op.eq_ignore_ascii_case(AVG) => Some((AVG, &FN_AVG)),
        op if op.eq_ignore_ascii_case(OPEN) => Some((OPEN, &FN_OPEN)),
        op if op.eq_ignore_ascii_case(CLOSE) => Some((CLOSE, &FN_CLOSE)),
        op if op.eq_ignore_ascii_case(LOW) => Some((LOW, &FN_LOW)),
        op if op.eq_ignore_ascii_case(HIGH) => Some((HIGH, &FN_HIGH)),
        _ => None,
    }
}


#[derive(Clone, Debug)]
pub struct TagFunction {
    pub tag_value: &'static str,
    pub func: RollupHandler,
}

pub type PreFunctionVec = SmallVec<PreFunction, 4>;
pub type TagFunctionVec = SmallVec<TagFunction, 4>;
pub type RollupConfigVec = SmallVec<RollupConfig, 4>;

#[derive(Clone, Default, Debug)]
pub struct RollupFunctionHandlerMeta {
    may_adjust_window: bool,
    samples_scanned_per_call: usize,
    is_default_rollup: bool,
    pre_funcs: PreFunctionVec,
    functions: TagFunctionVec,
}

pub(crate) fn get_rollup_configs(
    func: RollupFunction,
    rf: &RollupHandler,
    expr: &Expr,
    start: Timestamp,
    end: Timestamp,
    step: Duration,
    window: Duration,
    max_points_per_series: usize,
    min_staleness_interval: Duration,
    lookback_delta: Duration,
    shared_timestamps: &Arc<Vec<i64>>,
) -> RuntimeResult<(RollupConfigVec, PreFunctionVec)> {

    let mut meta = get_rollup_function_handler_meta(expr, func, rf)?;
    let pre_funcs = std::mem::take(&mut meta.pre_funcs);
    let rcs = get_rollup_configs_from_meta(
        meta,
        start,
        end,
        step,
        window,
        max_points_per_series,
        min_staleness_interval,
        lookback_delta,
        shared_timestamps,
    )?;

    Ok((rcs, pre_funcs))
}

pub(crate) fn get_rollup_configs_from_meta(
    meta: RollupFunctionHandlerMeta,
    start: Timestamp,
    end: Timestamp,
    step: Duration,
    window: Duration,
    max_points_per_series: usize,
    min_staleness_interval: Duration,
    lookback_delta: Duration,
    shared_timestamps: &Arc<Vec<i64>>,
) -> RuntimeResult<RollupConfigVec> {

    let new_rollup_config = |rf: RollupHandler, tag_value: &'static str| -> RollupConfig {
        RollupConfig {
            tag_value,
            handler: rf,
            start,
            end,
            step,
            window,
            may_adjust_window: meta.may_adjust_window,
            lookback_delta,
            timestamps: Arc::clone(shared_timestamps),
            is_default_rollup: meta.is_default_rollup,
            max_points_per_series,
            min_staleness_interval,
            samples_scanned_per_call: meta.samples_scanned_per_call,
        }
    };

    let rcs = meta
        .functions
        .into_iter()
        .map(|nf| new_rollup_config(nf.func, nf.tag_value))
        .collect::<RollupConfigVec>();

    Ok(rcs)
}

#[derive(Clone)]
pub(crate) struct RollupConfig {
    /// This tag value must be added to "rollup" tag if non-empty.
    pub tag_value: &'static str,
    pub handler: RollupHandler,
    pub start: Timestamp,
    pub end: Timestamp,
    pub step: Duration,
    pub window: Duration,

    /// Whether window may be adjusted to 2 x interval between data points.
    /// This is needed for functions which have dt in the denominator
    /// such as rate, deriv, etc.
    /// Without the adjustment their value would jump in unexpected directions
    /// when using window smaller than 2 x scrape_interval.
    pub may_adjust_window: bool,

    pub timestamps: Arc<Vec<i64>>,

    /// lookback_delta is the analog to `-query.lookback-delta` from Prometheus world.
    pub lookback_delta: Duration,

    /// Whether default_rollup is used.
    pub is_default_rollup: bool,

    /// The maximum number of points which can be generated per each series.
    pub max_points_per_series: usize,

    /// The minimum interval for staleness calculations. This could be useful for removing gaps on
    /// graphs generated from time series with irregular intervals between samples.
    pub min_staleness_interval: Duration,

    /// The estimated number of samples scanned per Func call.
    ///
    /// If zero, then it is considered that Func scans all the samples passed to it.
    pub samples_scanned_per_call: usize,
}

impl Default for RollupConfig {
    fn default() -> Self {
        Self {
            tag_value: EMPTY_STRING,
            handler: RollupHandler::Fake("uninitialized"),
            start: 0,
            end: 0,
            step: Duration::ZERO,
            window: Duration::ZERO,
            may_adjust_window: false,
            timestamps: Arc::new(vec![]),
            lookback_delta: Duration::ZERO,
            is_default_rollup: false,
            max_points_per_series: 0,
            min_staleness_interval: Duration::ZERO,
            samples_scanned_per_call: 0,
        }
    }
}

impl Display for RollupConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RollupConfig(start={}, end={}, step={}, window={}, points={}, max_points_per_series={})",
            self.start, self.end, humanize_duration(&self.step), humanize_duration(&self.window), 
            self.timestamps.len(), self.max_points_per_series
        )
    }
}

impl RollupConfig {
    #[cfg(test)]
    pub(crate) fn ensure_timestamps(&mut self) -> RuntimeResult<()> {
        if self.timestamps.is_empty() {
            let timestamps =
                get_timestamps(self.start, self.end, self.step, self.max_points_per_series)?;
            self.timestamps = Arc::new(timestamps);
        }
        Ok(())
    }

    /// calculates rollup for the given timestamps and values, appends
    /// them to dst_values and returns results.
    ///
    /// rc.timestamps are used as timestamps for dst_values.
    ///
    /// timestamps must cover time range [rc.start - rc.window - MAX_SILENCE_INTERVAL ... rc.end].
    pub(crate) fn exec(
        &self,
        dst_values: &mut Vec<f64>,
        values: &[f64],
        timestamps: &[Timestamp],
    ) -> RuntimeResult<u64> {
        self.exec_internal(dst_values, None, values, timestamps)
    }

    pub(crate) fn process_rollup(
        &self,
        func: RollupFunction,
        metric: &MetricName,
        keep_metric_names: bool,
        values: &[f64],
        timestamps: &[Timestamp],
        shared_timestamps: &Arc<Vec<Timestamp>>,
    ) -> RuntimeResult<(u64, Vec<Timeseries>)> {
        let func_keeps_metric_name = func.keep_metric_name();
        if TimeSeriesMap::is_valid_function(func) {
            let tsm = Arc::new(TimeSeriesMap::new(
                keep_metric_names || func_keeps_metric_name,
                shared_timestamps,
                metric,
            ));
            let scanned = self.do_timeseries_map(tsm.clone(), values, timestamps)?;
            let len = tsm.series_len();
            if len == 0 {
                return Ok((0u64, vec![]));
            }
            let mut series = Vec::with_capacity(len);
            tsm.append_timeseries_to(&mut series);
            return Ok((scanned, series));
        }

        let mut ts_dst: Timeseries = Timeseries::default();
        ts_dst.metric_name.copy_from(metric);
        if !self.tag_value.is_empty() {
            ts_dst.metric_name.set("rollup", &self.tag_value)
        }
        if !keep_metric_names && !func_keeps_metric_name {
            ts_dst.metric_name.reset_measurement();
        }
        let samples_scanned = self.exec(&mut ts_dst.values, values, timestamps)?;
        ts_dst.timestamps = Arc::clone(shared_timestamps);
        let tss = vec![ts_dst];

        Ok((samples_scanned, tss))
    }

    /// calculates rollup for the given timestamps and values and puts them to tsm.
    /// returns the number of samples scanned
    fn do_timeseries_map(
        &self,
        tsm: Arc<TimeSeriesMap>,
        values: &[f64],
        timestamps: &[Timestamp],
    ) -> RuntimeResult<u64> {
        let mut ts = get_timeseries();
        self.exec_internal(&mut ts.values, Some(tsm), values, timestamps)
    }

    fn exec_internal(
        &self,
        dst_values: &mut Vec<f64>,
        tsm: Option<Arc<TimeSeriesMap>>,
        values: &[f64],
        timestamps: &[Timestamp],
    ) -> RuntimeResult<u64> {
        self.validate()?;
        dst_values.reserve(self.timestamps.len());

        let scrape_interval = get_scrape_interval(timestamps);
        let mut max_prev_interval = get_max_prev_interval(scrape_interval);
        if !self.lookback_delta.is_zero() && max_prev_interval > self.lookback_delta {
            max_prev_interval = self.lookback_delta;
        }
        if !self.min_staleness_interval.is_zero() {
            let msi = self.min_staleness_interval;
            if !msi.is_zero() && max_prev_interval < msi {
                max_prev_interval = msi;
            }
        }
        
        let mut window = if !self.window.is_zero() { self.window } else { self.step };
        if self.may_adjust_window && window < max_prev_interval {
            window = max_prev_interval;
        }
        if self.is_default_rollup && !self.lookback_delta.is_zero() && window > self.lookback_delta {
            window = self.lookback_delta;
        }

        let mut samples_scanned = values.len() as u64;
        let samples_scanned_per_call = self.samples_scanned_per_call as u64;
        let window_ms = window.as_millis() as i64;
        let max_prev_interval = max_prev_interval.as_millis() as i64;

        let mut i = 0;
        let mut j = 0;
        let mut ni = 0;
        let mut nj = 0;

        let func_args: Vec<_> = self.timestamps.iter().enumerate().map(|(idx, &t_end)| {
            let t_start = t_end - window_ms;

            ni = seek_first_timestamp_idx_after(&timestamps[i..], t_start, ni);
            i += ni;
            if j < i {
                j = i;
            }

            nj = seek_first_timestamp_idx_after(&timestamps[j..], t_end, nj);
            j += nj;

            let mut rfa = RollupFuncArg::default();
            rfa.window = window_ms;
            rfa.prev_value = if i > 0 && timestamps[i - 1] > t_start - max_prev_interval {
                values[i - 1]
            } else {
                f64::NAN
            };
            rfa.prev_timestamp = if i > 0 { timestamps[i - 1] } else { t_start - max_prev_interval };
            rfa.values = &values[i..j];
            rfa.timestamps = &timestamps[i..j];
            rfa.real_prev_value = if i > 0 { values[i - 1] } else { f64::NAN };
            rfa.real_next_value = if j < values.len() { values[j] } else { f64::NAN };
            rfa.curr_timestamp = t_end;
            rfa.idx = idx;
            rfa.tsm = tsm.as_ref().map(Arc::clone);

            if samples_scanned_per_call > 0 {
                samples_scanned += samples_scanned_per_call;
            } else {
                samples_scanned += rfa.values.len() as u64;
            }

            rfa
        }).collect();

        match func_args.len() {
            0 => {}
            1 => dst_values.push(self.handler.eval(&func_args[0])),
            2 => {
                let (first, second) = rayon::join(
                    || self.handler.eval(&func_args[0]),
                    || self.handler.eval(&func_args[1]),
                );
                dst_values.push(first);
                dst_values.push(second);
            }
            _ => {
                func_args.par_iter()
                    .map(|rfa| self.handler.eval(rfa))
                    .collect_into_vec(dst_values);
            }
        }

        Ok(samples_scanned)
    }

    fn validate(&self) -> RuntimeResult<()> {
        // Sanity checks.
        if self.step.is_zero() {
            let msg = format!("BUG: step must be bigger than 0; got {}", humanize_duration(&self.step));
            return Err(RuntimeError::from(msg));
        }
        if self.start > self.end {
            let msg = format!(
                "BUG: start cannot exceed end; got {} vs {}",
                self.start, self.end
            );
            return Err(RuntimeError::from(msg));
        }
        match validate_max_points_per_timeseries(
            self.start,
            self.end,
            self.step,
            self.max_points_per_series,
        ) {
            Err(err) => {
                let msg = format!(
                    "BUG: {:?}; this must be validated before the call to rollupConfig.exec",
                    err
                );
                Err(RuntimeError::from(msg))
            }
            _ => Ok(()),
        }
    }
}

/// `rollup_samples_scanned_per_call` contains functions which scan a lower number of samples
/// than is passed to the rollup func.
///
/// It is expected that the remaining rollupFuncs scan all the samples passed to them.
const fn rollup_samples_scanned_per_call(rf: RollupFunction) -> usize {
    use RollupFunction::*;

    match rf {
        AbsentOverTime => 1,
        CountOverTime => 1,
        DefaultRollup => 1,
        Delta => 2,
        DeltaPrometheus => 2,
        DerivFast => 2,
        FirstOverTime => 1,
        IDelta => 2,
        IDeriv => 2,
        Increase => 2,
        IncreasePrometheus => 2,
        IncreasePure => 2,
        IRate => 2,
        Lag => 1,
        LastOverTime => 1,
        Lifetime => 2,
        PresentOverTime => 1,
        Rate => 2,
        ScrapeInterval => 2,
        TFirstOverTime => 1,
        Timestamp => 1,
        TimestampWithName => 1,
        TLastOverTime => 1,
        _ => 0, // == num rows
    }
}

fn seek_first_timestamp_idx_after(
    timestamps: &[Timestamp],
    seek_timestamp: Timestamp,
    n_hint: usize,
) -> usize {
    let count = timestamps.len();
    if count == 0 || timestamps[0] > seek_timestamp {
        return 0;
    }

    let start_idx = (n_hint.saturating_sub(2)).min(count - 1);
    let end_idx = (n_hint + 2).min(count);

    let slice_start = if timestamps[start_idx] <= seek_timestamp { start_idx } else { 0 };
    let slice_end = if end_idx < count && timestamps[end_idx] > seek_timestamp { end_idx } else { count };

    let slice = &timestamps[slice_start..slice_end];

    if slice.len() < 32 {
        slice.iter()
            .position(|&t| t > seek_timestamp)
            .map_or(slice.len(), |pos| pos)
    } else {
        match slice.binary_search(&(seek_timestamp + 1)) {
            Ok(pos) | Err(pos) => pos,
        }
    }
    .saturating_add(slice_start)
}

fn get_scrape_interval(timestamps: &[Timestamp]) -> Duration {
    if timestamps.len() < 2 {
        return MAX_SILENCE_INTERVAL;
    }

    // Estimate scrape interval as 0.6 quantile for the first 20 intervals.
    let mut ts_prev = timestamps[0];
    let timestamps = &timestamps[1..];
    let len = timestamps.len().clamp(0, 20);

    let mut intervals: [f64; 20] = [0_f64; 20];
    for (interval, ts) in intervals.iter_mut().zip(timestamps.iter()) {
        *interval = (ts - ts_prev) as f64;
        ts_prev = *ts
    }
    let scrape_interval = quantile(0.6, &intervals[0..len]) as i64;
    if scrape_interval <= 0 {
        return MAX_SILENCE_INTERVAL;
    }
    Duration::from_millis(scrape_interval as u64)
}

const fn get_max_prev_interval(scrape_interval: Duration) -> Duration {
    let scrape_interval = scrape_interval.as_millis() as i64;
    // Increase scrape_interval more for smaller scrape intervals in order to hide possible gaps
    // when high jitter is present.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/139 .
    let interval = if scrape_interval <= 2_000i64 {
        scrape_interval + 4 * scrape_interval
    } else if scrape_interval <= 4_000i64 {
        scrape_interval + 2 * scrape_interval
    } else if scrape_interval <= 8_000i64 {
        scrape_interval + scrape_interval
    } else if scrape_interval <= 16_000i64 {
        scrape_interval + scrape_interval / 2
    } else if scrape_interval <= 32_000i64 {
        scrape_interval + scrape_interval / 4
    } else {
        scrape_interval + scrape_interval / 8
    };

    if interval < 0 {
        Duration::ZERO
    } else {
        Duration::from_millis(interval as u64)
    }
}


fn get_rollup_function_handler_meta(
    expr: &Expr,
    func: RollupFunction,
    rf: &RollupHandler,
) -> RuntimeResult<RollupFunctionHandlerMeta> {
    let mut pre_funcs: PreFunctionVec = PreFunctionVec::new();

    if func.should_remove_counter_resets() {
        pre_funcs.push(remove_counter_resets_pre_func);
    }

    let new_function_config = |func: &RollupHandler, tag_value: &'static str| -> TagFunction {
        TagFunction {
            tag_value,
            func: func.clone(),
        }
    };

    let new_function_configs =
        |dst: &mut TagFunctionVec, tag: Option<&String>, valid: &'static [&str]| -> RuntimeResult<()> {
            if let Some(tag_value) = tag {
                let (name, func) = get_tag_fn_from_str(tag_value).ok_or_else(|| {
                    RuntimeError::ArgumentError(format!(
                        "unexpected rollup tag value {tag_value}; wanted {}",
                        valid
                            .iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<String>>()
                            .join(", ")
                    ))
                })?;
                dst.push(new_function_config(func, name));
            } else {
                for tag_value in valid {
                    let (name, func) = get_tag_fn_from_str(tag_value).unwrap();
                    dst.push(TagFunction {
                        tag_value: name,
                        func: func.clone(),
                    });
                }
            }

            Ok(())
        };

    let append_stats_function = |dst: &mut TagFunctionVec, expr: &Expr| -> RuntimeResult<()> {
        static VALID: [&str; 3] = [MIN, MAX, AVG];
        let tag = get_rollup_tag(expr)?;
        new_function_configs(dst, tag, &VALID)
    };

    let mut funcs: TagFunctionVec = TagFunctionVec::new();
    match func {
        RollupFunction::Rollup => {
            append_stats_function(&mut funcs, expr)?;
        }
        RollupFunction::RollupRate | RollupFunction::RollupDeriv => {
            pre_funcs.push(deriv_values);
            append_stats_function(&mut funcs, expr)?;
        }
        RollupFunction::RollupIncrease | RollupFunction::RollupDelta => {
            pre_funcs.push(delta_values_pre_func);
            append_stats_function(&mut funcs, expr)?;
        }
        RollupFunction::RollupCandlestick => {
            static VALID: [&str; 4] = [OPEN, CLOSE, LOW, HIGH];
            let tag = get_rollup_tag(expr)?;
            new_function_configs(&mut funcs, tag, &VALID)?;
        }
        RollupFunction::RollupScrapeInterval => {
            pre_funcs.push(calc_sample_intervals_pre_fn);
            append_stats_function(&mut funcs, expr)?;
        }
        RollupFunction::AggrOverTime => {
            let fns = get_rollup_aggr_functions(expr)?;
            for rf in fns {
                if rf.should_remove_counter_resets() {
                    // There is no need to save the previous pre_func, since it is either empty or the same.
                    pre_funcs.clear();
                    pre_funcs.push(remove_counter_resets_pre_func);
                }
                let rollup_fn = get_rollup_fn(&rf)?;
                let handler = RollupHandler::wrap(rollup_fn);
                funcs.push(TagFunction {
                    tag_value: &*rf.name(),
                    func: handler,
                });
            }
        }
        _ => {
            funcs.push(TagFunction {
                tag_value: EMPTY_STRING,
                func: rf.clone(),
            });
        }
    }

    let may_adjust_window = func.can_adjust_window();
    let is_default_rollup = func == RollupFunction::DefaultRollup;
    let samples_scanned_per_call = rollup_samples_scanned_per_call(func);

    Ok(RollupFunctionHandlerMeta {
        may_adjust_window,
        is_default_rollup,
        samples_scanned_per_call,
        pre_funcs,
        functions: funcs,
    })
}

/// get_rollup_tag returns the possible second arg from the expr.
///
/// The expr can have the following forms:
/// - rollup_func(q, tag)
/// - aggr_func(rollup_func(q, tag)) - this form is used during incremental aggregate calculations
fn get_rollup_tag(expr: &Expr) -> RuntimeResult<Option<&String>> {
    if let Expr::Function(fe) = expr {
        if fe.args.len() < 2 {
            return Ok(None);
        }
        if fe.args.len() != 2 {
            let msg = format!(
                "unexpected number of args for rollup function {}; got {:?}; want 2",
                fe.name(),
                fe.args
            );
            return Err(RuntimeError::ArgumentError(msg));
        }
        let arg = &fe.args[1];
        if let Expr::StringLiteral(se) = arg {
            if se.is_empty() {
                return Err(RuntimeError::ArgumentError("unexpected empty rollup tag value".to_string()));
            }
            Ok(Some(se))
        } else {
            Err(RuntimeError::ArgumentError(format!(
                "unexpected rollup tag value {arg}; wanted min, max or avg",
            )))
        }
    } else {
        let msg = format!("BUG: unexpected expression; want FunctionExpr; got {expr};");
        Err(RuntimeError::ArgumentError(msg))
    }
}

// todo: use in optimize so it's cached in the DAG node
fn get_rollup_aggr_functions(expr: &Expr) -> RuntimeResult<Vec<RollupFunction>> {
    fn get_func_by_name(name: &str) -> RuntimeResult<RollupFunction> {
        if let Ok(func) = get_rollup_func_by_name(name) {
            if !func.is_aggregate_function() {
                let msg = format!(
                    "{name} cannot be used in `aggr_over_time` function; expecting aggregate function name",
                );
                return Err(RuntimeError::General(msg));
            }
            Ok(func)
        } else {
            let msg =
                format!("Unknown aggregate function {name} used in `aggr_over_time` function;",);
            Err(RuntimeError::ArgumentError(msg))
        }
    }

    fn get_func_from_expr(expr: &Expr) -> RuntimeResult<RollupFunction> {
        if let Expr::StringLiteral(name) = expr {
            get_func_by_name(name.as_str())
        } else {
            let msg =
                format!("{expr} cannot be passed here; expecting quoted aggregate function name",);
            Err(RuntimeError::ArgumentError(msg))
        }
    }

    let expr = if let Expr::Aggregation(afe) = expr {
        // This is for incremental aggregate function case:
        //
        //     sum(aggr_over_time(...))
        // See aggr_incremental.rs for details.
        &afe.args[0]
    } else {
        expr
    };

    if let Expr::Function(fe) = expr {
        if !fe.is_rollup_function(RollupFunction::AggrOverTime) {
            let msg = format!(
                "BUG: unexpected function {}; want `aggr_over_time`",
                fe.name()
            );
            return Err(RuntimeError::ArgumentError(msg));
        }
        if fe.args.len() < 2 {
            let msg = format!(
                "unexpected number of args to aggr_over_time(); got {}; want at least 2",
                fe.args.len()
            );
            return Err(RuntimeError::ArgumentError(msg));
        }
        let mut functions = Vec::with_capacity(fe.args.len() - 1);
        for arg in fe.args[1..].iter() {
            functions.push(get_func_from_expr(arg)?)
        }
        Ok(functions)
    } else {
        let msg = format!(
            "BUG: unexpected expression; want FunctionExpr; got {}; value: {expr}",
            expr.variant_name()
        );
        Err(RuntimeError::ArgumentError(msg))
    }
}
