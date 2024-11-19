use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use tracing::{field, trace_span, Span};
use metricsql_common::atomic_counter::{AtomicCounter, RelaxedU64Counter};
use metricsql_common::pool::{get_pooled_vec_f64, get_pooled_vec_i64};
use metricsql_parser::ast::{AggregationExpr, DurationExpr, Expr, MetricExpr, RollupExpr};
use metricsql_parser::functions::RollupFunction;
use crate::cache::rollup_result_cache::merge_timeseries;
use crate::common::math::is_stale_nan;
use crate::execution::{align_start_end, eval_number, get_timestamps, validate_max_points_per_timeseries, Context, EvalConfig};
use crate::execution::new_eval::exec_expr;
use crate::functions::aggregate::IncrementalAggrFuncContext;
use crate::functions::rollup::{
    eval_prefuncs,
    get_rollup_configs, 
    RollupConfig,
    RollupHandler,
    MAX_SILENCE_INTERVAL,
    TimeSeriesMap
};
use crate::functions::transform::extract_labels_from_expr;
use crate::prelude::{get_timeseries, MetricName};
use crate::provider::{QueryResult, QueryResults, SearchQuery};
use crate::rayon::iter::ParallelIterator;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::{QueryValue, Timeseries, Timestamp};

pub(crate) struct RollupExecutor<'a> {
    /// Source expression
    expr: &'a Expr,
    re: &'a RollupExpr,
    func: RollupFunction,
    func_handler: RollupHandler,
    keep_metric_names: bool,
    is_tracing: bool,
    /// Max number of timeseries to return
    pub(crate) timeseries_limit: usize,
    pub(crate) is_incr_aggregate: bool,
}

impl<'a> RollupExecutor<'a> {
    pub(crate) fn new(
        function: RollupFunction,
        handler: RollupHandler,
        // expr may contain:
        // -: RollupFunc(m) if iafc is None
        // - aggrFunc(rollupFunc(m)) if iafc isn't None
        expr: &'a Expr,
        re: &'a RollupExpr,
    ) -> Self {
        Self {
            expr,
            re,
            func: function,
            func_handler: handler,
            keep_metric_names: function.keep_metric_name(),
            timeseries_limit: 0,
            is_incr_aggregate: false,
            is_tracing: false,
        }
    }

    pub(crate) fn eval(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        self.is_tracing = ctx.trace_enabled();
        let _ = if self.is_tracing {
            trace_span!(
                "rollup",
                "function" = self.func.name(),
                "expr" = self.expr.to_string().as_str(),
                "rollup_expr" = self.re.to_string().as_str(),
                "series" = field::Empty
            )
        } else {
            Span::none()
        };

        if let Some(at_expr) = &self.re.at {
            let at_timestamp = get_at_timestamp(ctx, ec, at_expr)?;
            let mut ec_new = ec.copy_no_timestamps();
            ec_new.start = at_timestamp;
            ec_new.end = at_timestamp;
            let mut tss = self.eval_without_at(ctx, &ec_new)?;

            // expand single-point tss to the original time range.
            let timestamps = ec.get_timestamps()?;
            for ts in tss.iter_mut() {
                ts.timestamps = Arc::clone(&timestamps);
                ts.values = vec![ts.values[0]; timestamps.len()];
            }

            Ok(QueryValue::InstantVector(tss))
        } else {
            let value = self.eval_without_at(ctx, ec)?;
            Ok(QueryValue::InstantVector(value))
        }
    }

    #[inline]
    fn adjust_eval_range(&self, ec: &'a EvalConfig) -> RuntimeResult<(i64, Cow<'a, EvalConfig>)> {
        let offset: i64 = duration_value(&self.re.offset, ec.step);

        let mut adjustment = 0 - offset;
        if self.func == RollupFunction::RollupCandlestick {
            // Automatically apply `offset -step` to `rollup_candlestick` function
            // in order to obtain expected OHLC results.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
            adjustment += ec.step
        }

        if adjustment != 0 {
            let mut result = ec.copy_no_timestamps();
            result.start += adjustment;
            result.end += adjustment;
            // There is no need in calling adjust_start_end() on ec_new if ec_new.may_cache is set to true,
            // since the time range alignment has been already performed by the caller,
            // so cache hit rate should be quite good.
            // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
            Ok((offset, Cow::Owned(result)))
        } else {
            Ok((offset, Cow::Borrowed(ec)))
        }
    }

    fn eval_without_at(&self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        let (offset, ec_new) = self.adjust_eval_range(ec)?;

        let mut rvs = if let Expr::MetricExpression(me) = &*self.re.expr {
            self.eval_with_metric_expr(ctx, &ec_new, me)?
        } else {
            if self.is_incr_aggregate {
                return Err(RuntimeError::from(format!(
                    "BUG:is_incr_aggregate must be false for rollup {} over subquery {}",
                    self.func, self.re
                )));
            }
            self.eval_with_subquery(ctx, &ec_new)?
        };

        if self.func == RollupFunction::AbsentOverTime {
            rvs = aggregate_absent_over_time(ec, &self.re.expr, &rvs)?;
        }

        if offset != 0 && !rvs.is_empty() {
            let src_timestamps = &rvs[0].timestamps;
            let shared = Arc::new(src_timestamps.iter().map(|&x| x + offset).collect::<Vec<_>>());
            for ts in rvs.iter_mut() {
                ts.timestamps = Arc::clone(&shared);
            }
        }

        Ok(rvs)
    }

    fn eval_with_subquery(&self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        // TODO: determine whether to use rollup result cache here.

        let span = if self.is_tracing {
            let function = self.func.name();
            trace_span!(
                "subquery",
                function,
                series = field::Empty,
                source_series = field::Empty,
                samples_scanned = field::Empty,
            )
        } else {
            Span::none()
        }
            .entered();

        let step = get_step(self.re, ec.step);
        let window = duration_value(&self.re.window, ec.step);

        let mut ec_sq = ec.copy_no_timestamps();
        ec_sq.start -= window + MAX_SILENCE_INTERVAL + step;
        ec_sq.end += step;
        ec_sq.step = step;
        validate_max_points_per_timeseries(
            ec_sq.start,
            ec_sq.end,
            ec_sq.step,
            ec_sq.max_points_per_series,
        )?;

        // unconditionally align start and end args to step for subquery as Prometheus does.
        (ec_sq.start, ec_sq.end) = align_start_end(ec_sq.start, ec_sq.end, ec_sq.step);

        // force refresh of timestamps
        let _ = ec_sq.get_timestamps()?;
        let tss_sq = exec_expr(ctx, &ec_sq, &self.re.expr)?;

        let tss_sq = tss_sq.as_instant_vec(&ec_sq)?;
        if tss_sq.len() == 0 {
            return Ok(vec![]);
        }

        let shared_timestamps = Arc::new(get_timestamps(
            ec.start,
            ec.end,
            ec.step,
            ec.max_points_per_series,
        )?);

        let min_staleness_interval = ctx.config.min_staleness_interval.num_milliseconds() as usize;
        let (rcs, pre_funcs) = get_rollup_configs(
            self.func,
            &self.func_handler,
            self.expr,
            ec.start,
            ec.end,
            ec.step,
            window,
            ec.max_points_per_series,
            min_staleness_interval,
            ec.lookback_delta,
            &shared_timestamps,
        )?;

        let (res, samples_scanned_total) = do_parallel(
            &tss_sq,
            move |ts_sq: &Timeseries,
                  values: &mut [f64],
                  timestamps: &[i64]|
                  -> RuntimeResult<(Vec<Timeseries>, u64)> {
                let mut res: Vec<Timeseries> = Vec::with_capacity(ts_sq.len());

                eval_prefuncs(&pre_funcs, values, timestamps);
                let mut scanned_total = 0_u64;

                for rc in rcs.iter() {
                    if let Some(tsm) = new_timeseries_map(
                        self.func,
                        self.keep_metric_names,
                        &shared_timestamps,
                        &ts_sq.metric_name,
                    ) {
                        let scanned = rc.do_timeseries_map(tsm.clone(), values, timestamps)?;
                        tsm.as_ref().append_timeseries_to(&mut res);
                        continue;
                    }

                    let mut ts: Timeseries = Default::default();

                    let scanned_samples = do_rollup_for_timeseries(
                        self.keep_metric_names,
                        rc,
                        &mut ts,
                        &ts_sq.metric_name,
                        values,
                        timestamps,
                        &shared_timestamps,
                    )?;

                    scanned_total += scanned_samples;

                    res.push(ts);
                }

                Ok((res, scanned_total))
            },
        )?;

        if !span.is_disabled() {
            span.record("series", res.len());
            span.record("source_series", tss_sq.len());
            span.record("samples_scanned", samples_scanned_total);
        }

        Ok(res)
    }

    fn eval_with_metric_expr(
        &self,
        ctx: &Context,
        ec: &EvalConfig,
        me: &MetricExpr,
    ) -> RuntimeResult<Vec<Timeseries>> {
        let window = duration_value(&self.re.window, ec.step);

        let span = {
            if self.is_tracing {
                trace_span!(
                    "rollup",
                    start = ec.start,
                    end = ec.end,
                    step = ec.step,
                    window,
                    function = self.func.name(),
                    needed_memory_bytes = field::Empty
                )
            } else {
                Span::none()
            }
        }
            .entered();

        if me.is_empty() {
            return eval_number(ec, f64::NAN);
        }

        // Search for partial results in cache.

        let tss_cached: Vec<Timeseries>;
        let start: i64;
        {
            let (cached, _start) = ctx.rollup_result_cache.get(ec, self.expr, window)?;
            tss_cached = cached.unwrap();
            start = _start;
        }

        if start > ec.end {
            // The result is fully cached.
            ctx.rollup_result_cache.full_hits.inc();
            return Ok(tss_cached);
        }

        if start > ec.start {
            ctx.rollup_result_cache.partial_hits.inc();
        } else {
            ctx.rollup_result_cache.misses.inc();
        }

        // Obtain rollup configs before fetching data from db,
        // so type errors can be caught earlier.
        let shared_timestamps = Arc::new(get_timestamps(
            start,
            ec.end,
            ec.step,
            ec.max_points_per_series,
        )?);

        let min_staleness_interval = ctx.config.min_staleness_interval.num_milliseconds() as usize;
        let (rcs, pre_funcs) = get_rollup_configs(
            self.func,
            &self.func_handler,
            self.expr,
            start,
            ec.end,
            ec.step,
            window,
            ec.max_points_per_series,
            min_staleness_interval,
            ec.lookback_delta,
            &shared_timestamps,
        )?;

        let pre_func = move |values: &mut [f64], timestamps: &[i64]| {
            eval_prefuncs(&pre_funcs, values, timestamps)
        };

        // Fetch the remaining part of the result.
        let tfs = vec![me.label_filters.clone()];
        let tfss = join_tag_filter_list(&tfs, &ec.enforced_tag_filters);
        let mut min_timestamp = start - MAX_SILENCE_INTERVAL;
        if window > ec.step {
            min_timestamp -= &window
        } else {
            min_timestamp -= ec.step
        }
        let filters = tfss.to_vec();
        let sq = SearchQuery::new(min_timestamp, ec.end, filters, ec.max_series);
        let mut rss = ctx.process_search_query(&sq, &ec.deadline)?;
        let rss_len = rss.len();
        if rss_len == 0 {
            rss.cancel();
            let dst: Vec<Timeseries> = vec![];
            let tss = merge_timeseries(tss_cached, dst, start, ec)?;
            return Ok(tss);
        }

        let rollup_memory_size = self.reserve_rollup_memory(ctx, ec, &rss, rcs.len())?;

        defer! {
           ctx.rollup_result_cache.release_memory(rollup_memory_size).unwrap();
           span.record("needed_memory_bytes", rollup_memory_size);
        }

        // Evaluate rollup
        // shadow timestamps
        let shared_timestamps = shared_timestamps.clone(); // TODO: do we need to clone ?
        let ignore_staleness = ec.no_stale_markers;
        let tss = match self.expr {
            Expr::Aggregation(ae) => self.eval_with_incremental_aggregate(
                ae,
                &mut rss,
                rcs,
                pre_func,
                &shared_timestamps,
                ignore_staleness,
            ),
            _ => self.eval_no_incremental_aggregate(
                &mut rss,
                rcs,
                pre_func,
                &shared_timestamps,
                ignore_staleness,
            ),
        }?;

        merge_timeseries(tss_cached, tss, start, ec).and_then(|res| {
            ctx.rollup_result_cache.put(ec, self.expr, window, &res)?;
            Ok(res)
        })
    }

    fn eval_with_incremental_aggregate<F>(
        &self,
        ae: &AggregationExpr,
        rss: &mut QueryResults,
        rcs: Vec<RollupConfig>,
        pre_func: F,
        shared_timestamps: &Arc<Vec<i64>>,
        ignore_staleness: bool,
    ) -> RuntimeResult<Vec<Timeseries>>
    where
        F: Fn(&mut [f64], &[i64]) + Send + Sync,
    {
        let span = if self.is_tracing {
            let function = self.func.name();
            trace_span!(
                "rollup",
                function,
                incremental = true,
                series = rss.len(),
                aggregation = ae.function.name(),
                samples_scanned = field::Empty
            )
        } else {
            Span::none()
        };

        let iafc = Arc::new(IncrementalAggrFuncContext::new(ae)?);

        struct Context<'a> {
            func: &'a RollupFunction,
            keep_metric_names: bool,
            iafc: Arc<IncrementalAggrFuncContext<'a>>,
            rcs: Vec<RollupConfig>,
            timestamps: Arc<Vec<i64>>,
            ignore_staleness: bool,
            samples_scanned_total: RelaxedU64Counter,
        }

        let mut ctx = Context {
            keep_metric_names: self.keep_metric_names,
            func: &self.func,
            iafc,
            rcs,
            timestamps: Arc::clone(shared_timestamps),
            ignore_staleness,
            samples_scanned_total: Default::default(),
        };

        rss.run_parallel(
            &mut ctx,
            |ctx: Arc<&mut Context>, rs: &mut QueryResult, worker_id: u64| {
                if !ctx.ignore_staleness {
                    drop_stale_nans(ctx.func, &mut rs.values, &mut rs.timestamps);
                }
                pre_func(&mut rs.values, &rs.timestamps);

                for rc in ctx.rcs.iter() {
                    if let Some(tsm) = new_timeseries_map(
                        *ctx.func,
                        ctx.keep_metric_names,
                        &ctx.timestamps,
                        &rs.metric,
                    ) {
                        rc.do_timeseries_map(tsm.clone(), &rs.values, &rs.timestamps)?;
                        for ts in tsm.as_ref() {
                            ctx.iafc.update_timeseries(ts, worker_id);
                        }
                        continue;
                    }

                    let mut ts = get_timeseries();
                    let samples_scanned = do_rollup_for_timeseries(
                        ctx.keep_metric_names,
                        rc,
                        &mut ts,
                        &rs.metric,
                        &rs.values,
                        &rs.timestamps,
                        &ctx.timestamps,
                    )?;

                    ctx.samples_scanned_total.add(samples_scanned);
                    // todo: return result rather than unwrap
                    ctx.iafc.update_timeseries(&mut ts, worker_id);
                }
                Ok(())
            },
        )?;

        let tss = ctx.iafc.finalize();

        if self.is_tracing {
            let samples_scanned = ctx.samples_scanned_total.get();
            span.record("series", tss.len());
            span.record("samples_scanned", samples_scanned);
        }

        Ok(tss)
    }

    fn eval_no_incremental_aggregate<F>(
        &self,
        rss: &mut QueryResults,
        rcs: Vec<RollupConfig>,
        pre_func: F,
        shared_timestamps: &Arc<Vec<Timestamp>>,
        no_stale_markers: bool,
    ) -> RuntimeResult<Vec<Timeseries>>
    where
        F: Fn(&mut [f64], &[i64]) + Send + Sync,
    {
        let span = if self.is_tracing {
            let function = self.func.name();
            let source_series = rss.len();
            // ("aggregation", ae.name.as_str()),
            // todo: add rcs to properties
            trace_span!(
                "rollup",
                function,
                incremental = false,
                source_series,
                series = field::Empty,
                samples_scanned = field::Empty
            )
        } else {
            Span::none()
        }
            .entered();

        struct TaskCtx<'a> {
            series: Arc<Mutex<Vec<Timeseries>>>,
            keep_metric_names: bool,
            func: RollupFunction,
            rcs: Vec<RollupConfig>,
            timestamps: &'a Arc<Vec<i64>>,
            no_stale_markers: bool,
            samples_scanned_total: RelaxedU64Counter,
        }

        let series = Arc::new(Mutex::new(Vec::with_capacity(rss.len() * rcs.len())));
        let mut ctx = TaskCtx {
            series: Arc::clone(&series),
            keep_metric_names: self.keep_metric_names,
            func: self.func,
            rcs,
            no_stale_markers,
            timestamps: shared_timestamps,
            samples_scanned_total: Default::default(),
        };

        rss.run_parallel(
            &mut ctx,
            |ctx: Arc<&mut TaskCtx>, rs: &mut QueryResult, _: u64| {
                if !ctx.no_stale_markers {
                    drop_stale_nans(&ctx.func, &mut rs.values, &mut rs.timestamps);
                }
                pre_func(&mut rs.values, &rs.timestamps);
                for rc in ctx.rcs.iter() {
                    let samples_scanned = process_result(
                        rs,
                        ctx.func,
                        rc,
                        ctx.series.clone(),
                        ctx.timestamps,
                        ctx.keep_metric_names,
                    )?;

                    ctx.samples_scanned_total.add(samples_scanned);
                }
                Ok(())
            },
        )?;

        // https://users.rust-lang.org/t/how-to-move-the-content-of-mutex-wrapped-by-arc/10259/7
        let res = Arc::try_unwrap(series).unwrap().into_inner().unwrap();

        if self.is_tracing {
            let samples_scanned = ctx.samples_scanned_total.get();
            span.record("series", res.len());
            span.record("samples_scanned", samples_scanned);
        }

        Ok(res)
    }

    fn reserve_rollup_memory(
        &self,
        ctx: &Context,
        ec: &EvalConfig,
        rss: &QueryResults,
        rcs_len: usize,
    ) -> RuntimeResult<usize> {
        // Verify timeseries fit available memory after the rollup.
        // Take into account points from tss_cached.
        let points_per_timeseries = 1 + (ec.end - ec.start) / ec.step;

        let rss_len = rss.len();
        let timeseries_len = if self.timeseries_limit > 0 {
            // The maximum number of output time series is limited by rss_len.
            if self.timeseries_limit > rss_len {
                rss_len
            } else {
                self.timeseries_limit
            }
        } else {
            rss_len
        };

        let rollup_points =
            mul_no_overflow(points_per_timeseries, (timeseries_len * rcs_len) as i64);
        let rollup_memory_size = rollup_points.saturating_mul(16) as usize;

        let memory_limit = ctx.rollup_result_cache.memory_limit();

        if !ctx.rollup_result_cache.reserve_memory(rollup_memory_size) {
            let msg = format!("not enough memory for processing {} data points across {} time series with {} points in each time series; \n
                                  total available memory for concurrent requests: {} bytes; requested memory: {} bytes; \n
                                  possible solutions are: reducing the number of matching time series; switching to node with more RAM; \n
                                  increasing -memory.allowedPercent; increasing `step` query arg ({})",
                              rollup_points,
                              timeseries_len * rcs_len,
                              points_per_timeseries,
                              memory_limit,
                              rollup_memory_size as u64,
                              ec.step as f64 / 1e3
            );

            return Err(RuntimeError::ResourcesExhausted(msg));
        }

        Ok(rollup_memory_size)
    }
}

#[inline]
fn new_timeseries_map(
    func: RollupFunction,
    keep_metric_names: bool,
    shared_timestamps: &Arc<Vec<Timestamp>>,
    mn: &MetricName,
) -> Option<Arc<TimeSeriesMap>> {
    if !TimeSeriesMap::is_valid_function(func) {
        return None;
    }
    let map = TimeSeriesMap::new(keep_metric_names, shared_timestamps, mn);
    Some(Arc::new(map))
}

fn process_result(
    rs: &mut QueryResult,
    func: RollupFunction,
    rc: &RollupConfig,
    series: Arc<Mutex<Vec<Timeseries>>>,
    timestamps: &Arc<Vec<i64>>,
    keep_metric_names: bool,
) -> RuntimeResult<u64> {
    if let Some(tsm) =
        new_timeseries_map(func, keep_metric_names, timestamps, &rs.metric)
    {
        rc.do_timeseries_map(tsm.clone(), &rs.values, &rs.timestamps)?;
        let mut tss = series.lock().unwrap();
        tsm.as_ref().append_timeseries_to(&mut tss);
        Ok(0_u64)
    } else {
        let mut ts: Timeseries = Timeseries::default();
        let samples_scanned = do_rollup_for_timeseries(
            keep_metric_names,
            rc,
            &mut ts,
            &rs.metric,
            &rs.values,
            &rs.timestamps,
            timestamps,
        )?;

        let mut tss = series.lock().unwrap();
        tss.push(ts);
        Ok(samples_scanned)
    }
}

fn get_at_timestamp(ctx: &Context, ec: &EvalConfig, expr: &Expr) -> RuntimeResult<i64> {
    match exec_expr(ctx, ec, expr) {
        Err(err) => {
            let msg = format!("cannot evaluate `@` modifier: {:?}", err);
            Err(RuntimeError::from(msg))
        }
        Ok(tss_at) => {
            match tss_at {
                QueryValue::Scalar(v) => Ok((v * 1000_f64) as i64),
                QueryValue::InstantVector(v) => {
                    if v.len() != 1 {
                        let msg = format!("`@` modifier must return a single series; it returns {} series instead", v.len());
                        return Err(RuntimeError::from(msg));
                    }
                    let ts = &v[0];
                    if ts.values.is_empty() {
                        let msg = "`@` modifier expression returned an empty value";
                        return Err(RuntimeError::from(msg));
                    }
                    Ok((ts.values[0] * 1000_f64) as i64)
                }
                _ => {
                    let val = tss_at.get_int()?;
                    Ok(val * 1000_i64)
                }
            }
        }
    }
}

/// aggregate_absent_over_time collapses tss to a single time series with 1 and nan values.
///
/// Values for returned series are set to nan if at least a single tss series contains nan at that point.
/// This means that tss contains a series with non-empty results at that point.
/// This follows Prometheus logic - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2130
fn aggregate_absent_over_time(
    ec: &EvalConfig,
    expr: &Expr,
    tss: &[Timeseries],
) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs = get_absent_timeseries(ec, expr)?;
    if tss.is_empty() {
        return Ok(rvs);
    }
    for i in 0..tss[0].values.len() {
        if tss.iter().any(|ts| ts.values[i].is_nan()) {
            rvs[0].values[i] = f64::NAN;
        }
    }
    Ok(rvs)
}
fn get_absent_timeseries(ec: &EvalConfig, expr: &Expr) -> RuntimeResult<Vec<Timeseries>> {
    // Copy tags from arg
    let mut rvs = eval_number(ec, 1.0)?;
    if let Some(labels) = extract_labels_from_expr(expr) {
        for label in labels {
            rvs[0].metric_name.set_label_value(&label.name, &label.value);
        }
    }
    Ok(rvs)
}


/// Executes `f` for each `Timeseries` in `tss` in parallel.
pub(super) fn do_parallel<F>(tss: &Vec<Timeseries>, f: F) -> RuntimeResult<(Vec<Timeseries>, u64)>
where
    F: Fn(&Timeseries, &mut [f64], &[i64]) -> RuntimeResult<(Vec<Timeseries>, u64)> + Send + Sync,
{
    let res: RuntimeResult<Vec<(Vec<Timeseries>, u64)>> = tss
        .par_iter()
        .map(|ts| {
            let len = ts.values.len();
            // todo: should we have an upper limit here to avoid OOM? Or explicitly size down
            // afterward if needed?
            let mut values = get_pooled_vec_f64(len);
            let mut timestamps = get_pooled_vec_i64(len);

            // todo(perf): have param for if values have NaNs
            remove_nan_values(&mut values, &mut timestamps, &ts.values, &ts.timestamps);

            f(ts, &mut values, &mut timestamps)
        })
        .collect();

    let mut series: Vec<Timeseries> = Vec::with_capacity(tss.len());
    let tss = res?;
    let mut sample_total = 0_u64;
    for (timeseries, sample_count) in tss.into_iter() {
        sample_total += sample_count;
        series.extend::<Vec<Timeseries>>(timeseries)
    }

    Ok((series, sample_total))
}

fn remove_nan_values(
    dst_values: &mut Vec<f64>,
    dst_timestamps: &mut Vec<i64>,
    values: &[f64],
    timestamps: &[i64],
) {
    let mut has_nan = false;
    for v in values {
        if v.is_nan() {
            has_nan = true;
            break;
        }
    }

    if !has_nan {
        // Fast path - no NaNs.
        dst_values.extend_from_slice(values);
        dst_timestamps.extend_from_slice(timestamps);
        return;
    }

    // Slow path - remove NaNs.
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        dst_values.push(*v);
        dst_timestamps.push(timestamps[i])
    }
}

fn do_rollup_for_timeseries(
    keep_metric_names: bool,
    rc: &RollupConfig,
    ts_dst: &mut Timeseries,
    mn_src: &MetricName,
    values_src: &[f64],
    timestamps_src: &[i64],
    shared_timestamps: &Arc<Vec<i64>>,
) -> RuntimeResult<u64> {
    ts_dst.metric_name.copy_from(mn_src);
    if !rc.tag_value.is_empty() {
        ts_dst.metric_name.set_label_value("rollup", &rc.tag_value)
    }
    if !keep_metric_names {
        ts_dst.metric_name.reset_measurement();
    }
    let samples_scanned = rc.exec(&mut ts_dst.values, values_src, timestamps_src)?;
    ts_dst.timestamps = Arc::clone(shared_timestamps);

    Ok(samples_scanned)
}

fn mul_no_overflow(a: i64, b: i64) -> i64 {
    a.saturating_mul(b)
}

pub(crate) fn drop_stale_nans(
    func: &RollupFunction,
    values: &mut Vec<f64>,
    timestamps: &mut Vec<i64>,
) {
    if *func == RollupFunction::DefaultRollup || *func == RollupFunction::StaleSamplesOverTime {
        // do not drop Prometheus staleness marks (aka stale NaNs) for default_rollup() function,
        // since it uses them for Prometheus-style staleness detection.
        // do not drop staleness marks for stale_samples_over_time() function, since it needs
        // to calculate the number of staleness markers.
        return;
    }
    // Remove Prometheus staleness marks, so non-default rollup functions don't hit NaN values.
    let has_stale_samples = values.iter().any(|x| is_stale_nan(*x));

    if !has_stale_samples {
        // Fast path: values have no Prometheus staleness marks.
        return;
    }

    // Slow path: drop Prometheus staleness marks from values.
    let mut k = 0;
    for i in 0..values.len() {
        let v = values[i];
        if !is_stale_nan(v) {
            values[k] = v;
            timestamps[k] = timestamps[i];
            k += 1;
        }
    }

    values.truncate(k);
    timestamps.truncate(k);
}

fn duration_value(dur: &Option<DurationExpr>, step: i64) -> i64 {
    if let Some(ofs) = dur {
        ofs.value(step)
    } else {
        0
    }
}

#[inline]
fn get_step(re: &RollupExpr, step: i64) -> i64 {
    let res = duration_value(&re.step, step);
    if res == 0 {
        step
    } else {
        res
    }
}