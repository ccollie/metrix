use std::borrow::Cow;
use std::ops::Div;
use crate::cache::rollup_result_cache::merge_timeseries;
use crate::common::math::is_stale_nan;
use crate::execution::aggregate::get_timeseries_limit;
use crate::execution::exec::eval_expr;
use crate::execution::{
    align_start_end, eval_number, get_timestamps, validate_max_points_per_timeseries, Context,
    EvalConfig,
};
use crate::functions::aggregate::IncrementalAggrFuncContext;
use crate::functions::rollup::{
    eval_pre_funcs, get_rollup_configs, RollupConfig, RollupConfigVec, RollupHandler, TimeSeriesMap,
    MAX_SILENCE_INTERVAL,
};
use crate::functions::transform::extract_labels_from_expr;
use crate::prelude::{get_timeseries, is_empty_extra_matchers, MetricName};
use crate::provider::{
    join_matchers_with_extra_filters_owned, QueryResult, QueryResults, SearchQuery,
};
use crate::rayon::iter::ParallelIterator;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::{QueryValue, Timeseries, Timestamp};
use metricsql_common::atomic_counter::{AtomicCounter, RelaxedU64Counter};
use metricsql_common::pool::{get_pooled_vec_f64, get_pooled_vec_i64};
use metricsql_parser::ast::{AggregationExpr, Expr, MetricExpr, RollupExpr};
use metricsql_parser::functions::RollupFunction;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{field, trace_span, Span};
use crate::execution::utils::{adjust_eval_range, duration_value, get_step};

/// Struct managing state for rollup execution.
pub(crate) struct RollupEvaluator<'a> {
    /// Source expression
    expr: &'a Expr,
    re: Cow<'a, RollupExpr>,
    func: RollupFunction,
    func_handler: RollupHandler,
    is_tracing: bool,
    keep_metric_names: bool,
    /// Max number of timeseries to return
    pub(crate) timeseries_limit: usize,
    pub(crate) is_incr_aggregate: bool,
}

impl<'a> RollupEvaluator<'a> {
    pub(crate) fn new(
        function: RollupFunction,
        handler: RollupHandler,
        // expr may contain:
        // -: RollupFunc(m) if iafc is None
        // - aggrFunc(rollupFunc(m)) if iafc isn't None
        expr: &'a Expr,
        re: Cow<'a, RollupExpr>,
    ) -> Self {
        let mut keep_metric_names = get_keep_metric_names(expr) || function.keep_metric_name();
        Self {
            expr,
            re,
            func: function,
            func_handler: handler,
            keep_metric_names,
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


    fn eval_without_at(&self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        let (offset, ec_new) = adjust_eval_range(self.func, &self.re.offset, ec)?;

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

        if !offset.is_zero() && !rvs.is_empty() {
            let src_timestamps = &rvs[0].timestamps;
            let ofs = offset.as_millis() as i64;
            let shared = Arc::new(
                src_timestamps
                    .iter()
                    .map(|&x| x + ofs)
                    .collect::<Vec<_>>(),
            );
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

        let step = get_step(&self.re.step, ec.step);
        let window = duration_value(&self.re.window, ec.step);

        let mut ec_sq = ec.copy_no_timestamps();
        ec_sq.start -= (window + step + max_silence_interval(ctx)).as_millis() as i64;
        ec_sq.end += step.as_millis() as i64;
        ec_sq.step = step;
        ec_sq.max_points_per_series = ctx.config.max_points_subquery_per_timeseries;
        validate_max_points_per_timeseries(
            ec_sq.start,
            ec_sq.end,
            ec_sq.step,
            ec_sq.max_points_per_series,
        )?;

        // unconditionally align start and end args to step for subquery as Prometheus does.
        (ec_sq.start, ec_sq.end) = align_start_end(ec_sq.start, ec_sq.end, &ec_sq.step);

        // force refresh of timestamps
        // let _ = ec_sq.get_timestamps()?;
        let tss_sq = eval_expr(ctx, &ec_sq, &self.re.expr)?;

        let tss_sq = tss_sq.as_instant_vec(&ec_sq)?;
        if tss_sq.is_empty() {
            return Ok(vec![]);
        }

        let shared_timestamps = Arc::new(get_timestamps(
            ec.start,
            ec.end,
            ec.step,
            ec.max_points_per_series,
        )?);

        let min_staleness_interval = ctx.config.min_staleness_interval;
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

                eval_pre_funcs(&pre_funcs, values, timestamps);
                let mut scanned_total = 0_u64;

                for rc in rcs.iter() {
                    if let Some(tsm) = new_timeseries_map(
                        self.func,
                        self.keep_metric_names,
                        &shared_timestamps,
                        &ts_sq.metric_name,
                    ) {
                        scanned_total += rc.do_timeseries_map(tsm.clone(), values, timestamps)?;
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
                    step = ec.step.as_millis() as u64,
                    window = window.as_millis() as u64,
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

        let min_staleness_interval = ctx.config.min_staleness_interval;
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
            eval_pre_funcs(&pre_funcs, values, timestamps)
        };

        let mut min_timestamp = start;
        if self.func.need_silence_interval() {
            min_timestamp -= MAX_SILENCE_INTERVAL.as_millis() as i64;
        }

        min_timestamp -= (if window > ec.step { window } else { ec.step }).as_millis() as i64;

        // if we don't have additional filters, borrow the existing matchers instead of cloning
        let mut rss = if is_empty_extra_matchers(&ec.enforced_tag_filters) {
            // I hate to clone, but
            let matchers = me.matchers.clone();
            let sq = SearchQuery::new(min_timestamp, ec.end, matchers, ec.max_series);
            ctx.search(sq, ec.deadline)?
        } else {
            let tfss =
                join_matchers_with_extra_filters_owned(&me.matchers, &ec.enforced_tag_filters);
            let sq = SearchQuery::new(min_timestamp, ec.end, tfss, ec.max_series);
            ctx.search(sq, ec.deadline)?
        };

        if rss.is_empty() {
            let dst: Vec<Timeseries> = vec![];
            let tss = merge_timeseries(tss_cached, dst, start, ec)?;
            return Ok(tss);
        }

        let mut ae: Option<&AggregationExpr> = None;
        let mut timeseries_limit = 0usize;

        if let Expr::Aggregation(_ae) = &self.expr {
            timeseries_limit = get_timeseries_limit(_ae)?;
            ae = Some(_ae);
        }

        let rollup_memory_size = self.reserve_rollup_memory(ctx, ec, &rss, timeseries_limit, rcs.len())?;

        defer! {
           ctx.rollup_result_cache.release_memory(rollup_memory_size).unwrap();
           span.record("needed_memory_bytes", rollup_memory_size);
        }

        // Evaluate rollup
        // shadow timestamps
        let shared_timestamps = shared_timestamps.clone(); // TODO: do we need to clone ?
        let ignore_staleness = ec.no_stale_markers;

        let tss = if let Some(ae) = ae {
            self.eval_with_incremental_aggregate(
                ae,
                &mut rss,
                rcs,
                pre_func,
                &shared_timestamps,
                ignore_staleness,
            )
        } else {
            self.eval_no_incremental_aggregate(
                &mut rss,
                rcs,
                pre_func,
                &shared_timestamps,
                ignore_staleness,
            )
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
        rcs: RollupConfigVec,
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
            rcs: RollupConfigVec,
            timestamps: Arc<Vec<i64>>,
            ignore_staleness: bool,
            samples_scanned_total: RelaxedU64Counter,
        }

        let ctx = Context {
            keep_metric_names: self.keep_metric_names,
            func: &self.func,
            iafc,
            rcs,
            timestamps: Arc::clone(shared_timestamps),
            ignore_staleness,
            samples_scanned_total: Default::default(),
        };

        rss.series
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(id, rs)| {
                if !ctx.ignore_staleness {
                    drop_stale_nans(ctx.func, &mut rs.values, &mut rs.timestamps);
                }
                pre_func(&mut rs.values, &rs.timestamps);

                let worker_id = id as u64;
                for rc in ctx.rcs.iter() {
                    if let Some(tsm) = new_timeseries_map(
                        *ctx.func,
                        ctx.keep_metric_names,
                        &ctx.timestamps,
                        &rs.metric,
                    ) {
                        let samples_scanned = rc.do_timeseries_map(tsm.clone(), &rs.values, &rs.timestamps)?;
                        tsm.with_all_series(|tss| ctx.iafc.update_single_timeseries(tss, worker_id));
                        ctx.samples_scanned_total.add(samples_scanned);
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
                    ctx.iafc.update_single_timeseries(&mut ts, worker_id);
                }
                Ok::<(), RuntimeError>(())
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
        rcs: RollupConfigVec,
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
            rcs: RollupConfigVec,
            timestamps: &'a Arc<Vec<i64>>,
            no_stale_markers: bool,
            samples_scanned_total: RelaxedU64Counter,
        }

        let series = Arc::new(Mutex::new(Vec::with_capacity(rss.len() * rcs.len())));
        let ctx = TaskCtx {
            series: Arc::clone(&series),
            keep_metric_names: self.keep_metric_names,
            func: self.func,
            rcs,
            no_stale_markers,
            timestamps: shared_timestamps,
            samples_scanned_total: Default::default(),
        };

        rss.series.par_iter_mut().try_for_each(|rs| {
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
            Ok::<(), RuntimeError>(())
        })?;

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
        timeseries_limit: usize,
        rcs_len: usize,
    ) -> RuntimeResult<usize> {
        // Verify timeseries fit available memory after the rollup.
        // Take into account points from tss_cached.
        let points_per_timeseries = 1 + (ec.end - ec.start).abs().div(ec.step.as_millis() as i64);

        let rss_len = rss.len();
        let timeseries_len = if timeseries_limit > 0 {
            timeseries_limit.min(rss_len)
        } else {
            rss_len
        };
        let rollup_points =
            mul_no_overflow(points_per_timeseries, (timeseries_len * rcs_len) as i64);
        let rollup_memory_size = mul_no_overflow(rollup_points, 16) as usize;

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
                              ec.step.as_millis() as f64 / 1e3
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
    if let Some(tsm) = new_timeseries_map(func, keep_metric_names, timestamps, &rs.metric) {
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
    match eval_expr(ctx, ec, expr) {
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
            rvs[0]
                .metric_name
                .set(&label.name, &label.value);
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
        ts_dst.metric_name.set("rollup", rc.tag_value)
    }
    if !keep_metric_names {
        ts_dst.metric_name.reset_measurement();
    }
    let samples_scanned = rc.exec(&mut ts_dst.values, values_src, timestamps_src)?;
    ts_dst.timestamps = Arc::clone(shared_timestamps);

    Ok(samples_scanned)
}

const FIVE_MINUTES: Duration = Duration::from_secs(5 * 60);

fn max_silence_interval(ctx: &Context) -> Duration {
    if ctx.config.min_staleness_interval.is_zero() {
        FIVE_MINUTES
    } else {
        ctx.config.min_staleness_interval
    }
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


fn get_keep_metric_names(expr: &Expr) -> bool {
    match expr {
        // If expr is a FuncExpr, return its KeepMetricNames flag.
        Expr::Function(fe)  => fe.keep_metric_names,
        Expr::Aggregation(ae) => {
            // This case is possible when optimized aggrFunc calculations are used
            // such as `sum(rate(...))`.
            if ae.args.len() == 1 {
                matches!(&ae.args[0], Expr::Function(fe) if fe.keep_metric_names)
            } else {
                false
            }
        }
        _ => false,
    }
}