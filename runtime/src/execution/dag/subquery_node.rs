use std::sync::Arc;

use tracing::{field, trace_span, Span};

use metricsql_parser::ast::{DurationExpr, Expr};
use metricsql_parser::prelude::{RollupExpr, RollupFunction};

use crate::execution::dag::builder::DAGBuilder;
use crate::execution::dag::utils::{
    adjust_series_by_offset, expand_single_value, get_at_value, handle_aggregate_absent_over_time,
    resolve_at_value, resolve_rollup_handler,
};
use crate::execution::dag::{DAGNode, ExecutableNode, NodeArg};
use crate::execution::utils::{
    adjust_eval_range, duration_value, get_step, process_series_in_parallel,
};
use crate::execution::{
    align_start_end, eval_number, get_timestamps, validate_max_points_per_timeseries, Context,
    EvalConfig,
};
use crate::functions::rollup::{
    eval_pre_funcs, get_rollup_configs, RollupHandler, MAX_SILENCE_INTERVAL,
};
use crate::{RuntimeError, RuntimeResult};
use crate::types::{QueryValue, Timeseries, Timestamp};

/// Node for non-selector sub-queries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SubqueryNode {
    pub func: RollupFunction,
    pub(crate) func_handler: RollupHandler,
    pub expr: Expr,
    pub expr_node: Box<DAGNode>,
    pub keep_metric_names: bool,
    pub step: Option<DurationExpr>,
    pub offset: Option<DurationExpr>,
    pub window: Option<DurationExpr>,
    pub at: Option<i64>,
    pub at_arg: Option<NodeArg>,
    pub args: Vec<NodeArg>,
    pre_exec_called: bool,
}

impl SubqueryNode {
    pub(crate) fn new(
        expr: &Expr,
        re: &RollupExpr,
        rf: RollupFunction,
        handler: RollupHandler,
    ) -> RuntimeResult<Self> {
        let mut node = SubqueryNode {
            func: rf,
            func_handler: handler,
            expr: expr.clone(),
            expr_node: Box::<DAGNode>::default(),
            keep_metric_names: get_keep_metric_names(expr),
            step: re.step.clone(),
            offset: re.offset.clone(),
            window: re.window.clone(),
            ..Default::default()
        };

        let expr_dag = DAGBuilder::compile(re.expr.as_ref().clone())?;
        node.expr_node = Box::new(expr_dag);

        Ok(node)
    }

    fn get_at_timestamp(&self) -> RuntimeResult<Option<Timestamp>> {
        // value was set in pre-execute
        if self.at.is_some() {
            return Ok(self.at);
        }
        // possibly const value set during build, and pre-execute not called
        if let Some(at_arg) = &self.at_arg {
            return match at_arg {
                NodeArg::Value(v) => {
                    let value = get_at_value(v)?;
                    Ok(Some(value))
                }
                _ => {
                    unreachable!("@ timestamp should be a const value or resolved in pre-execute")
                }
            };
        }
        Ok(None)
    }
}

impl ExecutableNode for SubqueryNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.at = resolve_at_value(&self.at_arg, dependencies)?;
        self.func_handler = resolve_rollup_handler(self.func, &self.args, dependencies)?;
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let val = if let Some(at_timestamp) = self.get_at_timestamp()? {
            let mut ec_new = ec.copy_no_timestamps();
            ec_new.start = at_timestamp;
            ec_new.end = at_timestamp;

            // todo: simply return the scalar. The value would be expanded later in the evaluator
            let mut val = self.eval_without_at(ctx, &ec_new)?;
            expand_single_value(&mut val, ec)?;
            val
        } else {
            self.eval_without_at(ctx, ec)?
        };

        Ok(QueryValue::from(val))
    }
}

impl SubqueryNode {
    fn eval_without_at(
        &mut self,
        ctx: &Context,
        ec: &EvalConfig,
    ) -> RuntimeResult<Vec<Timeseries>> {
        // TODO: determine whether to use rollup result cache here.

        let span = if ctx.trace_enabled() {
            trace_span!(
                "subquery",
                expr = self.expr.to_string().as_str(),
                rollup = self.func.name(),
                series = field::Empty,
                source_series = field::Empty,
                samples_scanned = field::Empty,
            )
        } else {
            Span::none()
        }
        .entered();

        let (offset, ec_new) = adjust_eval_range(&self.func, &self.offset, ec)?;

        let ec = ec_new;
        // todo: validate that step and window are non negative
        let step = get_step(&self.step, ec.step);
        let window = duration_value(&self.window, ec.step);

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
        let (start, end) = align_start_end(ec_sq.start, ec_sq.end, ec_sq.step);
        ec_sq.start = start;
        ec_sq.end = end;

        let tss_sq = self.exec_expression(ctx, &ec_sq)?;
        if tss_sq.is_empty() {
            return Ok(vec![]);
        }

        let shared_timestamps = Arc::new(get_timestamps(
            ec.start,
            ec.end,
            ec.step,
            ec.max_points_per_series,
        )?);

        // avoid issues later with borrow checker
        let is_absent_over_time = self.func == RollupFunction::AbsentOverTime;

        let min_staleness_interval = ctx.config.min_staleness_interval.num_milliseconds() as usize;
        let (rcs, pre_funcs) = get_rollup_configs(
            self.func,
            &self.func_handler,
            &self.expr,
            ec.start,
            ec.end,
            ec.step,
            window,
            ec.max_points_per_series,
            min_staleness_interval,
            ec.lookback_delta,
            &shared_timestamps,
        )?;

        let (mut res, samples_scanned_total) = process_series_in_parallel(
            &tss_sq,
            move |ts_sq: &Timeseries, values: &mut [f64], timestamps: &[i64]| -> RuntimeResult<(Vec<Timeseries>, u64)> {
                let mut res = Vec::with_capacity(ts_sq.len());

                eval_pre_funcs(&pre_funcs, values, timestamps);
                let mut scanned_total = 0_u64;

                for rc in rcs.iter() {
                    let (samples_scanned, mut series) = rc.process_rollup(
                        self.func,
                        &ts_sq.metric_name,
                        self.keep_metric_names,
                        values,
                        timestamps,
                        &shared_timestamps,
                    )?;

                    scanned_total += samples_scanned;
                    res.append(&mut series);
                }

                Ok((res, scanned_total))
            },
        )?;

        if !span.is_disabled() {
            span.record("series", res.len());
            span.record("source_series", tss_sq.len());
            span.record("samples_scanned", samples_scanned_total);
        }

        if is_absent_over_time {
            res = handle_aggregate_absent_over_time(&ec, &res, None)?
        }

        adjust_series_by_offset(&mut res, offset);
        Ok(res)
    }

    fn exec_expression(
        &mut self,
        ctx: &Context,
        ec: &EvalConfig,
    ) -> RuntimeResult<Vec<Timeseries>> {
        // force refresh of timestamps
        let _ = ec.get_timestamps();
        let val = self.expr_node.execute(ctx, ec)?;
        match val {
            QueryValue::InstantVector(v) => Ok(v),
            QueryValue::Scalar(n) => Ok(eval_number(ec, n)?),
            _ => {
                let msg = format!("cannot cast {} to an instant vector", val.data_type());
                Err(RuntimeError::TypeCastError(msg))
            }
        }
    }
}

fn get_keep_metric_names(expr: &Expr) -> bool {
    match expr {
        Expr::Function(f) => f.keep_metric_names,
        Expr::Aggregation(ae) => {
            // Extract rollupFunc(...) from aggrFunc(rollupFunc(...)).
            // This case is possible when optimized aggregate calculations are used
            // such as `sum(rate(...))`
            if ae.args.len() == 1 {
                return get_keep_metric_names(&ae.args[0]);
            }
            // ae.keep_metric_names
            false
        }
        _ => false,
    }
}
