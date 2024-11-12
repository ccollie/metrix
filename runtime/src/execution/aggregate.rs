use std::ops::Deref;

use tracing::{field, trace_span, Span};
use metricsql_parser::ast::{AggregationExpr, Expr, FunctionExpr, MetricExpr};
use metricsql_parser::functions::BuiltinFunction;
use crate::common::cpu::num_cpus;
use crate::execution::{Context, EvalConfig};
use crate::execution::new_eval::{eval_exprs_in_parallel, eval_rollup_func_args};
use crate::execution::rollups::RollupExecutor;
use crate::functions::aggregate::{exec_aggregate_fn, AggrFuncArg, IncrementalAggregationHandler};
use crate::functions::rollup::get_rollup_function_factory;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::QueryValue;

pub(super) fn eval_aggr_func(
    ctx: &Context,
    ec: &EvalConfig,
    expr: &Expr,
    ae: &AggregationExpr,
) -> RuntimeResult<QueryValue> {
    let span = if ctx.trace_enabled() {
        // done this way to avoid possible string alloc in the case where
        // logging is disabled
        let name = ae.name();
        trace_span!("aggregate", name, series = field::Empty)
    } else {
        Span::none()
    }
        .entered();

    // todo: ensure that this is serialized otherwise the contained block will not be executed
    if ae.can_incrementally_eval() && IncrementalAggregationHandler::handles(ae.function) {
        if let Some(fe) = try_get_arg_rollup_func_with_metric_expr(ae)? {
            // There is an optimized path for calculating `AggrFuncExpr` over: RollupFunc
            // over MetricExpr.
            // The optimized path saves RAM for aggregates over big number of time series.
            let (args, re, _) = eval_rollup_func_args(ctx, ec, &fe)?;

            let rf = match fe.function {
                BuiltinFunction::Rollup(rf) => rf,
                _ => {
                    // should not happen
                    unreachable!(
                        "Expected a rollup function in aggregation. Found \"{}\"",
                        fe.function
                    )
                }
            };

            let nrf = get_rollup_function_factory(rf);
            let func_handler = nrf(&args)?;
            let mut executor = RollupExecutor::new(rf, func_handler, expr, &re);
            executor.timeseries_limit = get_timeseries_limit(ae)?;
            executor.is_incr_aggregate = true;

            let val = executor.eval(ctx, ec)?;
            span.record("series", val.len());
            return Ok(val);
        }
    }

    let args = eval_exprs_in_parallel(ctx, ec, &ae.args)?;
    let mut afa = AggrFuncArg {
        args,
        ec,
        modifier: &ae.modifier,
        limit: get_timeseries_limit(ae)?,
    };

    match exec_aggregate_fn(ae.function, &mut afa) {
        Ok(res) => {
            span.record("series", res.len());
            Ok(QueryValue::InstantVector(res))
        }
        Err(e) => {
            let res = format!("cannot evaluate {}: {:?}", ae, e);
            Err(RuntimeError::General(res))
        }
    }
}

// todo: move to metricsql crate - optimize phase
fn try_get_arg_rollup_func_with_metric_expr(
    ae: &AggregationExpr,
) -> RuntimeResult<Option<FunctionExpr>> {
    if !ae.can_incrementally_eval() {
        return Ok(None);
    }

    if ae.args.len() != 1 {
        return Ok(None);
    }

    let expr = &ae.args[0];
    // Make sure e contains one of the following:
    // - metricExpr
    // - metricExpr[d]
    // -: RollupFunc(metricExpr)
    // -: RollupFunc(metricExpr[d])

    fn create_func(
        me: &MetricExpr,
        expr: &Expr,
        name: &str,
        for_subquery: bool,
    ) -> RuntimeResult<Option<FunctionExpr>> {
        if me.is_empty() || for_subquery {
            return Ok(None);
        }

        let func_name = if name.is_empty() {
            "default_rollup"
        } else {
            name
        };

        match FunctionExpr::from_single_arg(func_name, expr.clone()) {
            Err(e) => Err(RuntimeError::General(format!(
                "Error creating function {func_name}: {:?}",
                e
            ))),
            Ok(fe) => Ok(Some(fe)),
        }
    }

    match expr {
        Expr::MetricExpression(me) => create_func(me, expr, "", false),
        Expr::Rollup(re) => {
            match re.expr.deref() {
                Expr::MetricExpression(me) => {
                    // e = metricExpr[d]
                    create_func(me, expr, "", re.for_subquery())
                }
                _ => Ok(None),
            }
        }
        Expr::Function(fe) => {
            match fe.function {
                BuiltinFunction::Rollup(_) => {
                    if let Some(arg) = fe.arg_for_optimization() {
                        match arg.deref() {
                            Expr::MetricExpression(me) => create_func(me, expr, &fe.name(), false),
                            Expr::Rollup(re) => {
                                match &*re.expr {
                                    Expr::MetricExpression(me) => {
                                        if me.is_empty() || re.for_subquery() {
                                            Ok(None)
                                        } else {
                                            // e = RollupFunc(metricExpr[d])
                                            // todo: use COW to avoid clone
                                            Ok(Some(fe.clone()))
                                        }
                                    }
                                    _ => Ok(None),
                                }
                            }
                            _ => Ok(None),
                        }
                    } else {
                        // Incorrect number of args for rollup func.
                        // TODO: this should be an error
                        // all rollup functions should have a value for this
                        Ok(None)
                    }
                }
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

pub(super) fn get_timeseries_limit(aggr_expr: &AggregationExpr) -> RuntimeResult<usize> {
    // Incremental aggregates require holding only num_cpus() timeseries in memory.
    let timeseries_len = usize::from(num_cpus()?);
    let res = if aggr_expr.limit > 0 {
        // There is an explicit limit on the number of output time series.
        timeseries_len * aggr_expr.limit
    } else {
        // Increase the number of timeseries for non-empty group list: `aggr() by (something)`,
        // since each group can have own set of time series in memory.
        timeseries_len * 1000
    };

    Ok(res)
}