use std::ops::Deref;
use std::str::FromStr;
use std::sync::Arc;
use tracing::{field, trace_span, Span};

use metricsql::ast::{AggregationExpr, Expr, FunctionExpr, MetricExpr};
use metricsql::common::{AggregateModifier, ValueType};
use metricsql::functions::{AggregateFunction, BuiltinFunction, RollupFunction, Volatility};
use metricsql::prelude::Value;

use crate::context::Context;
use crate::eval::arg_list::ArgList;
use crate::eval::rollup::{compile_rollup_func_args, RollupEvaluator};
use crate::functions::aggregate::{
    exec_aggregate_fn, try_get_incremental_aggr_handler, AggrFuncArg,
};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::utils::num_cpus;
use crate::{EvalConfig, QueryValue};

use super::{Evaluator, ExprEvaluator};

pub(super) fn create_aggr_evaluator(ae: &AggregationExpr) -> RuntimeResult<ExprEvaluator> {
    match try_get_incremental_aggr_handler(&ae.name) {
        Some(handler) => {
            match try_get_arg_rollup_func_with_metric_expr(ae)? {
                Some(fe) => {
                    // There is an optimized path for calculating `Expression::AggrFuncExpr` over: RollupFunc
                    // over Expression::MetricExpr.
                    // The optimized path saves RAM for aggregates over big number of time series.
                    let (args, re) = compile_rollup_func_args(&fe)?;
                    let expr = Expr::Aggregation(ae.clone());
                    let func = get_rollup_function(&fe)?;

                    let mut res = RollupEvaluator::create_internal(func, &re, expr, args)?;

                    res.timeseries_limit = get_timeseries_limit(ae)?;
                    res.is_incr_aggregate = true;
                    res.incremental_aggr_handler = Some(handler);

                    return Ok(ExprEvaluator::Rollup(res));
                }
                _ => {}
            }
        }
        _ => {}
    }

    Ok(ExprEvaluator::Aggregate(AggregateEvaluator::new(ae)?))
}

pub struct AggregateEvaluator {
    pub expr: String,
    pub function: AggregateFunction, // smol_str ?
    args: ArgList,
    /// optional modifier such as `by (...)` or `without (...)`.
    modifier: Option<AggregateModifier>,
    /// Max number of timeseries to return
    pub limit: usize,
    pub may_sort_results: bool,
}

impl AggregateEvaluator {
    pub fn new(ae: &AggregationExpr) -> RuntimeResult<Self> {
        // todo: remove unwrap and return a Result
        let function = AggregateFunction::from_str(&ae.name).unwrap();
        let signature = function.signature();
        let args = ArgList::new(&signature, &ae.args)?;

        Ok(Self {
            args,
            function,
            modifier: ae.modifier.clone(),
            limit: ae.limit,
            expr: ae.to_string(),
            may_sort_results: function.may_sort_results(),
        })
    }

    pub fn is_idempotent(&self) -> bool {
        self.volatility() != Volatility::Volatile && self.args.all_const()
    }
}

impl Evaluator for AggregateEvaluator {
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let span = if ctx.trace_enabled() {
            // done this way to avoid possible string alloc in the case where
            // logging is disabled
            let name = self.function.name();
            trace_span!("aggregate", name, series = field::Empty)
        } else {
            Span::none()
        }
        .entered();

        let args = self.args.eval(ctx, ec)?;

        //todo: use tinyvec for args
        let mut afa = AggrFuncArg::new(ec, args, &self.modifier, self.limit);
        match exec_aggregate_fn(self.function, &mut afa) {
            Ok(res) => {
                span.record("series", res.len());
                Ok(QueryValue::InstantVector(res))
            }
            Err(e) => {
                let res = format!("cannot evaluate {}: {:?}", self.expr, e);
                Err(RuntimeError::General(res))
            }
        }
    }

    fn volatility(&self) -> Volatility {
        self.args.volatility
    }
}

impl Value for AggregateEvaluator {
    fn value_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

fn try_get_arg_rollup_func_with_metric_expr(
    ae: &AggregationExpr,
) -> RuntimeResult<Option<FunctionExpr>> {
    if ae.args.len() != 1 {
        return Ok(None);
    }
    let e = &ae.args[0];
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

        let func_name = if name.len() == 0 {
            "default_rollup"
        } else {
            name
        };

        match FunctionExpr::from_single_arg(func_name, expr.clone()) {
            Err(e) => Err(RuntimeError::General(format!(
                "Error creating function {}: {:?}",
                func_name, e
            ))),
            Ok(fe) => Ok(Some(fe)),
        }
    }

    return match &ae.args[0] {
        Expr::MetricExpression(me) => return create_func(me, e, "", false),
        Expr::Rollup(re) => {
            match re.expr.deref() {
                Expr::MetricExpression(me) => {
                    // e = metricExpr[d]
                    create_func(me, e, "", re.for_subquery())
                }
                _ => Ok(None),
            }
        }
        Expr::Function(fe) => {
            let function: BuiltinFunction;
            match BuiltinFunction::from_str(&fe.name) {
                Err(_) => return Err(RuntimeError::UnknownFunction(fe.name.to_string())),
                Ok(f) => function = f,
            };

            match function {
                BuiltinFunction::Rollup(_) => {
                    match fe.get_arg_for_optimization() {
                        None => {
                            // Incorrect number of args for rollup func.
                            // TODO: this should be an error
                            // all rollup functions should have a value for this
                            return Ok(None);
                        }
                        Some(arg) => {
                            match arg.deref() {
                                Expr::MetricExpression(me) => create_func(me, e, &fe.name, false),
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
                        }
                    }
                }
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    };
}

fn get_rollup_function(fe: &FunctionExpr) -> RuntimeResult<RollupFunction> {
    match RollupFunction::from_str(&fe.name) {
        Ok(rf) => Ok(rf),
        _ => {
            // should not happen
            Err(RuntimeError::General(format!(
                "Invalid rollup function \"{}\"",
                fe.name
            )))
        }
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
