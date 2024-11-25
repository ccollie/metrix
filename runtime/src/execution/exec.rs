use tracing::info;
use std::fmt::Display;
use std::sync::Arc;
use ahash::AHashSet;
use rayon::join;
use rayon::prelude::IntoParallelRefIterator;
use tracing::{field, trace, trace_span, Span};
use metricsql_common::hash::Signature;
use metricsql_common::prelude::current_time_millis;
use metricsql_parser::ast::{BinaryExpr, Expr, FunctionExpr, Operator, ParensExpr, RollupExpr, UnaryExpr};
use metricsql_parser::functions::{BuiltinFunction, RollupFunction, TransformFunction};
use crate::execution::{Context, EvalConfig};
use crate::functions::rollup::{get_rollup_function_factory, rollup_default, RollupHandler};
use crate::functions::transform::{exec_transform_fn, TransformFuncArg};
use crate::rayon::iter::ParallelIterator;
use crate::prelude::{QueryValue, Timeseries};
use crate::{QueryResult, RuntimeError, RuntimeResult};
use crate::common::math::round_to_decimal_digits;
use crate::execution::aggregate::eval_aggr_func;
use crate::execution::binary::*;
use crate::execution::parser_cache::{ParseCacheResult, ParseCacheValue};
use crate::execution::rollups::RollupEvaluator;
use crate::execution::vectors::vector_vector_binop;
use crate::prelude::binary::scalar_binary_operation;

// see git branch fd75173
type Value = QueryValue;

pub(crate) fn parse_promql_internal(
    context: &Context,
    query: &str,
) -> RuntimeResult<Arc<ParseCacheValue>> {
    let span = trace_span!("parse", cached = field::Empty).entered();
    let (parsed, cached) = context.parse_promql(query)?;
    span.record("cached", cached == ParseCacheResult::CacheHit);
    Ok(parsed)
}

pub(crate) fn exec_internal(
    context: &Context,
    ec: &mut EvalConfig,
    q: &str,
) -> RuntimeResult<(QueryValue, Arc<ParseCacheValue>)> {
    let start_time = current_time_millis();
    if context.stats_enabled() {
        defer! {
            context.query_stats.register_query(q, ec.end - ec.start, start_time)
        }
    }

    ec.validate()?;

    let parsed = parse_promql_internal(context, q)?;

    match (&parsed.expr, &parsed.has_subquery) {
        (Some(expr), has_subquery) => {
            if *has_subquery {
                let _ = ec.get_timestamps()?;
            }

            let qid = context
                .active_queries
                .register(ec, q, Some(start_time));

            defer! {
                context.active_queries.remove(qid);
            }

            let is_tracing = context.trace_enabled();

            let span = if is_tracing {
                let mut query = q.to_string();
                query.truncate(300);

                trace_span!(
                    "execution",
                    query,
                    may_cache = ec.may_cache(),
                    start = ec.start,
                    end = ec.end,
                    series = field::Empty,
                    points = field::Empty,
                    points_per_series = field::Empty
                )
            } else {
                Span::none()
            }
                .entered();

            let rv = eval_expr(context, ec, expr)?;

            if is_tracing {
                let ts_count: usize;
                let series_count: usize;
                match &rv {
                    QueryValue::RangeVector(iv) | QueryValue::InstantVector(iv) => {
                        series_count = iv.len();
                        if series_count > 0 {
                            ts_count = iv[0].timestamps.len();
                        } else {
                            ts_count = 0;
                        }
                    }
                    _ => {
                        ts_count = ec.data_points();
                        series_count = 1;
                    }
                }
                let mut points_per_series = 0;
                if series_count > 0 {
                    points_per_series = ts_count
                }

                let points_count = series_count * points_per_series;
                span.record("series", series_count);
                span.record("points", points_count);
                span.record("points_per_series", points_per_series);
            }

            Ok((rv, Arc::clone(&parsed)))
        }
        _ => {
            panic!("Bug: Invalid parse state")
        }
    }
}

/// executes q for the given config.
pub fn exec(
    context: &Context,
    ec: &mut EvalConfig,
    q: &str,
    is_first_point_only: bool,
) -> RuntimeResult<Vec<QueryResult>> {
    let (rv, parsed) = exec_internal(context, ec, q)?;

    // we ignore empty timeseries
    if let QueryValue::Scalar(val) = rv {
        if val.is_nan() {
            return Ok(vec![]);
        }
    }

    let mut rv = rv.into_instant_vector(ec)?;
    remove_empty_series(&mut rv);
    if rv.is_empty() {
        return Ok(vec![]);
    }

    if is_first_point_only {
        if rv[0].timestamps.len() > 0 {
            let timestamps = Arc::new(vec![rv[0].timestamps[0]]);
            // Remove all the points except the first one from every time series.
            for ts in rv.iter_mut() {
                ts.values.resize(1, f64::NAN);
                ts.timestamps = Arc::clone(&timestamps);
            }
        } else {
            return Ok(vec![]);
        }
    }

    let mut result = timeseries_to_result(&mut rv, parsed.sort_results)?;

    let n = ec.round_digits;
    if n < 100 {
        for r in result.iter_mut() {
            for v in r.values.iter_mut() {
                *v = round_to_decimal_digits(*v, n as i16);
            }
        }
    }

    info!(
        "sorted = {}, round_digits = {}",
        parsed.sort_results, ec.round_digits
    );

    Ok(result)
}


pub(crate) fn timeseries_to_result(
    tss: &mut Vec<Timeseries>,
    may_sort: bool,
) -> RuntimeResult<Vec<QueryResult>> {
    remove_empty_series(tss);
    if tss.is_empty() {
        return Ok(vec![]);
    }

    let mut result: Vec<QueryResult> = Vec::with_capacity(tss.len());
    let mut m: AHashSet<Signature> = AHashSet::with_capacity(tss.len());

    for ts in tss.iter_mut() {
        ts.metric_name.sort_labels();

        let key = ts.metric_name.signature();

        if m.insert(key) {
            let res = QueryResult {
                metric: ts.metric_name.clone(), // todo(perf) into vs clone/take
                values: ts.values.clone(),      // perf) .into vs clone/take
                timestamps: ts.timestamps.as_ref().clone(), // todo(perf): declare field as Rc<Vec<i64>>
            };

            result.push(res);
        } else {
            return Err(RuntimeError::from(format!(
                "duplicate output timeseries: {}",
                ts.metric_name
            )));
        }
    }

    if may_sort {
        result.sort_by(|a, b| a.metric.partial_cmp(&b.metric).unwrap())
    }

    Ok(result)
}

#[inline]
pub(crate) fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    if tss.is_empty() {
        return;
    }
    tss.retain(|ts| !ts.is_all_nans());
}

fn map_error<E: Display>(err: RuntimeError, e: E) -> RuntimeError {
    RuntimeError::General(format!("cannot evaluate {e}: {}", err))
}

pub fn eval_expr(ctx: &Context, ec: &EvalConfig, expr: &Expr) -> RuntimeResult<QueryValue> {
    let tracing = ctx.trace_enabled();
    match expr {
        Expr::StringLiteral(s) => Ok(QueryValue::String(s.to_string())),
        Expr::NumberLiteral(n) => Ok(QueryValue::Scalar(n.value)),
        Expr::Duration(de) => {
            let d = de.value(ec.step);
            let d_sec = d as f64 / 1000_f64;
            Ok(QueryValue::Scalar(d_sec))
        }
        Expr::BinaryOperator(be) => {
            let span = if tracing {
                trace_span!("binary op", "op" = be.op.as_str(), series = field::Empty)
            } else {
                Span::none()
            }
                .entered();

            let rv = exec_binary_op(ctx, ec, be)?;

            span.record("series", rv.len());

            Ok(rv)
        }
        Expr::Parens(pe) => {
            trace_span!("parens");
            let rv = eval_parens_op(ctx, ec, pe)?;
            Ok(rv)
        }
        Expr::MetricExpression(_me) => {
            // todo: avoid this clone 
            let re = RollupExpr::new(expr.clone());
            let handler = RollupHandler::Wrapped(rollup_default);
            let mut executor =
                RollupEvaluator::new(RollupFunction::DefaultRollup, handler, expr, &re);
            let val = executor.eval(ctx, ec).map_err(|err| map_error(err, expr))?;
            Ok(val)
        }
        Expr::Rollup(re) => {
            let handler = RollupHandler::Wrapped(rollup_default);
            let mut executor =
                RollupEvaluator::new(RollupFunction::DefaultRollup, handler, expr, re);
            executor.eval(ctx, ec).map_err(|err| map_error(err, expr))
        }
        Expr::Aggregation(ae) => {
            trace!("aggregate {}()", ae.function.name());
            let rv = eval_aggr_func(ctx, ec, expr, ae).map_err(|err| map_error(err, ae))?;
            trace!("series={}", rv.len());
            Ok(rv)
        }
        Expr::Function(fe) => eval_function(ctx, ec, expr, fe),
        Expr::UnaryOperator(ue) => eval_unary_op(ctx, ec, ue),
        _ => {
            Err(RuntimeError::NotImplemented(format!("No handler for {:?}", expr)))
        },
    }
}

fn eval_function(
    ctx: &Context,
    ec: &EvalConfig,
    expr: &Expr,
    fe: &FunctionExpr,
) -> RuntimeResult<QueryValue> {
    match fe.function {
        BuiltinFunction::Transform(tf) => {
            let span = if ctx.trace_enabled() {
                trace_span!("transform", function = tf.name(), series = field::Empty)
            } else {
                Span::none()
            }
                .entered();

            let rv = eval_transform_func(ctx, ec, fe, tf)?;
            span.record("series", rv.len());

            Ok(QueryValue::InstantVector(rv))
        }
        BuiltinFunction::Rollup(rf) => {
            let nrf = get_rollup_function_factory(rf);
            let (args, re, _) = eval_rollup_func_args(ctx, ec, fe)?;
            let func_handler = nrf(&args)?;
            let mut rollup_handler = RollupEvaluator::new(rf, func_handler, expr, &re);
            // todo: record samples_scanned in span
            let val = rollup_handler
                .eval(ctx, ec)
                .map_err(|err| map_error(err, fe))?;
            Ok(val)
        }
        _ => Err(RuntimeError::NotImplemented(fe.function.name().to_string())),
    }
}

fn eval_parens_op(ctx: &Context, ec: &EvalConfig, pe: &ParensExpr) -> RuntimeResult<QueryValue> {
    if pe.expressions.is_empty() {
        // should not happen !!
        return Err(RuntimeError::Internal(
            "BUG: empty parens expression".to_string(),
        ));
    }
    if pe.expressions.len() == 1 {
        return eval_expr(ctx, ec, &pe.expressions[0]);
    }
    let args = eval_exprs_in_parallel(ctx, ec, &pe.expressions)?;
    let rv = crate::functions::transform::handle_union(args, ec)?;
    let val = QueryValue::InstantVector(rv);
    Ok(val)
}

fn exec_binary_op(ctx: &Context, ec: &EvalConfig, be: &BinaryExpr) -> RuntimeResult<QueryValue> {
    let is_tracing = ctx.trace_enabled();
    // first are cheap binary ops that can be handled without invoking rayon overhead
    let res = match (&be.left.as_ref(), &be.right.as_ref()) {
        // vector op vector needs special handling where both vectors contain selectors
        (Expr::MetricExpression(_), Expr::MetricExpression(_))
        | (Expr::Rollup(_), Expr::Rollup(_))
        | (Expr::MetricExpression(_), Expr::Rollup(_))
        | (Expr::Rollup(_), Expr::MetricExpression(_)) => vector_vector_binop(be, ctx, ec),
        // the following cases can be handled cheaply without invoking rayon overhead (or maybe not :-) )
        (Expr::NumberLiteral(left), Expr::NumberLiteral(right)) => {
            let value = scalar_binary_operation(be.op, left.value, right.value, be.returns_bool())?;
            Ok(Value::Scalar(value))
        }
        (Expr::Duration(left), Expr::Duration(right)) => {
            eval_duration_duration_binop(left, right, be.op, ec.step)
        }
        (Expr::Duration(dur), Expr::NumberLiteral(scalar)) => {
            eval_duration_scalar_binop(dur, scalar.value, be.op, ec.step)
        }
        (Expr::NumberLiteral(scalar), Expr::Duration(dur)) => {
            eval_duration_scalar_binop(dur, scalar.value, be.op, ec.step)
        }
        (Expr::StringLiteral(left), Expr::StringLiteral(right)) => {
            eval_string_string_binop(be.op, left, right, be.returns_bool())
        }
        (left, right) => {
            // maybe chili here instead of rayon
            let (lhs, rhs) = join(|| eval_expr(ctx, ec, left), || eval_expr(ctx, ec, right));

            match (lhs?, rhs?) {
                (QueryValue::Scalar(left), QueryValue::Scalar(right)) => {
                    let value = scalar_binary_operation(be.op, left, right, be.returns_bool())?;
                    Ok(Value::Scalar(value))
                }
                (QueryValue::InstantVector(left), QueryValue::InstantVector(right)) => {
                    exec_vector_vector_binop(ctx, left, right, be.op, &be.modifier)
                }
                (QueryValue::InstantVector(vector), QueryValue::Scalar(scalar)) => {
                    eval_vector_scalar_binop(vector, be.op, scalar, be.returns_bool(), be.should_reset_metric_group(), is_tracing)
                }
                (QueryValue::Scalar(scalar), QueryValue::InstantVector(vector)) => {
                    eval_scalar_vector_binop(scalar, be.op, vector, be.returns_bool(), be.should_reset_metric_group(), is_tracing)
                }
                (QueryValue::String(left), QueryValue::String(right)) => {
                    eval_string_string_binop(be.op, &left, &right, be.returns_bool())
                }
                _ => {
                    return Err(RuntimeError::NotImplemented(format!(
                        "invalid binary operation: {} {} {}",
                        be.left.variant_name(),
                        be.op,
                        be.right.variant_name()
                    )));
                }
            }
        }
    };
    res
}

fn eval_unary_op(ctx: &Context, ec: &EvalConfig, ue: &UnaryExpr) -> RuntimeResult<QueryValue> {
    let is_tracing = ctx.trace_enabled();

    let value = eval_expr(ctx, ec, &ue.expr)?;

    match value {
        QueryValue::Scalar(left) => {
            Ok((-1.0 * left).into())
        }
        QueryValue::InstantVector(vector) => {
            eval_scalar_vector_binop(-1.0, Operator::Mul, vector, false, false, is_tracing)
        }
        _ => {
            Err(RuntimeError::NotImplemented(format!(
                "invalid unary operand: {}", ue.expr.variant_name(),
            )))
        }
    }
}

fn eval_transform_func(
    ctx: &Context,
    ec: &EvalConfig,
    fe: &FunctionExpr,
    func: TransformFunction,
) -> RuntimeResult<Vec<Timeseries>> {
    let args = if func == TransformFunction::Union {
        eval_exprs_in_parallel(ctx, ec, &fe.args)?
    } else {
        eval_exprs_sequentially(ctx, ec, &fe.args)?
    };
    let mut tfa = TransformFuncArg {
        ec,
        args,
        keep_metric_names: func.keep_metric_name(),
    };
    exec_transform_fn(func, &mut tfa).map_err(|err| map_error(err, fe))
}

#[inline]
fn eval_args(ctx: &Context, ec: &EvalConfig, args: &[Expr]) -> RuntimeResult<Vec<Value>> {
    // see if we can evaluate all args in parallel
    // todo: if rayon in cheap enough, we can avoid the check and always go parallel
    // todo: see https://docs.rs/rayon/1.0.3/rayon/iter/trait.IndexedParallelIterator.html#method.with_min_len
    let count = args.iter()
        .skip(5)
        .filter(|arg| !matches!(arg, Expr::StringLiteral(_) | Expr::Duration(_) | Expr::NumberLiteral(_)))
        .count();
    
    if count > 1 {
        eval_exprs_in_parallel(ctx, ec, args)
    } else {
        eval_exprs_sequentially(ctx, ec, args)
    }
}

pub(super) fn eval_exprs_sequentially(
    ctx: &Context,
    ec: &EvalConfig,
    args: &[Expr],
) -> RuntimeResult<Vec<Value>> {
    if args.is_empty() {
        return Ok(Vec::new());
    }
    args.iter()
        .map(|expr| eval_expr(ctx, ec, expr))
        .collect::<RuntimeResult<Vec<Value>>>()
}

pub(super) fn eval_exprs_in_parallel(
    ctx: &Context,
    ec: &EvalConfig,
    args: &[Expr],
) -> RuntimeResult<Vec<Value>> {
    if args.is_empty() {
        return Ok(Vec::new());
    }
    let res: RuntimeResult<Vec<Value>> = args
        .par_iter()
        .map(|expr| eval_expr(ctx, ec, expr))
        .collect();

    res
}

pub(super) fn eval_rollup_func_args(
    ctx: &Context,
    ec: &EvalConfig,
    fe: &FunctionExpr,
) -> RuntimeResult<(Vec<Value>, RollupExpr, usize)> {
    let mut re: RollupExpr = Default::default();
    // todo: i dont think we can have a empty arg_idx_for_optimization
    let rollup_arg_idx = fe.arg_idx_for_optimization().expect("rollup_arg_idx is None");

    if fe.args.len() <= rollup_arg_idx {
        let msg = format!(
            "expecting at least {} args to {}; got {} args; expr: {}",
            rollup_arg_idx + 1,
            fe.name(),
            fe.args.len(),
            fe
        );
        return Err(RuntimeError::from(msg));
    }

    let mut args = Vec::with_capacity(fe.args.len());
    // todo(perf): extract rollup arg first, then evaluate the rest in parallel
    for (i, arg) in fe.args.iter().enumerate() {
        if i == rollup_arg_idx {
            re = get_rollup_expr_arg(arg)?;
            args.push(QueryValue::Scalar(f64::NAN)); // placeholder
            continue;
        }
        let value = eval_expr(ctx, ec, arg).map_err(|err| {
            let msg = format!("cannot evaluate arg #{} for {}: {}", i + 1, fe, err);
            RuntimeError::ArgumentError(msg)
        })?;

        args.push(value);
    }

    Ok((args, re, rollup_arg_idx))
}

// todo: COW
fn get_rollup_expr_arg(arg: &Expr) -> RuntimeResult<RollupExpr> {
    match arg {
        Expr::Rollup(re) => {
            let mut re = re.clone();
            if !re.for_subquery() {
                // Return standard rollup if it doesn't contain subquery.
                return Ok(re);
            }

            match &re.expr.as_ref() {
                Expr::MetricExpression(_) => {
                    // Convert me[w:step] -> default_rollup(me)[w:step]

                    let arg = Expr::Rollup(RollupExpr::new(*re.expr.clone()));

                    match FunctionExpr::default_rollup(arg) {
                        Err(e) => Err(RuntimeError::General(format!("{:?}", e))),
                        Ok(fe) => {
                            re.expr = Box::new(Expr::Function(fe));
                            Ok(re)
                        }
                    }
                }
                _ => {
                    // arg contains subquery.
                    Ok(re)
                }
            }
        }
        _ => {
            // Wrap non-rollup arg into RollupExpr.
            Ok(RollupExpr::new(arg.clone()))
        }
    }
}