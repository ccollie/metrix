use std::fmt::Display;

use rayon::join;
use rayon::prelude::IntoParallelRefIterator;
use tracing::{field, trace, trace_span, Span};

use metricsql_parser::ast::{BinaryExpr, Expr, FunctionExpr, ParensExpr, RollupExpr};
use metricsql_parser::functions::{BuiltinFunction, RollupFunction, TransformFunction};
use crate::execution::{Context, EvalConfig};
use crate::functions::rollup::{get_rollup_function_factory, rollup_default, RollupHandler};
use crate::functions::transform::{exec_transform_fn, handle_union, TransformFuncArg};
use crate::rayon::iter::ParallelIterator;
use crate::prelude::{QueryValue, Timeseries};
use crate::{RuntimeError, RuntimeResult};
use crate::execution::aggregate::eval_aggr_func;
use crate::execution::binary::*;
use crate::execution::rollups::RollupExecutor;
use crate::execution::vectors::vector_vector_binop;
use crate::prelude::binary::scalar_binary_operation;

// see git branch fd75173
type Value = QueryValue;

fn map_error<E: Display>(err: RuntimeError, e: E) -> RuntimeError {
    RuntimeError::General(format!("cannot evaluate {e}: {}", err))
}

pub fn exec_expr(ctx: &Context, ec: &EvalConfig, expr: &Expr) -> RuntimeResult<QueryValue> {
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
                RollupExecutor::new(RollupFunction::DefaultRollup, handler, expr, &re);
            let val = executor.eval(ctx, ec).map_err(|err| map_error(err, expr))?;
            Ok(val)
        }
        Expr::Rollup(re) => {
            let handler = RollupHandler::Wrapped(rollup_default);
            let mut executor =
                RollupExecutor::new(RollupFunction::DefaultRollup, handler, expr, re);
            executor.eval(ctx, ec).map_err(|err| map_error(err, expr))
        }
        Expr::Aggregation(ae) => {
            trace!("aggregate {}()", ae.function.name());
            let rv = eval_aggr_func(ctx, ec, expr, ae).map_err(|err| map_error(err, ae))?;
            trace!("series={}", rv.len());
            Ok(rv)
        }
        Expr::Function(fe) => eval_function_op(ctx, ec, expr, fe),
        _ => unimplemented!(),
    }
}

fn eval_function_op(
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
            let mut rollup_handler = RollupExecutor::new(rf, func_handler, expr, &re);
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
        return exec_expr(ctx, ec, &pe.expressions[0]);
    }
    let args = eval_exprs_in_parallel(ctx, ec, &pe.expressions)?;
    let rv = handle_union(args, ec)?;
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
            let (lhs, rhs) = join(|| exec_expr(ctx, ec, left), || exec_expr(ctx, ec, right));

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
        .map(|expr| exec_expr(ctx, ec, expr))
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
        .map(|expr| exec_expr(ctx, ec, expr))
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
        let value = exec_expr(ctx, ec, arg).map_err(|err| {
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