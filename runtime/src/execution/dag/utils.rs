use std::sync::Arc;
use smallvec::SmallVec;
use tracing::{field, trace_span, Span};

use metricsql_parser::functions::RollupFunction;
use metricsql_parser::prelude::{BinModifier, MetricExpr, Operator};

use crate::execution::binary::{exec_binop, BinaryOpFuncArg};
use crate::execution::dag::NodeArg;
use crate::execution::utils::series_len;
use crate::execution::{eval_number, Context, EvalConfig};
use crate::functions::rollup::{get_rollup_function_handler, RollupHandler};
use crate::functions::transform::extract_labels;
use crate::{RuntimeError, RuntimeResult};
use crate::types::{InstantVector, QueryValue, Timeseries};

pub fn resolve_value(index: usize, value: &mut QueryValue, computed: &mut [QueryValue]) {
    // Note: we return values in this particular way because of an optimization in the evaluator.
    // Consider a call like
    //
    //     sort_desc(2 * (label_set(time(), "foo", "bar", "__name__", "q1")
    //          or label_set(10, "foo", "qwerty", "__name__", "q2")
    //      ) keep_metric_names)
    //
    //  Notice that we have many constant dependencies (args) here. Rather than create a node for
    //  each constant, we just store it directly in the computed array.
    //  We won't ever have non-primitive (string/scalar) constants, so we can just swap the value
    //  of InstantVector/RangeVectors, but we need to preserve the value of constants.
    let dependency = &mut computed[index];
    match dependency {
        QueryValue::InstantVector(_) | QueryValue::RangeVector(_) => {
            std::mem::swap(value, dependency)
        }
        QueryValue::Scalar(v) => *value = QueryValue::Scalar(*v),
        QueryValue::String(s) => *value = QueryValue::from(s.as_str()),
    }
}

pub fn resolve_node_args(
    node_args: &[NodeArg],
    args: &mut Vec<QueryValue>,
    computed: &mut [QueryValue],
) {
    if args.len() != node_args.len() {
        args.resize(node_args.len(), QueryValue::default());
    }
    for (node_arg, arg) in node_args.iter().zip(args.iter_mut()) {
        node_arg.resolve(arg, computed);
    }
}

pub fn resolve_vector(
    index: usize,
    computed: &mut [QueryValue],
) -> RuntimeResult<InstantVector> {
    let dependency = &mut computed[index];
    // use std::mem::take to avoid clone()
    if let QueryValue::InstantVector(vector) = dependency {
        Ok(std::mem::take(vector))
    } else {
        // todo: argument error
        Err(RuntimeError::TypeCastError(format!("expected vector, got {}", dependency.data_type_name())))
    }
}

pub(super) fn resolve_rollup_handler(
    func: RollupFunction,
    arg_nodes: &[NodeArg],
    dependencies: &mut [QueryValue],
) -> RuntimeResult<RollupHandler> {
    if arg_nodes.is_empty() {
        let empty = vec![];
        return get_rollup_function_handler(func, &empty);
    }
    let mut args: SmallVec<[QueryValue; 4]> = SmallVec::with_capacity(arg_nodes.len());
    for arg in arg_nodes {
        let mut value = QueryValue::default();
        arg.resolve(&mut value, dependencies);
        args.push(value);
    }
    get_rollup_function_handler(func, &args)
}

pub(super) fn resolve_at_value(
    arg: &Option<NodeArg>,
    computed: &mut [QueryValue],
) -> RuntimeResult<Option<i64>> {
    if let Some(arg_value) = arg {
        let mut value = QueryValue::default();
        arg_value.resolve(&mut value, computed);
        Ok(Some(get_at_value(&value)?))
    } else {
        Ok(None)
    }
}

pub(super) fn get_at_value(value: &QueryValue) -> RuntimeResult<i64> {
    let v = value.get_scalar().map_err(|_| {
        RuntimeError::TypeCastError(format!(
            "cannot evaluate '{value}' as a timestamp in `@` modifier expression"
        ))
    })?;

    Ok((v * 1000_f64) as i64)
}

pub fn expand_single_value(tss: &mut [Timeseries], ec: &EvalConfig) -> RuntimeResult<()> {
    // expand single-point tss to the original time range.
    let timestamps = ec.get_timestamps()?;
    for ts in tss.iter_mut() {
        ts.timestamps = Arc::clone(&timestamps);
        ts.values = vec![ts.values[0]; timestamps.len()];
    }
    Ok(())
}

pub fn adjust_series_by_offset(rvs: &mut [Timeseries], offset: i64) {
    if offset != 0 && !rvs.is_empty() {
        // Make a copy of timestamps, since they may be used in other values.
        let src_timestamps = &rvs[0].timestamps;
        let dst_timestamps = src_timestamps.iter().map(|x| x + offset).collect();
        let shared = Arc::new(dst_timestamps);
        for ts in rvs.iter_mut() {
            ts.timestamps = Arc::clone(&shared);
        }
    }
}

/// `aggregate_absent_over_time` collapses tss to a single time series with 1 and nan values.
///
/// Values for returned series are set to nan if at least a single tss series contains nan at that point.
/// This means that tss contains a series with non-empty results at that point.
/// This follows Prometheus logic - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2130
pub fn handle_aggregate_absent_over_time(
    ec: &EvalConfig,
    tss: &[Timeseries],
    expr: Option<&MetricExpr>,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs = eval_number(ec, 1.0)?;
    if let Some(expr) = expr {
        let labels = extract_labels(expr);
        for label in labels {
            rvs[0].metric_name.set(&label.name, &label.value);
        }
    }
    if tss.is_empty() {
        return Ok(rvs);
    }
    for i in 0..tss[0].values.len() {
        for ts in tss {
            if ts.values[i].is_nan() {
                rvs[0].values[i] = f64::NAN;
                break;
            }
        }
    }
    Ok(rvs)
}

