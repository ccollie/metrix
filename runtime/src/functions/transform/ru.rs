use crate::execution::eval_number;
use crate::functions::transform::TransformFuncArg;
use crate::prelude::Timeseries;
use crate::types::QueryValue;
use crate::{RuntimeError, RuntimeResult};
use crate::functions::transform::utils::{clamp_min};

/// Resource utilization
pub(crate) fn transform_ru(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args_len = tfa.args.len();
    if args_len != 2 {
        return Err(RuntimeError::ArgumentError("unexpected number of arguments: want 2".into()));
    }
    let (first, second) = tfa.args.split_at_mut(1);
    match (first, second) {
        ([QueryValue::Scalar(left)], [QueryValue::Scalar(right)]) => {
            let value = ru(*left, *right);
            eval_number(tfa.ec, value)
        }
        ([QueryValue::InstantVector(left)], [QueryValue::InstantVector(right)]) => {
            calculate_vector_vector(left, right)?;
            Ok(std::mem::take(left))
        }
        ([QueryValue::InstantVector(left)], [QueryValue::Scalar(right)]) => {
            calculate_vector_scalar(left, *right);
            Ok(std::mem::take(left))
        }
        ([QueryValue::Scalar(left)], [QueryValue::InstantVector(right)]) => {
            calculate_vector_scalar(right, *left);
            Ok(std::mem::take(right))
        }
        _ => {
            Err(RuntimeError::ArgumentError("unexpected argument types".into()))
        }
    }
}

fn calculate_vector_vector(free_series: &mut [Timeseries], max_series: &Vec<Timeseries>) -> RuntimeResult<()> {
    if free_series.len() != max_series.len() {
        return Err(RuntimeError::ArgumentError("both vectors must have the same length".into()));
    }
    for (free_ts, max_ts) in free_series.iter_mut().zip(max_series) {
        for (free_value, max_value) in free_ts.values.iter_mut().zip(max_ts.values.iter()) {
            *free_value = ru(*free_value, *max_value);
        }
    }
    Ok(())
}
fn calculate_vector_scalar(free_series: &mut [Timeseries], max_scalar: f64) {
    for ts in free_series.iter_mut() {
        for v in ts.values.iter_mut() {
            *v = ru(*v, max_scalar);
        }
    }
}

fn ru(free_value: f64, max_value: f64) -> f64 {
    // ru(freev, maxv) = clamp_min(maxv - clamp_min(freev, 0), 0) / clamp_min(maxv, 0) * 100
    let used = clamp_min(max_value - clamp_min(free_value, 0.0), 0.0);
    let max =  clamp_min(max_value, 0.0);
    let utilization = used / max;
    utilization * 100_f64
}
