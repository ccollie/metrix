use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeResult, types::Timeseries};
use crate::functions::utils::get_first_non_nan_index;

pub(crate) fn running_avg(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    running_func_impl(tfa, handle_avg)
}

pub(crate) fn running_sum(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    running_func_impl(tfa, handle_sum)
}

pub(crate) fn running_min(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    running_func_impl(tfa, handle_min)
}

pub(crate) fn running_max(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    running_func_impl(tfa, handle_max)
}

#[inline]
fn handle_sum(a: f64, b: f64, _idx: usize) -> f64 {
    a + b
}

#[inline]
fn handle_max(a: f64, b: f64, _idx: usize) -> f64 {
    a.max(b)
}

#[inline]
fn handle_min(a: f64, b: f64, _idx: usize) -> f64 {
    a.min(b)
}

#[inline]
fn handle_avg(a: f64, b: f64, idx: usize) -> f64 {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
    a + (b - a) / (idx + 1) as f64
}

fn running_func_impl(
    tfa: &mut TransformFuncArg,
    rf: fn(a: f64, b: f64, idx: usize) -> f64,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut res = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in res.iter_mut() {
        ts.metric_name.reset_measurement();

        // skip leading NaN values
        let mut start = get_first_non_nan_index(&ts.values);

        // make sure there's at least 2 items remaining
        if ts.values.len() - start < 2 {
            continue;
        }

        let mut prev_value = ts.values[start];
        start += 1;

        for (i, v) in ts.values[start..].iter_mut().enumerate() {
            if !v.is_nan() {
                prev_value = rf(prev_value, *v, i + 1);
            }
            *v = prev_value;
        }
    }

    Ok(res)
}
