use metricsql_common::pool::get_pooled_vec_f64;
use std::ops::Deref;

use crate::common::math::{
    linear_regression, mad, mean, quantile, quantile_sorted, stddev, stdvar,
};
use crate::functions::arg_parse::{get_float_arg, get_series_arg};
use crate::functions::skip_trailing_nans;
use crate::functions::transform::running::{running_avg, running_max, running_min, running_sum};
use crate::functions::transform::utils::expect_transform_args_num;
use crate::functions::transform::{TransformFn, TransformFuncArg};
use crate::functions::utils::get_first_non_nan_index;
use crate::types::Timeseries;
use crate::RuntimeResult;

pub(crate) fn range_avg(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_range_impl(tfa, running_avg)
}

pub(crate) fn range_max(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_range_impl(tfa, running_max)
}

pub(crate) fn range_min(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_range_impl(tfa, running_min)
}

pub(crate) fn range_sum(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_range_impl(tfa, running_sum)
}

fn transform_range_impl(
    tfa: &mut TransformFuncArg,
    running_fn: impl TransformFn,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs = running_fn(tfa)?;
    set_last_values(&mut rvs);
    Ok(rvs)
}

pub(crate) fn transform_range_quantile(
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    let phi = get_float_arg(&tfa.args, 0, Some(0_f64))?;

    let mut series = get_series_arg(&tfa.args, 1, tfa.ec)?;
    range_quantile(phi, &mut series);
    set_last_values(&mut series);
    Ok(series)
}

pub(crate) fn range_median(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    range_quantile(0.5, &mut series);
    Ok(series)
}

pub(crate) fn range_quantile(phi: f64, series: &mut [Timeseries]) {
    let mut values = get_pooled_vec_f64(series.len()).to_vec();

    for ts in series.iter_mut() {
        let mut last_idx = 0;
        values.clear();
        for (i, v) in ts.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            values.push(*v);
            last_idx = i;
        }
        if last_idx > 0 {
            values.sort_by(|a, b| a.total_cmp(b));
            ts.values[last_idx] = quantile_sorted(phi, &values)
        }
    }
    set_last_values(series);
}

pub(crate) fn range_first(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in series.iter_mut() {
        let len = ts.values.len();
        let first = get_first_non_nan_index(&ts.values);
        if first >= len - 1 {
            continue;
        }

        let v_first = ts.values[first];
        for v in ts.values.iter_mut() {
            *v = v_first;
        }
    }

    Ok(series)
}

pub(crate) fn range_last(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    set_last_values(&mut series);
    Ok(series)
}

pub(super) fn set_last_values(tss: &mut [Timeseries]) {
    for ts in tss.iter_mut() {
        let values = skip_trailing_nans(&ts.values);
        if values.is_empty() {
            continue;
        }
        let v_last = ts.values[values.len() - 1];

        for v in ts.values.iter_mut() {
            *v = v_last;
        }
    }
}

pub(crate) fn range_trim_outliers(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let k = get_float_arg(&tfa.args, 0, Some(0_f64))?;

    // Trim samples satisfying the `abs(v - range_median(q)) > k*range_mad(q)`
    let mut rvs = get_series_arg(&tfa.args, 1, tfa.ec)?;
    for ts in rvs.iter_mut() {
        let d_max = k * mad(&ts.values);
        let q_median = quantile(0.5, &ts.values);
        for v in ts.values.iter_mut() {
            if (*v - q_median).abs() > d_max {
                *v = f64::NAN
            }
        }
    }
    Ok(rvs)
}

pub(crate) fn range_trim_spikes(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let mut phi = get_float_arg(&tfa.args, 0, Some(0_f64))?;

    // Trim 100% * (phi / 2) samples with the lowest / highest values per each time series
    phi /= 2.0;
    let phi_upper = 1.0 - phi;
    let phi_lower = phi;
    let mut rvs = get_series_arg(&tfa.args, 1, tfa.ec)?;
    let value_count = rvs[0].values.len();
    let mut values = get_pooled_vec_f64(value_count);

    for ts in rvs.iter_mut() {
        values.clear();

        for v in ts.values.iter() {
            if !v.is_nan() {
                values.push(*v);
            }
        }

        values.sort_by(|a, b| a.total_cmp(b));

        // todo: chili::join or rayon::join
        let v_max = quantile_sorted(phi_upper, &values);
        let v_min = quantile_sorted(phi_lower, &values);
        for v in ts.values.iter_mut() {
            if !v.is_nan() && (*v > v_max || *v < v_min) {
                *v = f64::NAN;
            }
        }
    }

    Ok(rvs)
}

pub(crate) fn range_linear_regression(
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?; // todo: get_matrix
    for ts in series.iter_mut() {
        let timestamps = ts.timestamps.deref();
        if timestamps.is_empty() {
            continue;
        }
        let intercept_timestamp = timestamps[0];
        let (v, k) = linear_regression(&ts.values, timestamps, intercept_timestamp);
        for (value, timestamp) in ts.values.iter_mut().zip(timestamps.iter()) {
            *value = v + k * ((timestamp - intercept_timestamp) as f64 / 1e3)
        }
    }

    Ok(series)
}

pub(crate) fn range_mad(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

    for ts in series.iter_mut() {
        let v = mad(&ts.values);
        for val in ts.values.iter_mut() {
            *val = v
        }
    }

    Ok(series)
}

pub(crate) fn range_stddev(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?; // todo: get_matrix
    for ts in series.iter_mut() {
        let dev = stddev(&ts.values);
        for v in ts.values.iter_mut() {
            *v = dev;
        }
    }
    Ok(series)
}

pub(crate) fn range_stdvar(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?; // todo: get_matrix
    for ts in series.iter_mut() {
        let v = stdvar(&ts.values);
        for v1 in ts.values.iter_mut() {
            *v1 = v
        }
    }
    Ok(series)
}

pub(crate) fn range_normalize(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs: Vec<Timeseries> = vec![];
    let mut selected: Vec<usize> = Vec::with_capacity(tfa.args.len());
    for i in 0..tfa.args.len() {
        let mut series = get_series_arg(&tfa.args, i, tfa.ec)?; // todo: get_matrix
        for (j, ts) in series.iter_mut().enumerate() {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for v in ts.values.iter().copied() {
                if v.is_nan() {
                    continue;
                }
                min = min.min(v);
                max = max.max(v);
            }
            let d = max - min;
            if d.is_infinite() {
                continue;
            }
            for v in ts.values.iter_mut() {
                *v = (*v - min) / d;
            }
            selected.push(j);
        }

        selected.iter().for_each(|i| {
            rvs.push(series.remove(*i));
        });

        selected.clear();
    }
    Ok(rvs)
}

pub(crate) fn range_trim_zscore(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let z = get_float_arg(&tfa.args, 0, None)?.abs();

    // Trim samples with z-score above z.
    let mut rvs = get_series_arg(&tfa.args, 1, tfa.ec)?;
    for ts in rvs.iter_mut() {
        // todo: use rapid calculation methods for mean and stddev.
        let q_stddev = stddev(&ts.values);
        let avg = mean(&ts.values);
        for v in ts.values.iter_mut() {
            let z_curr = (*v - avg).abs() / q_stddev;
            if z_curr > z {
                *v = f64::NAN
            }
        }
    }
    Ok(rvs)
}

pub(crate) fn range_zscore(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in rvs.iter_mut() {
        // todo: use rapid calculation methods for mean and stddev.
        let q_stddev = stddev(&ts.values);
        let avg = mean(&ts.values);
        for v in ts.values.iter_mut() {
            *v = (*v - avg) / q_stddev
        }
    }
    Ok(rvs)
}
