use metricsql_common::pool::get_pooled_vec_f64;
use num_traits::Pow;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::ops::DerefMut;
use crate::types::Timestamp;

/// STALE_NAN_BITS is bit representation of Prometheus staleness mark (aka stale NaN).
/// This mark is put by Prometheus at the end of time series for improving staleness detection.
/// See https://www.robustperception.io/staleness-and-promql
/// StaleNaN is a special NaN value, which is used as Prometheus staleness mark.
pub const STALE_NAN_BITS: u64 = 0x7ff0000000000002;

/// is_stale_nan returns true if f represents Prometheus staleness mark.
#[inline]
pub fn is_stale_nan(f: f64) -> bool {
    f.to_bits() == STALE_NAN_BITS
}

pub static IQR_PHIS: [f64; 2] = [0.25, 0.75];

/// mode_no_nans returns mode for a.
///
/// It is expected that a doesn't contain NaNs.
///
/// The function modifies contents for a, so the caller must prepare it accordingly.
///
/// See https://en.wikipedia.org/wiki/Mode_(statistics)
pub fn mode_no_nans(prev_value: f64, a: &mut [f64]) -> f64 {
    let mut prev_value = prev_value;
    if a.is_empty() {
        return prev_value;
    }
    a.sort_by(|a, b| a.total_cmp(b));
    let mut j: i32 = -1;
    let mut i: i32 = 0;

    let mut d_max = 0;
    let mut mode = prev_value;
    for v in a.iter_mut() {
        if prev_value == *v {
            i += 1;
            continue;
        }
        let d = i - j;
        if d > d_max || mode.is_nan() {
            d_max = d;
            mode = prev_value;
        }
        j = i;
        i += 1;
        prev_value = *v;
    }
    let d = a.len() as i32 - j;
    if d > d_max || mode.is_nan() {
        mode = prev_value
    }
    mode
}

pub(crate) fn mean(values: &[f64]) -> f64 {
    let mut sum: f64 = 0.0;
    let mut n = 0;
    for v in values.iter() {
        if v.is_nan() {
            continue;
        }
        sum += v;
        n += 1;
    }
    sum / n as f64
}

pub(crate) fn stdvar(values: &[f64]) -> f64 {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation

    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        // Fast path.
        return 0.0;
    }
    let mut avg: f64 = 0.0;
    let mut count: usize = 0;
    let mut q: f64 = 0.0;
    for v in values {
        if v.is_nan() {
            continue;
        }
        count += 1;
        let avg_new = avg + (*v - avg) / count as f64;
        q += (*v - avg) * (*v - avg_new);
        avg = avg_new
    }
    if count == 0 {
        return f64::NAN;
    }
    q / count as f64
}

pub(crate) fn stddev(values: &[f64]) -> f64 {
    let std_var = stdvar(values);
    std_var.sqrt()
}

/// quantiles calculates the given phis from originValues without modifying origin_values, appends
/// them to qs and returns the result.
pub(crate) fn quantiles(qs: &mut [f64], phis: &[f64], origin_values: &[f64]) {
    if origin_values.len() <= 64 {
        let mut vec = SmallVec::<f64, 64>::new();
        prepare_tv_for_quantile_float64(&mut vec, origin_values);
        return quantiles_sorted(qs, phis, &vec);
    }

    let mut block = get_pooled_vec_f64(phis.len());
    let a = block.deref_mut();
    prepare_for_quantile_float64(a, origin_values);
    quantiles_sorted(qs, phis, a)
}

/// calculates the given phi from origin_values without modifying origin_values
pub(crate) fn quantile(phi: f64, origin_values: &[f64]) -> f64 {
    // todo: tinyvec ?
    let mut block = get_pooled_vec_f64(origin_values.len());
    prepare_for_quantile_float64(&mut block, origin_values);
    quantile_sorted(phi, &block)
}

/// prepare_for_quantile_float64 copies items from src to dst but removes NaNs and sorts the dst
fn prepare_for_quantile_float64(dst: &mut Vec<f64>, src: &[f64]) {
    dst.extend(src.iter().filter(|v| !v.is_nan()));
    dst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
}

/// copies items from src to dst but removes NaNs and sorts the dst
fn prepare_tv_for_quantile_float64(dst: &mut SmallVec<f64, 64>, src: &[f64]) {
    for v in src.iter() {
        if v.is_nan() {
            continue;
        }
        dst.push(*v);
    }
    dst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
}

/// calculates the given phis over a sorted list of values, appends them to qs and returns the result.
///
/// It is expected that values won't contain NaN items.
/// The implementation mimics Prometheus implementation for compatibility's sake.
pub(crate) fn quantiles_sorted(qs: &mut [f64], phis: &[f64], values: &[f64]) {
    // todo(perf): rayon ?
    for (phi, qs) in phis.iter().zip(qs.iter_mut()) {
        *qs = quantile_sorted(*phi, values);
    }
}

/// quantile_sorted calculates the given quantile over a sorted list of values.
///
/// It is expected that values won't contain NaN items.
/// The implementation mimics Prometheus implementation for compatibility's sake.
pub(crate) fn quantile_sorted(phi: f64, values: &[f64]) -> f64 {
    if values.is_empty() || phi.is_nan() {
        return f64::NAN;
    }
    if phi < 0.0 {
        return f64::NEG_INFINITY;
    }
    if phi > 1.0 {
        return f64::INFINITY;
    }
    let n = values.len();
    let rank = phi * (n - 1) as f64;

    let lower_index = std::cmp::max(0, rank.floor() as usize);
    let upper_index = std::cmp::min(n - 1, lower_index + 1);

    let weight = rank - rank.floor();
    values[lower_index] * (1.0 - weight) + values[upper_index] * weight
}

pub(crate) fn median(values: &[f64]) -> f64 {
    quantile(0.5, values)
}

pub(crate) fn mad(values: &[f64]) -> f64 {
    // See https://en.wikipedia.org/wiki/Median_absolute_deviation
    let med = median(values);
    let mut ds = get_pooled_vec_f64(values.len());
    for v in values.iter() {
        ds.push((v - med).abs())
    }
    median(&ds)
}

pub(crate) fn linear_regression(
    values: &[f64],
    timestamps: &[Timestamp],
    intercept_time: i64,
) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    if are_const_values(values) {
        return (values[0], 0.0);
    }

    // See https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    let mut v_sum: f64 = 0.0;
    let mut t_sum: f64 = 0.0;
    let mut tv_sum: f64 = 0.0;
    let mut tt_sum: f64 = 0.0;

    for (ts, v) in timestamps.iter().zip(values.iter()) {
        let dt = (ts - intercept_time) as f64 / 1e3_f64;
        v_sum += v;
        t_sum += dt;
        tv_sum += dt * v;
        tt_sum += dt * dt
    }

    let mut k: f64 = 0.0;
    let n = n as f64;
    let t_diff = tt_sum - t_sum * t_sum / n;
    if t_diff.abs() >= 1e-6 {
        // Prevent from incorrect division for too small t_diff values.
        k = (tv_sum - t_sum * v_sum / n) / t_diff;
    }
    let v = v_sum / n - k * t_sum / n;
    (v, k)
}

pub(crate) fn are_const_values(values: &[f64]) -> bool {
    if values.len() <= 1 {
        return true;
    }
    let mut v_prev = values[0];
    for v in &values[1..] {
        if *v != v_prev {
            return false;
        }
        v_prev = *v
    }

    true
}

/// rounds f to the given number of decimal digits after the point.
///
/// See also RoundToSignificantFigures.
pub fn round_to_decimal_digits(f: f64, digits: i16) -> f64 {
    if is_stale_nan(f) {
        // Do not modify stale nan mark value.
        return f;
    }
    if digits <= -100 || digits >= 100 {
        return f;
    }
    let m = 10_f64.pow(digits);
    let mult = (f * m).round();
    mult / m
}
