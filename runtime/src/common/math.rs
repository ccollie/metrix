use std::cmp::Ordering;
use std::ops::DerefMut;

use tinyvec::TinyVec;

use lib::get_pooled_vec_f64;

/// mode_no_nans returns mode for a.
///
/// It is expected that a doesn't contain NaNs.
///
/// The function modifies contents for a, so the caller must prepare it accordingly.
///
/// See https://en.wikipedia.org/wiki/Mode_(statistics)
pub fn mode_no_nans(prev_value: f64, a: &mut Vec<f64>) -> f64 {
    let mut prev_value = prev_value;
    if a.len() == 0 {
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
    return mode;
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
    return sum / n as f64;
}

pub(crate) fn stdvar(values: &[f64]) -> f64 {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation

    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
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
    return q / count as f64;
}

pub(crate) fn stddev(values: &[f64]) -> f64 {
    let std_var = stdvar(values);
    return std_var.sqrt();
}

/// quantiles calculates the given phis from originValues without modifying origin_values, appends
/// them to qs and returns the result.
pub(crate) fn quantiles(qs: &mut [f64], phis: &[f64], origin_values: &[f64]) {
    if origin_values.len() <= 64 {
        let mut vec = tiny_vec!([f64; 64]);
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
fn prepare_tv_for_quantile_float64(dst: &mut TinyVec<[f64; 64]>, src: &[f64]) {
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
    for (phi, qs) in phis.iter().zip(qs.iter_mut()) {
        *qs = quantile_sorted(*phi, values);
    }
}

/// quantile_sorted calculates the given quantile over a sorted list of values.
///
/// It is expected that values won't contain NaN items.
/// The implementation mimics Prometheus implementation for compatibility's sake.
pub(crate) fn quantile_sorted(phi: f64, values: &[f64]) -> f64 {
    if values.len() == 0 || phi.is_nan() {
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
    let upper_index = std::cmp::min(n - 1, lower_index + 1) as usize;

    let weight = rank - rank.floor();
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight;
}

pub(crate) fn median(values: &[f64]) -> f64 {
    quantile(0.5, values)
}
