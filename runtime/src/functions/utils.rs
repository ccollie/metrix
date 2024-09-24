use std::cmp::Ordering;

use chrono_tz::Tz;

use metricsql_common::time::get_local_tz;

use crate::{RuntimeError, RuntimeResult};

pub fn remove_nan_values_in_place(values: &mut Vec<f64>, timestamps: &mut Vec<i64>) {
    let len = values.len();

    if len == 0 {
        return;
    }

    // Slow path: drop nans from values.
    let mut k = 0;
    let mut nan_found = false;
    for i in 0..len {
        let v = values[i];
        if v.is_nan() {
            values[k] = v;
            timestamps[k] = timestamps[i];
            k += 1;
            nan_found = true;
        }
    }

    if nan_found {
        values.truncate(k);
        timestamps.truncate(k);
    }
}

#[inline]
pub fn get_first_non_nan_index(values: &[f64]) -> usize {
    for (index, v) in values.iter().enumerate() {
        if !v.is_nan() {
            return index;
        }
    }
    0
}

pub fn skip_trailing_nans(values: &[f64]) -> &[f64] {
    let i = values.iter().rposition(|&v| !v.is_nan()).map_or(0, |i| i + 1);
    &values[0..i]
}

#[inline]
pub fn get_last_non_nan_index(values: &[f64]) -> usize {
    values.iter().rposition(|&v| !v.is_nan()).unwrap_or(0)
}

pub(crate) fn float_to_int_bounded(f: f64) -> i64 {
    (f as i64).clamp(i64::MIN, i64::MAX)
}

// todo: move to common
pub(crate) fn parse_timezone(tz_name: &str) -> RuntimeResult<Tz> {
    if tz_name.is_empty() || tz_name.eq_ignore_ascii_case("local") {
        return if let Some(tz) = get_local_tz() {
            Ok(tz)
        } else {
            Err(RuntimeError::ArgumentError(
                "cannot get local timezone".to_string(),
            ))
        };
    }
    match tz_name.parse() {
        Ok(zone) => Ok(zone),
        Err(e) => Err(RuntimeError::ArgumentError(format!(
            "unable to parse tz: {:?}",
            e
        ))),
    }
}

pub(crate) fn float_cmp_with_nans(a: f64, b: f64) -> Ordering {
    if a.is_nan() {
        if b.is_nan() {
            return Ordering::Equal;
        }
        return Ordering::Less;
    } else if b.is_nan() {
        return Ordering::Greater;
    }
    a.total_cmp(&b)
}

pub(crate) fn float_cmp_with_nans_desc(a: f64, b: f64) -> Ordering {
    float_cmp_with_nans(b, a)
}

// todo: can we use SIMD here?
pub(crate) fn max_with_nans(values: &[f64]) -> f64 {
    let mut max = f64::NAN;
    let mut iter = values.iter().skip_while(|v| v.is_nan());
    if let Some(&v) = iter.next() {
        max = v;
        for v in iter {
            if !v.is_nan() && max < *v {
                max = *v;
            }
        }
    }
    max
}

pub(crate) fn min_with_nans(values: &[f64]) -> f64 {
    let mut min = f64::NAN;
    let mut iter = values.iter().skip_while(|v| v.is_nan());
    if let Some(&v) = iter.next() {
        min = v;
        for v in iter {
            if !v.is_nan() && min > *v {
                min = *v;
            }
        }
    }
    min
}
