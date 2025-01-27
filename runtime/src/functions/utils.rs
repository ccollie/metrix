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
#[inline]
pub(crate) fn max_with_nans(values: &[f64]) -> f64 {
    let max =
        values
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(f64::NAN);
    max
}

pub(crate) fn min_with_nans(values: &[f64]) -> f64 {
    let min =
        values
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(f64::NAN);
    min
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_with_nans_all_valid() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(max_with_nans(&values), 5.0);
    }

    #[test]
    fn test_max_with_nans_with_nans() {
        let values = vec![1.0, f64::NAN, 3.0, 4.0, f64::NAN];
        assert_eq!(max_with_nans(&values), 4.0);
    }

    #[test]
    fn test_max_with_nans_all_nans() {
        let values = vec![f64::NAN, f64::NAN, f64::NAN];
        assert!(max_with_nans(&values).is_nan());
    }

    #[test]
    fn test_max_with_nans_empty() {
        let values: Vec<f64> = vec![];
        assert!(max_with_nans(&values).is_nan());
    }

    #[test]
    fn test_max_with_nans_single_value() {
        let values = vec![42.0];
        assert_eq!(max_with_nans(&values), 42.0);
    }

    #[test]
    fn test_min_with_nans_all_valid() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(min_with_nans(&values), 1.0);
    }

    #[test]
    fn test_min_with_nans_with_nans() {
        let values = vec![1.0, f64::NAN, 3.0, 4.0, f64::NAN];
        assert_eq!(min_with_nans(&values), 1.0);
    }

    #[test]
    fn test_min_with_nans_all_nans() {
        let values = vec![f64::NAN, f64::NAN, f64::NAN];
        assert!(min_with_nans(&values).is_nan());
    }

    #[test]
    fn test_min_with_nans_empty() {
        let values: Vec<f64> = vec![];
        assert!(min_with_nans(&values).is_nan());
    }

    #[test]
    fn test_min_with_nans_single_value() {
        let values = vec![42.0];
        assert_eq!(min_with_nans(&values), 42.0);
    }
}