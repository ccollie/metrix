use itertools::izip;
use metricsql_parser::prelude::Value;

use crate::{QueryResult, RuntimeResult};
use crate::types::{MetricName, QueryValue, Timeseries};

pub fn test_results_equal(result: &[QueryResult], result_expected: &[QueryResult]) {
    assert_eq!(
        result.len(),
        result_expected.len(),
        "unexpected timeseries count; got {}; want {}",
        result.len(),
        result_expected.len()
    );

    let mut i = 0;
    for (actual, expected) in result.iter().zip(result_expected) {
        test_metric_names_equal(&actual.metric, &expected.metric, i);
        test_rows_equal(
            &actual.values,
            &actual.timestamps,
            &expected.values,
            &expected.timestamps,
        );
        i = i + 1;
    }
}

pub fn test_query_values_equal(actual: &QueryValue, expected: &QueryValue) {
    use QueryValue::*;

    let actual_type = actual.value_type();
    let expected_type = expected.value_type();
    assert_eq!(
        actual_type, expected_type,
        "unexpected value type; got {}; want {}",
        actual_type, expected_type
    );
    match (&actual, &expected) {
        (Scalar(actual), Scalar(expected)) => {
            assert_eq!(
                actual, expected,
                "unexpected scalar value; got {}; want {}",
                actual, expected
            );
        }
        (String(actual), String(expected)) => {
            assert_eq!(
                actual, expected,
                "unexpected string value; got {}; want {}",
                actual, expected
            );
        }
        (InstantVector(actual), InstantVector(expected)) => {
            test_timeseries_equal(actual, expected);
        }
        (RangeVector(actual), RangeVector(expected)) => {
            test_timeseries_equal(actual, expected);
        }
        (left, right) => {
            assert_eq!(
                left, right,
                "unexpected value; got {}; want {}",
                left, right
            );
        }
    }
}

pub fn test_rows_equal(
    values: &[f64],
    timestamps: &[i64],
    values_expected: &[f64],
    timestamps_expected: &[i64],
) {
    compare_values(values, values_expected).unwrap();
    assert_eq!(timestamps.len(), timestamps_expected.len(),
               "unexpected timestamps.len(); got {}; want {}\ntimestamps=\n{:?}\ntimestamps_expected=\n{:?}",
               timestamps.len(), timestamps_expected.len(), timestamps, timestamps_expected);

    assert_eq!(
        values.len(),
        timestamps.len(),
        "values.len() doesn't match timestamps.len(); got {} vs {}",
        values.len(),
        timestamps.len()
    );

    for (i, val, val_expected, ts, ts_expected) in izip!(
        0..1000,
        values.iter(),
        values_expected.iter(),
        timestamps.iter(),
        timestamps_expected.iter()
    ) {
        assert_eq!(
            ts, ts_expected,
            "unexpected timestamp at timestamps[{}]; got {}; want {}\ntimestamps=\n{:?}\ntimestamps_expected=\n{:?}",
            i, ts, ts_expected, timestamps, timestamps_expected
        );

        if val.is_nan() {
            assert!(val_expected.is_nan(),
                    "unexpected nan value at values[{}]; want %{}\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                    i, val_expected, values, values_expected);
            continue;
        }
        if val_expected.is_nan() {
            assert!(val.is_nan(), "unexpected value at values[{}]; got {}; want nan\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                    i, val, values, values_expected);
            continue;
        }

        // Compare values with the reduced precision because of different precision errors
        // on different OS/architectures. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1738
        // and https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1653
        if (val - val_expected).abs() / (val_expected).abs() > 1e-13 {
            panic!("unexpected value at values[{}]; got {}; want {}\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                   i, val, val_expected, values, values_expected)
        }
    }
}

pub fn test_metric_names_equal(mn: &MetricName, expected: &MetricName, pos: usize) {
    assert_eq!(
        mn.measurement, expected.measurement,
        "unexpected MetricGroup at #{pos}; got {}; want {}; got metric={mn}, expected metric={expected}",
        mn.measurement, expected.measurement,
    );

    assert_eq!(
        mn.labels.len(),
        expected.labels.len(),
        "unexpected labels count at #{pos}; got {}; want {}; got metric={mn}, expected metric={expected}",
        mn.labels.len(),
        expected.labels.len(),
    );

    for (i, (tag, tag_expected)) in mn.labels.iter().zip(expected.labels.iter()).enumerate() {
        assert_eq!(
            tag.name, tag_expected.name,
            "unexpected tag key at #{pos}, {i}; got {}; want {}; got metric={mn}, expected metric={expected}",
            tag.name, tag_expected.name
        );

        assert_eq!(
            tag.value, tag_expected.value,
            "unexpected tag value at #{pos},{i}; got {}; want {}; got metric={mn}, expected metric={expected}",
            tag.value, tag_expected.value,
        )
    }
}

pub fn test_timeseries_equal(tss: &[Timeseries], tss_expected: &[Timeseries]) {
    assert_eq!(
        tss.len(),
        tss_expected.len(),
        "unexpected timeseries count; got {}; want {}",
        tss.len(),
        tss_expected.len()
    );

    for (i, ts) in tss.iter().enumerate() {
        let ts_expected = &tss_expected[i];
        test_metric_names_equal(&ts.metric_name, &ts_expected.metric_name, i);
        test_rows_equal(
            &ts.values,
            &ts.timestamps,
            &ts_expected.values,
            &ts_expected.timestamps,
        )
    }
}

pub const EPSILON: f64 = 1e-14;

pub fn compare_values(actual: &[f64], expected: &[f64]) -> RuntimeResult<()> {
    assert_eq!(
        actual.len(),
        expected.len(),
        "unexpected number of values; got {}; want {}",
        actual.len(),
        expected.len()
    );
    for (i, (got, wanted)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            compare_floats(*wanted, *got),
            "unexpected value comparing slice; expected {v2}, got {v1} at index {i}\nactual={actual:?}\nexpected={expected:?}",
            v1 = *got,
            v2 = *wanted
        );
    }
    Ok(())
}

pub fn compare_floats(expected: f64, actual: f64) -> bool {
    match (expected.is_finite(), actual.is_finite()) {
        (true, true) => {
            let eps = (actual - expected).abs();
            eps <= EPSILON
        }
        (false, false) => {
            if expected.is_nan() {
                return actual.is_nan();
            }
            if expected == f64::INFINITY {
                return actual == f64::INFINITY;
            }
            actual == f64::NEG_INFINITY
        }
        _ => false,
    }
}

// Implementation of get_lines function
fn get_lines(input: String) -> Vec<String> {
    input
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.starts_with("#"))
        .collect()
}
