#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use std::time::Duration;

    use crate::execution::{compile_expression, Context, EvalConfig};
    use crate::{test_query_values_equal, Deadline};
    use crate::types::{MetricName, QueryValue, Timeseries};

    const NAN: f64 = f64::NAN;
    const START: i64 = 1000000_i64;
    const END: i64 = 2000000_i64;
    const STEP: Duration = Duration::from_millis(200000_u64);
    const TEST_ITERATIONS: usize = 3;

    fn run_query(q: &str) -> QueryValue {
        let mut ec = EvalConfig::new(START, END, STEP);
        ec.max_series = 1000;
        ec.max_points_per_series = 15000;
        ec.round_digits = 100;
        let timeout = Duration::from_secs(60);
        ec.deadline = Deadline::new(timeout).unwrap();
        let context = Context::default();
        let expr = metricsql_parser::prelude::parse(q).unwrap();
        let mut node = compile_expression(&expr).unwrap();
        node.execute(&context, &mut ec).unwrap()
    }

    fn test_query(q: &str, result_expected: QueryValue) {
        for _ in 0..TEST_ITERATIONS {
            let result = run_query(q);
            test_query_values_equal(&result, &result_expected);
        }
    }

    fn make_result(vals: &[f64]) -> QueryValue {
        let mut start = 1000000;
        let vals = Vec::from(vals);
        let mut timestamps: Vec<i64> = Vec::with_capacity(vals.len());
        (0..vals.len()).for_each(|_| {
            timestamps.push(start);
            start += 200000;
        });

        let series = Timeseries {
            metric_name: MetricName::default(),
            values: vals,
            timestamps: Arc::new(timestamps),
        };
        QueryValue::InstantVector(vec![series])
    }

    fn assert_result_eq(q: &str, values: &[f64]) {
        let r = make_result(values);
        test_query(q, r);
    }

    #[test]
    fn simple_number() {
        let q = "123";
        test_query(q, QueryValue::from(123.0));
    }

    #[test]
    fn simple_arithmetic() {
        test_query("-1+2 * 3 ^ 4+5%6", QueryValue::from(166.0));
    }

    #[test]
    fn simple_string() {
        let q = r#""foobar""#;
        test_query(q, QueryValue::from("foobar"));
    }

    #[test]
    fn simple_string_concat() {
        let q = r#""bar" + "baz""#;
        test_query(q, QueryValue::from("barbaz"));
    }

    #[test]
    fn scalar_vector_arithmetic() {
        let q = "scalar(-1)+2 *vector(3) ^ scalar(4)+5";
        assert_result_eq(q, &[166.0, 166.0, 166.0, 166.0, 166.0, 166.0]);
    }

    #[test]
    fn vector_eq_bool() {
        // vector(1) == bool time()
        assert_result_eq("vector(1) == bool time()", &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn vector_eq_scalar() {
        assert_result_eq("vector(1) == time()", &[NAN, NAN, NAN, NAN, NAN, NAN]);
    }

    #[test]
    fn compare_to_nan_right() {
        test_query("1 != bool NAN", QueryValue::from(1.0));
    }

    #[test]
    fn compare_to_nan_left() {
        test_query("NAN != bool 1", QueryValue::from(1.0));
    }

    #[test]
    fn function_cmp_scalar() {
        assert_result_eq("time() >= bool 2", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_simple() {
        assert_result_eq("1e3/time()*2*9*7", &[126.0, 105.0, 90.0, 78.75, 70.0, 63.0]);
    }

    #[test]
    fn test_and() {
        assert_result_eq(
            "time() and 2",
            &[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "time() and time() > 1300",
            &[NAN, NAN, 1400.0, 1600.0, 1800.0, 2000.0],
        );
    }

    #[test]
    fn test_time() {
        assert_result_eq(
            "time()",
            &[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "time()[300s]",
            &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "time()[300s] offset 100s",
            &[800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0],
        );
        assert_result_eq(
            "time()[300s:100s] offset 100s",
            &[900.0, 1100.0, 1300.0, 1500.0, 1700.0, 1900.0],
        );
        assert_result_eq(
            "time()[300:100] offset 100",
            &[900.0, 1100.0, 1300.0, 1500.0, 1700.0, 1900.0],
        );

        assert_result_eq(
            "time() offset 0s",
            &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "time()[:100s] offset 0s",
            &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "time()[:100s] offset 100s",
            &[900.0, 1100.0, 1300.0, 1500.0, 1700.0, 1900.0],
        );

        assert_result_eq(
            "time()[:100] offset 0",
            &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "time() offset 1h40s0ms",
            &[-2800.0, -2600.0, -2400.0, -2200.0, -2000.0, -1800.0],
        );

        assert_result_eq(
            "time() offset 3640",
            &[-2800.0, -2600.0, -2400.0, -2200.0, -2000.0, -1800.0],
        );
        assert_result_eq(
            "time() offset -1h40s0ms",
            &[4600.0, 4800.0, 5000.0, 5200.0, 5400.0, 5600.0],
        );
        assert_result_eq(
            "time() offset -100s",
            &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );

        assert_result_eq(
            "time()[1.5i:0.5i] offset 0.5i",
            &[900.0, 1100.0, 1300.0, 1500.0, 1700.0, 1900.0],
        );

        assert_result_eq("1e3/time()*2*9*7", &[126.0, 105.0, 90.0, 78.75, 70.0, 63.0]);

        assert_result_eq(
            "time() + time()",
            &[2000.0, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0],
        );
    }

    #[test]
    fn test_transform_functions() {
        assert_result_eq(
            "clamp_min(time(), -time()+2500)",
            &[1500.0, 1300.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );

        assert_result_eq(
            "atanh(tanh((2000-time())/1000))",
            &[1.0, 0.8000000000000002, 0.6, 0.4000000000000001, 0.2, 0.0],
        );
    }

    #[test]
    fn absent() {
        let all_absent = make_result(&[NAN, NAN, NAN, NAN, NAN, NAN]);
        let q = "absent(time())";
        test_query(q, all_absent.clone());

        let q = "absent(123)";
        test_query(q, all_absent.clone());

        let q = "absent(vector(scalar(123)))";
        test_query(q, all_absent.clone());

        assert_result_eq("absent(NaN)", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    fn assert_label_value(mn: &MetricName, tag: &str, expected: &str) {
        let tag_value = mn.label_value(tag.into()).unwrap();
        assert_eq!(tag_value.as_str(), expected);
    }

    #[test]
    fn rollup_rate() {
        let q = "rollup_rate((2000-time())[600s])";
        let actual = run_query(q);
        if let QueryValue::InstantVector(iv) = actual {
            let r1 = &iv[0];
            let r2 = &iv[1];
            let r3 = &iv[2];

            assert_label_value(&r1.metric_name, "rollup", "avg");
            assert_eq!(&r1.values, &[5_f64, 4.0, 3.0, 2.0, 1.0, 0.0]);

            assert_label_value(&r2.metric_name, "rollup", "max");
            assert_eq!(&r2.values, &[6_f64, 5.0, 4.0, 3.0, 2.0, 1.0]);

            assert_label_value(&r3.metric_name, "rollup", "min");
            assert_eq!(&r3.values, &[4_f64, 3.0, 2.0, 1.0, 0.0, -1.0]);
        } else {
            panic!("expected instant vector");
        }
    }

    #[test]
    fn vector_multiplied_by_scalar() {
        assert_result_eq(
            "sum(time()) * 2",
            &[2000.0, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0],
        );
    }
}
