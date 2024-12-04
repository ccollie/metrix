#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::execution::exec;
    use crate::execution::{Context, EvalConfig};
    use crate::functions::parse_timezone;
    use crate::functions::transform::get_timezone_offset;
    use crate::{test_results_equal, Deadline, QueryResult};
    use std::time::Duration;
    use metricsql_common::label::Label;
    use metricsql_parser::parser::parse;
    use metricsql_parser::prelude::utils::is_likely_invalid;
    use crate::types::{MetricName, Timestamp};

    const NAN: f64 = f64::NAN;
    const INF: f64 = f64::INFINITY;
    const NEG_INF: f64 = f64::NEG_INFINITY;

    const START: Timestamp = 1000000_i64;
    const END: Timestamp = 2000000_i64;
    const STEP: Duration = Duration::from_millis(200000_u64);

    const TIMESTAMPS_EXPECTED: [Timestamp; 6] = [1000000, 1200000, 1400000, 1600000, 1800000, 2000000];

    fn make_result(vals: &[f64]) -> QueryResult {
        let mut start = 1000000;
        let vals = Vec::from(vals);
        let mut timestamps: Vec<Timestamp> = Vec::with_capacity(vals.len());
        (0..vals.len()).for_each(|_| {
            timestamps.push(start);
            start += 200000;
        });

        QueryResult {
            metric: MetricName::default(),
            values: vals,
            timestamps,
        }
    }

    const TEST_ITERATIONS: usize = 3;

    fn test_query(q: &str, expected: Vec<QueryResult>) {
        let mut ec = EvalConfig::new(START, END, STEP);
        ec.max_series = 1000;
        ec.max_points_per_series = 15000;
        ec.round_digits = 100;
        let timeout = Duration::from_millis(1000);
        ec.deadline = Deadline::new(timeout).unwrap();
        let context = Context::default(); // todo: have a test gated default;
        for _ in 0..TEST_ITERATIONS {
            match exec(&context, &mut ec, q, false) {
                Ok(result) => test_results_equal(&result, &expected),
                Err(e) => {
                    panic!("{}", e)
                }
            }
        }
    }

    fn assert_result_eq(q: &str, values: &[f64]) {
        let r = make_result(values);
        test_query(q, vec![r]);
    }

    #[test]
    fn simple_number() {
        let q = "123";
        assert_result_eq(q, &[123.0, 123.0, 123.0, 123.0, 123.0, 123.0]);
    }

    #[test]
    fn duration_constant() {
        let q = "1h23m5s";
        assert_result_eq(q, &[4985.0, 4985.0, 4985.0, 4985.0, 4985.0, 4985.0]);
    }

    #[test]
    fn num_with_suffix_1() {
        let n = 123e6f64;
        assert_result_eq("123M", &[n, n, n, n, n, n]);
    }

    #[test]
    fn num_with_suffix_2() {
        let n = 1.23e12;
        assert_result_eq("1.23TB", &[n, n, n, n, n, n]);
    }

    #[test]
    fn num_with_suffix_3() {
        let n = 1.23 * (1 << 20) as f64;
        assert_result_eq("1.23Mib", &[n, n, n, n, n, n]);
    }

    #[test]
    fn num_with_suffix_4() {
        let n = 1.23 * (1 << 20) as f64;
        assert_result_eq("1.23mib", &[n, n, n, n, n, n]);
    }

    #[test]
    fn num_with_suffix_5() {
        let n = 1234e6;
        assert_result_eq("1_234M", &[n, n, n, n, n, n]);
    }

    #[test]
    fn simple_arithmetic() {
        assert_result_eq(
            "-1+2 *3 ^ 4+5%6",
            &[166.0, 166.0, 166.0, 166.0, 166.0, 166.0],
        );
    }

    #[test]
    fn simple_string() {
        let q = r#""foobar""#;
        test_query(q, vec![])
    }

    #[test]
    fn scalar_vector_arithmetic() {
        let q = "scalar(-1)+2 *vector(3) ^ scalar(4)+5";
        assert_result_eq(q, &[166.0, 166.0, 166.0, 166.0, 166.0, 166.0]);
    }

    #[test]
    fn scalar_string_non_number() {
        let q = r#"scalar("fooobar")"#;
        test_query(q, vec![])
    }

    #[test]
    fn scalar_string_num() {
        assert_result_eq(
            r#"scalar("-12.34")"#,
            &[-12.34, -12.34, -12.34, -12.34, -12.34, -12.34],
        );
    }

    //
    #[test]
    fn seeded_rand_normal() {
        let q = "rand_normal(0)";
        assert_result_eq(
            q,
            &[
                0.7128130103834549,
                0.85833144681790008,
                -2.4362438894664367,
                0.1633442588933682,
                -1.2750102039848832,
                1.2871709906391997,
            ],
        );
    }

    #[test]
    fn bitmap_and() {
        assert_result_eq(
            "bitmap_and(0xB3, 0x11)",
            &[17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
        );
        assert_result_eq(
            "bitmap_and(time(), 0x11)",
            &[0.0, 16.0, 16.0, 0.0, 0.0, 16.0],
        );

        test_query("bitmap_and(NaN, 1)", vec![]);
    }

    #[test]
    fn bitmap_or() {
        assert_result_eq(
            "bitmap_or(0xA2, 0x11)",
            &[179.0, 179.0, 179.0, 179.0, 179.0, 179.0],
        );
        assert_result_eq(
            "bitmap_or(time(), 0x11)",
            &[1017.0, 1201.0, 1401.0, 1617.0, 1817.0, 2001.0],
        );

        test_query("bitmap_or(NaN, 1)", vec![]);
    }

    #[test]
    fn bitmap_xor() {
        assert_result_eq(
            "bitmap_xor(0xB3, 0x11)",
            &[162.0, 162.0, 162.0, 162.0, 162.0, 162.0],
        );
        assert_result_eq(
            "bitmap_xor(time(), 0x11)",
            &[1017.0, 1185.0, 1385.0, 1617.0, 1817.0, 1985.0],
        );

        test_query("bitmap_xor(NaN, 1)", vec![]);
    }

    #[test]
    fn timezone_offset_utc() {
        assert_result_eq(r#"timezone_offset("UTC")"#, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_timezone_offset_america_new_york() {
        let q = r#"timezone_offset("America/New_York")"#;
        let tz = parse_timezone("America/New_York").unwrap();
        let offset = get_timezone_offset(&tz, TIMESTAMPS_EXPECTED[0]);
        assert_ne!(offset, None);

        let off = offset.unwrap() as f64;
        let r = make_result(&[off, off, off, off, off, off]);
        let result_expected: Vec<QueryResult> = vec![r];
        test_query(q, result_expected)
    }

    #[test]
    fn timezone_offset_local() {
        let q = r#"timezone_offset("Local")"#;
        let tz = parse_timezone("Local").unwrap();
        let offset = get_timezone_offset(&tz, TIMESTAMPS_EXPECTED[0]).unwrap();
        let off = offset as f64;
        let r = make_result(&[off, off, off, off, off, off]);
        test_query(q, vec![r]);
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
    fn test_offset() {
        // (a, b) offset 0s
        let q = r#"sort((label_set(time(), "foo", "bar"), label_set(time()+10, "foo", "baz")) offset 0s)"#;
        let mut r1 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[1010_f64, 1210.0, 1410.0, 1610.0, 1810.0, 2010.0]);
        r2.metric.set("foo", "baz");
        test_query(q, vec![r1, r2]);

        // (a, b) offset 100s
        let q = r#"sort((label_set(time(), "foo", "bar"), label_set(time()+10, "foo", "baz")) offset 100s)"#;
        let mut r1 = make_result(&[800_f64, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[810_f64, 1010.0, 1210.0, 1410.0, 1610.0, 1810.0]);
        r2.metric.set("foo", "baz");
        test_query(q, vec![r1, r2]);

        // (a offset 100s, b offset 50s
        let q = r#"sort((label_set(time() offset 100s, "foo", "bar"), label_set(time()+10, "foo", "baz") offset 50s))"#;
        let mut r1 = make_result(&[800_f64, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[810_f64, 1010.0, 1210.0, 1410.0, 1610.0, 1810.0]);
        r2.metric.set("foo", "baz");
        test_query(q, vec![r1, r2]);

        // (a offset 100s, b offset 50s) offset 400s
        let q = r#"sort((label_set(time() offset 100s, "foo", "bar"), label_set(time()+10, "foo", "baz") offset 50s) offset 400s)"#;
        let mut r1 = make_result(&[400_f64, 600.0, 800.0, 1000.0, 1200.0, 1400.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[410_f64, 610.0, 810.0, 1010.0, 1210.0, 1410.0]);
        r2.metric.set("foo", "baz");
        test_query(q, vec![r1, r2]);

        // (a offset -100s, b offset -50s) offset -400s
        let q = r#"sort((label_set(time() offset -100s, "foo", "bar"), label_set(time()+10, "foo", "baz") offset -50s) offset -400s)"#;
        let mut r1 = make_result(&[1400_f64, 1600.0, 1800.0, 2000.0, 2200.0, 2400.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[1410_f64, 1610.0, 1810.0, 2010.0, 2210.0, 2410.0]);
        r2.metric.set("foo", "baz");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn f_1h() {
        assert_result_eq("1h", &[3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0]);
    }

    #[test]
    fn sum_over_time() {
        assert_result_eq(
            "sum_over_time(time()[1h]) / 1h",
            &[-3.5, -2.5, -1.5, -0.5, 0.5, 1.5],
        );
    }

    #[test]
    fn timestamp() {
        assert_result_eq(
            "timestamp(123)",
            &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "timestamp(time())",
            &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "timestamp(456/time()+123)",
            &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "timestamp(time()>=1600)",
            &[NAN, NAN, NAN, 1600.0, 1800.0, 2000.0],
        );

        let q = r#"timestamp(alias(time()>=1600.0,"foo"))"#;
        assert_result_eq(q, &[NAN, NAN, NAN, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn tlast_change_over_time() {
        let q = "tlast_change_over_time(
        time()[1h]
        )";
        assert_result_eq(q, &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);

        let q = "tlast_change_over_time(
            (time() >=bool 1600)[1h]
        )";
        assert_result_eq(q, &[NAN, NAN, NAN, 1600.0, 1600.0, 1600.0]);
    }

    #[test]
    fn tlast_change_over_time_miss() {
        let q = "tlast_change_over_time(1[1h])";
        test_query(q, vec![])
    }

    #[test]
    fn timestamp_with_name() {
        let q = r#"timestamp_with_name(alias(time()>=1600.0,"foo"))"#;
        let mut r = make_result(&[NAN, NAN, NAN, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("foo");
        test_query(q, vec![r]);
    }

    #[test]
    fn time() {
        assert_result_eq("time()/100", &[10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
        assert_result_eq("1e3/time()*2*9*7", &[126.0, 105.0, 90.0, 78.75, 70.0, 63.0]);
    }

    #[test]
    fn minute() {
        assert_result_eq("minute()", &[16.0, 20.0, 23.0, 26.0, 30.0, 33.0]);
        assert_result_eq("minute(30*60+time())", &[46.0, 50.0, 53.0, 56.0, 0.0, 3.0]);
    }

    #[test]
    fn minute_series_with_nans() {
        assert_result_eq(
            "minute(time() <= 1200 or time() > 1600)",
            &[16.0, 20.0, NAN, NAN, 30.0, 33.0],
        );
    }

    #[test]
    fn day_of_month() {
        assert_result_eq(
            "day_of_month(time()*1e4)",
            &[26.0, 19.0, 12.0, 5.0, 28.0, 20.0],
        );
    }

    #[test]
    fn day_of_week() {
        assert_result_eq("day_of_week(time()*1e4)", &[0.0, 2.0, 5.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn day_of_year() {
        assert_result_eq(
            "day_of_year(time()*1e4)",
            &[116.0, 139.0, 163.0, 186.0, 209.0, 232.0],
        );
    }

    #[test]
    fn days_in_month() {
        assert_result_eq(
            "days_in_month(time()*2e4)",
            &[31.0, 31.0, 30.0, 31.0, 28.0, 30.0],
        );
    }

    #[test]
    fn hour() {
        assert_result_eq("hour(time()*1e4)", &[17.0, 21.0, 0.0, 4.0, 8.0, 11.0]);
    }

    #[test]
    fn month() {
        assert_result_eq("month(time()*1e4)", &[4.0, 5.0, 6.0, 7.0, 7.0, 8.0]);
    }

    #[test]
    fn year() {
        assert_result_eq(
            "year(time()*1e5)",
            &[1973.0, 1973.0, 1974.0, 1975.0, 1975.0, 1976.0],
        );
    }

    #[test]
    fn abs() {
        assert_result_eq(
            "abs(1500-time())",
            &[500.0, 300.0, 100.0, 100.0, 300.0, 500.0],
        );
        assert_result_eq(
            "abs(-time()+1300)",
            &[300.0, 100.0, 100.0, 300.0, 500.0, 700.0],
        );
    }

    #[test]
    fn ceil() {
        assert_result_eq("ceil(time()/500)", &[2.0, 3.0, 3.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn absent() {
        let q = "absent(time())";
        test_query(q, vec![]);

        let q = "absent(123)";
        test_query(q, vec![]);

        let q = "absent(vector(scalar(123)))";
        test_query(q, vec![]);
    }

    #[test]
    fn absent_with_nan() {
        assert_result_eq("absent(NaN)", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn present_over_time_time() {
        // assert_result_eq("present_over_time(time())", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_result_eq(
            "present_over_time(time()[100:300])",
            &[NAN, 1.0, NAN, NAN, 1.0, NAN],
        );
        assert_result_eq(
            "present_over_time(time()<1600)",
            &[1.0, 1.0, 1.0, NAN, NAN, NAN],
        );
    }

    #[test]
    fn absent_over_time() {
        assert_result_eq(
            "absent_over_time(NAN[200s:10s])",
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        );

        let q = r#"absent(label_set(scalar(1 or label_set(2, "xx", "foo")), "yy", "foo"))"#;
        assert_result_eq(q, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn absent_over_time_non_nan() {
        let q = "absent_over_time(time())";
        test_query(q, vec![])
    }

    #[test]
    fn absent_over_time_nan() {
        assert_result_eq(
            "absent_over_time((time() < 1500)[300s:])",
            &[NAN, NAN, NAN, NAN, 1.0, 1.0],
        );

        assert_result_eq("absent(time() > 1500)", &[1.0, 1.0, 1.0, NAN, NAN, NAN]);
    }

    #[test]
    fn absent_over_time_multi_ts() {
        let q = r#"
        absent_over_time((
        alias((time() < 1400)[200s:], "one"),
        alias((time() > 1600)[200s:], "two"),
        ))"#;
        assert_result_eq(q, &[NAN, NAN, 1.0, 1.0, NAN, NAN]);
    }

    #[test]
    fn clamp() {
        assert_result_eq(
            "clamp(time(), 1400.0, 1800)",
            &[1400.0, 1400.0, 1400.0, 1600.0, 1800.0, 1800.0],
        );
    }

    #[test]
    fn clamp_max() {
        assert_result_eq(
            "clamp_max(time(), 1400)",
            &[1000.0, 1200.0, 1400.0, 1400.0, 1400.0, 1400.0],
        );

        let q = r#"clamp_max(alias(time(), "foobar"), 1400)"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1400.0, 1400.0, 1400.0]);
        r.metric.set_metric_group("foobar");
        test_query(q, vec![r]);

        let q = r#"CLAmp_MAx(alias(time(), "foobar"), 1400)"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1400.0, 1400.0, 1400.0]);
        r.metric.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn clamp_min() {
        assert_result_eq(
            "clamp_min(time(), -time()+2500)",
            &[1500.0, 1300.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "clamp_min(1500, time())",
            &[1500.0, 1500.0, 1500.0, 1600.0, 1800.0, 2000.0],
        );
    }

    #[test]
    fn test_exp() {
        let q = r#"exp(alias(time()/1e3, "foobar"))"#;
        let r = make_result(&[
            std::f64::consts::E,
            3.3201169227365472,
            4.0551999668446745,
            4.953032424395115,
            6.0496474644129465,
            7.38905609893065,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn test_exp_1() {
        let q = r#"exp(alias(time()/1e3, "foobar")) keep_metric_names"#;
        let mut r = make_result(&[
            std::f64::consts::E,
            3.3201169227365472,
            4.0551999668446745,
            4.953032424395115,
            6.0496474644129465,
            7.38905609893065,
        ]);
        r.metric.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn at() {
        assert_result_eq(
            "time() @ 1h",
            &[3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0],
        );
        assert_result_eq(
            "time() @ start()",
            &[1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        );
        assert_result_eq(
            "time() @ end()",
            &[2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
        );
        assert_result_eq(
            "time() @ end() offset 10m",
            &[1400.0, 1400.0, 1400.0, 1400.0, 1400.0, 1400.0],
        );
        assert_result_eq(
            "time() @ (end() - 10m)",
            &[1400.0, 1400.0, 1400.0, 1400.0, 1400.0, 1400.0],
        );
    }

    #[test]
    fn rand() {
        assert_result_eq("round(rand()/2)", &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_result_eq(
            "round(rand(0), 0.01)",
            &[0.73, 0.77, 0.03, 0.58, 0.26, 0.77],
        );
    }

    #[test]
    fn rand_normal() {
        assert_result_eq(
            "clamp_max(clamp_min(0, rand_normal()), 0)",
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_result_eq(
            "round(rand_normal(0), 0.01)",
            &[0.71, 0.86, -2.44, 0.16, -1.28, 1.29],
        );
    }

    #[test]
    fn rand_exponential() {
        let q = "clamp_max(clamp_min(0, rand_exponential()), 0)";
        assert_result_eq(q, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert_result_eq(
            "round(rand_exponential(0), 0.01)",
            &[1.23, 1.34, 0.11, 0.45, 1.15, 2.73],
        );
    }

    #[test]
    fn now() {
        assert_result_eq("round(now()/now())", &[1.0; 6]);
    }

    #[test]
    fn pi() {
        let q = "pi()";
        let expected = [std::f64::consts::PI; 6];
        let r = make_result(&expected);
        test_query(q, vec![r]);
    }

    #[test]
    fn sin() {
        let q = "sin(pi()*(2000-time())/1000)";
        let r = make_result(&[
            1.2246467991473515e-16,
            0.5877852522924732,
            0.9510565162951536,
            0.9510565162951535,
            0.5877852522924731,
            0.0,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn sinh() {
        let q = "sinh(pi()*(2000-time())/1000)";
        let r = make_result(&[
            11.548739357257748,
            6.132140673514712,
            3.217113080357038,
            1.6144880404748523,
            0.6704839982471175,
            0.0,
        ]);
        let result_expected: Vec<QueryResult> = vec![r];
        test_query(q, result_expected)
    }

    #[test]
    fn asin() {
        let q = "asin((2000-time())/1000)";
        let r = make_result(&[
            std::f64::consts::FRAC_PI_2,
            0.9272952180016123,
            0.6435011087932843,
            0.41151684606748806,
            0.20135792079033082,
            0.0,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn asinh_sinh() {
        let q = "asinh(sinh((2000-time())/1000))";
        assert_result_eq(
            q,
            &[1.0, 0.8000000000000002, 0.6, 0.4000000000000001, 0.2, 0.0],
        );
    }

    #[test]
    fn test_atan2() {
        let q = "time() atan2 time()/10";
        let r = make_result(&[
            0.07853981633974483,
            0.07853981633974483,
            0.07853981633974483,
            0.07853981633974483,
            0.07853981633974483,
            0.07853981633974483,
        ]);
        test_query(q, vec![r])
    }

    #[test]
    fn test_atan() {
        let q = "atan((2000-time())/1000)";
        let r = make_result(&[
            std::f64::consts::FRAC_PI_4,
            0.6747409422235526,
            0.5404195002705842,
            0.3805063771123649,
            0.19739555984988078,
            0.0,
        ]);
        let result_expected: Vec<QueryResult> = vec![r];
        test_query(q, result_expected)
    }

    #[test]
    fn atanh_tanh() {
        let q = "atanh(tanh((2000-time())/1000))";
        assert_result_eq(
            q,
            &[1.0, 0.8000000000000002, 0.6, 0.4000000000000001, 0.2, 0.0],
        );
    }

    #[test]
    fn cos() {
        let q = "cos(pi()*(2000-time())/1000)";
        let r = make_result(&[
            -1_f64,
            -0.8090169943749475,
            -0.30901699437494734,
            0.30901699437494745,
            0.8090169943749473,
            1.0,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn acos() {
        let q = "acos((2000-time())/1000)";
        let r = make_result(&[
            0_f64,
            0.6435011087932843,
            0.9272952180016123,
            1.1592794807274085,
            1.3694384060045657,
            std::f64::consts::FRAC_PI_2,
        ]);
        test_query(q, vec![r]);

        let q = "acosh(cosh((2000-time())/1000))";
        let r = make_result(&[
            1_f64,
            0.8000000000000002,
            0.5999999999999999,
            0.40000000000000036,
            0.20000000000000023,
            0.0,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn rad() {
        assert_result_eq(
            "rad(deg(time()/500))",
            &[2.0, 2.3999999999999995, 2.8, 3.2, 3.6, 4.0],
        );
    }

    #[test]
    fn floor() {
        assert_result_eq("floor(time()/500)", &[2.0, 2.0, 2.0, 3.0, 3.0, 4.0]);
    }

    #[test]
    fn sqrt() {
        assert_result_eq(
            "sqrt(time())",
            &[
                31.622776601683793,
                34.64101615137755,
                37.416573867739416,
                40.0,
                42.42640687119285,
                44.721359549995796,
            ],
        );

        let q = r#"round(sqrt(sum2(label_set(10, "foo", "bar") or label_set(time()/100, "baz", "sss"))))"#;
        assert_result_eq(q, &[14.0, 16.0, 17.0, 19.0, 21.0, 22.0]);
    }

    #[test]
    fn test_ln() {
        let q = "ln(time())";
        let r = make_result(&[
            6.907755278982137,
            7.090076835776092,
            7.24422751560335,
            7.3777589082278725,
            7.495541943884256,
            7.600902459542082,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn log2() {
        let q = "log2(time())";
        let r = make_result(&[
            9.965784284662087,
            10.228818690495881,
            10.451211111832329,
            10.643856189774725,
            10.813781191217037,
            10.965784284662087,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn log10() {
        let q = "log10(time())";
        let r = make_result(&[
            3_f64,
            3.0791812460476247,
            3.1461280356782377,
            3.2041199826559246,
            3.255272505103306,
            3.3010299956639813,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn pow() {
        let q = "time()*(-4)^0.5";
        test_query(q, vec![]);

        assert_result_eq(
            "time()*-4^0.5",
            &[-2000.0, -2400.0, -2800.0, -3200.0, -3600.0, -4000.0],
        );
    }

    #[test]
    fn default_for_nan_series() {
        let q = r#"label_set(0, "foo", "bar")/0 default 7"#;
        let mut r = make_result(&[7_f64, 7.0, 7.0, 7.0, 7.0, 7.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn alias() {
        let q = r#"alias(time(), "foobar")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_set_tag() {
        let q = r#"label_set(time(), "tagname", "tagvalue")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("tagname", "tagvalue");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_set_metric_name() {
        let q = r#"label_set(time(), "__name__", "foobar")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_set_metric_name_tag() {
        let q = r#"label_set(
        label_set(time(), "__name__", "foobar"),
        "tag_name", "tag_value"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("foobar");
        r.metric.set("tag_name", "tag_value");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_set_del_metric_name() {
        let q = r#"label_set(
        label_set(time(), "__name__", "foobar"),
        "__name__", ""
        )"#;
        let r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn r#label_set_del_tag() {
        let q = r#"label_set(
        label_set(time(), "tagname", "foobar"),
        "tagname", ""
        )"#;
        assert_result_eq(q, &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn r#label_set_multi() {
        let q = r#"label_set(time()+100, "t1", "v1", "t2", "v2", "__name__", "v3")"#;
        let mut r = make_result(&[1100_f64, 1300.0, 1500.0, 1700.0, 1900.0, 2100.0]);
        r.metric.set_metric_group("v3");
        r.metric.set("t1", "v1");
        r.metric.set("t2", "v2");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_map_match() {
        let q = r#"sort(label_map((
        label_set(time(), "label", "v1"),
        label_set(time()+100, "label", "v2"),
        label_set(time()+200, "label", "v3"),
        label_set(time()+300, "x", "y"),
        label_set(time()+400, "label", "v4"),
        ), "label", "v1", "foo", "v2", "bar", "", "qwe", "v4", ""))"#;
        let mut r1 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r1.metric.set("label", "foo");
        let mut r2 = make_result(&[1100_f64, 1300.0, 1500.0, 1700.0, 1900.0, 2100.0]);
        r2.metric.set("label", "bar");
        let mut r3 = make_result(&[1200_f64, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0]);
        r3.metric.set("label", "v3");
        let mut r4 = make_result(&[1300_f64, 1500.0, 1700.0, 1900.0, 2100.0, 2300.0]);
        r4.metric.set("label", "qwe");
        r4.metric.set("x", "y");

        let r5 = make_result(&[1400_f64, 1600.0, 1800.0, 2000.0, 2200.0, 2400.0]);
        let result_expected = vec![r1, r2, r3, r4, r5];
        test_query(q, result_expected)
    }

    #[test]
    fn label_uppercase() {
        let q = r#"label_uppercase(
        label_set(time(), "foo", "bAr", "XXx", "yyy", "zzz", "abc"),
        "foo", "XXx", "aaa"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("XXx", "YYY");
        r.metric.set("foo", "BAR");
        r.metric.set("zzz", "abc");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_lowercase() {
        let q = r#"label_lowercase(
        label_set(time(), "foo", "bAr", "XXx", "yyy", "zzz", "aBc"),
        "foo", "XXx", "aaa"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("XXx", "yyy");
        r.metric.set("foo", "bar");
        r.metric.set("zzz", "aBc");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy_new_tag() {
        let q = r#"label_copy(
        label_set(time(), "tagname", "foobar"),
        "tagname", "xxx"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("tagname", "foobar");
        r.metric.set("xxx", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_move_new_tag() {
        let q = r#"label_move(
        label_set(time(), "tagname", "foobar"),
        "tagname", "xxx"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("xxx", "foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy_same_tag() {
        let q = r#"label_copy(
        label_set(time(), "tagname", "foobar"),
        "tagname", "tagname"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("tagname", "foobar");
        test_query(q, vec![r])
    }

    #[test]
    fn label_move_same_tag() {
        let q = r#"label_move(
        label_set(time(), "tagname", "foobar"),
        "tagname", "tagname"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("tagname", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy_same_tag_non_existing_src() {
        let q = r#"label_copy(
        label_set(time(), "tagname", "foobar"),
        "non-existing-tag", "tagname"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("tagname", "foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_move_same_tag_non_existing_src() {
        let q = r#"label_move(
        label_set(time(), "tagname", "foobar"),
        "non-existing-tag", "tagname"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("tagname", "foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy_existing_tag() {
        let q = r#"label_copy(
        label_set(time(), "tagname", "foobar", "xx", "yy"),
        "xx", "tagname"
        )"#;

        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("tagname", "yy");
        r.metric.set("xx", "yy");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_move_existing_tag() {
        let q = r#"label_move(
        label_set(time(), "tagname", "foobar", "xx", "yy"),
        "xx", "tagname"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("tagname", "yy");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy_from_metric_group() {
        let q = r#"label_copy(
        label_set(time(), "tagname", "foobar", "__name__", "yy"),
        "__name__", "aa"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("yy");
        r.metric.set("aa", "yy");
        r.metric.set("tagname", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_move_from_metric_group() {
        let q = r#"label_move(
        label_set(time(), "tagname", "foobar", "__name__", "yy"),
        "__name__", "aa"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("aa", "yy");
        r.metric.set("tagname", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy_to_metric_group() {
        let q = r#"label_copy(
        label_set(time(), "tagname", "foobar"),
        "tagname", "__name__"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("foobar");
        r.metric.set("tagname", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_move_to_metric_group() {
        let q = r#"label_move(
        label_set(time(), "tagname", "foobar"),
        "tagname", "__name__"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn labels_equal() {
        let q = r#"sort(labels_equal((
      label_set(10, "instance", "qwe", "host", "rty"),
      label_set(20, "instance", "qwe", "host", "qwe"),
      label_set(30, "aaa", "bbb", "instance", "foo", "host", "foo"),
    ), "instance", "host"))"#;
        let mut r1 = make_result(&[20.0, 20.0, 20.0, 20.0, 20.0, 20.0]);
        r1.metric.set("host", "qwe");
        r1.metric.set("instance", "qwe");

        let mut r2 = make_result(&[30.0, 30.0, 30.0, 30.0, 30.0, 30.0]);
        r2.metric.set("aaa", "bbb");
        r2.metric.set("host", "foo");
        r2.metric.set("instance", "foo");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn drop_empty_series() {
        let q = r#"sort(drop_empty_series(
                            (
                            alias(time(), "foo"),
                            alias(500 + time(), "bar"),
                            ) > 2000
                            ) default 123)"#;

        let mut r = make_result(&[123.0, 123.0, 123.0, 2100.0, 2300.0, 2500.0]);
        r.metric.set_metric_group("bar");
        test_query(q, vec![r])
    }

    #[test]
    fn no_drop_empty_series() {
        let q = r#"sort((
                        (
                        alias(time(), "foo"),
                        alias(500 + time(), "bar"),
                        ) > 2000
                    ) default 123)"#;

        let mut r1 = make_result(&[123.0, 123.0, 123.0, 123.0, 123.0, 123.0]);
        r1.metric.set_metric_group("foo");
        let mut r2 = make_result(&[123.0, 123.0, 123.0, 2100.0, 2300.0, 2500.0]);
        r2.metric.set_metric_group("bar");
        test_query(q, vec![r1, r2])
    }

    #[test]
    fn drop_common_labels_single_series() {
        let q =
            r#"drop_common_labels(label_set(time(), "foo", "bar", "__name__", "xxx", "q", "we"))"#;
        assert_result_eq(q, &[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn drop_common_labels_multi_series() {
        let q = r#"sort_desc(drop_common_labels((
        label_set(time(), "foo", "bar", "__name__", "xxx", "q", "we"),
        label_set(time()/10, "foo", "bar", "__name__", "yyy"),
        )))"#;
        let mut r1 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r1.metric.set_metric_group("xxx");
        r1.metric.set("q", "we");
        let mut r2 = make_result(&[100_f64, 120.0, 140.0, 160.0, 180.0, 200.0]);
        r2.metric.set_metric_group("yyy");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn drop_common_labels_multi_args() {
        let q = r#"sort(drop_common_labels(
        label_set(time(), "foo", "bar", "__name__", "xxx", "q", "we"),
        label_set(time()/10, "foo", "bar", "__name__", "xxx"),
        ))"#;
        let r1 = make_result(&[100_f64, 120.0, 140.0, 160.0, 180.0, 200.0]);
        let mut r2 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r2.metric.set("q", "we");
        test_query(q, vec![r1, r2])
    }

    #[test]
    fn label_keep_no_labels() {
        let q = r#"label_keep(time(), "foo", "bar")"#;
        assert_result_eq(q, &[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn label_keep_certain_labels() {
        let q = r#"label_keep(label_set(time(), "foo", "bar", "__name__", "xxx", "q", "we"), "foo", "nonexisting-label")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_keep_metric_name() {
        let q = r#"label_keep(label_set(time(), "foo", "bar", "__name__", "xxx", "q", "we"), "nonexisting-label", "__name__")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("xxx");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_del_no_labels() {
        assert_result_eq(
            r#"label_del(time(), "foo", "bar")"#,
            &[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
    }

    #[test]
    fn label_del_certain_labels() {
        let q = r#"label_del(label_set(time(), "foo", "bar", "__name__", "xxx", "q", "we"), "foo", "nonexisting-label")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("xxx");
        r.metric.set("q", "we");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_del_metric_name() {
        let q = r#"label_del(label_set(time(), "foo", "bar", "__name__", "xxx", "q", "we"), "nonexisting-label", "__name__")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("foo", "bar");
        r.metric.set("q", "we");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_join_empty() {
        let q = r#"label_join(vector(time()), "tt", "(sep)", "BAR")"#;
        assert_result_eq(q, &[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn label_join_tt() {
        let q = r#"label_join(vector(time()), "tt", "(sep)", "foo", "BAR")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("tt", "(sep)");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_join_name() {
        let q = r#"label_join(time(), "__name__", "(sep)", "foo", "BAR", "")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("(sep)(sep)");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_join_label_join() {
        let q = r#"label_join(label_join(time(), "__name__", "(sep)", "foo", "BAR"), "xxx", ",", "foobar", "__name__")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("(sep)");
        r.metric.set("xxx", ",(sep)");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_join_dst_label_equals_src_label() {
        let q =
            r#"label_join(label_join(time(), "bar", "sep1", "a", "b"), "bar", "sep2", "a", "bar")"#;
        let mut r = make_result(&[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("bar", "sep2sep1");
        test_query(q, vec![r])
    }

    #[test]
    fn label_value() {
        let q = r#"with (
        x = (
        label_set(time() > 1500, "foo", "123.456", "__name__", "aaa"),
        label_set(-time(), "foo", "bar", "__name__", "bbb"),
        label_set(-time(), "__name__", "bxs"),
        label_set(-time(), "foo", "45", "bar", "xs"),
        )
        )
        sort(x + label_value(x, "foo"))"#;
        let mut r1 = make_result(&[-955_f64, -1155.0, -1355.0, -1555.0, -1755.0, -1955.0]);
        r1.metric.set("bar", "xs");
        r1.metric.set("foo", "45");

        let mut r2 = make_result(&[NAN, NAN, NAN, 1723.456, 1923.456, 2123.456]);
        r2.metric.set("foo", "123.456");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn label_transform_mismatch() {
        let q = r#"label_transform(time(), "__name__", "foobar", "xx")"#;
        assert_result_eq(q, &[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn label_transform_match() {
        let q = r#"label_transform(
        label_set(time(), "foo", "a.bar.baz"),
        "foo", "\\.", "-")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("foo", "a-bar-baz");
        test_query(q, vec![r])
    }

    #[test]
    fn label_replace_with_non_existing_src() {
        let q = r#"label_replace(time(), "__name__", "x${1}y", "foo", ".+")"#;
        let r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn label_replace_with_non_existing_src_match() {
        let q = r#"label_replace(time(), "foo", "x", "bar", "")"#;
        let mut r = make_result(&[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("foo", "x");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_replace_with_non_existing_src_mismatch() {
        let q = r#"label_replace(time(), "foo", "x", "bar", "y")"#;
        let r = make_result(&[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn label_replace_with_mismatch() {
        let q = r#"label_replace(label_set(time(), "foo", "foobar"), "__name__", "x${1}y", "foo", "bar(.+)")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("foo", "foobar");
        test_query(q, vec![r])
    }

    #[test]
    fn label_replace_match() {
        let q = r#"label_replace(time(), "__name__", "x${1}y", "foo", ".*")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("xy");
        test_query(q, vec![r])
    }

    #[test]
    fn label_replace_label_replace() {
        let q = r#"
        label_replace(
        label_replace(
        label_replace(time(), "__name__", "x${1}y", "foo", ".*"),
        "xxx", "foo${1}bar(${1})", "__name__", "(.+)"),
        "xxx", "AA$1", "xxx", "foox(.+)"
        )"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("xy");
        r.metric.set("xxx", "AAybar(xy)");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_match() {
        let q = r#"
        label_match((
        alias(time(), "foo"),
        alias(2*time(), "bar"),
        ), "__name__", "f.+")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("foo");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_mismatch() {
        let q = r#"
        label_mismatch((
        alias(time(), "foo"),
        alias(2*time(), "bar"),
        ), "__name__", "f.+")"#;
        let mut r = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r.metric.set_metric_group("bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_graphite_group() {
        let q = r#"sort(label_graphite_group((
        alias(1, "foo.bar.baz"),
        alias(2, "abc"),
        label_set(alias(3, "a.xx.zz.asd"), "qwe", "rty"),
        ), 1, 3))"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set_metric_group("bar.");
        let mut r2 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r2.metric.set_metric_group(".");
        let mut r3 = make_result(&[3_f64, 3.0, 3.0, 3.0, 3.0, 3.0]);
        r3.metric.set_metric_group("xx.asd");
        r3.metric.set("qwe", "rty");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3];
        test_query(q, result_expected)
    }

    #[test]
    fn limit_offset() {
        let q = r#"limit_offset(1, 1, sort_by_label((
        label_set(time()*1, "foo", "y"),
        label_set(time()*2, "foo", "a"),
        label_set(time()*3, "foo", "x"),
        ), "foo"))"#;
        let mut r = make_result(&[3000_f64, 3600.0, 4200.0, 4800.0, 5400.0, 6000.0]);
        r.metric.set("foo", "x");
        test_query(q, vec![r]);
    }

    #[test]
    fn limit_offset_nan() {
        // q returns 3 time series, where foo=3 contains only NaN values
        // limit_offset suppose to apply offset for non-NaN series only
        let q = r#"limit_offset(1, 1, sort_by_label_desc((
        label_set(time()*1, "foo", "1"),
        label_set(time()*2, "foo", "2"),
        label_set(time()*3, "foo", "3"),
        ) < 3000, "foo"))"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("foo", "1");
        test_query(q, vec![r]);
    }
    
    #[test]
    fn limit_offset_sort_by_label() {
        let q = r#"limit_offset(5, 0, sort_by_label_numeric_desc((
                    label_set(3, "foo", "1:0:3"),
                    label_set(4, "foo", "5:0:15"),
                    label_set(1, "foo", "1:0:2"),
                    label_set(5, "foo", "7:0:15"),
                    label_set(7, "foo", "3:0:1"),
                    label_set(6, "foo", "1:0:2"),
                    label_set(8, "foo", "9:0:15")
                ), "foo"))"#;
        
        let mut r1 = make_result(&[8.0, 8.0, 8.0, 8.0, 8.0, 8.0]);
        r1.metric.set("foo", "9:0:15");
        
        let r2 = make_result(&[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
        r1.metric.set("foo", "7:0:15");
        
        let mut r3 = make_result(&[4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
        r3.metric.set("foo", "5:0:15");
        
        let mut r4 = make_result(&[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
        r4.metric.set("foo", "3:0:1");
        
        let mut r5 = make_result(&[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
        r5.metric.set("foo", "1:0:3");
        
        test_query(q, vec![r1, r2, r3, r4, r5]);
    }

    #[test]
    fn sum_label_graphite_group() {
        let q = r#"sort(sum by (__name__) (
        label_graphite_group((
        alias(1, "foo.bar.baz"),
        alias(2, "x.y.z"),
        alias(3, "qe.bar.qqq"),
        ), 1)
        ))"#;
        let mut r1 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r1.metric.set_metric_group("y");
        let mut r2 = make_result(&[4_f64, 4.0, 4.0, 4.0, 4.0, 4.0]);
        r2.metric.set_metric_group("bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn two_timeseries() {
        let q = r#"sort_desc(time() or label_set(2, "xx", "foo"))"#;
        let r1 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        let mut r2 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r2.metric.set("xx", "foo");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn test_sgn() {
        assert_result_eq("sgn(time()-1400)", &[-1.0, -1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn round_to_integer() {
        assert_result_eq("round(time()/1e3)", &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn round_to_nearest() {
        assert_result_eq("round(time()/1e3, 0.5)", &[1.0, 1.0, 1.5, 1.5, 2.0, 2.0]);
        assert_result_eq(
            "round(-time()/1e3, 0.5)",
            &[-1.0, -1.0, -1.5, -1.5, -2.0, -2.0],
        );
    }

    #[test]
    fn scalar_multi_timeseries() {
        let q = r#"scalar(1 or label_set(2, "xx", "foo"))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn sort() {
        let q = r#"sort(2 or label_set(1, "xx", "foo"))"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set("xx", "foo");
        let r2 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_desc() {
        let q = r#"sort_desc(1 or label_set(2, "xx", "foo"))"#;
        let mut r1 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r1.metric.set("xx", "foo");
        let r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label() {
        let q = r#"sort_by_label((
        alias(1, "foo"),
        alias(2, "bar"),
        ), "__name__")"#;
        let mut r1 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r1.metric.set_metric_group("bar");
        let mut r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r2.metric.set_metric_group("foo");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label_desc() {
        let q = r#"sort_by_label_desc((
        alias(1, "foo"),
        alias(2, "bar"),
        ), "__name__")"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set_metric_group("foo");
        let mut r2 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r2.metric.set_metric_group("bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label_multiple_labels() {
        let q = r#"sort_by_label((
        label_set(1, "x", "b", "y", "aa"),
        label_set(2, "x", "a", "y", "aa"),
        ), "y", "x")"#;
        let mut r1 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r1.metric.set("x", "a");
        r1.metric.set("y", "aa");

        let mut r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r2.metric.set("x", "b");
        r2.metric.set("y", "aa");
        test_query(q, vec![r1, r2])
    }

    #[test]
    fn test_scalar() {
        assert_result_eq("-1 < 2", &[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        assert_result_eq(
            "123 < time()",
            &[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq("time() > 1234", &[NAN, NAN, 1400.0, 1600.0, 1800.0, 2000.0]);
        assert_result_eq("time() >bool 1234", &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        assert_result_eq(
            "(time() > 1234) >bool 1450",
            &[NAN, NAN, 0.0, 1.0, 1.0, 1.0],
        );
        assert_result_eq(
            "(time() > 1234) !=bool 1400",
            &[NAN, NAN, 0.0, 1.0, 1.0, 1.0],
        );
        assert_result_eq(
            "1400 !=bool (time() > 1234)",
            &[NAN, NAN, 0.0, 1.0, 1.0, 1.0],
        );
        let q = "123 > time()";
        test_query(q, vec![]);

        let q = "time() < 123";
        test_query(q, vec![]);

        assert_result_eq(
            "1300 < time() < 1700",
            &[NAN, NAN, 1400.0, 1600.0, NAN, NAN],
        );
    }

    #[test]
    fn array_cmp_scalar_leave_metric_group() {
        let q = r#"sort_desc((
        label_set(time(), "__name__", "foo", "a", "x"),
        label_set(time()+200, "__name__", "bar", "a", "x"),
        ) > 1300)"#;
        let mut r1 = make_result(&[NAN, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0]);
        r1.metric.set_metric_group("bar");
        r1.metric.set("a", "x");
        let mut r2 = make_result(&[NAN, NAN, 1400.0, 1600.0, 1800.0, 2000.0]);
        r2.metric.set_metric_group("foo");
        r2.metric.set("a", "x");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn a_cmp_bool_scalar_drop_metric_group() {
        let q = r#"sort_desc((
        label_set(time(), "__name__", "foo", "a", "x"),
        label_set(time()+200, "__name__", "bar", "a", "y"),
        ) >= bool 1200)"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set("a", "y");
        let mut r2 = make_result(&[0_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r2.metric.set("a", "x");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn numeric_greater() {
        let q = "1 > 2";
        test_query(q, vec![])
    }

    #[test]
    fn vector_eq_bool() {
        // vector(1) == bool time()
        assert_result_eq("vector(1) == bool time()", &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn vector_eq_scalar() {
        test_query("vector(1) == time()", vec![]);
    }

    #[test]
    fn compare_to_nan_right() {
        assert_result_eq("1 != bool NAN", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn compare_to_nan_left() {
        assert_result_eq("NAN != bool 1", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn function_cmp_scalar() {
        assert_result_eq("time() >= bool 2", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
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
    fn test_time_unless_time_greater_than_1500() {
        assert_result_eq(
            "time() unless time() > 1500",
            &[1000_f64, 1200.0, 1400.0, NAN, NAN, NAN],
        );
    }

    // todo: do the scalar vector versions of the following 2 tests
    #[test]
    fn test_time_unless_2() {
        test_query("time() unless 2", vec![]);
    }

    #[test]
    fn test_timeseries_with_tags_unless_2() {
        let q = r#"label_set(time(), "foo", "bar") unless 2"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn scalar_or_scalar() {
        assert_result_eq(
            "time() > 1400 or 123",
            &[123.0, 123.0, 123.0, 1600.0, 1800.0, 2000.0],
        );
    }

    #[test]
    fn scalar_default_scalar() {
        assert_result_eq(
            "time() > 1400 default 123",
            &[123.0, 123.0, 123.0, 1600.0, 1800.0, 2000.0],
        );
    }

    #[test]
    fn scalar_default_scalar_from_vector() {
        let q = r#"time() > 1400 default scalar(label_set(123, "foo", "bar"))"#;
        assert_result_eq(q, &[123.0, 123.0, 123.0, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn scalar_default_vector1() {
        let q = r#"time() > 1400 default label_set(123, "foo", "bar")"#;
        assert_result_eq(q, &[NAN, NAN, NAN, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn scalar_default_vector2() {
        let q = r#"time() > 1400 default (
        label_set(123, "foo", "bar"),
        label_set(456, "__name__", "xxx"),
        )"#;
        assert_result_eq(q, &[456.0, 456.0, 456.0, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn scalar_default_nan() {
        let q = "time() > 1400 default (time() < -100)";
        assert_result_eq(q, &[NAN, NAN, NAN, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn vector_default_scalar() {
        let q = r#"sort_desc(union(
        label_set(time() > 1400.0, "__name__", "x", "foo", "bar"),
        label_set(time() < 1700, "__name__", "y", "foo", "baz")) default 123)"#;
        let mut r1 = make_result(&[123_f64, 123.0, 123.0, 1600.0, 1800.0, 2000.0]);
        r1.metric.set_metric_group("x");
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 123.0, 123.0]);
        r2.metric.set_metric_group("y");
        r2.metric.set("foo", "baz");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_divided_by_scalar() {
        let q =
            r#"sort_desc((label_set(time(), "foo", "bar") or label_set(10, "foo", "qwert")) / 2)"#;
        let mut r1 = make_result(&[500_f64, 600.0, 700.0, 800.0, 900.0, 1000.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[5_f64, 5.0, 5.0, 5.0, 5.0, 5.0]);
        r2.metric.set("foo", "qwert");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_multiplied_by_scalar() {
        assert_result_eq(
            "sum(time()) * 2",
            &[2000.0, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0],
        );
    }

    #[test]
    fn vector_by_scalar_keep_metric_names() {
        let q = r#"sort_desc((label_set(time(), "foo", "bar", "__name__", "q1") or label_set(10, "foo", "qwert", "__name__", "q2")) / 2 keep_metric_names)"#;
        let mut r1 = make_result(&[500_f64, 600_f64, 700_f64, 800_f64, 900_f64, 1000_f64]);
        r1.metric.measurement = "q1".to_string();
        r1.metric.labels = vec![Label {
            name: "foo".to_string(),
            value: "bar".to_string(),
        }];

        let mut r2 = make_result(&[5_f64, 5_f64, 5_f64, 5_f64, 5_f64, 5_f64]);
        r2.metric.measurement = "q2".to_string();
        r2.metric.labels = vec![Label {
            name: "foo".to_string(),
            value: "qwert".to_string(),
        }];
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn scalar_multiplied_by_vector() {
        let q =
            r#"sort_desc(2 * (label_set(time(), "foo", "bar") or label_set(10, "foo", "qwert")))"#;
        let mut r1 = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[20_f64, 20.0, 20.0, 20.0, 20.0, 20.0]);
        r2.metric.set("foo", "qwert");
        let result_expected: Vec<QueryResult> = vec![r1, r2];
        test_query(q, result_expected)
    }

    #[test]
    fn scalar_multiplied_by_vector_keep_metric_names() {
        let q = r#"sort_desc(2 * (label_set(time(), "foo", "bar", "__name__", "q1"), label_set(10, "foo", "qwert", "__name__", "q2")) keep_metric_names)"#;
        let mut r1 = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r1.metric.measurement = "q1".to_string();
        r1.metric.set("foo", "bar");

        let mut r2 = make_result(&[20_f64, 20.0, 20.0, 20.0, 20.0, 20.0]);
        r2.metric.measurement = "q2".to_string();
        r2.metric.set("foo", "qwert");

        test_query(q, vec![r1, r2])
    }

    #[test]
    fn scalar_on_group_right_vector() {
        // scalar * on() group_right vector
        let q = r#"sort_desc(2 * on() group_right() (label_set(time(), "foo", "bar") or label_set(10, "foo", "qwert")))"#;
        let mut r1 = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[20_f64, 20.0, 20.0, 20.0, 20.0, 20.0]);
        r2.metric.set("foo", "qwert");

        test_query(q, vec![r1, r2])
    }

    #[test]
    fn scalar_on_group_right_vector_keep_metric_names() {
        // scalar * on() group_right vector keep_metric_names
        let q = r#"sort_desc(2 * on() group_right() (label_set(time(), "foo", "bar", "__name__", "q1"), label_set(10, "foo", "qwert", "__name__", "q2")) keep_metric_names)"#;
        let mut r1 = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r1.metric.measurement = "q1".to_string();

        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[20_f64, 20.0, 20.0, 20.0, 20.0, 20.0]);
        r2.metric.measurement = "q2".to_string();

        r2.metric.set("foo", "qwert");
        test_query(q, vec![r1, r2])
    }

    #[test]
    fn scalar_multiply_by_ignoring_foo_group_right_vector() {
        let q = r#"sort_desc(label_set(2, "a", "2") * ignoring(foo,a) group_right(a) (label_set(time(), "foo", "bar", "a", "1"), label_set(10, "foo", "qwert")))"#;
        let mut r1 = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r1.metric.set("a", "2");
        r1.metric.set("foo", "bar");

        let mut r2 = make_result(&[20_f64, 20.0, 20.0, 20.0, 20.0, 20.0]);
        r2.metric.set("a", "2");
        r2.metric.set("foo", "qwert");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn scalar_multiply_ignoring_vector() {
        let q = r#"sort_desc(label_set(2, "foo", "bar") * ignoring(a) (label_set(time(), "foo", "bar") or label_set(10, "foo", "qwert")))"#;
        let mut r = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn scalar_multiply_by_on_foo_vector() {
        //"scalar * on(foo) vector"
        let q = r#"sort_desc(label_set(2, "foo", "bar", "aa", "bb") * on(foo) (label_set(time(), "foo", "bar", "xx", "yy") or label_set(10, "foo", "qwert")))"#;
        let mut r = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn vector_multiply_by_on_foo_scalar() {
        let q = r#"sort_desc((label_set(time(), "foo", "bar", "xx", "yy"), label_set(10, "foo", "qwert")) * on(foo) label_set(2, "foo","bar","aa","bb"))"#;
        let mut r = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn vector_multiply_by_on_foo_scalar_keep_metric_names() {
        let q = r#"
                (
                    (
		                label_set(time(), "foo", "bar", "xx", "yy", "__name__", "q1"),
			            label_set(10, "foo", "qwert", "__name__", "q2")
		            ) * on(foo) label_set(2, "foo","bar","aa","bb", "__name__", "q2")
		        ) keep_metric_names
        "#;
        let mut r = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r.metric.measurement = "q1".to_string();
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn vector_multiply_by_on_foo_group_left() {
        let q = r#"sort(label_set(time()/10, "foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) group_left(op) (
        label_set(time() < 1400.0, "foo", "bar", "op", "le"),
        label_set(time() >= 1400.0, "foo", "bar", "op", "ge"),
        ))"#;
        let mut r1 = make_result(&[1100_f64, 1320.0, NAN, NAN, NAN, NAN]);
        r1.metric.set("foo", "bar");
        r1.metric.set("op", "le");
        r1.metric.set("xx", "yy");

        let mut r2 = make_result(&[NAN, NAN, 1540.0, 1760.0, 1980.0, 2200.0]);
        r2.metric.set("foo", "bar");
        r2.metric.set("op", "ge");
        r2.metric.set("xx", "yy");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_multiplied_by_on_foo_duplicate_nonoverlapping_timeseries() {
        let q = r#"label_set(time()/10, "foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) (
        label_set(time() < 1400.0, "foo", "bar", "op", "le"),
        label_set(time() >= 1400.0, "foo", "bar", "op", "ge"),
        )"#;
        let mut r1 = make_result(&[1100_f64, 1320.0, 1540.0, 1760.0, 1980.0, 2200.0]);
        r1.metric.set("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn vector_multiply_by_on_foo_group_left_duplicate_nonoverlapping_timeseries() {
        let q = r#"label_set(time()/10, "foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) group_left() (
        label_set(time() < 1400.0, "foo", "bar", "op", "le"),
        label_set(time() >= 1400.0, "foo", "bar", "op", "ge"),
        )"#;
        let mut r1 = make_result(&[1100_f64, 1320.0, 1540.0, 1760.0, 1980.0, 2200.0]);
        r1.metric.set("foo", "bar");
        r1.metric.set("xx", "yy");

        test_query(q, vec![r1]);
    }

    #[test]
    fn vector_multiplied_by_on_foo_group_left_name() {
        let q = r#"label_set(time()/10, "foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) group_left(__name__)
        label_set(time(), "foo", "bar", "__name__", "aaa")"#;
        let mut r1 = make_result(&[1100_f64, 1320.0, 1540.0, 1760.0, 1980.0, 2200.0]);
        r1.metric.set_metric_group("aaa");
        r1.metric.set("foo", "bar");
        r1.metric.set("xx", "yy");

        test_query(q, vec![r1]);
    }

    #[test]
    fn vector_multiplied_by_on_foo_group_right() {
        let q = r#"sort(label_set(time()/10, "foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) group_right(xx) (
        label_set(time(), "foo", "bar", "__name__", "aaa"),
        label_set(time()+3, "foo", "bar", "__name__", "yyy","ppp", "123"),
        ))"#;
        let mut r1 = make_result(&[1100_f64, 1320.0, 1540.0, 1760.0, 1980.0, 2200.0]);
        r1.metric.set("foo", "bar");
        r1.metric.set("xx", "yy");

        let mut r2 = make_result(&[1103_f64, 1323.0, 1543.0, 1763.0, 1983.0, 2203.0]);
        r2.metric.set("foo", "bar");
        r2.metric.set("ppp", "123");
        r2.metric.set("xx", "yy");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_multiply_by_on_group_left_scalar() {
        let q = r#"sort_desc((label_set(time(), "foo", "bar") or label_set(10, "foo", "qwerty")) * on() group_left 2)"#;
        let mut r1 = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[20_f64, 20.0, 20.0, 20.0, 20.0, 20.0]);
        r2.metric.set("foo", "qwerty");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_plus_vector_matching() {
        let q = r#"sort_desc(
        (label_set(time(), "t1", "v1") or label_set(10, "t2", "v2"))
        +
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v2"))
        )"#;
        let mut r1 = make_result(&[1100_f64, 1300.0, 1500.0, 1700.0, 1900.0, 2100.0]);
        r1.metric.set("t1", "v1");
        let mut r2 = make_result(&[1010_f64, 1210.0, 1410.0, 1610.0, 1810.0, 2010.0]);
        r2.metric.set("t2", "v2");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_vector_partial_matching() {
        let q = r#"sort_desc(
        (label_set(time(), "t1", "v1") or label_set(10, "t2", "v2"))
        +
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3"))
        )"#;
        let mut r = make_result(&[1100_f64, 1300.0, 1500.0, 1700.0, 1900.0, 2100.0]);
        r.metric.set("t1", "v1");
        test_query(q, vec![r])
    }

    #[test]
    fn vector_plus_vector_partial_matching_keep_metric_names() {
        let q = r#"(
		  (label_set(time(), "t1", "v1", "__name__", "q1") or label_set(10, "t2", "v2", "__name__", "q2"))
		    +
		  (label_set(100, "t1", "v1", "__name__", "q3") or label_set(time(), "t2", "v3"))
		) keep_metric_names
        "#;
        let mut r = make_result(&[1100_f64, 1300.0, 1500.0, 1700.0, 1900.0, 2100.0]);
        r.metric.measurement = "q1".to_string();
        r.metric.set("t1", "v1");
        test_query(q, vec![r])
    }

    #[test]
    fn vector_plus_vector_no_matching() {
        let q = r#"sort_desc(
        (label_set(time(), "t2", "v1") or label_set(10, "t2", "v2"))
        +
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3"))
        )"#;
        test_query(q, vec![]);
    }

    #[test]
    fn vector_plus_vector_on_matching() {
        let q = r#"sort_desc(
        (label_set(time(), "t1", "v123", "t2", "v3") or label_set(10, "t2", "v2"))
        + on (foo, t2)
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3"))
        )"#;
        let mut r = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r.metric.set("t2", "v3");

        test_query(q, vec![r])
    }

    #[test]
    fn vector_plus_vector_on_group_left_matching() {
        let q = r#"sort_desc(
        (label_set(time(), "t1", "v123", "t2", "v3"), label_set(10, "t2", "v3", "xxx", "yy"))
        + on (foo, t2) group_left (t1, noxxx)
        (label_set(100, "t1", "v1"), label_set(time(), "t2", "v3", "noxxx", "aa"))
        )"#;
        let mut r1 = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r1.metric.set("noxxx", "aa");
        r1.metric.set("t2", "v3");

        let mut r2 = make_result(&[1010_f64, 1210.0, 1410.0, 1610.0, 1810.0, 2010.0]);
        r2.metric.set("noxxx", "aa");
        r2.metric.set("t2", "v3");
        r2.metric.set("xxx", "yy");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_plus_vector_on_group_left_name() {
        let q = r#"sort_desc(
        (union(label_set(time(), "t2", "v3", "__name__", "vv3", "x", "y"), label_set(10, "t2", "v3", "__name__", "yy")))
        + on (t2, dfdf) group_left (__name__, xxx)
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3", "__name__", "abc"))
        )"#;
        let mut r1 = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r1.metric.set_metric_group("abc");
        r1.metric.set("t2", "v3");
        r1.metric.set("x", "y");

        let mut r2 = make_result(&[1010_f64, 1210.0, 1410.0, 1610.0, 1810.0, 2010.0]);
        r2.metric.set_metric_group("abc");
        r2.metric.set("t2", "v3");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_plus_vector_ignoring_matching() {
        let q = r#"sort_desc(
        (label_set(time(), "t1", "v123", "t2", "v3") or label_set(10, "t2", "v2"))
        + ignoring (foo, t1, bar)
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3"))
        )"#;
        let mut r = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r.metric.set("t2", "v3");

        test_query(q, vec![r]);
    }

    #[test]
    fn vector_plus_vector_ignoring_group_right_matching() {
        let q = r#"sort_desc(
        (label_set(time(), "t1", "v123", "t2", "v3") or label_set(10, "t2", "v321", "t1", "v123", "t32", "v32"))
        + ignoring (foo, t2) group_right ()
        (label_set(100, "t1", "v123") or label_set(time(), "t1", "v123", "t2", "v3"))
        )"#;
        let mut r1 = make_result(&[2000_f64, 2400.0, 2800.0, 3200.0, 3600.0, 4000.0]);
        r1.metric.set("t1", "v123");
        r1.metric.set("t2", "v3");

        let mut r2 = make_result(&[1100_f64, 1300.0, 1500.0, 1700.0, 1900.0, 2100.0]);
        r2.metric.set("t1", "v123");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn histogram_quantile_scalar() {
        let q = "histogram_quantile(0.6, time())";
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_share_scalar() {
        let q = "histogram_share(123, time())";
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_quantile_single_value_no_le() {
        let q = r#"histogram_quantile(0.6, label_set(100, "foo", "bar"))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_share_single_value_no_le() {
        let q = r#"histogram_share(123, label_set(100, "foo", "bar"))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_quantile_single_value_invalid_le() {
        let q = r#"histogram_quantile(0.6, label_set(100, "le", "foobar"))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_share_single_value_invalid_le() {
        let q = r#"histogram_share(50, label_set(100, "le", "foobar"))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_quantile_single_value_inf_le() {
        let q = r#"histogram_quantile(0.6, label_set(100, "le", "+Inf"))"#;
        test_query(q, vec![]);

        let q = r#"histogram_quantile(0.6, label_set(100, "le", "200"))"#;
        let r = make_result(&[120_f64, 120.0, 120.0, 120.0, 120.0, 120.0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantile_zero_value_inf_le() {
        let q = r#"histogram_quantile(0.6, (
        label_set(100, "le", "+Inf"),
        label_set(0, "le", "42"),
        ))"#;
        assert_result_eq(q, &[42.0, 42.0, 42.0, 42.0, 42.0, 42.0]);
    }

    #[test]
    fn stdvar_over_time() {
        assert_result_eq(
            "round(stdvar_over_time(rand(0)[200s:5s]), 0.001)",
            &[0.085, 0.082, 0.078, 0.101, 0.059, 0.074],
        );
    }

    #[test]
    fn histogram_stdvar() {
        let q = "round(histogram_stdvar(histogram_over_time(rand(0)[200s:5s])), 0.001)";
        assert_result_eq(q, &[0.079, 0.089, 0.089, 0.071, 0.1, 0.082]);
    }

    #[test]
    fn stddev_over_time() {
        let q = "round(stddev_over_time(rand(0)[200s:5s]), 0.001)";
        assert_result_eq(q, &[0.291, 0.287, 0.28, 0.318, 0.244, 0.272]);
    }

    #[test]
    fn histogram_stddev() {
        let q = "round(histogram_stddev(histogram_over_time(rand(0)[200s:5s])), 0.001)";
        assert_result_eq(q, &[0.288, 0.285, 0.278, 0.32, 0.239, 0.27]);
    }

    #[test]
    fn avg_over_time() {
        let q = "round(avg_over_time(rand(0)[200s:5s]), 0.001)";
        assert_result_eq(q, &[0.467, 0.488, 0.462, 0.486, 0.441, 0.474]);
    }

    #[test]
    fn histogram_avg() {
        let q = "round(histogram_avg(histogram_over_time(rand(0)[200s:5s])), 0.001)";
        assert_result_eq(q, &[0.467, 0.485, 0.464, 0.488, 0.44, 0.473]);
    }

    #[test]
    fn histogram_share_single_value_valid_le() {
        let q = r#"histogram_share(300, label_set(100, "le", "200"))"#;
        assert_result_eq(q, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let q = r#"histogram_share(80, label_set(100, "le", "200"))"#;
        assert_result_eq(q, &[0.4, 0.4, 0.4, 0.4, 0.4, 0.4]);

        let q = r#"histogram_share(200, label_set(100, "le", "200"))"#;
        assert_result_eq(q, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn histogram_quantile_single_value_valid_le_bounds_label() {
        let q = r#"sort(histogram_quantile(0.6, label_set(100, "le", "200"), "foobar"))"#;
        let mut r1 = make_result(&[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        r1.metric.set("foobar", "lower");
        let r2 = make_result(&[120_f64, 120.0, 120.0, 120.0, 120.0, 120.0]);
        let mut r3 = make_result(&[200_f64, 200.0, 200.0, 200.0, 200.0, 200.0]);
        r3.metric.set("foobar", "upper");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_share_single_value_valid_le_bounds_label() {
        let q = r#"sort(histogram_share(120, label_set(100, "le", "200"), "foobar"))"#;
        let mut r1 = make_result(&[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        r1.metric.set("foobar", "lower");
        let r2 = make_result(&[0.6, 0.6, 0.6, 0.6, 0.6, 0.6]);
        let mut r3 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r3.metric.set("foobar", "upper");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_quantile_single_value_valid_le_max_phi() {
        let q = r#"histogram_quantile(1, (
        label_set(100, "le", "200"),
        label_set(0, "le", "55"),
        ))"#;
        assert_result_eq(q, &[200.0, 200.0, 200.0, 200.0, 200.0, 200.0]);
    }

    #[test]
    fn histogram_quantile_single_value_valid_le_max_le() {
        let q = r#"histogram_share(200, (
        label_set(100, "le", "200"),
        label_set(0, "le", "55"),
        ))"#;
        assert_result_eq(q, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn histogram_quantile_single_value_valid_le_min_phi() {
        let q = r#"histogram_quantile(0, (
        label_set(100, "le", "200"),
        label_set(0, "le", "55"),
        ))"#;
        assert_result_eq(q, &[55.0, 55.0, 55.0, 55.0, 55.0, 55.0]);
    }

    #[test]
    fn histogram_share_single_value_valid_le_min_le() {
        let q = r#"histogram_share(0, (
        label_set(100, "le", "200"),
        label_set(0, "le", "55"),
        ))"#;
        assert_result_eq(q, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn histogram_share_single_value_valid_le_low_le() {
        let q = r#"histogram_share(55, (
        label_set(100, "le", "200"),
        label_set(0, "le", "55"),
        ))"#;
        assert_result_eq(q, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn histogram_share_single_value_valid_le_mid_le() {
        let q = r#"histogram_share(105, (
        label_set(100, "le", "200"),
        label_set(0, "le", "55"),
        ))"#;
        assert_result_eq(
            q,
            &[
                0.3448275862068966,
                0.3448275862068966,
                0.3448275862068966,
                0.3448275862068966,
                0.3448275862068966,
                0.3448275862068966,
            ],
        );
    }

    #[test]
    fn histogram_quantile_single_value_valid_le_min_phi_no_zero_bucket() {
        let q = r#"histogram_quantile(0, label_set(100, "le", "200"))"#;
        assert_result_eq(q, &[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn histogram_quantile_scalar_phi() {
        let q = r#"histogram_quantile(time() / 2 / 1e3, label_set(100, "le", "200"))"#;
        assert_result_eq(q, &[100.0, 120.0, 140.0, 160.0, 180.0, 200.0]);
    }

    #[test]
    fn histogram_share_scalar_phi() {
        let q = r#"histogram_share(time() / 8, label_set(100, "le", "200"))"#;
        assert_result_eq(q, &[0.625, 0.75, 0.875, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn histogram_quantile_valid() {
        let q = r#"sort(histogram_quantile(0.6,
        label_set(90, "foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        or label_set(200, "tag", "xx", "le", "10")
        or label_set(300, "tag", "xx", "le", "30")
        ))"#;
        let mut r1 = make_result(&[9_f64, 9.0, 9.0, 9.0, 9.0, 9.0]);
        r1.metric.set("tag", "xx");
        let mut r2 = make_result(&[30_f64, 30.0, 30.0, 30.0, 30.0, 30.0]);
        r2.metric.set("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn histogram_share_valid() {
        let q = r#"sort(histogram_share(25,
        label_set(90, "foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        or label_set(200, "tag", "xx", "le", "10")
        or label_set(300, "tag", "xx", "le", "30")
        ))"#;
        let mut r1 = make_result(&[0.325, 0.325, 0.325, 0.325, 0.325, 0.325]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[
            0.9166666666666666,
            0.9166666666666666,
            0.9166666666666666,
            0.9166666666666666,
            0.9166666666666666,
            0.9166666666666666,
        ]);
        r2.metric.set("tag", "xx");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn histogram_quantile_negative_bucket_count() {
        let q = r#"histogram_quantile(0.6,
        label_set(90, "foo", "bar", "le", "10")
        or label_set(-100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        )"#;
        let mut r = make_result(&[30_f64, 30.0, 30.0, 30.0, 30.0, 30.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantile_nan_bucket_count_some() {
        let q = r#"round(histogram_quantile(0.6,
        label_set(90, "foo", "bar", "le", "10")
        or label_set(NaN, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        ),0.01)"#;
        let mut r = make_result(&[30.0, 30.0, 30.0, 30.0, 30.0, 30.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantile_normal_bucket_count() {
        let q = r#"histogram_quantile(0.2,
        label_set(0, "foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        )"#;
        let mut r = make_result(&[22_f64, 22.0, 22.0, 22.0, 22.0, 22.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantiles() {
        let q = r#"sort_by_label(histogram_quantiles("phi", 0.2, 0.3,
        label_set(0, "foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        ), "phi")"#;
        let mut r1 = make_result(&[22_f64, 22.0, 22.0, 22.0, 22.0, 22.0]);
        r1.metric.set("foo", "bar");
        r1.metric.set("phi", "0.2");

        let mut r2 = make_result(&[28_f64, 28.0, 28.0, 28.0, 28.0, 28.0]);
        r2.metric.set("foo", "bar");
        r2.metric.set("phi", "0.3");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn histogram_share_normal_bucket_count() {
        let q = r#"histogram_share(35,
        label_set(0, "foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        )"#;
        let mut r = make_result(&[
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
        ]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantile_normal_bucket_count_bounds_label() {
        let q = r#"sort(histogram_quantile(0.2,
        label_set(0, "foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf"),
        "xxx"
        ))"#;

        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        r1.metric.set("xxx", "lower");

        let mut r2 = make_result(&[22_f64, 22.0, 22.0, 22.0, 22.0, 22.0]);
        r2.metric.set("foo", "bar");

        let mut r3 = make_result(&[30_f64, 30.0, 30.0, 30.0, 30.0, 30.0]);
        r3.metric.set("foo", "bar");
        r3.metric.set("xxx", "upper");

        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_share_normal_bucket_count_bounds_label() {
        let q = r#"sort(histogram_share(22,
        label_set(0, "foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf"),
        "xxx"
        ))"#;
        let mut r1 = make_result(&[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        r1.metric.set("foo", "bar");
        r1.metric.set("xxx", "lower");

        let mut r2 = make_result(&[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]);
        r2.metric.set("foo", "bar");
        let mut r3 = make_result(&[
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
        ]);
        r3.metric.set("foo", "bar");
        r3.metric.set("xxx", "upper");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_quantile_zero_bucket_count() {
        let q = r#"histogram_quantile(0.6,
        label_set(0, "foo", "bar", "le", "10")
        or label_set(0, "foo", "bar", "le", "30")
        or label_set(0, "foo", "bar", "le", "+Inf")
        )"#;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_quantile_nan_bucket_count_all() {
        let q = r#"histogram_quantile(0.6,
        label_set(NAN, "foo", "bar", "le", "10")
        or label_set(NAN, "foo", "bar", "le", "30")
        or label_set(NAN, "foo", "bar", "le", "+Inf")
        )"#;
        test_query(q, vec![]);
    }

    #[test]
    fn buckets_limit_zero() {
        let q = r#"buckets_limit(0, (
        alias(label_set(100, "le", "INF", "x", "y"), "metric"),
        alias(label_set(50, "le", "120", "x", "y"), "metric"),
        ))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn buckets_limit_unused() {
        let q = r#"sort(buckets_limit(5, (
        alias(label_set(100, "le", "INF", "x", "y"), "metric"),
        alias(label_set(50, "le", "120", "x", "y"), "metric"),
        )))"#;

        let mut r1 = make_result(&[50_f64, 50.0, 50.0, 50.0, 50.0, 50.0]);
        r1.metric.set_metric_group("metric");
        r1.metric.set("le", "120");
        r1.metric.set("x", "y");

        let mut r2 = make_result(&[100_f64, 100.0, 100.0, 100.0, 100.0, 100.0]);
        r2.metric.set_metric_group("metric");
        r2.metric.set("le", "INF");
        r2.metric.set("x", "y");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn buckets_limit_used() {
        let q = r#"sort(buckets_limit(2, (
        alias(label_set(100, "le", "INF", "x", "y"), "metric"),
        alias(label_set(98, "le", "300", "x", "y"), "metric"),
        alias(label_set(52, "le", "200", "x", "y"), "metric"),
        alias(label_set(50, "le", "120", "x", "y"), "metric"),
        alias(label_set(20, "le", "70", "x", "y"), "metric"),
        alias(label_set(10, "le", "30", "x", "y"), "metric"),
        alias(label_set(9, "le", "10", "x", "y"), "metric"),
        )))"#;
        let mut r1 = make_result(&[9_f64, 9.0, 9.0, 9.0, 9.0, 9.0]);
        r1.metric.set_metric_group("metric");
        r1.metric.set("le", "10");
        r1.metric.set("x", "y");

        let mut r2 = make_result(&[98_f64, 98.0, 98.0, 98.0, 98.0, 98.0]);
        r2.metric.set_metric_group("metric");
        r2.metric.set("le", "300");
        r2.metric.set("x", "y");

        let mut r3 = make_result(&[100_f64, 100.0, 100.0, 100.0, 100.0, 100.0]);
        r3.metric.set_metric_group("metric");
        r3.metric.set("le", "INF");
        r3.metric.set("x", "y");

        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn prometheus_buckets_missing_vmrange() {
        let q = r#"sort(prometheus_buckets((
        alias(label_set(time()/20, "foo", "bar", "le", "0.2"), "xyz"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "foobar"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "30...foobar"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "30...40"), "xxx"),
        alias(label_set(time()/80, "foo", "bar", "vmrange", "0...900", "le", "54"), "yyy"),
        alias(label_set(time()/40, "foo", "bar", "vmrange", "900...+Inf", "le", "2343"), "yyy"),
        )))"#;
        let mut r1 = make_result(&[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        r1.metric.set_metric_group("xxx");
        r1.metric.set("foo", "bar");
        r1.metric.set("le", "30");

        let mut r2 = make_result(&[10_f64, 12.0, 14.0, 16.0, 18.0, 20.0]);
        r2.metric.set_metric_group("xxx");
        r2.metric.set("foo", "bar");
        r2.metric.set("le", "40");

        let mut r3 = make_result(&[10_f64, 12.0, 14.0, 16.0, 18.0, 20.0]);
        r3.metric.set_metric_group("xxx");
        r3.metric.set("foo", "bar");
        r3.metric.set("le", "+Inf");

        let mut r4 = make_result(&[12.5, 15.0, 17.5, 20.0, 22.5, 25.0]);
        r4.metric.set_metric_group("yyy");
        r4.metric.set("foo", "bar");
        r4.metric.set("le", "900");

        let mut r5 = make_result(&[37.5, 45.0, 52.5, 60.0, 67.5, 75.0]);
        r5.metric.set_metric_group("yyy");
        r5.metric.set("foo", "bar");
        r5.metric.set("le", "+Inf");

        let mut r6 = make_result(&[50_f64, 60.0, 70.0, 80.0, 90.0, 100.0]);
        r6.metric.set_metric_group("xyz");
        r6.metric.set("foo", "bar");
        r6.metric.set("le", "0.2");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5, r6];
        test_query(q, result_expected)
    }

    #[test]
    fn prometheus_buckets_zero_vmrange_value() {
        let q = r#"sort(prometheus_buckets(label_set(0, "vmrange", "0...0")))"#;
        test_query(q, vec![])
    }

    #[test]
    fn prometheus_buckets_valid() {
        let q = r#"sort(prometheus_buckets((
        alias(label_set(90, "foo", "bar", "vmrange", "0...0"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.2"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "0.2...40"), "xxx"),
        alias(label_set(time()/10, "foo", "bar", "vmrange", "40...Inf"), "xxx"),
        )))"#;
        let mut r1 = make_result(&[90_f64, 90.0, 90.0, 90.0, 90.0, 90.0]);
        r1.metric.set_metric_group("xxx");
        r1.metric.set("foo", "bar");
        r1.metric.set("le", "0");

        let mut r2 = make_result(&[140_f64, 150.0, 160.0, 170.0, 180.0, 190.0]);
        r2.metric.set_metric_group("xxx");
        r2.metric.set("foo", "bar");
        r2.metric.set("le", "0.2");

        let mut r3 = make_result(&[150_f64, 162.0, 174.0, 186.0, 198.0, 210.0]);
        r3.metric.set_metric_group("xxx");
        r3.metric.set("foo", "bar");
        r3.metric.set("le", "40");

        let mut r4 = make_result(&[250_f64, 282.0, 314.0, 346.0, 378.0, 410.0]);
        r4.metric.set_metric_group("xxx");
        r4.metric.set("foo", "bar");
        r4.metric.set("le", "Inf");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4];
        test_query(q, result_expected);
    }

    #[test]
    fn prometheus_buckets_overlapped_ranges() {
        let q = r#"sort(prometheus_buckets((
        alias(label_set(90, "foo", "bar", "vmrange", "0...0"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.2"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0.2...0.25"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.26"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "0.2...40"), "xxx"),
        alias(label_set(time()/10, "foo", "bar", "vmrange", "40...Inf"), "xxx"),
        )))"#;
        let mut r1 = make_result(&[90_f64, 90.0, 90.0, 90.0, 90.0, 90.0]);
        r1.metric.set_metric_group("xxx");
        r1.metric.set("foo", "bar");
        r1.metric.set("le", "0");

        let mut r2 = make_result(&[140_f64, 150.0, 160.0, 170.0, 180.0, 190.0]);
        r2.metric.set_metric_group("xxx");
        r2.metric.set("foo", "bar");
        r2.metric.set("le", "0.2");

        let mut r3 = make_result(&[190_f64, 210.0, 230.0, 250.0, 270.0, 290.0]);
        r3.metric.set_metric_group("xxx");
        r3.metric.set("foo", "bar");
        r3.metric.set("le", "0.25");

        let mut r4 = make_result(&[240_f64, 270.0, 300.0, 330.0, 360.0, 390.0]);
        r4.metric.set_metric_group("xxx");
        r4.metric.set("foo", "bar");
        r4.metric.set("le", "0.26");

        let mut r5 = make_result(&[250_f64, 282.0, 314.0, 346.0, 378.0, 410.0]);
        r5.metric.set_metric_group("xxx");
        r5.metric.set("foo", "bar");
        r5.metric.set("le", "40");

        let mut r6 = make_result(&[350_f64, 402.0, 454.0, 506.0, 558.0, 610.0]);
        r6.metric.set_metric_group("xxx");
        r6.metric.set("foo", "bar");
        r6.metric.set("le", "Inf");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5, r6];
        test_query(q, result_expected)
    }

    #[test]
    fn prometheus_buckets_overlapped_ranges_at_the_end() {
        let q = r#"sort(prometheus_buckets((
        alias(label_set(90, "foo", "bar", "vmrange", "0...0"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.2"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0.2...0.25"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.25"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "0.2...40"), "xxx"),
        alias(label_set(time()/10, "foo", "bar", "vmrange", "40...Inf"), "xxx"),
        )))"#;
        let mut r1 = make_result(&[90_f64, 90.0, 90.0, 90.0, 90.0, 90.0]);
        r1.metric.set_metric_group("xxx");
        r1.metric.set("foo", "bar");
        r1.metric.set("le", "0");

        let mut r2 = make_result(&[140_f64, 150.0, 160.0, 170.0, 180.0, 190.0]);
        r2.metric.set_metric_group("xxx");
        r2.metric.set("foo", "bar");
        r2.metric.set("le", "0.2");

        let mut r3 = make_result(&[190_f64, 210.0, 230.0, 250.0, 270.0, 290.0]);
        r3.metric.set_metric_group("xxx");
        r3.metric.set("foo", "bar");
        r3.metric.set("le", "0.25");

        let mut r4 = make_result(&[200_f64, 222.0, 244.0, 266.0, 288.0, 310.0]);
        r4.metric.set_metric_group("xxx");
        r4.metric.set("foo", "bar");
        r4.metric.set("le", "40");

        let mut r5 = make_result(&[300_f64, 342.0, 384.0, 426.0, 468.0, 510.0]);
        r5.metric.set_metric_group("xxx");
        r5.metric.set("foo", "bar");
        r5.metric.set("le", "Inf");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5];
        test_query(q, result_expected)
    }

    #[test]
    fn median_over_time() {
        let q = "median_over_time({})";
        test_query(q, vec![]);

        // assert_result_eq(r#"median_over_time("foo")"#, &[]);
        assert_result_eq(
            "median_over_time(12)",
            &[12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
        );
    }

    #[test]
    fn sum() {
        assert_result_eq("sum(time()/100)", &[10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    }

    #[test]
    fn sum_scalar() {
        assert_result_eq("sum(123)", &[123.0, 123.0, 123.0, 123.0, 123.0, 123.0]);
    }

    #[test]
    fn sum_multi_arg() {
        assert_result_eq("sum(1, 2, 3)", &[6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn sum_union_scalars() {
        assert_result_eq("sum((1, 2, 3))", &[6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn sum_union_vectors() {
        let q = r#"sum((
            alias(1, "foo"),
            alias(2, "foo"),
            alias(3, "foo"),
        ))"#;
        assert_result_eq(q, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn sum_scalar_by_empty_parens_expr() {
        assert_result_eq(
            "sum(123) by ()",
            &[123.0, 123.0, 123.0, 123.0, 123.0, 123.0],
        );
    }

    #[test]
    fn sum_scalar_without_empty_parens_expr() {
        assert_result_eq(
            "sum(123) without ()",
            &[123.0, 123.0, 123.0, 123.0, 123.0, 123.0],
        );
    }

    #[test]
    fn test_mode() {
        let q = r#"mode((
        alias(3, "m1"),
        alias(2, "m2"),
        alias(3, "m3"),
        alias(4, "m4"),
        alias(3, "m5"),
        alias(2, "m6"),
        ))"#;
        assert_result_eq(q, &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn share() {
        let q = r#"sort_by_label(round(share((
            label_set(time()/100+10, "k", "v1"),
            label_set(time()/200+5, "k", "v2"),
            label_set(time()/110-10, "k", "v3"),
            label_set(time()/90-5, "k", "v4"),
        )), 0.001), "k")"#;
        let mut r1 = make_result(&[0.554, 0.521, 0.487, 0.462, 0.442, 0.426]);
        r1.metric.set("k", "v1");

        let mut r2 = make_result(&[0.277, 0.26, 0.243, 0.231, 0.221, 0.213]);
        r2.metric.set("k", "v2");

        let mut r3 = make_result(&[f64::NAN, 0.022, 0.055, 0.081, 0.1, 0.116]);
        r3.metric.set("k", "v3");

        let mut r4 = make_result(&[0.169, 0.197, 0.214, 0.227, 0.237, 0.245]);
        r4.metric.set("k", "v4");
        let result_expected = vec![r1, r2, r3, r4];
        test_query(q, result_expected);
    }

    #[test]
    fn sum_share() {
        let q = r#"round(sum(share((
            label_set(time()/100+10, "k", "v1"),
            label_set(time()/200+5, "k", "v2"),
            label_set(time()/110-10, "k", "v3"),
            label_set(time()/90-5, "k", "v4"),
        ))), 0.001)"#;
        let r = make_result(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let result_expected = vec![r];
        test_query(q, result_expected);
    }

    #[test]
    fn sum_share_by() {
        let q = r#"round(sum(share((
                label_set(time()/100+10, "k", "v1"),
                label_set(time()/200+5, "k", "v2", "a", "b"),
                label_set(time()/110-10, "k", "v1", "a", "b"),
                label_set(time()/90-5, "k", "v2"),
            )) by (k)), 0.001)"#;
        let r = make_result(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
        let result_expected = vec![r];
        test_query(q, result_expected)
    }

    #[test]
    fn zscore() {
        let q = r#"sort_by_label(round(zscore((
        label_set(time()/100+10, "k", "v1"),
        label_set(time()/200+5, "k", "v2"),
        label_set(time()/110-10, "k", "v3"),
        label_set(time()/90-5, "k", "v4"),
        )), 0.001), "k")"#;
        let mut r1 = make_result(&[1.482, 1.511, 1.535, 1.552, 1.564, 1.57]);
        r1.metric.set("k", "v1");
        let mut r2 = make_result(&[0.159, 0.058, -0.042, -0.141, -0.237, -0.329]);
        r2.metric.set("k", "v2");
        let mut r3 = make_result(&[-1.285, -1.275, -1.261, -1.242, -1.219, -1.193]);
        r3.metric.set("k", "v3");
        let mut r4 = make_result(&[-0.356, -0.294, -0.232, -0.17, -0.108, -0.048]);
        r4.metric.set("k", "v4");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4];
        test_query(q, result_expected);
    }

    #[test]
    fn avg_without() {
        assert_result_eq(
            "avg without (xx, yy) (123)",
            &[123.0, 123.0, 123.0, 123.0, 123.0, 123.0],
        );
    }

    #[test]
    fn histogram_scalar() {
        let q = r#"sort(histogram(123)+(
        label_set(0, "le", "1.000e2"),
        label_set(0, "le", "1.136e2"),
        label_set(0, "le", "1.292e2"),
        label_set(1, "le", "+Inf"),
        ))"#;
        let mut r1 = make_result(&[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        r1.metric.set("le", "1.136e2");

        let mut r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r2.metric.set("le", "1.292e2");

        let mut r3 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r3.metric.set("le", "+Inf");

        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_vector() {
        let q = r#"sort(histogram((
        label_set(1, "foo", "bar"),
        label_set(1.1, "xx", "yy"),
        alias(1.15, "foobar"),
        ))+(
        label_set(0, "le", "8.799e1"),
        label_set(0, "le", "1.000e+00"),
        label_set(0, "le", "1.292e+00"),
        label_set(1, "le", "+Inf"),
        ))"#;
        let mut r1 = make_result(&[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        r1.metric.set("le", "8.799e1");

        let mut r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r2.metric.set("le", "1.000e+00");
        let mut r3 = make_result(&[3_f64, 3.0, 3.0, 3.0, 3.0, 3.0]);
        r3.metric.set("le", "1.292e+00");
        let mut r4 = make_result(&[4_f64, 4.0, 4.0, 4.0, 4.0, 4.0]);
        r4.metric.set("le", "+Inf");

        test_query(q, vec![r1, r2, r3, r4])
    }

    #[test]
    fn geomean() {
        assert_result_eq("geomean(time()/100)", &[10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    }

    #[test]
    fn geomean_over_time() {
        let q = r#"round(geomean_over_time(alias(time()/100, "foobar")[3i]), 0.1)"#;
        let mut r = make_result(&[7.8, 9.9, 11.9, 13.9, 15.9, 17.9]);
        r.metric.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn sum2_time() {
        assert_result_eq(
            "sum2(time()/100)",
            &[100.0, 144.0, 196.0, 256.0, 324.0, 400.0],
        );
    }

    #[test]
    fn sum2_over_time() {
        assert_result_eq(
            r#"sum2_over_time(alias(time()/100, "foobar")[3i])"#,
            &[200.0, 308.0, 440.0, 596.0, 776.0, 980.0],
        );
    }

    #[test]
    fn range_over_time() {
        let q = r#"range_over_time(alias(time()/100, "foobar")[3i])"#;
        assert_result_eq(q, &[4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn sum_multi_vector() {
        let q = r#"sum(label_set(10, "foo", "bar") or label_set(time()/100, "baz", "sss"))"#;
        assert_result_eq(q, &[20.0, 22.0, 24.0, 26.0, 28.0, 30.0]);
    }

    #[test]
    fn geomean_multi_vector() {
        let q = r#"round(geomean(label_set(10, "foo", "bar") or label_set(time()/100, "baz", "sss")), 0.1)"#;
        assert_result_eq(q, &[10.0, 11.0, 11.8, 12.6, 13.4, 14.1]);
    }

    #[test]
    fn sum2_multi_vector() {
        let q = r#"sum2(label_set(10, "foo", "bar") or label_set(time()/100, "baz", "sss"))"#;
        assert_result_eq(q, &[200.0, 244.0, 296.0, 356.0, 424.0, 500.0]);
    }

    #[test]
    fn avg_multi_vector() {
        let q = r#"avg(label_set(10, "foo", "bar") or label_set(time()/100, "baz", "sss"))"#;
        assert_result_eq(q, &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn stddev_multi_vector() {
        let q = r#"stddev(label_set(10, "foo", "bar") or label_set(time()/100, "baz", "sss"))"#;
        assert_result_eq(q, &[0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn count_multi_vector() {
        let q = r#"count(label_set(time()<1500, "foo", "bar") or label_set(time()<1800.0, "baz", "sss"))"#;
        assert_result_eq(q, &[2.0, 2.0, 2.0, 1.0, NAN, NAN]);
    }

    #[test]
    fn sum_multi_vector_by_known_tag() {
        // sum(multi-vector) by (known-tag)
        let q = r#"sort(sum(label_set(10, "foo", "bar") or label_set(time()/100, "baz", "sss")) by (foo))"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        let r2 = make_result(&[10_f64, 12.0, 14.0, 16.0, 18.0, 20.0]);
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sum_multi_vector_by_known_tag_limit_1() {
        let q = r#"sum(label_set(10, "foo", "bar") or label_set(time()/100, "baz", "sss")) by (foo) limit 1"#;
        let mut r = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn sum_multi_vector_by_known_tags() {
        let q = r#"sum(label_set(10, "foo", "bar", "baz", "sss", "x", "y") or label_set(time()/100, "baz", "sss", "foo", "bar")) by (foo, baz, foo)"#;
        let mut r = make_result(&[20_f64, 22.0, 24.0, 26.0, 28.0, 30.0]);
        r.metric.set("baz", "sss");
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn sum_multi_vector_by_name() {
        let q = r#"sort(sum(label_set(10, "__name__", "bar", "baz", "sss", "x", "y") or label_set(time()/100, "baz", "sss", "__name__", "aaa")) by (__name__))"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set_metric_group("bar");
        let mut r2 = make_result(&[10_f64, 12.0, 14.0, 16.0, 18.0, 20.0]);
        r2.metric.set_metric_group("aaa");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn min_multi_vector_by_unknown_tag() {
        let q = r#"min(label_set(10, "foo", "bar") or label_set(time()/100/1.5, "baz", "sss")) by (unknowntag)"#;
        assert_result_eq(
            q,
            &[6.666666666666667, 8.0, 9.333333333333334, 10.0, 10.0, 10.0],
        );
    }

    #[test]
    fn max_multi_vector_by_unknown_tag() {
        let q = r#"max(label_set(10, "foo", "bar") or label_set(time()/100/1.5, "baz", "sss")) by (unknowntag)"#;
        assert_result_eq(
            q,
            &[
                10.0,
                10.0,
                10.0,
                10.666666666666666,
                12.0,
                13.333333333333334,
            ],
        );
    }

    #[test]
    fn quantile_over_time() {
        let q = r#"quantile_over_time(0.9, label_set(round(rand(0), 0.01), "__name__", "foo", "xx", "yy")[200s:5s])"#;
        let mut r = make_result(&[
            0.893,
            0.892,
            0.9510000000000001,
            0.8730000000000001,
            0.9250000000000002,
            0.891,
        ]);
        r.metric.set_metric_group("foo");
        r.metric.set("xx", "yy");

        test_query(q, vec![r]);
    }

    #[test]
    fn quantiles_over_time_single_sample() {
        let q = r#"sort_by_label(
        quantiles_over_time("phi", 0.5, 0.9, time()[100s:100s]),
        "phi",
        )"#;
        let mut r1 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r1.metric.set("phi", "0.5");

        let mut r2 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r2.metric.set("phi", "0.9");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn quantiles_over_time_multiple_samples() {
        let q = r#"sort_by_label(
        quantiles_over_time("phi", 0.5, 0.9,
        label_set(round(rand(0), 0.01), "__name__", "foo", "xx", "yy")[200s:5s]
        ),
        "phi",
        )"#;
        let mut r1 = make_result(&[0.46499999999999997, 0.57, 0.485, 0.54, 0.555, 0.515]);
        r1.metric.set_metric_group("foo");
        r1.metric.set("phi", "0.5");
        r1.metric.set("xx", "yy");

        let mut r2 = make_result(&[
            0.893,
            0.892,
            0.9510000000000001,
            0.8730000000000001,
            0.9250000000000002,
            0.891,
        ]);
        r2.metric.set_metric_group("foo");
        r2.metric.set("phi", "0.9");
        r2.metric.set("xx", "yy");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn count_values_over_time() {
        let q = r##"sort_by_label(
            count_values_over_time("foo", round(label_set(rand(0), "x", "y"), 0.4)[200s:5s]),
            "foo",
        )"##;
        let mut r1 = make_result(&[4.0, 8.0, 7.0, 6.0, 10.0, 9.0]);
        r1.metric.set("foo", "0");
        r1.metric.set("x", "y");

        let mut r2 = make_result(&[20.0, 13.0, 19.0, 18.0, 14.0, 13.0]);
        r2.metric.set("foo", "0.4");
        r2.metric.set("x", "y");

        let mut r3 = make_result(&[16.0, 19.0, 14.0, 16.0, 16.0, 18.0]);
        r3.metric.set("foo", "0.8");
        r3.metric.set("x", "y");

        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_over_time() {
        let q = r#"sort_by_label(histogram_over_time(alias(label_set(rand(0)*1.3+1.1, "foo", "bar"), "xxx")[200s:5s]), "vmrange")"#;
        let mut r1 = make_result(&[1_f64, 2.0, 2.0, 2.0, NAN, 1.0]);
        r1.metric.set("foo", "bar");
        r1.metric.set("vmrange", "1.000e+00...1.136e+00");

        let mut r2 = make_result(&[3_f64, 3.0, 4.0, 2.0, 8.0, 3.0]);
        r2.metric.set("foo", "bar");
        r2.metric.set("vmrange", "1.136e+00...1.292e+00");

        let mut r3 = make_result(&[7_f64, 7.0, 5.0, 3.0, 3.0, 9.0]);
        r3.metric.set("foo", "bar");
        r3.metric.set("vmrange", "1.292e+00...1.468e+00");

        let mut r4 = make_result(&[4_f64, 6.0, 5.0, 6.0, 4.0]);
        r4.metric.set("foo", "bar");
        r4.metric.set("vmrange", "1.468e+00...1.668e+00");

        let mut r5 = make_result(&[6_f64, 6.0, 9.0, 13.0, 7.0, 7.0]);
        r5.metric.set("foo", "bar");
        r5.metric.set("vmrange", "1.668e+00...1.896e+00");

        let mut r6 = make_result(&[5_f64, 9.0, 4.0, 6.0, 7.0, 9.0]);
        r6.metric.set("foo", "bar");
        r6.metric.set("vmrange", "1.896e+00...2.154e+00");

        let mut r7 = make_result(&[11_f64, 9.0, 10.0, 9.0, 9.0, 7.0]);
        r7.metric.set("foo", "bar");
        r7.metric.set("vmrange", "2.154e+00...2.448e+00");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5, r6, r7];
        test_query(q, result_expected)
    }

    #[test]
    fn sum_histogram_over_time_by_vmrange() {
        let q = r#"sort_by_label(
        buckets_limit(
        3,
        sum(histogram_over_time(alias(label_set(rand(0)*1.3+1.1, "foo", "bar"), "xxx")[200s:5s])) by (vmrange)
        ), "le"
        )"#;
        let mut r1 = make_result(&[40_f64, 40.0, 40.0, 40.0, 40.0, 40.0]);
        r1.metric.set("le", "+Inf");

        let mut r2 = make_result(&[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        r2.metric.set("le", "1.000e0");

        let mut r3 = make_result(&[40_f64, 40.0, 40.0, 40.0, 40.0, 40.0]);
        r3.metric.set("le", "2.448e0");

        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn sum_histogram_over_time() {
        let q = r#"sum(histogram_over_time(alias(label_set(rand(0)*1.3+1.1, "foo", "bar"), "xxx")[200s:5s]))"#;
        assert_result_eq(q, &[40.0, 40.0, 40.0, 40.0, 40.0, 40.0]);
    }

    #[test]
    fn duration_over_time() {
        let q = "duration_over_time((time()<1200)[600s:10s], 20s)";
        assert_result_eq(q, &[590.0, 580.0, 380.0, 180.0, NAN, NAN]);
    }

    #[test]
    fn share_gt_over_time() {
        let q = "share_gt_over_time(round(rand(0))[200s:10s], 0.7)";
        assert_result_eq(q, &[0.55, 0.45, 0.6, 0.5, 0.45, 0.35]);
    }

    #[test]
    fn share_eq_over_time() {
        let q = "share_eq_over_time(rand(0)[200s:10s], 0.7)";
        assert_result_eq(q, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn share_le_over_time() {
        let q = "share_le_over_time(rand(0)[200s:10s], 0.7)";
        assert_result_eq(q, &[0.75, 0.9, 0.5, 0.65, 0.8, 0.8]);
    }

    #[test]
    fn count_gt_over_time() {
        let q = "count_gt_over_time(rand(0)[200s:10s], 0.7)";
        assert_result_eq(q, &[5.0, 2.0, 10.0, 7.0, 4.0, 4.0]);
    }

    #[test]
    fn count_le_over_time() {
        let q = "count_le_over_time(rand(0)[200s:10s], 0.7)";
        assert_result_eq(q, &[15.0, 18.0, 10.0, 13.0, 16.0, 16.0]);
    }

    #[test]
    fn count_eq_over_time() {
        let q = "count_eq_over_time(round(5*rand(0))[200s:10s], 1)";
        assert_result_eq(q, &[3.0, 4.0, 3.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn count_ne_over_time() {
        let q = "count_ne_over_time(round(5*rand(0))[200s:10s], 1)";
        assert_result_eq(q, &[17.0, 16.0, 17.0, 14.0, 14.0, 14.0]);
    }

    #[test]
    fn sum_gt_over_time() {
        let q = "round(sum_gt_over_time(rand(0)[200s:10s], 0.7), 0.1)";
        assert_result_eq(q, &[4.3, 1.8, 8.5, 5.9, 3.4, 3.4]);
    }

    #[test]
    fn sum_le_over_time() {
        let q = "round(sum_le_over_time(rand(0)[200s:10s], 0.7), 0.1)";
        assert_result_eq(q, &[5.0, 6.5, 3.0, 4.1, 6.2, 4.1]);
    }

    #[test]
    fn sum_eq_over_time() {
        let q = "round(sum_eq_over_time(rand(0)[200s:10s], 0.7), 0.1)";
        assert_result_eq(q, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn increases_over_time() {
        assert_result_eq(
            "increases_over_time(rand(0)[200s:10s])",
            &[9.0, 14.0, 12.0, 11.0, 9.0, 11.0],
        );
    }

    #[test]
    fn decreases_over_time() {
        assert_result_eq(
            "decreases_over_time(rand(0)[200s:10s])",
            &[11.0, 6.0, 8.0, 9.0, 11.0, 9.0],
        );
    }

    #[test]
    fn limitk() {
        let q = r#"limitk(-1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn limitk_1() {
        // NOTE: the answer here is dependent on the hashing algo used to preserve consistent
        // ordering of the series. As such it depends on the hash and not the data. If the
        // implementation changes, it's legit to change the answer here.
        let q = r#"limitk(1, label_set(10, "foo", "bar") or label_set(time()/150, "xbaz", "sss"))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("xbaz", "sss");
        test_query(q, vec![r1]);
    }

    #[test]
    fn limitk_10() {
        let q = r#"sort(limitk(10, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r2.metric.set("baz", "sss");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn limitk_inf() {
        let q = r#"sort(limitk(inf, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r2.metric.set("baz", "sss");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn any() {
        let q = r#"any(label_set(10, "__name__", "x", "foo", "bar") or label_set(time()/150, "__name__", "y", "baz", "sss"))"#;
        let mut r = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r.metric.set_metric_group("x");
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn any_empty_series() {
        let q = r#"any(label_set(time()<0, "foo", "bar"))"#;
        test_query(q, vec![])
    }

    #[test]
    fn group_by_test() {
        let q = r#"group((
        label_set(5, "__name__", "data", "test", "three samples", "point", "a"),
        label_set(6, "__name__", "data", "test", "three samples", "point", "b"),
        label_set(7, "__name__", "data", "test", "three samples", "point", "c"),
        )) by (test)"#;
        let mut r = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.reset_measurement();
        r.metric.set("test", "three samples");
        test_query(q, vec![r]);
    }

    #[test]
    fn group_without_point() {
        let q = r#"group((
        label_set(5, "__name__", "data", "test", "three samples", "point", "a"),
        label_set(6, "__name__", "data", "test", "three samples", "point", "b"),
        label_set(7, "__name__", "data", "test", "three samples", "point", "c"),
        )) without (point)"#;
        let mut r = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.reset_measurement();
        r.metric.set("test", "three samples");
        test_query(q, vec![r]);
    }

    #[test]
    fn top_k() {
        let q =
            r#"sort(topk(-1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        test_query(q, vec![]);

        let q = r#"topk(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"))"#;
        let mut r1 = make_result(&[NAN, NAN, NAN, 10.666666666666666, 12.0, 13.333333333333334]);
        r1.metric.set("baz", "sss");
        let mut r2 = make_result(&[10_f64, 10.0, 10.0, NAN, NAN, NAN]);
        r2.metric.set("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn topk_min() {
        let q = r#"sort(topk_min(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn bottomk_min() {
        let q = r#"sort(bottomk_min(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        test_query(q, vec![r1]);
    }

    #[test]
    fn topk_max_1() {
        let q = r#"topk_max(1, histogram_over_time(alias(label_set(rand(0)*1.3+1.1, "foo", "bar"), "xxx")[200s:5s]))"#;
        let mut r = make_result(&[6_f64, 6.0, 9.0, 13.0, 7.0, 7.0]);
        r.metric.set("foo", "bar");
        r.metric.set("vmrange", "1.668e+00...1.896e+00");

        test_query(q, vec![r]);
    }

    #[test]
    fn topk_max() {
        let q =
            r#"topk_max(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        test_query(q, vec![r1]);

        let q = r#"sort_desc(topk_max(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"), "remaining_sum=foo"))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        let mut r2 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r2.metric.set("remaining_sum", "foo");

        test_query(q, vec![r1, r2]);

        let q = r#"sort_desc(topk_max(2, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"), "remaining_sum"))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        let mut r2 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r2.metric.set("foo", "bar");

        test_query(q, vec![r1, r2]);

        let q = r#"sort_desc(topk_max(3, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"), "remaining_sum"))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        let mut r2 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r2.metric.set("foo", "bar");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn bottomk_max() {
        let q = r#"sort(bottomk_max(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn topk_avg() {
        let q = r#"sort(topk_avg(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        test_query(q, vec![r1]);
    }

    #[test]
    fn bottomk_avg() {
        let q = r#"sort(bottomk_avg(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        test_query(q, vec![r1])
    }

    #[test]
    fn topk_median_1() {
        let q = r#"sort(topk_median(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        test_query(q, vec![r1])
    }

    #[test]
    fn topk_last_1() {
        let q = r#"sort(topk_last(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        test_query(q, vec![r1]);
    }

    #[test]
    fn bottomk_median() {
        let q = r#"sort(bottomk_median(1, label_set(10, "foo", "bar") or label_set(time()/15, "baz", "sss")))"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn bottomk_last() {
        let q = r#"sort(bottomk_last(1, label_set(10, "foo", "bar") or label_set(time()/15, "baz", "sss")))"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn topk_nan_timeseries() {
        let q = r#"topk(1, label_set(NaN, "foo", "bar") or label_set(time()/150, "baz", "sss")) default 0"#;
        let mut r1 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r1.metric.set("baz", "sss");
        test_query(q, vec![r1]);
    }

    #[test]
    fn topk_2() {
        let q =
            r#"sort(topk(2, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r2.metric.set("baz", "sss");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn topk_nan() {
        let q = r#"sort(topk(NaN, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn topk_100500() {
        let q = r#"sort(topk(100500, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[
            6.666666666666667,
            8.0,
            9.333333333333334,
            10.666666666666666,
            12.0,
            13.333333333333334,
        ]);
        r2.metric.set("baz", "sss");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn bottomk() {
        let q = r#"bottomk(1, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss") or label_set(time()<100, "a", "b"))"#;
        let mut r1 = make_result(&[NAN, NAN, NAN, 10.0, 10.0, 10.0]);
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[6.666666666666667, 8.0, 9.333333333333334, NAN, NAN, NAN]);
        r2.metric.set("baz", "sss");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn keep_last_value() {
        let q = r#"keep_last_value(label_set(time() < 1300 default time() > 1700, "__name__", "foobar", "x", "y"))"#;
        let mut r1 = make_result(&[1000_f64, 1200.0, 1200.0, 1200.0, 1800.0, 2000.0]);
        r1.metric.set_metric_group("foobar");
        r1.metric.set("x", "y");
        test_query(q, vec![r1]);
    }

    #[test]
    fn keep_next_value() {
        let q = r#"keep_next_value(label_set(time() < 1300 default time() > 1700, "__name__", "foobar", "x", "y"))"#;
        let mut r1 = make_result(&[1000_f64, 1200.0, 1800.0, 1800.0, 1800.0, 2000.0]);
        r1.metric.set_metric_group("foobar");
        r1.metric.set("x", "y");
        test_query(q, vec![r1]);
    }

    #[test]
    fn interpolate() {
        let q = r#"interpolate(label_set(time() < 1300 default time() > 1700, "__name__", "foobar", "x", "y"))"#;
        let mut r1 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r1.metric.set_metric_group("foobar");
        r1.metric.set("x", "y");
        test_query(q, vec![r1]);
    }

    #[test]
    fn interpolate_tail() {
        let q = "interpolate(time() < 1300)";
        assert_result_eq(q, &[1000_f64, 1200.0, NAN, NAN, NAN, NAN]);
    }

    #[test]
    fn interpolate_head() {
        let q = "interpolate(time() > 1500)";
        assert_result_eq(q, &[NAN, NAN, NAN, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn interpolate_tail_head_and_middle() {
        let q =
            "interpolate(time() > 1100 and time() < 1300 default time() > 1700 and time() < 1900)";
        assert_result_eq(q, &[NAN, 1200.0, 1400.0, 1600.0, 1800.0, NAN]);
    }

    #[test]
    fn distinct_over_time_err() {
        assert_result_eq(
            "distinct_over_time((time() < 1700)[2.5i])",
            &[3.0, 3.0, 3.0, 3.0, 2.0, 1.0],
        );
    }

    #[test]
    fn distinct_over_time() {
        assert_result_eq(
            "distinct_over_time((time() < 1700)[500s])",
            &[3.0, 3.0, 3.0, 3.0, 2.0, 1.0],
        );
        assert_result_eq(
            "distinct_over_time((time() < 1700)[2.5i])",
            &[3.0, 3.0, 3.0, 3.0, 2.0, 1.0],
        );
    }

    #[test]
    fn distinct() {
        let q = r#"distinct(union(
        1+time() > 1100,
        label_set(time() > 1700, "foo", "bar"),
        ))"#;
        assert_result_eq(q, &[NAN, 1.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn vector2_if_vector1() {
        let q = r#"(
        label_set(time()/10, "x", "y"),
        label_set(time(), "foo", "bar", "__name__", "x"),
        ) if (
        label_set(time()>1400.0, "foo", "bar"),
        )"#;
        let mut r = make_result(&[NAN, NAN, NAN, 1600.0, 1800.0, 2000.0]);
        r.metric.set_metric_group("x");
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn vector2_if_vector2() {
        let q = r#"sort((
        label_set(time()/10, "x", "y"),
        label_set(time(), "foo", "bar", "__name__", "x"),
        ) if (
        label_set(time()>1400.0, "foo", "bar"),
        label_set(time()<1400.0, "x", "y"),
        ))"#;
        let mut r1 = make_result(&[100_f64, 120.0, NAN, NAN, NAN, NAN]);
        r1.metric.set("x", "y");
        let mut r2 = make_result(&[NAN, NAN, NAN, 1600.0, 1800.0, 2000.0]);
        r2.metric.set_metric_group("x");
        r2.metric.set("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn scalar_if_vector1() {
        let q = r#"time() if (
        label_set(123, "foo", "bar"),
        )"#;
        test_query(q, vec![]);
    }

    #[test]
    fn scalar_if_vector2() {
        let q = r#"time() if (
        label_set(123, "foo", "bar"),
        alias(time() > 1400.0, "xxx"),
        )"#;
        assert_result_eq(q, &[NAN, NAN, NAN, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn if_default() {
        let q = "time() if time() > 1400 default -time()";
        assert_result_eq(q, &[-1000.0, -1200.0, -1400.0, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn ifnot_default() {
        let q = "time() ifnot time() > 1400 default -time()";
        assert_result_eq(q, &[1000_f64, 1200.0, 1400.0, -1600.0, -1800.0, -2000.0]);
    }

    #[test]
    fn ifnot() {
        let q = "time() ifnot time() > 1400";
        assert_result_eq(q, &[1000_f64, 1200.0, 1400.0, NAN, NAN, NAN]);
    }

    #[test]
    fn ifnot_no_matching_timeseries() {
        let q = r#"label_set(time(), "foo", "bar") ifnot label_set(time() > 1400.0, "x", "y")"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn test_quantile() {
        let expected = [NEG_INF; 6];
        let q =
            r#"quantile(-2, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"))"#;
        assert_result_eq(q, &expected);

        let q =
            r#"quantile(0.2, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"))"#;
        assert_result_eq(
            q,
            &[
                7.333333333333334,
                8.4,
                9.466666666666669,
                10.133333333333333,
                10.4,
                10.666666666666668,
            ],
        );

        let q =
            r#"quantile(0.5, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"))"#;
        assert_result_eq(
            q,
            &[
                8.333333333333334,
                9.0,
                9.666666666666668,
                10.333333333333332,
                11.0,
                11.666666666666668,
            ],
        );
    }

    #[test]
    fn quantiles() {
        let q = r#"sort(quantiles("phi", 0.2, 0.5, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"#;
        let mut r1 = make_result(&[
            7.333333333333334,
            8.4,
            9.466666666666669,
            10.133333333333333,
            10.4,
            10.666666666666668,
        ]);
        r1.metric.set("phi", "0.2");
        let mut r2 = make_result(&[
            8.333333333333334,
            9.0,
            9.666666666666668,
            10.333333333333332,
            11.0,
            11.666666666666668,
        ]);
        r2.metric.set("phi", "0.5");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn median() {
        let q = r#"median(label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"))"#;
        let r = make_result(&[
            8.333333333333334,
            9.0,
            9.666666666666668,
            10.333333333333332,
            11.0,
            11.666666666666668,
        ]);
        test_query(q, vec![r]);

        let q = r#"median(union(label_set(10, "foo", "bar"), label_set(time()/150, "baz", "sss"), time()/200))"#;
        assert_result_eq(
            q,
            &[6.666666666666667, 8.0, 9.333333333333334, 10.0, 10.0, 10.0],
        );
    }

    #[test]
    fn quantile_3() {
        let q =
            r#"quantile(3, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"))"#;
        assert_result_eq(q, &[INF, INF, INF, INF, INF, INF]);
    }

    #[test]
    fn quantile_nan() {
        let q =
            r#"quantile(NaN, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss"))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn mad() {
        let q = r#"mad(
        alias(time(), "metric1"),
        alias(time()*1.5, "metric2"),
        label_set(time()*0.9, "baz", "sss"),
        )"#;
        assert_result_eq(q, &[100.0, 120.0, 140.0, 160.0, 180.0, 200.0]);
    }

    #[test]
    fn outliers_iqr() {
        let q = r#"sort(outliers_iqr((
            alias(time(), "m1"),
            alias(time()*1.5, "m2"),
            alias(time()*10, "m3"),
            alias(time()*1.2, "m4"),
            alias(time()*0.1, "m5"),
        )))"#;
        let mut r1 = make_result(&[100.0, 120.0, 140.0, 160.0, 180.0, 200.0]);
        r1.metric.measurement = "m5".to_string();
        let mut r2 = make_result(&[10000.0, 12000.0, 14000.0, 16000.0, 18000.0, 20000.0]);
        r2.metric.measurement = "m3".to_string();
        test_query(q, vec![r1, r2])
    }

    #[test]
    fn outliers_mad_1() {
        let q = r#"outliers_mad(1, (
        alias(time(), "metric1"),
        alias(time()*1.5, "metric2"),
        label_set(time()*0.9, "baz", "sss"),
        ))"#;
        let mut r = make_result(&[1500_f64, 1800.0, 2100.0, 2400.0, 2700.0, 3000.0]);
        r.metric.set_metric_group("metric2");
        test_query(q, vec![r]);
    }

    #[test]
    fn outliers_mad_5() {
        let q = r#"outliers_mad(5, (
        alias(time(), "metric1"),
        alias(time()*1.5, "metric2"),
        label_set(time()*0.9, "baz", "sss"),
        ))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn outliersk_0() {
        let q = r#"outliersk(0, (
        label_set(1300, "foo", "bar"),
        label_set(time(), "baz", "sss"),
        ))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn outliersk_1() {
        let q = r#"outliersk(1, (
        label_set(2000.0, "foo", "bar"),
        label_set(time(), "baz", "sss"),
        ))"#;
        let mut r = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r.metric.set("baz", "sss");
        test_query(q, vec![r]);
    }

    #[test]
    fn outliersk_3() {
        let q = r#"sort_desc(outliersk(3, (
        label_set(1300, "foo", "bar"),
        label_set(time(), "baz", "sss"),
        )))"#;
        let mut r1 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r1.metric.set("baz", "sss");
        let mut r2 = make_result(&[1300_f64, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0]);
        r2.metric.set("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn range_trim_outliers() {
        let q = "range_trim_outliers(0.5, time())";
        let r = make_result(&[f64::NAN, f64::NAN, 1400.0, 1600.0, f64::NAN, f64::NAN]);
        let result_expected = vec![r];
        test_query(q, result_expected);
    }

    #[test]
    fn range_trim_outliers_1() {
        let q = "range_trim_outliers(0.5, time() > 1200)";
        let r = make_result(&[f64::NAN, f64::NAN, f64::NAN, 1600.0, 1800.00, f64::NAN]);
        let result_expected = vec![r];
        test_query(q, result_expected);
    }

    #[test]
    fn range_trim_spikes() {
        let q = "range_trim_spikes(0.2, time())";
        let mut r = QueryResult::default();
        r.metric = MetricName::default();
        r.values = vec![f64::NAN, 1200_f64, 1400_f64, 1600_f64, 1800_f64, f64::NAN];
        r.timestamps = Vec::from(TIMESTAMPS_EXPECTED);
        test_query(q, vec![r])
    }

    #[test]
    fn range_trim_spikes_1() {
        let q = "range_trim_spikes(0.2, time() > 1200 <= 1800)";
        let r = make_result(&[f64::NAN, f64::NAN, f64::NAN, 1600.0, f64::NAN, f64::NAN]);
        let result_expected = vec![r];
        test_query(q, result_expected);
    }

    #[test]
    fn range_trim_zscore() {
        let q = r#"range_trim_zscore(0.9, time())"#;
        let r = make_result(&[f64::NAN, 1200.0, 1400.0, 1600.0, 1800.0, f64::NAN]);
        let result_expected = vec![r];
        test_query(q, result_expected)
    }

    #[test]
    fn range_trim_zscore_1() {
        let q = r#"round(range_zscore(time() > 1200 < 1800), 0.1)"#;
        let r = make_result(&[f64::NAN, f64::NAN, -1.0, 1.0, f64::NAN, f64::NAN]);
        let result_expected = vec![r];
        test_query(q, result_expected)
    }

    #[test]
    fn range_zscore() {
        let q = "round(range_zscore(time()), 0.1)";
        let r = make_result(&[-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]);
        test_query(q, vec![r])
    }

    #[test]
    fn range_quantile() {
        let q = "range_quantile(0.5, time())";
        let r = make_result(&[1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn range_quantile_1() {
        let q = "range_quantile(0.5, time() > 1200 < 2000)";
        let r = make_result(&[1600.0, 1600.0, 1600.0, 1600.0, 1600.0, 1600.0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn range_stddev() {
        let q = "round(range_stddev(time()), 0.01)";
        let r = make_result(&[341.57, 341.57, 341.57, 341.57, 341.57, 341.57]);
        test_query(q, vec![r]);
    }

    #[test]
    fn range_stddev_1() {
        let q = "round(range_stddev(time() > 1200 < 1800),0.01)";
        let r = make_result(&[100.0, 100.0, 100.0, 100.0, 100.0, 100.0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn range_stdvar() {
        let q = "round(range_stdvar(time()), 0.01)";
        let r = make_result(&[
            116666.67, 116666.67, 116666.67, 116666.67, 116666.67, 116666.67,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn range_stdvar_1() {
        let q = "round(range_stdvar(time() > 1200 < 1800),0.01)";
        let r = make_result(&[
            10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0,
        ]);
        test_query(q, vec![r]);
    }

    #[test]
    fn range_median() {
        let q = "range_median(time())";
        let r = make_result(&[1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0]);
        test_query(q, vec![r])
    }

    #[test]
    fn test_ru() {
        assert_result_eq("ru(time(), 2000)", &[50.0, 40.0, 30.0, 20.0, 10.0, 0.0]);

        assert_result_eq(
            "ru(time() offset 100s, 2000)",
            &[60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
        );

        assert_result_eq(
            "ru(time() offset 0.5i, 2000)",
            &[60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
        );

        assert_result_eq(
            "ru(time() offset 1.5i, 2000)",
            &[70.0, 60.0, 50.0, 40.0, 30.0, 20.0],
        );

        assert_result_eq("ru(time(), 1600)", &[37.5, 25.0, 12.5, 0.0, 0.0, 0.0]);

        assert_result_eq(
            "ru(1500-time(), 1000)",
            &[50.0, 70.0, 90.0, 100.0, 100.0, 100.0],
        );
    }

    #[test]
    fn mode_over_time() {
        let q = "mode_over_time(round(time()/500)[100s:1s])";
        assert_result_eq(q, &[2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn rate_over_sum() {
        let q = "rate_over_sum(round(time()/500)[100s:5s])";
        assert_result_eq(q, &[0.4, 0.4, 0.6, 0.6, 0.71, 0.8]);
    }

    #[test]
    fn zscore_over_time_rand() {
        let q = "round(zscore_over_time(rand(0)[100s:10s]), 0.01)";
        assert_result_eq(q, &[-1.12, 0.5, 1.05, 1.88, -1.16, 0.79]);
    }

    #[test]
    fn zscore_over_time_const() {
        assert_result_eq(
            "zscore_over_time(1[100s:10s])",
            &[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn integrate() {
        assert_result_eq("integrate(1)", &[200.0, 200.0, 200.0, 200.0, 200.0, 200.0]);
        assert_result_eq(
            "integrate(time()/1e3)",
            &[160.0, 200.0, 240.0, 280.0, 320.0, 360.0],
        );
    }

    #[test]
    fn rate() {
        // test_query("rate({})", vec![]);

        assert_result_eq("rate(2000-time())", &[5.5, 4.5, 3.5, 2.5, 1.5, 0.5]);

        assert_result_eq("rate((2000-time())[100s])", &[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);

        assert_result_eq(
            "rate((2000-time())[100s:100s])",
            &[0_f64, 0.0, 6.0, 4.0, 2.0, 0.0],
        );

        let q = "rate((2000-time())[100s:100s] offset 100s)";
        assert_result_eq(q, &[0.0, 0.0, 7.0, 5.0, 3.0, 1.0]);

        let q = "rate((2000-time())[100s:100s] offset 100s)[:] offset 100s";
        assert_result_eq(q, &[0.0, 0.0, 0.0, 7.0, 5.0, 3.0]);

        test_query("rate({}[:5s])", vec![]);
    }

    #[test]
    fn rate_time() {
        let q = r#"rate(label_set(alias(time(), "foo"), "x", "y"))"#;
        let mut r = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.set("x", "y");
        test_query(q, vec![r]);
    }

    #[test]
    fn rate_time_keep_metric_names() {
        let q = r#"rate(label_set(alias(time(), "foo"), "x", "y")) keep_metric_names"#;
        let mut r = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.set_metric_group("foo");
        r.metric.set("x", "y");
        test_query(q, vec![r]);
    }

    #[test]
    fn sum_rate_time_by__name__keep_metric_names() {
        let q = r#"sum(rate(label_set(alias(time(), "foo"), "x", "y")) keep_metric_names) by (__name__)"#;
        let mut r = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.set_metric_group("foo");
        test_query(q, vec![r]);
    }

    #[test]
    fn increase_pure() {
        assert_result_eq(
            "increase_pure(time())",
            &[200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
        );
    }

    #[test]
    fn increase() {
        assert_result_eq(
            "increase(time())",
            &[200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
        );

        assert_result_eq(
            "increase(2000-time())",
            &[1000_f64, 800.0, 600.0, 400.0, 200.0, 0.0],
        );
    }

    #[test]
    fn increase_prometheus() {
        let q = "increase_prometheus(time())";
        test_query(q, vec![]);

        assert_result_eq(
            "increase_prometheus(time()[201s])",
            &[200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
        );
    }

    #[test]
    fn running_max() {
        assert_result_eq("running_max(1)", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_result_eq(
            "running_max(abs(1300-time()))",
            &[300.0, 300.0, 300.0, 300.0, 500.0, 700.0],
        );
        assert_result_eq(
            "running_max(abs(1300-time()) > 300 < 700)",
            &[f64::NAN, f64::NAN, f64::NAN, f64::NAN, 500.0, 500.0],
        );
    }

    #[test]
    fn running_min() {
        assert_result_eq(
            "running_min(abs(1500-time()))",
            &[500.0, 300.0, 100.0, 100.0, 100.0, 100.0],
        );
    }

    #[test]
    fn running_min_1() {
        assert_result_eq(
            "running_min(abs(1500-time()) < 400 > 100)",
            &[f64::NAN, 300.0, 300.0, 300.0, 300.0, 300.0],
        );
    }

    #[test]
    fn running_sum_1() {
        assert_result_eq("running_sum(1)", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn running_sum_time() {
        assert_result_eq("running_sum(time()/1e3)", &[1.0, 2.2, 3.6, 5.2, 7.0, 9.0]);
    }

    #[test]
    fn running_sum_time_ex() {
        assert_result_eq("running_sum(time()/1e3 > 1.2 < 1.8)",
                         &[f64::NAN, f64::NAN, 1.4, 3.3, 3.3, 3.3]);
    }

    #[test]
    fn running_avg_time() {
        assert_result_eq(
            "running_avg(time())",
            &[1000_f64, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
        );
        assert_result_eq(
            "running_avg(time() > 1200 < 1800)",
            &[f64::NAN, f64::NAN, 1400.0, 1500.0, 1500.0, 1500.0],
        );
    }

    #[test]
    fn smooth_exponential() {
        assert_result_eq(
            "smooth_exponential(time(), 1)",
            &[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
        assert_result_eq(
            "smooth_exponential(time(), 0)",
            &[1000_f64, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        );
        assert_result_eq(
            "smooth_exponential(time(), 0.5)",
            &[1000_f64, 1100.0, 1250.0, 1425.0, 1612.5, 1806.25],
        );
    }

    #[test]
    fn remove_resets() {
        assert_result_eq(
            "remove_resets(abs(1500-time()))",
            &[500.0, 800.0, 900.0, 900.0, 1100.0, 1300.0],
        );
    }

    #[test]
    fn remove_resets_sum() {
        let q = r#"remove_resets(sum(
        alias(time(), "full"),
        alias(time()/5 < 300, "partial"),
        ))"#;
        assert_result_eq(q, &[1200.0, 1440.0, 1680.0, 1680.0, 1880.0, 2080.0]);
    }

    #[test]
    fn range_avg() {
        assert_result_eq(
            "range_avg(time())",
            &[1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0],
        );
    }

    #[test]
    fn range_min() {
        assert_result_eq(
            "range_min(time())",
            &[1000_f64, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        );
        assert_result_eq(
            "range_min(time() > 1200 < 1800)",
            &[1400_f64, 1400.0, 1400.0, 1400.0, 1400.0, 1400.0],
        );
    }

    #[test]
    fn range_normalize() {
        let q = r#"range_normalize(time(),alias(-time(),"negative"))"#;
        let r1 = make_result(&[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
        let mut r2 = make_result(&[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]);
        r2.metric.measurement = "negative".to_string();
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn range_normalize_1() {
        let q = r#"range_normalize(time() > 1200 < 1800,alias(-(time() > 1200 < 2000), "negative"))"#;
        let r1 = make_result(&[f64::NAN, f64::NAN, 0.0, 1.0, f64::NAN, f64::NAN]);
        let mut r2 = make_result(&[f64::NAN, f64::NAN, 1.0, 0.5, 0.0, f64::NAN]);
        r2.metric.measurement = "negative".to_string();
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn range_first() {
        assert_result_eq(
            "range_first(time())",
            &[1000_f64, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        );
        assert_result_eq(
            "range_first(time() > 1200 < 1800)",
            &[1400.0, 1400.0, 1400.0, 1400.0, 1400.0, 1400.0],
        );
    }

    #[test]
    fn range_mad() {
        assert_result_eq(
            "range_mad(time())",
            &[300.0, 300.0, 300.0, 300.0, 300.0, 300.0],
        );
        assert_result_eq(
            "range_mad(time() > 1200 < 1800)",
            &[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        );
    }

    #[test]
    fn range_max() {
        assert_result_eq(
            "range_max(time())",
            &[2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
        );
        assert_result_eq(
            "range_max(time() > 1200 < 1800)",
            &[1600.0, 1600.0, 1600.0, 1600.0, 1600.0, 1600.0],
        );
    }

    #[test]
    fn range_sum() {
        assert_result_eq(
            "range_sum(time())",
            &[9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0],
        );
        assert_result_eq(
            "range_sum(time() > 1200 < 1800)",
            &[3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
        );
    }

    #[test]
    fn range_last() {
        assert_result_eq(
            "range_last(time())",
            &[2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
        );
        assert_result_eq(
            "range_last(time() > 1200 < 1800)",
            &[1600.0, 1600.0, 1600.0, 1600.0, 1600.0, 1600.0],
        );
    }

    #[test]
    fn range_linear_regression_time() {
        assert_result_eq("range_linear_regression(time())",&[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
    }

    #[test]
    fn range_linear_regression_negative_time() {
        assert_result_eq("range_linear_regression(-time())", &[-1000.0, -1200.0, -1400.0, -1600.0, -1800.0, -2000.0]);
    }

    #[test]
    fn range_linear_regression_custom() {
        assert_result_eq(
            "range_linear_regression(time() > 1200 < 1800)",
            &[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        );
    }

    #[test]
    fn deriv() {
        assert_result_eq("deriv(-time())", &[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        assert_result_eq("deriv(1000)", &[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_result_eq("deriv(2*time())", &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_delta() {
        assert_result_eq("delta(time())", &[200.0, 200.0, 200.0, 200.0, 200.0, 200.0]);
        assert_result_eq("delta(delta(2*time()))", &[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_result_eq(
            "delta(-time())",
            &[-200.0, -200.0, -200.0, -200.0, -200.0, -200.0],
        );
        assert_result_eq("delta(1)", &[0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn delta_prometheus() {
        let q = "delta_prometheus(time())";
        test_query(q, vec![]);

        assert_result_eq(
            "delta_prometheus(time()[201s])",
            &[200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
        );
    }

    #[test]
    fn hoeffding_bound_lower() {
        let q = "hoeffding_bound_lower(0.9, rand(0)[:10s])";
        assert_result_eq(
            q,
            &[
                0.30156740146540006,
                0.23471994636231044,
                0.18209903174990535,
                0.3637583992631742,
                0.28199209507225204,
                0.2956205035589421,
            ],
        );
    }

    #[test]
    fn hoeffding_bound_upper() {
        let q = r#"hoeffding_bound_upper(0.9, alias(rand(0), "foobar")[:10s])"#;
        let mut r = make_result(&[
            0.7514246534914918,
            0.6961710293162271,
            0.6460729683292205,
            0.784636439070813,
            0.7173342952485633,
            0.6585571598032428,
        ]);
        r.metric.set_metric_group("foobar");
        test_query(q, vec![r])
    }

    #[test]
    fn aggr_over_time_single_func() {
        let q = r#"round(aggr_over_time(rand(0)[:10s], "increase"), 0.01)"#;
        let mut r1 = make_result(&[6.76, 4.59, 3.78, 5.86, 5.93, 6.45]);
        r1.metric.set("rollup", "increase");
        test_query(q, vec![r1]);
    }

    #[test]
    fn aggr_over_time_multi_func() {
        let q = r#"sort(aggr_over_time(round(rand(0),0.1)[:10s], "min_over_time", "median_over_time", "max_over_time"))"#;
        let mut r1 = make_result(&[0.0, 0.0, 0.0, 0.0, 0.1, 0.1]);
        r1.metric.set("rollup", "min_over_time");
        let mut r2 = make_result(&[1.0, 1.0, 1.0, 0.9, 1.0, 0.9]);
        r2.metric.set("rollup", "median_over_time");
        let mut r3 = make_result(&[20_f64, 20.0, 20.0, 20.0, 20.0, 20.0]);
        r3.metric.set("rollup", "max_over_time");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3];
        test_query(q, result_expected)
    }

    #[test]
    fn test_avg_aggr_over_time() {
        let q = r#"avg(aggr_over_time(time()[:10s], "min_over_time", "max_over_time"))"#;
        assert_result_eq(q, &[905.0, 1105.0, 1305.0, 1505.0, 1705.0, 1905.0]);
    }

    #[test]
    fn test_avg_aggr_over_time_by_rollup() {
        // avg(aggr_over_time(multi-func)) by (rollup)
        let q = r#"sort(avg(aggr_over_time(time()[:10s], "min_over_time", "max_over_time")) by (rollup))"#;
        let mut r1 = make_result(&[810_f64, 1010.0, 1210.0, 1410.0, 1610.0, 1810.0]);
        r1.metric.set("rollup", "min_over_time");
        let mut r2 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r2.metric.set("rollup", "max_over_time");
        test_query(q, vec![r1, r2]);
    }


    #[test]
    fn rollup_candlestick() {
        let q = r#"sort(rollup_candlestick(alias(round(rand(0),0.01),"foobar")[:10s]))"#;
        let mut r1 = make_result(&[0.02, 0.02, 0.03, 0.0, 0.03, 0.02]);
        r1.metric.set_metric_group("foobar");
        r1.metric.set("rollup", "low");
        let mut r2 = make_result(&[0.9, 0.32, 0.82, 0.13, 0.28, 0.86]);
        r2.metric.set_metric_group("foobar");
        r2.metric.set("rollup", "open");
        let mut r3 = make_result(&[0.1, 0.04, 0.49, 0.46, 0.57, 0.92]);
        r3.metric.set_metric_group("foobar");
        r3.metric.set("rollup", "close");
        let mut r4 = make_result(&[0.9, 0.94, 0.97, 0.93, 0.98, 0.92]);
        r4.metric.set_metric_group("foobar");
        r4.metric.set("rollup", "high");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4];
        test_query(q, result_expected)
    }

    // revise
    #[test]
    fn rollup_candlestick_high() {
        let q = r#"rollup_candlestick(alias(round(rand(0),0.01),"foobar")[:10s], "high")"#;
        let mut r = make_result(&[0.99, 0.98, 0.98, 0.92, 0.98, 0.99]);
        r.metric.set_metric_group("foobar");
        r.metric.set("rollup", "high");
        let result_expected: Vec<QueryResult> = vec![r];
        test_query(q, result_expected);
    }

    #[test]
    fn rollup_increase() {
        let q = "sort(rollup_increase(time()))";
        let mut r1 = make_result(&[200_f64, 200.0, 200.0, 200.0, 200.0, 200.0]);
        r1.metric.set("rollup", "min");
        let mut r2 = make_result(&[200_f64, 200.0, 200.0, 200.0, 200.0, 200.0]);
        r2.metric.set("rollup", "max");
        let mut r3 = make_result(&[200_f64, 200.0, 200.0, 200.0, 200.0, 200.0]);
        r3.metric.set("rollup", "avg");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn rollup_scrape_interval() {
        let q = r#"sort_by_label(rollup_scrape_interval(1[5m:10s]), "rollup")"#;
        let mut r1 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r1.metric.set("rollup", "avg");
        let mut r2 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r2.metric.set("rollup", "max");
        let mut r3 = make_result(&[10_f64, 10.0, 10.0, 10.0, 10.0, 10.0]);
        r3.metric.set("rollup", "min");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn rollup() {
        let q = "sort(rollup(time()[:50s]))";
        let mut r1 = make_result(&[850_f64, 1050.0, 1250.0, 1450.0, 1650.0, 1850.0]);
        r1.metric.set("rollup", "min");
        let mut r2 = make_result(&[925_f64, 1125.0, 1325.0, 1525.0, 1725.0, 1925.0]);
        r2.metric.set("rollup", "avg");
        let mut r3 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r3.metric.set("rollup", "max");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn rollup_rate() {
        let q = "rollup_rate((2200-time())[600s])";
        let mut r1 = make_result(&[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        r1.metric.set("rollup", "avg");

        let mut r2 = make_result(&[7.0, 6.0, 5.0, 4.0, 3.0, 2.0]);
        r2.metric.set("rollup", "max");

        let mut r3 = make_result(&[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
        r3.metric.set("rollup", "min");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn rollup_rate_max() {
        let q = r#"rollup_rate((2200-time())[600s], "max")"#;
        let mut r = make_result(&[7.0, 6.0, 5.0, 4.0, 3.0, 2.0]);
        r.metric.set("rollup", "max");
        test_query(q, vec![r]);
    }

    #[test]
    fn rollup_rate_avg() {
        let q = r#"rollup_rate((2200-time())[600s], "avg")"#;
        let mut r = make_result(&[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        r.metric.set("rollup", "avg");
        test_query(q, vec![r]);
    }

    #[test]
    fn rollup_deriv() {
        let q = "sort(rollup_deriv(time()[100s:50s]))";
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set("rollup", "min");
        let mut r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r2.metric.set("rollup", "max");
        let mut r3 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r3.metric.set("rollup", "avg");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn rollup_deriv_max() {
        let q = r#"sort(rollup_deriv(time()[100s:50s], "max"))"#;
        let mut r = make_result(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.set("rollup", "max");
        test_query(q, vec![r]);
    }

    #[test]
    fn empty_selector() {
        let q = "{}";
        test_query(q, vec![]);
    }

    #[test]
    fn start() {
        let q = "time() - start()";
        assert_result_eq(q, &[0_f64, 200.0, 400.0, 600.0, 800.0, 1000.0]);
    }

    #[test]
    fn end() {
        assert_result_eq(
            "end() - time()",
            &[1000_f64, 800.0, 600.0, 400.0, 200.0, 0.0],
        );
    }

    #[test]
    fn step() {
        assert_result_eq("time() / step()", &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn lag() {
        assert_result_eq("lag(time()[60s:17s])", &[14.0, 10.0, 6.0, 2.0, 15.0, 11.0]);
    }

    #[test]
    fn parens_expr() {
        test_query("()", vec![]);

        assert_result_eq("(1)", &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // identical_labels
        let q = r#"(label_set(1, "foo", "bar"), label_set(2, "foo", "bar"))"#;
        let mut r = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn parens_expr_identical_labels_with_names() {
        let q = r#"(label_set(1, "foo", "bar", "__name__", "xx"), label_set(2, "__name__", "xx", "foo", "bar"))"#;
        let mut r = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.set_metric_group("xx");
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn union() {
        let q = "union(1)";
        assert_result_eq(q, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn union_identical_labels() {
        let q = r#"union(label_set(1, "foo", "bar"), label_set(2, "foo", "bar"))"#;
        let mut r = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.set("foo", "bar");
        test_query(q, vec![r])
    }

    #[test]
    fn union_identical_labels_with_names() {
        let q = r#"union(label_set(1, "foo", "bar", "__name__", "xx"), label_set(2, "__name__", "xx", "foo", "bar"))"#;
        let mut r = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r.metric.set_metric_group("xx");
        r.metric.set("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn union_more_than_two() {
        let q = r#"union(
    label_set(1, "foo", "bar", "__name__", "xx"),
    label_set(2, "__name__", "yy", "foo", "bar"),
    label_set(time(), "qwe", "123") or label_set(3, "__name__", "rt"))"#;
        let mut r1 = make_result(&[1000_f64, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]);
        r1.metric.set("qwe", "123");
        let mut r2 = make_result(&[3_f64, 3.0, 3.0, 3.0, 3.0, 3.0]);
        r2.metric.set_metric_group("rt");
        let mut r3 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r3.metric.set_metric_group("xx");
        r3.metric.set("foo", "bar");
        let mut r4 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r4.metric.set_metric_group("yy");
        r4.metric.set("foo", "bar");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4];
        test_query(q, result_expected)
    }

    #[test]
    fn union_identical_labels_different_names() {
        let q = r#"union(label_set(1, "foo", "bar", "__name__", "xx"), label_set(2, "__name__", "yy", "foo", "bar"))"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set_metric_group("xx");
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r2.metric.set_metric_group("yy");
        r2.metric.set("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn parens_expr_identical_labels_different_names() {
        let q = r#"(label_set(1, "foo", "bar", "__name__", "xx"), label_set(2, "__name__", "yy", "foo", "bar"))"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set_metric_group("xx");
        r1.metric.set("foo", "bar");
        let mut r2 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r2.metric.set_metric_group("yy");
        r2.metric.set("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn nested_parens_expr() {
        let q = r#"((
        alias(1, "x1"),
        ),(
        alias(2, "x2"),
        alias(3, "x3"),
        ))"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set_metric_group("x1");
        let mut r2 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r2.metric.set_metric_group("x2");
        let mut r3 = make_result(&[3_f64, 3.0, 3.0, 3.0, 3.0, 3.0]);
        r3.metric.set_metric_group("x3");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn count_values_big_numbers() {
        let q = r#"sort_by_label(
        count_values("xxx", (alias(772424014, "first"), alias(772424230, "second"))),
        "xxx"
        )"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set("xxx", "772424014");

        let mut r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r2.metric.set("xxx", "772424230");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn count_values() {
        let q = r#"count_values("xxx", label_set(10, "foo", "bar") or label_set(time()/100, "foo", "bar", "baz", "xx"))"#;
        let mut r1 = make_result(&[2_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set("xxx", "10");

        let mut r2 = make_result(&[NAN, 1.0, NAN, NAN, NAN, NAN]);
        r2.metric.set("xxx", "12");

        let mut r3 = make_result(&[NAN, NAN, 1.0, NAN, NAN, NAN]);
        r3.metric.set("xxx", "14");

        let mut r4 = make_result(&[NAN, NAN, NAN, 1.0, NAN, NAN]);
        r4.metric.set("xxx", "16");
        let mut r5 = make_result(&[NAN, NAN, NAN, NAN, 1.0, NAN]);
        r5.metric.set("xxx", "18");

        let mut r6 = make_result(&[NAN, NAN, NAN, NAN, NAN, 1.0]);
        r6.metric.set("xxx", "20");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5, r6];
        test_query(q, result_expected)
    }

    #[test]
    fn count_values_by_xxx() {
        let q = r#"count_values("xxx", label_set(10, "foo", "bar", "xxx", "aaa") or label_set(floor(time()/600), "foo", "bar", "baz", "xx")) by (xxx)"#;
        let mut r1 = make_result(&[1_f64, NAN, NAN, NAN, NAN, NAN]);
        r1.metric.set("xxx", "1");

        let mut r2 = make_result(&[NAN, 1.0, 1.0, 1.0, NAN, NAN]);
        r2.metric.set("xxx", "2");

        let mut r3 = make_result(&[NAN, NAN, NAN, NAN, 1.0, 1.0]);
        r3.metric.set("xxx", "3");

        let mut r4 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r4.metric.set("xxx", "10");

        // expected sorted output for strings 1, 10, 2, 3
        let result_expected: Vec<QueryResult> = vec![r1, r4, r2, r3];
        test_query(q, result_expected);
    }

    #[test]
    fn count_values_without_baz() {
        let q = r#"count_values("xxx", label_set(floor(time()/600), "foo", "bar")) without (baz)"#;
        let mut r1 = make_result(&[1_f64, NAN, NAN, NAN, NAN, NAN]);
        r1.metric.set("foo", "bar");
        r1.metric.set("xxx", "1");

        let mut r2 = make_result(&[NAN, 1.0, 1.0, 1.0, NAN, NAN]);
        r2.metric.set("foo", "bar");
        r2.metric.set("xxx", "2");

        let mut r3 = make_result(&[NAN, NAN, NAN, NAN, 1.0, 1.0]);
        r3.metric.set("foo", "bar");
        r3.metric.set("xxx", "3");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn result_sorting() {
        let q = r#"(
        label_set(1, "instance", "localhost:1001", "type", "free"),
        label_set(1, "instance", "localhost:1001", "type", "buffers"),
        label_set(1, "instance", "localhost:1000", "type", "buffers"),
        label_set(1, "instance", "localhost:1000", "type", "free")
        )"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        test_add_labels(
            &mut r1.metric,
            &["instance", "localhost:1000", "type", "buffers"],
        );
        let mut r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        test_add_labels(
            &mut r2.metric,
            &["instance", "localhost:1000", "type", "free"],
        );
        let mut r3 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        test_add_labels(
            &mut r3.metric,
            &["instance", "localhost:1001", "type", "buffers"],
        );
        let mut r4 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        test_add_labels(
            &mut r4.metric,
            &["instance", "localhost:1001", "type", "free"],
        );
        test_query(q, vec![r1, r2, r3, r4]);
    }

    #[test]
    fn sort_by_label_numeric_multiple_labels_only_string() {
        let q = r#"sort_by_label_numeric((
        label_set(1, "x", "b", "y", "aa"),
        label_set(2, "x", "a", "y", "aa"),
        ), "y", "x")"#;
        let mut r1 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r1.metric.set("x", "a");
        r1.metric.set("y", "aa");

        let mut r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r2.metric.set("x", "b");
        r2.metric.set("y", "aa");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label_numeric_multiple_labels_numbers_special_chars() {
        let q = r#"sort_by_label_numeric((
        label_set(1, "x", "1:0:2", "y", "1:0:1"),
        label_set(2, "x", "1:0:15", "y", "1:0:1"),
        ), "x", "y")"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set("x", "1:0:2");
        r1.metric.set("y", "1:0:1");

        let mut r2 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r2.metric.set("x", "1:0:15");
        r2.metric.set("y", "1:0:1");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label_numeric_desc_multiple_labels_numbers_special_chars() {
        let q = r#"sort_by_label_numeric_desc((
        label_set(1, "x", "1:0:2", "y", "1:0:1"),
        label_set(2, "x", "1:0:15", "y", "1:0:1"),
        ), "x", "y")"#;
        let mut r1 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r1.metric.set("x", "1:0:15");
        r1.metric.set("y", "1:0:1");

        let mut r2 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r2.metric.set("x", "1:0:2");
        r2.metric.set("y", "1:0:1");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label_numeric_alias_numbers_with_special_chars() {
        let q = r#"sort_by_label_numeric((
        label_set(4, "a", "DS50:1/0/15"),
        label_set(1, "a", "DS50:1/0/0"),
        label_set(2, "a", "DS50:1/0/1"),
        label_set(3, "a", "DS50:1/0/2"),
        ), "a")"#;
        let mut r1 = make_result(&[1_f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        r1.metric.set("a", "DS50:1/0/0");

        let mut r2 = make_result(&[2_f64, 2.0, 2.0, 2.0, 2.0, 2.0]);
        r2.metric.set("a", "DS50:1/0/1");

        let mut r3 = make_result(&[3_f64, 3.0, 3.0, 3.0, 3.0, 3.0]);
        r3.metric.set("a", "DS50:1/0/2");

        let mut r4 = make_result(&[4_f64, 4.0, 4.0, 4.0, 4.0, 4.0]);
        r4.metric.set("a", "DS50:1/0/15");

        test_query(q, vec![r1, r2, r3, r4])
    }

    #[test]
    fn test_exec_error() {
        fn f(q: &str) {
            let mut ec = EvalConfig::new(1000, 2000, Duration::from_millis(100));
            ec.max_points_per_series = 100000;
            ec.max_series = 1000;
            let context = Arc::new(Context::default());

            (0..4).for_each(|_| {
                let rv = exec(&context, &mut ec, q, false);
                assert_eq!(rv.is_err(), true, "expecting exec error: {}", q);
                let rv = exec(&context, &mut ec, q, true);
                assert_eq!(rv.is_err(), true, "expecting exec error: {}", q);
            });
        }

        f("pi(123)");

        // Empty expr
        f("");
        f("    ");

        // Invalid expr
        f("1-");

        // Non-existing func
        f("nonexisting()");

        // Invalid number of args
        f("range_quantile()");
        f("range_quantile(1, 2, 3)");
        f("range_median()");
        f("abs()");
        f("abs(1,2)");
        f("absent(1, 2)");
        f("clamp()");
        f("clamp_max()");
        f("clamp_min(1,2,3)");
        f("hour(1,2)");
        f("label_join()");
        f("label_replace(1)");
        f("label_transform(1)");
        f("label_set()");
        f(r#"label_set(1, "foo")"#);
        f("label_map()");
        f("label_map(1)");
        f("label_del()");
        f("label_keep()");
        f("label_match()");
        f("label_mismatch()");
        f("label_graphite_group()");
        f("round()");
        f("round(1,2,3)");
        f("sgn()");
        f("scalar()");
        f("sort(1,2)");
        f("sort_desc()");
        f("sort_by_label()");
        f("sort_by_label_desc()");
        f("sort_by_label_numeric()");
        f("sort_by_label_numeric_desc()");
        f("timestamp()");
        f("timestamp_with_name()");
        f("vector()");
        f("histogram_quantile()");
        f("histogram_quantiles()");
        f("sum()");
        f("count_values()");
        f("quantile()");
        f("any()");
        f("group()");
        f("topk()");
        f("topk_min()");
        f("topk_max()");
        f("topk_avg()");
        f("topk_median()");
        f("topk_last()");
        f("limitk()");
        f("bottomk()");
        f("bottomk_min()");
        f("bottomk_max()");
        f("bottomk_avg()");
        f("bottomk_median()");
        f("bottomk_last()");
        f("time(123)");
        f("start(1)");
        f("end(1)");
        f("step(1)");
        f("running_sum(1, 2)");
        f("range_sum(1, 2)");
        f("range_trim_outliers()");
        f("range_trim_spikes()");
        f("range_trim_zscore()");
        f("range_zscore()");
        f("range_first(1,  2)");
        f("range_last(1, 2)");
        f("smooth_exponential()");
        f("smooth_exponential(1)");
        f("remove_resets()");
        f("sin()");
        f("sinh()");
        f("cos()");
        f("cosh()");
        f("asin()");
        f("asinh()");
        f("acos()");
        f("acosh()");
        f("rand(123, 456)");
        f("rand_normal(123, 456)");
        f("rand_exponential(122, 456)");
        f("pi(123)");
        f("now(123)");
        f("label_copy()");
        f("label_move()");
        f("median_over_time()");
        f("median()");
        f("keep_last_value()");
        f("keep_next_value()");
        f("interpolate()");
        f("distinct_over_time()");
        f("distinct()");
        f("alias()");
        f("alias(1)");
        f(r#"alias(1, "foo", "bar")"#);
        f("lifetime()");
        f("lag()");
        f("aggr_over_time()");
        f("aggr_over_time(foo)");
        f(r#"aggr_over_time("foo", bar, 1)"#);
        f("sum(aggr_over_time())");
        f("sum(aggr_over_time(foo))");
        f(r#"count(aggr_over_time("foo", bar, 1))"#);
        f("hoeffding_bound_lower()");
        f("hoeffding_bound_lower(1)");
        f("hoeffding_bound_lower(0.99, foo, 1)");
        f("hoeffding_bound_upper()");
        f("hoeffding_bound_upper(1)");
        f("hoeffding_bound_upper(0.99, foo, 1)");
        f("mad()");
        f("outliers_mad()");
        f("outliers_mad(1)");
        f("outliersk()");
        f("outliersk(1)");
        f("mode_over_time()");
        f("rate_over_sum()");
        f("zscore_over_time()");
        f("mode()");
        f("share()");
        f("zscore()");
        f("prometheus_buckets()");
        f("buckets_limit()");
        f("buckets_limit(1)");
        f("duration_over_time()");
        f("share_le_over_time()");
        f("share_gt_over_time()");
        f("count_le_over_time()");
        f("count_gt_over_time()");
        f("count_eq_over_time()");
        f("count_ne_over_time()");
        f("timezone_offset()");
        f("bitmap_and()");
        f("bitmap_or()");
        f("bitmap_xor()");
        f("quantiles()");
        f("limit_offset()");
        f("increase()");
        f("increase_prometheus()");
        f("changes()");
        f("changes_prometheus()");
        f("delta()");
        f("delta_prometheus()");
        f("rollup_candlestick()");
        f("rollup()");

        // Invalid argument type
        f("median_over_time({}, 2)");
        f(r#"smooth_exponential(1, 1 or label_set(2, "x", "y"))"#);
        f("count_values(1, 2)");
        f(r#"count_values(1 or label_set(2, "xx", "yy"), 2)"#);
        f(r#"quantile(1 or label_set(2, "xx", "foo"), 1)"#);
        f(r#"clamp_max(1, 1 or label_set(2, "xx", "foo"))"#);
        f(r#"clamp_min(1, 1 or label_set(2, "xx", "foo"))"#);
        f(r#"topk(label_set(2, "xx", "foo") or 1, 12)"#);
        f(r#"topk_avg(label_set(2, "xx", "foo") or 1, 12)"#);
        f(r#"limitk(label_set(2, "xx", "foo") or 1, 12)"#);
        f(r#"limit_offet((alias(1,"foo"),alias(2,"bar")), 2, 10)"#);
        f(r#"limit_offet(1, (alias(1,"foo"),alias(2,"bar")), 10)"#);
        f(r#"round(1, 1 or label_set(2, "xx", "foo"))"#);
        f(r#"histogram_quantile(1 or label_set(2, "xx", "foo"), 1)"#);
        f(r#"histogram_quantiles("foo", 1 or label_set(2, "xxx", "foo"), 2)"#);
        f("sort_by_label_numeric(1, 2)");
        f("label_set(1, 2, 3)");
        f(r#"label_set(1, "foo", (label_set(1, "foo", bar") or label_set(2, "xxx", "yy")))"#);
        f(r#"label_set(1, "foo", 3)"#);
        f("label_del(1, 2)");
        f("label_copy(1, 2)");
        f("label_move(1, 2, 3)");
        f(r#"label_move(1, "foo", 3)"#);
        f("label_keep(1, 2)");
        f("label_join(1, 2, 3)");
        f(r#"label_join(1, "foo", 2)"#);
        f(r#"label_join(1, "foo", "bar", 2)"#);
        f("label_replace(1, 2, 3, 4, 5)");
        f(r#"label_replace(1, "foo", 3, 4, 5)"#);
        f(r#"label_replace(1, "foo", "bar", 4, 5)"#);
        f(r#"label_replace(1, "foo", "bar", "baz", 5)"#);
        f(r#"label_replace(1, "foo", "bar", "baz", "invalid(regexp")"#);
        f("label_transform(1, 2, 3, 4)");
        f(r#"label_transform(1, "foo", 3, 4)"#);
        f(r#"label_transform(1, "foo", "bar", 4)"#);
        f(r#"label_transform(1, "foo", "invalid(regexp", "baz""#);
        f("label_match(1, 2, 3)");
        f("label_mismatch(1, 2, 3)");
        f("label_uppercase()");
        f("label_lowercase()");
        f("alias(1, 2)");
        f("aggr_over_time(1, 2)");
        f(r#"aggr_over_time(("foo", "bar"), 3)"#);
        f(r#"outliersk((label_set(1, "foo", "bar"), label_set(2, "x", "y")), 123)"#);

        // Duplicate timeseries
        f(
            r#"(label_set(1, "foo", "bar") or label_set(2, "foo", "baz"))
+ on(xx)
(label_set(1, "foo", "bar") or label_set(2, "foo", "baz"))"#,
        );

        // Invalid binary op groupings
        f(r#"1 + group_left() (label_set(1, "foo", bar"), label_set(2, "foo", "baz"))"#);
        f(r#"1 + on() group_left() (label_set(1, "foo", bar"), label_set(2, "foo", "baz"))"#);
        f(r#"1 + on(a) group_left(b) (label_set(1, "foo", bar"), label_set(2, "foo", "baz"))"#);
        f(
            r#"label_set(1, "foo", "bar") + on(foo) group_left() (label_set(1, "foo", "bar", "a", "b"), label_set(1, "foo", "bar", "a", "c"))"#,
        );
        f(r#"(label_set(1, "foo", bar"), label_set(2, "foo", "baz")) + group_right 1"#);
        f(r#"(label_set(1, "foo", bar"), label_set(2, "foo", "baz")) + on() group_right 1"#);
        f(r#"(label_set(1, "foo", bar"), label_set(2, "foo", "baz")) + on(a) group_right(b,c) 1"#);
        f(r#"(label_set(1, "foo", bar"), label_set(2, "foo", "baz")) + on() 1"#);
        f(
            r#"(label_set(1, "foo", "bar", "a", "b"), label_set(1, "foo", "bar", "a", "c")) + on(foo) group_right() label_set(1, "foo", "bar")"#,
        );
        f(r#"1 + on() (label_set(1, "foo", bar"), label_set(2, "foo", "baz"))"#);

        // duplicate metrics after binary op
        f(r#"(
label_set(time(), "__name__", "foo", "a", "x"),
label_set(time()+200, "__name__", "bar", "a", "x"),
) > bool 1300"#);
        f(r#"(
label_set(time(), "__name__", "foo", "a", "x"),
label_set(time()+200, "__name__", "bar", "a", "x"),
) + 10"#);

        // Invalid aggregates
        f("sum(1) foo (bar)");
        f("sum foo () (bar)");
        f("sum(foo) by (1)");
        f(r#"count(foo) without ("bar")"#);

        // With expressions
        f("ttf()");
        f("ttf(1, 2)");

        // Invalid rollup tags
        f(r#"rollup_rate(time()[5m], "")"#);
        f(r#"rollup_rate(time()[5m], "foo")"#);
        f(r#"rollup_rate(time()[5m], "foo", "bar")"#);
        f(r#"rollup_candlestick(time(), "foo")"#);
    }

    fn test_add_labels(mn: &mut MetricName, labels: &[&str]) {
        assert_eq!(
            labels.len() % 2,
            0,
            "uneven number of labels passed: {}",
            labels.join(",")
        );
        for i in (0..labels.len()).step_by(2) {
            mn.set(labels[i], labels[i + 1])
        }
    }

    #[test]
    fn test_metricsql_is_likely_invalid_false() {
        fn f(q: &str) {
            let expr = parse(q).unwrap();
            assert!(
                !is_likely_invalid(&expr),
                "unexpected result for is_likely_invalid({}); got true; want false",
                q
            );
        }

        f("http_total[5m]");
        f("sum(http_total)");
        f("sum(foo, bar)");
        f("absent(http_total)");
        f("rate(http_total[1m])");
        f("avg_over_time(up[1m])");
        f("sum(rate(http_total[1m]))");
        f("sum(sum(http_total))");

        f("sum(sum_over_time(http_total[1m] )) by (instance)");
        f("sum(up{cluster='a'}[1m] or up{cluster='b'}[1m])");
        f("(avg_over_time(alarm_test1[1m]) - avg_over_time(alarm_test1[1m] offset 5m)) > 0.1");
        f("http_total[1m] offset 1m");
        f("sum(http_total offset 1m)");

        // subquery
        f("rate(http_total[5m])[5m:1m]");
        f("rate(sum(http_total)[5m:1m])");
        f("rate(rate(http_total[5m])[5m:1m])");
        f("sum(rate(http_total[1m]))");
        f("sum(rate(sum(http_total)[5m:1m]))");
        f("rate(sum(rate(http_total[5m]))[5m:1m])");
        f("rate(sum(sum(http_total))[5m:1m])");
        f("rate(sum(rate(http_total[5m]))[5m:1m])");
        f("rate(sum(sum(http_total))[5m:1m])");
        f("avg_over_time(rate(http_total[5m])[5m:1m])");
        f("delta(avg_over_time(up[1m])[5m:1m]) > 0.1");
        f("avg_over_time(avg by (site) (metric)[2m:1m])");

        f("sum(http_total)[5m:1m] offset 1m");
        f("round(sum(sum_over_time(http_total[1m])) by (instance))[5m:1m] offset 1m");

        f("rate(sum(http_total)[5m:1m]) - rate(sum(http_total)[5m:1m])");
        f("avg_over_time((rate(http_total[5m])-rate(http_total[5m]))[5m:1m])");

        f("sum_over_time((up{cluster='a'} or up{cluster='b'})[5m:1m])");
        f("sum_over_time((up{cluster='a'} or up{cluster='b'})[5m:1m])");
        f("sum(sum_over_time((up{cluster='a'} or up{cluster='b'})[5m:1m])) by (instance)");

        // step (or resolution) is optional in subqueries
        f("max_over_time(rate(my_counter_total[5m])[1h:])");
        f("max_over_time(rate(my_counter_total[5m])[1h:1m])[5m:1m]");
        f("max_over_time(rate(my_counter_total[5m])[1h:])[5m:]");

        f(r#"
        WITH (
        cpuSeconds = node_cpu_seconds_total{instance=~"$node:$port",job=~"$job"},
        cpuIdle = rate(cpuSeconds{mode='idle'}[5m])
        )
        max_over_time(cpuIdle[1h:])"#);

        // These queries are mostly harmless, e.g. they return mostly correct results.
        f("rate(http_total)[5m:1m]");
        f("up[:5m]");
        f("sum(up[:5m])");
        f("absent(foo[5m])");
        f("sum(up[5m])");
        f("avg(foo[5m])");
        f("sort(foo[5m])");

        // These are valid subqueries with MetricsQL extention, which allows omitting lookbehind window for rollup functions
        f("rate(rate(http_total)[5m:1m])");
        f("rate(sum(rate(http_total))[5m:1m])");
        f("rate(sum(rate(http_total))[5m:1m])");
        f("avg_over_time((rate(http_total)-rate(http_total))[5m:1m])");

        // These are valid MetricsQL queries, which return correct result most of the time
        f("count_over_time(http_total)");

        // The following queries are from https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3974
        //
        // They are mostly correct. It is better to teach metricsql parser converting them to proper ones
        // instead of denying them.
        f("sum(http_total) offset 1m");
        f("round(sum(sum_over_time(http_total[1m])) by (instance)) offset 1m")
    }

    #[test]
    fn test_metricsql_is_likely_invalid_true() {
        fn f(q: &str) {
            let expr = parse(q).unwrap();
            assert!(
                is_likely_invalid(&expr),
                "unexpected result for is_likely_invalid({}); got false; want true",
                q
            );
        }

        f("rate(sum(http_total))");
        f("rate(rate(http_total))");
        f("sum(rate(sum(http_total)))");
        f("rate(sum(rate(http_total)))");
        f("rate(sum(sum(http_total)))");
        f("avg_over_time(rate(http_total[5m]))");

        f("rate(sum(http_total)) - rate(sum(http_total))");
        f("avg_over_time(rate(http_total)-rate(http_total))");

        // These queries are from https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3996
        f("sum_over_time(up{cluster='a'} or up{cluster='b'})");
        f("sum_over_time(up{cluster='a'}[1m] or up{cluster='b'}[1m])");
        f("sum(sum_over_time(up{cluster='a'}[1m] or up{cluster='b'}[1m])) by (instance)");

        f(r#"
        WITH (
        cpuSeconds = node_cpu_seconds_total{instance=~"$node:$port",job=~"$job"},
        cpuIdle = rate(cpuSeconds{mode='idle'}[5m])
        )
        max_over_time(cpuIdle)"#);
    }
} // mod tests
