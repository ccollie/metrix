#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;
    use crate::common::math::{linear_regression, STALE_NAN};
    use crate::functions::rollup::delta::*;
    use crate::functions::rollup::deriv::*;
    use crate::functions::rollup::integrate::rollup_integrate;
    use crate::functions::rollup::outlier_iqr::rollup_outlier_iqr;
    use crate::functions::rollup::rollup_fns::{
        remove_counter_resets, rollup_avg, rollup_changes, rollup_changes_prometheus, rollup_count,
        rollup_default, rollup_distinct, rollup_first, rollup_lag, rollup_lifetime, rollup_max,
        rollup_min, rollup_mode_over_time, rollup_rate_over_sum, rollup_resets,
        rollup_scrape_interval, rollup_stddev, rollup_sum, rollup_zscore_over_time, ROLLUP_LAST,
    };
    use crate::functions::rollup::{
        get_rollup_func_by_name, get_rollup_function_factory, get_rollup_function_handler,
        RollupConfig, RollupFuncArg, RollupHandler, RollupHandlerFactory,
    };
    use crate::types::{QueryValue, Timeseries, Timestamp};
    use crate::{compare_floats, compare_values, test_rows_equal, RuntimeResult};
    // https://github.com/VictoriaMetrics/VictoriaMetrics/blob/master/app/vmselect/promql/rollup_test.go

    const NAN: f64 = f64::NAN;
    const INF: f64 = f64::INFINITY;
    const TEST_VALUES: [f64; 12] = [
        123.0, 34.0, 44.0, 21.0, 54.0, 34.0, 99.0, 12.0, 44.0, 32.0, 34.0, 34.0,
    ];
    const TEST_TIMESTAMPS: [i64; 12] = [5, 15, 24, 36, 49, 60, 78, 80, 97, 115, 120, 130];

    fn get_rollup_function_factory_by_name(name: &str) -> RuntimeResult<RollupHandlerFactory> {
        let func = get_rollup_func_by_name(name)?;
        Ok(get_rollup_function_factory(func))
    }

    #[test]
    fn test_rollup_outlier_iqr() {
        fn f(values: &[f64], expected: f64) {
            let mut rfa = RollupFuncArg::default();
            rfa.values = values;
            rfa.timestamps = &[];

            let result = rollup_outlier_iqr(&mut rfa);
            if result.is_nan() {
                assert!(
                    expected.is_nan(),
                    "unexpected value; got {result}; want {}",
                    expected
                )
            } else {
                if expected.is_nan() {
                    panic!("unexpected value; got {result}; want {expected}")
                }
                assert_eq!(
                    result, expected,
                    "unexpected value; got {result}; want {expected}",
                )
            }
        }

        f(&[1.0, 2.0, 3.0, 4.0, 5.0], NAN);
        f(&[1.0, 2.0, 3.0, 4.0, 7.0], NAN);
        f(&[1.0, 2.0, 3.0, 4.0, 8.0], 8.0);
        f(&[1.0, 2.0, 3.0, 4.0, -2.0], NAN);
        f(&[1.0, 2.0, 3.0, 4.0, -3.0], -3.0)
    }

    #[test]
    fn test_rollup_ideriv_duplicate_timestamps() {
        let mut rfa = RollupFuncArg::default();
        rfa.values = &[1.0, 2.0, 3.0, 4.0, 5.0];
        rfa.timestamps = &[100, 100, 200, 300, 300];

        let n = rollup_ideriv(&mut rfa);
        assert_eq!(n, 20_f64, "unexpected value; got {n}; want 20");

        let mut rfa = RollupFuncArg::default();
        rfa.values = &[1.0, 2.0, 3.0, 4.0, 5.0];
        rfa.timestamps = &[100, 100, 300, 300, 300];
        let n = rollup_ideriv(&mut rfa);
        assert_eq!(n, 15_f64, "unexpected value; got {n}; want 15");

        let mut rfa = RollupFuncArg::default();
        rfa.prev_value = NAN;
        let n = rollup_ideriv(&mut rfa);
        assert!(n.is_nan(), "unexpected value; got {n}; want NAN");

        let mut rfa = RollupFuncArg::default();
        rfa.prev_value = NAN;
        rfa.values = &[15.0];
        rfa.timestamps = &[100];

        let n = rollup_ideriv(&mut rfa);
        assert!(n.is_nan(), "unexpected value; got {}; want {}", n, NAN);

        let mut rfa = RollupFuncArg::default();
        rfa.prev_timestamp = 90;
        rfa.prev_value = 10_f64;
        rfa.values = &[15_f64];
        rfa.timestamps = &[100];

        let n = rollup_ideriv(&mut rfa);
        assert_eq!(n, 500_f64, "unexpected value; got {n}; want 500");

        let mut rfs = RollupFuncArg::default();
        rfs.prev_timestamp = 100;
        rfs.prev_value = 10_f64;
        rfs.values = &[15_f64];
        rfs.timestamps = &[100];

        let n = rollup_ideriv(&mut rfs);
        assert!(
            compare_floats(INF, n),
            "unexpected value; got {n}; want INF",
        );

        let mut rfs = RollupFuncArg::default();
        rfs.prev_timestamp = 100;
        rfs.prev_value = 10_f64;
        rfs.values = &[15_f64, 20_f64];
        rfs.timestamps = &[100, 100];

        let n = rollup_ideriv(&mut rfs);
        assert!(
            compare_floats(INF, n),
            "unexpected value; got {n}; want INF",
        );
    }

    #[test]
    fn test_remove_counter_resets() {
        let mut values = Vec::from(TEST_VALUES);
        remove_counter_resets(&mut values);
        let values_expected: Vec<f64> = vec![
            123_f64, 157.0, 167.0, 188.0, 221.0, 255.0, 320.0, 332.0, 364.0, 396.0, 398.0, 398.0,
        ];
        test_rows_equal(
            &values,
            &TEST_TIMESTAMPS,
            &values_expected,
            &TEST_TIMESTAMPS,
        );

        // removeCounterResets doesn't expect negative values, so it doesn't work properly with them.
        let mut values: Vec<f64> = vec![-100.0, -200.0, -300.0, -400.0];
        remove_counter_resets(&mut values);
        let values_expected = vec![-100.0, -100.0, -100.0, -100.0];
        let timestamps_expected: Vec<i64> = vec![0, 1, 2, 3];
        test_rows_equal(
            &values,
            &timestamps_expected,
            &values_expected,
            &timestamps_expected,
        );

        // verify how partial counter reset is handled.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2787
        let mut values = vec![100_f64, 95.0, 120.0, 119.0, 139.0, 50.0];
        remove_counter_resets(&mut values);
        let values_expected = vec![100_f64, 100.0, 125.0, 125.0, 145.0, 195.0];
        let timestamps_expected = vec![0, 1, 2, 3, 4, 5];
        test_rows_equal(
            &values,
            &timestamps_expected,
            &values_expected,
            &timestamps_expected,
        );

        // verify results always increase monotonically with possible float operations precision error
        let mut values = [
            34.094223, 2.7518, 2.140669, 0.044878, 1.887095, 2.546569, 2.490149, 0.045, 0.035684,
            0.062454, 0.058296,
        ];
        remove_counter_resets(&mut values);

        let mut prev: f64 = values[0];
        for (i, v) in values.iter().enumerate() {
            if *v < prev {
                panic!("error: unexpected value keep getting bigger {i}; cur {v}; pre {prev}\n");
            }
            prev = *v;
        }
    }

    #[test]
    fn test_delta_values() {
        let mut values: Vec<f64> = vec![123_f64];
        delta_values(&mut values);
        let values_expected: Vec<f64> = vec![0_f64];
        test_rows_equal(
            &values,
            &TEST_TIMESTAMPS[0..1],
            &values_expected,
            &TEST_TIMESTAMPS[0..1],
        );

        values.clear();
        values.extend_from_slice(&TEST_VALUES);
        delta_values(&mut values);
        let values_expected: Vec<f64> = vec![
            -89_f64, 10.0, -23.0, 33.0, -20.0, 65.0, -87.0, 32.0, -12.0, 2.0, 0.0, 0.0,
        ];
        test_rows_equal(
            &values,
            &TEST_TIMESTAMPS,
            &values_expected,
            &TEST_TIMESTAMPS,
        );

        values.clear();
        // remove counter resets
        values.extend_from_slice(&TEST_VALUES);
        remove_counter_resets(&mut values);
        delta_values(&mut values);

        let values_expected = vec![
            34_f64, 10.0, 21.0, 33.0, 34.0, 65.0, 12.0, 32.0, 32.0, 2.0, 0.0, 0.0,
        ];
        test_rows_equal(
            &values,
            &TEST_TIMESTAMPS,
            &values_expected,
            &TEST_TIMESTAMPS,
        )
    }

    #[test]
    fn test_deriv_values() {
        let mut values: Vec<f64> = vec![123_f64];
        deriv_values(&mut values, &TEST_TIMESTAMPS[0..1]);
        let values_expected: Vec<f64> = vec![0_f64];
        test_rows_equal(
            &values,
            &TEST_TIMESTAMPS[0..1],
            &values_expected,
            &TEST_TIMESTAMPS[0..1],
        );

        values.clear();
        values.extend_from_slice(&TEST_VALUES);
        deriv_values(&mut values, &TEST_TIMESTAMPS);
        let values_expected: Vec<f64> = vec![
            -8900.0,
            1111.111111111111,
            -1916.6666666666665,
            2538.4615384615386,
            -1818.1818181818182,
            3611.1111111111113,
            -43500.0,
            1882.3529411764705,
            -666.6666666666667,
            400.0,
            0.0,
            0.0,
        ];
        test_rows_equal(
            &values,
            &TEST_TIMESTAMPS,
            &values_expected,
            &TEST_TIMESTAMPS,
        );

        values.clear();
        // remove counter resets
        values.extend_from_slice(&TEST_VALUES);
        remove_counter_resets(&mut values);
        deriv_values(&mut values, &TEST_TIMESTAMPS);
        let mut values_expected = vec![
            3400_f64,
            1111.111111111111,
            1750.0,
            2538.4615384615386,
            3090.909090909091,
            3611.1111111111113,
            6000.0,
            1882.3529411764705,
            1777.7777777777778,
            400.0,
            0.0,
            0.0,
        ];
        test_rows_equal(
            &values,
            &TEST_TIMESTAMPS,
            &values_expected,
            &TEST_TIMESTAMPS,
        );

        // duplicate timestamps
        let mut values = vec![1_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let timestamps = vec![100, 100, 200, 200, 300, 400, 400];
        deriv_values(&mut values, &timestamps);
        values_expected = vec![0_f64, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0];
        test_rows_equal(&values, &timestamps, &values_expected, &timestamps);
    }

    fn test_rollup_func(func_name: &str, args: Vec<QueryValue>, expected: f64) {
        let func = get_rollup_func_by_name(func_name).unwrap();

        let rf = get_rollup_function_handler(func, &args).unwrap();
        let mut rfa = RollupFuncArg::default();
        let mut values = Vec::from(TEST_VALUES);
        if func.should_remove_counter_resets() {
            let mut slice = values.as_mut_slice();
            remove_counter_resets(&mut slice);
            rfa.values = slice;
        } else {
            rfa.values = &values;
        }
        rfa.prev_value = NAN;
        rfa.prev_timestamp = 0;
        rfa.timestamps = &TEST_TIMESTAMPS;
        rfa.window = rfa.timestamps[rfa.timestamps.len() - 1] - &rfa.timestamps[0];
        let args_as_string = args
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        (0..5).for_each(|_| {
            let v = rf.eval(&mut rfa);
            assert!(
                compare_floats(expected, v),
                "unexpected value for {}({}); got {}; want {}",
                func_name,
                args_as_string,
                v,
                expected
            );
        });
    }

    #[test]
    fn test_rollup_duration_over_time() {
        let f = |max_interval: f64, expected: f64| {
            let max_intervals = QueryValue::from(max_interval);
            let args = vec![QueryValue::from(0), max_intervals];
            test_rollup_func("duration_over_time", args, expected)
        };

        f(-123.0, 0.0);
        f(0.0, 0.0);
        f(0.001, 0.0);
        f(0.005, 0.007);
        f(0.01, 0.036);
        f(0.02, 0.125);
        f(1.0, 0.125);
        f(100.0, 0.125)
    }

    #[test]
    fn test_rollup_share_le_over_time() {
        let f = |le: f64, expected: f64| {
            let les = QueryValue::from(le);
            let args = vec![QueryValue::from(0), les];
            test_rollup_func("share_le_over_time", args, expected)
        };

        f(-123_f64, 0.0);
        f(0_f64, 0.0);
        f(10_f64, 0.0);
        f(12_f64, 0.08333333333333333);
        f(30_f64, 0.16666666666666666);
        f(50_f64, 0.75);
        f(100_f64, 0.9166666666666666);
        f(123_f64, 1_f64);
        f(1000_f64, 1_f64)
    }

    #[test]
    fn test_rollup_share_eq_over_time() {
        fn f(eq: f64, expected: f64) {
            let eqs = QueryValue::from(eq);
            let args = vec![QueryValue::from(0), eqs];
            test_rollup_func("share_eq_over_time", args, expected)
        }

        f(-123.0, 0.0);
        f(34.0, 0.3333333333333333);
        f(44.0, 0.16666666666666666);
        f(123.0, 0.08333333333333333);
        f(1000.0, 0.0)
    }

    #[test]
    fn test_rollup_share_gt_over_time() {
        let f = |gt: f64, v_expected: f64| {
            let gts = QueryValue::from(gt);
            let args = vec![QueryValue::from(0), gts];
            test_rollup_func("share_gt_over_time", args, v_expected)
        };

        f(-123_f64, 1_f64);
        f(0_f64, 1_f64);
        f(10_f64, 1_f64);
        f(12_f64, 0.9166666666666666);
        f(30_f64, 0.8333333333333334);
        f(50_f64, 0.25);
        f(100_f64, 0.08333333333333333);
        f(123_f64, 0_f64);
        f(1000_f64, 0_f64)
    }

    #[test]
    fn test_rollup_count_le_over_time() {
        let f = |le: f64, expected: f64| {
            let le_param = QueryValue::from(le);
            let args = vec![QueryValue::from(0), le_param];
            test_rollup_func("count_le_over_time", args, expected);
        };

        f(-123.0, 0.0);
        f(0.0, 0.0);
        f(10.0, 0.0);
        f(12.0, 1.0);
        f(30.0, 2.0);
        f(50.0, 9.0);
        f(100.0, 11.0);
        f(123.0, 12.0);
        f(1000.0, 12.0)
    }

    #[test]
    fn test_rollup_count_gt_over_time() {
        let f = |gt: f64, expected: f64| {
            let gt_param = QueryValue::from(gt);
            let args = vec![QueryValue::from(0), gt_param];
            test_rollup_func("count_gt_over_time", args, expected)
        };

        f(-123_f64, 12_f64);
        f(0_f64, 12_f64);
        f(10_f64, 12_f64);
        f(12_f64, 11_f64);
        f(30_f64, 10_f64);
        f(50_f64, 3_f64);
        f(100_f64, 1_f64);
        f(123_f64, 0_f64);
        f(1000_f64, 0_f64)
    }

    #[test]
    fn test_rollup_count_eq_over_time() {
        let f = |eq: f64, v_expected: f64| {
            let eqs = QueryValue::from(eq);
            let args = vec![QueryValue::from(0), eqs];
            test_rollup_func("count_eq_over_time", args, v_expected)
        };

        f(-123.0, 0.0);
        f(0.0, 0.0);
        f(34.0, 4.0);
        f(123.0, 1.0);
        f(12.0, 1.0)
    }

    #[test]
    fn test_rollup_count_ne_over_time() {
        let f = |ne: f64, expected: f64| {
            let nes = QueryValue::from(ne);
            let args = vec![QueryValue::from(0), nes];
            test_rollup_func("count_ne_over_time", args, expected)
        };

        f(-123.0, 12.0);
        f(0.0, 12.0);
        f(34.0, 8.0);
        f(123.0, 11.0);
        f(12.0, 11.0);
    }

    #[test]
    fn test_rollup_sum_le_over_time() {
        let f = |le: f64, expected: f64| {
            let les = QueryValue::from(le);
            let args = vec![QueryValue::from(0), les];
            test_rollup_func("sum_le_over_time", args, expected);
        };

        f(-123.0, 0_f64);
        f(0.0, 0_f64);
        f(10.0, 0_f64);
        f(12.0, 12_f64);
        f(30.0, 33_f64);
        f(50.0, 289_f64);
        f(100.0, 442_f64);
        f(123.0, 565_f64);
        f(1000.0, 565_f64)
    }

    #[test]
    fn test_rollup_sum_gt_over_time() {
        let f = |le, expected: f64| {
            let les = QueryValue::from(le);
            let args = vec![QueryValue::from(0), les];
            test_rollup_func("sum_gt_over_time", args, expected);
        };

        f(-123.0, 565.0);
        f(0.0, 565.0);
        f(10.0, 565.0);
        f(12.0, 553.0);
        f(30.0, 532.0);
        f(50.0, 276.0);
        f(100.0, 123.0);
        f(123.0, 0.0);
        f(1000.0, 0.0)
    }

    #[test]
    fn test_rollup_sum_eq_over_time() {
        let f = |le, expected: f64| {
            let les = QueryValue::from(le);
            let args = vec![QueryValue::from(0), les];
            test_rollup_func("sum_eq_over_time", args, expected);
        };

        f(-123.0, 0.0);
        f(0.0, 0.0);
        f(10.0, 0.0);
        f(12.0, 12.0);
        f(30.0, 0.0);
        f(50.0, 0.0);
        f(100.0, 0.0);
        f(123.0, 123.0);
        f(1000.0, 0.0);
    }

    #[test]
    fn test_rollup_quantile_over_time() {
        let f = |phi: f64, expected: f64| {
            let phis = QueryValue::from(phi);
            let args = vec![phis, QueryValue::from(0)];
            test_rollup_func("quantile_over_time", args, expected)
        };

        f(-123.0, f64::NEG_INFINITY);
        f(-0.5, f64::NEG_INFINITY);
        f(0.0, 12.0);
        f(0.1, 22.1);
        f(0.5, 34.0);
        f(0.9, 94.50000000000001);
        f(1.0, 123.0);
        f(234.0, f64::INFINITY)
    }

    #[test]
    fn test_rollup_predict_linear() {
        let f = |sec: f64, expected: f64| {
            let secs = QueryValue::from(sec);
            let args = vec![QueryValue::from(0), secs];
            test_rollup_func("predict_linear", args, expected)
        };

        f(0e-3, 65.07405077267295);
        f(50e-3, 51.7311206569699);
        f(100e-3, 38.38819054126685);
        f(200e-3, 11.702330309860756)
    }

    #[test]
    fn test_linear_regression() {
        let f = |values: &[f64], timestamps: &[Timestamp], exp_v: f64, exp_k: f64| {
            let ts = &timestamps[0] + 100;
            let (v, k) = linear_regression(values, timestamps, ts);
            compare_values(&[v], &[exp_v]).unwrap();
            compare_values(&vec![k], &vec![exp_k]).unwrap();
        };

        f(&[1.0], &[1], f64::NAN, f64::NAN);
        f(&[1.0, 2.0], &[100, 300], 1.5, 5.0);
        f(
            &[2.0, 4.0, 6.0, 8.0, 10.0],
            &[100, 200, 300, 400, 500],
            4.0,
            20.0,
        );
    }

    #[test]
    fn test_rollup_holt_winters() {
        let f = |sf: f64, tf: f64, expected: f64| {
            let sf_param = QueryValue::from(sf);
            let tf_param = QueryValue::from(tf);
            let args = vec![QueryValue::from(0), sf_param, tf_param];
            test_rollup_func("holt_winters", args, expected)
        };

        f(-1.0, 0.5, NAN);
        f(0.0, 0.5, NAN);
        f(1.0, 0.5, NAN);
        f(2.0, 0.5, NAN);
        f(0.5, -1.0, NAN);
        f(0.5, 0.0, NAN);
        f(0.5, 1.0, NAN);
        f(0.5, 2.0, NAN);
        f(0.5, 0.5, 34.97794532775879);
        f(0.1, 0.5, -131.30529492371622);
        f(0.1, 0.1, -397.3307790780296);
        f(0.5, 0.1, -5.791530520284198);
        f(0.5, 0.9, 25.498906408926757);
        f(0.9, 0.9, 33.99637566941818)
    }

    #[test]
    fn test_rollup_hoeffding_bound_lower() {
        let f = |phi: f64, expected: f64| {
            let phis = QueryValue::from(phi);
            let args = vec![phis, QueryValue::from(0)];
            test_rollup_func("hoeffding_bound_lower", args, expected)
        };

        f(0.5, 28.21949401521037);
        f(-1.0, 47.083333333333336);
        f(0.0, 47.083333333333336);
        f(1.0, f64::NEG_INFINITY);
        f(2.0, f64::NEG_INFINITY);
        f(0.1, 39.72878000047643);
        f(0.9, 12.701803086472331)
    }

    #[test]
    fn test_rollup_hoeffding_bound_upper() {
        let f = |phi: f64, expected: f64| {
            let phis = QueryValue::from(phi);
            let args = vec![phis, QueryValue::from(0)];
            test_rollup_func("hoeffding_bound_upper", args, expected)
        };

        f(0.5, 65.9471726514563);
        f(-1.0, 47.083333333333336);
        f(0.0, 47.083333333333336);
        f(1.0, INF);
        f(2.0, INF);
        f(0.1, 54.43788666619024);
        f(0.9, 81.46486358019433)
    }

    #[test]
    fn test_rollup_new_rollup_func_success() {
        let f = |func_name: &str, expected: f64| {
            let args = vec![QueryValue::from(0)];
            test_rollup_func(func_name, args, expected)
        };

        f("default_rollup", 34_f64);
        f("changes", 11_f64);
        f("changes_prometheus", 10_f64);
        f("delta", 34_f64);
        f("delta_prometheus", -89_f64);
        f("deriv", -266.85860231406093);
        f("deriv_fast", -712_f64);
        f("idelta", 0_f64);
        f("increase", 398_f64);
        f("increase_prometheus", 275_f64);
        f("irate", 0_f64);
        f("outlier_iqr_over_time", NAN);
        f("rate", 2200_f64);
        f("resets", 5_f64);
        f("range_over_time", 111_f64);
        f("avg_over_time", 47.083333333333336);
        f("min_over_time", 12_f64);
        f("max_over_time", 123_f64);
        f("tmin_over_time", 0.08);
        f("tmax_over_time", 0.005);
        f("tfirst_over_time", 0.005);
        f("tlast_change_over_time", 0.12);
        f("tlast_over_time", 0.13);
        f("sum_over_time", 565.0);
        f("sum2_over_time", 37951_f64);
        f("geomean_over_time", 39.33466603189148);
        f("count_over_time", 12_f64);
        f("stale_samples_over_time", 0_f64);
        f("stddev_over_time", 30.752935722554287);
        f("stdvar_over_time", 945.7430555555555);
        f("first_over_time", 123_f64);
        f("last_over_time", 34_f64);
        f("integrate", 0.817);
        f("distinct_over_time", 8_f64);
        f("ideriv", 0_f64);
        f("decreases_over_time", 5_f64);
        f("increases_over_time", 5_f64);
        f("increase_pure", 398_f64);
        f("ascent_over_time", 142_f64);
        f("descent_over_time", 231_f64);
        f("zscore_over_time", -0.4254336383156416);
        f("timestamp", 0.13);
        f("timestamp_with_name", 0.13);
        f("mode_over_time", 34_f64);
        f("rate_over_sum", 4520_f64)
    }

    #[test]
    fn test_rollup_new_rollup_func_error() {
        let nrf = get_rollup_function_factory_by_name("non-existing-func");
        assert!(nrf.is_err(), "expecting err; got a factory function");

        let f = |func_name: &str, args: &[QueryValue]| {
            let nrf = get_rollup_function_factory_by_name(func_name).unwrap();
            let args = Vec::from(args);
            let _rf = nrf(&args);
            assert!(
                _rf.is_err(),
                "expecting err for {}({})",
                func_name,
                args.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
        };

        // Invalid number of args
        f("default_rollup", &[]);
        f("holt_winters", &[]);
        f("predict_linear", &[]);
        f("quantile_over_time", &[]);
        f("quantiles_over_time", &[]);

        // Invalid arg type
        let scalar_ts = QueryValue::InstantVector(vec![Timeseries::new(vec![123], vec![321_f64])]);
        let me = QueryValue::from(0);
        let _123 = QueryValue::from(123);
        let _321 = QueryValue::from(321);
        f("holt_winters", &[_123.clone(), _123.clone(), _123.clone()]);
        f("holt_winters", &[me.clone(), _123.clone(), _321.clone()]);
        f("holt_winters", &[me.clone(), scalar_ts, _321.clone()]);
        f("predict_linear", &[_123.clone(), _123.clone()]);
        f("predict_linear", &[me.clone(), _123.clone()]);
        f("quantile_over_time", &[_123.clone(), _123.clone()]);
        f("quantiles_over_time", &[_123.clone(), _123.clone()]);
    }

    fn test_rollup(rc: &mut RollupConfig, values_expected: &[f64], timestamps_expected: &[i64]) {
        rc.max_points_per_series = 10000;
        // rc.ensure_timestamps().unwrap();
        let mut values: Vec<f64> = vec![];
        let samples_scanned = rc
            .exec(&mut values, &TEST_VALUES, &TEST_TIMESTAMPS)
            .unwrap();
        assert_ne!(
            samples_scanned, 0,
            "expecting non-zero samples_scanned from rollupConfig.exec"
        );
        test_rows_equal(
            &values,
            &rc.timestamps,
            &values_expected,
            &timestamps_expected,
        )
    }

    #[test]
    fn test_rollup_no_window_no_points_before_start() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_first);
        rc.start = 0;
        rc.end = 4;
        rc.step = Duration::from_millis(1);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, NAN, NAN, NAN, NAN], &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_rollup_no_window_no_points_after_end() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_delta);
        rc.start = 120;
        rc.end = 148;
        rc.step = Duration::from_millis(4);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[2_f64, 0.0, 0.0, 0.0, NAN, NAN, NAN, NAN],
            &[120, 124, 128, 132, 136, 140, 144, 148],
        )
    }

    #[test]
    fn test_window_no_points_before_start() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_first);
        rc.start = 0;
        rc.end = 4;
        rc.step = Duration::from_millis(1);
        rc.window = Duration::from_millis(3);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, NAN, NAN, NAN, NAN], &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_window_no_points_after_end() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_first);
        rc.start = 161;
        rc.end = 191;
        rc.step = Duration::from_millis(10);
        rc.window = Duration::from_millis(3);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, NAN, NAN, NAN], &[161, 171, 181, 191]);
    }

    #[test]
    fn test_no_window_partial_points_before_start() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_first);
        rc.start = 0;
        rc.end = 25;
        rc.step = Duration::from_millis(5);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 123.0, NAN, 34.0, NAN, 44.0],
            &[0, 5, 10, 15, 20, 25],
        );
    }

    #[test]
    fn test_no_window_partial_points_after_end() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_first);
        rc.start = 100;
        rc.end = 160;
        rc.step = Duration::from_millis(20);
        rc.window = Duration::ZERO;
        rc.max_points_per_series = 10000;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[44_f64, 32.0, 34.0, NAN], &[100, 120, 140, 160]);
    }

    #[test]
    fn test_no_window_partial_points_middle() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_first);
        rc.start = -50;
        rc.end = 150;
        rc.step = Duration::from_millis(50);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, NAN, 123.0, 34.0, 32.0],
            &[-50, 0, 50, 100, 150],
        );
    }

    #[test]
    fn test_window_partial_points_before_start() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(ROLLUP_LAST);
        rc.start = 0;
        rc.end = 20;
        rc.step = Duration::from_millis(5);
        rc.window = Duration::from_millis(8);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 123_f64, 123_f64, 34_f64, 34_f64],
            &[0, 5, 10, 15, 20],
        );
    }

    #[test]
    fn test_window_partial_points_after_end() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(ROLLUP_LAST);
        rc.start = 100;
        rc.end = 160;
        rc.step = Duration::from_millis(20);
        rc.window = Duration::from_millis(18);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[44_f64, 34_f64, 34_f64, NAN],
            &[100, 120, 140, 160],
        );
    }

    #[test]
    fn test_window_partial_points_middle() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(ROLLUP_LAST);
        rc.start = 0;
        rc.end = 150;
        rc.step = Duration::from_millis(50);
        rc.window = Duration::from_millis(19);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, 54_f64, 44_f64, NAN], &[0, 50, 100, 150]);
    }

    #[test]
    fn test_lookback_delta_1() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_first);
        rc.start = 80;
        rc.end = 140;
        rc.step = Duration::from_millis(10);
        rc.lookback_delta = Duration::from_millis(1); // secs
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[99_f64, NAN, 44.0, NAN, 32.0, 34.0, NAN],
            &[80, 90, 100, 110, 120, 130, 140],
        );
    }

    #[test]
    fn test_lookback_delta_7() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_first);
        rc.start = 80;
        rc.end = 140;
        rc.step = Duration::from_millis(10);
        rc.lookback_delta = Duration::from_millis(7);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[99_f64, NAN, 44.0, NAN, 32.0, 34.0, NAN],
            &[80, 90, 100, 110, 120, 130, 140],
        );
    }

    #[test]
    fn test_lookback_delta_0() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_first);
        rc.start = 80;
        rc.end = 140;
        rc.step = Duration::from_millis(10);
        rc.lookback_delta = Duration::from_millis(0);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[99_f64, NAN, 44.0, NAN, 32.0, 34.0, NAN],
            &[80, 90, 100, 110, 120, 130, 140],
        );
    }

    #[test]
    fn test_rollup_increase_with_staleness_step_gt_gap() {
        // there is a gap between samples in the dataset below
        let timestamps = Arc::new(vec![0, 15000, 30000, 70000]);
        let values = vec![1.0, 1.0, 1.0, 1.0];

        // step > gap
        let mut rc = RollupConfig {
            handler: RollupHandler::Wrapped(rollup_delta),
            start: 0,
            end: 70000,
            step: Duration::from_millis(35000),
            window: Duration::ZERO,
            max_points_per_series: 10000,
            ..Default::default()
        };

        rc.ensure_timestamps().expect("failed to ensure timestamps");
        let mut dst_values: Vec<f64> = vec![];

        let samples_scanned = rc.exec_internal(&mut dst_values, None, &values, &timestamps).expect("failed to exec");
        assert_eq!(samples_scanned, 8);

        let values_expected = vec![1.0, 0.0, 0.0];
        let timestamps_expected = vec![0, 35000, 70000];
        test_rows_equal(&dst_values, &rc.timestamps, &values_expected, &timestamps_expected);
    }

    #[test]
    fn test_rollup_increase_with_staleness_step_lt_gap() {
        // there is a gap between samples in the dataset below
        let timestamps = vec![0, 15000, 30000, 70000];
        let values = vec![1.0, 1.0, 1.0, 1.0];

        // step < gap
        let mut rc = RollupConfig {
            handler: RollupHandler::Wrapped(rollup_delta),
            start: 0,
            end: 70000,
            step: Duration::from_millis(10000),
            window: Duration::ZERO,
            max_points_per_series: 10000,
           ..Default::default()
        };

        rc.ensure_timestamps().expect("failed to ensure timestamps");
        let mut dst_values: Vec<f64> = vec![];

        let samples_scanned = rc.exec_internal(&mut dst_values, None, &values, &timestamps).expect("failed to exec");
        assert_eq!(samples_scanned, 8);

        let values_expected = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let timestamps_expected: Vec<i64> = vec![0, 10000, 20000, 30000, 40000, 50000, 60000, 70000];

        test_rows_equal(&dst_values, &rc.timestamps, &values_expected, &timestamps_expected);
    }

    #[test]
    fn test_rollup_increase_with_staleness_marker() {
        // there is a gap between samples in the dataset below
        let timestamps = Arc::new(vec![0, 10000, 20000, 30000, 40000]);
        let values = vec![1.0, 1.0, 1.0, STALE_NAN, 1.0];

        // step > gap
        let mut rc = RollupConfig {
            handler: RollupHandler::Wrapped(rollup_delta),
            start: 0,
            end: 40000,
            step: Duration::from_millis(10000),
            window: Duration::ZERO,
            max_points_per_series: 10000,
            ..Default::default()
        };

        rc.ensure_timestamps().expect("failed to ensure timestamps");

        let mut dst_values: Vec<f64> = vec![];

        let samples_scanned = rc.exec_internal(&mut dst_values, None, &values, &timestamps).expect("failed to exec");
        assert_eq!(samples_scanned, 10);

        let values_expected = vec![1.0, 0.0, 0.0, f64::NAN, 1.0];
        let timestamps_expected = vec![0, 10000, 20000, 30000, 40000];
        test_rows_equal(&dst_values, &rc.timestamps, &values_expected, &timestamps_expected);
    }

    #[test]
    fn test_rollup_count_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_count);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, 4.0, 4.0, 3.0, 1.0], &[0, 40, 80, 120, 160]);
    }

    #[test]
    fn test_rollup_min_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_min);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 21.0, 12.0, 32.0, 34.0],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_max_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_max);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 123.0, 99.0, 44.0, 34.0],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_sum_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_sum);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.max_points_per_series = 10000;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 222.0, 199.0, 110.0, 34.0],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_delta_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_delta);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 21.0, -9.0, 22.0, 0.0],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_delta_prometheus_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_delta_prometheus);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, -102.0, -42.0, -10.0, NAN],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_idelta_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_idelta);
        rc.start = 10;
        rc.end = 130;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[123_f64, 33.0, -87.0, 0.0], &[10, 50, 90, 130]);
    }

    #[test]
    fn test_rollup_lag_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_lag);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 0.004, 0.0, 0.0, 0.03],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_lifetime_1_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_lifetime);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 0.031, 0.044, 0.04, 0.01],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_lifetime_2_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_lifetime);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::from_millis(200);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 0.031, 0.075, 0.115, 0.125],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_scrape_interval_1_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_scrape_interval);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 0.010333333333333333, 0.011, 0.013333333333333334, 0.01],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_scrape_interval_2_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_scrape_interval);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::from_millis(80);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[
                NAN,
                0.010333333333333333,
                0.010714285714285714,
                0.012,
                0.0125,
            ],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_changes_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_changes);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, 4.0, 4.0, 3.0, 0.0], &[0, 40, 80, 120, 160]);
    }

    #[test]
    fn test_rollup_changes_prometheus_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_changes_prometheus);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, 3.0, 3.0, 2.0, 0.0], &[0, 40, 80, 120, 160]);
    }

    #[test]
    fn test_rollup_changes_small_window_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_changes);
        rc.start = 0;
        rc.end = 45;
        rc.step = Duration::from_millis(9);
        rc.window = Duration::from_millis(9);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 1.0, 1.0, 1.0, 1.0, 0.0],
            &[0, 9, 18, 27, 36, 45],
        );
    }

    #[test]
    fn test_rollup_resets_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_resets);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, 2.0, 2.0, 1.0, 0.0], &[0, 40, 80, 120, 160]);
    }

    #[test]
    fn test_rollup_avg_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_avg);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 55.5, 49.75, 36.666666666666664, 34.0],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_deriv_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_deriv_slow);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[
                NAN,
                -2879.310344827588,
                127.87627310448904,
                -496.5831435079728,
                0.0,
            ],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_deriv_fast_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_deriv_fast);
        rc.start = 0;
        rc.end = 20;
        rc.step = Duration::from_millis(4);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, NAN, NAN, 0.0, -8900.0, 0.0],
            &[0, 4, 8, 12, 16, 20],
        );
    }

    #[test]
    fn test_rollup_ideriv_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_ideriv);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, -1916.6666666666665, -43500.0, 400.0, 0.0],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_stddev_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_stddev);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.max_points_per_series = 10000;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[
                NAN,
                39.81519810323691,
                32.080952292598795,
                5.2493385826745405,
                0.0,
            ],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_integrate_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_integrate);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::ZERO;
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 2.148, 1.593, 1.156, 1.36],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_distinct_over_time_1_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_distinct);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, 4.0, 4.0, 3.0, 1.0], &[0, 40, 80, 120, 160]);
    }

    #[test]
    fn test_rollup_distinct_over_time_2_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_distinct);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::from_millis(80);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(&mut rc, &[NAN, 4.0, 7.0, 6.0, 3.0], &[0, 40, 80, 120, 160]);
    }

    #[test]
    fn test_rollup_mode_over_time_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_mode_over_time);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::from_millis(80);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 21.0, 34.0, 34.0, 34.0],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_rate_over_sum_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_rate_over_sum);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::from_millis(80);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[NAN, 2775.0, 5262.5, 3862.5, 1800.0],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_zscore_over_time_no_window() {
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_zscore_over_time);
        rc.start = 0;
        rc.end = 160;
        rc.step = Duration::from_millis(40);
        rc.window = Duration::from_millis(80);
        rc.ensure_timestamps().expect("failed to ensure timestamps");
        test_rollup(
            &mut rc,
            &[
                NAN,
                -0.86650328627136,
                -1.1200838283548589,
                -0.40035755084856683,
                NAN,
            ],
            &[0, 40, 80, 120, 160],
        );
    }

    #[test]
    fn test_rollup_big_number_of_values() {
        const SRC_VALUES_COUNT: i64 = 10000;
        let mut rc = RollupConfig::default();
        rc.handler = RollupHandler::Wrapped(rollup_default);
        rc.end = SRC_VALUES_COUNT;
        rc.step = Duration::from_millis((SRC_VALUES_COUNT / 5) as u64);
        rc.window = Duration::from_millis((SRC_VALUES_COUNT / 4) as u64);
        rc.max_points_per_series = 10000;
        // rc.ensure_timestamps().unwrap();
        let mut src_values: Vec<f64> = Vec::with_capacity(SRC_VALUES_COUNT as usize);
        let mut src_timestamps: Vec<Timestamp> = Vec::with_capacity(SRC_VALUES_COUNT as usize);
        for i in 0..SRC_VALUES_COUNT {
            src_values.push(i as f64);
            src_timestamps.push(i / 2)
        }
        let mut values: Vec<f64> = vec![];
        let samples_scanned = rc.exec(&mut values, &src_values, &src_timestamps).unwrap();
        assert_ne!(
            samples_scanned, 0,
            "expecting non-zero samples_scanned from rollupConfig.co"
        );
        let values_expected: Vec<f64> = vec![1_f64, 4001.0, 8001.0, 9999.0, NAN, NAN];
        let timestamps_expected: Vec<Timestamp> = vec![0, 2000, 4000, 6000, 8000, 10000];
        test_rows_equal(
            &values,
            &rc.timestamps,
            &values_expected,
            &timestamps_expected,
        );
    }

    #[test]
    fn test_rollup_delta() {
        let f = |prev_value: f64,
                 real_prev_value: f64,
                 real_next_value: f64,
                 values: &[f64],
                 result_expected: f64| {
            let mut rfa = RollupFuncArg::default();
            rfa.prev_value = prev_value;
            rfa.values = values;
            rfa.real_prev_value = real_prev_value;
            rfa.real_next_value = real_next_value;
            let result = rollup_delta(&mut rfa);
            if result.is_nan() {
                assert!(
                    result_expected.is_nan(),
                    "unexpected result; got {}; want {}",
                    result,
                    result_expected
                );
                return;
            }
            assert_eq!(
                result, result_expected,
                "unexpected result; got {}; want {}",
                result, result_expected
            );
        };

        f(NAN, NAN, NAN, &[], NAN);

        // Small initial value
        f(NAN, NAN, NAN, &[1.0], 1_f64);
        f(NAN, NAN, NAN, &[10.0], 0_f64);
        f(NAN, NAN, NAN, &[100.0], 0_f64);
        f(NAN, NAN, NAN, &[1.0, 2.0, 3.0], 3_f64);
        f(1_f64, NAN, NAN, &[1.0, 2.0, 3.0], 2_f64);
        f(NAN, NAN, NAN, &[5.0, 6.0, 8.0], 8_f64);
        f(2_f64, NAN, NAN, &[5.0, 6.0, 8.0], 6.0);

        f(NAN, NAN, NAN, &[100.0, 100.0], 0.0);

        // Big initial value with zero delta after that.
        f(NAN, NAN, NAN, &[1000.0], 0.0);
        f(NAN, NAN, NAN, &[1000.0, 1000.0], 0.0);

        // Big initial value with small delta after that.
        f(NAN, NAN, NAN, &[1000.0, 1001.0, 1002.0], 2_f64);

        // Non-NAN real_prev_value
        f(NAN, 900.0, NAN, &[1000.0], 100_f64);
        f(NAN, 1000.0, NAN, &[1000.0], 0_f64);
        f(NAN, 1100.0, NAN, &[1000.0], -100_f64);
        f(NAN, 900.0, NAN, &[1000.0, 1001.0, 1002.0], 102_f64);

        // Small delta between realNextValue and values
        f(NAN, NAN, 990.0, &[1000.0], 0_f64);
        f(NAN, NAN, 1005.0, &[1000.0], 0_f64);

        // Big delta between relaNextValue and values
        f(NAN, NAN, 800.0, &[1000.0], 1000_f64);
        f(NAN, NAN, 1300.0, &[1000.0], 1000_f64);

        // Empty values
        f(1_f64, NAN, NAN, &[], 0_f64);
        f(100_f64, NAN, NAN, &[], 0_f64)
    }
}
