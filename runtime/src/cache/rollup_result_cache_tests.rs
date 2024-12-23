#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;
    use metricsql_parser::ast::{AggregationExpr, Expr, FunctionExpr, MetricExpr};
    use metricsql_parser::functions::AggregateFunction;
    use metricsql_parser::label::{Matcher, MatchOp};

    use crate::cache::rollup_result_cache::{merge_timeseries, RollupResultCache};
    use crate::execution::EvalConfig;
    use crate::{test_timeseries_equal};
    use crate::types::{MetricName, Timeseries};

    const NAN: f64 = f64::NAN;

    struct TestContext {
        fe: Expr,
        ae: Expr,
        ec: EvalConfig,
        cache: RollupResultCache,
        window: Duration,
    }

    fn setup() -> TestContext {
        let window = Duration::from_millis(456_u64);
        let mut ec = EvalConfig::default();
        ec.start = 1000;
        ec.end = 2000;
        ec.step = Duration::from_millis(200);
        ec.max_points_per_series = 1e4 as usize;
        ec.set_caching(true);

        let me = MetricExpr::default()
            .append(Matcher::new(MatchOp::Equal, "aaa", "xxx").unwrap());

        let fe = FunctionExpr::from_single_arg("avg", Expr::MetricExpression(me)).unwrap();

        let mut ae = AggregationExpr::new(AggregateFunction::Sum, vec![]);
        ae.args.push(Expr::Function(fe.clone()));

        let cache = RollupResultCache::default();

        TestContext {
            ec,
            ae: Expr::Aggregation(ae),
            fe: Expr::Function(fe),
            cache,
            window,
        }
    }

    fn create_ts(timestamps: &[i64], values: &[f64]) -> Timeseries {
        Timeseries {
            metric_name: MetricName::default(),
            timestamps: Arc::new(Vec::from(timestamps)),
            values: Vec::from(values),
        }
    }

    // Try obtaining an empty value.
    #[test]
    fn test_empty() {
        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();
        let (tss, new_start) = cache.get(&ec, &fe, window).unwrap();
        assert_eq!(
            new_start, ec.start,
            "unexpected new_start; got {}; want {}",
            new_start, ec.start
        );

        expect_empty(tss);
    }

    // Store timeseries overlapping with start
    #[test]
    fn start_overlap_no_ae() {
        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();
        let tss = vec![create_ts(&[800, 1000, 1200], &[0_f64, 1_f64, 2_f64])];

        cache
            .put(&ec, &fe, window, &tss)
            .expect("error putting to cache");
        let (tss, new_start) = cache.get(&ec, &fe, window).unwrap();
        assert_eq!(
            new_start, 1400,
            "unexpected new_start; got {}; want {}",
            new_start, 1400
        );

        let tss_expected = vec![create_ts(&[1000, 1200], &[1_f64, 2_f64])];

        let tss = tss.unwrap();
        test_timeseries_equal(&tss, &tss_expected)
    }

    #[test]
    fn start_overlap_with_ae() {
        let tss = vec![create_ts(&[800, 1000, 1200], &[0_f64, 1_f64, 2_f64])];

        let TestContext {
            cache,
            ec,
            ae,
            window,
            ..
        } = setup();

        cache
            .put(&ec, &ae, window, &tss)
            .expect("error putting to cache");

        let (tss, new_start) = cache.get(&ec, &ae, window).unwrap();
        assert_eq!(
            new_start, 1400,
            "unexpected new_start; got {}; want {}",
            new_start, 1400
        );

        let tss_expected = vec![create_ts(&[1000, 1200], &[1_f64, 2_f64])];

        let tss = tss.unwrap();
        test_timeseries_equal(&tss, &tss_expected)
    }

    // Store timeseries overlapping with end
    #[test]
    fn end_overlap() {
        let tss = vec![create_ts(
            &[1800, 2000, 2200, 2400],
            &[333_f64, 0_f64, 1_f64, 2_f64],
        )];

        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();

        cache
            .put(&ec, &fe, window, &tss)
            .expect("error putting to cache");

        let (tss, new_start) = cache.get(&ec, &fe, window).unwrap();

        assert_eq!(
            new_start, 1000,
            "unexpected new_start; got {}; want {}",
            new_start, 1000
        );
        expect_empty(tss);
    }

    // Store timeseries covered by [start ... end]
    #[test]
    fn full_cover() {
        let tss = vec![create_ts(&[1200, 1400, 1600], &[0_f64, 1_f64, 2_f64])];

        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();
        cache
            .put(&ec, &fe, window, &tss)
            .expect("error putting to cache");

        let (tss, new_start) = cache.get(&ec, &fe, window).unwrap();

        assert_eq!(
            new_start, 1000,
            "unexpected new_start; got {}; want {}",
            new_start, 1000
        );
        expect_empty(tss);
    }

    // Store timeseries below start
    #[test]
    fn before_start() {
        let tss = vec![create_ts(&[200, 400, 600], &[0_f64, 1_f64, 2_f64])];

        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();
        cache
            .put(&ec, &fe, window, &tss)
            .expect("error putting to cache");

        let (tss, new_start) = cache.get(&ec, &fe, window).unwrap();

        assert_eq!(
            new_start, 1000,
            "unexpected new_start; got {}; want {}",
            new_start, 1000
        );
        expect_empty(tss);
    }

    // Store timeseries after end
    #[test]
    fn after_end() {
        let tss = vec![create_ts(&[2200, 2400, 2600], &[0_f64, 1_f64, 2_f64])];

        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();
        cache
            .put(&ec, &fe, window, &tss)
            .expect("error putting to cache");

        let (tss, new_start) = cache.get(&ec, &fe, window).unwrap();

        assert_eq!(
            new_start, 1000,
            "unexpected new_start; got {}; want {}",
            new_start, 1000
        );
        expect_empty(tss);
    }

    // Store timeseries bigger than the interval [start ... end]
    #[test]
    fn bigger_than_start_end() {
        let tss = vec![create_ts(
            &[800, 1000, 1200, 1400, 1600, 1800, 2000, 2200],
            &[0_f64, 1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64, 7_f64],
        )];

        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();
        cache
            .put(&ec, &fe, window, &tss)
            .expect("error putting to cache");

        let (tss, new_start) = cache.get(&ec, &fe, window).unwrap();

        assert_eq!(
            new_start, 2200,
            "unexpected new_start; got {}; want {}",
            new_start, 2200
        );

        let tss_expected = vec![create_ts(
            &[1000, 1200, 1400, 1600, 1800, 2000],
            &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64],
        )];

        test_timeseries_equal(tss.unwrap().as_slice(), &tss_expected)
    }

    // Store timeseries matching the interval [start ... end]
    #[test]
    fn start_end_match() {
        let tss = vec![create_ts(
            &[1000, 1200, 1400, 1600, 1800, 2000],
            &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64],
        )];

        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();

        cache
            .put(&ec, &fe, window, &tss)
            .expect("error putting to cache");

        let (tss, new_start) = cache.get(&ec, &fe, window).unwrap();

        assert_eq!(
            new_start, 2200,
            "unexpected new_start; got {}; want {}",
            new_start, 2200
        );

        let tss_expected = vec![create_ts(
            &[1000, 1200, 1400, 1600, 1800, 2000],
            &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64],
        )];

        let series = tss.unwrap();
        test_timeseries_equal(&series, &tss_expected)
    }

    // Store big timeseries, so their marshaled size exceeds 64Kb.
    #[test]
    fn big_timeseries() {
        let mut tss: Vec<Timeseries> = vec![];
        (0..1000).for_each(|_| {
            let ts = create_ts(
                &[1000, 1200, 1400, 1600, 1800, 2000],
                &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64],
            );
            tss.push(ts);
        });

        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();

        cache
            .put(&ec, &fe, window, &tss)
            .expect("error putting to cache");

        let (tss_result, new_start) = cache.get(&ec, &fe, window).unwrap();

        assert_eq!(
            new_start, 2200,
            "unexpected new_start; got {}; want {}",
            new_start, 2200
        );

        let tss_result = tss_result.unwrap();
        test_timeseries_equal(&tss_result, &tss)
    }

    // Store multiple time series
    #[test]
    fn multi_timeseries() {
        let tss1 = vec![create_ts(&[800, 1000, 1200], &[0_f64, 1_f64, 2_f64])];

        let tss2 = vec![create_ts(
            &[1800, 2000, 2200, 2400],
            &[333_f64, 0_f64, 1_f64, 2_f64],
        )];

        let tss3 = vec![create_ts(&[1200, 1400, 1600], &[0_f64, 1_f64, 2_f64])];

        let TestContext {
            cache,
            ec,
            fe,
            window,
            ..
        } = setup();

        cache
            .put(&ec, &fe, window, &tss1)
            .expect("error putting value in cache");
        cache
            .put(&ec, &fe, window, &tss2)
            .expect("error putting value in cache");
        cache
            .put(&ec, &fe, window, &tss3)
            .expect("error putting value in cache");

        let (tss, new_start) = cache.get(&ec, &fe, window).unwrap();
        assert_eq!(
            new_start, 1400,
            "unexpected new_start; got {}; want {}",
            new_start, 1400
        );

        let tss_expected = vec![create_ts(&[1000, 1200], &[1_f64, 2_f64])];

        let tss = tss.unwrap();

        test_timeseries_equal(&tss, &tss_expected)
    }

    struct MergeTestContext {
        ec: EvalConfig,
        bstart: i64,
    }

    fn setup_merge() -> MergeTestContext {
        let mut ec = EvalConfig::default();
        ec.start = 1000;
        ec.end = 2000;
        ec.step = Duration::from_millis(200);
        ec.max_points_per_series = 1e4 as usize;

        MergeTestContext {
            ec,
            bstart: 1400_i64,
        }
    }

    fn expect_empty(tss: Option<Vec<Timeseries>>) {
        let res = tss.unwrap_or(vec![]);
        assert_eq!(
            res.len(),
            0,
            "got {} timeseries, while expecting zero",
            res.len()
        );
    }

    #[test]
    fn merge_bstart_eq_ec_start() {
        let a: Vec<Timeseries> = vec![];
        let b = vec![create_ts(
            &[1000, 1200, 1400, 1600, 1800, 2000],
            &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64],
        )];

        let MergeTestContext { ec, .. } = setup_merge();
        let tss = merge_timeseries(a, b, 1000, &ec).expect("unable to merge timeseries");
        let tss_expected = vec![create_ts(
            &[1000, 1200, 1400, 1600, 1800, 2000],
            &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64],
        )];

        test_timeseries_equal(&tss, &tss_expected)
    }

    #[test]
    fn merge_a_empty() {
        let a: Vec<Timeseries> = vec![];
        let b = vec![create_ts(
            &[1400, 1600, 1800, 2000],
            &[3_f64, 4_f64, 5_f64, 6_f64],
        )];

        let MergeTestContext { ec, bstart } = setup_merge();
        let tss = merge_timeseries(a, b, bstart, &ec).expect("unable to merge timeseries");
        let tss_expected = vec![create_ts(
            &[1000, 1200, 1400, 1600, 1800, 2000],
            &[NAN, NAN, 3_f64, 4_f64, 5_f64, 6_f64],
        )];

        test_timeseries_equal(&tss, &tss_expected)
    }

    #[test]
    fn merge_b_empty() {
        let a = vec![create_ts(&[1000, 1200], &[2_f64, 1_f64])];

        let b = vec![Timeseries::default()];

        let MergeTestContext { ec, bstart } = setup_merge();
        let tss = merge_timeseries(a, b, bstart, &ec).expect("unable to merge timeseries");

        let tss_expected = vec![create_ts(
            &[1000, 1200, 1400, 1600, 1800, 2000],
            &[2_f64, 1_f64, NAN, NAN, NAN, NAN],
        )];

        test_timeseries_equal(&tss, &tss_expected);
    }

    #[test]
    fn merge_non_empty() {
        let a = vec![create_ts(&[1000, 1200], &[2_f64, 1_f64])];

        let b = vec![create_ts(
            &[1400, 1600, 1800, 2000],
            &[3_f64, 4_f64, 5_f64, 6_f64],
        )];

        let MergeTestContext { ec, bstart } = setup_merge();
        let tss = merge_timeseries(a, b, bstart, &ec).expect("unable to merge timeseries");
        let tss_expected = vec![create_ts(
            &[1000, 1200, 1400, 1600, 1800, 2000],
            &[2_f64, 1_f64, 3_f64, 4_f64, 5_f64, 6_f64],
        )];

        test_timeseries_equal(&tss, &tss_expected)
    }

    #[test]
    fn merge_non_empty_distinct_metric_names() {
        let mut a = vec![create_ts(&[1000, 1200], &[2_f64, 1_f64])];

        a.get_mut(0).unwrap().metric_name.measurement = "bar".to_string();

        let mut b = vec![create_ts(
            &[1400, 1600, 1800, 2000],
            &[3_f64, 4_f64, 5_f64, 6_f64],
        )];

        b.get_mut(0).unwrap().metric_name.measurement = "foo".to_string();

        let MergeTestContext { ec, bstart } = setup_merge();
        let tss = merge_timeseries(a, b, bstart, &ec).expect("unable to merge timeseries");

        let mut foo = Timeseries::default();
        foo.metric_name.measurement = "foo".to_string();

        let mut bar = Timeseries::default();
        bar.metric_name.measurement = "bar".to_string();

        let tss_expected = vec![
            foo,
            create_ts(
                &[1000, 1200, 1400, 1600, 1800, 2000],
                &[NAN, NAN, 3_f64, 4_f64, 5_f64, 6_f64],
            ),
            bar,
            create_ts(
                &[1000, 1200, 1400, 1600, 1800, 2000],
                &[2_f64, 1_f64, NAN, NAN, NAN, NAN],
            ),
        ];

        test_timeseries_equal(&tss, &tss_expected)
    }
}
