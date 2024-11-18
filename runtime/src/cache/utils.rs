use crate::execution::EvalConfig;
use crate::types::{Timeseries, Timestamp};
use ahash::{AHashMap, AHashSet};
use tracing::{error, info, warn};
use metricsql_common::hash::Signature;


pub fn merge_series(
    // qt: &Arc<Mutex<QueryTracer>>,
    a: Vec<Timeseries>,
    b: Vec<Timeseries>,
    b_start: Timestamp,
    ec: &EvalConfig,
) -> (Vec<Timeseries>, bool) {
    // let qt = qt.lock().unwrap();
    // if qt.enabled() {
    //     info!(&format!(
    //         "merge series on time range {} with step={}ms; len(a)={}, len(b)={}, bStart={}",
    //         ec.time_range_string(), ec.step, a.len(), b.len(), b_start
    //     ));
    // }

    let shared_timestamps = ec.get_timestamps().unwrap(); // todo: handle error
    let mut i = 0;
    while i < shared_timestamps.len() && shared_timestamps[i] < b_start {
        i += 1;
    }
    let a_timestamps = &shared_timestamps[..i];
    let b_timestamps = &shared_timestamps[i..];

    let len_a = a.len();
    let mut b = b;
    if b_timestamps.len() == shared_timestamps.len() {
        for ts_b in b.iter_mut() {
            if *ts_b.timestamps != b_timestamps {
                error!("BUG: invalid timestamps in b series {}; got {:?}; want {:?}", ts_b.metric_name, ts_b.timestamps, b_timestamps);
                panic!("BUG: invalid timestamps in b series");
            }
            ts_b.timestamps = shared_timestamps.clone();
        }
        return (b, true);
    }

    let mut m_a: AHashMap<Signature, Timeseries> = AHashMap::with_capacity(a.len());
    for ts in a.into_iter() {
        if *ts.timestamps != a_timestamps {
            error!("BUG: invalid timestamps in a series {}; got {:?}; want {:?}", ts.metric_name, ts.timestamps, a_timestamps);
            panic!("BUG: invalid timestamps in a series");
        }
        let key = ts.signature();
        if m_a.contains_key(&key) {
            warn!("cannot merge series because a series contain duplicate {}", ts.metric_name);
            return (vec![], false);
        }
        m_a.insert(key, ts);
    }

    let mut m_b: AHashSet<Signature> = AHashSet::with_capacity(b.len());
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(len_a);
    let mut a_nans = Vec::new();
    for ts_b in b.into_iter() {
        if *ts_b.timestamps != b_timestamps {
            error!("BUG: invalid timestamps for b series {}; got {:?}; want {:?}", ts_b.metric_name, ts_b.timestamps, b_timestamps);
            panic!("BUG: invalid timestamps for b series");
        }

        let key = ts_b.signature();
        if m_b.contains(&key) {
            warn!("cannot merge series because b series contain duplicate {}", ts_b.metric_name);
            return (vec![], false);
        }
        m_b.insert(key);

        let mut tmp = Timeseries {
            timestamps: shared_timestamps.clone(),
            values: Vec::with_capacity(shared_timestamps.len()),
            metric_name: ts_b.metric_name.clone(),
        };

        if let Some(ts_a) = m_a.remove(&key) {
            tmp.values.extend_from_slice(&ts_a.values);
        } else {
            if a_nans.is_empty() {
                let mut t_start = ec.start;
                let step = ec.step.as_millis() as i64;
                while t_start < b_start {
                    a_nans.push(f64::NAN);
                    t_start += step;
                }
            }
            tmp.values.extend_from_slice(&a_nans);
        }
        tmp.values.extend_from_slice(&ts_b.values);
        rvs.push(tmp);
    }

    // Copy the remaining timeseries from mA.
    let mut b_nans = Vec::new();
    for (_, ts_a) in m_a.into_iter() {
        let mut tmp = Timeseries {
            timestamps: shared_timestamps.clone(),
            values: Vec::with_capacity(shared_timestamps.len()),
            metric_name: ts_a.metric_name,
        };
        tmp.values.extend_from_slice(&ts_a.values);

        if b_nans.is_empty() {
            let mut t_start = b_start;
            let step = ec.step.as_millis() as i64;
            while t_start <= ec.end {
                b_nans.push(f64::NAN);
                t_start += step;
            }
        }
        tmp.values.extend_from_slice(&b_nans);
        rvs.push(tmp);
    }

    info!("resulting series={}", rvs.len());
    (rvs, true)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;
    use super::*;
    use crate::types::MetricName;

    fn test_timeseries_equal(t: &Timeseries, t_expected: &Timeseries) {
        assert_eq!(t.timestamps, t_expected.timestamps);
        assert_eq!(t.values, t_expected.values);
        assert_eq!(t.metric_name, t_expected.metric_name);
    }

    #[test]
    fn test_merge_series() {
        let mut ec = EvalConfig::new(1000, 2000, Duration::from_millis(200));
        ec.max_points_per_series = 10000;

        // Test case: bStart=ec.Start
        let a = vec![];
        let b = vec![Timeseries {
            timestamps: Arc::new(vec![1000, 1200, 1400, 1600, 1800, 2000]),
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            metric_name: MetricName::default(),
        }];
        let (tss, ok) = merge_series(a, b, 1000, &ec);
        assert!(ok, "unexpected failure to merge series");
        let tss_expected = vec![Timeseries {
            timestamps: Arc::new(vec![1000, 1200, 1400, 1600, 1800, 2000]),
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            metric_name: MetricName::default(),
        }];
        for (t, t_expected) in tss.iter().zip(tss_expected.iter()) {
            test_timeseries_equal(t, t_expected);
        }

        // Test case: a-empty
        let a = vec![];
        let b = vec![Timeseries {
            timestamps: Arc::new(vec![1400, 1600, 1800, 2000]),
            values: vec![3.0, 4.0, 5.0, 6.0],
            metric_name: MetricName::default(),
        }];
        let (tss, ok) = merge_series(a, b, 1000, &ec);
        assert!(ok, "unexpected failure to merge series");
        let tss_expected = vec![Timeseries {
            timestamps: Arc::new(vec![1000, 1200, 1400, 1600, 1800, 2000]),
            values: vec![f64::NAN, f64::NAN, 3.0, 4.0, 5.0, 6.0],
            metric_name: MetricName::default(),
        }];
        for (t, t_expected) in tss.iter().zip(tss_expected.iter()) {
            test_timeseries_equal(t, t_expected);
        }

        // Test case: b-empty
        let a = vec![Timeseries {
            timestamps: Arc::new(vec![1000, 1200]),
            values: vec![2.0, 1.0],
            metric_name: MetricName::default(),
        }];
        let b = vec![];
        let (tss, ok) = merge_series(a, b, 1000, &ec);
        assert!(ok, "unexpected failure to merge series");
        let tss_expected = vec![Timeseries {
            timestamps: Arc::new(vec![1000, 1200, 1400, 1600, 1800, 2000]),
            values: vec![2.0, 1.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            metric_name: MetricName::default(),
        }];
        for (t, t_expected) in tss.iter().zip(tss_expected.iter()) {
            test_timeseries_equal(t, t_expected);
        }

        // Test case: non-empty
        let a = vec![Timeseries {
            timestamps: Arc::new(vec![1000, 1200]),
            values: vec![2.0, 1.0],
            metric_name: MetricName::default(),
        }];
        let b = vec![Timeseries {
            timestamps: Arc::new(vec![1400, 1600, 1800, 2000]),
            values: vec![3.0, 4.0, 5.0, 6.0],
            metric_name: MetricName::default(),
        }];

        let (tss, ok) = merge_series(a, b, 1000, &ec);
        assert!(ok, "unexpected failure to merge series");
        let tss_expected = vec![Timeseries {
            timestamps: Arc::new(vec![1000, 1200, 1400, 1600, 1800, 2000]),
            values: vec![2.0, 1.0, 3.0, 4.0, 5.0, 6.0],
            metric_name: MetricName::default(),
        }];
        for (t, t_expected) in tss.iter().zip(tss_expected.iter()) {
            test_timeseries_equal(t, t_expected);
        }

        // Test case: non-empty-distinct-metric-names
        let a = vec![Timeseries {
            timestamps: Arc::new(vec![1000, 1200]),
            values: vec![2.0, 1.0],
            metric_name: MetricName::new("bar"),
        }];
        let b = vec![Timeseries {
            timestamps: Arc::new(vec![1400, 1600, 1800, 2000]),
            values: vec![3.0, 4.0, 5.0, 6.0],
            metric_name: MetricName::new("foo"),
        }];

        let (tss, ok) = merge_series(a, b, 1000, &ec);
        assert!(ok, "unexpected failure to merge series");
        let tss_expected = vec![
            Timeseries {
                timestamps: Arc::new(vec![1000, 1200, 1400, 1600, 1800, 2000]),
                values: vec![f64::NAN, f64::NAN, 3.0, 4.0, 5.0, 6.0],
                metric_name: MetricName::new("foo"),
            },
            Timeseries {
                timestamps: Arc::new(vec![1000, 1200, 1400, 1600, 1800, 2000]),
                values: vec![2.0, 1.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
                metric_name: MetricName::new("bar"),
            },
        ];
        for (t, t_expected) in tss.iter().zip(tss_expected.iter()) {
            test_timeseries_equal(t, t_expected);
        }

        // Test case: duplicate-series-a
        let a = vec![
            Timeseries {
                timestamps: Arc::new(vec![1000, 1200]),
                values: vec![2.0, 1.0],
                metric_name: MetricName::default(),
            },
            Timeseries {
                timestamps: Arc::new(vec![1000, 1200]),
                values: vec![3.0, 3.0],
                metric_name: MetricName::default(),
            },
        ];
        let b = vec![Timeseries {
            timestamps: Arc::new(vec![1400, 1600, 1800, 2000]),
            values: vec![3.0, 4.0, 5.0, 6.0],
            metric_name: MetricName::default(),
        }];

        let (tss, ok) = merge_series(a, b, 1000, &ec);
        assert!(!ok, "expecting failure to merge series");
        assert!(tss.is_empty());

        // Test case: duplicate-series-b
        let a = vec![Timeseries {
            timestamps: Arc::new(vec![1000, 1200]),
            values: vec![1.0, 2.0],
            metric_name: MetricName::default(),
        }];
        let b = vec![
            Timeseries {
                timestamps: Arc::new(vec![1400, 1600, 1800, 2000]),
                values: vec![3.0, 4.0, 5.0, 6.0],
                metric_name: MetricName::default(),
            },
            Timeseries {
                timestamps: Arc::new(vec![1400, 1600, 1800, 2000]),
                values: vec![13.0, 14.0, 15.0, 16.0],
                metric_name: MetricName::default(),
            },
        ];

        let (tss, ok) = merge_series(a, b, 1000, &ec);
        assert!(!ok, "expecting failure to merge series");
        assert!(tss.is_empty());
    }
}