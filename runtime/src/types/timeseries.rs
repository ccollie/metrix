use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;
use ahash::AHashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use metricsql_common::hash::Signature;
use metricsql_common::prelude::humanize_duration;
use metricsql_parser::ast::VectorMatchModifier;

use crate::runtime_error::{RuntimeError, RuntimeResult};
use super::{MetricName, Timestamp};

pub type TimeseriesHashMap = AHashMap<Signature, Vec<Timeseries>>;
pub type TimeseriesHashMapRef<'a> = AHashMap<Signature, &'a [Timeseries]>;

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Timeseries {
    pub metric_name: MetricName,
    pub values: Vec<f64>,
    pub timestamps: Arc<Vec<Timestamp>>, //Arc used vs Rc since Rc is !Send
}

impl Timeseries {
    pub fn new(timestamps: Vec<Timestamp>, values: Vec<f64>) -> Self {
        Timeseries {
            metric_name: MetricName::default(),
            values,
            timestamps: Arc::new(timestamps),
        }
    }

    pub fn with_shared_timestamps(timestamps: &Arc<Vec<Timestamp>>, values: &[f64]) -> Self {
        Timeseries {
            metric_name: MetricName::default(),
            values: Vec::from(values),
            // see https://pkolaczk.github.io/server-slower-than-a-laptop/ under the section #the fix
            timestamps: Arc::new(timestamps.as_ref().clone()), // clones the value under Arc and wraps it in a new counter
        }
    }

    pub fn reset(&mut self) {
        self.values.clear();
        self.timestamps = Arc::new(vec![]);
        self.metric_name.reset();
    }

    pub fn is_all_nans(&self) -> bool {
        self.values.iter().all(|v| v.is_nan())
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[inline]
    pub fn tag_count(&self) -> usize {
        self.metric_name.labels.len()
    }

    pub fn signature(&self) -> Signature {
        self.metric_name.signature()
    }
}

pub(crate) struct SeriesSlice<'a> {
    pub metric_name: &'a MetricName,
    pub timestamps: &'a [Timestamp],
    pub values: &'a [f64],
}

impl<'a> SeriesSlice<'a> {
    pub fn new(metric_name: &'a MetricName, timestamps: &'a [Timestamp], values: &'a [f64]) -> Self {
        SeriesSlice {
            metric_name,
            timestamps,
            values,
        }
    }

    pub fn from_timeseries(ts: &'a Timeseries, range: Option<(usize, usize)>) -> Self {
        if let Some((start, end)) = range {
            SeriesSlice {
                metric_name: &ts.metric_name,
                timestamps: &ts.timestamps[start..end],
                values: &ts.values[start..end],
            }
        } else {
            SeriesSlice {
                metric_name: &ts.metric_name,
                timestamps: &ts.timestamps,
                values: &ts.values,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }
}

pub(crate) fn assert_identical_timestamps(tss: &[Timeseries], step: Duration) -> RuntimeResult<()> {
    if tss.is_empty() {
        return Ok(());
    }
    let ts_golden = &tss[0];
    if ts_golden.values.len() != ts_golden.timestamps.len() {
        let msg = format!(
            "BUG: ts_golden.values.len() must match ts_golden.timestamps.len(); got {} vs {}",
            ts_golden.values.len(),
            ts_golden.timestamps.len()
        );
        return Err(RuntimeError::from(msg));
    }
    if ts_golden.timestamps.len() > 0 {
        let mut prev_timestamp = ts_golden.timestamps[0];
        let step_ms = step.as_millis() as i64;
        for timestamp in ts_golden.timestamps.iter().skip(1) {
            if timestamp - prev_timestamp != step_ms {
                let msg = format!(
                    "BUG: invalid step between timestamps; got {}; want {};",
                    timestamp - prev_timestamp,
                    humanize_duration(&step)
                );
                return Err(RuntimeError::from(msg));
            }
            prev_timestamp = *timestamp
        }
    }
    for ts in tss.iter() {
        if ts.values.len() != ts_golden.values.len() {
            let msg = format!(
                "BUG: unexpected ts.values.len(); got {}; want {}; ts.values={}",
                ts.values.len(),
                ts_golden.values.len(),
                ts.values.len()
            );
            return Err(RuntimeError::from(msg));
        }
        if ts.timestamps.len() != ts_golden.timestamps.len() {
            let msg = format!(
                "BUG: unexpected ts.timestamps.len(); got {}; want {};",
                ts.timestamps.len(),
                ts_golden.timestamps.len()
            );
            return Err(RuntimeError::from(msg));
        }
        if ts.timestamps.len() == 0 {
            continue;
        }
        if ts.timestamps[0] == ts_golden.timestamps[0] {
            // Fast path - shared timestamps.
            continue;
        }
        for (ts, golden) in ts.timestamps.iter().zip(ts_golden.timestamps.iter()) {
            if ts != golden {
                let msg = format!("BUG: timestamps mismatch; got {}; want {};", ts, golden);
                return Err(RuntimeError::from(msg));
            }
        }
    }
    Ok(())
}

fn assert_identical_timestamps_internal<'a>(
    ts_iter: &mut impl Iterator<Item = &'a [Timestamp]>,
    values_iter: &mut impl Iterator<Item = &'a [f64]>,
    step: i64,
) -> RuntimeResult<()> {
    let ts_golden = ts_iter.next().unwrap_or_default();
    let values_golden = values_iter.next().unwrap_or_default();
    if values_golden.len() != ts_golden.len() {
        let msg = format!(
            "BUG: ts_golden.values.len() must match ts_golden.timestamps.len(); got {} vs {}",
            values_golden.len(),
            ts_golden.len()
        );
        return Err(RuntimeError::from(msg));
    }
    if !ts_golden.is_empty() {
        let mut prev_timestamp = ts_golden[0];
        for timestamp in ts_golden.iter() {
            if timestamp - prev_timestamp != step {
                let msg = format!(
                    "BUG: invalid step between timestamps; got {}; want {};",
                    timestamp - prev_timestamp,
                    step
                );
                return Err(RuntimeError::from(msg));
            }
            prev_timestamp = *timestamp
        }
    }
    for (ts, values) in ts_iter.zip(values_iter) {
        if values.len() != values_golden.len() {
            let msg = format!(
                "BUG: unexpected ts.values.len(); got {}; want {}; ts.values={}",
                values.len(),
                values_golden.len(),
                values.len()
            );
            return Err(RuntimeError::from(msg));
        }
        if ts.len() != ts_golden.len() {
            let msg = format!(
                "BUG: unexpected ts.timestamps.len(); got {}; want {};",
                ts.len(),
                ts_golden.len()
            );
            return Err(RuntimeError::from(msg));
        }
        if ts.is_empty() {
            continue;
        }
        if ts[0] == ts_golden[0] {
            // Fast path - shared timestamps.
            continue;
        }
        for (ts, golden) in ts.iter().zip(ts_golden.iter()) {
            if ts != golden {
                let msg = format!("BUG: timestamps mismatch; got {}; want {};", ts, golden);
                return Err(RuntimeError::from(msg));
            }
        }
    }

    Ok(())
}

pub(crate) fn get_timeseries() -> Timeseries {
    // timeseries_pool().pull().timeseries.borrow_mut()
    Timeseries::default()
}

/// The minimum threshold of timeseries tags to process in parallel when computing signatures.
pub(crate) const SIGNATURE_PARALLELIZATION_THRESHOLD: usize = 8;

pub fn group_series_by_match_modifier(
    series: &mut Vec<Timeseries>,
    modifier: &Option<VectorMatchModifier>,
    with_metric_name: bool,
) -> TimeseriesHashMap {
    let sigs: Vec<Signature> = if series.len() >= SIGNATURE_PARALLELIZATION_THRESHOLD {
        series
            .par_iter()
            .map_with(modifier, |modifier, timeseries| {
                if with_metric_name {
                    timeseries.metric_name.signature_by_match_modifier(modifier)
                } else {
                    timeseries
                        .metric_name
                        .tags_signature_by_match_modifier(modifier)
                }
            })
            .collect::<Vec<Signature>>()
    } else {
        series
            .iter()
            .map(|timeseries| {
                if with_metric_name {
                    timeseries.metric_name.signature_by_match_modifier(modifier)
                } else {
                    timeseries
                        .metric_name
                        .tags_signature_by_match_modifier(modifier)
                }
            })
            .collect::<Vec<Signature>>()
    };

    let mut m: TimeseriesHashMap = AHashMap::with_capacity(series.len());
    for (ts, sig) in series.iter_mut().zip(sigs.iter()) {
        m.entry(*sig).or_default().push(std::mem::take(ts));
    }

    m
}
