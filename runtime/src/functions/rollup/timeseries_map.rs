use std::collections::hash_map::ValuesMut;
use std::collections::HashMap;
use std::iter;
use std::sync::Arc;

use metricsql::functions::RollupFunction;

use crate::{MetricName, Timeseries};
use crate::histogram::{Histogram, NonZeroBuckets};

#[derive(Clone)]
pub(crate) struct TimeseriesMap {
    origin: Timeseries,
    hist: Histogram,
    pub(crate) series: HashMap<String, Timeseries>
}

impl TimeseriesMap {
    pub fn new(
        func: &RollupFunction,
        keep_metric_names: bool,
        shared_timestamps: &Arc<Vec<i64>>,
        mn_src: &MetricName) -> Option<TimeseriesMap> {

        if !TimeseriesMap::is_eligible_function(func) {
            return None;
        }

        let ts_len = shared_timestamps.len();
        let mut values: Vec<f64> = Vec::with_capacity(shared_timestamps.len());
        values.extend(iter::repeat(f64::NAN).take(ts_len));

        let mut origin: Timeseries = Timeseries::default();
        origin.metric_name.copy_from(mn_src);
        if !keep_metric_names {
            origin.metric_name.reset_metric_group()
        }
        origin.timestamps = shared_timestamps.clone();
        origin.values = values;
        let m: HashMap<String, Timeseries> = HashMap::new();

        Some(TimeseriesMap {
            origin,
            hist: Histogram::new(),
            series: m
        })
    }

    pub fn is_eligible_function(func: &RollupFunction) -> bool {
        *func == RollupFunction::HistogramOverTime || *func == RollupFunction::QuantilesOverTime
    }

    pub fn update(&mut self, value: f64) {
        self.hist.update(value);
    }

    pub fn get_or_create_timeseries(&mut self, label_name: &str, label_value: &str) -> &mut Timeseries {
        let value = label_value.to_string();
        let timestamps = &self.origin.timestamps;
        self.series.entry(value)
            .or_insert_with_key(move |value| {
                let values: Vec<f64> = Vec::with_capacity(1);
                let mut ts = Timeseries::with_shared_timestamps(&timestamps, &values);
                ts.metric_name.remove_tag(label_name);
                ts.metric_name.add_tag(label_name, &value);
                ts
            })
    }

    /// Copy all timeseries to dst. The map should not be used after this call
    pub fn append_timeseries_to(&mut self, dst: &mut Vec<Timeseries>) {
        for (_, mut ts) in self.series.drain() {
            dst.push(std::mem::take(&mut ts))
        }
    }
    
    pub(crate) fn reset(&mut self) {
        self.hist.reset();
    }

    pub(crate) fn values_mut(&mut self) -> ValuesMut<'_, String, Timeseries> {
        self.series.values_mut()
    }

    pub fn non_zero_buckets(&mut self) -> NonZeroBuckets {
        self.hist.non_zero_buckets()
    }
}

pub(crate) fn reset_timeseries_map(map: &mut TimeseriesMap) {
    map.hist.reset()
}
