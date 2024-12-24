use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use metricsql_common::hash::Signature;
use metricsql_parser::prelude::{Matcher, Matchers};

use crate::{
    Deadline, MetricStorage, QueryResult, QueryResults, RuntimeResult, SearchQuery,
};
use crate::types::{MetricName};

#[derive(Debug, Clone)]
pub struct Point {
    t: i64,
    v: f64,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub metric: MetricName,
    pub timestamp: i64,
    pub value: f64,
}

/// In-memory implementation of MetricDataProvider primarily for testing
#[derive(Default, Debug)]
pub struct MemoryMetricProvider {
    inner: RwLock<Storage>,
}

#[derive(Default, Debug, Clone)]
struct Storage {
    labels_hash: BTreeMap<Signature, Arc<MetricName>>,
    sample_values: BTreeMap<Signature, Vec<Point>>,
}

impl Storage {
    fn append(&mut self, labels: MetricName, t: i64, v: f64) -> RuntimeResult<()> {
        let h = labels.signature();
        match self.labels_hash.entry(h) {
            Entry::Vacant(e) => {
                let metric = Arc::new(labels);
                e.insert(metric);
                self.sample_values.insert(h, vec![Point { t, v }]);
            }
            Entry::Occupied(_) => {
                let values = self.sample_values.entry(h).or_default();
                values.push(Point { t, v });
            }
        }
        Ok(())
    }

    pub fn search(&self, start: i64, end: i64, filters: &Matchers) -> RuntimeResult<QueryResults> {
        let mut results: Vec<QueryResult> = vec![];
        // todo: handle or
        for (k, labels) in &self.labels_hash {
            for matchers in filters.iter() {
                if matches_filters(labels, matchers) {
                    if let Some(res) = self.get_range(*k, start, end) {
                        results.push(res)
                    }
                }
            }
        }

        Ok(QueryResults::new(results))
    }

    fn get_range(&self, metric_id: Signature, start: i64, end: i64) -> Option<QueryResult> {
        if let Some(values) = self.sample_values.get(&metric_id) {
            if let Some(start) = find_first_index(values, start) {
                let mut points = &mut values[start..]
                    .iter()
                    .filter(|p| p.t <= end)
                    .collect::<Vec<_>>();

                points.sort_by(|a, b| a.t.cmp(&b.t));

                let mut timestamps = Vec::with_capacity(points.len());
                let mut values = Vec::with_capacity(points.len());

                for point in points {
                    timestamps.push(point.t);
                    values.push(point.v);
                }

                let metric_name = if let Some(mn) = self.labels_hash.get(&metric_id) {
                    let copy: MetricName = mn.deref().clone();
                    copy
                } else {
                    MetricName::default()
                };

                return Some(QueryResult {
                    metric: metric_name,
                    values,
                    timestamps,
                });
            }
        }
        None
    }

    fn clear(&mut self) {
        self.labels_hash.clear();
        self.sample_values.clear();
    }
}

impl MemoryMetricProvider {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Storage {
                labels_hash: Default::default(),
                sample_values: Default::default(),
            }),
        }
    }

    pub fn append(&self, labels: MetricName, t: i64, v: f64) -> RuntimeResult<()> {
        let mut inner = self.inner.write().unwrap();
        inner.append(labels, t, v)
    }

    pub fn add_sample(&self, sample: Sample) -> RuntimeResult<()> {
        let mut inner = self.inner.write().unwrap();
        inner.append(sample.metric, sample.timestamp, sample.value)
    }

    pub fn clear(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.clear();
    }

    fn search_internal(
        &self,
        start: i64,
        end: i64,
        filters: &Matchers,
    ) -> RuntimeResult<QueryResults> {
        let inner = self.inner.read().unwrap();
        inner.search(start, end, filters)
    }

    pub fn search(&self, start: i64, end: i64, filters: &Matchers) -> RuntimeResult<QueryResults> {
        self.search_internal(start, end, filters)
    }
}

#[async_trait]
impl MetricStorage for MemoryMetricProvider {
    async fn search(&self, sq: SearchQuery, _deadline: Deadline) -> RuntimeResult<QueryResults> {
        self.search_internal(sq.start, sq.end, &sq.matchers)
    }
}

fn matches_filter(mn: &MetricName, filter: &Matcher) -> bool {
    if let Some(v) = mn.label_value(filter.label.as_str()) {
        return filter.matches(v);
    }
    false
}

fn matches_filters(mn: &MetricName, filters: &[Matcher]) -> bool {
    filters.iter().all(|f| matches_filter(mn, f))
}

fn find_first_index(range_values: &[Point], ts: i64) -> Option<usize> {
    // Find the index of the first item where `range.start <= key`.
    match range_values.binary_search_by_key(&ts, |point| point.t) {
        Ok(index) => Some(index),

        // If the requested key is smaller than the smallest range in the slice,
        // we would be computing `0 - 1`, which would underflow an `usize`.
        // We use `checked_sub` to get `None` instead.
        Err(index) => index.checked_sub(1),
    }
}

#[cfg(test)]
mod tests {
    use crate::types::MetricName;

    use super::*;

    #[test]
    fn append_new_metric_creates_new_entry() {
        let provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();

        let inner = provider.inner.read().unwrap();
        assert!(inner.labels_hash.contains_key(&labels.signature()));
    }

    #[test]
    fn append_existing_metric_adds_point() {
        let provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();
        provider.append(labels.clone(), 2, 2.0).unwrap();

        let inner = provider.inner.read().unwrap();
        assert_eq!(
            inner.sample_values.get(&labels.signature()).unwrap().len(),
            2
        );
    }

    #[test]
    fn search_returns_matching_metrics() {
        let provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();

        let matchers = Matchers::new(vec![Matcher::equal("foo", "bar")]);
        let results = provider.search(0, 2, &matchers).unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_returns_empty_for_no_match() {
        let provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();

        let matchers = Matchers::new(vec![Matcher::equal("foo", "baz")]);
        let results = provider.search(0, 2, &matchers).unwrap();

        assert_eq!(results.len(), 0);
    }
}
