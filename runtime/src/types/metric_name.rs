use metricsql_common::hash::Signature;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::str::FromStr;

use crate::common::encoding::{read_string, read_usize, write_string, write_usize};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use ahash::{AHashMap, AHashSet};
use enquote::enquote;
use metricsql_common::prelude::Label;
use metricsql_parser::parser::{parse_metric_name, ParseError, ParseResult};
use metricsql_parser::prelude::{AggregateModifier, VectorMatchModifier};
use serde::{Deserialize, Serialize};

/// The maximum length of label name.
///
/// Longer names are truncated.
pub const MAX_LABEL_NAME_LEN: usize = 256;

pub const METRIC_NAME_LABEL: &str = "__name__";

// for tag manipulation (removing, adding, etc.), name vectors longer than this will be converted to a hashmap
// for comparison, otherwise we do a linear probe
const SET_SEARCH_MIN_THRESHOLD: usize = 16;


/// MetricName represents a metric name.
#[derive(Debug, PartialEq, Eq, Clone, Default, Hash, Serialize, Deserialize)]
pub struct MetricName {
    pub measurement: String,
    pub labels: Vec<Label>,
}

impl MetricName {
    pub fn new(name: &str) -> Self {
        MetricName {
            measurement: name.to_string(),
            labels: vec![],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.measurement.is_empty() && self.labels.is_empty()
    }

    /// from_strings creates new labels from pairs of strings.
    pub fn from_strings(ss: &[&str]) -> RuntimeResult<Self> {
        if ss.len() % 2 != 0 {
            return Err(RuntimeError::from("invalid number of strings"));
        }

        let mut res = MetricName::default();
        for i in (0..ss.len()).step_by(2) {
            res.set(ss[i], ss[i + 1]);
        }

        Ok(res)
    }

    pub fn parse(s: &str) -> ParseResult<Self> {
        let mut labels = parse_metric_name(s)?;
        let mut mn = MetricName::default();
        for label in labels.drain(..) {
            if label.name == METRIC_NAME_LABEL {
                mn.measurement = label.value;
            } else {
                mn.labels.push(label);
            }
        }
        mn.sort_labels();
        Ok(mn)
    }

    pub fn reset_measurement(&mut self) {
        self.measurement.clear();
    }

    pub fn copy_from(&mut self, other: &MetricName) {
        self.measurement.clone_from(&other.measurement);
        self.labels.clone_from(&other.labels);
    }

    /// Reset resets the mn.
    pub fn reset(&mut self) {
        self.measurement = "".to_string();
        self.labels.clear();
    }

    pub fn set_metric_group(&mut self, value: &str) {
        self.measurement = value.to_string();
    }

    /// adds new label to mn with the given key and value.
    pub fn add_label(&mut self, key: &str, value: &str) {
        if key == METRIC_NAME_LABEL {
            self.measurement = value.into();
            return;
        }
        self.upsert(key, value);
    }

    pub fn add_labels_from_hashmap(&mut self, tags: &HashMap<String, String>) {
        for (key, value) in tags {
            if key == METRIC_NAME_LABEL {
                self.measurement = value.into();
                return;
            }
            self.upsert(key, value);
        }
    }

    fn upsert(&mut self, key: &str, value: &str) {
        match self.labels.binary_search_by_key(&key, |tag| &tag.name) {
            Ok(idx) => {
                let tag = &mut self.labels[idx];
                tag.value.clear();
                tag.value.push_str(value);
            }
            Err(idx) => {
                let tag = Label {
                    name: key.to_string(),
                    value: value.to_string(),
                };
                self.labels.insert(idx, tag);
            }
        }
    }

    /// adds new tag to mn with the given key and value.
    pub fn set(&mut self, key: &str, value: &str) {
        if key == METRIC_NAME_LABEL {
            self.measurement = value.into();
        } else {
            self.upsert(key, value);
        }
    }

    /// removes a tag with the given tagKey
    pub fn remove_label(&mut self, key: &str) {
        if key == METRIC_NAME_LABEL {
            self.reset_measurement();
        } else {
            self.labels.retain(|x| x.name != key);
        }
    }

    pub fn has_label(&self, name: &str) -> bool {
        self.labels.iter().any(|x| x.name == name)
    }

    fn get_label_index(&self, name: &str) -> Option<usize> {
        if self.labels.is_empty() {
            return None;
        }
        if self.labels.len() < 8 {
            return self.labels.iter().position(|x| x.name == name);
        }
        self.labels.binary_search_by_key(&name, |label| &label.name).ok()
    }

    /// returns the value for a Label with the given name.
    pub fn label_value(&self, key: &str) -> Option<&String> {
        if key == METRIC_NAME_LABEL {
            return Some(&self.measurement);
        }
        if let Some(index) = self.get_label_index(key) {
            return Some(&self.labels[index].value);
        }
        None
    }

    #[allow(unused)]
    pub(crate) fn get_value_mut(&mut self, name: &str) -> Option<&mut String> {
        if name == METRIC_NAME_LABEL {
            return Some(&mut self.measurement);
        }
        if let Some(index) = self.get_label_index(name) {
            return Some(&mut self.labels[index].value);
        }
        None
    }

    /// removes all the tags not included in on_tags.
    /// don't stare too deeply. Just convince yourself that this is the correct behavior.
    /// https://github.com/VictoriaMetrics/VictoriaMetrics/blob/cde5029bcecac116b59e245330f6caf625e75eea/lib/storage/metric_name.go#L247
    pub fn remove_labels_on(&mut self, on_tags: &[String]) {
        if !on_tags.iter().any(|x| *x == METRIC_NAME_LABEL) {
            self.reset_measurement()
        }
        if on_tags.is_empty() {
            self.labels.clear();
            return;
        }
        if on_tags.len() > SET_SEARCH_MIN_THRESHOLD {
            let set: AHashSet<_> = AHashSet::from_iter(on_tags);
            self.labels.retain(|tag| set.contains(&tag.name));
        } else {
            self.labels.retain(|tag| on_tags.contains(&tag.name));
        }
    }

    /// remove_tags_ignoring removes all the tags included in ignoring_tags.
    pub fn remove_labels_ignoring(&mut self, ignoring_tags: &[String]) {
        self.remove_labels(ignoring_tags);
    }

    /// removes all the labels included in labels.
    pub fn remove_labels(&mut self, labels: &[String]) {
        if labels.is_empty() {
            return;
        }
        if labels.iter().any(|x| x.as_str() == METRIC_NAME_LABEL) {
            self.reset_measurement();
        }

        if labels.len() > SET_SEARCH_MIN_THRESHOLD {
            let set: AHashSet<_> = AHashSet::from_iter(labels);
            self.labels.retain(|tag| !set.contains(&tag.name));
        } else {
            self.labels.retain(|tag| !labels.contains(&tag.name));
        }
    }

    pub fn retain_labels(&mut self, tags: &[String]) {
        if !tags.iter().any(|x| *x == METRIC_NAME_LABEL) {
            self.reset_measurement()
        }
        if tags.is_empty() {
            self.labels.clear();
            return;
        }
        if tags.len() >= SET_SEARCH_MIN_THRESHOLD {
            let set: AHashSet<_> = AHashSet::from_iter(tags);
            self.labels.retain(|tag| set.contains(&tag.name));
        } else {
            self.labels.retain(|tag| tags.contains(&tag.name));
        }
    }

    /// sets labels from src with keys matching add_tags.
    pub(crate) fn set_labels(
        &mut self,
        prefix: &str,
        add_labels: &[String],
        skip_labels: &[String],
        src: &mut MetricName,
    ) {
        if add_labels.len() == 1 && add_labels[0] == "*" {
            // Special case for copying all the tags except of skipTags from src to mn.
            self.set_all_labels(prefix, skip_labels, src);
            return;
        }

        for tag_name in add_labels {
            if skip_labels.contains(tag_name) {
                continue;
            }

            // todo: use iterators instead
            match src.label_value(tag_name) {
                Some(tag_value) => {
                    if !prefix.is_empty() {
                        let key = format!("{prefix}{}", tag_name);
                        self.set(&key, tag_value);
                    } else {
                        self.set(tag_name, tag_value);
                    }
                }
                None => {
                    self.remove_label(tag_name);
                }
            }
        }
    }

    fn set_all_labels(&mut self, prefix: &str, skip_tags: &[String], src: &MetricName) {
        for tag in src.labels.iter() {
            if skip_tags.contains(&tag.name) {
                continue;
            }
            if !prefix.is_empty() {
                let key = format!("{prefix}{}", tag.name);
                self.set(&key, &tag.value);
            } else {
                self.set(&tag.name, &tag.value);
            }
        }
    }

    pub fn append_labels_to_string(&self, dst: &mut Vec<u8>) {
        dst.extend_from_slice("{{".as_bytes());

        let len = &self.labels.len();
        for (i, Label { name: k, value: v }) in self.labels.iter().enumerate() {
            dst.extend_from_slice(format!("{}={}", k, enquote('"', v)).as_bytes());
            if i + 1 < *len {
                dst.extend_from_slice(", ".as_bytes())
            }
        }
        // ??????????????
        dst.extend_from_slice('}'.to_string().as_bytes());
    }

    pub(crate) fn marshal_labels_fast(&self, dst: &mut Vec<u8>) {
        // Calculate the required size and pre-allocate space in dst
        let required_size = self
            .labels
            .iter()
            .fold(0, |acc, tag| acc + tag.name.len() + tag.value.len() + 8);
        dst.reserve(required_size + 4);
        write_usize(dst, self.labels.len());
        for tag in self.labels.iter() {
            tag.marshal(dst);
        }
    }

    fn unmarshal_labels<'a>(&mut self, src: &'a [u8]) -> RuntimeResult<&'a [u8]> {
        let (mut src, len) = read_usize(src, "tag count")?;
        self.labels = Vec::with_capacity(len);
        for _ in 0..len {
            let (label, new_src) = Label::unmarshal(src);
            if label.name.is_empty() {
                return Err(RuntimeError::from("empty label name"));
            }
            self.labels.push(label);
            src = new_src;
        }
        Ok(src)
    }

    /// marshal appends marshaled mn to dst.
    ///
    /// `self.sort_labels` must be called before calling this function
    /// in order to sort and de-duplicate labels.
    pub fn marshal(&self, dst: &mut Vec<u8>) {
        // Calculate the required size and pre-allocate space in dst
        let required_size = self.measurement.len() + 8;
        dst.reserve(required_size);

        write_string(dst, &self.measurement);
        self.marshal_labels_fast(dst);
    }

    /// unmarshals mn from src.
    /// Todo(perf) this is not necessarily as performant as can be, even for
    /// simple serialization
    pub fn unmarshal(src: &[u8]) -> RuntimeResult<(&[u8], MetricName)> {
        let mut mn = MetricName::default();
        let (mut src, group) = read_string(src, "metric group")?;

        mn.measurement = group;
        src = mn.unmarshal_labels(src)?;
        Ok((src, mn))
    }

    pub(crate) fn serialized_size(&self) -> usize {
        let mut n = 2 + self.measurement.len();
        n += 2; // Length of labels.
        for Label { name: k, value: v } in self.labels.iter() {
            n += 2 + k.len();
            n += 2 + v.len();
        }
        n
    }

    pub fn remove_group_labels(&mut self, modifier: &Option<AggregateModifier>) {
        if let Some(m) = modifier {
            match m {
                AggregateModifier::By(labels) => {
                    // we're grouping by `labels, so keep only those
                    self.retain_labels(labels);
                }
                AggregateModifier::Without(labels) => {
                    self.remove_labels(labels);
                    // Reset metric group as Prometheus does on `aggr(...) without (...)` call.
                    self.reset_measurement();
                }
            }
        } else {
            // No grouping. Remove all labels.
            self.remove_labels_on(&[]);
        };
    }

    pub(crate) fn count_label_values(&self, hm: &mut AHashMap<String, AHashMap<String, usize>>) {
        // duplication, I know
        let label_counts = hm.entry(METRIC_NAME_LABEL.to_string()).or_default();
        *label_counts
            .entry(self.measurement.to_string())
            .or_insert(0) += 1;
        for label in self.labels.iter() {
            count_label_value(hm, &label.name, &label.value);
        }
    }

    pub(crate) fn signature(&self) -> Signature {
        Signature::from_name_and_labels(&self.measurement, self.labels.iter())
    }

    pub fn sort_labels(&mut self) {
        if self.labels.len() > 1 {
            self.labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
    }

    /// generate a Signature using tags only (excluding the measurement name)
    pub fn labels_signature(&self) -> Signature {
        Signature::from_vec(&self.labels)
    }

    /// Compute a signature hash using only the tags passed in `labels`. `metric_group` is ignored.
    pub fn tags_signature_with_labels(&self, labels: &[String]) -> Signature {
        let empty: &str = "";
        self.signature_with_labels_internal(empty, labels)
    }

    /// Compute a signature hash using only the tags passed in `labels`.
    pub fn signature_with_labels(&self, labels: &[String]) -> Signature {
        self.signature_with_labels_internal(&self.measurement, labels)
    }

    fn signature_with_labels_internal(&self, name: &str, labels: &[String]) -> Signature {
        let iter = self.labels.iter().filter(|tag| labels.contains(&tag.name));
        Signature::from_name_and_labels(name, iter)
    }

    /// Compute a signature hash ignoring the passed in labels.
    pub fn signature_without_labels(&self, labels: &[String]) -> Signature {
        self.signature_with_labels_internal(&self.measurement, labels)
    }

    /// Compute a signature hash from tags ignoring the passed in labels.
    /// The `metric_group` is ignored.
    pub fn tags_signature_without_labels(&self, labels: &[String]) -> Signature {
        self.signature_without_labels_internal("", labels)
    }

    fn signature_without_labels_internal(&self, name: &str, labels: &[String]) -> Signature {
        if labels.is_empty() {
            let iter = self.labels.iter();
            return Signature::from_name_and_labels(name, iter);
        }
        let includes_metric_group = labels.iter().any(|x| *x == METRIC_NAME_LABEL);
        let group_name = if includes_metric_group { name } else { "" };
        let iter = self.labels.iter().filter(|tag| !labels.contains(&tag.name));
        Signature::from_name_and_labels(group_name, iter)
    }

    pub(crate) fn get_hash_signature(&self, modifier: &Option<VectorMatchModifier>, keep_metric_name: bool) -> Signature {
        match modifier {
            None => {
                if keep_metric_name {
                    Signature::from_name_and_labels(&self.measurement, self.labels.iter())
                } else {
                    Signature::from_name_and_labels("", self.labels.iter())
                }
            },
            Some(m) => match m {
                VectorMatchModifier::On(on_tags) => {
                    // removes all the tags not included to on_tags.
                    let keep_names = if !on_tags.contains(METRIC_NAME_LABEL) {
                        false
                    } else {
                        keep_metric_name
                    };
                    signature_with_labels(self, on_tags.as_ref(), keep_names)
                },
                VectorMatchModifier::Ignoring(labels) => {
                    signature_without_labels(self, labels.as_ref(), keep_metric_name)
                }
            },
        }
    }

    pub(crate) fn get_aggregate_hash_signature(&self, modifier: &Option<AggregateModifier>) -> Signature {
        match modifier {
            None => Signature::from_name_and_labels(&self.measurement, self.labels.iter()),
            Some(AggregateModifier::By(by_tags)) => {
                let keep_name = by_tags.iter().find(|x| *x == METRIC_NAME_LABEL).is_some();
                signature_with_labels(self, by_tags, keep_name)
            },
            Some(AggregateModifier::Without(labels)) => {
                // reset metric group as Prometheus does on `aggr(...) without (...)` call.
                signature_without_labels(self, labels, false)
            }
        }
    }
}

fn signature_without_labels(mn: &MetricName, labels: &[String], keep_metric_name: bool) -> Signature {
    let group_name = if keep_metric_name {
        &mn.measurement
    } else {
        ""
    };
    if labels.is_empty() {
        let iter = mn.labels.iter();
        return Signature::from_name_and_labels(group_name, iter);
    }
    let iter = mn.labels.iter().filter(|tag| !labels.contains(&tag.name));
    Signature::from_name_and_labels(group_name, iter)
}

fn signature_with_labels(mn: &MetricName, labels: &[String], keep_metric_name: bool) -> Signature {
    let group_name = if keep_metric_name {
        &mn.measurement
    } else {
        ""
    };
    let iter = mn.labels.iter().filter(|tag| labels.contains(&tag.name));
    Signature::from_name_and_labels(group_name, iter)
}


impl FromStr for MetricName {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        MetricName::parse(s)
    }
}

impl TryFrom<&str> for MetricName {
    type Error = ParseError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        MetricName::parse(s)
    }
}

impl TryFrom<String> for MetricName {
    type Error = ParseError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        MetricName::parse(&s)
    }
}

fn count_label_value(
    hm: &mut AHashMap<String, AHashMap<String, usize>>,
    label: &String,
    value: &String,
) {
    let label_counts = hm.entry(label.to_string()).or_default();
    *label_counts.entry(value.to_string()).or_insert(0) += 1;
}

impl Display for MetricName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{{", self.measurement)?;
        let len = self.labels.len();
        for (i, Label { name: k, value: v }) in self.labels.iter().enumerate() {
            write!(f, "{}={}", k, enquote('"', v))?;
            if i < len - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "}}")?;
        Ok(())
    }
}

impl PartialOrd for MetricName {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl Ord for MetricName {
    fn cmp(&self, other: &Self) -> Ordering {
        let order = self.measurement.cmp(&other.measurement);
        if order != Ordering::Equal {
            return order;
        }
        // Metric names for a and b match. Compare tags.
        // Tags must be already sorted by the caller, so just compare them.
        let ats = &self.labels;
        let bts = &other.labels;

        let other_len = bts.len();

        for (i, a) in ats.iter().enumerate() {
            if i >= other_len {
                // self contains more tags than other and all the previous tags were identical,
                // so self is considered bigger than other.
                return Ordering::Greater;
            }
            let bt = &bts[i];
            let ordering = a.cmp(bt);
            if ordering != Ordering::Equal {
                return ordering;
            }
        }

        self.labels.len().cmp(&other.labels.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_name() {
        let mut mn = MetricName::default();
        mn.set_metric_group("foo");
        mn.add_label("bar", "baz");
        mn.add_label("qux", "quux");
        mn.add_label("qux", "quuz");
        mn.add_label("corge", "grault");
        mn.add_label("garply", "waldo");
        mn.add_label("fred", "plugh");
        mn.add_label("xyzzy", "thud");
        mn.add_label("xyzzy", "thud");
        mn.add_label("xyzzy", "thud");
        assert_eq!(mn.measurement, "foo");
        assert_eq!(mn.labels.len(), 6);
        assert_eq!(mn.labels[0].name, "bar");
        assert_eq!(mn.labels[0].value, "baz");

        let mut prev = &mn.labels[0];
        for curr in mn.labels.iter().skip(1) {
            assert!(prev.name < curr.name, "labels are not sorted");
            prev = curr;
        }

        assert_eq!(mn.label_value("qux"), Some(&String::from("quuz")));
        assert_eq!(mn.label_value("xyzzy"), Some(&String::from("thud")));
    }

    #[test]
    fn test_add_tag() {
        let mut mn = MetricName::default();
        mn.add_label("foo", "bar");
        assert_eq!(mn.labels.len(), 1);
        assert_eq!(mn.labels[0].name, "foo");
        assert_eq!(mn.labels[0].value, "bar");

        // replace value if key exists already
        mn.add_label("foo", "baz");
        assert_eq!(mn.labels.len(), 1);
        assert_eq!(mn.labels[0].name, "foo");
        assert_eq!(mn.labels[0].value, "baz");

        // ensure sort order is maintained
        mn.add_label("bar", "baz");
        assert_eq!(mn.labels.len(), 2);
        assert_eq!(mn.labels[0].name, "bar");
        assert_eq!(mn.labels[1].name, "foo");
    }

    #[test]
    fn test_duplicate_keys() {
        let mut mn = MetricName::default();
        mn.measurement = "xxx".to_string();
        mn.add_label("foo", "bar");
        mn.add_label("duplicate", "tag1");
        mn.add_label("duplicate", "tag2");
        mn.add_label("tt", "xx");
        mn.add_label("foo", "abc");
        mn.add_label("duplicate", "tag3");

        let mut mn_expected = MetricName::default();
        mn_expected.measurement = "xxx".to_string();
        mn_expected.add_label("duplicate", "tag3");
        mn_expected.add_label("foo", "abc");
        mn_expected.add_label("tt", "xx");

        assert_eq!(mn, mn_expected);
    }

    #[test]
    fn test_remove_tags_on() {
        let mut empty_mn = MetricName::default();
        empty_mn.measurement = "name".to_string();
        empty_mn.add_label("key", "value");
        empty_mn.remove_labels_on(&vec![]);
        assert!(
            empty_mn.measurement.is_empty() && empty_mn.labels.is_empty(),
            "expecting empty metric name got {}",
            &empty_mn
        );

        let mut as_is_mn = MetricName::default();
        as_is_mn.measurement = "name".to_string();
        as_is_mn.add_label("key", "value");
        let tags = vec![METRIC_NAME_LABEL.to_string(), "key".to_string()];
        as_is_mn.remove_labels_on(&tags);
        let mut exp_as_is_mn = MetricName::default();
        exp_as_is_mn.measurement = "name".to_string();
        exp_as_is_mn.add_label("key", "value");
        assert_eq!(
            exp_as_is_mn, as_is_mn,
            "expecting {} got {}",
            &exp_as_is_mn, &as_is_mn
        );

        let mut mn = MetricName::default();
        mn.measurement = "name".to_string();
        mn.add_label("foo", "bar");
        mn.add_label("baz", "qux");
        let tags = vec!["baz".to_string()];
        mn.remove_labels_on(&tags);
        let mut exp_mn = MetricName::default();
        exp_mn.add_label("baz", "qux");
        assert_eq!(exp_mn, mn, "expecting {} got {}", &exp_mn, &mn);
    }

    #[test]
    fn test_remove_tag() {
        let mut mn = MetricName::default();
        mn.measurement = "name".to_string();
        mn.add_label("foo", "bar");
        mn.add_label("baz", "qux");
        mn.remove_label("__name__");
        assert!(
            mn.measurement.is_empty(),
            "expecting empty metric group name got {}",
            &mn
        );
        mn.remove_label("foo");
        let mut exp_mn = MetricName::default();
        exp_mn.add_label("baz", "qux");
        assert_eq!(exp_mn, mn, "expecting {} got {}", &exp_mn, &mn);
    }

    #[test]
    fn test_remove_tags_ignoring() {
        let mut mn = MetricName::default();
        mn.measurement = "name".to_string();
        mn.add_label("foo", "bar");
        mn.add_label("baz", "qux");
        let tags = vec![METRIC_NAME_LABEL.to_string(), "foo".to_string()];
        mn.remove_labels_ignoring(&tags);
        let mut exp_mn = MetricName::default();
        exp_mn.add_label("baz", "qux");
        assert_eq!(exp_mn, mn, "expecting {} got {}", &exp_mn, &mn);
    }

    #[test]
    fn test_tags_signature_with_labels() {
        let mut mn = MetricName::new("name");
        mn.add_label("le", "8.799e1");
        mn.add_label("foo", "bar");
        mn.add_label("baz", "qux");
        let mut exp_mn = MetricName::default();
        exp_mn.add_label("baz", "qux");
        let actual = mn.tags_signature_with_labels(&vec!["baz".to_string()]);
        let expected = exp_mn.tags_signature_with_labels(&vec!["baz".to_string()]);
        assert_eq!(
            actual, expected,
            "expecting {:?} got {:?}",
            expected, actual
        );
    }
}
