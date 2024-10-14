use metricsql_common::prelude::{Label, StringMatchHandler};
use std::cmp::Ordering;
use std::fmt;
use std::ops::Deref;

pub type LabelFilterOp = metricsql_parser::label::LabelFilterOp;


/// labelFilter contains PromQL filter for `{label op "value"}`
#[derive(Debug, Clone)]
pub struct LabelFilter {
    pub label: String,
    pub op: LabelFilterOp,
    // todo: enum
    pub value: String,
    // re contains compiled regexp for `=~` and `!~` op.
    pub re: Option<StringMatchHandler>,
}

impl PartialEq<LabelFilter> for LabelFilter {
    fn eq(&self, other: &Self) -> bool {
        self.op.eq(&other.op) && self.label.eq(&other.label) && self.value.eq(&other.value)
    }
}


impl PartialOrd for LabelFilter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.label != other.label {
            return Some(self.label.cmp(&other.label));
        }
        if self.op != other.op {
            return Some(self.op.cmp(&other.op));
        }
        self.value.partial_cmp(&other.value)
    }
}


impl LabelFilter {
    pub fn matches(&self, labels: &[Label]) -> bool {
        match self.op {
            LabelFilterOp::Equal => self.equal_value(labels),
            LabelFilterOp::NotEqual => !self.equal_value(labels),
            LabelFilterOp::RegexEqual => self.match_regexp(labels),
            LabelFilterOp::RegexNotEqual => !self.match_regexp(labels),
        }
    }

    fn equal_value(&self, labels: &[Label]) -> bool {
        let mut label_name_matches = 0;
        for label in labels {
            if to_canonical_label_name(&label.name) != self.label {
                continue;
            }
            label_name_matches += 1;
            if label.value == self.value {
                return true;
            }
        }
        if label_name_matches == 0 {
            // Special case for {non_existing_label=""}, which matches anything except of non-empty non_existing_label
            return self.value == "";
        }
        false
    }

    fn match_regexp(&self, labels: &[Label]) -> bool {
        let mut label_name_matches = 0;

        if let Some(re) = &self.re {
            for label in labels {
                if to_canonical_label_name(&label.name) != self.label {
                    continue;
                }
                label_name_matches += 1;
                if re.matches(&label.value) {
                    return true;
                }
            }
            if label_name_matches == 0 {
                // Special case for {non_existing_label=~"something|"}, which matches empty non_existing_label
                return re.matches("");
            }
        }
        false
    }
}

pub(crate) fn to_canonical_label_name(label_name: &str) -> &str {
    if label_name == "__name__" {
        return "";
    }
    label_name
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct LabelMatchers(Vec<LabelFilter>);

impl LabelMatchers {
    pub fn new(filters: Vec<LabelFilter>) -> Self {
        LabelMatchers(filters)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// find the matcher's value whose name equals the specified name. This function
    /// is designed to prepare error message of invalid promql expression.
    pub fn find_matcher_value(&self, name: &str) -> Option<String> {
        for m in &self.0 {
            if m.label.eq(name) {
                return Some(m.value.clone());
            }
        }
        None
    }

    /// find matchers whose name equals the specified name
    pub fn find_matchers(&self, name: &str) -> Vec<LabelFilter> {
        self.0
            .iter()
            .filter(|m| m.label.eq(name))
            .cloned()
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn push(&mut self, m: LabelFilter) {
        self.0.push(m);
    }

    pub fn iter(&self) -> impl Iterator<Item=&LabelFilter> {
        self.0.iter()
    }
}

impl fmt::Display for LabelMatchers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", &self.0)
    }
}

impl Deref for LabelMatchers {
    type Target = Vec<LabelFilter>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}