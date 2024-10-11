use super::utils::{
    are_equal_label_values,
    concat_label_values,
    contains_all_label_values,
    get_label_value,
    is_regex_matcher,
    set_label_value
};
use super::{is_default_regex_for_config, GraphiteLabelRule, GraphiteMatchTemplate, IfExpression, RelabelAction, METRIC_NAME_LABEL};
use dynamic_lru_cache::DynamicCache;
use enquote::enquote;
use metricsql_common::bytes_util::FastStringTransformer;
use metricsql_common::prelude::StringMatchHandler;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::sync::{LazyLock, OnceLock};
use bon::Builder;
use xxhash_rust::xxh3::xxh3_64;
use metricsql_common::regex_util::PromRegex;
use metricsql_parser::label::Label;


pub trait Action {
    fn apply(&self, labels: &mut Vec<Label>, labels_offset: usize);
}

/// DebugStep contains debug information about a single relabeling rule step
#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
pub struct DebugStep {
    /// rule contains string representation of the rule step
    pub rule: String,

    /// In contains the input labels before the execution of the rule step
    pub r#in: String,

    /// Out contains the output labels after the execution of the rule step
    pub out: String,
}

/// ParsedRelabelConfig contains parsed `relabel_config`.
///
/// See https://prometheus.io/docs/prometheus/latest/configuration/configuration/#relabel_config
#[derive(Debug, Clone, Builder)]
pub struct ParsedRelabelConfig {
    /// rule_original contains the original relabeling rule for the given ParsedRelabelConfig.
    pub rule_original: String,

    pub source_labels: Vec<String>,
    pub separator: String,
    pub target_label: String,
    pub modulus: u64,
    pub replacement: String,
    pub action: RelabelAction,
    pub r#if: Option<IfExpression>,

    pub regex: PromRegex,
    pub regex_anchored: Regex,
    pub regex_original: Regex,

    pub has_capture_group_in_target_label: bool,
    pub has_capture_group_in_replacement: bool,
    pub has_label_reference_in_replacement: bool,

    pub graphite_match_template: Option<GraphiteMatchTemplate>,
    pub graphite_label_rules: Vec<GraphiteLabelRule>,
    pub(crate) string_replacer_cache: DynamicCache<String, String>, // todo: AHash/gxhash
    pub(crate) submatch_cache: DynamicCache<String, String>,
}

impl ParsedRelabelConfig {

    pub fn apply_debug(&mut self, labels: &[Label], _labels_offset: usize) -> (Vec<Label>, DebugStep) {
        self.apply_internal(labels, 0, true)
    }

    fn apply_internal(&mut self, labels: &[Label], labels_offset: usize, debug: bool) -> (Vec<Label>, DebugStep) {
        let mut labels = labels.to_vec();
        let labels_offset = labels_offset;
        let mut debug_step = DebugStep::default();
        if debug {
            debug_step.rule = self.rule_original.clone();
            debug_step.r#in = labels_to_string(&labels);
        }
        self.apply(&mut labels, labels_offset);
        if debug {
            debug_step.out = labels_to_string(&labels);
        }
        (labels, debug_step)
    }

    /// `apply` applies relabeling according to the given labels.
    ///
    /// See https://prometheus.io/docs/prometheus/latest/configuration/configuration/#relabel_config
    pub fn apply(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        use RelabelAction::*;
        let src = &labels[labels_offset..];
        if let Some(if_expr) = &self.r#if {
            if !if_expr.is_match(src) {
                if self.action == Keep {
                    // Drop the target on `if` mismatch for `action: keep`
                    labels.truncate(labels_offset);
                }
                // Do not apply actions on `if` mismatch.
                return;
            }
        }
        match &self.action {
            Drop => self.handle_drop(labels, labels_offset),
            DropEqual => self.drop_equal(labels, labels_offset),
            DropIfContains => self.drop_if_contains(labels, labels_offset),
            DropIfEqual => self.drop_if_equal(labels, labels_offset),
            Graphite => self.graphite(labels, labels_offset),
            HashMod => self.hashmod(labels, labels_offset),
            Keep => self.handle_keep(labels, labels_offset),
            KeepEqual => self.handle_keep_equal(labels, labels_offset),
            KeepIfContains => self.keep_if_contains(labels, labels_offset),
            KeepIfEqual => self.keep_if_equal(labels, labels_offset),
            Lowercase => self.lowercase(labels, labels_offset),
            LabelMap => self.label_map(labels, labels_offset),
            LabelDrop => self.label_drop(labels, labels_offset),
            LabelMapAll => self.label_map_all(labels, labels_offset),
            LabelKeep => self.label_keep(labels, labels_offset),
            Uppercase => self.uppercase(labels, labels_offset),
            Replace => handle_replace(self, labels, labels_offset),
            ReplaceAll => self.replace_all(labels, labels_offset),
            _=> {
                panic!("BUG: unsupported action: {}", self.action);
            }
        }
    }

    /// Drop the entry if `source_labels` joined with `separator` matches `regex`
    fn handle_drop(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        if is_default_regex_for_config(&self.regex_anchored) {
            // Fast path for the case with `if` and without explicitly set `regex`:
            //
            // - action: drop
            //   if: 'some{label=~"filters"}'
            //
            return;
        }

        let buf = concat_label_values(&labels, &self.source_labels, &self.separator);
        let drop = self.regex.is_match(&buf);
        if drop {
            labels.truncate(labels_offset);
        }
    }

    fn drop_equal(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        // Drop the entry if `source_labels` joined with `separator` matches `target_label`
        let buf = concat_label_values(&labels, &self.source_labels, &self.separator);
        let target_value = get_label_value(&labels[labels_offset..], &self.target_label);
        let drop = buf == target_value;
        if !drop {
            return;
        }
        labels.truncate(labels_offset);
    }

    /// Drop the entry if target_label contains all the label values listed in source_labels.
    /// For example, the following relabeling rule would drop the entry if __meta_consul_tags
    /// contains values of __meta_required_tag1 and __meta_required_tag2:
    ///
    ///   - action: drop_if_contains
    ///     target_label: __meta_consul_tags
    ///     source_labels: [__meta_required_tag1, __meta_required_tag2]
    ///
    fn drop_if_contains(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        if contains_all_label_values(labels, &self.target_label, &self.source_labels) {
            labels.truncate(labels_offset);
        }
    }

    /// Drop the entry if all the label values in source_labels are equal.
    /// For example:
    ///
    ///   - source_labels: [foo, bar]
    ///     action: drop_if_equal
    ///
    /// Would drop the entry if `foo` value equals `bar` value
    fn drop_if_equal(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        if are_equal_label_values(labels, &self.source_labels) {
            labels.truncate(labels_offset);
        }
    }

    fn graphite(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        let metric_name = get_label_value(&labels, METRIC_NAME_LABEL);
        if let Some(gmt) = &self.graphite_match_template {
            // todo: use pool
            let mut matches: Vec<String> = Vec::with_capacity(4);
            if !gmt.is_match(&mut matches, metric_name) {
                // Fast path - name mismatch
                return;
            }
            // Slow path - extract labels from graphite metric name
            for gl in self.graphite_label_rules.iter() {
                let value_str = gl.expand(&matches);
                set_label_value(labels, labels_offset, &gl.target_label, value_str)
            }
        } else {
            return
        }
    }

    /// Store the hashmod of `source_labels` joined with `separator` at `target_label`
    fn hashmod(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        let buf = concat_label_values(&labels, &self.source_labels, &self.separator);
        let hash_mod = xxh3_64(&buf.as_bytes()) % self.modulus;
        let value_str = hash_mod.to_string();
        set_label_value(labels, labels_offset, &self.target_label, value_str)
    }

    fn handle_keep(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        // Keep the entry if `source_labels` joined with `separator` matches `regex`
        if is_default_regex_for_config(&self.regex_anchored) {
            // Fast path for the case with `if` and without explicitly set `regex`:
            //
            // - action: keep
            //   if: 'some{label=~"filters"}'
            //
            return;
        }
        let buf = concat_label_values(&labels, &self.source_labels, &self.separator);
        let keep = self.regex.is_match(&buf);
        if !keep {
            labels.truncate(labels_offset);
        }
    }

    /// keep the entry if `source_labels` joined with `separator` matches `target_label`
    fn handle_keep_equal(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        let buf = concat_label_values(&labels, &self.source_labels, &self.separator);
        let target_value = get_label_value(&labels[labels_offset..], &self.target_label);
        let keep = buf == target_value;
        if keep {
            return;
        }
        labels.truncate(labels_offset);
    }

    /// Keep the entry if target_label contains all the label values listed in source_labels.
    /// For example, the following relabeling rule would leave the entry if __meta_consul_tags
    /// contains values of __meta_required_tag1 and __meta_required_tag2:
    ///
    ///   - action: keep_if_contains
    ///     target_label: __meta_consul_tags
    ///     source_labels: [__meta_required_tag1, __meta_required_tag2]
    ///
    fn keep_if_contains(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        if contains_all_label_values(labels, &self.target_label, &self.source_labels) {
            return
        }
        labels.truncate(labels_offset);
    }

    /// Keep the entry if all the label values in source_labels are equal.
    /// For example:
    ///
    ///   - source_labels: [foo, bar]
    ///     action: keep_if_equal
    ///
    /// Would leave the entry if `foo` value equals `bar` value
    fn keep_if_equal(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        if !are_equal_label_values(labels, &self.source_labels) {
            labels.truncate(labels_offset);
        }
    }

    fn label_drop(&self, labels: &mut Vec<Label>, _labels_offset: usize) {
        // Drop all the labels matching `regex`
        labels.retain(|label| !self.regex.is_match(&label.name))
    }

    fn label_keep(&self, labels: &mut Vec<Label>, _labels_offset: usize) {
        // Keep all the labels matching `regex`
        labels.retain(|label| self.regex.is_match(&label.name))
    }

    fn label_map(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        let mut values = HashMap::with_capacity(labels.len());
        // let mut keys = HashSet::with_capacity(labels.len());
        // Copy `source_labels` to `target_label`
        // Replace label names with the `replacement` if they match `regex`
        for label in labels.iter() {
            let label_name = self.replace_full_string_fast(&label.name);
            if label_name != label.name {
                values.insert(label_name, label.value.clone());
            }
        }
        for (k, v) in values.iter() {
            let value_str = v.clone();
            set_label_value(labels, labels_offset, k, value_str)
        }
    }

    /// replace all the occurrences of `regex` at label names with `replacement`
    fn label_map_all(&self, labels: &mut Vec<Label>, _labels_offset: usize) {
        for label in labels.iter_mut() {
            label.name = self.replace_string_submatches_fast(&label.name)
        }
    }

    fn lowercase(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        let buf = concat_label_values(&labels, &self.source_labels, &self.separator);
        let value_str = buf.to_uppercase();
        set_label_value(labels, labels_offset, &self.target_label, value_str)
    }

    /// Replace all the occurrences of `regex` at `source_labels` joined with `separator` with the `replacement`
    /// and store the result at `target_label`
    /// todo: use buffer pool
    fn replace_all(&self, labels: &mut Vec<Label>, label_offset: usize) {
        let buf = concat_label_values(labels, &self.source_labels, &self.separator);
        let value_str = self.replace_string_submatches_fast(&buf);
        if value_str != buf {
            set_label_value(labels, label_offset, &self.target_label, value_str)
        }
    }

    fn uppercase(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        let buf = concat_label_values(&labels, &self.source_labels, &self.separator);
        let value_str = buf.to_uppercase();
        set_label_value(labels, labels_offset, &self.target_label, value_str)
    }

    /// replaces s with the replacement if s matches '^regex$'.
    ///
    /// s is returned as if it doesn't match '^regex$'.
    pub(super) fn replace_full_string_fast(&self, s: &str) -> String {
        let replacement = &self.replacement;
        if !self.has_capture_group_in_target_label {
            match &self.regex.matcher {
                StringMatchHandler::Literal(pattern) => {
                    if s == pattern {
                        return replacement.clone();
                    }
                    return s.to_string();
                }
                _ => {}
            }
        }

        if replacement == "$1" {
            // Fast path for commonly used rule for deleting label prefixes such as:
            //
            // - action: labelmap
            //   regex: __meta_kubernetes_node_label_(.+)
            //
            let re_str = self.regex_original.as_str();
            let prefix = &self.regex.prefix;
            if re_str.starts_with(prefix) {
                let suffix = &s[prefix.len()..];
                let re_suffix = &re_str[prefix.len()..];
                if re_suffix == "(.*)" {
                    return suffix.to_string();
                } else if re_suffix == "(.+)" {
                    if !suffix.is_empty() {
                        return suffix.to_string();
                    }
                    return s.to_string();
                }
            }
        }
        if !self.regex.is_match(s) {
            // Fast path - regex mismatch
            return s.to_string();
        }
        // Slow path - handle the rest of cases.
        self.replace_string(s)
    }

    fn replace_string(&self, val: &str) -> String {
        // how to avoid this alloc ?
        let key = val.to_string();
        let res = self.string_replacer_cache.get_or_insert(&key, || {
            self.replace_full_string_slow(val)
        });
        res.to_string()
    }

    /// replaces s with the replacement if s matches '^regex$'.
    ///
    /// s is returned as is if it doesn't match '^regex$'.
    pub fn replace_full_string_slow(&self, s: &str) -> String {
        // Slow path - regexp processing
        self.expand_capture_groups(&self.replacement, s)
    }

    fn replace_submatches(&self, val: &str) -> String {
        // how to avoid this alloc ?
        let key = val.to_string();
        let res = self.submatch_cache.get_or_insert(&key, || {
            self.replace_string_submatches_slow(val)
        });
        res.to_string()
    }

    /// replaces all the regex matches with the replacement in s.
    pub(crate) fn replace_string_submatches_fast(&self, s: &str) -> String {
        if !self.has_capture_group_in_replacement && !is_regex_matcher(&self.regex.matcher) && !self.regex.is_match(s) {
            // Fast path - no regex matches in s.
            return s.to_string();
        }

        // Slow path - replace all the regex matches in s with the replacement.
        self.replace_submatches(s)
    }

    /// replaces all the regex matches with the replacement in s.
    fn replace_string_submatches_slow(&self, s: &str) -> String {
        let res = self.regex_original.replace_all(s, &self.replacement);
        res.to_string()
    }

    fn expand_capture_groups(&self, template: &str, source: &str) -> String {
        if let Some(captures) = self.regex_anchored.captures(source) {
            let mut s = String::with_capacity(template.len() + 16);
            captures.expand(template, &mut s);
            return s
        }
        source.to_string()
    }
}

impl Display for ParsedRelabelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.rule_original)
    }
}

fn handle_replace(prc: &ParsedRelabelConfig, labels: &mut Vec<Label>, labels_offset: usize) {
    // Store `replacement` at `target_label` if the `regex` matches `source_labels` joined with `separator`
    let replacement = if prc.has_label_reference_in_replacement {
        let mut buf: String = String::with_capacity(128);
        // Fill {{labelName}} references in the replacement
        fill_label_references(&mut buf, &prc.replacement, &labels[labels_offset..]);
        buf
    } else {
        prc.replacement.clone()
    };

    let buf = concat_label_values(labels, &prc.source_labels, &prc.separator);
    if is_default_regex_for_config(&prc.regex_anchored) && !prc.has_capture_group_in_target_label {
        if replacement == "$1" {
            // Fast path for the rule that copies source label values to destination:
            // - source_labels: [...]
            //   target_label: foobar
            let value_str = buf;
            return set_label_value(labels, labels_offset, &prc.target_label, value_str);
        }
        if !prc.has_label_reference_in_replacement {
            // Fast path for the rule that sets label value:
            // - target_label: foobar
            //   replacement: something-here
            set_label_value(labels, labels_offset, &prc.target_label, replacement);
            return;
        }
    }
    let source_str = &buf;
    if !prc.regex.is_match(source_str) {
        // Fast path - regexp mismatch.
        return;
    }
    let value_str = if replacement == prc.replacement {
        // Fast path - the replacement wasn't modified, so it is safe calling stringReplacer.Transform.
        prc.replace_string(source_str)
    } else {
        // Slow path - the replacement has been modified, so the valueStr must be calculated
        // from scratch based on the new replacement value.
        prc.expand_capture_groups(&replacement, source_str)
    };
    let name_str = &prc.target_label;
    if prc.has_capture_group_in_target_label {
        // Slow path - target_label contains regex capture groups, so the target_label
        // must be calculated from the regex match.
        let name_str = prc.expand_capture_groups(name_str, source_str);
        set_label_value(labels, labels_offset, &name_str, value_str)
    } else {
        set_label_value(labels, labels_offset, &prc.target_label, value_str)
    }
}

/// `labels_to_string` returns Prometheus string representation for the given labels.
///
/// Labels in the returned string are sorted by name,
/// while the __name__ label is put in front of {} labels.
pub fn labels_to_string(labels: &[Label]) -> String {
    let mut labels_copy = Vec::with_capacity(labels.len());
    let mut mname = "";
    let mut capacity = 0;
    for label in labels.iter() {
        if label.name == METRIC_NAME_LABEL {
            mname = &label.value;
            capacity += label.value.len();
        } else {
            capacity += label.name.len() + label.value.len() + 2;
            labels_copy.push(label);
        }
    }
    if !mname.is_empty() && labels_copy.is_empty() {
        return mname.to_string();
    }
    labels_copy.sort();

    let mut b = String::with_capacity(capacity);
    if !mname.is_empty() {
        b.push_str(mname);
    }
    b.push('{');
    for (i, label) in labels_copy.iter().enumerate() {
        b.push_str(&label.name);
        b.push('=');
        b.push_str(&*enquote('"', &label.value));
        if i + 1 < labels_copy.len() {
            b.push(',');
        }
    }
    b.push('}');
    b
}

pub(crate) fn fill_label_references(dst: &mut String, replacement: &str, labels: &[Label]) {
    let mut s = replacement;
    while !s.is_empty() {
        if let Some(n) = s.find("{{") {
            dst.push_str(&s[0..n]);
            s = &s[n + 2..];
        } else {
            dst.push_str(s);
            return;
        }
        if let Some(n) = s.find("}}") {
            let label_name = &s[0..n];
            let label_value = get_label_value(labels, label_name);
            s = &s[n + 2..];
            dst.push_str(label_value);
        } else {
            dst.push_str("{{");
            dst.push_str(s);
            return;
        }
    }
}

static UNSUPPORTED_METRIC_NAME_CHARS_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-zA-Z0-9_]").unwrap());
static UNSUPPORTED_LABEL_NAME_REGEX: LazyLock<Regex> = LazyLock::new(||Regex::new(r"[^a-zA-Z0-9_:]").unwrap());

pub fn is_valid_metric_name(name: &str) -> bool {
    !UNSUPPORTED_METRIC_NAME_CHARS_REGEX.is_match(name)
}

pub fn is_valid_label_name(name: &str) -> bool {
    !UNSUPPORTED_LABEL_NAME_REGEX.is_match(name)
}

static LABEL_NAME_SANITIZER: OnceLock<FastStringTransformer> = OnceLock::new();

fn get_metric_name_sanitizer() -> &'static FastStringTransformer {
    static METRIC_NAME_SANITIZER: OnceLock<FastStringTransformer> = OnceLock::new();
    METRIC_NAME_SANITIZER.get_or_init(|| {
        FastStringTransformer::new(|s: &str| -> String {
            return UNSUPPORTED_METRIC_NAME_CHARS_REGEX.replace_all(s, "_").to_string();
        })
    })
}

fn label_name_sanitizer() -> &'static FastStringTransformer {
    LABEL_NAME_SANITIZER.get_or_init(|| {
        FastStringTransformer::new(|s: &str| -> String {
            return UNSUPPORTED_LABEL_NAME_REGEX.replace_all(s, "_").to_string();
        })
    })
}

/// `sanitize_label_name` replaces unsupported by Prometheus chars in label names with _.
///
/// See https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
pub fn sanitize_label_name(name: &str) -> String {
    label_name_sanitizer().transform(name)
}

/// `sanitize_metric_name` replaces unsupported by Prometheus chars in metric names with _.
///
// See https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
pub fn sanitize_metric_name(value: &str) -> String {
    get_metric_name_sanitizer().transform(value)
}