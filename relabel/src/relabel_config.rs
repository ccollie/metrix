use super::{labels_to_string, new_graphite_label_rules, DebugStep, GraphiteMatchTemplate, IfExpression, ParsedRelabelConfig, RelabelError};
use dynamic_lru_cache::DynamicCache;
use lazy_static::lazy_static;
use metricsql_common::prelude::{remove_start_end_anchors, Label, simplify, PromRegex};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::str::FromStr;
use bon::Builder;
use crate::relabel_error::RelabelResult;

pub const METRIC_NAME_LABEL: &str = "__name__";

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RelabelAction {
    Drop,
    DropEqual,
    DropIfContains,
    DropIfEqual,
    DropMetrics,
    Graphite,
    HashMod,
    Keep,
    KeepEqual,
    KeepIfContains,
    KeepIfEqual,
    KeepMetrics,
    Lowercase,
    LabelMap,
    LabelMapAll,
    LabelDrop,
    LabelKeep,
    #[default]
    Replace,
    ReplaceAll,
    Uppercase,
}

impl RelabelAction {
    pub fn as_str(&self) -> &'static str {
        use RelabelAction::*;
        match self {
            Graphite => "graphite",
            Replace => "replace",
            ReplaceAll => "replace_all",
            KeepIfEqual => "keep_if_equal",
            DropIfEqual => "drop_if_equal",
            KeepEqual => "keepequal",
            DropEqual => "dropequal",
            Keep => "keep",
            Drop => "drop",
            DropIfContains => "drop_if_contains",
            DropMetrics => "drop_metrics",
            HashMod => "hashmod",
            KeepMetrics => "keep_metrics",
            Uppercase => "uppercase",
            Lowercase => "lowercase",
            LabelMap => "labelmap",
            LabelMapAll => "labelmap_all",
            LabelDrop => "labeldrop",
            LabelKeep => "labelkeep",
            KeepIfContains => "keep_if_contains",
        }
    }
}

impl Display for RelabelAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for RelabelAction {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use RelabelAction::*;
        match s.to_lowercase().as_str() {
            "graphite" => Ok(Graphite),
            "replace" => Ok(Replace),
            "replace_all" => Ok(ReplaceAll),
            "keep_if_equal" => Ok(KeepIfEqual),
            "drop" => Ok(Drop),
            "drop_equal" | "dropequal" => Ok(DropEqual),
            "drop_if_equal" => Ok(DropIfEqual),
            "drop_if_contains" => Ok(DropIfContains),
            "drop_metrics" => Ok(DropMetrics),
            "keep_equal" | "keepequal" => Ok(KeepEqual),
            "keep" => Ok(Keep),
            "hashmod" => Ok(HashMod),
            "keep_metrics" => Ok(KeepMetrics),
            "keep_if_contains" => Ok(KeepIfContains),
            "lowercase" => Ok(Lowercase),
            "labelmap" => Ok(LabelMap),
            "labelmap_all" => Ok(LabelMapAll),
            "labeldrop" | "label_drop" => Ok(LabelDrop),
            "labelkeep" | "label_keep" => Ok(LabelKeep),
            "uppercase" => Ok(Uppercase),
            _ => Err(format!("unknown action: {}", s)),
        }
    }
}

/// `RelabelConfig` represents relabel config.
///
/// See https://prometheus.io/docs/prometheus/latest/configuration/configuration/#relabel_config
#[derive(Debug, Default, Clone, Builder)]
pub struct RelabelConfig {
    pub if_expr: Option<String>,
    pub action: RelabelAction,
    pub source_labels: Vec<String>,
    pub separator: String,
    pub target_label: String,
    pub regex: Option<String>,
    pub modulus: u64,
    #[builder(default = "$1".to_string())]
    pub replacement: String,

    /// match is used together with Labels for `action: graphite`. For example:
    /// - action: graphite
    ///   match: 'foo.*.*.bar'
    ///   labels:
    ///     job: '$1'
    ///     instance: '${2}:8080'
    pub r#match: String,

    /// Labels is used together with match for `action: graphite`. For example:
    /// - action: graphite
    ///   match: 'foo.*.*.bar'
    ///   labels:
    ///     job: '$1'
    ///     instance: '${2}:8080'
    pub labels: HashMap<String, String>,
}

impl RelabelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn keep(source_labels: Option<Vec<String>>, if_expr: Option<String>) -> RelabelResult<Self> {
        let valid = match (&source_labels, &if_expr) {
            (Some(labels), None) => !labels.is_empty(),
            (None, Some(expr)) => !expr.is_empty(),
            _ => false,
        };
        if !valid {
            return Err(RelabelError::InvalidConfiguration("missing `source_labels` or if expression for `action=keep`".to_string()));
        }

        Ok(Self {
            if_expr,
            action: RelabelAction::Keep,
            source_labels: source_labels.unwrap_or_default(),
            ..Default::default()
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct ParsedConfigs(pub Vec<ParsedRelabelConfig>);

impl ParsedConfigs {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// apply_debug applies pcs to labels in debug mode.
    ///
    /// It returns DebugStep list - one entry per each applied relabeling step.
    pub fn apply_debug(&self, labels: &mut Vec<Label>) -> Vec<DebugStep> {
        self.apply_internal(labels, 0, true)
    }

    /// applies self to labels starting from the labelsOffset.
    pub fn apply(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        let _ = self.apply_internal(labels, labels_offset, false);
    }

    fn apply_internal(
        &self,
        labels: &mut Vec<Label>,
        labels_offset: usize,
        debug: bool,
    ) -> Vec<DebugStep> {
        let mut dss: Vec<DebugStep> = Vec::with_capacity(labels.len());
        let mut in_str: String = "".to_string();
        if debug {
            in_str = labels_to_string(&labels[labels_offset..])
        }
        for prc in self.0.iter() {
            prc.apply(labels, labels_offset);
            if debug {
                let out_str = labels_to_string(&labels[labels_offset..]);
                dss.push(DebugStep {
                    rule: prc.to_string(),
                    r#in: in_str,
                    out: out_str.clone(),
                });
                in_str = out_str
            }
            if labels.len() == labels_offset {
                // All the labels have been removed.
                return dss;
            }
        }

        remove_empty_labels(labels, labels_offset);
        if debug {
            let out_str = labels_to_string(&labels[labels_offset..]);
            if out_str != in_str {
                dss.push(DebugStep {
                    rule: "remove empty labels".to_string(),
                    r#in: in_str,
                    out: out_str,
                });
            }
        }

        dss
    }
}

// todo:
// https://github.com/VictoriaMetrics/VictoriaMetrics/blob/master/lib/promrelabel/config.go#L123
impl Display for ParsedConfigs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        for prc in &self.0 {
            s.push_str(&prc.to_string());
        }
        write!(f, "{}", s)
    }
}

/// parses relabel configs from the given data.
// pub fn parse_relabel_configs_data(data: &str) -> Result<ParsedConfigs, String> {
//     let rcs: Vec<RelabelConfig> =  serde_yaml::from_str(data)
//         .map_err(|e| format!("cannot parse relabel configs from data: {:?}", e))?;
//     parse_relabel_configs(rcs)
// }

/// parse_relabel_configs parses rcs to dst.
pub fn parse_relabel_configs(rcs: Vec<RelabelConfig>) -> RelabelResult<ParsedConfigs> {
    if rcs.is_empty() {
        return Ok(ParsedConfigs::default());
    }
    let mut prcs: Vec<ParsedRelabelConfig> = Vec::with_capacity(rcs.len());
    for (i, item) in rcs.into_iter().enumerate() {
        let prc = parse_relabel_config(item);
        if let Err(err) = prc {
            return Err(RelabelError::InvalidRule(format!(
                "error when parsing `relabel_config` #{}: {:?}",
                i + 1,
                err
            )));
        }
        prcs.push(prc?);
    }
    Ok(ParsedConfigs(prcs))
}

const DEFAULT_ORIGINAL_REGEX_STR_FOR_RELABEL_CONFIG: &str = "(.*)" ;

// todo: use OnceLock
lazy_static! {
    pub static ref DEFAULT_ORIGINAL_REGEX_FOR_RELABEL_CONFIG: Regex = Regex::new(".*").unwrap();
    pub static ref DEFAULT_REGEX_FOR_RELABEL_CONFIG: Regex = Regex::new("^(.*)$").unwrap();
}

pub(crate) fn is_default_regex_for_config(regex: &Regex) -> bool {
    regex.as_str() == "^(.*)$"
}

fn validate_labels(
    action: RelabelAction,
    source_labels: &Vec<String>,
    target_label: &str,
) -> Result<(), String> {
    if source_labels.is_empty() {
        return Err(format!("missing `source_labels` for `action={action}`"));
    }
    if target_label.is_empty() {
        return Err(format!("missing `target_label` for `action={action}`"));
    }
    Ok(())
}

pub fn parse_relabel_config(rc: RelabelConfig) -> RelabelResult<ParsedRelabelConfig> {
    let separator = if rc.separator.is_empty() { ";" } else { &rc.separator };

    let (regex_anchored, regex_original_compiled, prom_regex) = compile_regex(&rc)
        .map_err(|err| RelabelError::InvalidConfiguration(err))?;

    let modulus = rc.modulus;
    let replacement = if rc.replacement.is_empty() { "$1".to_string() } else { rc.replacement.clone() };

    let graphite_match_template = if !rc.r#match.is_empty() {
        Some(GraphiteMatchTemplate::new(&rc.r#match))
    } else {
        None
    };

    let graphite_label_rules = if !rc.labels.is_empty() {
        new_graphite_label_rules(&rc.labels)
    } else {
        vec![]
    };

    let if_expr =  if let Some(expr) = &rc.if_expr {
        Some(IfExpression::parse(expr)?)
    } else {
        None
    };

    validate_action(&rc, &replacement)
        .map_err(|s| RelabelError::InvalidRule(s))?;

    // TODO:
    let rule_original = format!("{:?}", &rc);

    let has_capture_group_in_target_label = rc.target_label.contains("$");
    let prc = ParsedRelabelConfig {
        rule_original: rule_original.to_string(),
        source_labels: rc.source_labels,
        separator: separator.to_string(),
        target_label: rc.target_label,
        regex_anchored,
        modulus,
        action: rc.action,
        r#if: if_expr,
        graphite_match_template,
        graphite_label_rules,
        string_replacer_cache: DynamicCache::new(64), // todo: pass in config ?
        regex: prom_regex,
        regex_original: regex_original_compiled,
        has_capture_group_in_target_label,
        has_capture_group_in_replacement: replacement.contains("$"),
        has_label_reference_in_replacement: replacement.contains("{{"),
        replacement,
        submatch_cache: DynamicCache::new(64),
    };
    Ok(prc)
}

fn compile_regex(rc: &RelabelConfig) -> Result<(Regex, Regex, PromRegex), String> {
    use RelabelAction::*;

    let default_regex_original = DEFAULT_ORIGINAL_REGEX_STR_FOR_RELABEL_CONFIG;

    let (is_empty, reg_str) = if let Some(regex) = &rc.regex {
        let empty = is_empty_or_default_regex(regex);
        if empty {
            (true, default_regex_original)
        } else {
            (false, regex.as_str())
        }
    } else {
        (true, default_regex_original)
    };

    if is_empty {
        return Ok((
            DEFAULT_REGEX_FOR_RELABEL_CONFIG.clone(),
            DEFAULT_ORIGINAL_REGEX_FOR_RELABEL_CONFIG.clone(),
            PromRegex::new(default_regex_original).unwrap(),
        ));
    }

    let regex = if rc.action != ReplaceAll && rc.action != LabelMapAll {
        remove_start_end_anchors(&reg_str)
    } else {
        reg_str
    };

    let regex_anchored = Regex::new(&format!("^(?:{})$", regex))
        .map_err(|e| format!("cannot parse `regex` {}: {:?}", reg_str, e))?;
    let regex_original_compiled = Regex::new(&regex)
        .map_err(|e| format!("cannot parse `regex` {}: {:?}", regex, e))?;
    let prom_regex = PromRegex::new(&regex).map_err(|err| {
        format!("BUG: cannot parse already parsed regex {}: {:?}", regex, err)
    })?;

    Ok((regex_anchored, regex_original_compiled, prom_regex))
}


fn validate_action(
    rc: &RelabelConfig,
    replacement: &str,
) -> Result<(), String> {
    use RelabelAction::*;

    let source_labels = &rc.source_labels;
    let target_label = &rc.target_label;

    match rc.action {
        Graphite => {
            if rc.r#match.is_empty() {
                return Err("missing `match` for `action=graphite`".to_string());
            }
            if rc.labels.is_empty() {
                return Err("missing `labels` for `action=graphite`".to_string());
            }
            if !source_labels.is_empty() {
                return Err("`source_labels` cannot be used with `action=graphite`".to_string());
            }
            if !target_label.is_empty() {
                return Err("`target_label` cannot be used with `action=graphite`".to_string());
            }
            if !replacement.is_empty() {
                return Err("`replacement` cannot be used with `action=graphite`".to_string());
            }
            if rc.regex.is_some() {
                return Err("`regex` cannot be used for `action=graphite`".to_string());
            }
        }
        Replace => {
            if target_label.is_empty() {
                return Err("missing `target_label` for `action=replace`".to_string());
            }
        }
        ReplaceAll | DropIfContains | DropIfEqual | KeepIfContains | KeepIfEqual | LabelMap | LabelMapAll
        | LabelDrop | LabelKeep | KeepEqual | DropEqual | HashMod | Uppercase | Lowercase => {
            validate_labels(rc.action, source_labels, target_label)?;
        }
        Keep | Drop => {
            if source_labels.is_empty() && rc.if_expr.is_none() {
                return Err(format!("missing `source_labels` for `action={}`", rc.action));
            }
        }
        KeepMetrics | DropMetrics => {
            if is_empty_regex(&rc.regex) && rc.if_expr.is_none() {
                return Err(format!("`regex` must be non-empty for `action={}`", rc.action));
            }
            if !source_labels.is_empty() {
                return Err(format!("`source_labels` must be empty for `action={}`", rc.action));
            }
        }
    }
    Ok(())
}

fn is_default_regex(expr: &str) -> bool {
    match simplify(expr) {
        Ok((prefix, suffix)) => prefix == "" && suffix == ".*",
        _ => false,
    }
}

fn is_empty_regex(regex: &Option<String>) -> bool {
    if let Some(regex) = regex {
        return is_empty_regex_str(regex)
    }
    true
}

fn is_empty_or_default_regex(expr: &str) -> bool {
    is_empty_regex_str(expr) || is_default_regex(expr)
}

fn is_empty_regex_str(regex: &str) -> bool {
    regex.is_empty() || regex == "(?:)"
}

fn remove_empty_labels(labels: &mut Vec<Label>, labels_offset: usize) {
    let mut i: usize = labels.len() - 1;
    while i >= labels_offset {
        let label = &labels[i];
        if label.name.is_empty() || label.value.is_empty() {
            labels.remove(i);
        }
        i -= 1;
    }
}
