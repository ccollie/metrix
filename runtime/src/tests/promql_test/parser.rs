// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
use super::test_command::{EvalCmd, LoadCmd, TestCommand};
use super::types::{raise, ParseErr, SequenceValue};
use metricsql_common::time::current_time_millis;
use metricsql_parser::ast::Expr;
use metricsql_parser::parse_duration_value;
use prometheus_parse::{Labels, Scrape, Value};
use regex::Regex;
use std::error::Error;
use std::io::BufRead;
use std::time::{Duration, SystemTime};
use std::io;
use std::sync::LazyLock;
use crate::types::MetricName;

// Constants
pub const TEST_START_TIME: SystemTime = SystemTime::UNIX_EPOCH;


// Regex patterns
static PAT_EVAL_INSTANT: LazyLock<Regex> = LazyLock::new(||
    Regex::new(r"^eval(?:_(fail|warn|ordered))?\s+instant\s+(?:at\s+(.+?))?\s+(.+)$")
        .expect("failed to compile regex pattern")
);

static PAT_EVAL_RANGE: LazyLock<Regex> = LazyLock::new(||
   Regex::new(r"^eval(?:_(fail|warn))?\s+range\s+from\s+(.+)\s+to\s+(.+)\s+step\s+(.+?)\s+(.+)$")
       .expect("failed to compile regex pattern")
);

static PAT_LOAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^load(?:_(with_nhcb))?\s+(.+?)$").unwrap());

// Implementation of parse_duration function
pub(super) fn parse_duration(s: &str) -> Result<Duration, Box<dyn Error>> {
    let secs = s.parse::<u64>()?;
    Ok(Duration::from_secs(secs))
}

pub(super) fn enrich_parse_error(err: &mut ParseErr, f: impl FnOnce(&mut ParseErr)) {
    f(err);
}

// Implementation of parse_number function
pub(super) fn parse_number(s: &str) -> Result<f64, Box<dyn Error>> {
    metricsql_parser::prelude::parse_number(s)
    .map_err(|e| e.into())
}

pub(super) fn parse_expr(s: &str) -> Result<Expr, Box<dyn Error>> {
    metricsql_parser::prelude::parse(s).map_err(|e| e.into())
}

fn parse_timestamp(timestamp: Option<&str>) -> Result<i64, Box<dyn Error>> {
    // Parse value or skip
    // Parse timestamp or use given sample time
    if let Some(time) = timestamp
        .and_then(|x| x.parse::<i64>().ok())
    {
        Ok(time)
    } else {
        Ok(current_time_millis())
    }
}

fn labels_to_metric_name(labels: &Labels) -> MetricName {
    let mut metric_name = MetricName::default();
    for (k, v) in labels.iter() {
        metric_name.add_label(k, v);
    }
    metric_name.sort_labels();
    metric_name
}

/// Implementation of parse_series function
pub(super) fn parse_series(def_line: &str, line: usize) -> Result<(MetricName, Vec<SequenceValue>), ParseErr> {
    let br = io::BufReader::new(def_line.as_bytes());
    let scrape = Scrape::parse(br.lines()).unwrap(); // todo: remove unwrap

    let first = scrape.samples.first().unwrap(); // todo: remove unwrap

    match &first.value {
        Value::Counter(v) | Value::Gauge(v) | Value::Untyped(v) => {
            let vals = vec![SequenceValue {
                value: *v,
                omitted: false,
            }];
            let metric_name = labels_to_metric_name(&first.labels);
            Ok((metric_name, vals))
        }
        Value::Histogram(_) | Value::Summary(_) => {
            let msg = format!("unsupported value type in line {}: {}", line, def_line);
            Err(raise(line, msg))
        }
    }
}


// Implementation of parse_load function
pub fn parse_load(lines: &[String], i: usize) -> Result<(usize, TestCommand), ParseErr> {
    if !PAT_LOAD.is_match(&lines[i]) {
        return Err(
            raise(i, "invalid load command. (load[_with_nhcb] <step:duration>)".to_string())
            );
    }
    let parts: Vec<&str> = PAT_LOAD.captures(&lines[i]).unwrap()
        .iter()
        .skip(1)
        .map(|m| m.unwrap().as_str())
        .collect();

    //let with_nhcb = parts[1] == "with_nhcb";
    let step = parts[2];
    let gap_millis = parse_duration_value(step, 1)
        .map_err(|e| raise(i, format!("invalid test definition, failed to parse step: {:?}", e)))?;
    let gap = Duration::from_millis(gap_millis as u64);
    let mut cmd = LoadCmd::new(gap);
    let mut j = i + 1;
    while j < lines.len() {
        let def_line = &lines[j];
        if def_line.is_empty() {
            j -= 1;
            break;
        }
        let (metric, vals) = parse_series(def_line, j)?;
        cmd.set(metric, vals);
        j += 1;
    }
    Ok((j, TestCommand::Load(cmd)))
}


/// Implementation of parse_eval function
pub fn parse_eval(lines: &[String], mut i: usize) -> Result<(usize, TestCommand), ParseErr> {
    let instant_parts = PAT_EVAL_INSTANT.captures(&lines[i]);
    let range_parts = PAT_EVAL_RANGE.captures(&lines[i]);

    if instant_parts.is_none() && range_parts.is_none() {
        const MSG: &str = "invalid evaluation command. Must be either 'eval[_fail|_warn|_ordered] instant [at <offset:duration>] <query>' or 'eval[_fail|_warn] range from <from> to <to> step <step> <query>'";
        return Err(raise(i, MSG.to_string()));
    }

    let is_instant = instant_parts.is_some();

    let (expr, parts) = if is_instant {
        let parts = instant_parts.unwrap();
        (
            parts.get(3).unwrap().as_str(),
            parts
        )
    } else {
        let parts = range_parts.unwrap();
        (parts.get(5).unwrap().as_str(), parts)
    };

    let mod_str= parts.get(1).map(|m| m.as_str()).unwrap_or("");

    if let Err(_err) = parse_expr(expr) {
        let pos_offset = lines[i].find(expr).unwrap_or(0);
        let parse_err = ParseErr {
            line_offset: i,
            position_range: (pos_offset, pos_offset),
            query: lines[i].clone(),
            err: "invalid expression".to_string(),
        };
        return Err(parse_err);
    }

    let mut cmd: EvalCmd = if is_instant {
        let at = parts.get(2).map(|m| m.as_str()).unwrap_or("");
        let offset = parse_duration(at)
            .map_err(|e| raise(i, format!("invalid test definition, failed to parse offset: {:?}", e)))?;

        let ts = TEST_START_TIME + offset;
        EvalCmd::new_instant_eval_cmd(expr.to_string(), ts, i + 1)
    } else {
        let from = parts.get(2).unwrap().as_str();
        let to = parts.get(3).unwrap().as_str();
        let step = parts.get(4).unwrap().as_str();

        let parsed_from = parse_duration(from)
            .map_err(|e| raise(i, format!("invalid test definition, failed to parse start timestamp: {:?}", e)))?;

        let parsed_to = parse_duration(to)
            .map_err(|e| raise(i, format!("invalid test definition, failed to parse end timestamp: {:?}", e)))?;

        if parsed_to < parsed_from {
            return Err(raise(i,format!("invalid test definition, end timestamp ({to}) is before start timestamp ({from})")));
        }

        let parsed_step = parse_duration(step)
            .map_err(|e| raise(i, format!("invalid test definition, failed to parse step: {:?}", e)))?;

        EvalCmd::new_range_eval_cmd(expr.to_string(),
                                    TEST_START_TIME + parsed_from,
                                    TEST_START_TIME + parsed_to, parsed_step, i + 1)
    };

    match mod_str {
        "ordered" => cmd.ordered = true,
        "fail" => cmd.fail = true,
        "warn" => cmd.warn = true,
        _ => {}
    }

    let mut j = 1;
    while i + 1 < lines.len() {
        i += 1;
        let def_line = &lines[i];
        if def_line.is_empty() {
            i -= 1;
            break;
        }

        if cmd.fail && def_line.starts_with("expected_fail_message") {
            cmd.expected_fail_message = Some(def_line.trim_start_matches("expected_fail_message").trim().to_string());
            break;
        }

        if cmd.fail && def_line.starts_with("expected_fail_regexp") {
            let pattern = def_line.trim_start_matches("expected_fail_regexp").trim();
            let regex = Regex::new(pattern)
                .map_err(|e| raise(i, format!("invalid regex pattern in line {i}: {:?}", e)))?;
            cmd.expected_fail_regexp = Some(regex);
            break;
        }

        if let Ok(f) = parse_number(def_line) {
            cmd.expect(0, vec![SequenceValue { value: f, omitted: false }]);
            break;
        }

        let (metric, vals) = parse_series(def_line, i)
            .map_err(|_e| raise(i, "error parsing series".to_string()) )?;

        if vals.len() > 1 && is_instant {
            return Err(raise(i,"expecting multiple values in instant evaluation not allowed".to_string()));
        }

        cmd.expect_metric(j, metric, vals);
        j += 1;
    }

    Ok((i, TestCommand::Eval(cmd)))
}
