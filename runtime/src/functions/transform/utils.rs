use std::cmp::Ordering;

use chrono::{Offset, TimeZone};

use metricsql_common::time::timestamp_ms_to_datetime;
use metricsql_parser::ast::{Expr, MetricExpr};

use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult};
use crate::types::{Label, Labels, Timeseries};

/// copy_timeseries returns a copy of tss.
pub(super) fn copy_timeseries(tss: &[Timeseries]) -> Vec<Timeseries> {
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());
    for src in tss {
        rvs.push(src.clone());
    }
    rvs
}

pub(super) fn expect_transform_args_num(
    tfa: &TransformFuncArg,
    expected: usize,
) -> RuntimeResult<()> {
    let arg_count = tfa.args.len();
    if arg_count == expected {
        return Ok(());
    }
    Err(RuntimeError::ArgumentError(format!(
        "unexpected number of args; got {arg_count}; want {expected}"
    )))
}

// Todo: test this, making sure to account for dst

pub fn get_timezone_offset(zone: &impl TimeZone, timestamp_msecs: i64) -> Option<i64> {
    match timestamp_ms_to_datetime(timestamp_msecs) {
        None => None,
        Some(naive) => {
            let offset = zone.offset_from_utc_datetime(&naive);
            let fixed = offset.fix();
            Some(fixed.local_minus_utc() as i64)
        }
    }
}

#[inline]
/// This exists solely for readability
pub(super) fn clamp_min(val: f64, limit: f64) -> f64 {
    if val < limit {
        limit
    } else {
        val
    }
}

pub(crate) fn ru(free_value: f64, max_value: f64) -> f64 {
    // ru(freev, maxv) = clamp_min(maxv - clamp_min(freev, 0), 0) / clamp_min(maxv, 0) * 100
    let used = clamp_min(max_value - clamp_min(free_value, 0.0), 0.0);
    let max =  clamp_min(max_value, 0.0);
    let utilization = used / max;
    utilization * 100_f64
}

pub fn extract_labels_from_expr(arg: &Expr) -> Option<Labels> {
    if let Expr::MetricExpression(me) = arg {
        return Some(extract_labels(me));
    }
    None
}

pub fn extract_labels(expr: &MetricExpr) -> Labels {
    expr.matchers
        .matchers
        .iter()
        .chain(expr.matchers.or_matchers.iter().flatten())
        .filter(|tf| !tf.label.is_empty() && !tf.is_regexp() && !tf.is_negative())
        .map(|tf| Label {
            name: tf.label.to_string(),
            value: tf.value.clone(),
        })
        .collect()
}

pub(super) fn is_inf(x: f64, sign: i8) -> bool {
    match sign.cmp(&0_i8) {
        Ordering::Greater => x == f64::INFINITY,
        Ordering::Less => x == f64::NEG_INFINITY,
        Ordering::Equal => x.is_infinite(),
    }
}
