use std::sync::Arc;
use std::vec;

use crate::common::math::round_to_decimal_digits;
use crate::execution::context::Context;
use crate::execution::parser_cache::{ParseCacheResult, ParseCacheValue};
use crate::execution::EvalConfig;
use crate::provider::QueryResult;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::{QueryValue, Timeseries};
use ahash::AHashSet;
use chrono::Utc;
use tracing::{field, info, trace_span, Span};
use metricsql_common::hash::Signature;
use metricsql_common::time::current_time_millis;

pub(crate) fn parse_promql_internal(
    context: &Context,
    query: &str,
) -> RuntimeResult<Arc<ParseCacheValue>> {
    let span = trace_span!("parse", cached = field::Empty).entered();
    let (parsed, cached) = context.parse_promql(query)?;
    span.record("cached", cached == ParseCacheResult::CacheHit);
    Ok(parsed)
}

pub(crate) fn exec_internal(
    context: &Context,
    ec: &mut EvalConfig,
    q: &str,
) -> RuntimeResult<(QueryValue, Arc<ParseCacheValue>)> {
    let start_time = current_time_millis();
    if context.stats_enabled() {
        defer! {
            context.query_stats.register_query(q, ec.end - ec.start, start_time)
        }
    }

    ec.validate()?;

    let parsed = parse_promql_internal(context, q)?;

    match (&parsed.eval_node, &parsed.has_subquery) {
        (Some(eval_node), has_subquery) => {
            if *has_subquery {
                let _ = ec.get_timestamps()?;
            }

            let qid = context
                .active_queries
                .register(ec, q, Some(start_time));

            defer! {
                context.active_queries.remove(qid);
            }

            let is_tracing = context.trace_enabled();

            let span = if is_tracing {
                let mut query = q.to_string();
                query.truncate(300);

                trace_span!(
                    "execution",
                    query,
                    may_cache = ec.may_cache(),
                    start = ec.start,
                    end = ec.end,
                    series = field::Empty,
                    points = field::Empty,
                    points_per_series = field::Empty
                )
            } else {
                Span::none()
            }
            .entered();

            // DAGNodes can be stateful, therefore we need to clone the node before
            // executing it.
            // todo: possible optimization: if we're not from a cache, we can avoid cloning
            let mut node = eval_node.clone();

            let rv = node.execute(context, ec)?;

            if is_tracing {
                let ts_count: usize;
                let series_count: usize;
                match &rv {
                    QueryValue::RangeVector(iv) | QueryValue::InstantVector(iv) => {
                        series_count = iv.len();
                        if series_count > 0 {
                            ts_count = iv[0].timestamps.len();
                        } else {
                            ts_count = 0;
                        }
                    }
                    _ => {
                        ts_count = ec.data_points();
                        series_count = 1;
                    }
                }
                let mut points_per_series = 0;
                if series_count > 0 {
                    points_per_series = ts_count
                }

                let points_count = series_count * points_per_series;
                span.record("series", series_count);
                span.record("points", points_count);
                span.record("points_per_series", points_per_series);
            }

            Ok((rv, Arc::clone(&parsed)))
        }
        _ => {
            panic!("Bug: Invalid parse state")
        }
    }
}

/// executes q for the given config.
pub fn exec(
    context: &Context,
    ec: &mut EvalConfig,
    q: &str,
    is_first_point_only: bool,
) -> RuntimeResult<Vec<QueryResult>> {
    let (rv, parsed) = exec_internal(context, ec, q)?;

    // we ignore empty timeseries
    if let QueryValue::Scalar(val) = rv {
        if val.is_nan() {
            return Ok(vec![]);
        }
    }

    let mut rv = rv.into_instant_vector(ec)?;
    remove_empty_series(&mut rv);
    if rv.is_empty() {
        return Ok(vec![]);
    }

    if is_first_point_only {
        if rv[0].timestamps.len() > 0 {
            let timestamps = Arc::new(vec![rv[0].timestamps[0]]);
            // Remove all the points except the first one from every time series.
            for ts in rv.iter_mut() {
                ts.values.resize(1, f64::NAN);
                ts.timestamps = Arc::clone(&timestamps);
            }
        } else {
            return Ok(vec![]);
        }
    }

    let mut result = timeseries_to_result(&mut rv, parsed.sort_results)?;

    let n = ec.round_digits;
    if n < 100 {
        for r in result.iter_mut() {
            for v in r.values.iter_mut() {
                *v = round_to_decimal_digits(*v, n as i16);
            }
        }
    }

    info!(
        "sorted = {}, round_digits = {}",
        parsed.sort_results, ec.round_digits
    );

    Ok(result)
}

pub(crate) fn timeseries_to_result(
    tss: &mut Vec<Timeseries>,
    may_sort: bool,
) -> RuntimeResult<Vec<QueryResult>> {
    remove_empty_series(tss);
    if tss.is_empty() {
        return Ok(vec![]);
    }

    let mut result: Vec<QueryResult> = Vec::with_capacity(tss.len());
    let mut m: AHashSet<Signature> = AHashSet::with_capacity(tss.len());

    for ts in tss.iter_mut() {
        ts.metric_name.sort_labels();

        let key = ts.metric_name.signature();

        if m.insert(key) {
            let res = QueryResult {
                metric: ts.metric_name.clone(), // todo(perf) into vs clone/take
                values: ts.values.clone(),      // perf) .into vs clone/take
                timestamps: ts.timestamps.as_ref().clone(), // todo(perf): declare field as Rc<Vec<i64>>
            };

            result.push(res);
        } else {
            return Err(RuntimeError::from(format!(
                "duplicate output timeseries: {}",
                ts.metric_name
            )));
        }
    }

    if may_sort {
        result.sort_by(|a, b| a.metric.partial_cmp(&b.metric).unwrap())
    }

    Ok(result)
}

#[inline]
pub(crate) fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    if tss.is_empty() {
        return;
    }
    tss.retain(|ts| !ts.is_all_nans());
}