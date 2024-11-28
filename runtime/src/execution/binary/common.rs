use ahash::{AHashMap, AHashSet};
use regex::escape;

use metricsql_parser::prelude::{BinaryExpr, Expr, LabelFilter, Operator};

use crate::types::{Label, Timeseries};

pub(crate) fn can_push_down_common_filters(be: &BinaryExpr) -> bool {
    if be.op == Operator::Or || be.op == Operator::Default {
        return false;
    }
    match (&be.left.as_ref(), &be.right.as_ref()) {
        (Expr::Aggregation(left), Expr::Aggregation(right)) => {
            if left.is_non_grouping() || right.is_non_grouping() {
                return false;
            }
            true
        }
        _ => true,
    }
}

pub(crate) fn get_common_label_filters(tss: &[Timeseries]) -> Vec<LabelFilter> {
    let mut kv_map: AHashMap<String, AHashSet<String>> = AHashMap::new();
    for ts in tss.iter() {
        for Label { name: k, value: v } in ts.metric_name.labels.iter() {
            kv_map
                .entry(k.to_string())
                .or_default()
                .insert(v.to_string());
        }
    }

    let mut lfs: Vec<LabelFilter> = Vec::with_capacity(kv_map.len());
    for (key, values) in kv_map {
        if values.len() != tss.len() {
            // Skip the tag, since it doesn't belong to all the time series.
            continue;
        }

        if values.len() > 1000 {
            // Skip the filter on the given tag, since it needs to enumerate too many unique values.
            // This may slow down the provider for matching time series.
            continue;
        }

        let vals: Vec<&String> = values.iter().collect::<Vec<_>>();

        let lf = if values.len() == 1 {
            LabelFilter::equal(key, vals[0].into())
        } else {
            let str_value = join_regexp_values(&vals);
            LabelFilter::regex_equal(key, str_value).unwrap()
        };

        lfs.push(lf);
    }
    // todo(perf): does this need to be sorted ?
    lfs.sort();
    lfs
}

fn join_regexp_values(a: &[&String]) -> String {
    let init_size = a.iter().fold(0, |res, x| res + x.len() + 3);
    let mut res = String::with_capacity(init_size);
    for (i, s) in a.iter().enumerate() {
        let s_quoted = escape(s);
        res.push_str(s_quoted.as_str());
        if i < a.len() - 1 {
            res.push('|')
        }
    }
    res
}