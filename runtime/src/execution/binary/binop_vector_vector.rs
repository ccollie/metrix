use std::borrow::Cow;
use std::collections::hash_map::Entry;
use ahash::AHashMap;
use tracing::{field, trace_span, Span};
use metricsql_common::hash::Signature;
use metricsql_parser::ast::{Operator, VectorMatchCardinality, VectorMatchModifier};
use metricsql_parser::binaryop::{
    get_scalar_binop_handler, get_scalar_comparison_handler, BinopFunc,
};
use metricsql_parser::prelude::{BinModifier, Labels};
use crate::execution::Context;
use crate::execution::utils::{remove_empty_series, series_len};
use crate::prelude::QueryValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::{
    group_series_by_match_modifier,
    Timeseries,
    InstantVector,
    TimeseriesHashMap,
    METRIC_NAME_LABEL
};

pub struct BinaryOpFuncArg<'a> {
    op: Operator,
    modifier: &'a Option<BinModifier>,
    left: InstantVector,
    right: InstantVector,
}

impl<'a> BinaryOpFuncArg<'a> {
    pub fn new(
        left: InstantVector,
        op: Operator,
        right: InstantVector,
        modifier: &'a Option<BinModifier>,
    ) -> Self {
        Self {
            left,
            right,
            op,
            modifier,
        }
    }

    pub fn returns_bool(&self) -> bool {
        matches!(self.modifier, Some(modifier) if modifier.return_bool)
    }
}

pub type BinaryOpFuncResult = RuntimeResult<InstantVector>;

macro_rules! make_binary_func {
    ($name: ident, $op: expr) => {
        fn $name(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<InstantVector> {
            const FUNC: BinopFunc = get_scalar_binop_handler($op, false);
            binary_op_func_impl(FUNC, bfa)
        }
    };
}

make_binary_func!(binary_op_add, Operator::Add);
make_binary_func!(binary_op_atan2, Operator::Atan2);
make_binary_func!(binary_op_sub, Operator::Sub);
make_binary_func!(binary_op_mul, Operator::Mul);
make_binary_func!(binary_op_div, Operator::Div);
make_binary_func!(binary_op_mod, Operator::Mod);
make_binary_func!(binary_op_pow, Operator::Pow);

fn binary_op_comparison(bfa: &mut BinaryOpFuncArg) -> BinaryOpFuncResult {
    let bf = get_scalar_comparison_handler(bfa.op, bfa.returns_bool());
    binary_op_func_impl(bf, bfa)
}

pub fn exec_vector_vector_binop(
    ctx: &Context,
    left: InstantVector,
    right: InstantVector,
    op: Operator,
    modifier: &Option<BinModifier>,
) -> RuntimeResult<QueryValue> {
    let is_tracing = ctx.trace_enabled();

    let span = if is_tracing {
        trace_span!("binary op", "op" = op.as_str(), series = field::Empty)
    } else {
        Span::none()
    }.entered();

    let mut bfa = BinaryOpFuncArg::new(left, op, right, modifier);

    let result = exec_binop(&mut bfa).map(QueryValue::InstantVector)?;

    if is_tracing {
        let series_count = series_len(&result);
        span.record("series", series_count);
    }

    Ok(result)
}

pub(crate) fn exec_binop(bfa: &mut BinaryOpFuncArg) -> BinaryOpFuncResult {
    use Operator::*;
    match bfa.op {
        Add => binary_op_add(bfa),
        Atan2 => binary_op_atan2(bfa),
        Sub => binary_op_sub(bfa),
        Mul => binary_op_mul(bfa),
        Div => binary_op_div(bfa),
        Mod => binary_op_mod(bfa),
        Pow => binary_op_pow(bfa),
        // comparison operators
        Eql | NotEq | Gt | Gte | Lt | Lte => binary_op_comparison(bfa),
        And => binary_op_and(bfa),
        Or => binary_op_or(bfa),
        Unless => binary_op_unless(bfa),
        If => binary_op_if(bfa),
        IfNot => binary_op_if_not(bfa),
        Default => binary_op_default(bfa),
    }
}

fn binary_op_func_impl(bf: BinopFunc, bfa: &mut BinaryOpFuncArg) -> RuntimeResult<InstantVector> {
    if bfa.left.is_empty() || bfa.right.is_empty() {
        return Ok(vec![]);
    }

    // todo: should this also be applied to scalar/vector and vector/scalar?
    if bfa.op.is_comparison() {
        // Do not remove empty series for comparison operations,
        // since this may lead to missing result.
    } else {
        remove_empty_series(&mut bfa.left);
        remove_empty_series(&mut bfa.right);
    }

    let (mut left, mut right) = adjust_binary_op_tags(bfa)?;
    if left.len() != right.len() {
        return Err(RuntimeError::InvalidState(format!(
            "BUG: left.len() must match right.len(); got {} vs {}",
            left.len(),
            right.len(),
        )));
    }

    let is_right = is_group_right(bfa.modifier);

    // todo: use rayon for items over a given length
    for (left_ts, right_ts) in left.iter_mut().zip(right.iter_mut()) {
        if left_ts.values.len() != right_ts.values.len() {
            let msg = format!(
                "BUG: left_values.len() must match right_values.len(); got {} vs {}",
                left_ts.values.len(),
                right_ts.values.len()
            );
            return Err(RuntimeError::InvalidState(msg));
        }

        // todo: how to simplify this?
        if is_right {
            for (left_val, right_val) in left_ts.values.iter_mut().zip(right_ts.values.iter_mut()) {
                *right_val = bf(*left_val, *right_val);
            }
        } else {
            for (left_val, right_val) in left_ts.values.iter_mut().zip(right_ts.values.iter_mut()) {
                *left_val = bf(*left_val, *right_val);
            }
        }
    }

    // do not remove time series containing only NaNs, since then the `(foo op bar) default N`
    // won't work as expected if `(foo op bar)` results in NaN series.
    if is_right {
        Ok(right)
    } else {
        Ok(left)
    }
}

fn adjust_binary_op_tags(
    bfa: &mut BinaryOpFuncArg,
) -> RuntimeResult<(
    InstantVector, // left
    InstantVector, // right
)> {
    // `vector op vector` or `foo op {on|ignoring} {group_left|group_right} bar`
    let (mut m_left, mut m_right) = create_series_map_by_tag_set(bfa);

    // TODO: I think if we wanted we could reuse bfa.left and bfa.right here
    let mut rvs_left = std::mem::take(&mut bfa.left);
    let mut rvs_right = std::mem::take(&mut bfa.right);

    let card = VectorMatchCardinality::OneToOne;
    let none_matching = None;

    let (matching, grouping, keep_metric_names, return_bool) = if let Some(modifier) = bfa.modifier
    {
        let keep_metric_names = modifier.keep_metric_names;
        (
            &modifier.matching,
            &modifier.card,
            keep_metric_names,
            modifier.return_bool,
        )
    } else {
        (&none_matching, &card, false, false)
    };

    let reset_metric_group = should_reset_metric_group(bfa.op, keep_metric_names, return_bool);

    for (k, tss_left) in m_left.iter_mut() {
        let mut tss_right = m_right.remove(k).unwrap_or(vec![]);
        if tss_right.is_empty() {
            continue;
        }

        match grouping {
            // group_left
            VectorMatchCardinality::ManyToOne(_) => group_join(
                "right",
                bfa,
                reset_metric_group,
                &mut rvs_left,
                &mut rvs_right,
                tss_left,
                &mut tss_right,
            )?,
            // group_right
            VectorMatchCardinality::OneToMany(_) => group_join(
                "left",
                bfa,
                reset_metric_group,
                &mut rvs_right,
                &mut rvs_left,
                &mut tss_right,
                tss_left,
            )?,
            _ => {
                let mut is_on = false;
                let group_tags = match matching {
                    Some(VectorMatchModifier::On(labels)) => {
                        is_on = true;
                        // Add `__name__` to group_tags if metric name must be preserved.
                        if keep_metric_names {
                            let mut changed = labels.clone();
                            changed.push(METRIC_NAME_LABEL.to_string());
                            changed.sort();
                            Cow::Owned(changed)
                        } else {
                            Cow::Borrowed(labels)
                        }
                    }
                    Some(VectorMatchModifier::Ignoring(labels)) => Cow::Borrowed(labels),
                    None => Cow::Owned(Labels::default()),
                };

                let mut ts_left = ensure_single_timeseries("left", bfa.op, bfa.modifier, tss_left)?;
                let ts_right =
                    ensure_single_timeseries("right", bfa.op, bfa.modifier, &mut tss_right)?;

                if reset_metric_group {
                    ts_left.metric_name.reset_measurement();
                }

                let labels = group_tags.as_ref();
                if is_on {
                    ts_left.metric_name.remove_labels_on(labels.as_ref());
                } else {
                    ts_left.metric_name.remove_labels_ignoring(labels.as_ref());
                }

                rvs_left.push(ts_left);
                rvs_right.push(ts_right);
            }
        }
    }

    Ok((rvs_left, rvs_right))
}

fn should_reset_metric_group(op: Operator, keep_metric_names: bool, returns_bool: bool) -> bool {
    if op.is_comparison() && !returns_bool {
        // Do not reset metric_group for non-boolean `compare` binary ops like Prometheus does.
        return false;
    }
    if keep_metric_names {
        // Do not reset metric_group if it is explicitly requested via `foo op bar keep_metric_names`
        // See https://docs.victoriametrics.com/MetricsQL.html#keep_metric_names
        return false;
    }

    // in the original code, the metric group is not reset for logical ops
    if op.is_logical_op() {
        return false;
    }

    true
}

fn ensure_single_timeseries(
    side: &str,
    op: Operator,
    modifier: &Option<BinModifier>,
    tss: &mut Vec<Timeseries>,
) -> RuntimeResult<Timeseries> {
    if tss.is_empty() {
        return Err(RuntimeError::General(
            "BUG: tss must contain at least one value".to_string(),
        ));
    }

    let mut acc = tss.pop().unwrap();

    for ts in tss.iter() {
        if !merge_non_overlapping_timeseries(&mut acc, ts) {
            let msg = format!(
                "duplicate time series on the {side} side of {} {}: {} and {}",
                op,
                group_modifier_to_string(modifier),
                acc.metric_name,
                ts.metric_name
            );

            return Err(RuntimeError::from(msg));
        }
    }

    Ok(acc)
}

fn group_join(
    single_timeseries_side: &str,
    bfa: &BinaryOpFuncArg,
    reset_metric_group: bool,
    rvs_left: &mut InstantVector,
    rvs_right: &mut InstantVector,
    tss_left: &mut [Timeseries],
    tss_right: &mut InstantVector,
) -> RuntimeResult<()> {
    let empty_prefix = "";
    let empty_labels = Labels::default();

    let (join_tags, skip_tags) = if let Some(modifier) = bfa.modifier {
        let join = match &modifier.card {
            VectorMatchCardinality::ManyToOne(labels) |
            VectorMatchCardinality::OneToMany(labels) => labels,
            _ => &empty_labels,
        };
        let skip = if let Some(VectorMatchModifier::On(labels)) = &modifier.matching {
            labels
        } else {
            &empty_labels
        };
        (join, skip)
    } else {
        (&empty_labels, &empty_labels)
    };

    struct TsPair {
        left: Timeseries,
        right: Timeseries,
    }

    // todo(perf): IntMap
    let mut map: AHashMap<Signature, TsPair> = AHashMap::with_capacity(tss_left.len());

    for ts_left in tss_left.iter_mut() {
        if reset_metric_group {
            ts_left.metric_name.reset_measurement();
        }

        // Easy case - right part contains only a single matching time series.
        if tss_right.len() == 1 {
            let mut right = tss_right.remove(0);
            ts_left.metric_name.set_labels(
                empty_prefix,
                join_tags.as_ref(),
                skip_tags.as_ref(),
                &mut right.metric_name,
            );

            rvs_left.push(std::mem::take(ts_left));
            rvs_right.push(right);
            continue;
        }

        // Hard case - right part contains multiple matching time series.
        // Verify it doesn't result in duplicate MetricName values after adding missing tags.
        map.clear();

        for mut ts_right in tss_right.drain(..) {
            let mut ts_copy = ts_left.clone(); // todo(perf): how to avoid clone ?
            ts_copy.metric_name.set_labels(
                empty_prefix,
                join_tags.as_ref(),
                skip_tags.as_ref(),
                &mut ts_right.metric_name,
            );

            let key = ts_copy.metric_name.labels_signature();

            match map.entry(key) {
                Entry::Vacant(entry) => {
                    entry.insert(TsPair {
                        left: ts_copy,
                        right: ts_right,
                    });
                }
                Entry::Occupied(entry) => {
                    let pair = entry.into_mut();
                    // Try merging pair.right with ts_right if they don't overlap.
                    if !merge_non_overlapping_timeseries(&mut pair.right, &ts_right) {
                        let err = format!(
                            "duplicate time series on the {} side of `{} {} {}`: {} and {}",
                            single_timeseries_side,
                            bfa.op,
                            group_modifier_to_string(bfa.modifier),
                            join_modifier_to_string(bfa.modifier),
                            pair.right.metric_name,
                            ts_right.metric_name
                        );
                        return Err(RuntimeError::from(err));
                    }
                }
            }
        }

        for (_, pair) in map.iter_mut() {
            rvs_left.push(std::mem::take(&mut pair.left));
            rvs_right.push(std::mem::take(&mut pair.right));
        }
    }

    Ok(())
}

pub fn merge_non_overlapping_timeseries(dst: &mut Timeseries, src: &Timeseries) -> bool {
    // Verify whether the time series can be merged.
    let mut overlaps = 0;

    for (src_val, dst_val) in src.values.iter().zip(dst.values.iter()) {
        if src_val.is_nan() {
            continue;
        }
        if !dst_val.is_nan() {
            overlaps += 1;
            // Allow up to two overlapping data points, which can appear due to staleness algorithm,
            // which can add a few data points in the end of time series.
            if overlaps > 2 {
                return false;
            }
        }
    }

    // do not merge time series with too small number of data points.
    // This can be the case during evaluation of instant queries (alerting or recording rules).
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1141
    if src.values.len() <= 2 && dst.values.len() <= 2 {
        return false;
    }
    // Time series can be merged. Merge them.
    for (src_val, dst_val) in src.values.iter().zip(dst.values.iter_mut()) {
        if !src_val.is_nan() {
            *dst_val = *src_val
        }
    }
    true
}

fn binary_op_if(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<InstantVector> {
    let (mut m_left, m_right) = create_series_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m_left.len());

    for (k, tss_left) in m_left.iter_mut() {
        if let Some(tss_right) = series_by_key(&m_right, k) {
            add_right_nans_to_left(tss_left, tss_right);
            rvs.append(tss_left)
        }
    }

    Ok(rvs)
}

/// `vector1 and vector2` results in a vector consisting of the elements of vector1 for which there
/// are elements in vector2 with exactly matching label sets.
/// Other elements are dropped. The metric name and values are carried over from the left-hand side vector.
///
/// https://prometheus.io/docs/prometheus/latest/querying/operators/#logical-set-binary-operators
fn binary_op_and(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<InstantVector> {
    if bfa.left.is_empty() || bfa.right.is_empty() {
        return Ok(vec![]); // Short-circuit: AND with nothing is nothing.
    }

    let (mut left_sigs, right_sigs) = create_series_map_by_tag_set(bfa);

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(left_sigs.len());

    for (sig, tss_right) in right_sigs.into_iter() {
        if let Some(tss_left) = left_sigs.get_mut(&sig) {
            // Add gaps to tss_left if there are gaps at tss_right.
            add_right_nans_to_left(tss_left, &tss_right);
            rvs.append(tss_left);
        }
    }

    Ok(rvs)
}

fn binary_op_default(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<InstantVector> {
    if bfa.left.is_empty() {
        return Ok(std::mem::take(&mut bfa.right)); // Short-circuit: default with nothing is nothing.
    }

    let (mut m_left, m_right) = create_series_map_by_tag_set(bfa);

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m_left.len());
    for (k, tss_left) in m_left.iter_mut() {
        if let Some(tss_right) = series_by_key(&m_right, k) {
            fill_left_nans_with_right_values(tss_left, tss_right);
        }
        rvs.append(tss_left);
    }

    Ok(rvs)
}

fn sort_series_by_metric_name(tss: &mut [Timeseries]) {
    tss.sort_by(|a, b| a.metric_name.cmp(&b.metric_name));
}

/// `vector1 or vector2` results in a vector that contains all original elements (label sets + values)
/// of vector1 and additionally all elements of vector2 which do not have matching label sets in vector1.
///
/// https://prometheus.io/docs/prometheus/latest/querying/operators/#logical-set-binary-operators
fn binary_op_or(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<InstantVector> {
    if bfa.left.is_empty() {
        // Short-circuit.
        sort_series_by_metric_name(&mut bfa.right);
        return Ok(std::mem::take(&mut bfa.right));
    }

    if bfa.right.is_empty() {
        // Short-circuit.
        sort_series_by_metric_name(&mut bfa.left);
        return Ok(std::mem::take(&mut bfa.left));
    }

    let (mut m_left, m_right) = create_series_map_by_tag_set(bfa);

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m_right.len());

    for (sig, ref mut tss_right) in m_right.into_iter() {
        if let Some(tss_left) = m_left.get_mut(&sig) {
            fill_left_nans_with_right_values(tss_left, tss_right);
        } else {
            // add right if it is not in left
            rvs.append(tss_right);
        };
    }

    // todo(perf): there is a lot of copying and moving here, can we avoid it?
    let mut left = m_left.into_values().flatten().collect::<Vec<Timeseries>>();
    if rvs.is_empty() {
        return Ok(left);
    }

    left.append(&mut rvs);
    // Sort series by metric name as Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/5393
    sort_series_by_metric_name(&mut left);

    Ok(left)
}

/// `vector1 unless vector2` results in a vector consisting of the elements of vector1
/// for which there are no elements in vector2 with exactly matching label sets.
/// All matching elements in both vectors are dropped.
///
/// https://prometheus.io/docs/prometheus/latest/querying/operators/#logical-set-binary-operators
fn binary_op_unless(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<InstantVector> {
    // If right is empty, we simply return the left
    // if left is empty we will return it anyway.
    if bfa.right.is_empty() || bfa.left.is_empty() {
        return Ok(std::mem::take(&mut bfa.left));
    }

    let (m_left, m_right) = create_series_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m_left.len());

    for (k, mut tss_left) in m_left.into_iter() {
        if let Some(tss_right) = m_right.get(&k) {
            // Add gaps to tss_left if there are no gaps at tss_right.
            add_left_nans_if_no_right_nans(&mut tss_left, tss_right);
        }
        rvs.append(&mut tss_left);
    }

    Ok(rvs)
}

/// q1 ifnot q2 removes values from q1 for existing values from q2.
fn binary_op_if_not(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<InstantVector> {
    let (m_left, m_right) = create_series_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m_left.len());

    for (k, mut tss_left) in m_left.into_iter() {
        if let Some(tss_right) = series_by_key(&m_right, &k) {
            add_left_nans_if_no_right_nans(&mut tss_left, tss_right);
        }
        rvs.extend(tss_left.into_iter());
    }

    Ok(rvs)
}

#[inline]
/// Fill gaps in tss_left with values from tss_right as Prometheus does.
/// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/552
fn fill_left_nans_with_right_values(tss_left: &mut [Timeseries], tss_right: &[Timeseries]) {
    for ts_left in tss_left.iter_mut() {
        for (i, left_value) in ts_left.values.iter_mut().enumerate() {
            if !left_value.is_nan() {
                continue;
            }
            for ts_right in tss_right.iter() {
                let v_right = ts_right.values[i];
                if !v_right.is_nan() {
                    *left_value = v_right;
                    break;
                }
            }
        }
    }
}

fn add_right_nans_to_left(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    for ts_left in tss_left.iter_mut() {
        for i in 0..ts_left.values.len() {
            let mut has_value = false;
            for ts_right in tss_right {
                if !ts_right.values[i].is_nan() {
                    has_value = true;
                    break;
                }
            }
            if !has_value {
                ts_left.values[i] = f64::NAN
            }
        }
    }

    remove_empty_series(tss_left)
}

fn add_left_nans_if_no_right_nans(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    for ts_left in tss_left.iter_mut() {
        for (i, left_value) in ts_left.values.iter_mut().enumerate() {
            for ts_right in tss_right {
                if !ts_right.values[i].is_nan() {
                    *left_value = f64::NAN;
                    break;
                }
            }
        }
    }

    remove_empty_series(tss_left);
}

fn create_series_map_by_tag_set(
    bfa: &mut BinaryOpFuncArg,
) -> (TimeseriesHashMap, TimeseriesHashMap) {
    let empty_matching = None;

    let matching = if let Some(modifier) = bfa.modifier.as_ref() {
        &modifier.matching
    } else {
        &empty_matching
    };

    let m_left = group_series_by_match_modifier(&mut bfa.left, matching, false);
    let m_right = group_series_by_match_modifier(&mut bfa.right, matching, false);

    (m_left, m_right)
}

fn is_scalar(arg: &[Timeseries]) -> bool {
    arg.len() == 1 && arg[0].metric_name.is_empty()
}

fn series_by_key<'a>(m: &'a TimeseriesHashMap, key: &Signature) -> Option<&'a Vec<Timeseries>> {
    if let Some(v) = m.get(key) {
        return Some(v);
    }
    if m.len() != 1 {
        return None;
    }
    if let Some(tss) = m.values().next() {
        if is_scalar(tss) {
            return Some(tss);
        }
    }
    None
}

fn is_group_right(modifier: &Option<BinModifier>) -> bool {
    matches!(modifier, Some(modifier) if modifier.card.is_group_right())
}

fn group_modifier_to_string(modifier: &Option<BinModifier>) -> String {
    if let Some(modifier) = modifier {
        if let Some(matching) = &modifier.matching {
            return matching.to_string();
        }
    }
    "None".to_string()
}

fn join_modifier_to_string(modifier: &Option<BinModifier>) -> String {
    if let Some(modifier) = modifier {
        match modifier.card {
            VectorMatchCardinality::ManyToOne(_) | VectorMatchCardinality::OneToMany(_) => {
                return modifier.card.to_string()
            }
            _ => {}
        }
    }
    "None".to_string()
}
