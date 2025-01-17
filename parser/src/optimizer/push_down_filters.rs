use crate::ast::{
    AggregateModifier, AggregationExpr, BinaryExpr, Expr, Operator, RollupExpr, VectorMatchModifier,
};
use crate::functions::BuiltinFunction::Transform;
use crate::functions::{AggregateFunction, BuiltinFunction, RollupFunction, TransformFunction};
use crate::label::{Matcher, Matchers, NAME_LABEL};
use crate::parser::{ParseError, ParseResult};
use crate::prelude::{can_accept_multiple_args_for_aggr_func, VectorMatchCardinality};
use metricsql_common::hash::{FastHashSet, HashSetExt, Signature};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::hash::Hash;
use std::iter::FromIterator;
use std::vec::Vec;

/// `push_down_filters` optimizes e in order to improve its performance.
///
/// It performs the following optimizations:
///
/// - Adds missing filters to `foo{filters1} op bar{filters2}`
///   according to https://utcc.utoronto.ca/~cks/space/blog/sysadmin/PrometheusLabelNonOptimization
pub fn push_down_filters(expr: &Expr) -> Cow<Expr> {
    if can_pushdown_filters(expr) {
        let mut clone = expr.clone();
        optimize_in_place(&mut clone);
        Cow::Owned(clone)
    } else {
        Cow::Borrowed(expr)
    }
}

pub(crate) fn can_pushdown_filters(expr: &Expr) -> bool {
    use Expr::*;

    match expr {
        Rollup(RollupExpr { expr, at, .. }) => {
            if let Some(at) = at {
                can_pushdown_filters(expr) || can_pushdown_filters(at)
            } else {
                can_pushdown_filters(expr)
            }
        }
        Function(f) => f.args.iter().any(can_pushdown_filters),
        Aggregation(agg) => agg.args.iter().any(can_pushdown_filters),
        BinaryOperator(BinaryExpr { left, right, .. }) => {
            can_pushdown_filters(left) || can_pushdown_filters(right)
        }
        _ => false,
    }
}

pub(crate) fn optimize_in_place(expr: &mut Expr) {
    use Expr::*;

    match expr {
        Rollup(re) => {
            optimize_in_place(&mut re.expr);
            if let Some(ref mut at) = re.at {
                optimize_in_place(at.as_mut());
            }
        }
        Function(f) => {
            for arg in f.args.iter_mut() {
                optimize_in_place(arg);
            }
        }
        Aggregation(agg) => {
            for arg in agg.args.iter_mut() {
                optimize_in_place(arg);
            }
        }
        BinaryOperator(be) => {
            optimize_in_place(&mut be.left);
            optimize_in_place(&mut be.right);
            let mut lfs = get_common_label_filters(expr);
            push_down_binary_op_filters_in_place(expr, &mut lfs);
        }
        _ => {}
    }
}

pub fn get_common_label_filters(e: &Expr) -> Vec<Matcher> {
    use Expr::*;

    match e {
        MetricExpression(m) => get_common_label_filters_without_metric_name(&m.matchers),
        Rollup(r) => get_common_label_filters(&r.expr),
        Function(fe) => {
            use RollupFunction::*;
            use TransformFunction::*;

            match fe.function {
                BuiltinFunction::Rollup(rf) => {
                    if rf == CountValuesOverTime {
                        return get_common_label_filters_for_count_values_over_time(&fe.args);
                    }
                }
                Transform(tf) => match tf {
                    LabelSet => return get_common_label_filters_for_label_set(&fe.args),
                    LabelMap | LabelJoin | LabelMatch | LabelMismatch | LabelReplace
                    | LabelTransform => {
                        return get_common_label_filters_for_label_replace(&fe.args)
                    }
                    LabelCopy | LabelMove => {
                        return get_common_label_filters_for_label_copy(&fe.args)
                    }
                    LabelDel | LabelsEqual | LabelLowercase | LabelUppercase => {
                        return get_common_label_filters_for_label_del(&fe.args)
                    }
                    LabelKeep => return get_common_label_filters_for_label_keep(&fe.args),
                    RangeNormalize | Union => {
                        return intersect_label_filters_for_all_args(&fe.args)
                    }
                    _ => {}
                },
                _ => {}
            }
            if let Some(arg) = fe.arg_for_optimization() {
                get_common_label_filters(arg)
            } else {
                vec![]
            }
        }
        Aggregation(agg) => {
            use AggregateFunction::*;
            if agg.function == CountValues {
                if agg.args.len() != 2 {
                    return vec![];
                }
                let mut lfs = get_common_label_filters(&agg.args[1]);
                lfs = drop_label_filters_for_label_name(&lfs, &agg.args[0]);
                trim_filters_by_aggr_modifier(&mut lfs, agg);
                return lfs;
            }
            if can_accept_multiple_args_for_aggr_func(agg.function) {
                let mut lfs = intersect_label_filters_for_all_args(&agg.args);
                trim_filters_by_aggr_modifier(&mut lfs, agg);
                return lfs;
            }
            let func = BuiltinFunction::Aggregate(agg.function);
            if let Ok(Some(argument)) = get_func_arg_for_optimization(func, &agg.args) {
                let mut filters = get_common_label_filters(argument);
                trim_filters_by_aggr_modifier(&mut filters, agg);
                filters
            } else {
                vec![]
            }
        }
        UnaryOperator(unary) => get_common_label_filters(&unary.expr),
        BinaryOperator(binary) => {
            let mut lfs_left = get_common_label_filters(&binary.left);
            let mut lfs_right = get_common_label_filters(&binary.right);
            let card = VectorMatchCardinality::OneToOne;
            let group_modifier: Option<VectorMatchModifier> = None;

            let (group_modifier, join_modifier) = if let Some(modifier) = &binary.modifier {
                (&modifier.matching, &modifier.card)
            } else {
                (&group_modifier, &card)
            };

            match binary.op {
                Operator::Or => {
                    // {fCommon, f1} or {fCommon, f2} -> {fCommon}
                    // {fCommon, f1} or on() {fCommon, f2} -> {}
                    // {fCommon, f1} or on(fCommon) {fCommon, f2} -> {fCommon}
                    // {fCommon, f1} or on(f1) {fCommon, f2} -> {}
                    // {fCommon, f1} or on(f2) {fCommon, f2} -> {}
                    // {fCommon, f1} or on(f3) {fCommon, f2} -> {}
                    lfs_left = intersect_label_filters(lfs_left, lfs_right);
                    trim_filters_by_match_modifier(&mut lfs_left, group_modifier);
                    lfs_left
                }
                Operator::Unless => {
                    // {f1} unless {f2} -> {f1}
                    // {f1} unless on() {f2} -> {}
                    // {f1} unless on(f1) {f2} -> {f1}
                    // {f1} unless on(f2) {f2} -> {}
                    // {f1} unless on(f1, f2) {f2} -> {f1}
                    // {f1} unless on(f3) {f2} -> {}
                    trim_filters_by_match_modifier(&mut lfs_left, group_modifier);
                    lfs_left
                }
                _ => {
                    match join_modifier {
                        // group_left
                        VectorMatchCardinality::ManyToOne(_) => {
                            // {f1} * group_left() {f2} -> {f1, f2}
                            // {f1} * on() group_left() {f2} -> {f1}
                            // {f1} * on(f1) group_left() {f2} -> {f1}
                            // {f1} * on(f2) group_left() {f2} -> {f1, f2}
                            // {f1} * on(f1, f2) group_left() {f2} -> {f1, f2}
                            // {f1} * on(f3) group_left() {f2} -> {f1}
                            trim_filters_by_match_modifier(&mut lfs_right, group_modifier);
                            union_label_filters(lfs_left, lfs_right)
                        }
                        // group_right
                        VectorMatchCardinality::OneToMany(_) => {
                            // {f1} * group_right() {f2} -> {f1, f2}
                            // {f1} * on() group_right() {f2} -> {f2}
                            // {f1} * on(f1) group_right() {f2} -> {f1, f2}
                            // {f1} * on(f2) group_right() {f2} -> {f2}
                            // {f1} * on(f1, f2) group_right() {f2} -> {f1, f2}
                            // {f1} * on(f3) group_right() {f2} -> {f2}
                            trim_filters_by_match_modifier(&mut lfs_left, group_modifier);
                            union_label_filters(lfs_left, lfs_right)
                        }
                        _ => {
                            // {f1} * {f2} -> {f1, f2}
                            // {f1} * on() {f2} -> {}
                            // {f1} * on(f1) {f2} -> {f1}
                            // {f1} * on(f2) {f2} -> {f2}
                            // {f1} * on(f1, f2) {f2} -> {f2}
                            // {f1} * on(f3} {f2} -> {}
                            lfs_left = union_label_filters(lfs_left, lfs_right);
                            trim_filters_by_match_modifier(&mut lfs_left, group_modifier);
                            lfs_left
                        }
                    }
                }
            }
        }
        _ => {
            vec![]
        }
    }
}

fn intersect_label_filters_for_all_args(args: &[Expr]) -> Vec<Matcher> {
    if args.is_empty() {
        return vec![];
    }
    let mut lfs_a = get_common_label_filters(&args[0]);
    for arg in &args[1..] {
        let lfs_next = get_common_label_filters(arg);
        lfs_a = intersect_label_filters(lfs_a, lfs_next)
    }
    lfs_a
}

fn get_common_label_filters_for_count_values_over_time(args: &[Expr]) -> Vec<Matcher> {
    if args.len() != 2 {
        return vec![];
    }
    let lfs = get_common_label_filters(&args[1]);
    drop_label_filters_for_label_name(&lfs, &args[0])
}

fn get_common_label_filters_for_label_keep(args: &[Expr]) -> Vec<Matcher> {
    if args.is_empty() {
        return vec![];
    }
    let mut lfs = get_common_label_filters(&args[0]);
    keep_label_filters_for_label_names(&mut lfs, args[1..].iter());
    lfs
}

fn get_common_label_filters_for_label_del(args: &[Expr]) -> Vec<Matcher> {
    if args.is_empty() {
        return vec![];
    }
    let lfs = get_common_label_filters(&args[0]);
    drop_label_filters_for_label_names(&lfs, args[1..].iter())
}

/// Gets common label filters for `label_copy`.
fn get_common_label_filters_for_label_copy(args: &[Expr]) -> Vec<Matcher> {
    if args.len() > 1 {
        return Vec::new();
    }

    let lfs = get_common_label_filters(&args[0]);
    let label_names  = args.iter().skip(1).step_by(2);

    drop_label_filters_for_label_names(&lfs, label_names)
}

fn get_common_label_filters_for_label_replace(args: &[Expr]) -> Vec<Matcher> {
    if args.len() < 2 {
        return vec![];
    }
    let lfs = get_common_label_filters(&args[0]);
    drop_label_filters_for_label_name(&lfs, &args[1])
}

fn get_common_label_filters_for_label_set(args: &[Expr]) -> Vec<Matcher> {
    if args.is_empty() {
        return vec![];
    }

    // Get the initial label filters from the first argument
    let mut lfs = get_common_label_filters(&args[0]);

    // Iterate over the remaining arguments in pairs (label name, label value)
    let mut args_iter = args.iter().skip(1);
    while let Some(label_name_expr) = args_iter.next() {
        let label_value = match args_iter.next() {
            Some(expr) => expr,
            None => return vec![], // Missing label value
        };

        let label_name = if let Some(value) = get_expr_as_string(label_name_expr) {
            value
        } else {
            return vec![]; // Invalid label name
        };

        // Skip if the label name is "__name__"
        if NAME_LABEL == label_name {
            continue;
        }

        let label_value = if let Some(value) = get_expr_as_string(label_value) {
            value
        } else {
            return vec![]; // Invalid label value
        };

        // Drop existing filters for the label name
        lfs = drop_label_filters_for_label_name(&lfs, label_name_expr);

        // Add the new label filter
        lfs.push(Matcher::equal(label_name, label_value));
    }

    lfs
}

fn is_string_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::StringExpr(_) | Expr::StringLiteral(_))
}

fn get_expr_as_string(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::StringExpr(se) => {
            if let Ok(Some(s)) = se.get_literal() {
                Some(s.as_str())
            } else {
                None
            }
        },
        Expr::StringLiteral(se) => Some(se.as_str()),
        _ => None,
    }
}

fn trim_filters_by_aggr_modifier(lfs: &mut Vec<Matcher>, afe: &AggregationExpr) {
    match &afe.modifier {
        None => lfs.clear(),
        Some(modifier) => match modifier {
            AggregateModifier::By(args) => filter_label_filters_on(lfs, args),
            AggregateModifier::Without(args) => filter_label_filters_ignoring(lfs, args),
        },
    }
}

/// trims lfs by the specified be.modifier.matching (e.g. on() or ignoring()).
///
/// The following cases are possible:
/// - It returns lfs as is if be doesn't contain any group modifier
/// - It returns only filters specified in on()
/// - It drops filters specified inside ignoring()
pub fn trim_filters_by_match_modifier(
    lfs: &mut Vec<Matcher>,
    group_modifier: &Option<VectorMatchModifier>,
) {
    match group_modifier {
        None => {}
        Some(modifier) => match modifier {
            VectorMatchModifier::On(labels) => filter_label_filters_on(lfs, labels.as_ref()),
            VectorMatchModifier::Ignoring(labels) => {
                filter_label_filters_ignoring(lfs, labels.as_ref())
            }
        },
    }
}

fn get_common_label_filters_without_metric_name(matchers: &Matchers) -> Vec<Matcher> {
    if !matchers.or_matchers.is_empty() {
        let lfss = &matchers.or_matchers;
        let head = &lfss[0];
        let mut lfs_a = get_label_filters_without_metric_name(head);
        for lfs in lfss[1..].iter() {
            if lfs_a.is_empty() {
                return vec![];
            }
            let lfs_b = get_label_filters_without_metric_name(lfs);
            lfs_a = intersect_label_filters(lfs_a, lfs_b);
        }
        return lfs_a;
    }
    if !matchers.matchers.is_empty() {
        return get_label_filters_without_metric_name(&matchers.matchers);
    }
    vec![]
}

// todo: use lifetimes instead of cloning
fn get_label_filters_without_metric_name(lfs: &[Matcher]) -> Vec<Matcher> {
    lfs.iter()
        .filter(|x| x.label != NAME_LABEL)
        .cloned()
        .collect::<Vec<_>>()
}

/// Pushes down the given common_filters to `expr` if possible.
///
/// `expr` must be a part of a binary operation - either left or right.
///
/// For example, if e contains `foo + sum(bar)` and common_filters=`{x="y"}`,
/// then the returned expression will contain `foo{x="y"} + sum(bar)`.
///
/// The `{x="y"}` cannot be pushed down to `sum(bar)`, since this
/// may change binary operation results.
pub fn pushdown_binary_op_filters(
    expr: &Expr,
    common_filters: Vec<Matcher>,
) -> Cow<Expr> {
    // according to pushdown_binary_op_filters_in_place, only the following types need to be
    // handled, so exit otherwise
    if common_filters.is_empty() || !can_pushdown_op_filters(expr) {
        return Cow::Borrowed(expr);
    }

    let mut copy = expr.clone();
    let mut common_filters = common_filters;
    push_down_binary_op_filters_in_place(&mut copy, &mut common_filters);
    Cow::Owned(copy)
}

fn can_pushdown_op_filters(expr: &Expr) -> bool {
    use Expr::*;
    // these are the types handled below in pushdown_binary_op_filters_in_place
    matches!(
        expr,
        MetricExpression(_)
            | Function(_)
            | Rollup(_)
            | BinaryOperator(_)
            | Aggregation(_)
            | UnaryOperator(_)
    )
}

pub fn push_down_binary_op_filters_in_place(e: &mut Expr, common_filters: &mut Vec<Matcher>) {
    use Expr::*;

    if common_filters.is_empty() {
        return;
    }

    match e {
        MetricExpression(me) => {
            if me.matchers.is_empty() {
                return;
            }
            if !me.matchers.matchers.is_empty() {
                union_label_filters_internal(&mut me.matchers.matchers, common_filters);
                // union_label_filters_internal(&mut me.matchers.matchers, &common_filters);
            } else if !me.matchers.or_matchers.is_empty() {
                for matcher in me.matchers.or_matchers.iter_mut() {
                    union_label_filters_internal(matcher, &common_filters);
                }
            } else {
                me.matchers.matchers = common_filters.clone();
            }
            // do we need to sort this ?
            me.matchers.sort_filters();
        }
        Function(fe) => {
            use TransformFunction::*;

            match fe.function {
                BuiltinFunction::Rollup(rf) => {
                    if rf == RollupFunction::CountValuesOverTime {
                        return pushdown_label_filters_for_count_values_over_time(
                            &mut fe.args,
                            common_filters,
                        );
                    }
                }
                Transform(tf) => match tf {
                    LabelSet => {
                        return pushdown_label_filters_for_label_set(&mut fe.args, common_filters)
                    }
                    LabelMap | LabelJoin | LabelMatch | LabelMismatch | LabelReplace | LabelTransform => {
                        return pushdown_label_filters_for_label_replace(
                            &mut fe.args,
                            common_filters,
                        )
                    }
                    LabelCopy | LabelMove => {
                        return pushdown_label_filters_for_label_copy(&mut fe.args, common_filters)
                    }
                    LabelDel | LabelsEqual | LabelLowercase | LabelUppercase => {
                        return pushdown_label_filters_for_label_del(&mut fe.args, common_filters)
                    }
                    LabelKeep => {
                        return pushdown_label_filters_for_label_keep(&mut fe.args, common_filters)
                    }
                    RangeNormalize | Union => {
                        return pushdown_label_filters_for_all_args(common_filters, &mut fe.args)
                    }
                    _ => {}
                },
                _ => {}
            }
            if let Some(idx) = fe.arg_idx_for_optimization() {
                if let Some(val) = fe.args.get_mut(idx) {
                    push_down_binary_op_filters_in_place(val, common_filters);
                }
            }
        }
        UnaryOperator(unary) => {
            push_down_binary_op_filters_in_place(&mut unary.expr, common_filters);
        }
        BinaryOperator(bo) => {
            if let Some(modifier) = &bo.modifier {
                trim_filters_by_match_modifier(common_filters, &modifier.matching);
            }
            push_down_binary_op_filters_in_place(&mut bo.left, common_filters);
            push_down_binary_op_filters_in_place(&mut bo.right, common_filters);
        }
        Aggregation(aggr) => {
            trim_filters_by_aggr_modifier(common_filters, aggr);
            if aggr.function == AggregateFunction::CountValues {
                if aggr.args.len() == 2 {
                    *common_filters = drop_label_filters_for_label_name(&common_filters, &aggr.args[0]);
                    push_down_binary_op_filters_in_place(&mut aggr.args[1], common_filters);
                }
            } else if can_accept_multiple_args_for_aggr_func(aggr.function) {
                pushdown_label_filters_for_all_args(common_filters, &mut aggr.args);
            } else if let Some(arg_idx) = aggr.arg_idx_for_optimization() {
                if let Some(expr) = aggr.args.get_mut(arg_idx) {
                    push_down_binary_op_filters_in_place(expr, common_filters);
                }
            }
        }
        Rollup(re) => {
            push_down_binary_op_filters_in_place(&mut re.expr, common_filters);
        }
        _ => {
        }
    }
}

fn pushdown_label_filters_for_all_args(lfs: &mut Vec<Matcher>, args: &mut [Expr]) {
    for arg in args {
        push_down_binary_op_filters_in_place(arg, lfs)
    }
}

fn pushdown_label_filters_for_count_values_over_time(
    args: &mut [Expr],
    lfs: &mut Vec<Matcher>,
) {
    if args.len() != 2 {
        return;
    }
    *lfs = drop_label_filters_for_label_name(&lfs, &args[0]);
    push_down_binary_op_filters_in_place(&mut args[1], lfs);
}

fn pushdown_label_filters_for_label_keep(args: &mut [Expr], lfs: &mut Vec<Matcher>) {
    if args.is_empty() {
        return;
    }
    keep_label_filters_for_label_names(lfs, args[1..].iter());
    if let Some(arg) = args.get_mut(0) {
        push_down_binary_op_filters_in_place(arg, lfs);
    }
}

fn pushdown_label_filters_for_label_del(args: &mut [Expr], lfs: &mut Vec<Matcher>) {
    if args.is_empty() {
        return;
    }

    *lfs = drop_label_filters_for_label_names(lfs, args[1..].iter());
    if let Some(arg) = args.get_mut(0) {
        push_down_binary_op_filters_in_place(arg, lfs);
    }
}

fn pushdown_label_filters_for_label_copy(args: &mut [Expr], lfs: &mut Vec<Matcher>) {
    if args.is_empty() {
        return;
    }

    let label_names = args.iter()
        .skip(1)
        .step_by(2);

    *lfs = drop_label_filters_for_label_names(lfs, label_names);

    if let Some(arg) = args.get_mut(0) {
        push_down_binary_op_filters_in_place(arg, lfs);
    }
}

fn pushdown_label_filters_for_label_replace(args: &mut [Expr], lfs: &mut Vec<Matcher>) {
    if args.len() < 2 {
        return;
    }
    *lfs = drop_label_filters_for_label_name(lfs, &args[1]);
    if let Some(arg) = args.get_mut(0) {
        push_down_binary_op_filters_in_place(arg, lfs);
    }
}

fn pushdown_label_filters_for_label_set(args: &mut [Expr], lfs: &mut Vec<Matcher>) {
    if args.is_empty() {
        return;
    }

    let label_names = args.iter()
        .skip(1)
        .step_by(2);

    let mut lfs = drop_label_filters_for_label_names(lfs, label_names);

    if let Some(arg) = args.get_mut(0) {
        push_down_binary_op_filters_in_place(arg, &mut lfs);
    }
}

#[inline]
fn get_label_filters_map(filters: &[Matcher]) -> FastHashSet<Signature> {
    let mut set: FastHashSet<Signature> = FastHashSet::with_capacity(filters.len());
    for label in filters.iter() {
        let sig = Signature::new(label);
        set.insert(sig);
    }
    set
}

fn intersect_label_filters(first: Vec<Matcher>, second: Vec<Matcher>) -> Vec<Matcher> {
    if first.is_empty() || second.is_empty() {
        return vec![];
    }
    let set = get_label_filters_map(&first);
    let mut result = Vec::with_capacity(first.len());
    for matcher in second.into_iter() {
        let sig = Signature::new(&matcher);
        if set.contains(&sig) {
            result.push(matcher);
        }
    }
    result
}

fn union_label_filters(first: Vec<Matcher>, second: Vec<Matcher>) -> Vec<Matcher> {
    if first.is_empty() {
        return second;
    }
    if second.is_empty() {
        return first;
    }
    // use SmallVec here because generally the number of filters is small, and we want to avoid allocations
    let set: SmallVec<Signature, 6> = SmallVec::from_iter(first.iter().map(Signature::new));
    // reuse first to avoid allocations
    let mut result = first;
    for matcher in second.into_iter() {
        let signature = Signature::new(&matcher);
        if !set.contains(&signature) {
            result.push(matcher);
        }
    }
    result
}

fn union_label_filters_internal(first: &mut Vec<Matcher>, second: &Vec<Matcher>) {
    // use SmallVec here because generally the number of filters is small, and we want to avoid allocations
    let set: SmallVec<Signature, 6> = SmallVec::from_iter(first.iter().map(Signature::new));
    for matcher in second.iter() {
        let sig = Signature::new(&matcher);
        if !set.contains(&sig) {
            first.push(matcher.clone());
        }
    }
}

fn keep_label_filters_for_label_names<'a>(lfs: &mut Vec<Matcher>, label_names: impl Iterator<Item=&'a Expr>) {
    let mut names_set: SmallVec<&str, 4> = SmallVec::new();
    for label_name in label_names {
        if let Expr::StringLiteral(se_label_name) = label_name {
            names_set.push(se_label_name.as_str());
        } else {
            return;
        }
    }
    lfs.retain(|x| names_set.contains(&x.label.as_str()))
}

fn drop_label_filters_for_label_names<'a>(lfs: &[Matcher], label_names: impl Iterator<Item=&'a Expr>) -> Vec<Matcher> {
    let mut names_set: SmallVec<&str, 4> = SmallVec::new();
    for label_name in label_names {
        if let Some(v) = get_expr_as_string(label_name) {
            names_set.push(v);
        } else {
            return vec![];
        }
    }
    lfs.iter().filter(|x|!names_set.contains(&x.label.as_str())).cloned().collect()
}

fn drop_label_filters_for_label_name(lfs: &[Matcher], label_name: &Expr) -> Vec<Matcher> {
    let name = if let Some(v) = get_expr_as_string(label_name) {
        v
    } else {
        return vec![];
    };
    lfs.iter().filter(|x|!x.label.eq(name)).cloned().collect()
}

fn filter_label_filters_on(lfs: &mut Vec<Matcher>, args: &[String]) {
    if !args.is_empty() {
        let m: FastHashSet<&String> = FastHashSet::from_iter(args.iter());
        lfs.retain(|x| m.contains(&x.label))
    } else {
        lfs.clear()
    }
}

fn filter_label_filters_ignoring(lfs: &mut Vec<Matcher>, args: &[String]) {
    if !args.is_empty() {
        let m: FastHashSet<&String> = FastHashSet::from_iter(args.iter());
        lfs.retain(|x| !m.contains(&x.label))
    }
}

fn get_func_arg_for_optimization(
    func: BuiltinFunction,
    args: &[Expr],
) -> ParseResult<Option<&Expr>> {
    let idx = get_func_arg_idx_for_optimization(func, args)?;
    if let Some(idx) = idx {
        if idx >= args.len() {
            return Ok(None);
        }
        return Ok(Some(&args[idx]));
    }
    Ok(None)
}

fn get_func_arg_idx_for_optimization(
    func: BuiltinFunction,
    args: &[Expr],
) -> ParseResult<Option<usize>> {
    use BuiltinFunction::*;
    match func {
        Aggregate(agg) => get_aggr_arg_idx_for_optimization(agg, args),
        Rollup(rollup) => get_rollup_arg_idx_for_optimization(rollup, args),
        Transform(transform) => get_transform_arg_idx_for_optimization(transform, args),
    }
}

fn get_aggr_arg_idx_for_optimization(
    func: AggregateFunction,
    args: &[Expr],
) -> ParseResult<Option<usize>> {
    use AggregateFunction::*;
    match func {
        CountValues => Err(ParseError::ArgumentError(
            "BUG: count_values must be already handled".to_string(),
        )),
        Bottomk | BottomkAvg | BottomkLast | BottomkMax | BottomkMedian | BottomkMin | Limitk
        | Outliersk | OutliersMAD | Quantile | Topk | TopkAvg | TopkLast | TopkMax | TopkMin
        | TopkMedian => Ok(Some(1)),
        Quantiles => Ok(Some(args.len() - 1)),
        _ => {
            if func.can_accept_multiple_args() {
                let msg = format!("BUG: {} must be already handled", func);
                return Err(ParseError::ArgumentError(msg));
            }
            Ok(Some(0))
        }
    }
}

fn get_rollup_arg_idx_for_optimization(
    func: RollupFunction,
    args: &[Expr],
) -> ParseResult<Option<usize>> {
    use RollupFunction::*;
    // This must be kept in sync with GetRollupArgIdx()
    match func {
        CountValuesOverTime => Err(ParseError::ArgumentError(
            "BUG: count_values_over_time must be already handled".to_string(),
        )),
        AbsentOverTime => Ok(None),
        QuantileOverTime | AggrOverTime | HoeffdingBoundLower | HoeffdingBoundUpper => Ok(Some(1)),
        QuantilesOverTime => Ok(Some(args.len() - 1)),
        _ => Ok(Some(0)),
    }
}

fn get_transform_arg_idx_for_optimization(
    func_name: TransformFunction,
    args: &[Expr],
) -> ParseResult<Option<usize>> {
    use TransformFunction::*;
    match func_name {
        LabelCopy | LabelDel | LabelJoin | LabelKeep | LabelLowercase | LabelMap | LabelMatch
        | LabelMismatch | LabelMove | LabelReplace | LabelSet | LabelTransform | LabelUppercase
        | LabelsEqual | RangeNormalize | Union => {
            let msg = format!("BUG: {} must be already handled", func_name);
            // todo: different error type
            Err(ParseError::ArgumentError(msg))
        }
        DropCommonLabels => Ok(None),
        Absent | Scalar => Ok(None),
        End | Now | Pi | Ru | Start | Step | Time => Ok(None),
        LimitOffset => Ok(Some(2)),
        BucketsLimit | HistogramQuantile | HistogramShare | RangeQuantile | RangeTrimOutliers
        | RangeTrimSpikes | RangeTrimZScore => Ok(Some(1)),
        HistogramQuantiles => Ok(Some(args.len() - 1)),
        _ => Ok(Some(0)),
    }
}
