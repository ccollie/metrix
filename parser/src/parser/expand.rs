use std::ops::Deref;

use ahash::AHashSet;

use crate::ast::{
    AggregateModifier, AggregationExpr, BinaryExpr, Expr, FunctionExpr, MetricExpr, ParensExpr,
    RollupExpr, StringExpr, StringSegment, VectorMatchCardinality, VectorMatchModifier,
    WithArgExpr, WithExpr,
};
use crate::label::{LabelFilter, Labels};
use crate::parser::symbol_provider::SymbolProviderRef;
use crate::parser::{syntax_error, ParseError, ParseResult};
use crate::prelude::InterpolatedSelector;

pub(super) fn expand_with_expr(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    expr: Expr,
) -> ParseResult<Expr> {
    use Expr::*;

    //print!("expanding {} => ", &expr);

    let res = match expr {
        BinaryOperator(be) => expand_binary_operator(symbols, was, be),
        Function(fe) => expand_function(symbols, was, fe),
        Aggregation(ae) => expand_aggregation(symbols, was, ae),
        MetricExpression(me) => expand_metric_expression(symbols, was, me),
        Parens(pe) => expand_parens(symbols, was, pe),
        Rollup(re) => expand_rollup(symbols, was, re),
        StringExpr(se) => expand_string_expr(symbols, was, &se),
        With(we) => expand_with(symbols, was, we),
        WithSelector(ws) => expand_with_selector_expression(symbols, was, ws),
        _ => Ok(expr),
    }?;

    // println!("{}", res);

    Ok(res)
}

fn expand_with(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    we: WithExpr,
) -> ParseResult<Expr> {
    let mut was_new = Vec::with_capacity(was.len() + we.was.len());
    was_new.append(&mut was.clone());
    was_new.append(&mut we.was.clone());

    expand_with_expr(symbols, &was_new, *we.expr)
}

pub fn should_expand(expr: &Expr) -> bool {
    use Expr::*;

    match expr {
        StringLiteral(_) | NumberLiteral(_) | Duration(_) => false,
        BinaryOperator(be) => should_expand(&be.left) || should_expand(&be.right),
        MetricExpression(me) => me.is_only_metric_name(), // todo: is this correct ?????
        StringExpr(se) => !se.is_expanded(),
        Parens(pe) => !pe.expressions.is_empty() && pe.expressions.iter().any(should_expand),
        Rollup(re) => {
            if should_expand(&re.expr) {
                return true;
            }
            if let Some(at) = &re.at {
                return should_expand(at);
            }
            false
        }
        WithSelector(_) => true,
        _ => true,
    }
}

fn expand_with_args(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    args: Vec<Expr>,
) -> ParseResult<Vec<Expr>> {
    let mut dst_args: Vec<Expr> = Vec::with_capacity(args.len());
    for arg in args {
        let dst_arg = expand_with_expr(symbols, was, arg)?;
        dst_args.push(dst_arg);
    }
    Ok(dst_args)
}

fn expand_with_expr_box(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    expr: Box<Expr>,
) -> ParseResult<Box<Expr>> {
    if should_expand(&expr) {
        let dst_expr = expand_with_expr(symbols, was, *expr)?;
        Ok(Box::new(dst_expr))
    } else {
        Ok(expr)
    }
}

fn expand_binary_operator(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    be: BinaryExpr,
) -> ParseResult<Expr> {
    let mut be = be;

    be.left = expand_with_expr_box(symbols, was, be.left)?;
    be.right = expand_with_expr_box(symbols, was, be.right)?;

    if let Some(modifier) = &mut be.modifier {
        if let Some(matching) = &mut modifier.matching {
            match matching {
                VectorMatchModifier::On(labels) => {
                    let new_labels = expand_modifier_args(symbols, was, labels.as_ref())?;
                    if labels != &new_labels {
                        *labels = Labels::new_from_iter(new_labels);
                    }
                }
                VectorMatchModifier::Ignoring(labels) => {
                    let new_labels = expand_modifier_args(symbols, was, labels.as_ref())?;
                    if labels != &new_labels {
                        *labels = Labels::new_from_iter(new_labels);
                    }
                }
            }

            match &modifier.card {
                VectorMatchCardinality::ManyToOne(labels) => {
                    let new_labels = expand_modifier_args(symbols, was, labels.as_ref())?;
                    modifier.card =
                        VectorMatchCardinality::ManyToOne(Labels::new_from_iter(new_labels));
                }
                VectorMatchCardinality::OneToMany(labels) => {
                    let new_labels = expand_modifier_args(symbols, was, labels.as_ref())?;
                    modifier.card =
                        VectorMatchCardinality::ManyToOne(Labels::new_from_iter(new_labels));
                }
                _ => {}
            }
        }
    }

    Ok(Expr::BinaryOperator(be))
}

fn expand_with_selector_expression(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    me: InterpolatedSelector,
) -> ParseResult<Expr> {
    if me.is_resolved() {
        // Already expanded.
        let matchers = me.to_matchers()?;
        let res = MetricExpr {
            name: None,
            matchers,
        };
        return Ok(Expr::MetricExpression(res));
    }

    let mut new_selector: MetricExpr = MetricExpr::default();

    // Populate me.LabelFilters
    for lfes in &me.matchers {
        for lfe in lfes {
            let label = lfe.name();

            if lfe.value.is_empty() || lfe.is_variable() {
                // Expand lfe.Label into vec<LabelFilter>.
                // we have something like: foo{commonFilters} and we want to expand it into
                // foo{bar="bax", job="trace"}
                let wa = get_with_arg_expr(symbols, was, &label);
                if wa.is_none() {
                    let msg = format!("cannot find WITH template for {label} inside {me}");
                    return Err(ParseError::General(msg));
                }
                let e_new = expand_with_expr_ext(symbols, was, wa.unwrap(), vec![])?;

                let mut has_non_empty_metric_group = false;
                let wme = match e_new {
                    Expr::MetricExpression(ref me) => {
                        has_non_empty_metric_group = me.metric_name().is_some();
                        Some(me)
                    }
                    _ => None,
                };
                if wme.is_none() || has_non_empty_metric_group {
                    let msg =
                        format!("WITH template {label} inside {me} must be {{...}}; got {e_new}");
                    return Err(ParseError::General(msg));
                }

                let wme = wme.unwrap();

                if wme.is_empty() {
                    continue;
                }

                let substitute = if wme.has_or_matchers() {
                    if wme.matchers.or_matchers.len() > 1 {
                        let msg = format!(
                            "WITH template {label} at {me} must be {{...}} without 'or'; got {wme}"
                        );
                        return Err(ParseError::General(msg));
                    }
                    let first = &wme.matchers.or_matchers[0];
                    if first.is_empty() {
                        continue;
                    }
                    &first[0]
                } else {
                    if wme.matchers.is_empty() {
                        continue;
                    }
                    &wme.matchers.matchers[0]
                };

                new_selector.matchers = new_selector.matchers.append(substitute.clone());

                continue;
            }

            // convert lfe to LabelFilter.
            let se = expand_string_expr(symbols, was, &lfe.value)?;
            let value = get_expr_as_string(&se)?;
            let lf = LabelFilter::new(lfe.op, label, value)?;

            new_selector = new_selector.append(lf);
        }
    }

    new_selector.matchers.dedup();
    new_selector.sort_filters();

    let metric_name = me.metric_name();
    if metric_name.is_none() {
        return Ok(Expr::MetricExpression(new_selector));
    }

    let k = metric_name.unwrap();
    let wa = get_with_arg_expr(symbols, was, k);
    if wa.is_none() {
        return Ok(Expr::MetricExpression(new_selector));
    }

    let expanded = expand_with_expr_ext(symbols, was, wa.unwrap(), vec![])?;

    let inner = match &expanded {
        Expr::MetricExpression(me) => Some(me),
        Expr::Rollup(re) => match re.expr.deref() {
            Expr::MetricExpression(me) => Some(me),
            _ => None,
        },
        _ => None,
    };

    let is_only_metric_name = new_selector.is_only_metric_name();

    if inner.is_none() {
        if is_only_metric_name {
            return Ok(expanded);
        }
        let msg = format!("cannot expand {new_selector} to non-metric expression {expanded}");
        return Err(ParseError::General(msg));
    }

    let wme = inner.unwrap();

    let lfss_src: Vec<_> = wme.matchers.iter().collect();
    let mut or_matchers: Vec<Vec<LabelFilter>> = vec![];
    if lfss_src.len() != 1 {
        // template_name{filters} where template_name is {... or ...}
        if is_only_metric_name {
            // {filters} is empty. Return {... or ...}
            return Ok(expanded);
        }

        if new_selector.has_or_matchers() {
            let name = metric_name.unwrap_or("");
            // {filters} contain {... or ...}. It cannot be merged with {... or ...}
            let msg = format!("metric {name} must not contain 'or' filters; got {wme}");
            return Err(ParseError::General(msg));
        }

        let list_to_merge = new_selector.matchers.iter().next().unwrap();
        let non_metric_name = &list_to_merge[1..];

        // {filters} doesn't contain `or`. Merge it with {... or ...} into {...,filters or ...,filters}
        for lfs in lfss_src {
            let mut filters = lfs.clone();
            for filter in non_metric_name {
                filters.push(filter.clone());
            }
            or_matchers.push(filters);
        }
    } else {
        // template_name{... or ...} where template_name is an ordinary {filters} without 'or'.
        // Merge it into {filters,... or filters,...}
        for lfs in new_selector.matchers.iter() {
            let mut filters = lfs.clone();
            for filter in lfss_src[0] {
                filters.push(filter.clone());
            }
            for filter in lfs[1..].iter() {
                filters.push(filter.clone());
            }
            or_matchers.push(filters);
        }
    }

    let me = if or_matchers.len() == 1 {
        MetricExpr::with_filters(or_matchers.pop().unwrap())
    } else {
        MetricExpr::with_or_filters(or_matchers)
    };

    match expanded {
        Expr::Rollup(mut re) => {
            re.expr = Box::new(Expr::MetricExpression(me));
            Ok(Expr::Rollup(re))
        }
        _ => Ok(Expr::MetricExpression(me)),
    }
}

fn get_expr_as_string(expr: &Expr) -> ParseResult<String> {
    match expr {
        Expr::StringExpr(se) => {
            if !se.is_literal_only() {
                let msg = format!("BUG: string expression segments must be empty; got {}", se);
                return Err(ParseError::General(msg));
            }
            Ok(se.to_string())
        }
        Expr::StringLiteral(s) => Ok(s.to_string()),
        _ => {
            let msg = format!("must be string expression; got {}", expr);
            Err(ParseError::General(msg))
        }
    }
}

fn expand_metric_expression(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    me: MetricExpr,
) -> ParseResult<Expr> {
    if !me.is_only_metric_name() {
        // Already expanded.
        return Ok(Expr::MetricExpression(me));
    }
    if let Some(metric_name) = me.metric_name() {
        if let Some(wa) = get_with_arg_expr(symbols, was, metric_name) {
            return expand_with_expr_ext(symbols, was, wa, vec![]);
        }
    }

    Ok(Expr::MetricExpression(me))
}

fn expand_function(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    func: FunctionExpr,
) -> ParseResult<Expr> {
    let name = func.name();
    let args = expand_with_args(symbols, was, func.args)?;
    if let Some(wa) = get_with_arg_expr(symbols, was, name) {
        return expand_with_expr_ext(symbols, was, wa, args);
    }
    let res = FunctionExpr {
        args,
        keep_metric_names: func.keep_metric_names,
        function: func.function,
    };
    Ok(Expr::Function(res))
}

fn expand_aggregation(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    ae: AggregationExpr,
) -> ParseResult<Expr> {
    let mut ae = ae;
    let name = ae.name();
    let args = expand_with_args(symbols, was, ae.args)?;
    if let Some(wa) = get_with_arg_expr(symbols, was, name) {
        // TODO:: if were in this method at all, Its a confirmed aggregate, so we should ensure
        // new name is also an aggregate
        return expand_with_expr_ext(symbols, was, wa, args);
    }
    ae.args = args;

    if let Some(modifier) = &ae.modifier {
        match modifier {
            AggregateModifier::By(args) => {
                let new_args = expand_modifier_args(symbols, was, args)?;
                if args != &new_args {
                    ae.modifier = Some(AggregateModifier::By(new_args));
                }
            }
            AggregateModifier::Without(args) => {
                let new_args = expand_modifier_args(symbols, was, args)?;
                if args != &new_args {
                    ae.modifier = Some(AggregateModifier::Without(new_args));
                }
            }
        };
    }

    Ok(Expr::Aggregation(ae))
}

fn expand_parens(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    p: ParensExpr,
) -> ParseResult<Expr> {
    let mut exprs = expand_with_args(symbols, was, p.expressions)?;
    if exprs.len() == 1 {
        return Ok(exprs.remove(0));
    }
    let pe = ParensExpr::new(exprs);
    Ok(Expr::Parens(pe))
}

fn expand_rollup(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    re: RollupExpr,
) -> ParseResult<Expr> {
    let mut re = re;

    re.expr = expand_with_expr_box(symbols, was, re.expr)?;
    if let Some(at) = re.at {
        let expr = expand_with_expr(symbols, was, *at)?;
        re.at = Some(Box::new(expr));
    }
    Ok(Expr::Rollup(re))
}

fn expand_string_expr(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    se: &StringExpr,
) -> ParseResult<Expr> {
    if se.is_expanded() {
        // Already expanded.
        return match se.get_literal()? {
            Some(s) => Ok(Expr::from(s.to_string())),
            None => Ok(Expr::from("")),
        };
    }

    if se.is_empty() {
        return Ok(Expr::from(""));
    }

    // todo: calculate size
    let mut b = String::with_capacity(64);
    for token in se.iter() {
        match token {
            StringSegment::Literal(s) => {
                b.push_str(s);
                continue;
            }
            StringSegment::Ident(ident) => {
                if let Some(expr) = resolve_ident(symbols, was, ident, vec![])? {
                    let value = get_expr_as_string(&expr)?;
                    b.push_str(&value);
                } else {
                    let msg = format!("missing {ident} value inside string expression");
                    return Err(ParseError::General(msg));
                }
            }
        }
    }

    Ok(Expr::from(b))
}

pub(super) fn resolve_ident(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    ident: &str,
    args: Vec<Expr>,
) -> ParseResult<Option<Expr>> {
    if let Some(wa) = get_with_arg_expr(symbols, was, ident) {
        let expr = expand_with_expr_ext(symbols, was, wa, args)?;
        return Ok(Some(expr));
    }

    Ok(None)
}

fn expand_modifier_args(
    symbols: &SymbolProviderRef,
    was: &[WithArgExpr],
    args: &[String],
) -> ParseResult<Vec<String>> {
    if args.is_empty() {
        return Ok(vec![]);
    }

    let mut dst_args: Vec<String> = Vec::with_capacity(args.len());
    for arg in args {
        let wa = get_with_arg_expr(symbols, was, arg);
        if let Some(wa) = wa {
            if wa.args.is_empty() {
                match &wa.expr {
                    Expr::MetricExpression(_) => {
                        dst_args.push(handle_metric_expr(&wa.expr, arg, args)?);
                    }
                    Expr::Parens(pe) => {
                        for p_arg in &pe.expressions {
                            dst_args.push(handle_metric_expr(p_arg, arg, args)?);
                        }
                    }
                    _ => return Err(syntax_error(&format!("cannot use {} instead of {} in {}", wa.expr, arg, args.join(", ")), &wa.token_range, "".to_string())),
                }
            } else {
                dst_args.push(arg.clone());
            }
        } else {
            dst_args.push(arg.clone());
        }
    }

    // Remove duplicate args from dst_args
    let m: AHashSet<&String> = AHashSet::from_iter(dst_args.iter());
    Ok(m.iter().map(|x| x.to_string()).collect::<Vec<String>>())
}

fn handle_metric_expr(expr: &Expr, arg: &String, args: &[String]) -> ParseResult<String> {
    match expr {
        Expr::MetricExpression(me) => {
            if !me.is_only_metric_name() {
                return error(expr, arg, args);
            }
            Ok(String::from(me.metric_name().unwrap()))
        }
        _ => error(expr, arg, args),
    }
}

fn error(expr: &Expr, arg: &String, args: &[String]) -> ParseResult<String> {
    Err(ParseError::General(format!("cannot use {expr} instead of {arg} in {}", args.join(", "))))
}

fn get_with_arg_expr<'a>(
    symbols: &'a SymbolProviderRef,
    was: &'a [WithArgExpr],
    name: &str,
) -> Option<&'a WithArgExpr> {
    // Scan wes backwards, since certain expressions may override
    // previously defined expressions
    for wa in was.iter().rev() {
        if wa.name == name {
            return Some(wa);
        }
    }
    symbols.get_symbol(name)
}

fn expand_with_expr_ext(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    wa: &WithArgExpr,
    args: Vec<Expr>,
) -> ParseResult<Expr> {
    if wa.args.len() != args.len() {
        if args.is_empty() {
            // Just return MetricExpr with the wa.Name name.
            return Ok(Expr::MetricExpression(MetricExpr::new(&wa.name)));
        }
        let msg = format!(
            "invalid number of args for {}(); got {}; want {}",
            wa.name,
            args.len(),
            wa.args.len()
        );
        return Err(ParseError::General(msg));
    }

    let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len() + args.len());

    for wa_tmp in was {
        if wa_tmp == wa {
            break;
        }
        was_new.push(wa_tmp.clone())
    }

    let arg_names = &wa.args[..];

    for (i, arg) in args.iter().enumerate() {
        was_new.push(WithArgExpr {
            name: arg_names[i].clone(),
            args: vec![],
            expr: arg.clone(),
            token_range: Default::default(),
        });
    }

    expand_with_expr(symbols, &was_new, wa.expr.clone())
}
