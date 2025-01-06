use smallvec::SmallVec;
use crate::ast::{Expr, InterpolatedSelector, MetricExpr};
use crate::label::{LabelFilterExpr, MatchOp, Matcher};
use crate::parser::expr::parse_string_expr;
use crate::parser::parse_error::unexpected;
use crate::parser::{ParseResult, Parser};

use super::tokens::Token;

/// parse_metric_expr parses a metric selector.
///
///    <label_set>
///
///    <metric_identifier> [<label_set>]
///
pub fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expr> {
    let can_expand = p.can_lookup();
    let mut name: Option<String> = None;

    fn create_metric_expr(
        name: Option<String>,
        mut filters: Vec<Vec<LabelFilterExpr>>,
    ) -> ParseResult<Expr> {
        let mut me = if let Some(name) = name {
            MetricExpr::new(name)
        } else {
            MetricExpr::default()
        };

        if filters.is_empty() {
            return Ok(Expr::MetricExpression(me));
        }

        // AND matchers only. No OR matchers found.
        if filters.len() == 1 {
            let first = filters.pop().expect("filters is not empty");
            if first.is_empty() {
                return Ok(Expr::MetricExpression(me));
            }
            let converted = first
                .into_iter()
                .map(|x| x.into_matcher())
                .collect::<ParseResult<Vec<_>>>()?;

            me.matchers.matchers.extend(converted);
            me.sort_filters();
            return Ok(Expr::MetricExpression(me));
        }

        let mut or_matchers = vec![];
        for filter in filters {
            let converted = filter
                .into_iter()
                .map(|x| x.into_matcher())
                .collect::<ParseResult<Vec<_>>>()?;
            or_matchers.push(converted);
        }

        me.matchers.or_matchers = or_matchers;
        me.sort_filters();
        Ok(Expr::MetricExpression(me))
    }

    if p.at(&Token::Identifier) {
        let token = p.expect_identifier()?;

        if !p.at(&Token::LeftBrace) {
            let me = MetricExpr::new(token);
            return Ok(Expr::MetricExpression(me));
        }

        name = Some(token);
    }

    let filters = parse_label_filters(p)?;
    // symbol table is empty and we're not parsing a WITH statement
    if !can_expand {
        create_metric_expr(name, filters)
    } else {
        // no identifiers in the label filters, create a MetricExpr
        if filters
            .iter()
            .all(|x| x.iter().all(|filter| filter.is_resolved()))
        {
            return create_metric_expr(name, filters);
        }
        p.needs_expansion = true;
        let mut with_me = if let Some(name) = name {
            InterpolatedSelector::new(name)
        } else {
            InterpolatedSelector::default()
        };
        with_me.matchers = filters;
        Ok(Expr::WithSelector(with_me))
    }
}

/// parse_label_filters parses a set of label matchers.
///
/// `{` [ <label_name> <match_op> <match_string>, ... [or <label_name> <match_op> <match_string>, ...] `}`
///
fn parse_label_filters(p: &mut Parser) -> ParseResult<Vec<Vec<LabelFilterExpr>>> {
    use Token::*;

    p.expect(&LeftBrace)?;

    if p.at(&RightBrace) {
        p.bump();
        return Ok(vec![]);
    }

    let mut result: Vec<Vec<LabelFilterExpr>> = Vec::with_capacity(2);
    loop {
        let filters = parse_label_filters_internal(p)?;
        if !filters.is_empty() {
            result.push(filters);
        }

        let tok = p.current_token()?;
        match tok.kind {
            RightBrace => {
                p.bump();
                break;
            }
            OpOr => p.bump(),
            _ => return Err(unexpected("label filter", tok.text, "OR or }", None)),
        }
    }

    Ok(result)
}

/// parse_label_filters parses a set of label matchers.
///
/// [ <label_name> <match_op> <match_string>, ... ]
///
fn parse_label_filters_internal(p: &mut Parser) -> ParseResult<Vec<LabelFilterExpr>> {
    use Token::*;

    let mut filters = Vec::with_capacity(4);
    loop {
        let filter = parse_label_filter(p)?;
        filters.push(filter);

        let tok = p.current_token()?;
        match tok.kind {
            Comma => {
                p.bump();
                if p.at(&RightBrace) {
                    break;
                }
                continue;
            }
            RightBrace | OpOr => {
                break;
            }
            _ => return Err(unexpected("label filter", tok.text, "OR or }", None)),
        }
    }

    if !p.can_lookup() {
        // if we're not parsing a WITH statement, we need to make sure we have no unresolved identifiers
        for filter in &filters {
            if filter.is_variable() {
                return Err(unexpected(
                    "label filter",
                    &filter.label,
                    "unresolved identifier",
                    None,
                ));
            }
        }
    }

    Ok(filters)
}

/// parse_label_filter parses a single label matcher.
///
///   <label_name> <match_op> <match_string> | identifier
///
fn parse_label_filter(p: &mut Parser) -> ParseResult<LabelFilterExpr> {
    use Token::*;

    let label = p.expect_identifier()?;
    let op: MatchOp;

    let tok = p.current_token()?;
    match tok.kind {
        Equal => op = MatchOp::Equal,
        OpNotEqual => op = MatchOp::NotEqual,
        RegexEqual => op = MatchOp::RegexEqual,
        RegexNotEqual => op = MatchOp::RegexNotEqual,
        Comma | RightBrace => return Ok(LabelFilterExpr::variable(&label)),
        _ => {
            return Err(unexpected(
                "label filter",
                tok.text,
                "=, !=, =~ or !~",
                Some(&tok.span),
            ))
        }
    };

    p.bump();

    // todo: if we're parsing a WITH, we can accept an ident. IOW, we can have metric{s=ident}
    let value = parse_string_expr(p)?;

    LabelFilterExpr::new(label, op, value)
}

fn optimize_or_matchers(filters: Vec<Vec<Matcher>>) -> Vec<Vec<Matcher>> {
    let names: SmallVec<String, 4> = filters
        .iter()
        .flat_map(|x| x.iter().map(|y| y.label.clone()))
        .collect();
    let mut optimized = vec![];
    for mut or_filters in filters {
        or_filters.sort();
        optimized.push(or_filters);
    }
    optimized
}