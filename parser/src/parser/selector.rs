use super::tokens::Token;
use crate::ast::{Expr, MetricExpr};
use crate::label::{MatchOp, Matcher, Matchers};
use crate::parser::expr::parse_string_expr;
use crate::parser::parse_error::unexpected;
use crate::parser::{unescape_ident, ParseResult, Parser};
use crate::prelude::ParseError;
use smallvec::SmallVec;

/// parse_metric_expr parses a metric selector.
///
///    <label_set>
///
///    <metric_identifier> [<label_set>]
///
pub fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expr> {
    let mut name: Option<String> = None;

    let tok = p.current_token()?;
    if tok.kind.is_ident_like() {
        name = match tok.kind {
            Token::Identifier => Some(unescape_ident(tok.text)?.to_string()),
            _ => Some(tok.text.to_string())
        };
        p.bump();
    }

    if !p.at(&Token::LeftBrace) {
        if name.is_none() {
            return Err(ParseError::InvalidSelector("missing metric name".to_string()));
        }
        return Ok(Expr::MetricExpression(MetricExpr {
            matchers: Matchers {
                name,
                ..Default::default()
            },
        }));
    }

    let mut matchers = parse_label_filters(p)?;

    // now normalize the matchers
    let mut need_normalization = true;
    if !matchers.or_matchers.is_empty() {
        if let Some(metric_name) = normalize_matcher_list(&mut matchers.or_matchers) {
            if name.is_none() {
                name = Some(metric_name.to_string());
            }
            if matchers.or_matchers.len() == 1 {
                let mut matcher = matchers.or_matchers.pop().expect("or_matchers is not empty");
                std::mem::swap(&mut matchers.matchers, &mut matcher);
            }
            need_normalization = false;
        }
    }

    if need_normalization && !matchers.matchers.is_empty() {
        for (i, matcher) in matchers.matchers.iter_mut().enumerate() {
            if matcher.is_metric_name_filter() {
                if name.is_none() {
                    let value = std::mem::take(&mut matcher.value);
                    name = Some(value);
                }
                matchers.matchers.remove(i);
                break;
            }
        }
    }

    matchers.name = name;
    let mut me = MetricExpr { matchers };

    me.sort_filters();
    Ok(Expr::MetricExpression(me))
}

/// parse_label_filters parses a set of label matchers.
///
/// `{` [ <label_name> <match_op> <match_string>, ... [or <label_name> <match_op> <match_string>, ...] `}`
///
fn parse_label_filters(p: &mut Parser) -> ParseResult<Matchers> {
    use Token::*;

    p.expect(&LeftBrace)?;

    if p.at(&RightBrace) {
        p.bump();
        return Ok(Matchers::default());
    }

    let mut or_matchers: Vec<Vec<Matcher>> = Vec::new();
    let mut matchers: Vec<Matcher> = Vec::new();
    let mut has_or_matchers = false;

    loop {
        if has_or_matchers && !matchers.is_empty() {
            let last_matchers = std::mem::take(&mut matchers);
            or_matchers.push(last_matchers);
        }
        matchers = parse_label_filters_internal(p)?;

        let tok = p.current_token()?;
        match tok.kind {
            RightBrace => {
                p.bump();
                break;
            }
            OpOr => {
                has_or_matchers = true;
                p.bump()
            },
            _ => return Err(unexpected("label filter", tok.text, "OR or }", None)),
        }
    }

    if has_or_matchers {
        if !matchers.is_empty() {
            or_matchers.push(matchers);
        }
        // todo: validate name
        return Ok(Matchers::with_or_matchers(None, or_matchers));
    }

    Ok(Matchers::new(matchers))
}

/// parse_label_filters parses a set of label matchers.
///
/// [ <label_name> <match_op> <match_string>, ... ]
///
fn parse_label_filters_internal(p: &mut Parser) -> ParseResult<Vec<Matcher>> {
    use Token::*;

    let mut matchers: Vec<Matcher> = vec![];

    loop {
        let matcher = parse_label_filter(p)?;
        if matcher.is_metric_name_filter() {
            if matchers.is_empty() {
                matchers.push(matcher);
            } else {
                matchers.insert(0, matcher)
            }
        } else {
            matchers.push(matcher);
        }

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

    Ok(matchers)
}

fn normalize_matcher_list(matchers: &mut Vec<Vec<Matcher>>) -> Option<String> {
    // if we have a __name__ filter, we need to ensure that all matchers have the same name
    // if so, we pull out the name and return it while removing the __name__ filter from all matchers

    // track name filters. Use Smallvec instead of HashSet to avoid allocations
    let mut to_remove: SmallVec<(usize, usize, bool), 4> = SmallVec::new();

    let name = {
        let mut metric_name: &str = "";

        let first = matchers.first()?;
        for (i, m) in first.iter().enumerate() {
            if m.is_metric_name_filter() {
                metric_name = m.value.as_str();
                to_remove.push((0, i, first.len() == 1));
                break;
            }
        }

        if metric_name.is_empty() {
            return None;
        }

        let mut i: usize = 1;

        for match_list in matchers.iter().skip(1) {
            let mut found = false;
            for (j, m) in match_list.iter().enumerate() {
                if m.is_metric_name_filter() {
                    if m.value.as_str() != metric_name {
                        return None;
                    }
                    found = true;
                    to_remove.push((i, j, match_list.len() == 1));
                    break;
                }
            }
            if !found {
                return None;
            }
            i += 1;
        }

        metric_name.to_string()
    };

    // remove the __name__ filter from all matchers
    for (i, j, remove) in to_remove.iter().rev() {
        if *remove {
            matchers.remove(*i);
        } else {
            matchers[*i].remove(*j);
        }
    }

    Some(name)
}

/// parse_label_filter parses a single label matcher.
///
///   <label_name> <match_op> <match_string> | identifier
///
fn parse_label_filter(p: &mut Parser) -> ParseResult<Matcher> {
    use Token::*;

    let label = p.expect_identifier_ex()?;

    let op: MatchOp;

    let tok = p.current_token()?;
    match tok.kind {
        Equal => op = MatchOp::Equal,
        OpNotEqual => op = MatchOp::NotEqual,
        RegexEqual => op = MatchOp::RegexEqual,
        RegexNotEqual => op = MatchOp::RegexNotEqual,
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

    let value = parse_string_expr(p)?.into_literal()?;

    Matcher::new(op, label, value)
}
