use crate::ast::{DurationExpr, Expr, RollupExpr};
use crate::common::ValueType;
use crate::parser::expr::parse_single_expr_without_rollup_suffix;
use crate::parser::tokens::Token;
use crate::parser::{syntax_error, ParseError, ParseResult, Parser};

pub(super) fn parse_rollup_expr(p: &mut Parser, e: Expr) -> ParseResult<Expr> {
    let mut re = RollupExpr::new(e);

    let mut at: Option<Expr> = None;
    if p.at(&Token::LeftBracket) {
        let (window, step, inherit_step) = parse_window_and_step(p)?;
        re.window = window;
        re.step = step;
        re.inherit_step = inherit_step;
    }

    if p.at(&Token::At) {
        at = Some(parse_at_expr(p)?);
    }

    if p.at(&Token::Offset) {
        re.offset = Some(parse_offset(p)?);
    }

    if p.at(&Token::At) {
        if at.is_some() {
            let span = p.last_token_range().unwrap_or_default();
            let msg = "duplicate '@' token".to_string();
            return Err(syntax_error(&msg, &span, "".to_string()));
        }
        at = Some(parse_at_expr(p)?);
    }

    if let Some(v) = at {
        re.at = Some(Box::new(v))
    }

    Ok(Expr::Rollup(re))
}

fn parse_at_expr(p: &mut Parser) -> ParseResult<Expr> {
    use Token::*;

    p.expect(&At)?;

    let span = p.last_token_range().unwrap_or_default();
    let expr = parse_single_expr_without_rollup_suffix(p)?;

    match expr.return_type() {
        ValueType::InstantVector | ValueType::Scalar => Ok(expr),
        _ => Err(syntax_error(
            "@ modifier Expr must return a scalar or instant vector",
            &span,
            "".to_string(),
        )),
    }
}

fn parse_window_and_step(
    p: &mut Parser,
) -> Result<(Option<DurationExpr>, Option<DurationExpr>, bool), ParseError> {
    p.expect(&Token::LeftBracket)?;

    let mut window: Option<DurationExpr> = None;

    if !p.at(&Token::Colon) {
        window = Some(p.parse_positive_duration()?);
    }

    let mut step: Option<DurationExpr> = None;
    let mut inherit_step = false;

    if p.at(&Token::Colon) {
        p.bump();
        // Parse step
        if p.at(&Token::RightBracket) {
            inherit_step = true;
        }
        if !p.at(&Token::RightBracket) {
            step = Some(p.parse_positive_duration()?);
        }
    }
    p.expect(&Token::RightBracket)?;

    Ok((window, step, inherit_step))
}

fn parse_offset(p: &mut Parser) -> ParseResult<DurationExpr> {
    p.expect(&Token::Offset)?;
    p.parse_duration()
}
