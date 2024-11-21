use chrono::DateTime;
pub use duration::parse_duration_value;
pub use function::validate_function_args;
pub use number::{get_number_suffix, parse_number};
pub use parse_error::*;
pub use parser::*;
pub use regexp_cache::{compile_regexp, is_empty_regex};
pub use selector::parse_metric_expr;
pub(crate) use utils::{escape_ident, extract_string_value, quote, unescape_ident};

use crate::ast::{check_ast, Expr};
use crate::optimizer::remove_parens_expr;
use crate::parser::expr::parse_expression;

mod aggregation;
mod expand;
mod expr;
mod function;
mod regexp_cache;
mod rollup;
mod selector;
mod with_expr;

pub mod duration;
pub mod number;
pub mod parse_error;
pub mod parser;
pub mod symbol_provider;
pub mod tokens;
mod utils;

// tests
#[cfg(test)]
mod expand_with_test;
#[cfg(test)]
mod parser_example_test;
#[cfg(test)]
mod parser_test;
mod metric_name;

pub use metric_name::parse_metric_name;
pub use utils::{is_valid_identifier};

pub fn parse(input: &str) -> ParseResult<Expr> {
    let mut parser = Parser::new(input)?;
    let expr = parse_expression(&mut parser)?;
    if !parser.is_eof() {
        let msg = "unparsed data".to_string();
        return Err(ParseError::General(msg));
    }
    let mut expr = parser.expand_if_needed(expr)?;
    expr = remove_parens_expr(expr);
    check_ast(expr).map_err(|err| ParseError::General(err.to_string()))
}

/// Expands WITH expressions inside q and returns the resulting
/// PromQL without WITH expressions.
pub fn expand_with_exprs(q: &str) -> Result<String, ParseError> {
    let e = parse(q)?;
    Ok(format!("{}", e))
}

/// Parses a string into a unix timestamp (milliseconds). Supports
/// - Unix timestamps in seconds with optional milliseconds after the point. For example, 1562529662.678.
/// - Unix timestamps in milliseconds. For example, 1562529662678.
/// - RFC3339. For example, 2022-03-29T01:02:03Z or 2022-03-29T01:02:03+02:30.
/// 
/// Included here only to avoid having to include chrono in the public API
pub fn parse_timestamp(s: &str) -> ParseResult<i64> {
    let value = if s.contains('.') {
        let value = s.parse::<f64>().map_err(|_| ParseError::InvalidTimestamp(s.to_string()))?;
        // split into whole seconds and milliseconds
        let secs = value.trunc() as i64;
        let ms = value.fract() as i64;
        secs * 1000 + ms
    } else if let Ok(dt) = s.parse::<i64>() {
        dt
    } else {
        DateTime::parse_from_rfc3339(s)
            .map(|value| value.timestamp_millis())
            .map_err(|_| ParseError::InvalidTimestamp(s.to_string()))?
    };
    if value < 0 {
        return Err(ParseError::InvalidTimestamp(s.to_string()));
    }
    Ok(value)
}