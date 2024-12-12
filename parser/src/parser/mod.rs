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

/// Parses a string into a unix timestamp (milliseconds). Accepts a positive integer or an RFC3339 timestamp.
/// Included here only to avoid having to include chrono in the public API
pub fn parse_timestamp(s: &str) -> ParseResult<i64> {
    let value = if let Ok(dt) = parse_numeric_timestamp(s) {
        dt
    } else {
        let value = DateTime::parse_from_rfc3339(s)
            .map_err(|_| ParseError::InvalidTimestamp(s.to_string()))?;
        value.timestamp_millis()
    };
    if value < 0 {
        return Err(ParseError::InvalidTimestamp(s.to_string()));
    }
    Ok(value)
}


/// parse_numeric_timestamp parses timestamp at s in seconds, milliseconds, microseconds or nanoseconds.
///
/// It returns milliseconds for the parsed timestamp.
pub fn parse_numeric_timestamp(s: &str) -> Result<i64, Box<dyn std::error::Error>> {
    const CHARS_TO_CHECK: &[char] = &['.', 'e', 'E'];
    if s.contains(CHARS_TO_CHECK) {
        // The timestamp is a floating-point number
        let ts: f64 = s.parse()?;
        if ts >= (1 << 32) as f64 {
            // The timestamp is in milliseconds
            return Ok(ts as i64);
        }
        return Ok(ts.round() as i64);
    }
    // The timestamp is an integer number
    let ts: i64 = s.parse()?;
    match ts {
        ts if ts >= (1 << 32) * 1_000_000 => {
            // The timestamp is in nanoseconds
            Ok(ts / 1_000_000)
        }
        ts if ts >= (1 << 32) * 1_000 => {
            // The timestamp is in microseconds
            Ok(ts / 1_000)
        }
        ts if ts >= (1 << 32) => {
            // The timestamp is in milliseconds
            Ok(ts)
        }
        _ => Ok(ts * 1_000),
    }
}



#[cfg(test)]
mod tests {

}