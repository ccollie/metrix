pub use duration::parse_duration_value;
pub use function::validate_function_args;
pub use number::{get_number_suffix, parse_number};
pub use parse_error::*;
pub use parser::*;
pub use selector::parse_metric_expr;
pub use timestamp::*;
pub(crate) use utils::{escape_ident, extract_string_value, quote, unescape_ident};

use crate::ast::{check_ast, Expr};
use crate::optimizer::remove_parens;

mod aggregation;
mod expr;
mod function;
mod rollup;
mod selector;
pub mod duration;
pub mod number;
pub mod parse_error;
pub mod parser;
pub mod tokens;
mod utils;
#[cfg(test)]
mod parser_example_test;
#[cfg(test)]
mod parser_test;
mod metric_name;
mod timestamp;

pub use metric_name::parse_metric_name;
pub use utils::is_valid_identifier;

use crate::parser::expr::parse_expression;

pub fn parse(input: &str) -> ParseResult<Expr> {
    let mut parser = Parser::new(input)?;
    let mut expr = parse_expression(&mut parser)?;
    if !parser.is_eof() {
        let msg = "unparsed data".to_string();
        return Err(ParseError::General(msg));
    }
    expr = remove_parens(expr); // todo: remove this (needs test refactoring)
    check_ast(expr).map_err(|err| ParseError::General(err.to_string()))
}