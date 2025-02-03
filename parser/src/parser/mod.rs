pub use duration::{parse_duration_value, parse_positive_duration_value};
pub use function::validate_function_args;
pub use number::{get_number_suffix, parse_number};
pub use parse_error::*;
pub use timestamp::*;
pub use tokens::*;
pub(crate) use utils::{
    escape_ident, extract_string_value, quote, unescape_ident
};

use crate::ast::{check_ast, Expr};
use crate::optimizer::remove_parens;
use parser::Parser;

pub use metric_name::parse_metric_name;
pub use utils::is_valid_identifier;
use crate::label::Matchers;
use crate::parser::expr::parse_expression;

mod aggregation;
mod expr;
mod function;
mod parser;
mod rollup;
mod selector;
pub mod duration;
mod number;
mod parse_error;
mod tokens;
mod utils;
#[cfg(test)]
mod parser_example_test;
#[cfg(test)]
mod parser_test;
mod metric_name;
mod timestamp;


pub fn parse(input: &str) -> ParseResult<Expr> {
    let mut parser = Parser::new(input)?;
    let mut expr = parse_expression(&mut parser)?;
    if !parser.is_eof() {
        return Err(ParseError::UnparsedData);
    }
    expr = remove_parens(expr); // todo: remove this (needs test refactoring)
    check_ast(expr).map_err(|err| ParseError::General(err.to_string()))
}

/// Parse a string representing a metric selector expression e,g. 
pub fn parse_metric_selector(input: &str) -> ParseResult<Matchers> {
    let mut parser = Parser::new(input)?;
    match parse_expression(&mut parser) {
        Ok(expr) => {
            if !parser.is_eof() {
                return Err(ParseError::UnparsedData);
            }
            match expr {
                Expr::MetricExpression(expr) => Ok(expr.matchers),
                _ => Err(ParseError::InvalidSelector(input.to_string()))
            }
        },
        Err(err) => Err(err),
    }
}