use crate::ast::{Expr, FunctionExpr};
use crate::common::ValueType;
use crate::functions::{BuiltinFunction, TypeSignature};
use crate::parser::tokens::Token;
use crate::parser::{ParseError, ParseResult, Parser};

pub(super) fn parse_func_expr(p: &mut Parser) -> ParseResult<Expr> {
    let name = p.expect_identifier()?;
    let args = p.parse_arg_list()?;

    let mut fe = FunctionExpr::new(&name, args)?;
    fe.keep_metric_names = if p.at(&Token::KeepMetricNames) {
        p.bump();
        true
    } else {
        false
    };

    // TODO: !!!! fix validate args
    // validate_args(&fe.function, &fe.args)?;

    Ok(Expr::Function(fe))
}

/// Note: MetricSQL is much looser than PromQL in terms of function argument types. In particular,
/// 1. MetricSQL allows scalar arguments to be passed to functions that expect vector arguments.
/// 2. For rollup function arguments without a lookbehind window, an implicit [1i] is added, which
///    essentially converts vectors into ranges
/// 3. non-rollup series selectors are wrapped in a default_rollup()
///    see https://docs.victoriametrics.com/MetricsQL.html
///
///    https://docs.victoriametrics.com/MetricsQL.html#implicit-query-conversions
pub fn validate_function_args(func: &BuiltinFunction, args: &[Expr]) -> ParseResult<()> {
    let sig = func.signature();
    sig.validate_arg_count(func.name(), args.len())?;

    match sig.type_signature {
        TypeSignature::VariadicAny(_) | TypeSignature::Any(_) => return Ok(()),
        _ => {}
    }

    for (i, (expected_type, actual)) in sig.types().zip(args.iter()).enumerate() {
        validate_return_type(actual.return_type(), expected_type, i)?;
        validate_expr_type(actual, expected_type)?;
    }
    Ok(())
}

fn validate_return_type(return_type: ValueType, expected: ValueType, index: usize) -> ParseResult<()> {
    match expected {
        ValueType::RangeVector => match return_type {
            ValueType::Scalar | ValueType::InstantVector | ValueType::RangeVector => Ok(()),
            _ => expect_type(return_type, ValueType::RangeVector, index),
        },
        ValueType::InstantVector => match return_type {
            ValueType::Scalar | ValueType::InstantVector => Ok(()),
            _ => expect_type(return_type, ValueType::InstantVector, index),
        },
        ValueType::Scalar => {
            if !return_type.is_operator_valid() {
                return Err(ParseError::ArgumentError(format!(
                    "Invalid argument #{} to function. Scalar or InstantVector expected, found {}",
                    index + 1,
                    return_type
                )));
            }
            Ok(())
        }
        ValueType::String => expect_type(return_type, ValueType::String, index),
    }
}

fn validate_expr_type(expr: &Expr, expected_type: ValueType) -> ParseResult<()> {
    match expr {
        Expr::Duration(_) | Expr::NumberLiteral(_) if expected_type.is_scalar() => Ok(()),
        Expr::StringLiteral(_) if expected_type == ValueType::String => Ok(()),
        _ => validate_return_type(expr.return_type(), expected_type, 0),
    }
}

fn expect_type(actual: ValueType, expected: ValueType, index: usize) -> ParseResult<()> {
    if actual != expected {
        return Err(ParseError::ArgumentError(format!(
            "Invalid argument #{} to function. {} expected, found {}",
            index + 1,
            expected,
            actual
        )));
    }
    Ok(())
}