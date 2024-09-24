use crate::ast::{Expr, FunctionExpr};
use crate::common::ValueType;
use crate::functions::{BuiltinFunction, TypeSignature};
use crate::parser::tokens::Token;
use crate::parser::{ParseError, ParseResult, Parser};

pub(super) fn parse_func_expr(p: &mut Parser) -> ParseResult<Expr> {
    let name = p.expect_identifier()?;
    let args = p.parse_arg_list()?;

    // with (f(x) = sum(x * 2))  f(x{a="b"}) => sum(x{a="b"}) * 2)
    // check if we have a function with the same name in the WITH stack
    if p.can_lookup() {
        let args_clone = args.clone();
        if let Some(expr) = p.resolve_ident(&name, args_clone)? {
            return Ok(expr);
        }
    }

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
/// see https://docs.victoriametrics.com/MetricsQL.html
/// https://docs.victoriametrics.com/MetricsQL.html#implicit-query-conversions
pub fn validate_function_args(func: &BuiltinFunction, args: &[Expr]) -> ParseResult<()> {
    let expect = |actual: ValueType, expected: ValueType, index: usize| -> ParseResult<()> {
        if actual != expected {
            return Err(ParseError::ArgumentError(format!(
                "Invalid argument #{} to {func}. {expected} expected, found {actual}",
                index + 1,
            )));
        }
        Ok(())
    };

    let validate_return_type = |return_type: ValueType, expected: ValueType, index: usize|
     -> ParseResult<()> {
        match expected {
            ValueType::RangeVector => {
                return match return_type {
                    // scalar and instant vector can be converted to RangeVector
                    ValueType::Scalar | ValueType::InstantVector | ValueType::RangeVector => Ok(()),
                    _ => expect(return_type, ValueType::RangeVector, index),
                };
            }
            ValueType::InstantVector => {
                return match return_type {
                    // scalar can be converted to InstantVector
                    ValueType::Scalar | ValueType::InstantVector => Ok(()),
                    _ => expect(return_type, ValueType::InstantVector, index),
                };
            }
            ValueType::Scalar => {
                if !return_type.is_operator_valid() {
                    return Err(ParseError::ArgumentError(format!(
                        "Invalid argument #{} to {func}. Scalar or InstantVector expected, found {return_type}",
                        index + 1,
                    )));
                }
            }
            ValueType::String => {
                return expect(return_type, ValueType::String, index);
            }
        }
        Ok(())
    };

    // validate function args
    let sig = func.signature();

    sig.validate_arg_count(func.name(), args.len())?;

    // arg counts match, so if we accept any type, we're done
    match sig.type_signature {
        TypeSignature::VariadicAny(_) | TypeSignature::Any(_) => return Ok(()),
        _ => {}
    }

    let mut i = 0;
    for (expected_type, actual) in sig.types().zip(args.iter()) {
        validate_return_type(actual.return_type(), expected_type, 0)?;

        match actual {
            // technically should not occur as a function parameter
            Expr::Duration(_) | Expr::NumberLiteral(_) => {
                if expected_type.is_scalar() {
                    continue;
                }
            }
            Expr::StringLiteral(_) => {
                if expected_type == ValueType::String {
                    continue;
                }
            }
            _ => {}
        }

        validate_return_type(actual.return_type(), expected_type, i)?;
        i += 1;
    }
    Ok(())
}
