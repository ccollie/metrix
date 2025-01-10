// integrate checks from
// https://github.com/prometheus/prometheus/blob/fa6e05903fd3ce52e374a6e1bf4eb98c9f1f45a7/promql/parser/parse.go#L436

// Original Source: https://github.com/GreptimeTeam/promql-parser/blob/main/src/parser/ast.rs
use crate::ast::{
    AggregationExpr, BExpression, BinModifier, BinaryExpr, Expr, FunctionExpr, MetricExpr,
    NumberLiteral, ParensExpr, RollupExpr, StringExpr, UnaryExpr, VectorMatchCardinality
};
use crate::common::{Value, ValueType};
use crate::functions::BuiltinFunction;
use crate::label::NAME_LABEL;

/// check_ast checks the validity of the provided AST. This includes type checking.
/// Recursively check correct typing for child nodes and raise errors in case of bad typing.
pub fn check_ast(expr: Expr) -> Result<Expr, String> {
    use Expr::*;
    match expr {
        UnaryOperator(ex) => {
            let modified = check_ast(*ex.expr)?;
            Ok(UnaryOperator(UnaryExpr {
                expr: Box::new(modified),
            }))
        }
        BinaryOperator(ex) => check_ast_for_binary_expr(ex),
        Aggregation(ex) => check_ast_for_aggregate_expr(ex),
        Function(ex) => check_ast_for_call(ex),
        MetricExpression(ex) => check_ast_for_vector_selector(ex),
        Rollup(ex) => check_ast_for_rollup(ex),
        Parens(ex) => check_ast_for_parens(ex),
        StringExpr(ex) => check_ast_for_string_expr(ex),
        StringLiteral(_) | NumberLiteral(_) | Duration(_) => Ok(expr),
    }
}

pub fn validate_func_args(func: &BuiltinFunction, args: &[Expr]) -> Result<(), String> {
    func.validate_args(args).map_err(|e| e.to_string())
}

fn check_ast_for_aggregate_expr(ex: AggregationExpr) -> Result<Expr, String> {
    let func = BuiltinFunction::Aggregate(ex.function);
    validate_func_args(&func, &ex.args)?;
    Ok(Expr::Aggregation(ex))
}

fn check_ast_for_call(expr: FunctionExpr) -> Result<Expr, String> {
    validate_func_args(&expr.function, &expr.args)?;
    Ok(Expr::Function(expr))
}

fn check_ast_for_parens(expr: ParensExpr) -> Result<Expr, String> {
    let mut expressions: Vec<Expr> = Vec::with_capacity(expr.len());
    for expr in expr.expressions.into_iter() {
        expressions.push(check_ast(expr)?);
    }
    Ok(Expr::Parens(ParensExpr::new(expressions)))
}

/// TODO
fn check_ast_for_string_expr(expr: StringExpr) -> Result<Expr, String> {
    Ok(Expr::StringExpr(expr))
}

/// the original logic is redundant in
/// prometheus, and the following coding blocks
/// have been optimized for readability, but all logic SHOULD be covered.
fn check_ast_for_binary_expr(mut ex: BinaryExpr) -> Result<Expr, String> {
    use ValueType::*;

    let operator = ex.op;
    let is_comparison = operator.is_comparison();

    if ex.returns_bool() && !is_comparison {
        return Err("bool modifier can only be used on comparison operators".into());
    }

    let left_type = ex.left.value_type();
    let right_type = ex.right.value_type();

    // we're more lenient than prometheus here
    // if is_comparison {
    //     match (&left_type, &right_type, ex.returns_bool()) {
    //         (ValueType::Scalar, ValueType::Scalar, false) => {
    //             return Err("comparisons between scalars must use BOOL modifier".into());
    //         }
    //         (ValueType::String, ValueType::String, false) => {
    //             return Err("comparisons between strings must use BOOL modifier".into());
    //         }
    //         _ => {}
    //     }
    // }

    // For `on` matching, a label can only appear in one of the lists.
    // Every time series of the result vector must be uniquely identifiable.
    if ex.is_matching_on() && ex.is_labels_joint() {
        if let Some(labels) = ex.intersect_labels() {
            if let Some(label) = labels.first() {
                return Err(format!(
                    "label '{label}' must not occur in ON and GROUP clause at once",
                ));
            }
        };
    }

    if operator.is_set_operator() {
        if left_type == String && right_type == String {
            return Err(format!(
                "operator '{operator}' not allowed in string string operations"
            ));
        }

        if left_type == String || right_type == String {
            return Err(format!(
                "set operator '{operator}' not allowed in binary {left_type}/{right_type} expression",
            ));
        }

        if left_type == InstantVector && right_type == InstantVector {
            if let Some(ref modifier) = ex.modifier {
                if matches!(modifier.card, VectorMatchCardinality::OneToMany(_))
                    || matches!(modifier.card, VectorMatchCardinality::ManyToOne(_))
                {
                    return Err(format!("no grouping allowed for '{operator}' operation"));
                }
            };
        }

        match &mut ex.modifier {
            Some(modifier) => {
                if modifier.card == VectorMatchCardinality::OneToOne {
                    modifier.card = VectorMatchCardinality::ManyToMany;
                }
            }
            None => {
                ex.modifier =
                    Some(BinModifier::default().with_card(VectorMatchCardinality::ManyToMany));
            }
        }
    }

    if left_type == String && right_type == String {
        if !operator.is_valid_string_op() {
            return Err(format!(
                "operator '{operator}' not allowed in string string operations"
            ));
        }
        return Ok(Expr::BinaryOperator(ex));
    }

    let valid_types = [Scalar, InstantVector, RangeVector];

    if !valid_types.contains(&left_type) || !valid_types.contains(&right_type) {
        return Err("mismatched operand types in binary expression".into());
    }

    if (left_type != InstantVector || right_type != InstantVector)
        && ex.is_matching_labels_not_empty()
    {
        return Err("vector matching only allowed between instant vectors".into());
    }

    Ok(Expr::BinaryOperator(ex))
}

fn check_ast_for_rollup(mut ex: RollupExpr) -> Result<Expr, String> {
    ex.expr = BExpression::from(check_ast(*ex.expr)?);
    let value_type = ex.expr.return_type();
    // we differ from Prometheus in that Scalars are treated to InstantVectors with no labels
    if value_type != ValueType::InstantVector && value_type != ValueType::Scalar {
        return Err(format!(
            "subquery is only allowed on instant vector, got {value_type} instead"
        ));
    }
    if let Some(at) = &ex.at {
        let at_type = at.return_type();
        if at_type != ValueType::Scalar && value_type != ValueType::InstantVector {
            return Err(format!(
                "subquery @ modifier must be a scalar or expression, got {at_type} instead",
            ));
        }
        if let Expr::NumberLiteral(NumberLiteral { value, .. }) = at.as_ref() {
            if value.is_infinite()
                || value.is_nan()
                || value >= &(i64::MAX as f64)
                || value <= &(i64::MIN as f64)
            {
                return Err(format!("timestamp out of bounds for @ modifier: {value}"));
            }
        }
    }

    Ok(Expr::Rollup(ex))
}

fn check_ast_for_vector_selector(ex: MetricExpr) -> Result<Expr, String> {
    match ex.metric_name() {
        Some(_) => {
            // make sure metric name is not set twice (to different values)
            // with OR matching, __name__ can appear multiple times]
            // ex: {__name__="foo" OR __name__="bar"}
            // {__name__="a",bar="baz" or __name__="a"}
            let mut is_duplicate = ex.matchers.matchers.iter().filter(|m| m.name() == NAME_LABEL).count() > 0;
            if !is_duplicate {
                // try the arms of the OR matchers
                let or_matchers = &ex.matchers.or_matchers;
                for m in or_matchers.iter() {
                    if m.iter().filter(|m| m.name() == NAME_LABEL).count() > 1 {
                        is_duplicate = true;
                        break;
                    }
                }
            }
            if is_duplicate {
                return Err("metric name must not be set twice".to_string());
            }
            Ok(Expr::MetricExpression(ex))
        }
        _ => Ok(Expr::MetricExpression(ex)),
    }
}
