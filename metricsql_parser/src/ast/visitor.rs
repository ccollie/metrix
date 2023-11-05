// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::ast::{AggregationExpr, BinaryExpr, Expr, FunctionExpr, ParensExpr, RollupExpr};

/// Trait that implements the [Visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern)
/// for a depth first walk on [Expr] AST. [`pre_visit`](ExprVisitor::pre_visit) is called
/// before any children are visited, and then [`post_visit`](ExprVisitor::post_visit) is called
/// after all children have been visited. Only [`pre_visit`](ExprVisitor::pre_visit) is required.
pub trait ExprVisitor {
    type Error;

    /// Called before any children are visited. Return `Ok(false)` to cut short the recursion
    /// (skip traversing and return).
    fn pre_visit(&mut self, plan: &Expr) -> Result<bool, Self::Error>;

    /// Called after all children are visited. Return `Ok(false)` to cut short the recursion
    /// (skip traversing and return).
    fn post_visit(&mut self, _plan: &Expr) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

/// A util function that traverses an AST [Expr] in depth-first order. Returns
/// `Ok(true)` if all nodes were visited, and `Ok(false)` if any call to
/// [`pre_visit`](ExprVisitor::pre_visit) or [`post_visit`](ExprVisitor::post_visit)
/// returned `Ok(false)` and may have cut short the recursion.
pub fn walk_expr<V: ExprVisitor>(visitor: &mut V, expr: &Expr) -> Result<bool, V::Error> {
    if !visitor.pre_visit(expr)? {
        return Ok(false);
    }

    let recurse = match expr {
        Expr::BinaryOperator(BinaryExpr { left, right, .. }) => {
            walk_expr(visitor, left)? || walk_expr(visitor, right)?
        }
        Expr::Parens(ParensExpr { expressions }) => {
            for expr in expressions {
                if !walk_expr(visitor, expr)? {
                    return Ok(false);
                }
            }
            true
        }
        Expr::Rollup(RollupExpr { expr, at, .. }) => {
            if !walk_expr(visitor, expr)? {
                return Ok(false);
            }
            if let Some(at_expr) = at {
                if !walk_expr(visitor, at_expr)? {
                    return Ok(false);
                }
            }
            true
        }
        Expr::Aggregation(AggregationExpr { args, .. })
        | Expr::Function(FunctionExpr { args, .. }) => {
            for func_argument_expr in args {
                if !walk_expr(visitor, func_argument_expr)? {
                    return Ok(false);
                }
            }
            true
        }
        Expr::StringLiteral(_)
        | Expr::StringExpr(_)
        | Expr::Number(_)
        | Expr::MetricExpression(_)
        | Expr::With(_)
        | Expr::WithSelector(_)
        | Expr::Duration(_) => true,
    };

    if !recurse {
        return Ok(false);
    }

    if !visitor.post_visit(expr)? {
        return Ok(false);
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use crate::ast::MetricExpr;
    use crate::common::LabelFilterOp;
    use crate::parser;

    use super::*;

    struct NamespaceVisitor {
        namespace: String,
    }

    fn vector_selector_includes_namespace(namespace: &str, vector_selector: &MetricExpr) -> bool {
        let mut includes_namespace = false;
        for filters in &vector_selector.label_filters {
            if filters.label.eq("namespace")
                && filters.value.eq(namespace)
                && filters.op == LabelFilterOp::Equal
            {
                includes_namespace = true;
                break;
            }
        }
        includes_namespace
    }

    impl ExprVisitor for NamespaceVisitor {
        type Error = &'static str;

        fn pre_visit(&mut self, expr: &Expr) -> Result<bool, Self::Error> {
            match expr {
                Expr::MetricExpression(matrix_selector) => {
                    let included = vector_selector_includes_namespace(
                        self.namespace.as_str(),
                        &matrix_selector,
                    );
                    return Ok(included);
                }
                Expr::Number(_) | Expr::StringLiteral(_) => return Ok(false),
                _ => (),
            }
            Ok(true)
        }
    }

    #[test]
    fn test_check_for_namespace_basic_query() {
        let expr = "pg_stat_activity_count{namespace=\"sample\"}";
        let ast = parser::parse(expr).unwrap();
        let mut visitor = NamespaceVisitor {
            namespace: "sample".to_string(),
        };
        assert!(walk_expr(&mut visitor, &ast).unwrap());
    }

    #[test]
    fn test_check_for_namespace_label_present() {
        let expr = "(sum by (namespace) (max_over_time(pg_stat_activity_count{namespace=\"sample\"}[1h])))";
        let ast = parser::parse(expr).unwrap();
        let mut visitor = NamespaceVisitor {
            namespace: "sample".to_string(),
        };
        assert!(walk_expr(&mut visitor, &ast).unwrap());
    }

    #[test]
    fn test_check_for_namespace_label_wrong_namespace() {
        let expr = "(sum by (namespace) (max_over_time(pg_stat_activity_count{namespace=\"sample\"}[1h])))";
        let ast = parser::parse(expr).unwrap();
        let mut visitor = NamespaceVisitor {
            namespace: "foobar".to_string(),
        };
        assert!(!walk_expr(&mut visitor, &ast).unwrap());
    }

    #[test]
    fn test_check_for_namespace_label_missing_namespace() {
        let expr = "(sum by (namespace) (max_over_time(pg_stat_activity_count{}[1h])))";
        let ast = parser::parse(expr).unwrap();
        let mut visitor = NamespaceVisitor {
            namespace: "sample".to_string(),
        };
        assert!(!walk_expr(&mut visitor, &ast).unwrap());
    }

    #[test]
    fn test_literal_expr() {
        let mut visitor = NamespaceVisitor {
            namespace: "sample".to_string(),
        };

        let ast = parser::parse("1").unwrap();
        assert!(!walk_expr(&mut visitor, &ast).unwrap());

        let ast = parser::parse("1 + 1").unwrap();
        assert!(!walk_expr(&mut visitor, &ast).unwrap());

        let ast = parser::parse(r#""1""#).unwrap();
        assert!(!walk_expr(&mut visitor, &ast).unwrap());
    }
}
