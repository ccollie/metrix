// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// https://github.com/apache/arrow-datafusion/tree/main/datafusion/optimizer/src
// https://github.com/apache/arrow-datafusion/blob/e222bd627b6e7974133364fed4600d74b4da6811/datafusion/optimizer/src/utils.rs

use std::ops::Deref;

use crate::ast::utils::{expr_contains, is_null, is_one, is_op_with, is_zero};
use crate::ast::{BinaryExpr, Expr, Operator};
use crate::common::{RewriteRecursion, TreeNode, TreeNodeRewriter};
use crate::optimizer::const_evaluator::ConstEvaluator;
use crate::optimizer::push_down_filters::{can_pushdown_filters, optimize_in_place};
use crate::optimizer::remove_parens;
use crate::parser::{ParseError, ParseResult};
use crate::prelude::BuiltinFunction;

// https://prometheus.io/docs/prometheus/latest/querying/operators
// Expression simplification API

pub fn simplify_expression(expr: Expr) -> ParseResult<Expr> {
    let simplifier = ExprSimplifier::new();
    simplifier.simplify(expr)
}

pub fn optimize(expr: Expr) -> ParseResult<Expr> {
    simplify_expression(expr)
}

/// This structure handles API for expression simplification
#[derive(Default)]
pub struct ExprSimplifier {}

impl ExprSimplifier {
    /// Create a new `ExprSimplifier` with the given `info` such as an
    /// instance of [`SimplifyContext`]. See
    /// [`simplify`](Self::simplify) for an example.
    ///
    /// [`SimplifyContext`]: crate::simplify_expressions::context::SimplifyContext
    pub fn new() -> Self {
        Self {}
    }

    /// Simplifies this [`Expr`]`s as much as possible, evaluating
    /// constants and applying simplifications.
    ///
    /// The types of the expression must match what operators expect,
    /// or else an error may occur trying to evaluate.
    ///
    /// # Example:
    ///
    /// `b > 2 AND b > 2`
    ///
    /// can be written to
    ///
    /// `b > 2`
    ///
    /// ``` rust
    /// use crate::metricsql_parser::prelude::{selector, number, Expr, ExprSimplifier};
    ///
    /// // Create the simplifier
    /// let simplifier = ExprSimplifier::new();
    ///
    /// // b < 2
    /// let b_lt_2 = selector("b").lt(number(2.0));
    ///
    /// // (b < 2) OR (b < 2)
    /// let expr = b_lt_2.clone().or(b_lt_2.clone());
    ///
    /// // (b < 2) OR (b < 2) --> (b < 2)
    /// let expr = simplifier.simplify(expr).unwrap();
    /// assert_eq!(expr, b_lt_2);
    /// ```
    pub fn simplify(&self, expr: Expr) -> ParseResult<Expr> {
        let mut simplifier = Simplifier::new();
        let mut const_evaluator = ConstEvaluator::new();

        let expr = remove_parens(expr);

        // TODO iterate until no changes are made during rewrite
        // (evaluating constants can enable new simplifications and
        // simplifications can enable new constant evaluation)
        // https://github.com/apache/arrow-datafusion/issues/1160
        let mut result = expr
            .rewrite(&mut const_evaluator)?
            .rewrite(&mut simplifier)?
            // run both passes twice to try and minimize simplifications that we missed
            .rewrite(&mut const_evaluator)?
            .rewrite(&mut simplifier)?;

        // push down filters
        optimize_in_place(&mut result);
        Ok(result)
    }
}

/// Rewrites [`Expr`]s by pushing down operations that can be evaluated
/// push_down_filters optimizes e in order to improve its performance.
///
/// It performs the following optimizations:
///
/// - Adds missing filters to `foo{filters1} op bar{filters2}`
///   according to https://utcc.utoronto.ca/~cks/space/blog/sysadmin/PrometheusLabelNonOptimization
#[derive(Debug, Default)]
pub struct PushDownFilterRewriter {
    /// `can_evaluate` is used during the depth-first-provider of the
    /// `Expr` tree to track if any siblings (or their descendants) were
    /// non evaluatable (e.g. had a column reference or volatile
    /// function)
    ///
    /// Specifically, `can_evaluate[N]` represents the state of
    /// traversal when we are N levels deep in the tree, one entry for
    /// this Expr and each of its parents.
    ///
    /// After visiting all siblings if `can_evaluate.top()`` is true, that
    /// means there were no non evaluatable siblings (or their
    /// descendants) so this `Expr` can be evaluated
    can_evaluate: Vec<bool>,
}

impl PushDownFilterRewriter {
    pub fn new() -> Self {
        Self {
            can_evaluate: vec![],
        }
    }

    fn push_down_filters(expr: Expr) -> ParseResult<Expr> {
        let mut expr = expr;
        optimize_in_place(&mut expr);
        Ok(expr)
    }
}

impl TreeNodeRewriter for PushDownFilterRewriter {
    type N = Expr;

    fn pre_visit(&mut self, expr: &Expr) -> ParseResult<RewriteRecursion> {
        // Default to being able to evaluate this node
        self.can_evaluate.push(true);

        // if this expr is not ok to evaluate, mark entire parent
        // stack as not ok (as all parents have at least one child or
        // descendant that can not be evaluated

        if !can_pushdown_filters(expr) {
            // walk back up stack, marking first parent that is not mutable
            let parent_iter = self.can_evaluate.iter_mut().rev();
            for p in parent_iter {
                if !*p {
                    // optimization: if we find an element on the
                    // stack already marked, know all elements above are also marked
                    break;
                }
                *p = false;
            }
        }

        // NB: do not short circuit recursion even if we find a
        // node wee can't evaluate node (so we can fold other children, args to
        // functions, etc.)
        Ok(RewriteRecursion::Continue)
    }

    fn mutate(&mut self, expr: Expr) -> ParseResult<Self::N> {
        match self.can_evaluate.pop() {
            Some(true) => Self::push_down_filters(expr),
            Some(false) => Ok(expr),
            // todo: specific optimize error
            _ => Err(ParseError::General(
                "Failed to pop can_evaluate".to_string(),
            )),
        }
    }
}

/// Simplifies [`Expr`]s by applying algebraic transformation rules
///
/// Example transformations that are applied:
/// * `expr == bool 1` and `expr != false` to `expr` when `expr` is of boolean type
/// * `expr != true` to `!expr` when `expr` is of boolean type
/// * `1 == bool 1` to `1`
/// * `0 == bool 1` to `0`
/// * `expr == NaN` and `expr != NaN` to `NaN`
#[derive(Default)]
pub struct Simplifier {}

impl Simplifier {
    pub fn new() -> Self {
        Self {}
    }

    pub fn simplify(&self, expr: Expr) -> Expr {
        use Operator::{Add, And, Div, Mod, Mul, Or};

        match expr {
            Expr::BinaryOperator(BinaryExpr {
                                     left,
                                     right,
                                     op,
                                     modifier,
                                 }) => {
                match op {
                    //
                    // Rules for Add
                    //

                    // A + 0 --> A
                    // Valid only for NumberLiteral, since MetricExpression, Rollup, Aggregation,
                    // etc. can return vectors containing NaN
                    Add if is_zero(&right) && Expr::is_number(&left) => *left,

                    // 0 + A --> A
                    // Valid only for NumberLiteral, since MetricExpression, Rollup, Aggregation,
                    // etc. can return vectors containing NaN
                    Add if is_zero(&left) && Expr::is_number(&right) => *right,

                    // A + A --> A * 2
                    // Our use case envisions that this expression involving metric selectors
                    // will need to make network calls to evaluate. If both sides are the same
                    // we can optimize by multiplying by 2 and only making one network call.
                    Add if left == right
                        && matches!(
                            left.deref(),
                            Expr::MetricExpression(_) | Expr::Rollup(_) | Expr::Aggregation(_)
                        ) =>
                        {
                            let two = Expr::from(2.0);
                            Expr::BinaryOperator(BinaryExpr {
                                right: Box::new(two),
                                left,
                                op: Mul,
                                modifier,
                            })
                        }

                    // Rules for OR
                    //
                    // (..A..) OR A --> (..A..)
                    Or if expr_contains(&left, &right, Or) => *left,
                    // A OR (..A..) --> (..A..)
                    Or if expr_contains(&right, &left, Or) => *right,
                    // A OR (A AND B) --> A
                    Or if is_op_with(And, &right, &left) => *left,
                    // (A AND B) OR A --> A
                    Or if is_op_with(And, &left, &right) => *right,

                    //
                    // Rules for AND
                    //
                    // (..A..) AND A --> (..A..)
                    And if expr_contains(&left, &right, And) => *left,
                    // A AND (..A..) --> (..A..)
                    And if expr_contains(&right, &left, And) => *right,
                    // A AND (A OR B) --> A
                    And if is_op_with(Or, &right, &left) => *left,
                    // (A OR B) AND A --> A
                    And if is_op_with(Or, &left, &right) => *right,

                    //
                    // Rules for Mul
                    //
                    // A * 1 --> A
                    Mul if is_one(&right) && Expr::is_number(&left) => *left,
                    // 1 * A --> A
                    Mul if is_one(&left) && Expr::is_number(&right) => *right,
                    // A * NaN --> NaN
                    Mul if is_null(&right) => *right,
                    // NaN * A --> NaN
                    Mul if is_null(&left) => *left,
                    // A * 0 --> 0
                    Mul if is_zero(&right) && Expr::is_number(&left) => *right,
                    // 0 * A --> 0
                    Mul if is_zero(&left) && Expr::is_number(&right) => *left,

                    //
                    // Rules for Div
                    //
                    // A / 1 --> A
                    Div if is_one(&right) => *left,
                    // NaN / A --> NaN
                    Div if is_null(&left) => *left,
                    // A / NaN --> NaN
                    Div if is_null(&right) => *right,
                    // A / A --> NAN if A.is_nan() else 1.0. The NaN comparison can be valid for
                    // NumberLiteral, but not for MetricExpression, Rollup, Aggregation, etc.
                    Div if left == right && Expr::is_number(&left) => {
                        if is_null(&right) {
                            Expr::from(f64::NAN)
                        } else {
                            Expr::from(1.0)
                        }
                    }
                    // 0 / 0 -> NaN
                    Div if is_zero(&left) && is_zero(&right) => Expr::from(f64::NAN),
                    // A / 0 -> NaN
                    Div if is_zero(&right) => {
                        // if we have an instant vector or sample, check if we need to maintain
                        // the label set
                        let mut should_keep_metric_names =
                            matches!(&modifier, Some(modifier) if modifier.keep_metric_names);
                        if !should_keep_metric_names {
                            if let Expr::Function(fe) = &left.as_ref() {
                                if fe.keep_metric_names {
                                    should_keep_metric_names = true;
                                } else if let BuiltinFunction::Transform(tf) = fe.function {
                                    should_keep_metric_names = tf.manipulates_labels();
                                }
                            }
                        }
                        if should_keep_metric_names {
                            return Expr::BinaryOperator(BinaryExpr {
                                left,
                                right,
                                op,
                                modifier,
                            });
                        }

                        Expr::from(f64::NAN)
                    }
                    // 0 / A -> 0
                    Div if is_zero(&left) && Expr::is_number(&right) => *left,

                    //
                    // Rules for Mod
                    //
                    // A % NaN --> NaN
                    Mod if is_null(&right) => *right,
                    // NaN % A --> NaN
                    Mod if is_null(&left) => *left,
                    // A % 1 --> 0
                    Mod if is_one(&right) && Expr::is_number(&left) => Expr::from(0.0),
                    // A % 0 --> NaN
                    Mod if is_zero(&right) => Expr::from(f64::NAN),
                    // A % A --> 0
                    Mod if left == right && Expr::is_number(&left) => {
                        if is_null(&right) {
                            Expr::from(f64::NAN)
                        } else {
                            Expr::from(0.0)
                        }
                    }
                    // no additional rewrites possible
                    _ => Expr::BinaryOperator(BinaryExpr {
                        left,
                        right,
                        op,
                        modifier,
                    }),
                }
            }
            expr => expr,
        }
    }
}

// see https://prometheus.io/docs/prometheus/latest/querying/operators/

impl TreeNodeRewriter for Simplifier {
    type N = Expr;

    /// rewrite the expression simplifying any constant expressions
    fn mutate(&mut self, expr: Expr) -> ParseResult<Expr> {
        Ok(self.simplify(expr))
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::binary_expr;
    use crate::ast::utils::{number, selector};
    use crate::parser::parse;

    use super::*;

    fn assert_expr_eq(expected: &Expr, actual: &Expr) {
        assert_eq!(
            expected, actual,
            "expected: \n{}\n but got: \n{}",
            expected, actual
        );
    }

    // ------------------------------
    // --- ExprSimplifier tests -----
    // ------------------------------
    #[test]
    fn api_basic() {
        let expr = number(1.0) + number(2.0);
        let expected = number(3.0);
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn simplify_and_constant_prop() {
        // should be able to simplify to false
        // (6 * (1 - 2)) > 0
        let expr = (number(6.0) * (number(1.0) - number(2.0))).gt(number(0.0));
        let expected = number(0.0);
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    // ------------------------------
    // --- Simplifier tests -----
    // ------------------------------

    #[test]
    fn test_simplify_or_same() {
        let expr = selector("c2").or(selector("c2"));
        let expected = selector("c2");
        let actual = simplify(expr);

        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_and_same() {
        let expr = selector("c2").and(selector("c2"));
        let expected = selector("c2");

        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_selector_plus_selector_same() {
        let expr = selector("c2") + selector("c2");
        let expected = selector("c2") * number(2.0);

        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_mul_by_one() {
        let expr_a = selector("c2") * number(1.0);
        let expected_a = expr_a.clone();

        let expr_b = number(1.0) * selector("c2");
        let expected_b = expr_b.clone();

        let a = simplify(expr_a);
        let b = simplify(expr_b);
        assert_expr_eq(&expected_a, &a);
        assert_expr_eq(&expected_b, &b);

        let expr = number(45.0) * number(1.0);
        let actual = simplify(expr);
        let expected = number(45.0);
        assert_expr_eq(&expected, &actual);

        let expr = number(1.0) * number(89.0);
        let expected = number(89.0);
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_mul_by_nan() {
        let null = Expr::from(f64::NAN);
        // A * NAN --> NAN
        {
            let expr = selector("c2") * null.clone();
            let actual = simplify(expr);
            assert_expr_eq(&null, &actual);
        }
        // NAN * A --> NAN
        {
            let expr = null.clone() * selector("c2");
            let actual = simplify(expr);
            assert_expr_eq(&null, &actual);
        }
    }

    #[test]
    fn test_simplify_add_zero() {
        let zero = number(0.0);
        // 0 + A --> A, where A is numeric
        {
            let expr = number(0.0) + number(5.0);
            let expected = number(5.0);
            let actual = simplify(expr);
            assert_expr_eq(&expected, &actual);
        }

        // 0 + A --> A
        // Only simplify when A is numeric
        {
            let expr = zero.clone() + selector("c2");
            let expected = expr.clone();
            let actual = simplify(expr);
            assert_expr_eq(&expected, &actual);
        }
        // A + 0 --> A if A
        // Only simplify when A is numeric
        {
            let expr = selector("foo") + number(0.0);
            let expected = expr.clone();
            let actual = simplify(expr);
            assert_expr_eq(&expected, &actual);
        }
    }

    #[test]
    fn test_simplify_mul_by_zero() {
        // 0 * A --> 0
        {
            // should remain unchanged if A is not numeric
            let mut expr = number(0.0) * selector("c2");
            let mut expected = expr.clone();
            let actual = simplify(expr);
            assert_expr_eq(&expected, &actual);

            // should return 0.0 if it is numeric
            expr = number(0.0) * number(12.5);
            expected = number(0.0);

            let actual = simplify(expr);
            assert_expr_eq(&expected, &actual);
        }

        // A * 0 --> 0
        {
            // should remain unchanged for non-numeric A
            let expr = selector("foo") * number(0.0);
            let expected = expr.clone();
            let actual = simplify(expr);
            assert_expr_eq(&expected, &actual);

            let expr = number(0.0) * number(65.4);
            let expected = number(0.0);

            let actual = simplify(expr);
            assert_expr_eq(&expected, &actual);
        }
    }

    #[test]
    fn test_simplify_div_by_one() {
        // A / 1 = A
        // should remain unchanged for non-numeric A
        let expr = selector("c2") / number(1.0);
        let expected = selector("c2");
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);

        // return A for numeric A
        let expr = number(42.0) / number(1.0);
        let expected = number(42.0);
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_div_nan() {
        // A / NAN --> NAN
        let null = Expr::from(f64::NAN);
        {
            let expr = selector("c1") * null.clone();
            let actual = simplify(expr);
            assert_expr_eq(&null, &actual);
        }
        // NAN / A --> NAN
        {
            let expr = null.clone() * selector("c2");
            let actual = simplify(expr);
            assert_expr_eq(&null, &actual);
        }

        // NAN / NAN --> NAN
        {
            let expr = null.clone() / null.clone();
            let actual = simplify(expr);
            assert_expr_eq(&null, &actual);
        }
    }

    #[test]
    fn test_simplify_div_zero_by_zero() {
        // 0 / 0 -> NAN
        let expr = number(0.0) / number(0.0);
        let expected = number(f64::NAN);
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_div_by_zero() {
        // A / 0 -> NaN

        let expected = Expr::from(f64::NAN);
        let expr = selector("c2") / number(0.0);
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_mod_by_nan() {
        let null = Expr::from(f64::NAN);
        // A % NaN --> NaN
        {
            let expr = selector("c2") % null.clone();
            let actual = simplify(expr);
            assert_expr_eq(&null, &actual);
        }
        // NaN % A --> NaN
        {
            let expr = null.clone() % selector("c2");
            let actual = simplify(expr);
            assert_expr_eq(&null, &actual);
        }
    }

    #[test]
    fn test_simplify_mod_by_one() {
        let expr = selector("c2") % number(1.0);
        let expected = expr.clone();
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);

        // test with number
        let expr = number(789.0) % number(1.0);
        let expected = number(0.0);

        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_mod_by_zero_non_nan() {
        let expr = selector("foo") % number(0.0);
        let expected = number(f64::NAN);
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_simple_and() {
        // (c > 5) AND (c > 5)
        let expr = selector("c2")
            .gt(number(5.0))
            .and(selector("c2").gt(number(5.0)));
        let expected = selector("c2").gt(number(5.0));
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_composed_and() {
        // ((c > 5) AND (c1 < 6)) AND (c > 5)
        let expr = binary_expr(
            binary_expr(
                selector("c2").gt(number(5.0)),
                Operator::And,
                selector("c1").lt(number(6.0)),
            ),
            Operator::And,
            selector("c2").gt(number(5.0)),
        );
        let expected = selector("c2").gt(number(5.0)) & selector("c1").lt(number(6.0));

        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_or_and() {
        let l = selector("c2").gt(number(5.0));
        let r = binary_expr(
            selector("c1").lt(number(6.0)),
            Operator::And,
            selector("c2").gt(number(5.0)),
        );

        // (c2 > 5) OR ((c1 < 6) AND (c2 > 5)) --> c2 > 5
        let expr = l.clone() | r.clone();

        // This is only true if `c1 < 6` is not nullable / can not be null.
        let expected = selector("c2").gt(number(5.0));

        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);

        // ((c1 < 6) AND (c2 > 5)) OR (c2 > 5) --> c2 > 5
        let expr = l | r;

        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_and_or() {
        let l = selector("c2").gt(number(5.0));
        let r = binary_expr(
            selector("c1").lt(number(6.0)),
            Operator::Or,
            selector("c2").gt(number(5.0)),
        );

        // (c2 > 5) AND ((c1 < 6) OR (c2 > 5)) --> c2 > 5
        let expr = l.clone() & r.clone();

        // This is only true if `c1 < 6` is not nullable / can not be null.
        let expected = selector("c2").gt(number(5.0));

        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);

        // ((c1 < 6) OR (c2 > 5)) AND (c2 > 5) --> c2 > 5
        let expr = l & r;

        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    fn lit_bool_null() -> Expr {
        number(f64::NAN)
    }

    #[test]
    fn test_simplify_nan_and_false() {
        let expr = lit_bool_null() & number(0.0);
        let expected = number(f64::NAN);
        let actual = simplify(expr.clone());

        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_div_nan_by_nan() {
        let null = number(f64::NAN);
        let expr_plus = null.clone() * null.clone();
        let expr_eq = null;
        let actual = simplify(expr_plus);

        assert_expr_eq(&expr_eq, &actual);
    }

    #[test]
    fn test_simplify_simplify_arithmetic_expr() {
        let expr_plus = number(1.0) + number(1.0);
        let expr_eq = number(1.0).eq(number(1.0));

        let actual_plus = simplify(expr_plus);
        let actual_eq = simplify(expr_eq);
        assert_expr_eq(&number(2.0), &actual_plus);
        assert_expr_eq(&number(1.0), &actual_eq);
    }

    // ------------------------------
    // ----- Simplifier tests -------
    // ------------------------------

    fn try_simplify(expr: Expr) -> ParseResult<Expr> {
        let simplifier = ExprSimplifier::new();
        simplifier.simplify(expr)
    }

    fn simplify(expr: Expr) -> Expr {
        try_simplify(expr).unwrap()
    }

    #[test]
    fn simplify_expr_nan_comparison() {
        let nan = number(f64::NAN);
        let zero = number(0.0);
        let one = number(1.0);

        // scalar == bool NAN is always false
        let actual = simplify(number(1.0).eq(number(f64::NAN)));
        assert_expr_eq(&zero, &actual);

        let expr = parse("NaN == NaN").unwrap();
        let actual = simplify(expr);
        assert_expr_eq(&nan, &actual);

        let expr = parse("NaN == bool NaN").unwrap();
        let actual = simplify(expr);
        assert_expr_eq(&one, &actual);

        // NAN != NAN is always 0
        let actual = simplify(number(f64::NAN).not_eq(number(f64::NAN)));
        assert_expr_eq(&zero, &actual);

        // scalar != NAN is always 1
        let actual = simplify(number(10.0).not_eq(number(f64::NAN)));
        assert_expr_eq(&one, &actual);
    }

    #[test]
    fn simplify_expr_eq() {
        let one = number(1.0);
        let zero = number(0.0);

        // true == true -> true
        let actual = simplify(one.clone().eq(one.clone()));
        assert_expr_eq(&one, &actual);

        // true == false -> false
        let actual = simplify(one.clone().eq(zero.clone()));
        assert_expr_eq(&zero, &actual);
    }

    #[test]
    fn simplify_expr_eq_skip_non_boolean_type() {
        // don't fold c1 = foo
        let actual = simplify(selector("c1").eq(selector("foo")));
        let expected = selector("c1").eq(selector("foo"));
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn simplify_expr_not_eq() {
        let zero = number(0.0);
        let one = number(1.0);
        // test constant
        let actual = simplify(number(1.0).not_eq(number(1.0)));
        assert_expr_eq(&zero, &actual);

        let actual = simplify(number(1.0).not_eq(number(0.0)));
        assert_expr_eq(&one, &actual);
    }

    #[test]
    fn simplify_expr_not_eq_skip_non_boolean_type() {
        let actual = simplify(selector("c1").not_eq(selector("foo")));
        let expected = selector("c1").not_eq(selector("foo"));
        assert_expr_eq(&expected, &actual);
    }

    #[test]
    fn test_simplify_parens() {
        let expr = parse("((foo))").unwrap();
        let expected = parse("foo").unwrap();
        let actual = simplify(expr);
        assert_expr_eq(&expected, &actual);
    }

    // TODO: BinaryExpr
}
