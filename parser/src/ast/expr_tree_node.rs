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

//! Tree node implementation for Ast expr

use crate::ast::{
    AggregationExpr, BExpression, BinaryExpr, Expr, FunctionExpr, ParensExpr, RollupExpr,
    UnaryExpr
};
use crate::common::{TreeNode, VisitRecursion};
use crate::parser::ParseResult;

pub type Result<T> = ParseResult<T>;

impl TreeNode for Expr {
    fn apply_children<F>(&self, op: &mut F) -> Result<VisitRecursion>
    where
        F: FnMut(&Self) -> Result<VisitRecursion>,
    {
        let children = match self {
            Expr::StringLiteral(_)
            | Expr::StringExpr(_)
            | Expr::NumberLiteral(_)
            | Expr::MetricExpression(_)
            | Expr::Duration(_) => vec![],
            Expr::UnaryOperator(u) => vec![u.expr.as_ref().clone()],
            Expr::BinaryOperator(BinaryExpr { left, right, .. }) => {
                vec![left.as_ref().clone(), right.as_ref().clone()]
            }
            Expr::Aggregation(AggregationExpr { args, .. })
            | Expr::Function(FunctionExpr { args, .. }) => args.clone(),
            Expr::Parens(p) => p.expressions.clone(),
            Expr::Rollup(RollupExpr { expr, at, .. }) => {
                let mut expr_vec = Vec::with_capacity(2);
                expr_vec.push(expr.as_ref().clone());
                if let Some(at_expr) = at {
                    expr_vec.push(at_expr.as_ref().clone());
                }
                // todo: window, step, offset
                expr_vec
            }
        };

        for child in children.iter() {
            match op(child)? {
                VisitRecursion::Continue => {}
                VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }

        Ok(VisitRecursion::Continue)
    }

    fn map_children<F>(self, transform: F) -> Result<Self>
    where
        F: FnMut(Self) -> Result<Self>,
    {
        let mut transform = transform;

        // recurse into all sub expressions(and cover all expression types)
        let expr = match self {
            Expr::Aggregation(AggregationExpr {
                function,
                args,
                modifier,
                limit,
                keep_metric_names,
            }) => Expr::Aggregation(AggregationExpr {
                function,
                args: transform_vec(args, &mut transform)?,
                modifier,
                limit,
                keep_metric_names,
            }),
            Expr::UnaryOperator(u) => Expr::UnaryOperator(UnaryExpr {
                expr: transform_boxed(u.expr, &mut transform)?,
            }),
            Expr::BinaryOperator(BinaryExpr {
                left,
                op,
                right,
                modifier,
            }) => Expr::BinaryOperator(BinaryExpr {
                left: transform_boxed(left, &mut transform)?,
                op,
                right: transform_boxed(right, &mut transform)?,
                modifier,
            }),
            Expr::Duration(_) => self.clone(),
            Expr::Function(FunctionExpr {
                args,
                keep_metric_names,
                function,
            }) => Expr::Function(FunctionExpr {
                args: transform_vec(args, &mut transform)?,
                keep_metric_names,
                function,
            }),
            Expr::NumberLiteral(_) => self.clone(),
            Expr::Rollup(RollupExpr {
                expr,
                window,
                step,
                offset,
                inherit_step,
                at,
            }) => Expr::Rollup(RollupExpr {
                expr: transform_boxed(expr, &mut transform)?,
                window,
                step,
                offset,
                inherit_step,
                at: transform_option_box(at, &mut transform)?,
            }),
            Expr::MetricExpression(_) => self.clone(),
            Expr::Parens(ParensExpr { expressions }) => Expr::Parens(ParensExpr {
                expressions: transform_vec(expressions, &mut transform)?,
            }),
            Expr::StringLiteral(_) | Expr::StringExpr(_) => self.clone(),
        };

        Ok(expr)
    }
}

#[allow(clippy::boxed_local)]
fn transform_boxed<F>(boxed_expr: Box<Expr>, transform: &mut F) -> Result<BExpression>
where
    F: FnMut(Expr) -> Result<Expr>,
{
    // TODO:
    // It might be possible to avoid an allocation (the Box::new) below by reusing the box.
    let expr: Expr = *boxed_expr;
    let rewritten_expr = transform(expr)?;
    Ok(Box::new(rewritten_expr))
}

fn transform_option_box<F>(
    option_box: Option<BExpression>,
    transform: &mut F,
) -> Result<Option<BExpression>>
where
    F: FnMut(Expr) -> Result<Expr>,
{
    option_box
        .map(|expr| transform_boxed(expr, transform))
        .transpose()
}

/// &mut transform a `Vec` of `Expr`s
fn transform_vec<F>(v: Vec<Expr>, transform: &mut F) -> Result<Vec<Expr>>
where
    F: FnMut(Expr) -> Result<Expr>,
{
    v.into_iter().map(transform).collect()
}
