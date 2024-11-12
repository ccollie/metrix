use serde::{Deserialize, Serialize};

use metricsql_parser::prelude::{BinModifier, Operator};

use crate::execution::binary::{eval_scalar_vector_binop, eval_string_string_binop, eval_vector_scalar_binop, exec_vector_vector_binop, scalar_binary_operation};
use crate::execution::context::Context;
use crate::execution::dag::utils::{ resolve_value};
use crate::execution::dag::ExecutableNode;
use crate::execution::{eval_number, EvalConfig};
use crate::{RuntimeError, RuntimeResult};
use crate::types::{QueryValue};

/// A node that represents a binary operation between two QueryValue nodes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BinopNode {
    pub op: Operator,
    pub modifier: Option<BinModifier>,
    pub left_idx: usize,
    pub right_idx: usize,
    #[serde(skip)]
    pub left: QueryValue,
    #[serde(skip)]
    pub right: QueryValue,
    pub(crate) reset_metric_group: bool,
}

impl BinopNode {
    pub fn new(
        left_idx: usize,
        right_idx: usize,
        op: Operator,
        modifier: Option<BinModifier>,
    ) -> Self {
        Self {
            op,
            left_idx,
            right_idx,
            modifier,
            ..Default::default()
        }
    }
}

impl ExecutableNode for BinopNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        resolve_value(self.left_idx, &mut self.left, dependencies);
        resolve_value(self.right_idx, &mut self.right, dependencies);
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        use QueryValue::*;

        let is_tracing = ctx.trace_enabled();
        let bool_modifier = if let Some(modifier) = &self.modifier {
            modifier.return_bool
        } else {
            false
        };

        match (&mut self.left, &mut self.right) {
            (Scalar(left), Scalar(right)) => {
                let value = scalar_binary_operation(self.op, *left, *right, bool_modifier)?;
                Ok(Scalar(value))
            }
            (InstantVector(ref mut left_vec), InstantVector(ref mut right_vec)) => {
                let left = std::mem::take(left_vec);
                let right = std::mem::take(right_vec);
                exec_vector_vector_binop(ctx, left, right, self.op, &self.modifier)
            }
            (InstantVector(ref mut vector), Scalar(scalar)) => {
                if self.op.is_logical_op() {
                    // expand scalar to vector. Note we handle `unless` specially, since it's
                    // possible to calculate without expanding the scalar (and allocating memory).
                    if self.op != Operator::Unless {
                        let left = std::mem::take(vector);
                        let right = eval_number(ec, *scalar)?;
                        return exec_vector_vector_binop(ctx, left, right, self.op, &self.modifier);
                    }
                }
                eval_vector_scalar_binop(
                    std::mem::take(vector),
                    self.op,
                    *scalar,
                    bool_modifier,
                    self.reset_metric_group,
                    is_tracing,
                )
            }
            (Scalar(scalar), InstantVector(ref mut vector)) => {
                if self.op.is_logical_op() {
                    let left = eval_number(ec, *scalar)?;
                    let right = std::mem::take(vector);
                    return exec_vector_vector_binop(ctx, left, right, self.op, &self.modifier);
                };
                eval_scalar_vector_binop(
                    *scalar,
                    self.op,
                    std::mem::take(vector),
                    bool_modifier,
                    self.reset_metric_group,
                    is_tracing,
                )
            }
            (String(left), String(right)) => {
                eval_string_string_binop(self.op, left, right, bool_modifier)
            }
            _ => Err(RuntimeError::NotImplemented(format!(
                "invalid binary operation: {} {} {}",
                self.left.data_type_name(),
                self.op,
                self.right.data_type_name()
            ))),
        }
    }
}
