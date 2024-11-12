use serde::{Deserialize, Serialize};

use metricsql_parser::ast::{BinModifier, BinaryExpr, Operator};

use crate::execution::binary::{eval_vector_scalar_binop, exec_vector_vector_binop};
use crate::execution::{Context, EvalConfig};
use crate::prelude::binary::should_reset_metric_group;
use crate::prelude::eval_number;
use crate::RuntimeResult;
use crate::types::{InstantVector, QueryValue};

use super::utils::{resolve_vector};
use super::ExecutableNode;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorScalarBinaryNode {
    pub right: f64,
    pub op: Operator,
    pub reset_metric_group: bool,
    pub left_idx: usize,
    pub modifier: Option<BinModifier>,
    #[serde(skip)]
    pub(crate) left: InstantVector,
}

impl VectorScalarBinaryNode {
    pub fn new(be: &BinaryExpr, left_idx: usize, scalar_value: f64) -> Self {
        let reset_metric_group = should_reset_metric_group(be);
        VectorScalarBinaryNode {
            left_idx,
            left: Default::default(),
            right: scalar_value,
            op: be.op,
            modifier: be.modifier.clone(),
            reset_metric_group,
        }
    }
}

impl ExecutableNode for VectorScalarBinaryNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.left = resolve_vector(self.left_idx, dependencies)?;
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        if self.op.is_logical_op() {
            // convert scalar to vector and execute vector-vector binary op
            let left = std::mem::take(&mut self.left);
            let right = eval_number(ec, self.right)?;
            // todo: can we not just execute vector-scalar binary op (for simple non-logical ops)
            // rather than allocating ?
            return exec_vector_vector_binop(ctx, left, right, self.op, &self.modifier);
        }
        let bool_modifier = if let Some(modifier) = &self.modifier {
            modifier.return_bool
        } else {
            false
        };
        eval_vector_scalar_binop(
            std::mem::take(&mut self.left),
            self.op,
            self.right,
            bool_modifier,
            self.reset_metric_group,
            ctx.trace_enabled(),
        )
    }
}
