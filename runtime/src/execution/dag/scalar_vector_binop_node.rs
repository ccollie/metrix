use serde::{Deserialize, Serialize};

use metricsql_parser::ast::{BinModifier, BinaryExpr, Operator};

use crate::execution::binary::{eval_scalar_vector_binop, exec_vector_vector_binop, should_reset_metric_group};
use crate::execution::{eval_number, Context, EvalConfig};
use crate::RuntimeResult;
use crate::types::{InstantVector, QueryValue};

use super::utils::{resolve_vector};
use super::ExecutableNode;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScalarVectorBinaryNode {
    pub left: f64,
    pub op: Operator,
    pub reset_metric_group: bool,
    pub right_idx: usize,
    pub modifier: Option<BinModifier>,
    #[serde(skip)]
    pub(crate) right: InstantVector,
}

impl ScalarVectorBinaryNode {
    pub fn new(be: &BinaryExpr, right_idx: usize, scalar_value: f64) -> Self {
        let reset_metric_group = should_reset_metric_group(be);
        ScalarVectorBinaryNode {
            right_idx,
            left: scalar_value,
            right: Default::default(),
            op: be.op,
            modifier: be.modifier.clone(),
            reset_metric_group,
        }
    }
}

impl ExecutableNode for ScalarVectorBinaryNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.right = resolve_vector(self.right_idx, dependencies)?;
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        if self.op.is_logical_op() {
            // convert scalar to vector and execute vector-vector binary op
            let right = std::mem::take(&mut self.right);
            let left = eval_number(ec, self.left)?;
            return exec_vector_vector_binop(ctx, left, right, self.op, &self.modifier);
        }
        let bool_modifier = if let Some(modifier) = &self.modifier {
            modifier.return_bool
        } else {
            false
        };
        eval_scalar_vector_binop(
            self.left,
            self.op,
            std::mem::take(&mut self.right),
            bool_modifier,
            self.reset_metric_group,
            ctx.trace_enabled(),
        )
    }
}
