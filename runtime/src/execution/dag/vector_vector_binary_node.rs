use serde::{Deserialize, Serialize};

use metricsql_parser::optimizer::{
    push_down_binary_op_filters_in_place, trim_filters_by_match_modifier,
};
use metricsql_parser::prelude::{BinModifier, Expr, Operator};

use crate::execution::binary::{exec_vector_vector_binop, get_common_label_filters};
use crate::execution::{compile_expression, Context, EvalConfig};
use crate::RuntimeResult;
use crate::types::{InstantVector, QueryValue};

use super::utils::{resolve_vector};
use super::ExecutableNode;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorVectorBinaryNode {
    pub op: Operator,
    pub modifier: Option<BinModifier>,
    pub left_idx: usize,
    pub right_idx: usize,
    #[serde(skip)]
    pub(crate) left: InstantVector,
    #[serde(skip)]
    pub(crate) right: InstantVector,
}

impl VectorVectorBinaryNode {
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

impl ExecutableNode for VectorVectorBinaryNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.left = resolve_vector(self.left_idx, dependencies)?;
        self.right = resolve_vector(self.right_idx, dependencies)?;
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, _ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let left = std::mem::take(&mut self.left);
        let right = std::mem::take(&mut self.right);
        exec_vector_vector_binop(ctx, left, right, self.op, &self.modifier)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorVectorPushDownNode {
    pub op: Operator,
    pub left_idx: usize,
    pub modifier: Option<BinModifier>,
    #[serde(skip)]
    pub(crate) left: InstantVector,
    #[serde(skip)]
    pub(crate) right: Expr,
    pub(super) is_swapped: bool,
}

impl ExecutableNode for VectorVectorPushDownNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.left = resolve_vector(self.left_idx, dependencies)?;
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let left = std::mem::take(&mut self.left);
        let right = self.fetch_right(ctx, ec)?;
        if self.is_swapped {
            exec_vector_vector_binop(ctx, right, left, self.op, &self.modifier)
        } else {
            exec_vector_vector_binop(ctx, left, right, self.op, &self.modifier)
        }
    }
}

impl VectorVectorPushDownNode {
    /// Execute binary operation in the following way:
    ///
    /// 1) execute the expr_first
    /// 2) get common label filters for series returned at step 1
    /// 3) push down the found common label filters to expr_second. This filters out unneeded series
    ///    during expr_second execution instead of spending compute resources on extracting and
    ///    processing these series before they are dropped later when matching time series according to
    ///    https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
    /// 4) execute the expr_second with possible additional filters found at step 3
    ///
    /// Typical use cases:
    /// - Kubernetes-related: show pod creation time with the node name:
    ///
    ///     kube_pod_created{namespace="prod"} * on (uid) group_left(node) kube_pod_info
    ///
    ///   Without the optimization `kube_pod_info` would select and spend compute resources
    ///   for more time series than needed. The selected time series would be dropped later
    ///   when matching time series on the right and left sides of binary operand.
    ///
    /// - Generic alerting queries, which rely on `info` metrics.
    ///   See https://grafana.com/blog/2021/08/04/how-to-use-promql-joins-for-more-effective-queries-of-prometheus-metrics-at-scale/
    ///
    /// - Queries, which get additional labels from `info` metrics.
    ///   See https://www.robustperception.io/exposing-the-software-version-to-prometheus
    ///
    /// Invariant: self.lhs and self.rhs are both ValueType::InstantVector
    fn fetch_right(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<InstantVector> {
        // if first.is_empty() && self.op == Or, the result will be empty,
        // since the "expr_first op expr_second" would return an empty result in any case.
        // https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3349

        if self.left.is_empty() && self.op == Operator::Or {
            return Ok(Default::default());
        }
        // push down filters if applicable
        let mut common_filters = get_common_label_filters(&self.left[0..]);
        if !common_filters.is_empty() {
            if let Some(modifier) = &self.modifier {
                trim_filters_by_match_modifier(&mut common_filters, &modifier.matching);
            }
            let mut copy = self.right.clone();
            push_down_binary_op_filters_in_place(&mut copy, &mut common_filters);
            execute_expr(ctx, ec, &copy)
        } else {
            execute_expr(ctx, ec, &self.right)
        }
    }
}

fn execute_expr(ctx: &Context, ec: &EvalConfig, expr: &Expr) -> RuntimeResult<InstantVector> {
    let mut eval_node = compile_expression(expr)?;
    let value = eval_node.execute(ctx, ec)?;
    value.into_instant_vector(ec)
}
