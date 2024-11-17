use std::time::Duration;
use std::default::Default;
use std::str::FromStr;

use metricsql_parser::ast::{DurationExpr, StringLiteral};

use crate::execution::dag::absent_transform_node::AbsentTransformNode;
use crate::execution::dag::subquery_node::SubqueryNode;
use crate::execution::dag::utils::resolve_value;
use crate::execution::dag::vector_vector_binary_node::VectorVectorPushDownNode;
use crate::execution::{Context, EvalConfig};
use crate::{RuntimeResult};
use crate::types::QueryValue;
use super::aggregate_node::AggregateNode;
use super::binop_node::BinopNode;
use super::duration_node::DurationNode;
use super::dynamic_node::DynamicNode;
use super::rollup_node::RollupNode;
use super::scalar_vector_binop_node::ScalarVectorBinaryNode;
use super::selector_node::SelectorNode;
use super::transform_node::TransformNode;
use super::vector_scalar_binop_node::VectorScalarBinaryNode;
use super::vector_vector_binary_node::VectorVectorBinaryNode;

pub trait ExecutableNode {
    // separated as a method so we don't have to clone() args or run afoul of the borrow
    // checker with the dependencies. It also has the benefit of not requiring locking during the
    // evaluation process
    fn pre_execute(&mut self, _dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        Ok(())
    }
    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodeArg {
    Index(usize),
    Value(QueryValue),
}

impl NodeArg {
    pub fn resolve(&self, value: &mut QueryValue, dependencies: &mut [QueryValue]) {
        match self {
            NodeArg::Index(i) => {
                resolve_value(*i, value, dependencies);
            }
            NodeArg::Value(v) => {
                match v {
                    QueryValue::InstantVector(_) | QueryValue::RangeVector(_) => {
                        *value = v.clone() // ?? i dont think we'll get here
                    }
                    QueryValue::Scalar(v) => *value = QueryValue::Scalar(*v),
                    QueryValue::String(s) => *value = QueryValue::from(s.as_str()),
                }
            }
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(self, NodeArg::Value(_))
    }
}

impl Default for NodeArg {
    fn default() -> Self {
        NodeArg::Value(QueryValue::Scalar(f64::NAN))
    }
}

impl From<usize> for NodeArg {
    fn from(i: usize) -> Self {
        NodeArg::Index(i)
    }
}

impl From<QueryValue> for NodeArg {
    fn from(qv: QueryValue) -> Self {
        NodeArg::Value(qv)
    }
}

impl From<f64> for NodeArg {
    fn from(f: f64) -> Self {
        NodeArg::Value(QueryValue::Scalar(f))
    }
}

impl From<&str> for NodeArg {
    fn from(s: &str) -> Self {
        NodeArg::Value(QueryValue::String(s.to_string()))
    }
}

impl From<i64> for NodeArg {
    fn from(i: i64) -> Self {
        NodeArg::Value(QueryValue::Scalar(i as f64))
    }
}

impl FromStr for NodeArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(NodeArg::Value(QueryValue::String(s.to_string())))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DAGNode {
    Value(QueryValue),
    Aggregate(AggregateNode),
    Absent(AbsentTransformNode),
    Transform(TransformNode),
    Rollup(RollupNode),
    Duration(DurationNode),
    Selector(SelectorNode),
    Subquery(SubqueryNode),
    ScalarVectorOp(ScalarVectorBinaryNode),
    VectorVectorOp(VectorVectorBinaryNode),
    VectorVectorPushDownOp(VectorVectorPushDownNode),
    VectorScalarOp(VectorScalarBinaryNode),
    BinOp(BinopNode),
    Dynamic(DynamicNode),
}

impl DAGNode {
    pub fn is_value(&self) -> bool {
        matches!(self, DAGNode::Value(_))
    }

    pub fn execute(&mut self, context: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        ExecutableNode::execute(self, context, ec)
    }
}

impl ExecutableNode for DAGNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        match self {
            DAGNode::Value(_) => Ok(()),
            DAGNode::Absent(node) => node.pre_execute(dependencies),
            DAGNode::Aggregate(node) => node.pre_execute(dependencies),
            DAGNode::Transform(node) => node.pre_execute(dependencies),
            DAGNode::Rollup(node) => node.pre_execute(dependencies),
            DAGNode::Duration(node) => node.pre_execute(dependencies),
            DAGNode::Selector(node) => node.pre_execute(dependencies),
            DAGNode::ScalarVectorOp(node) => node.pre_execute(dependencies),
            DAGNode::Subquery(sub) => sub.pre_execute(dependencies),
            DAGNode::VectorVectorOp(node) => node.pre_execute(dependencies),
            DAGNode::VectorScalarOp(node) => node.pre_execute(dependencies),
            DAGNode::BinOp(node) => node.pre_execute(dependencies),
            DAGNode::Dynamic(node) => node.pre_execute(dependencies),
            DAGNode::VectorVectorPushDownOp(node) => node.pre_execute(dependencies),
        }
    }
    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        match self {
            DAGNode::Value(node) => Ok(node.clone()),
            DAGNode::Aggregate(node) => node.execute(ctx, ec),
            DAGNode::Absent(node) => node.execute(ctx, ec),
            DAGNode::Transform(node) => node.execute(ctx, ec),
            DAGNode::Rollup(node) => node.execute(ctx, ec),
            DAGNode::Duration(node) => node.execute(ctx, ec),
            DAGNode::ScalarVectorOp(node) => node.execute(ctx, ec),
            DAGNode::Subquery(node) => node.execute(ctx, ec),
            DAGNode::VectorVectorOp(node) => node.execute(ctx, ec),
            DAGNode::VectorScalarOp(node) => node.execute(ctx, ec),
            DAGNode::BinOp(node) => node.execute(ctx, ec),
            DAGNode::Dynamic(node) => node.execute(ctx, ec),
            DAGNode::VectorVectorPushDownOp(node) => node.execute(ctx, ec),
            DAGNode::Selector(_) => {
                todo!("selector")
            }
        }
    }
}

impl Default for DAGNode {
    fn default() -> Self {
        DAGNode::Value(QueryValue::Scalar(f64::NAN))
    }
}

impl From<String> for DAGNode {
    fn from(s: String) -> Self {
        DAGNode::Value(QueryValue::String(s))
    }
}

impl From<StringLiteral> for DAGNode {
    fn from(s: StringLiteral) -> Self {
        DAGNode::Value(QueryValue::String(s.to_string()))
    }
}

impl From<&str> for DAGNode {
    fn from(s: &str) -> Self {
        DAGNode::Value(QueryValue::String(s.to_string()))
    }
}

impl From<f64> for DAGNode {
    fn from(f: f64) -> Self {
        DAGNode::Value(QueryValue::Scalar(f))
    }
}

impl From<i64> for DAGNode {
    fn from(i: i64) -> Self {
        DAGNode::Value(QueryValue::Scalar(i as f64))
    }
}

impl From<DurationExpr> for DAGNode {
    fn from(de: DurationExpr) -> Self {
        (&de).into()
    }
}

impl From<&DurationExpr> for DAGNode {
    fn from(de: &DurationExpr) -> Self {
        if !de.requires_step() {
            let val = de.value(Duration::from_millis(1));
            let d_sec = val as f64 / 1000_f64;
            DAGNode::from(d_sec)
        } else {
            DAGNode::Duration(DurationNode(de.clone()))
        }
    }
}
