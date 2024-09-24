use tracing::{field, trace_span, Span};

use metricsql_parser::ast::Expr;

use crate::execution::dag::{ExecutableNode, NodeArg};
use crate::execution::{eval_number, Context, EvalConfig};
use crate::functions::transform::extract_labels_from_expr;
use crate::RuntimeResult;
use crate::types::{Labels, QueryValue, Timeseries};

/// Hack. This is a copy of TransformNode. This exists solely because
/// `absent` is the only transform function that requires access to the
/// original expression in order to get the set of labels to apply to the result vector.
/// Rather than pass around a param that's redundant in 99.9% of case we handle it here and
/// hopefully keep the code cleaner.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AbsentTransformNode {
    pub keep_metric_names: bool,
    pub labels: Option<Labels>,
    pub node_arg: NodeArg,
    pub(crate) arg: QueryValue,
    pub(crate) arg_const: bool,
}

impl AbsentTransformNode {
    pub fn from_arg_index(expr: &Expr, arg_index: usize) -> Self {
        let labels = extract_labels_from_expr(expr);
        Self {
            keep_metric_names: false,
            labels,
            node_arg: NodeArg::Index(arg_index),
            arg: QueryValue::default(),
            arg_const: false,
        }
    }

    pub fn from_arg(expr: &Expr, node_arg: NodeArg) -> Self {
        let labels = extract_labels_from_expr(expr);
        // todo: dedup labels
        let (arg, arg_const) = if node_arg.is_const() {
            let mut arg = QueryValue::default();
            node_arg.resolve(&mut arg, &mut []);
            (arg, true)
        } else {
            (QueryValue::default(), false)
        };

        Self {
            keep_metric_names: false,
            labels,
            node_arg,
            arg,
            arg_const,
        }
    }

    fn get_absent_timeseries(&self, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        // Copy tags from arg
        let mut rvs = eval_number(ec, 1.0)?;
        if let Some(labels) = &self.labels {
            for label in labels {
                rvs[0].metric_name.set(&label.name, &label.value);
            }
        }
        Ok(rvs)
    }
}

impl ExecutableNode for AbsentTransformNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.node_arg.resolve(&mut self.arg, dependencies);
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let trace_enabled = ctx.trace_enabled();
        let span = if trace_enabled {
            trace_span!("transform", function = "absent", series = field::Empty)
        } else {
            Span::none()
        }
        .entered();

        let tss = self.arg.as_instant_vec(ec)?;

        let mut rvs = self.get_absent_timeseries(ec)?;
        if !tss.is_empty() {
            for i in 0..tss[0].values.len() {
                let mut is_absent = true;
                for ts in tss.iter() {
                    if !ts.values[i].is_nan() {
                        is_absent = false;
                        break;
                    }
                }
                if !is_absent {
                    rvs[0].values[i] = f64::NAN
                }
            }
        }

        span.record("series", tss.len());
        Ok(QueryValue::InstantVector(rvs))
    }
}
