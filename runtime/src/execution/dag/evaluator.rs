use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};

use crate::execution::dag::ExecutableNode;
use crate::execution::{Context, EvalConfig};
use crate::rayon::iter::ParallelIterator;
use crate::{types::QueryValue, RuntimeResult};

use super::DAGNode;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dependency {
    pub(crate) node: DAGNode,
    pub(crate) result_index: usize,
}

impl Dependency {
    pub(crate) fn new(node: DAGNode, result_index: usize) -> Dependency {
        Dependency { node, result_index }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DAGEvaluator {
    /// Adjacency list of DAG nodes
    /// The first vec contains the nodes that have no dependencies.
    /// Each subsequent vec contains the nodes that depend on the nodes in the previous layer.
    /// The last vec contains nodes that depend on all other nodes.
    pub(super) dag: Vec<Vec<Dependency>>,

    /// collects the results of the execution of dependencies (1 per dependency). The results are
    /// fed to the nodes that depend on them in a pre-execute phase.
    pub(super) computed: Vec<QueryValue>,

    max_list_len: usize,
}

struct ExecResult {
    index: usize,
    value: RuntimeResult<QueryValue>,
}

impl Default for ExecResult {
    fn default() -> Self {
        ExecResult {
            index: 0,
            value: Ok(QueryValue::default()),
        }
    }
}

impl DAGEvaluator {
    pub(crate) fn new(nodes: Vec<Vec<Dependency>>, node_count: usize) -> DAGEvaluator {
        // todo: SmallVec, chili
        // allocate scratch space for the results of the dependencies. We initialize the entire
        // vector with default values, and then we overwrite the values as we execute the
        // dependencies.
        let mut computed = Vec::with_capacity(node_count + 1);
        computed.resize_with(node_count + 1, Default::default);
        let max_list_len = nodes.iter().map(|list| list.len()).max().unwrap_or(0);

        DAGEvaluator {
            dag: nodes,
            computed,
            max_list_len,
        }
    }

    pub fn evaluate(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        // reusable buffer to avoid extra allocations in collect()
        let mut buf: Option<Vec<ExecResult>> = None;

        for (i, dependencies) in self.dag.iter_mut().enumerate() {
            // update dependencies
            if i > 0 {
                for dep in dependencies.iter_mut() {
                    dep.node.pre_execute(&mut self.computed)?;
                }
            }

            let len = dependencies.len();
            let computed = &mut self.computed[0..];
            match len {
                1 => {
                    let dependency = &mut dependencies[0];
                    let idx = dependency.result_index;
                    let node = &mut dependency.node;
                    computed[idx] = node.execute(ctx, ec)?;
                }
                2 => {
                    // arrrgh! rust treats vecs as a single unit wrt borrowing, but the following iterator trick
                    // seems to work
                    let mut iter = dependencies.iter_mut();
                    let first = iter.next().unwrap();
                    let second = iter.next().unwrap();
                    let (first_val, second_val) = rayon::join(
                        || first.node.execute(ctx, ec),
                        || second.node.execute(ctx, ec),
                    );
                    computed[first.result_index] = first_val?;
                    computed[second.result_index] = second_val?;
                }
                _ => {
                    if buf.is_none() {
                        buf = Some(Vec::with_capacity(self.max_list_len));
                    }
                    match buf.as_mut() {
                        Some(buf) => {
                            dependencies
                                .par_iter_mut()
                                .map(|dependency| {
                                    let value = dependency.node.execute(ctx, ec);
                                    ExecResult {
                                        index: dependency.result_index,
                                        value,
                                    }
                                })
                                .collect_into_vec(buf);

                            // set values for subsequent nodes
                            for ExecResult { index, value } in buf.iter_mut() {
                                match value {
                                    // todo: is it worth it to optimize here for scalars ??
                                    Ok(v) => computed[*index] = std::mem::take(v),
                                    Err(e) => return Err(e.clone()),
                                }
                            }
                        }
                        None => unreachable!(),
                    }
                }
            }
        }

        Ok(std::mem::take(&mut self.computed[0]))
    }
}
