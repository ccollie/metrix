pub(crate) use aggr_fns::{exec_aggregate_fn, AggrFuncArg};
pub(crate) use incremental::*;

mod aggr_fns;
#[cfg(test)]
mod aggr_incremental_test;
#[cfg(test)]
mod aggr_test;
mod incremental;