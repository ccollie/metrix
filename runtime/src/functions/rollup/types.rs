use std::fmt::Debug;
use std::sync::Arc;

use clone_dyn::clone_dyn;
use smallvec::SmallVec;

use crate::functions::rollup::TimeSeriesMap;
use crate::types::{QueryValue, Timestamp};
use crate::{RuntimeResult};

#[derive(Default, Clone, Debug)]
pub struct RollupFuncArg<'a> {
    /// The value preceding values if it fits staleness interval.
    pub(super) prev_value: f64,

    /// The timestamp for prev_value.
    pub(super) prev_timestamp: Timestamp,

    /// Values that fit window ending at curr_timestamp.
    pub(crate) values: &'a [f64],

    /// Timestamps for values.
    pub(crate) timestamps: &'a [Timestamp],

    /// Real value preceding values without restrictions on staleness interval.
    pub(super) real_prev_value: f64,

    /// Real value which goes after values.
    pub(crate) real_next_value: f64,

    /// Current timestamp for rollup evaluation.
    pub(super) curr_timestamp: Timestamp,

    /// Index for the currently evaluated point relative to time range for query evaluation.
    pub(super) idx: usize,

    /// Time window for rollup calculations.
    pub(super) window: i64,

    pub(super) tsm: Option<Arc<TimeSeriesMap>>,
}


impl RollupFuncArg<'_> {
    pub(crate) fn get_tsm(&self) -> Arc<TimeSeriesMap> {
        if let Some(tsm) = &self.tsm {
            tsm.clone()
        } else {
            panic!("BUG: tsm is None")
        }
    }
}

pub(crate) type RollupFunc = fn(rfa: &RollupFuncArg) -> f64;

#[clone_dyn]
/// RollupFunc must return rollup value for the given rfa.
///
/// prev_value may be NAN, values and timestamps may be empty.
pub trait RollupFn: Fn(&RollupFuncArg) -> f64 + Send + Sync {}

/// implement `Rollup` on any type that implements `Fn(&RollupFuncArg) -> f64`.
impl<T> RollupFn for T where T: Fn(&RollupFuncArg) -> f64 + Send + Sync + Clone {}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GenericRollupHandler<STATE, F>
where
    F: Fn(&RollupFuncArg, &STATE) -> f64 + Send + Sync,
    STATE: Clone + Debug,
{
    pub state: STATE,
    pub func: F,
}

impl<STATE, F> GenericRollupHandler<STATE, F>
where
    F: Fn(&RollupFuncArg, &STATE) -> f64 + Send + Sync,
    STATE: Clone + Debug,
{
    pub fn new(state: STATE, func: F) -> Self {
        Self { state, func }
    }

    pub(crate) fn eval(&self, arg: &RollupFuncArg) -> f64 {
        (self.func)(arg, &self.state)
    }
}

pub type RollupFuncFloatArg = fn(&RollupFuncArg, &f64) -> f64;
pub(crate) type RollupHandlerFloat = GenericRollupHandler<f64, fn(&RollupFuncArg, &f64) -> f64>;

pub(crate) type RollupHandlerVec =
    GenericRollupHandler<SmallVec<f64, 4>, fn(&RollupFuncArg, &SmallVec<f64, 4>) -> f64>;

#[derive(Clone, Debug)]
pub(crate) enum RollupHandler {
    Wrapped(RollupFunc),
    Fake(&'static str),
    FloatArg(RollupHandlerFloat),
    VecArg(RollupHandlerVec),
    General(Box<dyn RollupFn<Output = f64>>),
}

impl Debug for dyn RollupFn<Output = f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dyn RollupFn")
    }
}

impl RollupHandler {
    pub fn wrap(f: RollupFunc) -> Self {
        RollupHandler::Wrapped(f)
    }
    pub fn fake(name: &'static str) -> Self {
        RollupHandler::Fake(name)
    }

    pub fn float_arg(arg: f64, f: RollupFuncFloatArg) -> Self {
        let handler = RollupHandlerFloat::new(arg, f);
        RollupHandler::FloatArg(handler)
    }

    pub(crate) fn eval(&self, arg: &RollupFuncArg) -> f64 {
        match self {
            RollupHandler::Wrapped(wrapped) => wrapped(arg),
            RollupHandler::Fake(name) => {
                panic!("BUG: {} shouldn't be called", name);
            }
            RollupHandler::General(df) => df(arg),
            RollupHandler::FloatArg(f) => f.eval(arg),
            RollupHandler::VecArg(f) => f.eval(arg),
        }
    }
}

impl PartialEq for RollupHandler {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RollupHandler::Wrapped(left), RollupHandler::Wrapped(right)) => left == right,
            (RollupHandler::Fake(left), RollupHandler::Fake(right)) => left == right,
            (RollupHandler::FloatArg(left), RollupHandler::FloatArg(right)) => left == right,
            (RollupHandler::VecArg(left), RollupHandler::VecArg(right)) => left == right,
            (RollupHandler::General(_), RollupHandler::General(_)) => false,
            _ => false,
        }
    }
}

impl Default for RollupHandler {
    fn default() -> Self {
        RollupHandler::fake("default")
    }
}

pub(crate) type RollupHandlerFactory = fn(&[QueryValue]) -> RuntimeResult<RollupHandler>;
