use std::sync::Arc;

use metricsql::ast::*;
use metricsql::common::{LabelFilter, Value};
use metricsql::functions::{BuiltinFunction, Signature, Volatility};
use metricsql::prelude::{BinaryOpKind, ValueType};

use crate::context::Context;
use crate::eval::aggregate::{create_aggr_evaluator, AggregateEvaluator};
use crate::eval::binop_handlers::should_reset_metric_group;
use crate::eval::binop_scalar_scalar::BinaryEvaluatorScalarScalar;
use crate::eval::binop_scalar_vector::BinaryEvaluatorScalarVector;
use crate::eval::binop_vector_scalar::BinaryEvaluatorVectorScalar;
use crate::eval::binop_vector_vector::BinaryEvaluatorVectorVector;
use crate::eval::duration::DurationEvaluator;
use crate::eval::function::TransformEvaluator;
use crate::eval::instant_vector::InstantVectorEvaluator;
use crate::eval::scalar::ScalarEvaluator;
use crate::eval::string::StringEvaluator;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::Deadline;
use crate::types::{Timeseries, Timestamp};
use crate::{QueryValue, TimestampTrait};

use super::rollup::RollupEvaluator;
use super::traits::{Evaluator, NullEvaluator};

pub enum ExprEvaluator {
    Null(NullEvaluator),
    Aggregate(AggregateEvaluator),
    ScalarScalar(BinaryEvaluatorScalarScalar),
    ScalarVector(BinaryEvaluatorScalarVector),
    VectorScalar(BinaryEvaluatorVectorScalar),
    VectorVector(BinaryEvaluatorVectorVector),
    Duration(DurationEvaluator),
    Function(TransformEvaluator),
    Number(ScalarEvaluator),
    Rollup(RollupEvaluator),
    String(StringEvaluator),
    InstantVector(InstantVectorEvaluator),
}

impl ExprEvaluator {
    /// returns true if the evaluator returns a const value i.e. calling it is
    /// essentially idempotent
    pub fn is_const(&self) -> bool {
        match self {
            ExprEvaluator::Number(_) | ExprEvaluator::String(_) | ExprEvaluator::Null(_) => true,
            ExprEvaluator::Duration(d) => d.is_const(),
            _ => false,
        }
    }
}

impl Value for ExprEvaluator {
    fn value_type(&self) -> ValueType {
        match self {
            ExprEvaluator::Null(e) => e.value_type(),
            ExprEvaluator::Aggregate(ae) => ae.value_type(),
            ExprEvaluator::VectorVector(bo) => bo.value_type(),
            ExprEvaluator::ScalarVector(sv) => sv.value_type(),
            ExprEvaluator::VectorScalar(vs) => vs.value_type(),
            ExprEvaluator::ScalarScalar(ss) => ss.value_type(),
            ExprEvaluator::Duration(de) => de.value_type(),
            ExprEvaluator::Function(fe) => fe.value_type(),
            ExprEvaluator::Number(n) => n.value_type(),
            ExprEvaluator::Rollup(re) => re.value_type(),
            ExprEvaluator::String(se) => se.value_type(),
            ExprEvaluator::InstantVector(iv) => iv.value_type(),
        }
    }
}

impl Evaluator for ExprEvaluator {
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        match self {
            ExprEvaluator::Null(e) => e.eval(ctx, ec),
            ExprEvaluator::Aggregate(ae) => ae.eval(ctx, ec),
            ExprEvaluator::VectorVector(bo) => bo.eval(ctx, ec),
            ExprEvaluator::VectorScalar(vs) => vs.eval(ctx, ec),
            ExprEvaluator::ScalarVector(sv) => sv.eval(ctx, ec),
            ExprEvaluator::ScalarScalar(ss) => ss.eval(ctx, ec),
            ExprEvaluator::Duration(de) => de.eval(ctx, ec),
            ExprEvaluator::Function(fe) => fe.eval(ctx, ec),
            ExprEvaluator::Number(n) => n.eval(ctx, ec),
            ExprEvaluator::Rollup(re) => re.eval(ctx, ec),
            ExprEvaluator::String(se) => se.eval(ctx, ec),
            ExprEvaluator::InstantVector(iv) => iv.eval(ctx, ec),
        }
    }

    fn return_type(&self) -> ValueType {
        match self {
            ExprEvaluator::Null(e) => e.return_type(),
            ExprEvaluator::Aggregate(ae) => ae.return_type(),
            ExprEvaluator::VectorVector(bo) => bo.return_type(),
            ExprEvaluator::ScalarVector(sv) => sv.return_type(),
            ExprEvaluator::VectorScalar(vs) => vs.return_type(),
            ExprEvaluator::ScalarScalar(ss) => ss.return_type(),
            ExprEvaluator::Duration(de) => de.return_type(),
            ExprEvaluator::Function(fe) => fe.return_type(),
            ExprEvaluator::Number(n) => n.return_type(),
            ExprEvaluator::Rollup(re) => re.return_type(),
            ExprEvaluator::String(se) => se.return_type(),
            ExprEvaluator::InstantVector(iv) => iv.return_type(),
        }
    }
}

impl Default for ExprEvaluator {
    fn default() -> Self {
        ExprEvaluator::Null(NullEvaluator {})
    }
}

impl From<i64> for ExprEvaluator {
    fn from(val: i64) -> Self {
        Self::Number(ScalarEvaluator::from(val as f64))
    }
}

impl From<f64> for ExprEvaluator {
    fn from(val: f64) -> Self {
        Self::Number(ScalarEvaluator::from(val))
    }
}

impl From<String> for ExprEvaluator {
    fn from(val: String) -> Self {
        Self::String(StringEvaluator::new(&val))
    }
}

impl From<&str> for ExprEvaluator {
    fn from(val: &str) -> Self {
        Self::String(StringEvaluator::new(val))
    }
}

pub fn create_evaluator(expr: &Expr) -> RuntimeResult<ExprEvaluator> {
    use Expr::*;

    match expr {
        Aggregation(ae) => create_aggr_evaluator(ae),
        MetricExpression(me) => Ok(ExprEvaluator::Rollup(
            RollupEvaluator::from_metric_expression(me.clone())?,
        )),
        Rollup(re) => Ok(ExprEvaluator::Rollup(RollupEvaluator::new(re)?)),
        Function(fe) => create_function_evaluator(fe),
        BinaryOperator(be) => create_binary_evaluator(be),
        Number(ne) => Ok(ExprEvaluator::from(ne.value)),
        Parens(pe) => create_parens_evaluator(pe),
        StringLiteral(se) => Ok(ExprEvaluator::from(se.clone())),
        Duration(de) => {
            if de.requires_step {
                Ok(ExprEvaluator::Duration(DurationEvaluator::new(de)))
            } else {
                Ok(ExprEvaluator::from(de.value))
            }
        }
        StringExpr(se) => {
            if se.is_literal_only() {
                Ok(ExprEvaluator::from(se.to_string()))
            } else {
                unreachable!("String expression should have been removed in parser")
            }
        }
        With(_) => {
            unreachable!("With expression should have been removed in parser")
        }
    }
}

fn create_function_evaluator(fe: &FunctionExpr) -> RuntimeResult<ExprEvaluator> {
    match fe.function {
        BuiltinFunction::Rollup(_) => {
            let eval = RollupEvaluator::from_function(fe)?;
            Ok(ExprEvaluator::Rollup(eval))
        }
        // note: aggregations produce another ast node type, so we don't need to handle them here
        _ => {
            let fe = TransformEvaluator::new(fe)?;
            Ok(ExprEvaluator::Function(fe))
        }
    }
}

fn create_parens_evaluator(pe: &ParensExpr) -> RuntimeResult<ExprEvaluator> {
    return if pe.len() == 1 {
        create_evaluator(&pe.expressions[0])
    } else {
        let func = pe.clone().to_function();
        create_function_evaluator(&func)
    };
}

fn create_binary_evaluator(be: &BinaryExpr) -> RuntimeResult<ExprEvaluator> {
    use BinaryOpKind::*;

    let op_kind = be.op.kind();
    let keep_metric_names =
        be.keep_metric_names || should_reset_metric_group(&be.op, be.bool_modifier);
    match (be.left.return_type(), op_kind, be.right.return_type()) {
        (ValueType::Scalar, Arithmetic | Comparison, ValueType::Scalar) => {
            assert!(Comparison != op_kind || be.bool_modifier);
            debug_assert!(be.modifier.is_none());
            Ok(ExprEvaluator::ScalarScalar(
                BinaryEvaluatorScalarScalar::new(be.op, &be.left, &be.right, be.bool_modifier)?,
            ))
        }
        (ValueType::Scalar, Arithmetic | Comparison, ValueType::InstantVector) => {
            assert!(!be.bool_modifier || Comparison == op_kind);
            debug_assert!(be.modifier.is_none());
            Ok(ExprEvaluator::ScalarVector(
                BinaryEvaluatorScalarVector::new(
                    be.op,
                    &be.left,
                    &be.right,
                    be.bool_modifier,
                    keep_metric_names,
                )?,
            ))
        }
        (ValueType::InstantVector, Arithmetic | Comparison, ValueType::Scalar) => {
            assert!(!be.bool_modifier || Comparison == op_kind);
            debug_assert!(be.modifier.is_none());
            Ok(ExprEvaluator::VectorScalar(
                BinaryEvaluatorVectorScalar::new(
                    be.op,
                    &be.left,
                    &be.right,
                    be.bool_modifier,
                    keep_metric_names,
                )?,
            ))
        }
        (ValueType::InstantVector, Arithmetic | Comparison | Logical, ValueType::InstantVector) => {
            assert!(!be.bool_modifier || Comparison == op_kind);
            debug_assert!(be.modifier.is_none());
            Ok(ExprEvaluator::VectorVector(
                BinaryEvaluatorVectorVector::new(be)?,
            ))
        }
        (lk, ok, rk) => unimplemented!("{:?} {:?} {:?} operation is not supported", lk, ok, rk),
    }
}

pub(crate) fn create_evaluators(vec: &[Expr]) -> RuntimeResult<Vec<ExprEvaluator>> {
    let mut res: Vec<ExprEvaluator> = Vec::with_capacity(vec.len());
    for arg in vec.iter() {
        match create_evaluator(arg) {
            Err(e) => return Err(e),
            Ok(eval) => res.push(eval),
        }
    }
    Ok(res)
}

/// validate_max_points_per_timeseries checks the maximum number of points that
/// may be returned per each time series.
///
/// The number mustn't exceed max_points_per_timeseries.
pub(crate) fn validate_max_points_per_timeseries(
    start: Timestamp,
    end: Timestamp,
    step: i64,
    max_points_per_timeseries: usize,
) -> RuntimeResult<()> {
    let points = (end - start) / step + 1;
    if (max_points_per_timeseries > 0) && points > max_points_per_timeseries as i64 {
        let msg = format!(
            "too many points for the given step={}, start={} and end={}: {}; cannot exceed {}",
            step, start, end, points, max_points_per_timeseries
        );
        Err(RuntimeError::from(msg))
    } else {
        Ok(())
    }
}

/// The minimum number of points per timeseries for enabling time rounding.
/// This improves cache hit ratio for frequently requested queries over
/// big time ranges.
const MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: i64 = 50;

pub fn adjust_start_end(start: Timestamp, end: Timestamp, step: i64) -> (Timestamp, Timestamp) {
    // if disableCache {
    //     // do not adjust start and end values when cache is disabled.
    //     // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/563
    //     return (start, end);
    // }
    let points = (end - start) / step + 1;
    if points < MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING {
        // Too small number of points for rounding.
        return (start, end);
    }

    // Round start and end to values divisible by step in order
    // to enable response caching (see EvalConfig.mayCache).
    let (start, end) = align_start_end(start, end, step);

    // Make sure that the new number of points is the same as the initial number of points.
    let mut new_points = (end - start) / step + 1;
    let mut _end = end;
    while new_points > points {
        _end = end - step;
        new_points -= 1;
    }

    return (start, _end);
}

pub fn align_start_end(start: Timestamp, end: Timestamp, step: i64) -> (Timestamp, Timestamp) {
    // Round start to the nearest smaller value divisible by step.
    let new_start = start - start % step;
    // Round end to the nearest bigger value divisible by step.
    let adjust = end % step;
    let mut new_end = end;
    if adjust > 0 {
        new_end += step - adjust
    }
    return (new_start, new_end);
}

#[derive(Clone)]
pub struct EvalConfig {
    pub start: Timestamp,
    pub end: Timestamp,
    pub step: i64, // todo: Duration

    /// max_series is the maximum number of time series which can be scanned by the query.
    /// Zero means 'no limit'
    pub max_series: usize,

    /// quoted remote address.
    pub quoted_remote_addr: Option<String>,

    pub deadline: Deadline,

    /// Whether the response can be cached.
    _may_cache: bool,

    /// lookback_delta is analog to `-query.lookback-delta` from Prometheus.
    /// todo: change type to Duration
    pub lookback_delta: i64,

    /// How many decimal digits after the point to leave in response.
    pub round_digits: i16,

    /// enforced_tag_filters may contain additional label filters to use in the query.
    pub enforced_tag_filters: Vec<Vec<LabelFilter>>,

    /// Set this flag to true if the data doesn't contain Prometheus stale markers, so there is
    /// no need in spending additional CPU time on its handling. Staleness markers may exist only in
    /// data obtained from Prometheus scrape targets
    pub no_stale_markers: bool,

    /// The limit on the number of points which can be generated per each returned time series.
    pub max_points_per_series: usize,

    /// Whether to disable response caching. This may be useful during data back-filling
    pub disable_cache: bool,

    _timestamps: Arc<Vec<Timestamp>>,
}

impl EvalConfig {
    pub fn new(start: Timestamp, end: Timestamp, step: i64) -> Self {
        let mut result = EvalConfig::default();
        result.start = start;
        result.end = end;
        result.step = step;
        result
    }

    pub fn copy_no_timestamps(&self) -> EvalConfig {
        let ec = EvalConfig {
            start: self.start,
            end: self.end,
            step: self.step,
            deadline: self.deadline,
            max_series: self.max_series,
            quoted_remote_addr: self.quoted_remote_addr.clone(),
            _may_cache: self._may_cache,
            lookback_delta: self.lookback_delta,
            round_digits: self.round_digits,
            enforced_tag_filters: self.enforced_tag_filters.clone(),
            // do not copy src.timestamps - they must be generated again.
            _timestamps: Arc::new(vec![]),
            no_stale_markers: self.no_stale_markers,
            max_points_per_series: self.max_points_per_series,
            disable_cache: self.disable_cache,
        };
        return ec;
    }

    pub fn adjust_by_offset(&mut self, offset: i64) {
        self.start -= offset;
        self.end -= offset;
        self._timestamps = Arc::new(vec![]);
    }

    pub fn validate(&self) -> RuntimeResult<()> {
        if self.start > self.end {
            let msg = format!(
                "BUG: start cannot exceed end; got {} vs {}",
                self.start, self.end
            );
            return Err(RuntimeError::from(msg));
        }
        if self.step <= 0 {
            let msg = format!("BUG: step must be greater than 0; got {}", self.step);
            return Err(RuntimeError::from(msg));
        }
        Ok(())
    }

    pub fn may_cache(&self) -> bool {
        if self.disable_cache {
            return false;
        }
        if self._may_cache {
            return true;
        }
        if self.start % self.step != 0 {
            return false;
        }
        if self.end % self.step != 0 {
            return false;
        }

        true
    }

    pub fn no_cache(&mut self) {
        self._may_cache = false
    }

    pub fn set_caching(&mut self, may_cache: bool) {
        self._may_cache = may_cache;
    }

    pub fn update_from_context(&mut self, ctx: &Context) {
        let state_config = &ctx.config;
        self.disable_cache = state_config.disable_cache;
        self.max_points_per_series = state_config.max_points_subquery_per_timeseries;
        self.no_stale_markers = state_config.no_stale_markers;
        self.lookback_delta = state_config.max_lookback.num_milliseconds();
        self.max_series = state_config.max_unique_timeseries;
    }

    pub fn timestamps(&self) -> Arc<Vec<i64>> {
        Arc::clone(&self._timestamps)
    }

    pub fn get_timestamps(&mut self) -> Arc<Vec<Timestamp>> {
        self.ensure_timestamps().unwrap(); //???
        Arc::clone(&self._timestamps)
    }

    pub fn timerange_string(&self) -> String {
        format!("[{}..{}]", self.start.to_rfc3339(), self.end.to_rfc3339())
    }

    pub(crate) fn ensure_timestamps(&mut self) -> RuntimeResult<()> {
        if self._timestamps.len() == 0 {
            let ts = get_timestamps(self.start, self.end, self.step, self.max_points_per_series)?;
            self._timestamps = Arc::new(ts);
        }
        Ok(())
    }

    pub fn get_shared_timestamps(&mut self) -> Arc<Vec<i64>> {
        self.get_timestamps()
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            step: 0,
            max_series: 0,
            quoted_remote_addr: None,
            deadline: Deadline::default(),
            _may_cache: false,
            lookback_delta: 0,
            round_digits: 100,
            enforced_tag_filters: vec![],
            no_stale_markers: true,
            max_points_per_series: 0,
            disable_cache: false,
            _timestamps: Arc::new(vec![]),
        }
    }
}

impl From<&Context> for EvalConfig {
    fn from(ctx: &Context) -> Self {
        let mut config = EvalConfig::default();
        config.update_from_context(ctx);
        config
    }
}

pub fn get_timestamps(
    start: Timestamp,
    end: Timestamp,
    step: i64,
    max_timestamps_per_timeseries: usize,
) -> RuntimeResult<Vec<i64>> {
    // Sanity checks.
    if step <= 0 {
        let msg = format!("Step must be bigger than 0; got {}", step);
        return Err(RuntimeError::from(msg));
    }

    if start > end {
        let msg = format!("Start cannot exceed End; got {} vs {}", start, end);
        return Err(RuntimeError::from(msg));
    }

    if let Err(err) =
        validate_max_points_per_timeseries(start, end, step, max_timestamps_per_timeseries)
    {
        let msg = format!(
            "BUG: {:?}; this must be validated before the call to get_timestamps",
            err
        );
        return Err(RuntimeError::from(msg));
    }

    // Prepare timestamps.
    let n: usize = (1 + (end - start) / step) as usize;
    // todo: use a pool
    let mut timestamps: Vec<i64> = Vec::with_capacity(n);
    for ts in (start..=end).step_by(step as usize) {
        timestamps.push(ts);
    }

    return Ok(timestamps);
}

pub(crate) fn eval_number(ec: &EvalConfig, n: f64) -> Vec<Timeseries> {
    let timestamps = ec.timestamps();
    // HACK!!!  ec.ensure_timestamps() should have been called before this function
    if timestamps.len() == 0 {
        // todo: this is a hack, we should not call get_timestamps here
        let timestamps =
            get_timestamps(ec.start, ec.end, ec.step, ec.max_points_per_series).unwrap();
        let values = vec![n; timestamps.len()];
        let ts = Timeseries::new(timestamps, values);
        return vec![ts];
    }
    let ts = Timeseries {
        metric_name: Default::default(),
        timestamps: timestamps.clone(),
        values: vec![n; timestamps.len()],
    };
    vec![ts]
}

pub(crate) fn eval_time(ec: &EvalConfig) -> Vec<Timeseries> {
    let mut rv = eval_number(ec, f64::NAN);
    let timestamps = rv[0].timestamps.clone(); // this is an Arc, so it's cheap to clone
    for (ts, val) in timestamps.iter().zip(rv[0].values.iter_mut()) {
        *val = (*ts as f64) / 1e3_f64;
    }
    rv
}

pub(super) fn eval_volatility(sig: &Signature, args: &Vec<ExprEvaluator>) -> Volatility {
    if sig.volatility != Volatility::Immutable {
        return sig.volatility;
    }

    let mut has_volatile = false;
    let mut mutable = false;

    for arg in args.iter() {
        let vol = arg.volatility();
        if vol != Volatility::Immutable {
            mutable = true;
            has_volatile = vol == Volatility::Volatile;
        }
    }

    if mutable {
        return if has_volatile {
            Volatility::Volatile
        } else {
            Volatility::Stable
        };
    } else {
        Volatility::Immutable
    }
}
