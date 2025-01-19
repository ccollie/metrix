use std::cmp::Ordering;
use std::fmt::{Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, Neg};
use std::str::FromStr;
use std::time::{Duration, SystemTime};
use std::{fmt, iter, ops};

use enquote::enquote;
use serde::{Deserialize, Serialize};

use metricsql_common::duration::fmt_duration_ms;

use crate::ast::utils::string_vecs_equal_unordered;
use crate::ast::{
    indent, prettify_args, Operator, Prettier, StringExpr, MAX_CHARACTERS_PER_LINE,
};
use crate::common::{hash_f64, write_comma_separated, write_number, Value, ValueType};
use crate::functions::{AggregateFunction, BuiltinFunction, TransformFunction};
use crate::label::{Labels, MatchOp, Matcher, Matchers, NAME_LABEL};
use crate::parser::{ParseError, ParseResult};
use crate::prelude::{
    get_aggregate_arg_idx_for_optimization, BuiltinFunctionType,
    RollupFunction,
};

pub type BExpr = Box<Expr>;

/// Matching Modifier, for VectorMatching of binary expr.
/// Label lists provided to matching keywords will determine how vectors are combined.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorMatchModifier {
    On(Labels),
    Ignoring(Labels),
}

impl Display for VectorMatchModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use VectorMatchModifier::*;
        match self {
            On(labels) => write!(f, "on({:?})", labels)?,
            Ignoring(labels) => write!(f, "ignoring({:?})", labels)?,
        }
        Ok(())
    }
}

impl VectorMatchModifier {
    pub fn new(labels: Vec<String>, is_on: bool) -> Self {
        let names = Labels::new_from_iter(labels);
        if is_on {
            VectorMatchModifier::On(names)
        } else {
            VectorMatchModifier::Ignoring(names)
        }
    }

    pub fn labels(&self) -> &Labels {
        match self {
            VectorMatchModifier::On(l) => l,
            VectorMatchModifier::Ignoring(l) => l,
        }
    }

    pub fn is_on(&self) -> bool {
        matches!(*self, VectorMatchModifier::On(_))
    }
}

/// Binary Expr Modifier
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinModifier {
    /// The matching behavior for the operation if both operands are Vectors.
    /// If they are not this field is None.
    pub card: VectorMatchCardinality,

    /// on/ignoring on labels.
    /// like a + b, no match modifier is needed.
    #[serde(default, skip_serializing_if = "is_default")]
    pub matching: Option<VectorMatchModifier>,

    /// If `keep_metric_names` is set to true, then the operation should keep metric names.
    #[serde(default, skip_serializing_if = "is_default")]
    pub keep_metric_names: bool,

    /// If a comparison operator, return 0/1 rather than filtering.
    /// For example, `foo > bool bar`.
    #[serde(default, skip_serializing_if = "is_default")]
    pub return_bool: bool,
}

impl Default for BinModifier {
    fn default() -> Self {
        Self {
            card: VectorMatchCardinality::OneToOne,
            matching: None,
            keep_metric_names: false,
            return_bool: false,
        }
    }
}

impl BinModifier {
    pub fn with_card(mut self, card: VectorMatchCardinality) -> Self {
        self.card = card;
        self
    }

    pub fn with_matching(mut self, matching: Option<VectorMatchModifier>) -> Self {
        self.matching = matching;
        if self.matching.is_none() {
            self.card = VectorMatchCardinality::OneToOne;
        }
        self
    }

    pub fn with_return_bool(mut self, return_bool: bool) -> Self {
        self.return_bool = return_bool;
        self
    }

    pub fn with_keep_metric_names(mut self, keep_metric_names: bool) -> Self {
        self.keep_metric_names = keep_metric_names;
        self
    }

    pub fn is_labels_joint(&self) -> bool {
        matches!((self.card.labels(), &self.matching),
                 (Some(labels), Some(matching)) if !labels.is_joint(matching.labels()))
    }

    pub fn intersect_labels(&self) -> Option<Vec<String>> {
        if let Some(labels) = self.card.labels() {
            if let Some(matching) = &self.matching {
                let res = labels.intersect(matching.labels());
                return Some(res.0);
            }
        };
        None
    }

    pub fn is_matching_on(&self) -> bool {
        matches!(&self.matching, Some(matching) if matching.is_on())
    }

    pub fn is_matching_labels_not_empty(&self) -> bool {
        matches!(&self.matching, Some(matching) if !matching.labels().is_empty())
    }

    pub fn is_default(&self) -> bool {
        self.card == VectorMatchCardinality::OneToOne
            && self.matching.is_none()
            && !self.keep_metric_names
            && !self.return_bool
    }
}

impl Display for BinModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use VectorMatchCardinality::*;
        if self.return_bool {
            write!(f, "bool")?;
        }
        match &self.card {
            ManyToOne(labels) => {
                write!(f, " group_left")?;
                write_comma_separated(labels.iter(), f, true)?;
            }
            OneToMany(labels) => {
                write!(f, " group_right")?;
                write_comma_separated(labels.iter(), f, true)?;
            }
            _ => {}
        }
        if let Some(matching) = &self.matching {
            match matching {
                VectorMatchModifier::On(labels) => {
                    write!(f, " on")?;
                    write_comma_separated(labels.iter(), f, true)?;
                }
                VectorMatchModifier::Ignoring(labels) => {
                    write!(f, " ignoring")?;
                    write_comma_separated(labels.iter(), f, true)?;
                }
            }
        }
        if self.keep_metric_names {
            write!(f, " keep_metric_names")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub enum AggregateModifier {
    // todo: use BtreeSet<String>, since the runtime expects these to be sorted
    By(Vec<String>),
    Without(Vec<String>),
}

impl AggregateModifier {
    /// Creates a new AggregateModifier with the Left op
    pub fn by() -> Self {
        AggregateModifier::By(vec![])
    }

    /// Creates a new AggregateModifier with the Right op
    pub fn without() -> Self {
        AggregateModifier::Without(vec![])
    }

    /// Adds a label key to this AggregateModifier
    pub fn arg<S: Into<String>>(&mut self, arg: S) {
        match self {
            AggregateModifier::By(ref mut args) => {
                args.push(arg.into());
                args.sort();
            }
            AggregateModifier::Without(ref mut args) => {
                args.push(arg.into());
                args.sort();
            }
        }
    }

    pub fn get_args(&self) -> &Vec<String> {
        match self {
            AggregateModifier::By(val) => val,
            AggregateModifier::Without(val) => val,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.get_args().is_empty()
    }
}

impl Display for AggregateModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            AggregateModifier::By(vec) => {
                write!(f, "by ")?;
                write_comma_separated(vec.iter(), f, true)?;
            }
            AggregateModifier::Without(vec) => {
                write!(f, "without ")?;
                write_comma_separated(vec.iter(), f, true)?;
            }
        }
        Ok(())
    }
}

impl PartialEq<Self> for AggregateModifier {
    fn eq(&self, other: &AggregateModifier) -> bool {
        match (self, other) {
            (AggregateModifier::Without(left), AggregateModifier::Without(right)) => {
                string_vecs_equal_unordered(left, right)
            }
            (AggregateModifier::By(left), AggregateModifier::By(right)) => {
                string_vecs_equal_unordered(left, right)
            }
            _ => false,
        }
    }
}

// See https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize)]
pub enum GroupModifierOp {
    On,
    Ignoring,
}

impl Display for GroupModifierOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use GroupModifierOp::*;
        match self {
            On => write!(f, "on")?,
            Ignoring => write!(f, "ignoring")?,
        }
        Ok(())
    }
}

impl TryFrom<&str> for GroupModifierOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        use GroupModifierOp::*;

        match op {
            op if op.eq_ignore_ascii_case("on") => Ok(On),
            op if op.eq_ignore_ascii_case("ignoring") => Ok(Ignoring),
            _ => Err(ParseError::General(format!(
                "Unknown group_modifier op: {op}",
            ))),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum GroupType {
    GroupLeft,
    GroupRight,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum VectorMatchCardinality {
    OneToOne,
    /// on(labels)/ignoring(labels) GROUP_LEFT
    ManyToOne(Labels),
    /// on(labels)/ignoring(labels) GROUP_RIGHT
    OneToMany(Labels),
    /// logical/set binary operators
    ManyToMany,
}

impl VectorMatchCardinality {
    pub fn group_left(labels: Labels) -> Self {
        VectorMatchCardinality::ManyToOne(labels)
    }

    pub fn group_right(labels: Labels) -> Self {
        VectorMatchCardinality::OneToMany(labels)
    }
    pub fn is_group_left(&self) -> bool {
        matches!(self, VectorMatchCardinality::ManyToOne(_))
    }
    pub fn is_group_right(&self) -> bool {
        matches!(self, VectorMatchCardinality::OneToMany(_))
    }

    pub fn is_grouping(&self) -> bool {
        matches!(
            self,
            VectorMatchCardinality::ManyToOne(_) | VectorMatchCardinality::OneToMany(_)
        )
    }

    pub fn labels(&self) -> Option<&Labels> {
        match self {
            VectorMatchCardinality::ManyToOne(labels)
            | VectorMatchCardinality::OneToMany(labels) => Some(labels),
            _ => None,
        }
    }

    pub fn group_type(&self) -> Option<GroupType> {
        match self {
            VectorMatchCardinality::ManyToOne(_) => Some(GroupType::GroupLeft),
            VectorMatchCardinality::OneToMany(_) => Some(GroupType::GroupRight),
            _ => None,
        }
    }
}

impl Display for VectorMatchCardinality {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use VectorMatchCardinality::*;
        let str = match self {
            OneToOne => "OneToOne".to_string(),
            OneToMany(labels) => format!("group_right({:?})", labels),
            ManyToOne(labels) => format!("group_left({:?})", labels),
            ManyToMany => "ManyToMany".to_string(),
        };
        write!(f, "{}", str)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash, Serialize, Deserialize)]
pub enum JoinModifierOp {
    GroupLeft,
    GroupRight,
}

impl Display for JoinModifierOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use JoinModifierOp::*;
        match self {
            GroupLeft => write!(f, "group_left")?,
            GroupRight => write!(f, "group_right")?,
        }
        Ok(())
    }
}

impl TryFrom<&str> for JoinModifierOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        use JoinModifierOp::*;

        match op {
            op if op.eq_ignore_ascii_case("group_left") => Ok(GroupLeft),
            op if op.eq_ignore_ascii_case("group_right") => Ok(GroupRight),
            _ => {
                let msg = format!("Unknown join_modifier op: {}", op);
                Err(ParseError::General(msg))
            }
        }
    }
}

/// Expression Trait. Useful for cases where match is not ergonomic
pub trait ExpressionNode {
    fn cast(self) -> Expr;
}

/// NumberExpr represents number expression.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct NumberLiteral {
    /// value is the parsed number, i.e. `1.23`, `-234`, etc.
    pub value: f64,
}

impl NumberLiteral {
    pub fn new(v: f64) -> Self {
        NumberLiteral { value: v }
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl Value for NumberLiteral {
    fn value_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl From<f64> for NumberLiteral {
    fn from(value: f64) -> Self {
        NumberLiteral::new(value)
    }
}

impl From<i64> for NumberLiteral {
    fn from(value: i64) -> Self {
        NumberLiteral::new(value as f64)
    }
}

impl From<usize> for NumberLiteral {
    fn from(value: usize) -> Self {
        NumberLiteral::new(value as f64)
    }
}

impl PartialEq<NumberLiteral> for NumberLiteral {
    fn eq(&self, other: &Self) -> bool {
        are_floats_equal(self.value, other.value)
    }
}

impl PartialOrd<NumberLiteral> for NumberLiteral {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Eq for NumberLiteral {}

impl ExpressionNode for NumberLiteral {
    fn cast(self) -> Expr {
        Expr::NumberLiteral(self.clone())
    }
}

impl Display for NumberLiteral {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write_number(f, self.value)
    }
}

impl Prettier for NumberLiteral {
    fn needs_split(&self, _max: usize) -> bool {
        false
    }
}

impl Hash for NumberLiteral {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(state, self.value);
    }
}

impl Neg for NumberLiteral {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let value = -self.value;
        NumberLiteral { value }
    }
}

impl Deref for NumberLiteral {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StringLiteral(pub String);

impl StringLiteral {
    pub fn new<S: Into<String>>(s: S) -> Self {
        StringLiteral(s.into())
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl Display for StringLiteral {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Prettier for StringLiteral {
    fn needs_split(&self, _max: usize) -> bool {
        false
    }
}

impl Deref for StringLiteral {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<String> for StringLiteral {
    fn from(s: String) -> Self {
        StringLiteral(s)
    }
}

/// DurationExpr contains a duration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DurationExpr {
    /// duration value in milliseconds
    Millis(i64),
    /// a value that is multiplied at evaluation time by the step value
    StepValue(f64),
}

impl DurationExpr {
    pub fn from_secs(secs: i64) -> Self {
        Self::Millis(secs * 1000)
    }

    pub fn new(millis: i64) -> Self {
        Self::Millis(millis)
    }

    pub fn new_step(value: f64) -> Self {
        Self::StepValue(value)
    }

    pub fn requires_step(&self) -> bool {
        matches!(self, DurationExpr::StepValue(_))
    }

    /// Duration returns the duration from de in milliseconds.
    pub fn value(&self, step: Duration) -> i64 {
        match self {
            DurationExpr::Millis(v) => *v,
            DurationExpr::StepValue(v) => (*v * (step.as_millis() as f64)) as i64,
        }
    }

    pub fn as_duration(&self, step: Duration) -> Duration {
        match self {
            DurationExpr::Millis(v) => Duration::from_millis(*v as u64),
            DurationExpr::StepValue(v) => { 
                let millis = *v * step.as_millis() as f64;
                Duration::from_millis(millis as u64)
            },
        }
    }
    pub fn value_as_secs(&self, step: Duration) -> i64 {
        self.value(step) / 1000
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl Default for DurationExpr {
    fn default() -> Self {
        Self::Millis(0)
    }
}

impl Display for DurationExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            DurationExpr::Millis(v) => fmt_duration_ms(f, *v),
            DurationExpr::StepValue(v) => write!(f, "{v}i"),
        }
    }
}

impl TryFrom<NumberLiteral> for DurationExpr {
    type Error = String;

    fn try_from(num: NumberLiteral) -> Result<Self, Self::Error> {
        DurationExpr::try_from(num.value)
    }
}

impl TryFrom<Expr> for DurationExpr {
    type Error = String;

    fn try_from(ex: Expr) -> Result<Self, Self::Error> {
        match ex {
            Expr::NumberLiteral(nl) => DurationExpr::try_from(nl),
            _ => Err("invalid float value after @ modifier".into()),
        }
    }
}

impl TryFrom<f64> for DurationExpr {
    type Error = String;

    fn try_from(secs: f64) -> Result<Self, Self::Error> {
        let err_info = format!("timestamp out of bounds for @ modifier: {secs}");

        if secs.is_nan() || secs.is_infinite() || secs >= f64::MAX || secs <= f64::MIN {
            return Err(err_info);
        }
        let milli = (secs * 1000f64).round().abs() as u64;

        let duration = Duration::from_millis(milli);
        let mut st = Some(SystemTime::UNIX_EPOCH);
        if secs.is_sign_positive() {
            st = SystemTime::UNIX_EPOCH.checked_add(duration);
        }
        if secs.is_sign_negative() {
            st = SystemTime::UNIX_EPOCH.checked_sub(duration);
        }
        if let Some(st) = st {
            let millis = st
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64;
            return Ok(DurationExpr::Millis(millis));
        }
        Err(err_info)
    }
}

impl PartialEq<DurationExpr> for DurationExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DurationExpr::Millis(v1), DurationExpr::Millis(v2)) => v1 == v2,
            (DurationExpr::StepValue(v1), DurationExpr::StepValue(v2)) => {
                are_floats_equal(*v1, *v2)
            }
            _ => false,
        }
    }
}

impl Eq for DurationExpr {}

impl Prettier for DurationExpr {
    fn needs_split(&self, _max: usize) -> bool {
        false
    }
}

// todo: MetricExpr => VectorSelector
/// `MetricExpr` represents MetricsQL metric with optional filters, i.e. `foo{...}`.
///
/// Curly braces may contain or-delimited list of filters. For example:
///
/// `x{job="foo",instance="bar" or job="x",instance="baz"}`
///
/// In this case the filter returns all the series, which match at least one of the following filters:
///
/// `x{job="foo",instance="bar"}`
/// `x{job="x",instance="baz"}`
///
/// This allows using or-delimited list of filters inside rollup functions. For example,
/// the following query calculates rate per each matching series for the given or-delimited filters:
///
/// `rate(x{job="foo",instance="bar" or job="x",instance="baz"}[5m])`
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricExpr {
    /// matchers contains a list of label filters from curly braces.
    pub matchers: Matchers,
}

impl MetricExpr {
    pub fn new<S: Into<String>>(name: S) -> MetricExpr {
        // SAFETY: safe to unwrap since only regex errors are possible
        let name_filter = Matcher::new(MatchOp::Equal, NAME_LABEL, name.into()).unwrap();
        MetricExpr {
            matchers: Matchers::default().append(name_filter),
        }
    }

    pub fn with_filters(filters: Vec<Matcher>) -> Self {
        MetricExpr {
            matchers: Matchers::new(filters),
        }
    }

    pub fn with_or_filters(name: Option<String>, filters: Vec<Vec<Matcher>>) -> Self {
        MetricExpr {
            matchers: Matchers::with_or_matchers(name, filters),
        }
    }

    pub fn with_matchers(name: Option<String>, matchers: Vec<Matcher>) -> Self {
        MetricExpr {
            matchers: Matchers::with_matchers(name, matchers),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.matchers.is_empty()
    }

    pub fn is_only_metric_name(&self) -> bool {
        self.matchers.is_only_metric_name()
    }

    pub fn metric_name(&self) -> Option<&str> {
        self.matchers.metric_name()
    }

    pub fn append(mut self, filter: Matcher) -> Self {
        self.matchers = self.matchers.append(filter);
        self
    }

    pub fn append_or(mut self, filter: Matcher) -> Self {
        self.matchers = self.matchers.append_or(filter);
        self
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }

    pub fn is_empty_matchers(&self) -> bool {
        self.matchers.is_empty_matchers()
    }

    /// find all the matchers whose name equals the specified name.
    pub fn find_matchers(&self, name: &str) -> Vec<&Matcher> {
        self.matchers.find_matchers(name)
    }

    pub fn sort_filters(&mut self) {
        self.matchers.sort_filters();
    }

    pub fn has_or_matchers(&self) -> bool {
        !self.matchers.or_matchers.is_empty()
    }
}

impl Value for MetricExpr {
    fn value_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

impl Display for MetricExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.matchers)
    }
}

/// directly create an instant vector with only METRIC_NAME matcher.
///
/// # Examples
///
/// Basic usage:
///
/// ``` rust
/// use metricsql_parser::ast::MetricExpr;
/// use metricsql_parser::label::Matchers;
///
/// let vs = MetricExpr::new("foo");
///
/// assert_eq!(MetricExpr::from("foo"), vs);
/// ```
impl From<&str> for MetricExpr {
    fn from(name: &str) -> Self {
        MetricExpr::from(name.to_string())
    }
}

impl From<String> for MetricExpr {
    fn from(name: String) -> Self {
        MetricExpr::new(name)
    }
}

impl Neg for MetricExpr {
    type Output = UnaryExpr;

    fn neg(self) -> Self::Output {
        let ex = Expr::MetricExpression(self);
        UnaryExpr { expr: Box::new(ex) }
    }
}

impl ExpressionNode for MetricExpr {
    fn cast(self) -> Expr {
        Expr::MetricExpression(self)
    }
}

impl Prettier for MetricExpr {
    fn needs_split(&self, _max: usize) -> bool {
        false
    }
}

/// FuncExpr represents a MetricsQL function call such as `rate(...)`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionExpr {
    pub function: BuiltinFunction,

    /// Args contains function args.
    pub args: Vec<Expr>,

    /// If keep_metric_names is set to true, then the function should keep metric names.
    #[serde(default, skip_serializing_if = "is_default")]
    pub keep_metric_names: bool,
}

impl FunctionExpr {
    pub fn new(name: &str, args: Vec<Expr>) -> ParseResult<Self> {
        let func_name = if name.is_empty() { "union" } else { name };
        let function = BuiltinFunction::new(func_name)?;

        Ok(Self {
            args,
            keep_metric_names: false,
            function,
        })
    }

    pub fn name(&self) -> &'static str {
        self.function.name()
    }

    pub fn return_type(&self) -> ValueType {
        self.function
            .return_type(&self.args)
            .unwrap_or(ValueType::Scalar)
    }

    pub fn function_type(&self) -> BuiltinFunctionType {
        self.function.get_type()
    }

    pub fn arg_idx_for_optimization(&self) -> Option<usize> {
        self.function.get_arg_idx_for_optimization(self.args.len())
    }

    pub fn arg_for_optimization(&self) -> Option<&Expr> {
        self.arg_idx_for_optimization().and_then(|idx| self.args.get(idx))
    }

    pub fn default_rollup(arg: Expr) -> ParseResult<Self> {
        Self::from_single_arg("default_rollup", arg)
    }

    pub fn from_single_arg(name: &str, arg: Expr) -> ParseResult<Self> {
        let args = vec![arg];
        Self::new(name, args)
    }

    pub fn is_rollup(&self) -> bool {
        self.function_type() == BuiltinFunctionType::Rollup
    }

    pub fn is_rollup_function(&self, rf: RollupFunction) -> bool {
        matches!(self.function, BuiltinFunction::Rollup(r) if r == rf)
    }

    pub fn is_aggregate_function(&self, af: AggregateFunction) -> bool {
        matches!(self.function, BuiltinFunction::Aggregate(a) if a == af)
    }

    pub fn is_transform_function(&self, tf: TransformFunction) -> bool {
        matches!(self.function, BuiltinFunction::Transform(f) if f == tf)
    }
}

impl Display for FunctionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.function.name())?;
        write_comma_separated(&mut self.args.iter(), f, true)?;
        if self.keep_metric_names {
            write!(f, " keep_metric_names")?;
        }
        Ok(())
    }
}

impl Prettier for FunctionExpr {
    fn format(&self, level: usize, max: usize) -> String {
        let spaces = indent(level);
        format!(
            "{spaces}{}(\n{}\n{spaces}{})",
            self.name(),
            prettify_args(&self.args, level + 1, max),
            if self.keep_metric_names {
                " keep_metric_names"
            } else {
                ""
            }
        )
    }
}

/// `AggregationExpr` represents aggregate function such as `sum(...) by (...)`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregationExpr {
    /// function is the aggregation function.
    pub function: AggregateFunction,

    /// function args.
    pub args: Vec<Expr>,

    /// optional modifier such as `by (...)` or `without (...)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modifier: Option<AggregateModifier>,

    /// optional limit for the number of output time series.
    /// This is an MetricsQL extension.
    ///
    /// Example: `sum(...) by (...) limit 10` would return maximum 10 time series.
    #[serde(default, skip_serializing_if = "is_default")]
    pub limit: usize,

    #[serde(default, skip_serializing_if = "is_default")]
    pub keep_metric_names: bool,
}

impl AggregationExpr {
    pub fn new(function: AggregateFunction, args: Vec<Expr>) -> AggregationExpr {
        let mut ae = AggregationExpr {
            args,
            modifier: None,
            limit: 0,
            function,
            keep_metric_names: false,
        };

        ae.set_keep_metric_names();
        ae
    }

    /// the aggregation function name.
    pub fn name(&self) -> &'static str {
        self.function.name()
    }

    pub fn from_name(name: &str) -> ParseResult<Self> {
        let function = AggregateFunction::from_str(name)?;
        Ok(Self::new(function, vec![]))
    }

    pub fn with_modifier(mut self, modifier: AggregateModifier) -> Self {
        self.modifier = Some(modifier);
        self
    }

    pub fn with_args(mut self, args: &[Expr]) -> Self {
        self.args = args.to_vec();
        self.set_keep_metric_names();
        self
    }

    fn set_keep_metric_names(&mut self) {
        if self.args.len() != 1 {
            self.keep_metric_names = false;
            return;
        }
        match &self.args[0] {
            Expr::Function(fe) => {
                self.keep_metric_names = fe.keep_metric_names;
            }
            Expr::Aggregation(ae) => {
                self.keep_metric_names = ae.keep_metric_names;
            }
            _ => self.keep_metric_names = false,
        }
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }

    pub fn get_arg_for_optimization(&self) -> Option<&'_ Expr> {
        self.arg_idx_for_optimization().and_then(|idx| self.args.get(idx))
    }

    pub fn arg_idx_for_optimization(&self) -> Option<usize> {
        get_aggregate_arg_idx_for_optimization(self.function, self.args.len())
    }

    /// Check if args[0] contains one of the following:
    /// - metricExpr
    /// - metricExpr[d]
    /// - RollupFunc(metricExpr)
    /// - RollupFunc(metricExpr[d])
    pub fn can_incrementally_eval(&self) -> bool {
        if self.args.len() != 1 {
            return false;
        }

        fn validate(me: &MetricExpr, for_subquery: bool) -> bool {
            if me.is_empty() || for_subquery {
                return false;
            }

            true
        }

        fn validate_expr(expr: &Expr) -> bool {
            match expr {
                Expr::MetricExpression(me) => validate(me, false),
                Expr::Rollup(re) => match &*re.expr {
                    Expr::MetricExpression(me) => validate(me, re.for_subquery()),
                    _ => false,
                },
                _ => false,
            }
        }

        let first = &self.args[0];
        match first {
            Expr::Function(fe) => match fe.function {
                BuiltinFunction::Rollup(_) => {
                    if let Some(arg) = fe.arg_for_optimization() {
                        validate_expr(arg)
                    } else {
                        false
                    }
                }
                _ => false,
            },
            _ => validate_expr(first),
        }
    }

    pub fn is_non_grouping(&self) -> bool {
        match &self.modifier {
            Some(modifier) => modifier.is_empty(),
            _ => true,
        }
    }

    fn get_op_string(&self) -> String {
        let mut s = self.function.to_string();

        if let Some(modifier) = &self.modifier {
            match modifier {
                AggregateModifier::By(ls) if !ls.is_empty() => write!(s, " {modifier} ").unwrap(),
                AggregateModifier::Without(_) => write!(s, " {modifier} ").unwrap(),
                _ => (),
            }
        }
        s
    }
}

impl Display for AggregationExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.function)?;
        let args_len = self.args.len();
        if args_len > 0 {
            write_comma_separated(&mut self.args.iter(), f, true)?;
        }
        if let Some(modifier) = &self.modifier {
            write!(f, " {}", modifier)?;
        }
        if self.limit > 0 {
            write!(f, " limit {}", self.limit)?;
        }
        Ok(())
    }
}

impl Prettier for AggregationExpr {
    fn format(&self, level: usize, max: usize) -> String {
        let spaces = indent(level);
        let mut s = format!("{spaces}{}(\n", self.get_op_string());
        let args = prettify_args(&self.args, level + 1, max);
        if !args.is_empty() {
            writeln!(s, "{}", args).unwrap();
        }
        write!(s, "{spaces})").unwrap();
        if self.limit > 0 {
            write!(s, " limit {}", self.limit).unwrap();
        }
        if self.keep_metric_names {
            write!(s, " keep_metric_names").unwrap();
        }
        s
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// `RollupExpr` represents an MetricsQL expression which contains at least `offset` or `[...]` part.
pub struct RollupExpr {
    /// The expression for the rollup. Usually it is MetricExpr, but may be arbitrary expr
    /// if subquery is used. https://prometheus.io/blog/2019/01/28/subquery-support/
    pub expr: BExpression,

    /// window contains optional window value from square brackets. Equivalent to `range` in
    /// prometheus terminology
    ///
    /// For example, `http_requests_total[5m]` will have window value `5m`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<DurationExpr>,

    /// step contains optional step value from square brackets. Equivalent to `resolution`
    /// in the prometheus docs
    ///
    /// For example, `foobar[1h:3m]` will have step value `3m`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<DurationExpr>,

    /// offset contains optional value from `offset` part.
    ///
    /// For example, `foobar{baz="aa"} offset 5m` will have offset value `5m`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<DurationExpr>,

    /// if set to true, then `foo[1h:]` would print the same instead of `foo[1h]`.
    #[serde(default, skip_serializing_if = "is_default")]
    pub inherit_step: bool,

    /// at contains an optional expression after `@` modifier.
    ///
    /// For example, `foo @ end()` or `bar[5m] @ 12345`
    /// See https://prometheus.io/docs/prometheus/latest/querying/basics/#modifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at: Option<BExpression>,
}

impl RollupExpr {
    pub fn new(expr: Expr) -> Self {
        RollupExpr {
            expr: Box::new(expr),
            window: None,
            offset: None,
            step: None,
            inherit_step: false,
            at: None,
        }
    }

    pub fn for_subquery(&self) -> bool {
        self.step.is_some() || self.inherit_step
    }

    pub fn set_at(mut self, expr: impl ExpressionNode) -> Self {
        self.at = Some(Box::new(expr.cast()));
        self
    }

    pub fn set_offset(&mut self, expr: DurationExpr) {
        self.offset = Some(expr);
    }

    pub fn set_window(mut self, expr: DurationExpr) -> ParseResult<Self> {
        self.window = Some(expr);
        self.validate().map_err(ParseError::General)?;
        Ok(self)
    }

    pub fn set_expr(&mut self, expr: impl ExpressionNode) -> ParseResult<()> {
        self.expr = Box::new(expr.cast());
        self.validate().map_err(ParseError::General)
    }

    fn validate(&self) -> Result<(), String> {
        // range + subquery is not allowed (however this is syntactically invalid)
        // if self.window.is_some() && self.for_subquery() {
        //     return Err(
        //         "range and subquery are not allowed together in a rollup expression".to_string(),
        //     );
        // }
        Ok(())
    }

    pub fn return_type(&self) -> ValueType {
        // sub queries turn instant vectors into ranges
        match (self.window.is_some(), self.for_subquery()) {
            (false, false) => ValueType::InstantVector,
            (false, true) => ValueType::RangeVector,
            (true, false) => ValueType::RangeVector,
            (true, true) => {
                ValueType::RangeVector
                // unreachable!("range and subquery are not allowed together in a rollup expression")
            }
        }
    }

    pub fn wraps_metric_expr(&self) -> bool {
        matches!(*self.expr, Expr::MetricExpression(_))
    }

    fn get_time_suffix_string(&self) -> Result<String, fmt::Error> {
        let mut s = String::with_capacity(12);
        if self.window.is_some() || self.inherit_step || self.step.is_some() {
            s.push('[');
            if let Some(win) = &self.window {
                write!(s, "{}", win)?;
            }
            if let Some(step) = &self.step {
                s.push(':');
                write!(s, "{}", step)?;
            } else if self.inherit_step {
                s.push(':');
            }
            s.push(']');
        }
        if let Some(offset) = &self.offset {
            write!(s, " offset {}", offset)?;
        }
        if let Some(at) = &self.at {
            let parens_needed = at.is_binary_op();
            s.push_str(" @ ");
            if parens_needed {
                s.push('(');
            }
            write!(s, "{}", at)?;
            if parens_needed {
                s.push(')');
            }
        }
        Ok(s)
    }
}

impl Display for RollupExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let need_parens = match self.expr.as_ref() {
            Expr::Rollup(_) => true,
            Expr::BinaryOperator(_) => true,
            Expr::Aggregation(ae) => ae.modifier.is_some(),
            _ => false,
        };
        if need_parens {
            write!(f, "(")?;
        }
        write!(f, "{}", self.expr)?;
        if need_parens {
            write!(f, ")")?;
        }
        let suffix = self.get_time_suffix_string().map_err(|_| fmt::Error)?;

        write!(f, "{}", suffix)?;
        Ok(())
    }
}

impl Prettier for RollupExpr {
    fn pretty(&self, level: usize, max: usize) -> String {
        let suffix = self.get_time_suffix_string().unwrap_or("".to_string());
        format!("{}{}", self.expr.pretty(level, max), suffix)
    }
}

/// BinaryOpExpr represents a binary operation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BinaryExpr {
    /// `left` contains left arg for the `left op right` expression.
    pub left: BExpression,

    /// contains right arg for the `left op right` expression.
    pub right: BExpression,

    /// `op` is the operation itself, i.e. `+`, `-`, `*`, etc.
    pub op: Operator,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub modifier: Option<BinModifier>,
}

impl BinaryExpr {
    pub fn new(op: Operator, lhs: Expr, rhs: Expr) -> Self {
        // operators can only have instant vectors or scalars
        // TODO
        BinaryExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            modifier: None,
        }
    }

    pub fn is_matching_on(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.is_matching_on())
    }

    pub fn is_matching_labels_not_empty(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.is_matching_labels_not_empty())
    }

    /// check if labels of card and matching are joint
    pub fn is_labels_joint(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.is_labels_joint())
    }

    /// intersect labels of card and matching
    pub fn intersect_labels(&self) -> Option<Vec<String>> {
        self.modifier
            .as_ref()
            .and_then(|modifier| modifier.intersect_labels())
    }

    pub fn with_bool_modifier(mut self) -> Self {
        if !self.op.is_comparison() {
            panic!("bool modifier is only allowed for comparison operators");
        }
        if let Some(modifier) = &mut self.modifier {
            modifier.return_bool = true;
        } else {
            let modifier = BinModifier {
                return_bool: true,
                ..Default::default()
            };
            self.modifier = Some(modifier);
        }
        self
    }

    pub fn set_keep_metric_names(&mut self) {
        if let Some(modifier) = &mut self.modifier {
            modifier.keep_metric_names = true;
        } else {
            let modifier = BinModifier {
                keep_metric_names: true,
                ..Default::default()
            };
            self.modifier = Some(modifier);
        }
    }

    /// Convert `num cmpOp query` expression to `query reverseCmpOp num` expression
    /// like Prometheus does. For instance, `0.5 < foo` must be converted to `foo > 0.5`
    /// in order to return valid values for `foo` that are bigger than 0.5.
    pub fn adjust_comparison_op(&mut self) -> bool {
        if self.should_adjust_comparison_op() {
            self.op = self.op.get_reverse_cmp();
            std::mem::swap(&mut self.left, &mut self.right);
            return true;
        }
        false
    }

    pub fn should_adjust_comparison_op(&self) -> bool {
        if !self.op.is_comparison() {
            return false;
        }

        if Expr::is_number(&self.right) || !Expr::is_scalar(&self.left) {
            return false;
        }
        true
    }

    pub fn should_reset_metric_group(&self) -> bool {
        if matches!(&self.modifier, Some(modifier) if modifier.keep_metric_names) {
            return false;
        }
        let op = self.op;
        if op.is_comparison() && !self.returns_bool() {
            // do not reset MetricGroup for non-boolean `compare` binary ops like Prometheus does.
            return false;
        }
        !matches!(op, Operator::Default | Operator::If | Operator::IfNot)
    }

    pub fn return_type(&self) -> ValueType {
        let lhs_ret = self.left.return_type();
        let rhs_ret = self.right.return_type();

        match (lhs_ret, rhs_ret) {
            (ValueType::Scalar, ValueType::Scalar) => ValueType::Scalar,
            (ValueType::RangeVector, ValueType::RangeVector) => ValueType::RangeVector,
            (ValueType::InstantVector, ValueType::InstantVector) => ValueType::InstantVector,
            (ValueType::InstantVector, ValueType::Scalar) => ValueType::InstantVector,
            (ValueType::Scalar, ValueType::InstantVector) => ValueType::InstantVector,
            (ValueType::String, ValueType::String) => {
                if self.op.is_comparison() {
                    return ValueType::Scalar;
                }
                debug_assert!(
                    self.op == Operator::Add,
                    "Operator {} is not valid for (String, String)",
                    self.op
                );
                ValueType::String
            }
            _ => ValueType::InstantVector,
        }
    }

    /// indicates whether `bool` modifier is present.
    /// For example, `foo > bool bar`.
    pub fn returns_bool(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.return_bool)
    }

    /// Determines if the result of the operation should keep metric names.
    pub fn keep_metric_names(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.keep_metric_names)
    }

    pub fn vector_match_cardinality(&self) -> Option<&VectorMatchCardinality> {
        if let Some(modifier) = self.modifier.as_ref() {
            return Some(&modifier.card);
        }
        None
    }

    fn fmt_no_keep_metric_name(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{} {} {}",
            self.left,
            self.get_op_matching_string(),
            self.right
        )
    }

    fn get_op_matching_string(&self) -> String {
        match &self.modifier {
            Some(modifier) => format!("{} {modifier}", self.op),
            None => self.op.to_string(),
        }
    }
}

impl Display for BinaryExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.keep_metric_names() {
            write!(f, "(")?;
            self.fmt_no_keep_metric_name(f)?;
            write!(f, ") keep_metric_names")?;
        } else {
            self.fmt_no_keep_metric_name(f)?;
        }
        Ok(())
    }
}

impl Prettier for BinaryExpr {
    fn format(&self, level: usize, max: usize) -> String {
        format!(
            "{}\n{}{}{}\n{}",
            self.left.pretty(level + 1, max),
            indent(level),
            self.get_op_matching_string(),
            if self.keep_metric_names() {
                "\n keep_metric_names"
            } else {
                ""
            },
            self.right.pretty(level + 1, max)
        )
    }
}

/// UnaryExpr will negate the expr
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnaryExpr {
    pub expr: Box<Expr>,
}

impl UnaryExpr {
    pub fn new(expr: Expr) -> Self {
        UnaryExpr {
            expr: Box::new(expr),
        }
    }
}

impl Display for UnaryExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "-{}", self.expr)
    }
}

impl Prettier for UnaryExpr {
    fn pretty(&self, level: usize, max: usize) -> String {
        format!(
            "{}-{}",
            indent(level),
            self.expr.pretty(level, max).trim_start()
        )
    }
}

// TODO: ParensExpr => GroupExpr
/// Expression(s) explicitly grouped in parens
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParensExpr {
    pub expressions: Vec<Expr>,
}

impl ParensExpr {
    pub fn new(expressions: Vec<Expr>) -> Self {
        ParensExpr { expressions }
    }

    pub fn len(&self) -> usize {
        self.expressions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.expressions.is_empty()
    }

    pub fn return_type(&self) -> ValueType {
        if let Some(inner) = self.innermost_expr() {
            return inner.return_type();
        }
        if self.len() == 1 {
            return self.expressions[0].return_type();
        }

        // Treat as a function with empty name, i.e. union()
        TransformFunction::Union.return_type()
    }

    pub fn to_function(self) -> FunctionExpr {
        // Treat parensExpr as a function with empty name, i.e. union()
        let name = "union";
        let func = BuiltinFunction::from_str(name).unwrap(); // if union is not defined, we have a fatal issue

        FunctionExpr {
            args: self.expressions,
            keep_metric_names: false,
            function: func,
        }
    }

    /// Return the innermost expression wrapped by a `ParensExpr` if the `ParensExpr` contains
    /// exactly one expression. For example : (((x + y))) would return a ref to `x + y`
    pub fn innermost_expr(&self) -> Option<&Expr> {
        match self.len() {
            0 => None,
            1 => match &self.expressions[0] {
                Expr::Parens(pe2) => pe2.innermost_expr(),
                expr => Some(expr),
            },
            _ => None,
        }
    }
}

impl Value for ParensExpr {
    fn value_type(&self) -> ValueType {
        if let Some(inner) = self.innermost_expr() {
            return inner.return_type();
        }
        // Treat as a function with empty name, i.e. union()
        TransformFunction::Union.return_type()
    }
}

impl Display for ParensExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write_comma_separated(&mut self.expressions.iter(), f, true)?;
        Ok(())
    }
}

impl Prettier for ParensExpr {
    fn format(&self, level: usize, max: usize) -> String {
        let mut s = String::with_capacity(64);
        let sub_indent = indent(level + 1);

        for (i, expr) in self.expressions.iter().enumerate() {
            if i > 0 {
                s.push_str(",\n");
            }
            s.push_str(&sub_indent);
            s.push_str(&expr.pretty(level + 1, max));
        }

        format!("{}(\n{s}\n{})", indent(level), indent(level))
    }
}

impl ExpressionNode for ParensExpr {
    fn cast(self) -> Expr {
        Expr::Parens(self)
    }
}

/// A root expression node.
///
/// These are all valid root expression ast.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    /// A single scalar number.
    NumberLiteral(NumberLiteral),

    Duration(DurationExpr),

    /// A single scalar string.
    ///
    /// Prometheus' docs claim strings aren't currently implemented, but they're
    /// valid as function arguments.
    StringLiteral(StringLiteral),

    /// A function call
    Function(FunctionExpr),

    /// Aggregation represents aggregate functions such as `sum(...) by (...)`
    Aggregation(AggregationExpr),

    /// A unary operator expression
    UnaryOperator(UnaryExpr),

    /// A binary operator expression
    BinaryOperator(BinaryExpr),

    /// RollupExpr represents an MetricsQL expression which contains at least `offset` or `[...]` part.
    Rollup(RollupExpr),

    /// MetricExpr represents a MetricsQL metric with optional filters, i.e. `foo{...}`.
    MetricExpression(MetricExpr),

    /// A grouped expression wrapped in parentheses
    Parens(ParensExpr),

    /// String expression parsed in the context of a `with` statement.
    ///
    /// Prometheus' docs claim strings aren't currently implemented, but they're
    /// valid as function arguments.
    StringExpr(StringExpr),
}

pub type BExpression = Box<Expr>;

impl Expr {
    pub fn is_scalar(expr: &Expr) -> bool {
        match expr {
            Expr::Duration(_) | Expr::NumberLiteral(_) => true,
            Expr::Function(f) => f.function.is_scalar(),
            _ => false,
        }
    }

    pub fn is_number(expr: &Expr) -> bool {
        matches!(expr, Expr::NumberLiteral(_))
    }

    pub fn is_string(expr: &Expr) -> bool {
        matches!(expr, Expr::StringLiteral(_))
    }

    pub fn is_primitive(expr: &Expr) -> bool {
        matches!(expr, Expr::NumberLiteral(_) | Expr::StringLiteral(_))
    }

    pub fn is_duration(expr: &Expr) -> bool {
        matches!(expr, Expr::Duration(_))
    }

    pub fn vectors(&self) -> Box<dyn Iterator<Item = &Matcher> + '_> {
        match self {
            Self::MetricExpression(v) => Box::new(v.matchers.filter_iter()),
            Self::Rollup(re) => Box::new(re.expr.vectors().chain(if let Some(at) = &re.at {
                at.vectors()
            } else {
                Box::new(iter::empty())
            })),
            Self::UnaryOperator(u) => u.expr.vectors(),
            Self::BinaryOperator(be) => Box::new(be.left.vectors().chain(be.right.vectors())),
            Self::Aggregation(ae) => Box::new(ae.args.iter().flat_map(|node| node.vectors())),
            Self::Function(fe) => Box::new(fe.args.iter().flat_map(|node| node.vectors())),
            Self::Parens(pe) => Box::new(pe.expressions.iter().flat_map(|node| node.vectors())),
            Self::NumberLiteral(_)
            | Self::Duration(_)
            | Self::StringLiteral(_)
            | Self::StringExpr(_) => Box::new(iter::empty()),
        }
    }

    /**
    Return an iterator of series names present in this node.
    ``` rust
    let query = r#"
        sum(1 - something_used{env="production"} / something_total) by (instance)
        and ignoring (instance)
        sum(rate(some_queries{instance=~"localhost\\d+"} [5m])) > 100
    "#;
    let ast = metricsql_parser::parser::parse(query).expect("valid query");
    let series: Vec<String> = ast.series_names().collect();
    assert_eq!(series, vec![
            "something_used".to_string(),
            "something_total".to_string(),
            "some_queries".to_string(),
        ],
    );
    ```
     */
    pub fn series_names(&self) -> impl Iterator<Item = String> + '_ {
        self.vectors().map(|x| {
            if x.label == NAME_LABEL {
                x.value.clone()
                // String::from_utf8(x.value.clone())
                //     .expect("series names should always be valid utf8")
            } else {
                x.label.clone()
            }
        })
    }

    pub fn contains_subquery(&self) -> bool {
        use Expr::*;
        match self {
            Function(fe) => fe.args.iter().any(|e| e.contains_subquery()),
            BinaryOperator(bo) => bo.left.contains_subquery() || bo.right.contains_subquery(),
            Aggregation(aggr) => aggr.args.iter().any(|e| e.contains_subquery()),
            Rollup(re) => re.for_subquery(),
            _ => false,
        }
    }

    pub fn return_type(&self) -> ValueType {
        match self {
            Expr::NumberLiteral(_) => ValueType::Scalar,
            Expr::Duration(dur) => dur.return_type(),
            Expr::StringLiteral(_) | Expr::StringExpr(_) => ValueType::String,
            Expr::Function(fe) => fe.return_type(),
            Expr::Aggregation(ae) => ae.return_type(),
            Expr::UnaryOperator(u) => u.expr.return_type(),
            Expr::BinaryOperator(be) => be.return_type(),
            Expr::Rollup(re) => re.return_type(),
            Expr::Parens(me) => me.return_type(),
            Expr::MetricExpression(me) => me.return_type(),
        }
    }

    pub fn variant_name(&self) -> &'static str {
        match self {
            Expr::NumberLiteral(_) => "Scalar",
            Expr::Duration(_) => "Duration",
            Expr::StringLiteral(_) | Expr::StringExpr(_) => "String",
            Expr::Function(_) => "Function",
            Expr::Aggregation(_) => "Aggregation",
            Expr::UnaryOperator(_) => "UnaryOperator",
            Expr::BinaryOperator(_) => "BinaryOperator",
            Expr::Rollup(_) => "Rollup",
            Expr::Parens(_) => "Parens",
            Expr::MetricExpression(_) => "VectorSelector",
        }
    }

    pub fn cast(self) -> Expr {
        // this code seems suspicious
        match self {
            Expr::Aggregation(a) => Expr::Aggregation(a),
            Expr::UnaryOperator(u) => Expr::UnaryOperator(u),
            Expr::BinaryOperator(b) => Expr::BinaryOperator(b),
            Expr::Duration(d) => Expr::Duration(d),
            Expr::Function(f) => Expr::Function(f),
            Expr::NumberLiteral(n) => Expr::NumberLiteral(n),
            Expr::MetricExpression(m) => Expr::MetricExpression(m),
            Expr::Parens(m) => Expr::Parens(m),
            Expr::Rollup(r) => Expr::Rollup(r),
            Expr::StringLiteral(s) => Expr::StringLiteral(s),
            Expr::StringExpr(s) => Expr::StringExpr(s),
        }
    }

    pub fn is_metric_expression(&self) -> bool {
        matches!(self, Expr::MetricExpression(_))
    }

    pub fn is_binary_op(&self) -> bool {
        matches!(self, Expr::BinaryOperator(_))
    }

    pub fn is_rollup(&self) -> bool {
        matches!(self, Expr::Rollup(_))
    }

    /// returns a scalar expression
    pub fn scalar(value: f64) -> Expr {
        Expr::from(value)
    }

    /// returns a string literal expression
    pub fn string_literal(value: &str) -> Expr {
        Expr::from(value)
    }

    /// Return `self == other`
    pub fn eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Eql, other)
    }

    /// Return `self != other`
    pub fn not_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::NotEq, other)
    }

    /// Return `self > other`
    pub fn gt(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Gt, other)
    }

    /// Return `self >= other`
    pub fn gt_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Gte, other)
    }

    /// Return `self < other`
    pub fn lt(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Lt, other)
    }

    /// Return `self <= other`
    pub fn lt_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Lte, other)
    }

    /// Return `self AND other`
    pub fn and(self, other: Expr) -> Expr {
        binary_expr(self, Operator::And, other)
    }

    /// Return `self OR other`
    pub fn or(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Or, other)
    }

    /// Calculate the modulus of two expressions.
    /// Return `self % other`
    pub fn modulus(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Mod, other)
    }

    pub fn call(func: &str, args: Vec<Expr>) -> ParseResult<Expr> {
        let expr = FunctionExpr::new(func, args)?;
        Ok(Expr::Function(expr))
    }

    pub fn new_vector_selector<S: Into<String>>(name: Option<S>, matchers: Matchers) -> Expr {
        let mut selector = if let Some(name) = name {
            MetricExpr::new(name)
        } else {
            MetricExpr::default()
        };
        selector.matchers = matchers;
        Expr::MetricExpression(selector)
    }

    pub fn new_binary_expr(
        lhs: Expr,
        op: Operator,
        modifier: Option<BinModifier>,
        rhs: Expr,
    ) -> Result<Expr, String> {
        let ex = BinaryExpr {
            left: Box::new(lhs),
            right: Box::new(rhs),
            op,
            modifier,
        };
        Ok(Expr::BinaryOperator(ex))
    }

    pub fn at_expr(self, at: Expr) -> Result<Self, String> {
        let already_set_err = Err("@ <timestamp> may not be set multiple times".into());
        match self {
            Expr::Rollup(mut s) => match s.at {
                None => {
                    s.at = Some(Box::new(at));
                    Ok(Expr::Rollup(s))
                }
                Some(_) => already_set_err,
            },
            _ => {
                Err("@ modifier must be preceded by an vector selector or matrix selector or a subquery".into())
            }
        }
    }

    pub fn prettify(&self) -> String {
        self.pretty(0, MAX_CHARACTERS_PER_LINE)
    }

    pub fn keep_metric_names(&self) -> bool {
        match self {
            Expr::UnaryOperator(ue) => ue.expr.keep_metric_names(),
            Expr::BinaryOperator(be) => be.keep_metric_names(),
            Expr::Rollup(re) => re.wraps_metric_expr(),
            Expr::Function(fe) => fe.keep_metric_names,
            Expr::Aggregation(ae) => ae.keep_metric_names,
            Expr::Parens(pe) => {
                if let Some(expr) = pe.innermost_expr() {
                    return expr.keep_metric_names();
                }
                false
            }
            _ => false,
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Expr::Aggregation(a) => write!(f, "{}", a)?,
            Expr::UnaryOperator(ue) => write!(f, "{}", ue)?,
            Expr::BinaryOperator(be) => write!(f, "{}", be)?,
            Expr::Duration(d) => write!(f, "{}", d)?,
            Expr::Function(func) => write!(f, "{}", func)?,
            Expr::NumberLiteral(n) => write!(f, "{}", n)?,
            Expr::MetricExpression(me) => write!(f, "{}", me)?,
            Expr::Parens(p) => write!(f, "{}", p)?,
            Expr::Rollup(re) => write!(f, "{}", re)?,
            Expr::StringLiteral(s) => write!(f, "{}", enquote('"', s))?,
            Expr::StringExpr(s) => write!(f, "{}", s)?,
        }
        Ok(())
    }
}

impl Prettier for Expr {
    fn pretty(&self, level: usize, max: usize) -> String {
        match self {
            Expr::Aggregation(ex) => ex.pretty(level, max),
            Expr::UnaryOperator(ex) => ex.pretty(level, max),
            Expr::BinaryOperator(ex) => ex.pretty(level, max),
            Expr::Parens(ex) => ex.pretty(level, max),
            Expr::Rollup(ex) => ex.pretty(level, max),
            Expr::NumberLiteral(ex) => ex.pretty(level, max),
            Expr::StringLiteral(ex) => ex.pretty(level, max),
            Expr::MetricExpression(ex) => ex.pretty(level, max),
            Expr::Function(ex) => ex.pretty(level, max),
            Expr::Duration(d) => d.pretty(level, max),
            Expr::StringExpr(se) => se.pretty(level, max),
        }
    }
}

impl Value for Expr {
    fn value_type(&self) -> ValueType {
        match self {
            Expr::Aggregation(a) => a.return_type(),
            Expr::UnaryOperator(ue) => ue.expr.return_type(),
            Expr::BinaryOperator(be) => be.return_type(),
            Expr::Duration(d) => d.return_type(),
            Expr::Function(func) => func.return_type(),
            Expr::NumberLiteral(n) => n.return_type(),
            Expr::MetricExpression(me) => me.value_type(),
            Expr::Parens(p) => p.return_type(),
            Expr::Rollup(re) => re.return_type(),
            Expr::StringLiteral(_) => ValueType::String,
            Expr::StringExpr(_) => ValueType::String,
        }
    }
}

// crate private
impl Default for Expr {
    fn default() -> Self {
        Expr::from(1.0)
    }
}

impl From<f64> for Expr {
    fn from(v: f64) -> Self {
        Expr::NumberLiteral(NumberLiteral::new(v))
    }
}

impl From<i64> for Expr {
    fn from(v: i64) -> Self {
        Self::from(v as f64)
    }
}

impl From<String> for Expr {
    fn from(s: String) -> Self {
        Expr::StringLiteral(StringLiteral(s))
    }
}

impl From<&str> for Expr {
    fn from(s: &str) -> Self {
        Expr::StringLiteral(StringLiteral(s.to_string()))
    }
}

impl From<MetricExpr> for Expr {
    fn from(vs: MetricExpr) -> Self {
        Expr::MetricExpression(vs)
    }
}

pub(crate) fn binary_expr(left: Expr, op: Operator, right: Expr) -> Expr {
    let mut expr = BinaryExpr::new(op, left, right);
    if op.is_comparison() {
        expr = expr.with_bool_modifier();
    }
    Expr::BinaryOperator(expr)
}

impl ops::Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Add, rhs)
    }
}

impl ops::Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Sub, rhs)
    }
}

impl ops::Mul for Expr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Mul, rhs)
    }
}

impl ops::Div for Expr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Div, rhs)
    }
}

impl ops::Rem for Expr {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Mod, rhs)
    }
}

impl ops::BitAnd for Expr {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        binary_expr(self, Operator::And, rhs)
    }
}

impl ops::BitOr for Expr {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Or, rhs)
    }
}

impl ops::BitXor for Expr {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Pow, rhs)
    }
}

impl Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Expr::NumberLiteral(nl) => Expr::NumberLiteral(-nl),
            _ => Expr::UnaryOperator(UnaryExpr {
                expr: Box::new(self),
            }),
        }
    }
}

fn are_floats_equal(left: f64, right: f64) -> bool {
    // Special handling for nan == nan.
    left == right || left.is_nan() && right.is_nan()
}

fn is_default<T: Default + PartialEq>(t: &T) -> bool {
    t == &T::default()
}
