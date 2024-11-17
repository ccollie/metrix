use metricsql_parser::ast::DurationExpr;
use metricsql_parser::common::{Value, ValueType};
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Serialize, Serializer};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use std::time::Duration;
use crate::common::format::format_number;
use crate::execution::{eval_number, EvalConfig};
use crate::functions::types::get_single_timeseries;
use crate::{RuntimeError, RuntimeResult};
use crate::types::{Label, Timeseries, Timestamp};

pub type Labels = Vec<Label>;


#[derive(Debug, Default, Clone, Copy)]
pub struct Sample {
    /// Time in microseconds
    pub timestamp: Timestamp,
    pub value: f64,
}

impl Serialize for Sample {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(2))?;
        seq.serialize_element(&(self.timestamp / 1_000_000))?;
        seq.serialize_element(&self.value.to_string())?;
        seq.end()
    }
}

impl Sample {
    pub fn new(timestamp: Timestamp, value: f64) -> Self {
        Self { timestamp, value }
    }
}

impl PartialEq for Sample {
    #[inline]
    fn eq(&self, other: &Sample) -> bool {
        // Two data points are equal if their times are equal, and their values are either equal or are NaN.
        if self.timestamp == other.timestamp {
            return if self.value.is_nan() {
                other.value.is_nan()
            } else {
                self.value == other.value
            }
        }
        false
    }
}

impl Eq for Sample {}

impl Ord for Sample {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

impl PartialOrd for Sample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


pub type InstantVector = Vec<Timeseries>; // todo: Vec<(label, Sample)> or somesuch

#[derive(Debug, Clone)]
pub struct RangeValue {
    pub labels: Labels,
    pub samples: Vec<Sample>,
    //pub time_window: Option<TimeWindow>,
}

impl Serialize for RangeValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_struct("range_value", 2)?;
        let labels_map = self
            .labels
            .iter()
            .map(|l| (l.name.as_str(), l.value.as_str()))
            .collect::<HashMap<_, _>>();
        seq.serialize_field("metric", &labels_map)?;
        seq.serialize_field("values", &self.samples)?;
        seq.end()
    }
}

impl RangeValue {
    pub fn with_capacity(labels: Labels, capacity: usize) -> Self {
        Self {
            labels,
            samples: Vec::with_capacity(capacity),
            //time_window: None,
        }
    }

    pub fn new<S>(labels: Labels, samples: S) -> Self
    where
        S: IntoIterator<Item = Sample>,
    {
        Self {
            labels,
            samples: Vec::from_iter(samples),
            //time_window: None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum QueryValue {
    RangeVector(Vec<Timeseries>),
    InstantVector(InstantVector),
    Scalar(f64),
    String(String),
}

impl QueryValue {
    pub fn nan() -> Self {
        QueryValue::Scalar(f64::NAN)
    }
    pub fn is_numeric(&self) -> bool {
        matches!(self, QueryValue::Scalar(_))
    }
    pub fn data_type(&self) -> ValueType {
        match &self {
            QueryValue::RangeVector(_) => ValueType::RangeVector,
            QueryValue::Scalar(_) => ValueType::Scalar,
            QueryValue::String(_) => ValueType::String,
            QueryValue::InstantVector(_) => ValueType::InstantVector,
        }
    }
    pub fn from_duration_ms(dur: &DurationExpr, step: i64) -> Self {
        let d = dur.value(Duration::from_millis(step as u64));
        let d_sec = d as f64 / 1000_f64;
        QueryValue::Scalar(d_sec)
    }

    pub fn data_type_name(&self) -> &'static str {
        match &self {
            QueryValue::RangeVector(_) => "RangeVector",
            QueryValue::Scalar(_) => "Scalar",
            QueryValue::String(_) => "String",
            QueryValue::InstantVector(_) => "InstantVector",
        }
    }

    pub fn get_matrix(&self) -> &Vec<Timeseries> {
        match self {
            QueryValue::RangeVector(val) => val,
            _ => panic!("BUG: range selection value expected "),
        }
    }

    pub fn get_scalar(&self) -> RuntimeResult<f64> {
        match self {
            QueryValue::Scalar(val) => Ok(*val),
            QueryValue::InstantVector(series) => {
                let ts = get_single_timeseries(series)?;
                if ts.values.is_empty() {
                    let msg = "value::get_scalar - empty vector".to_string();
                    return Err(RuntimeError::ArgumentError(msg));
                }
                Ok(ts.values[0])
            }
            _ => Err(RuntimeError::TypeCastError(format!(
                "{} cannot be converted to a scalar",
                self.data_type()
            ))),
        }
    }

    pub fn get_int(&self) -> RuntimeResult<i64> {
        match self {
            QueryValue::Scalar(val) => Ok(*val as i64),
            _ => match self.get_scalar() {
                Ok(val) => Ok(val as i64),
                Err(e) => Err(e),
            },
        }
    }

    pub fn get_vector(&self) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            QueryValue::InstantVector(val) => {
                // TODO: into
                Ok(val.clone())
            } // ????
            QueryValue::RangeVector(val) => {
                // TODO: into
                Ok(val.clone())
            } // ????
            _ => Err(RuntimeError::InvalidNumber(
                "vector parameter expected ".to_string(),
            )),
        }
    }

    // todo: get_series_into()/ COW
    pub fn get_instant_vector(&self, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            QueryValue::InstantVector(val) => Ok(val.clone()), // ????
            QueryValue::Scalar(n) => eval_number(ec, *n),
            _ => {
                let msg = format!("cannot cast {} to an instant vector", self.data_type());
                Err(RuntimeError::TypeCastError(msg))
            }
        }
    }

    pub fn get_range_vector(&self) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            QueryValue::RangeVector(val) => Ok(val.clone()), // ????
            _ => panic!("BUG: invalid series parameter"),
        }
    }

    pub fn into_instant_vector(self, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            QueryValue::InstantVector(mut val) => Ok(std::mem::take(&mut val)), // use .into instead ????
            QueryValue::Scalar(n) => eval_number(ec, n),
            _ => {
                let msg = format!("cannot cast {} to an instant vector", self.data_type());
                Err(RuntimeError::TypeCastError(msg))
            }
        }
    }

    pub fn get_string(&self) -> RuntimeResult<String> {
        match self {
            QueryValue::String(s) => Ok(s.to_string()),
            QueryValue::InstantVector(series) => {
                let ts = get_single_timeseries(series)?;
                if !ts.values.is_empty() {
                    let all_nan = series[0].values.iter().all(|x| x.is_nan());
                    if !all_nan {
                        let msg = "series contains non-string timeseries".to_string();
                        return Err(RuntimeError::ArgumentError(msg));
                    }
                }
                let res = ts.metric_name.measurement.clone();
                // todo: return reference
                Ok(res)
            }
            _ => {
                let msg = format!("cannot cast {} to a string", self.data_type());
                Err(RuntimeError::TypeCastError(msg))
            }
        }
    }

    pub fn as_instant_vec(&self, ec: &EvalConfig) -> RuntimeResult<Cow<Vec<Timeseries>>> {
        match self {
            QueryValue::InstantVector(v) => Ok(Cow::Borrowed(v)),
            QueryValue::Scalar(n) => Ok(Cow::Owned(eval_number(ec, *n)?)),
            _ => {
                let msg = format!("cannot cast {} to an instant vector", self.data_type());
                Err(RuntimeError::TypeCastError(msg))
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            QueryValue::InstantVector(val) => val.is_empty(), // ????
            QueryValue::RangeVector(val) => val.is_empty(),
            QueryValue::String(v) => v.is_empty(),
            _ => false,
        }
    }

    pub fn empty_vec() -> Self {
        QueryValue::InstantVector(vec![])
    }

    pub fn is_string_or_scalar(&self) -> bool {
        matches!(self, QueryValue::String(_) | QueryValue::Scalar(_))
    }

    pub fn len(&self) -> usize {
        match self {
            QueryValue::InstantVector(val) => val.len(), // ????
            QueryValue::RangeVector(val) => val.len(),
            QueryValue::String(v) => v.len(),
            _ => 0,
        }
    }
}

impl Value for QueryValue {
    fn value_type(&self) -> ValueType {
        match &self {
            QueryValue::RangeVector(_) => ValueType::RangeVector,
            QueryValue::Scalar(_) => ValueType::Scalar,
            QueryValue::String(_) => ValueType::String,
            QueryValue::InstantVector(_) => ValueType::InstantVector,
        }
    }
}

impl Default for QueryValue {
    fn default() -> Self {
        QueryValue::Scalar(0_f64)
    }
}

impl FromStr for QueryValue {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(QueryValue::String(s.to_string()))
    }
}

impl From<String> for QueryValue {
    fn from(s: String) -> Self {
        QueryValue::String(s)
    }
}

impl From<&str> for QueryValue {
    fn from(s: &str) -> Self {
        QueryValue::String(s.to_string())
    }
}

impl From<f64> for QueryValue {
    fn from(v: f64) -> Self {
        QueryValue::Scalar(v)
    }
}

impl From<i64> for QueryValue {
    fn from(v: i64) -> Self {
        QueryValue::Scalar(v as f64)
    }
}

impl From<Duration> for QueryValue {
    fn from(d: Duration) -> Self {
        QueryValue::Scalar(d.as_secs_f64())
    }
}

impl From<Vec<Timeseries>> for QueryValue {
    fn from(vec: Vec<Timeseries>) -> Self {
        QueryValue::InstantVector(vec)
    }
}

impl Clone for QueryValue {
    fn clone(&self) -> Self {
        match self {
            QueryValue::RangeVector(m) => QueryValue::RangeVector(m.clone()),
            QueryValue::InstantVector(series) => QueryValue::InstantVector(series.clone()),
            QueryValue::Scalar(f) => QueryValue::Scalar(*f),
            QueryValue::String(s) => QueryValue::String(s.clone()),
        }
    }
}

impl Display for QueryValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryValue::RangeVector(m) => write!(f, "{:?}", m),
            QueryValue::InstantVector(series) => write!(f, "{:?}", series),
            QueryValue::Scalar(n) => format_number(f, *n),
            QueryValue::String(s) => write!(f, "{}", s),
        }
    }
}

pub enum ScalarIterator {
    Empty,
    Single(f64),
    Multiple(Vec<f64>),
}

impl ScalarIterator {}

impl Iterator for ScalarIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ScalarIterator::Empty => None,
            ScalarIterator::Single(v) => {
                let res = *v;
                *self = ScalarIterator::Empty;
                Some(res)
            }
            ScalarIterator::Multiple(v) => v.pop(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_ordering_test() {
        let dp_1 = Sample::new(20, 2.0);
        let dp_2 = Sample::new(10, 3.0);
        let dp_3 = Sample::new(10, 3.0);

        // The ordering of data points is based on time, so dp_2 will be less than dp_1.
        assert!(dp_2 < dp_1);

        // Data points are equal if their time and values are equal.
        assert!(dp_2 == dp_3 && dp_1 != dp_2);

        // Data points with NaN values are equal if their times are equal.
        let dp_4 = Sample::new(10, f64::NAN);
        let dp_5 = Sample::new(10, f64::NAN);
        assert_eq!(dp_4, dp_5);
    }
}