use std::fmt::{Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

use crate::common::ValueType;
use crate::functions::signature::Signature;
use crate::functions::{BuiltinFunction, FunctionMeta, MAX_ARG_COUNT};
use crate::parser::ParseError;

// TODO: ttf

/// Transform functions calculate transformations over rollup results.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash, EnumIter, Serialize, Deserialize)]
pub enum TransformFunction {
    Abs,
    Absent,
    Acos,
    Acosh,
    Alias,
    Asin,
    Asinh,
    Atan,
    Atanh,
    BitmapAnd,
    BitmapOr,
    BitmapXor,
    BucketsLimit,
    Ceil,
    Clamp,
    ClampMax,
    ClampMin,
    Cos,
    Cosh,
    DayOfMonth,
    DayOfWeek,
    DayOfYear,
    DaysInMonth,
    Deg,
    DropCommonLabels,
    DropEmptySeries,
    End,
    Exp,
    Floor,
    HistogramAvg,
    HistogramQuantile,
    HistogramQuantiles,
    HistogramShare,
    HistogramStddev,
    HistogramStdvar,
    Hour,
    Interpolate,
    KeepLastValue,
    KeepNextValue,
    LabelCopy,
    LabelDel,
    LabelGraphiteGroup,
    LabelJoin,
    LabelKeep,
    LabelLowercase,
    LabelMap,
    LabelMatch,
    LabelMismatch,
    LabelMove,
    LabelReplace,
    LabelSet,
    LabelTransform,
    LabelUppercase,
    LabelValue,
    LabelsEqual,
    LimitOffset,
    Ln,
    Log2,
    Log10,
    Minute,
    Month,
    Now,
    Pi,
    PrometheusBuckets,
    Rad,
    Random,
    RandExponential,
    RandNormal,
    RangeAvg,
    RangeFirst,
    RangeLast,
    RangeLinearRegression,
    RangeMAD,
    RangeMax,
    RangeMedian,
    RangeMin,
    RangeNormalize,
    RangeQuantile,
    RangeStdDev,
    RangeStdVar,
    RangeSum,
    RangeTrimOutliers,
    RangeTrimSpikes,
    RangeTrimZScore,
    RangeZScore,
    RemoveResets,
    Round,
    Ru,
    RunningAvg,
    RunningMax,
    RunningMin,
    RunningSum,
    Scalar,
    Sgn,
    Sin,
    Sinh,
    SmoothExponential,
    Sort,
    SortByLabel,
    SortByLabelDesc,
    SortByLabelNumeric,
    SortByLabelNumericDesc,
    SortDesc,
    Sqrt,
    Start,
    Step,
    Tan,
    Tanh,
    Time,
    // "timestamp" has been moved to rollup funcs. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/415
    TimezoneOffset,
    Union,
    Vector,
    Year,
}

impl Display for TransformFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl FromStr for TransformFunction {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(meta) = FunctionMeta::lookup(s) {
            if let BuiltinFunction::Transform(tf) = &meta.function {
                return Ok(*tf);
            }
        }
        Err(ParseError::InvalidFunction(s.to_string()))
    }
}

impl TransformFunction {
    pub const fn name(&self) -> &'static str {
        use TransformFunction::*;
        match self {
            Abs => "abs",
            Absent => "absent",
            Acos => "acos",
            Acosh => "acosh",
            Alias => "alias",
            Asin => "asin",
            Asinh => "asinh",
            Atan => "atan",
            Atanh => "atanh",
            BitmapAnd => "bitmap_and",
            BitmapOr => "bitmap_or",
            BitmapXor => "bitmap_xor",
            BucketsLimit => "buckets_limit",
            Ceil => "ceil",
            Clamp => "clamp",
            ClampMax => "clamp_max",
            ClampMin => "clamp_min",
            Cos => "cos",
            Cosh => "cosh",
            DayOfMonth => "day_of_month",
            DayOfWeek => "day_of_week",
            DayOfYear => "day_of_year",
            DaysInMonth => "days_in_month",
            Deg => "deg",
            DropCommonLabels => "drop_common_labels",
            DropEmptySeries => "drop_empty_series",
            End => "end",
            Exp => "exp",
            Floor => "floor",
            HistogramAvg => "histogram_avg",
            HistogramQuantile => "histogram_quantile",
            HistogramQuantiles => "histogram_quantiles",
            HistogramShare => "histogram_share",
            HistogramStddev => "histogram_stddev",
            HistogramStdvar => "histogram_stdvar",
            Hour => "hour",
            Interpolate => "interpolate",
            KeepLastValue => "keep_last_value",
            KeepNextValue => "keep_next_value",
            LabelCopy => "label_copy",
            LabelDel => "label_del",
            LabelGraphiteGroup => "label_graphite_group",
            LabelJoin => "label_join",
            LabelKeep => "label_keep",
            LabelLowercase => "label_lowercase",
            LabelMap => "label_map",
            LabelMatch => "label_match",
            LabelMismatch => "label_mismatch",
            LabelMove => "label_move",
            LabelReplace => "label_replace",
            LabelSet => "label_set",
            LabelTransform => "label_transform",
            LabelUppercase => "label_uppercase",
            LabelValue => "label_value",
            LabelsEqual => "labels_equal",
            LimitOffset => "limit_offset",
            Ln => "ln",
            Log2 => "log2",
            Log10 => "log10",
            Minute => "minute",
            Month => "month",
            Now => "now",
            Pi => "pi",
            PrometheusBuckets => "prometheus_buckets",
            Rad => "rad",
            Random => "rand",
            RandExponential => "rand_exponential",
            RandNormal => "rand_normal",
            RangeAvg => "range_avg",
            RangeFirst => "range_first",
            RangeLast => "range_last",
            RangeLinearRegression => "range_linear_regression",
            RangeMAD => "range_mad",
            RangeMax => "range_max",
            RangeMedian => "range_median",
            RangeMin => "range_min",
            RangeNormalize => "range_normalize",
            RangeQuantile => "range_quantile",
            RangeStdDev => "range_stddev",
            RangeStdVar => "range_stdvar",
            RangeSum => "range_sum",
            RangeTrimSpikes => "range_trim_spikes",
            RangeTrimOutliers => "range_trim_outliers",
            RangeTrimZScore => "range_trim_zscore",
            RangeZScore => "range_zscore",
            RemoveResets => "remove_resets",
            Round => "round",
            Ru => "ru",
            RunningAvg => "running_avg",
            RunningMax => "running_max",
            RunningMin => "running_min",
            RunningSum => "running_sum",
            Scalar => "scalar",
            Sgn => "sgn",
            Sin => "sin",
            Sinh => "sinh",
            SmoothExponential => "smooth_exponential",
            Sort => "sort",
            SortByLabel => "sort_by_label",
            SortByLabelDesc => "sort_by_label_desc",
            SortByLabelNumeric => "sort_by_label_numeric",
            SortByLabelNumericDesc => "sort_by_label_numeric_desc",
            SortDesc => "sort_desc",
            Sqrt => "sqrt",
            Start => "start",
            Step => "step",
            Tan => "tan",
            Tanh => "tanh",
            Time => "time",
            TimezoneOffset => "timezone_offset",
            Union => "union",
            Vector => "vector",
            Year => "year",
        }
    }

    pub const fn may_sort_results(&self) -> bool {
        use TransformFunction::*;
        matches!(
            &self,
            LimitOffset 
            | Sort 
            | SortDesc
            | SortByLabel
            | SortByLabelDesc
            | SortByLabelNumeric
            | SortByLabelNumericDesc
        )
    }

    pub const fn manipulates_labels(&self) -> bool {
        use TransformFunction::*;
        matches!(
            self,
            Alias
                | DropCommonLabels
                | DropEmptySeries
                | LabelCopy
                | LabelDel
                | LabelGraphiteGroup
                | LabelJoin
                | LabelKeep
                | LabelLowercase
                | LabelMap
                | LabelMove
                | LabelReplace
                | LabelSet
                | LabelTransform
                | LabelUppercase
        )
    }

    pub fn signature(&self) -> Signature {
        use TransformFunction::*;

        // note: the expression must accept the type returned by this function or the execution panics.
        match self {
            Alias => Signature::exact(vec![ValueType::InstantVector, ValueType::String]),
            BitmapAnd | BitmapOr | BitmapXor => Signature::exact(vec![ValueType::InstantVector, ValueType::Scalar]),
            BucketsLimit => Signature::exact(vec![ValueType::Scalar, ValueType::InstantVector]),
            Clamp => Signature::exact(
                vec![
                    ValueType::InstantVector,
                    ValueType::Scalar,
                    ValueType::Scalar,
                ],
            ),
            ClampMax | ClampMin => Signature::exact(
                vec![ValueType::InstantVector, ValueType::Scalar],
            ),
            Start | End => Signature::exact(vec![]),
            DropCommonLabels => {
                Signature::variadic_equal(ValueType::InstantVector, 1)
            }
            HistogramQuantile => Signature::exact(vec![ValueType::Scalar, ValueType::InstantVector]),
            HistogramQuantiles => {
                // histogram_quantiles("phiLabel", phi1, ..., phiN, buckets)
                // todo: need a better way to handle variadic args with specific types
                Signature::variadic_any(3)
            }
            // histogram_share(le, buckets)
            HistogramShare => Signature::exact(vec![ValueType::Scalar, ValueType::InstantVector]),
            LabelCopy | LabelMove | LabelSet => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact_with_min_args(types, 3)
            }
            LabelDel | LabelKeep | LabelLowercase | LabelUppercase => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact_with_min_args(types, 2)
            }
            LabelJoin => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact_with_min_args(types, 4)
            }
            LabelMap => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact_with_min_args(types, 4)
            }
            LabelMatch | LabelMismatch => {
                let types = vec![
                    ValueType::InstantVector,
                    ValueType::String,
                    ValueType::String,
                ];
                Signature::exact_with_min_args(types, 3)
            }
            LabelGraphiteGroup => {
                // label_graphite_group(q, groupNum1, ... groupNumN)
                let mut types = vec![ValueType::Scalar; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact(types)
            }
            LabelReplace => {
                // label_replace(q, "dst_label", "replacement", "src_label", "regex")
                Signature::exact(
                    vec![
                        ValueType::InstantVector,
                        ValueType::String,
                        ValueType::String,
                        ValueType::String,
                        ValueType::String,
                    ],
                )
            }
            LabelTransform => {
                // label_transform(q, "label", "regexp", "replacement")
                Signature::exact(
                    vec![
                        ValueType::InstantVector,
                        ValueType::String,
                        ValueType::String,
                        ValueType::String,
                    ],
                )
            }
            LabelValue => Signature::exact(vec![ValueType::InstantVector, ValueType::String]),
            LimitOffset => Signature::exact(
                vec![
                    ValueType::Scalar,
                    ValueType::Scalar,
                    ValueType::InstantVector,
                ],
            ),
            Now => Signature::exact(vec![]),
            Pi => Signature::exact(vec![]),
            Random | RandExponential | RandNormal => {
                Signature::exact_with_min_args(vec![ValueType::Scalar], 0)
            }
            RangeNormalize => {
                Signature::variadic_min(vec![ValueType::InstantVector], 1)
            }
            RangeTrimOutliers | RangeTrimSpikes | RangeTrimZScore => Signature::exact(
                vec![ValueType::Scalar, ValueType::InstantVector]),
            RangeQuantile => Signature::exact(vec![ValueType::Scalar, ValueType::InstantVector]),
            Round => Signature::exact_with_min_args(vec![ValueType::InstantVector, ValueType::Scalar], 1),
            Ru => Signature::exact(vec![ValueType::RangeVector, ValueType::RangeVector]),
            Scalar => Signature::any(1),
            SmoothExponential => Signature::exact(vec![ValueType::InstantVector, ValueType::Scalar]),
            Sort => Signature::exact(vec![ValueType::RangeVector]),
            SortByLabel | SortByLabelDesc | SortByLabelNumeric | SortByLabelNumericDesc => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::RangeVector);
                Signature::exact_with_min_args(types, 2)
            }
            Step => Signature::exact(vec![]),
            Time => Signature::exact(vec![]),
            TimezoneOffset => Signature::exact(vec![ValueType::String]),
            Union => {
                let types = vec![ValueType::InstantVector; MAX_ARG_COUNT];
                Signature::exact_with_min_args(types, 1)
            }
            Vector => Signature::exact(vec![ValueType::InstantVector]),
            // DateTime functions
            DayOfMonth | DayOfWeek | DayOfYear | DaysInMonth | Hour | Minute | Month | Year => {
                Signature::exact_with_min_args(vec![ValueType::InstantVector], 0)
            }
            _ => {
                // by default we take a single arg containing series
                Signature::exact(vec![ValueType::InstantVector])
            }
        }
    }

    pub const fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

pub const fn get_transform_arg_idx_for_optimization(
    func: TransformFunction,
    arg_count: usize,
) -> Option<usize> {

    use TransformFunction::*;
    match func {
        Absent | DropCommonLabels | Scalar => None,
        End | Now | Pi | RangeNormalize | Ru | Start | Step | Time | Union | Vector => None, // todo Ru
        LabelGraphiteGroup => Some(0),
        LimitOffset => Some(2),
        BucketsLimit | HistogramQuantile | HistogramShare | RangeQuantile |
        RangeTrimSpikes | RangeTrimOutliers | RangeTrimZScore => Some(1),
        HistogramQuantiles => Some(arg_count - 1),
        _ => {
            if func.manipulates_labels() {
                return None;
            }
            Some(0)
        },
    }
}
