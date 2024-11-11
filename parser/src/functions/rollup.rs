use std::fmt::{Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

use crate::common::ValueType;
use crate::functions::signature::{Signature, Volatility};
use crate::functions::{BuiltinFunction, FunctionMeta, MAX_ARG_COUNT};
use crate::parser::ParseError;

/// Built-in Rollup Functions
#[derive(
    Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Default, EnumIter, Serialize, Deserialize,
)]
pub enum RollupFunction {
    AbsentOverTime,
    AggrOverTime,
    AscentOverTime,
    AvgOverTime,
    Changes,
    ChangesPrometheus,
    CountEqOverTime,
    CountGtOverTime,
    CountLeOverTime,
    CountNeOverTime,
    CountOverTime,
    CountValuesOverTime,
    DecreasesOverTime,
    #[default]
    DefaultRollup,
    Delta,
    DeltaPrometheus,
    Deriv,
    DerivFast,
    DescentOverTime,
    DistinctOverTime,
    DurationOverTime,
    FirstOverTime,
    GeomeanOverTime,
    HistogramOverTime,
    HoeffdingBoundLower,
    HoeffdingBoundUpper,
    HoltWinters,
    IDelta,
    IDeriv,
    Increase,
    IncreasePrometheus,
    IncreasePure,
    IncreasesOverTime,
    Integrate,
    IQROverTime,
    IRate, // + rollupFuncsRemoveCounterResets
    Lag,
    LastOverTime,
    Lifetime,
    MadOverTime,
    MaxOverTime,
    MedianOverTime,
    MinOverTime,
    ModeOverTime,
    OutlierIQROverTime,
    PredictLinear,
    PresentOverTime,
    QuantileOverTime,
    QuantilesOverTime,
    RangeOverTime,
    Rate,
    RateOverSum,
    Resets,
    Rollup,
    RollupCandlestick,
    RollupDelta,
    RollupDeriv,
    RollupIncrease,
    RollupRate,
    RollupScrapeInterval,
    ScrapeInterval,
    ShareEqOverTime,
    ShareGtOverTime,
    ShareLeOverTime,
    StaleSamplesOverTime,
    StddevOverTime,
    StdvarOverTime,
    SumEqOverTime,
    SumGtOverTime,
    SumLeOverTime,
    SumOverTime,
    Sum2OverTime,
    TFirstOverTime,
    Timestamp,
    TimestampWithName,
    TLastChangeOverTime,
    TLastOverTime,
    TMaxOverTime,
    TMinOverTime,
    ZScoreOverTime,
}

impl Display for RollupFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl RollupFunction {
    pub const fn name(&self) -> &'static str {
        use RollupFunction::*;

        match self {
            AbsentOverTime => "absent_over_time",
            AggrOverTime => "aggr_over_time",
            AscentOverTime => "ascent_over_time",
            AvgOverTime => "avg_over_time",
            Changes => "changes",
            ChangesPrometheus => "changes_prometheus",
            CountEqOverTime => "count_eq_over_time",
            CountGtOverTime => "count_gt_over_time",
            CountLeOverTime => "count_le_over_time",
            CountNeOverTime => "count_ne_over_time",
            CountOverTime => "count_over_time",
            CountValuesOverTime => "count_values_over_time",
            DecreasesOverTime => "decreases_over_time",
            DefaultRollup => "default_rollup",
            Delta => "delta",
            DeltaPrometheus => "delta_prometheus",
            Deriv => "deriv",
            DerivFast => "deriv_fast",
            DescentOverTime => "descent_over_time",
            DistinctOverTime => "distinct_over_time",
            DurationOverTime => "duration_over_time",
            FirstOverTime => "first_over_time",
            GeomeanOverTime => "geomean_over_time",
            HistogramOverTime => "histogram_over_time",
            HoeffdingBoundLower => "hoeffding_bound_lower",
            HoeffdingBoundUpper => "hoeffding_bound_upper",
            HoltWinters => "holt_winters",
            IDelta => "idelta",
            IDeriv => "ideriv",
            Increase => "increase",
            IncreasePrometheus => "increase_prometheus",
            IncreasePure => "increase_pure",
            IncreasesOverTime => "increases_over_time",
            Integrate => "integrate",
            IQROverTime => "iqr_over_time",
            IRate => "irate",
            Lag => "lag",
            LastOverTime => "last_over_time",
            Lifetime => "lifetime",
            MadOverTime => "mad_over_time",
            MaxOverTime => "max_over_time",
            MedianOverTime => "median_over_time",
            MinOverTime => "min_over_time",
            ModeOverTime => "mode_over_time",
            OutlierIQROverTime => "outlier_iqr_over_time",
            PredictLinear => "predict_linear",
            PresentOverTime => "present_over_time",
            QuantileOverTime => "quantile_over_time",
            QuantilesOverTime => "quantiles_over_time",
            RangeOverTime => "range_over_time",
            Rate => "rate",
            RateOverSum => "rate_over_sum",
            Resets => "resets",
            Rollup => "rollup",
            RollupCandlestick => "rollup_candlestick",
            RollupDelta => "rollup_delta",
            RollupDeriv => "rollup_deriv",
            RollupIncrease => "rollup_increase",
            RollupRate => "rollup_rate",
            RollupScrapeInterval => "rollup_scrape_interval",
            ScrapeInterval => "scrape_interval",
            ShareEqOverTime => "share_eq_over_time",
            ShareGtOverTime => "share_gt_over_time",
            ShareLeOverTime => "share_le_over_time",
            StaleSamplesOverTime => "stale_samples_over_time",
            StddevOverTime => "stddev_over_time",
            StdvarOverTime => "stdvar_over_time",
            SumEqOverTime => "sum_eq_over_time",
            SumGtOverTime => "sum_gt_over_time",
            SumLeOverTime => "sum_le_over_time",
            SumOverTime => "sum_over_time",
            Sum2OverTime => "sum2_over_time",
            TFirstOverTime => "tfirst_over_time",
            Timestamp => "timestamp",
            TimestampWithName => "timestamp_with_name",
            TLastChangeOverTime => "tlast_change_over_time",
            TLastOverTime => "tlast_over_time",
            TMaxOverTime => "tmax_over_time",
            TMinOverTime => "tmin_over_time",
            ZScoreOverTime => "zscore_over_time",
        }
    }

    /// the signatures supported by the function `fun`.
    pub fn signature(&self) -> Signature {
        use RollupFunction::*;
        use ValueType::*;

        // note: the physical expression must accept the type returned by this function or the execution panics.
        match self {
            CountEqOverTime | CountLeOverTime | CountNeOverTime | CountGtOverTime
            | DurationOverTime | PredictLinear | ShareEqOverTime | ShareGtOverTime
            | ShareLeOverTime | SumEqOverTime | SumGtOverTime | SumLeOverTime | TFirstOverTime => {
                Signature::exact(vec![RangeVector, Scalar], Volatility::Immutable)
            }
            CountValuesOverTime => {
                Signature::exact(vec![String, RangeVector], Volatility::Immutable)
            }
            HoeffdingBoundLower | HoeffdingBoundUpper => {
                Signature::exact(vec![Scalar, RangeVector], Volatility::Immutable)
            }
            HoltWinters => {
                Signature::exact(vec![RangeVector, Scalar, Scalar], Volatility::Immutable)
            }
            AggrOverTime => {
                let mut quantile_types: Vec<ValueType> = vec![String; MAX_ARG_COUNT];
                quantile_types.insert(0, RangeVector);
                Signature::variadic_min(quantile_types, 2, Volatility::Volatile)
            }
            QuantilesOverTime => {
                let mut quantile_types: Vec<ValueType> = vec![RangeVector; MAX_ARG_COUNT];
                quantile_types.insert(0, RangeVector);
                Signature::variadic_min(quantile_types, 3, Volatility::Volatile)
            }
            Rollup | RollupDelta | RollupDeriv | RollupIncrease | RollupRate
            | RollupScrapeInterval | RollupCandlestick => {
                Signature::variadic_min(vec![RangeVector, String], 1, Volatility::Volatile)
            }
            _ => {
                // default
                Signature::uniform(1, RangeVector, Volatility::Immutable)
            }
        }
    }

    /// These functions don't change physical meaning of input time series,
    /// so they don't drop metric name
    pub const fn keep_metric_name(&self) -> bool {
        use RollupFunction::*;
        matches!(
            self,
            AvgOverTime
                | DefaultRollup
                | FirstOverTime
                | GeomeanOverTime
                | HoeffdingBoundLower
                | HoeffdingBoundUpper
                | HoltWinters
                | LastOverTime
                | MaxOverTime
                | MinOverTime
                | ModeOverTime
                | IQROverTime
                | PredictLinear
                | QuantileOverTime
                | QuantilesOverTime
                | Rollup
                | RollupCandlestick
                | TimestampWithName
        )
    }

    pub const fn should_remove_counter_resets(&self) -> bool {
        use RollupFunction::*;
        matches!(
            self,
            Increase
                | IncreasePrometheus
                | IncreasePure
                | IRate
                | Rate
                | RollupIncrease
                | RollupRate
        )
    }

    pub const fn is_aggregate_function(&self) -> bool {
        use RollupFunction::*;
        matches!(
            self,
            AbsentOverTime
                | AscentOverTime
                | AvgOverTime
                | Changes
                | CountOverTime
                | DecreasesOverTime
                | DefaultRollup
                | Delta
                | Deriv
                | DerivFast
                | DescentOverTime
                | DistinctOverTime
                | FirstOverTime
                | GeomeanOverTime
                | IDelta
                | IDeriv
                | Increase
                | IncreasePure
                | IncreasesOverTime
                | Integrate
                | IRate
                | Lag
                | LastOverTime
                | Lifetime
                | MaxOverTime
                | MinOverTime
                | MedianOverTime
                | ModeOverTime
                | PresentOverTime
                | RangeOverTime
                | Rate
                | RateOverSum
                | Resets
                | ScrapeInterval
                | StaleSamplesOverTime
                | StddevOverTime
                | StdvarOverTime
                | SumOverTime
                | Sum2OverTime
                | TFirstOverTime
                | Timestamp
                | TimestampWithName
                | TLastChangeOverTime
                | TLastOverTime
                | TMaxOverTime
                | TMinOverTime
                | ZScoreOverTime
        )
    }

    // All rollup the functions which do not rely on the previous sample
    // before the lookbehind window (aka prev_value), do not need silence interval.
    pub const fn need_silence_interval(&self) -> bool {
        use RollupFunction::*;
        !matches!(
            self,
            AscentOverTime
                | Changes
                | DecreasesOverTime
            // The default_rollup implicitly relies on the previous samples in order to fill gaps.
	        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/5388
                | DefaultRollup
                | Delta
                | DerivFast
                | DescentOverTime
                | IDelta
                | IDeriv
                | Increase
                | IncreasePure
                | IncreasesOverTime
                | Integrate
                | IRate
                | Lag
                | Lifetime
                | Rate
                | Resets
                | Rollup
                | RollupCandlestick
                | RollupDelta
                | RollupDeriv
                | RollupIncrease
                | RollupRate
                | RollupScrapeInterval
                | ScrapeInterval
                | TLastChangeOverTime
        )
    }
}

/// We can extend lookbehind window for these functions in order to make sure it contains enough
/// points for returning non-empty results.
///
/// This is needed for returning the expected non-empty graphs when zooming in the graph in Grafana,
/// which is built with `func_name(metric)` query.
pub const fn can_adjust_window(func: RollupFunction) -> bool {
    use RollupFunction::*;
    matches!(
        func,
        DefaultRollup
            | Deriv
            | DerivFast
            | IDeriv
            | IRate
            | Rate
            | RateOverSum
            | Rollup
            | RollupCandlestick
            | RollupDeriv
            | RollupRate
            | RollupScrapeInterval
            | ScrapeInterval
            | Timestamp
    )
}

impl FromStr for RollupFunction {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(meta) = FunctionMeta::lookup(s) {
            if let BuiltinFunction::Rollup(rf) = &meta.function {
                return Ok(*rf);
            }
        }
        Err(ParseError::InvalidFunction(s.to_string()))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash, EnumIter, Serialize, Deserialize)]
pub enum RollupTag {
    Min,
    Max,
    Avg,
}

impl Display for RollupTag {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RollupTag::Min => write!(f, "min"),
            RollupTag::Max => write!(f, "max"),
            RollupTag::Avg => write!(f, "avg"),
        }
    }
}

impl FromStr for RollupTag {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            s if s.eq_ignore_ascii_case("min") => Ok(RollupTag::Min),
            s if s.eq_ignore_ascii_case("max") => Ok(RollupTag::Max),
            s if s.eq_ignore_ascii_case("avg") => Ok(RollupTag::Avg),
            _ => Err(ParseError::InvalidFunction(format!(
                "invalid rollup tag::{s}",
            ))),
        }
    }
}

/// get_rollup_arg_idx returns the argument index for the given fe, which accepts the rollup argument.
///
/// -1 is returned if fe isn't a rollup function.
pub const fn get_rollup_arg_idx(fe: &RollupFunction, arg_count: usize) -> i32 {
    use RollupFunction::*;
    match fe {
        QuantileOverTime | HoeffdingBoundLower | HoeffdingBoundUpper => 1,
        QuantilesOverTime => (arg_count - 1) as i32,
        _ => 0,
    }
}

pub const fn get_rollup_arg_idx_for_optimization(
    func: RollupFunction,
    arg_count: usize,
) -> Option<usize> {
    // This must be kept in sync with GetRollupArgIdx()
    use RollupFunction::*;
    match func {
        AbsentOverTime => None,
        QuantileOverTime | HoeffdingBoundLower | HoeffdingBoundUpper => Some(1),
        QuantilesOverTime => Some(arg_count - 1),
        _ => Some(0),
    }
}

/// Determines if a given rollup function converts a range vector to an instant vector
///
/// Note that `_over_time` functions do not affect labels, unlike their regular
/// counterparts
pub fn is_rollup_aggregation_over_time(func: RollupFunction) -> bool {
    use RollupFunction::*;
    let name = func.name();

    if name.ends_with("over_time") {
        return true;
    }

    matches!(func, 
        | Delta
        | DeltaPrometheus
        | Deriv
        | DerivFast
        | IDelta
        | IDeriv
        | Increase
        | IncreasePure
        | IncreasePrometheus
        | IRate
        | PredictLinear
        | Rate
        | Resets
        | RollupDeriv
        | RollupDelta
        | RollupIncrease
        | RollupRate)
}
