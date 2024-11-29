use metricsql_common::pool::get_pooled_vec_f64;
use metricsql_parser::functions::RollupFunction;
use metricsql_parser::prelude::{BuiltinFunction, FunctionMeta};

use crate::common::math::{
    is_stale_nan, linear_regression, mad, mode_no_nans, quantile, stddev, stdvar,
};
use crate::functions::arg_parse::get_scalar_param_value;
use crate::functions::rollup::counts::{
    new_rollup_count_values, new_rollup_sum_eq, new_rollup_sum_gt, new_rollup_sum_le,
};
use crate::functions::rollup::types::RollupHandlerFactory;
use crate::functions::rollup::{
    counts::{
        new_rollup_count_eq, new_rollup_count_gt, new_rollup_count_le, new_rollup_count_ne,
        new_rollup_share_eq, new_rollup_share_gt, new_rollup_share_le,
    },
    delta::{
        new_rollup_delta, new_rollup_delta_prometheus, new_rollup_idelta, new_rollup_increase,
        rollup_delta, rollup_idelta,
    },
    deriv::{
        new_rollup_deriv, new_rollup_deriv_fast, new_rollup_ideriv, new_rollup_irate,
        new_rollup_rate, rollup_deriv_fast, rollup_deriv_slow, rollup_ideriv,
    },
    duration_over_time::new_rollup_duration_over_time,
    hoeffding_bound::{new_rollup_hoeffding_bound_lower, new_rollup_hoeffding_bound_upper},
    holt_winters::new_rollup_holt_winters,
    integrate::{new_rollup_integrate, rollup_integrate},
    outlier_iqr::rollup_outlier_iqr,
    quantiles::{new_rollup_quantile, new_rollup_quantiles},
    RollupHandlerFloat,
};
use crate::functions::rollup::{RollupFunc, RollupFuncArg, RollupHandler};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::QueryValue;

// https://github.com/VictoriaMetrics/VictoriaMetrics/blob/master/app/vmselect/promql/rollup.go

const NAN: f64 = f64::NAN;

pub(crate) const ROLLUP_LAST: fn(&RollupFuncArg) -> f64 = rollup_default;

pub(super) fn get_rollup_fn(f: &RollupFunction) -> RuntimeResult<RollupFunc> {
    use RollupFunction::*;

    let res = match f {
        AbsentOverTime => rollup_absent,
        AscentOverTime => rollup_ascent_over_time,
        AvgOverTime => rollup_avg,
        Changes => rollup_changes,
        CountOverTime => rollup_count,
        DecreasesOverTime => ROLLUP_DECREASES,
        DefaultRollup => rollup_default,
        Delta => rollup_delta,
        Deriv => rollup_deriv_slow,
        DerivFast => rollup_deriv_fast,
        DescentOverTime => rollup_descent_over_time,
        DistinctOverTime => rollup_distinct,
        FirstOverTime => rollup_first,
        GeomeanOverTime => rollup_geomean,
        IDelta => rollup_idelta,
        IDeriv => rollup_ideriv,
        Increase => rollup_delta,
        IncreasePure => rollup_increase_pure,
        IncreasesOverTime => rollup_increases,
        Integrate => rollup_integrate,
        IRate => rollup_ideriv,
        Lag => rollup_lag,
        LastOverTime => ROLLUP_LAST,
        Lifetime => rollup_lifetime,
        MaxOverTime => rollup_max,
        MinOverTime => rollup_min,
        MedianOverTime => rollup_median,
        ModeOverTime => rollup_mode_over_time,
        IQROverTime => rollup_outlier_iqr,
        PresentOverTime => rollup_present,
        RangeOverTime => rollup_range,
        Rate => rollup_deriv_fast,
        RateOverSum => rollup_rate_over_sum,
        Resets => rollup_resets,
        ScrapeInterval => rollup_scrape_interval,
        StaleSamplesOverTime => rollup_stale_samples,
        StddevOverTime => rollup_stddev,
        StdvarOverTime => rollup_stdvar,
        SumOverTime => rollup_sum,
        Sum2OverTime => rollup_sum2,
        TFirstOverTime => rollup_tfirst,
        Timestamp => rollup_tlast,
        TimestampWithName => rollup_tlast,
        TLastChangeOverTime => rollup_tlast_change,
        TLastOverTime => rollup_tlast,
        TMaxOverTime => rollup_tmax,
        TMinOverTime => rollup_tmin,
        ZScoreOverTime => rollup_zscore_over_time,
        _ => {
            return Err(RuntimeError::UnknownFunction(format!(
                "{} is not an rollup function",
                &f.name()
            )))
        }
    };

    Ok(res)
}

pub(crate) fn get_rollup_func_by_name(name: &str) -> RuntimeResult<RollupFunction> {
    if let Some(meta) = FunctionMeta::lookup(name) {
        if let BuiltinFunction::Rollup(f) = meta.function {
            return Ok(f);
        }
    }
    Err(RuntimeError::UnknownFunction(name.to_string()))
}

macro_rules! make_factory {
    ( $name: ident, $rf: expr ) => {
        #[inline]
        pub(super) fn $name(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
            Ok(RollupHandler::wrap($rf))
        }
    };
}

macro_rules! fake_wrapper {
    ( $funcName: ident, $name: expr ) => {
        #[inline]
        fn $funcName(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
            Ok(RollupHandler::fake($name))
        }
    };
}

// pass through to existing functions
make_factory!(new_rollup_absent_over_time, rollup_absent);
make_factory!(new_rollup_aggr_over_time, rollup_fake);
make_factory!(new_rollup_ascent_over_time, rollup_ascent_over_time);
make_factory!(new_rollup_avg_over_time, rollup_avg);
make_factory!(new_rollup_changes, rollup_changes);
make_factory!(new_rollup_changes_prometheus, rollup_changes_prometheus);
make_factory!(new_rollup_count_over_time, rollup_count);
make_factory!(new_rollup_decreases_over_time, ROLLUP_DECREASES);
make_factory!(new_rollup_default, rollup_default);
make_factory!(new_rollup_descent_over_time, rollup_descent_over_time);
make_factory!(new_rollup_distinct_over_time, rollup_distinct);
make_factory!(new_rollup_first_over_time, rollup_first);
make_factory!(new_rollup_geomean_over_time, rollup_geomean);
make_factory!(new_rollup_histogram_over_time, rollup_histogram);
make_factory!(new_rollup_increase_pure, rollup_increase_pure);
make_factory!(new_rollup_increases_over_time, rollup_increases);
make_factory!(new_rollup_lag, rollup_lag);
make_factory!(new_rollup_last_over_time, ROLLUP_LAST);
make_factory!(new_rollup_lifetime, rollup_lifetime);
make_factory!(new_rollup_mad_over_time, rollup_mad);
make_factory!(new_rollup_max_over_time, rollup_max);
make_factory!(new_rollup_min_over_time, rollup_min);
make_factory!(new_rollup_median_over_time, rollup_median);
make_factory!(new_rollup_mode_over_time, rollup_mode_over_time);
make_factory!(new_rollup_outlier_iqr_over_time, rollup_outlier_iqr);
make_factory!(new_rollup_present_over_time, rollup_present);
make_factory!(new_rollup_range_over_time, rollup_range);
make_factory!(new_rollup_rate_over_sum, rollup_rate_over_sum);
make_factory!(new_rollup_resets, rollup_resets);
make_factory!(new_rollup_scrape_interval, rollup_scrape_interval);
make_factory!(new_rollup_stale_samples_over_time, rollup_stale_samples);
make_factory!(new_rollup_stddev_over_time, rollup_stddev);
make_factory!(new_rollup_stdvar_over_time, rollup_stdvar);
make_factory!(new_rollup_sum_over_time, rollup_sum);
make_factory!(new_rollup_sum2_over_time, rollup_sum2);
make_factory!(new_rollup_tfirst_over_time, rollup_tfirst);
make_factory!(new_rollup_timestamp, rollup_tlast);
make_factory!(new_rollup_timestamp_with_name, rollup_tlast);
make_factory!(new_rollup_tlast_change_over_time, rollup_tlast_change);
make_factory!(new_rollup_tlast_over_time, rollup_tlast);
make_factory!(new_rollup_tmax_over_time, rollup_tmax);
make_factory!(new_rollup_tmin_over_time, rollup_tmin);
make_factory!(new_rollup_zscore_over_time, rollup_zscore_over_time);

fake_wrapper!(new_rollup, "rollup");
fake_wrapper!(new_rollup_candlestick, "rollup_candlestick");
fake_wrapper!(new_rollup_scrape_interval_fake, "rollup_scrape_interval");

pub(crate) fn get_rollup_function_handler(
    func: RollupFunction,
    args: &[QueryValue],
) -> RuntimeResult<RollupHandler> {
    let factory = get_rollup_function_factory(func);
    factory(args)
}

pub(crate) const fn get_rollup_function_factory(func: RollupFunction) -> RollupHandlerFactory {
    use RollupFunction::*;
    match func {
        AbsentOverTime => new_rollup_absent_over_time,
        AggrOverTime => new_rollup_aggr_over_time,
        AscentOverTime => new_rollup_ascent_over_time,
        AvgOverTime => new_rollup_avg_over_time,
        Changes => new_rollup_changes,
        ChangesPrometheus => new_rollup_changes_prometheus,
        CountEqOverTime => new_rollup_count_eq,
        CountGtOverTime => new_rollup_count_gt,
        CountLeOverTime => new_rollup_count_le,
        CountNeOverTime => new_rollup_count_ne,
        CountOverTime => new_rollup_count_over_time,
        CountValuesOverTime => new_rollup_count_values,
        DecreasesOverTime => new_rollup_decreases_over_time,
        DefaultRollup => new_rollup_default,
        Delta => new_rollup_delta,
        DeltaPrometheus => new_rollup_delta_prometheus,
        Deriv => new_rollup_deriv,
        DerivFast => new_rollup_deriv_fast,
        DescentOverTime => new_rollup_descent_over_time,
        DistinctOverTime => new_rollup_distinct_over_time,
        DurationOverTime => new_rollup_duration_over_time,
        FirstOverTime => new_rollup_first_over_time,
        GeomeanOverTime => new_rollup_geomean_over_time,
        HistogramOverTime => new_rollup_histogram_over_time,
        HoeffdingBoundLower => new_rollup_hoeffding_bound_lower,
        HoeffdingBoundUpper => new_rollup_hoeffding_bound_upper,
        HoltWinters => new_rollup_holt_winters,
        IDelta => new_rollup_idelta,
        IDeriv => new_rollup_ideriv,
        Increase => new_rollup_increase,
        IncreasePrometheus => new_rollup_delta_prometheus,
        IncreasePure => new_rollup_increase_pure,
        IncreasesOverTime => new_rollup_increases_over_time,
        Integrate => new_rollup_integrate,
        IRate => new_rollup_irate,
        IQROverTime => new_rollup_outlier_iqr_over_time,
        Lag => new_rollup_lag,
        LastOverTime => new_rollup_last_over_time,
        Lifetime => new_rollup_lifetime,
        MadOverTime => new_rollup_mad_over_time,
        MaxOverTime => new_rollup_max_over_time,
        MedianOverTime => new_rollup_median_over_time,
        MinOverTime => new_rollup_min_over_time,
        ModeOverTime => new_rollup_mode_over_time,
        OutlierIQROverTime => new_rollup_outlier_iqr_over_time,
        PredictLinear => new_rollup_predict_linear,
        PresentOverTime => new_rollup_present_over_time,
        QuantileOverTime => new_rollup_quantile,
        QuantilesOverTime => new_rollup_quantiles,
        RangeOverTime => new_rollup_range_over_time,
        Rate => new_rollup_rate,
        RateOverSum => new_rollup_rate_over_sum,
        Resets => new_rollup_resets,
        Rollup => new_rollup,
        RollupCandlestick => new_rollup_candlestick,
        RollupDelta => new_rollup_delta,
        RollupDeriv => new_rollup_deriv,
        RollupIncrease => new_rollup_increase,
        RollupRate => new_rollup_rate,
        RollupScrapeInterval => new_rollup_scrape_interval_fake,
        ScrapeInterval => new_rollup_scrape_interval,
        ShareEqOverTime => new_rollup_share_eq,
        ShareGtOverTime => new_rollup_share_gt,
        ShareLeOverTime => new_rollup_share_le,
        StaleSamplesOverTime => new_rollup_stale_samples_over_time,
        StddevOverTime => new_rollup_stddev_over_time,
        StdvarOverTime => new_rollup_stdvar_over_time,
        SumOverTime => new_rollup_sum_over_time,
        SumEqOverTime => new_rollup_sum_eq,
        SumGtOverTime => new_rollup_sum_gt,
        SumLeOverTime => new_rollup_sum_le,
        Sum2OverTime => new_rollup_sum2_over_time,
        TFirstOverTime => new_rollup_tfirst_over_time,
        Timestamp => new_rollup_timestamp,
        TimestampWithName => new_rollup_timestamp_with_name,
        TLastChangeOverTime => new_rollup_tlast_change_over_time,
        TLastOverTime => new_rollup_tlast_over_time,
        TMaxOverTime => new_rollup_tmax_over_time,
        TMinOverTime => new_rollup_tmin_over_time,
        ZScoreOverTime => new_rollup_zscore_over_time,
    }
}

pub(super) fn remove_counter_resets(values: &mut [f64]) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from storage.
    if values.is_empty() {
        return;
    }
    let mut correction: f64 = 0.0;
    let mut prev_value = values[0];

    for i in 0..values.len() {
        let mut val = values[i];
        let d = val - prev_value;
        if d < 0.0 {
            if (-d * 8.0) < prev_value {
                // This is likely a partial counter reset.
                // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2787
                correction += prev_value - val;
            } else {
                correction += prev_value;
            }
        }
        prev_value = val;
        val += correction;
        // Check again, there could be precision error in float operations,
        // see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/5571
        if i > 0 && val < prev_value {
            val = prev_value;
        }
        values[i] = val;
    }
}

fn new_rollup_predict_linear(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    let secs = get_scalar_param_value(args, 1, "predict_linear", "secs")?;

    let handler = RollupHandlerFloat::new(secs, |rfa: &RollupFuncArg, secs: &f64| {
        let secs = *secs;
        let (v, k) = linear_regression(rfa.values, rfa.timestamps, rfa.curr_timestamp);
        if v.is_nan() {
            return NAN;
        }
        v + k * secs
    });

    Ok(RollupHandler::FloatArg(handler))
}

pub(super) fn rollup_histogram(rfa: &RollupFuncArg) -> f64 {
    let map = rfa.get_tsm();
    map.process_rollup(rfa.values, rfa.idx);
    NAN
}

pub(super) fn rollup_avg(rfa: &RollupFuncArg) -> f64 {
    // do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
    // since it is slower and has no significant benefits in precision.

    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    let sum: f64 = rfa.values.iter().copied().sum();
    sum / rfa.values.len() as f64
}

pub(super) fn rollup_min(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }

    let min = rfa.values.iter().copied().min_by(|a, b| a.total_cmp(b)).unwrap();
    min
}

pub(crate) fn rollup_mad(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup funcs.
    mad(rfa.values)
}

pub(super) fn rollup_max(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }

    *rfa.values
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

pub(super) fn rollup_median(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    quantile(0.5, rfa.values)
}

pub(super) fn rollup_tmin(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    if values.is_empty() {
        return NAN;
    }
    let mut min_value = values[0];
    let mut min_timestamp = rfa.timestamps[0];
    for (v, ts) in rfa.values.iter().zip(rfa.timestamps.iter()) {
        // Get the last timestamp for the minimum value as most users expect.
        if v <= &min_value {
            min_value = *v;
            min_timestamp = *ts;
        }
    }
    min_timestamp as f64 / 1e3_f64
}

pub(super) fn rollup_tmax(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    if rfa.values.is_empty() {
        return NAN;
    }

    let mut max_value = rfa.values[0];
    let mut max_timestamp = rfa.timestamps[0];

    for (v, ts) in rfa.values.iter().zip(rfa.timestamps.iter()) {
        // Get the last timestamp for the maximum value as most users expect.
        if *v >= max_value {
            max_value = *v;
            max_timestamp = *ts;
        }
    }

    max_timestamp as f64 / 1e3_f64
}

pub(super) fn rollup_tfirst(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.timestamps.is_empty() {
        // do not take into account rfa.prev_timestamp, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    rfa.timestamps[0] as f64 / 1e3_f64
}

pub(super) fn rollup_tlast(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let timestamps = rfa.timestamps;
    if timestamps.is_empty() {
        // do not take into account rfa.prev_timestamp, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    timestamps[timestamps.len() - 1] as f64 / 1e3_f64
}

pub(super) fn rollup_tlast_change(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return NAN;
    }

    let values = rfa.values;
    let timestamps = rfa.timestamps;

    let last = values.len() - 1;
    let last_value = values[last];

    for i in (0..last).rev() {
        if values[i] != last_value {
            return timestamps[i + 1] as f64 / 1e3_f64;
        }
    }

    if rfa.prev_value.is_nan() || rfa.prev_value != last_value {
        return timestamps[0] as f64 / 1e3_f64;
    }
    NAN
}

pub(super) fn rollup_sum(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }

    rfa.values.iter().fold(0.0, |r, x| r + *x)
}

pub(super) fn rollup_rate_over_sum(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let timestamps = rfa.timestamps;
    if timestamps.is_empty() {
        return NAN;
    }
    let sum: f64 = rfa.values.iter().sum();
    sum / (rfa.window as f64 / 1e3_f64)
}

pub(super) fn rollup_range(rfa: &RollupFuncArg) -> f64 {
    if rfa.values.is_empty() {
        return NAN;
    }
    let values = rfa.values;
    let mut max = values[0];
    let mut min = max;
    for v in values.iter().cloned() {
        max = max.max(v);
        min = min.min(v);
    }
    max - min
}

pub(super) fn rollup_sum2(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return NAN;
    }
    rfa.values.iter().copied().fold(0.0, |r, x| r + (x * x))
}

pub(super) fn rollup_geomean(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let len = rfa.values.len();
    if len == 0 {
        return NAN;
    }

    let p = rfa.values.iter().fold(1.0, |r, v| r * *v);
    p.powf(1.0 / len as f64)
}

pub(super) fn rollup_absent(rfa: &RollupFuncArg) -> f64 {
    if rfa.values.is_empty() {
        return 1.0;
    }
    NAN
}

pub(super) fn rollup_present(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if !rfa.values.is_empty() {
        return 1.0;
    }
    NAN
}

pub(super) fn rollup_count(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return NAN;
    }
    rfa.values.len() as f64
}

pub(super) fn rollup_stale_samples(rfa: &RollupFuncArg) -> f64 {
    let values = rfa.values;
    if values.is_empty() {
        return NAN;
    }
    rfa.values.iter().filter(|v| is_stale_nan(**v)).count() as f64
}

pub(super) fn rollup_stddev(rfa: &RollupFuncArg) -> f64 {
    stddev(rfa.values)
}

pub(super) fn rollup_stdvar(rfa: &RollupFuncArg) -> f64 {
    stdvar(rfa.values)
}

pub(super) fn rollup_increase_pure(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    let values = rfa.values;
    let count = values.len();

    // restore to the real value because of potential staleness reset
    let mut prev_value = rfa.real_prev_value;

    if prev_value.is_nan() {
        if count == 0 {
            return NAN;
        }
        // Assume the counter starts from 0.
        prev_value = 0.0;
    }

    if rfa.values.is_empty() {
        // Assume the counter didn't change since prev_value.
        return 0f64;
    }
    values[count - 1] - prev_value
}

pub(super) fn rollup_lifetime(rfa: &RollupFuncArg) -> f64 {
    // Calculate the duration between the first and the last data points.
    let timestamps = rfa.timestamps;
    let count = timestamps.len();
    if rfa.prev_value.is_nan() {
        if count < 2 {
            return NAN;
        }
        return (timestamps[count - 1] - timestamps[0]) as f64 / 1e3_f64;
    }
    if count == 0 {
        return NAN;
    }
    (timestamps[count - 1] - rfa.prev_timestamp) as f64 / 1e3_f64
}

pub(super) fn rollup_lag(rfa: &RollupFuncArg) -> f64 {
    // Calculate the duration between the current timestamp and the last data point.
    let count = rfa.timestamps.len();
    if count == 0 {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        return (rfa.curr_timestamp - rfa.prev_timestamp) as f64 / 1e3_f64;
    }
    (rfa.curr_timestamp - rfa.timestamps[count - 1]) as f64 / 1e3_f64
}

/// Calculate the average interval between data points.
pub(super) fn rollup_scrape_interval(rfa: &RollupFuncArg) -> f64 {
    let count = rfa.timestamps.len();
    if rfa.prev_value.is_nan() {
        if count < 2 {
            return NAN;
        }
        return ((rfa.timestamps[count - 1] - rfa.timestamps[0]) as f64 / 1e3_f64)
            / (count - 1) as f64;
    }
    if count == 0 {
        return NAN;
    }
    ((rfa.timestamps[count - 1] - rfa.prev_timestamp) as f64 / 1e3_f64) / count as f64
}

#[inline]
fn change_below_tolerance(v: f64, prev_value: f64) -> bool {
    let tolerance = 1e-12 * v.abs();
    (v - prev_value).abs() < tolerance
}

pub(super) fn rollup_changes_prometheus(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    // do not take into account rfa.prev_value like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1962
    if rfa.values.is_empty() {
        return NAN;
    }
    let mut prev_value = rfa.values[0];
    let mut n = 0;
    for v in rfa.values.iter().skip(1) {
        if *v != prev_value {
            if change_below_tolerance(*v, prev_value) {
                // This may be precision error. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/767#issuecomment-1650932203
                continue;
            }
            n += 1;
            prev_value = *v;
        }
    }

    n as f64
}

pub(super) fn rollup_changes(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut n = 0;
    let mut values = &rfa.values[0..];
    let mut prev_value = rfa.prev_value;
    if prev_value.is_nan() {
        if rfa.values.is_empty() {
            return NAN;
        }
        prev_value = rfa.values[0];
        values = &values[1..];
        n += 1;
    }

    for v in values.iter() {
        if *v != prev_value {
            if change_below_tolerance(*v, prev_value) {
                // This may be precision error. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/767#issuecomment-1650932203
                continue;
            }
            n += 1;
            prev_value = *v;
        }
    }

    n as f64
}

pub(super) fn rollup_increases(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut prev_value = rfa.prev_value;
    let mut values = &rfa.values[0..];

    if values.is_empty() {
        if prev_value.is_nan() {
            return NAN;
        }
        return 0.0;
    }

    if prev_value.is_nan() {
        prev_value = values[0];
        values = &values[1..];
    }

    if values.is_empty() {
        return 0.0;
    }

    let mut n = 0;
    for v in values.iter() {
        if *v > prev_value {
            if change_below_tolerance(*v, prev_value) {
                // This may be precision error. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/767#issuecomment-1650932203
                continue;
            }
            n += 1;
        }
        prev_value = *v;
    }

    n as f64
}

// `decreases_over_time` logic is the same as `resets` logic.
const ROLLUP_DECREASES: RollupFunc = rollup_resets;

pub(super) fn rollup_resets(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    let mut values = &rfa.values[0..];
    if values.is_empty() {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        return 0.0;
    }

    let mut prev_value = rfa.prev_value;
    if prev_value.is_nan() {
        prev_value = values[0];
        values = &values[1..];
    }

    if values.is_empty() {
        return 0.0;
    }

    let mut n = 0;
    for v in values.iter() {
        if *v < prev_value {
            if change_below_tolerance(*v, prev_value) {
                // This may be precision error. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/767#issuecomment-1650932203
                continue;
            }
            n += 1;
        }
        prev_value = *v;
    }

    n as f64
}

pub(super) fn rollup_mode_over_time(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    // Copy rfa.values to a, since modeNoNaNs modifies a contents.
    if rfa.values.is_empty() {
        let mut a = vec![];
        return mode_no_nans(rfa.prev_value, &mut a);
    }
    let mut a = get_pooled_vec_f64(rfa.values.len());
    a.extend_from_slice(rfa.values);
    mode_no_nans(rfa.prev_value, &mut a)
}

pub(super) fn rollup_ascent_over_time(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    let mut prev_value = rfa.prev_value;
    let mut start: usize = 0;
    if prev_value.is_nan() {
        if values.is_empty() {
            return NAN;
        }
        prev_value = values[0];
        start = 1;
    }
    let mut s: f64 = 0.0;
    for v in &values[start..] {
        let d = v - prev_value;
        if d > 0.0 {
            s += d;
        }
        prev_value = *v;
    }
    s
}

pub(super) fn rollup_descent_over_time(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut ofs = 0;
    let mut prev_value = rfa.prev_value;
    if prev_value.is_nan() {
        if rfa.values.is_empty() {
            return NAN;
        }
        prev_value = rfa.values[0];
        ofs = 1;
    }

    let mut s: f64 = 0.0;
    for v in &rfa.values[ofs..] {
        let d = prev_value - *v;
        if d > 0.0 {
            s += d;
        }
        prev_value = *v;
    }

    s
}

pub(super) fn rollup_zscore_over_time(rfa: &RollupFuncArg) -> f64 {
    // See https://about.gitlab.com/blog/2019/07/23/anomaly-detection-using-prometheus/#using-z-score-for-anomaly-detection
    let scrape_interval = rollup_scrape_interval(rfa);
    let lag = rollup_lag(rfa);
    if scrape_interval.is_nan() || lag.is_nan() || lag > scrape_interval {
        return NAN;
    }
    let d = ROLLUP_LAST(rfa) - rollup_avg(rfa);
    if d == 0.0 {
        return 0.0;
    }
    d / rollup_stddev(rfa)
}

pub(super) fn rollup_first(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    if values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    values[0]
}

pub(crate) fn rollup_default(rfa: &RollupFuncArg) -> f64 {
    let values = rfa.values;
    if values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    // Intentionally do not skip the possible last Prometheus staleness mark.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1526 .
    *values.last().unwrap()
}

pub(super) fn rollup_distinct(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return NAN;
    }

    let mut copy = get_pooled_vec_f64(rfa.values.len());
    copy.extend_from_slice(rfa.values);
    copy.sort_by(|a, b| a.total_cmp(b));
    copy.dedup();

    copy.len() as f64
}

pub(super) fn rollup_fake(_rfa: &RollupFuncArg) -> f64 {
    panic!("BUG: rollup_fake shouldn't be called");
}
