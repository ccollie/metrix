use crate::common::math::linear_regression;
use crate::functions::rollup::{RollupFuncArg, RollupHandler};
use crate::{RuntimeResult};
use crate::types::{QueryValue, Timestamp};

#[inline]
pub(super) fn new_rollup_rate(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_deriv_fast))
}

#[inline]
pub(super) fn new_rollup_deriv(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_deriv_slow))
}

#[inline]
pub(super) fn new_rollup_deriv_fast(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_deriv_fast))
}

#[inline]
pub(super) fn new_rollup_ideriv(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_ideriv))
}

#[inline]
pub(super) fn new_rollup_irate(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_ideriv))
}

pub(super) fn deriv_values(values: &mut [f64], timestamps: &[Timestamp]) {
    // There is no need in handling NaNs here, since they are impossible on values from storage.
    if values.is_empty() {
        return;
    }
    let mut prev_deriv: f64 = 0.0;
    let mut prev_value = values[0];
    let mut prev_ts = timestamps[0];

    let mut j: usize = 0;
    for i in 1..values.len() {
        let v = values[i];
        let ts = timestamps[i];
        if ts == prev_ts {
            // Use the previous value for duplicate timestamps.
            values[j] = prev_deriv;
            j += 1;
            continue;
        }
        let dt = (ts - prev_ts) as f64 / 1e3_f64;
        prev_deriv = (v - prev_value) / dt;
        values[j] = prev_deriv;
        prev_value = v;
        prev_ts = ts;
        j += 1;
    }

    values[values.len() - 1] = prev_deriv
}

pub(super) fn rollup_deriv_slow(rfa: &RollupFuncArg) -> f64 {
    // Use linear regression like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/73
    let (_, k) = linear_regression(rfa.values, rfa.timestamps, rfa.curr_timestamp);
    k
}

pub(super) fn rollup_deriv_fast(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    let timestamps = rfa.timestamps;
    let mut prev_value = rfa.prev_value;
    let mut prev_timestamp = rfa.prev_timestamp;
    if prev_value.is_nan() {
        if values.is_empty() || values.len() == 1 {
            return f64::NAN;
        }
        prev_value = values[0];
        prev_timestamp = timestamps[0];
    } else if values.is_empty() {
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    let v_end = values[values.len() - 1];
    let t_end = timestamps[timestamps.len() - 1];
    let dv = v_end - prev_value;
    let dt = (t_end - prev_timestamp) as f64 / 1e3_f64;
    dv / dt
}

pub(super) fn rollup_ideriv(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    let timestamps = rfa.timestamps;
    let mut count = rfa.values.len();
    if count < 2 {
        if count == 0 {
            return f64::NAN;
        }
        if rfa.prev_value.is_nan() {
            // It is impossible to determine the duration during which the value changed
            // from 0 to the current value.
            // The following attempts didn't work well:
            // - using scrape interval as the duration. It fails on Prometheus restarts when it
            //   skips scraping for the counter. This results in too high rate() value for the first point
            //   after Prometheus restarts.
            // - using window or step as the duration. It results in too small rate() values for the first
            //   points of time series.
            //
            // So just return NAN
            return f64::NAN;
        }
        return (values[0] - rfa.prev_value)
            / ((timestamps[0] - rfa.prev_timestamp) as f64 / 1e3_f64);
    }
    let v_end = values[values.len() - 1];
    let t_end = timestamps[timestamps.len() - 1];

    let values = &values[0..count - 1];
    let mut timestamps = &timestamps[0..timestamps.len() - 1];

    // Skip data points with duplicate timestamps.
    while !timestamps.is_empty() && timestamps[timestamps.len() - 1] >= t_end {
        timestamps = &timestamps[0..timestamps.len() - 1];
    }
    count = timestamps.len();

    let t_start: i64;
    let v_start: f64;
    if count == 0 {
        if rfa.prev_value.is_nan() {
            return 0.0;
        }
        t_start = rfa.prev_timestamp;
        v_start = rfa.prev_value;
    } else {
        t_start = timestamps[count - 1];
        v_start = values[count - 1];
    }
    let dv = v_end - v_start;
    let dt = t_end - t_start;
    dv / (dt as f64 / 1e3_f64)
}
