use crate::functions::rollup::RollupHandler;
use crate::RuntimeResult;
use crate::types::QueryValue;
use super::RollupFuncArg;

pub(super) fn new_rollup_delta(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_delta))
}

pub(super) fn new_rollup_increase(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_delta))
}

pub(super) fn new_rollup_idelta(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_idelta))
}

pub(super) fn new_rollup_delta_prometheus(_: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_delta_prometheus))
}

pub(crate) fn delta_values(values: &mut [f64]) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from storage.
    if values.is_empty() {
        return;
    }

    let mut prev_delta: f64 = 0.0;
    let mut prev_value = values[0];

    for i in 1..values.len() {
        let v = values[i];
        prev_delta = v - prev_value;
        values[i - 1] = prev_delta;
        prev_value = v;
    }

    values[values.len() - 1] = prev_delta
}

pub(super) fn rollup_delta(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values[0..];
    let mut prev_value = rfa.prev_value;
    if prev_value.is_nan() {
        if values.is_empty() {
            return f64::NAN;
        }
        if !rfa.real_prev_value.is_nan() {
            // Assume that the value didn't change during the current gap.
            // This should fix high delta() and increase() values at the end of gaps.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/894
            return values[values.len() - 1] - rfa.real_prev_value;
        }
        // Assume that the previous non-existing value was 0 only in the following cases:
        //
        // - If the delta with the next value equals to 0.
        //   This is the case for slow-changing counter - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/962
        // - If the first value doesn't exceed too much the delta with the next value.
        //
        // This should prevent from improper increase() results for os-level counters
        // such as cpu time or bytes sent over the network interface.
        // These counters may start long ago before the first value appears in the db.
        //
        // This also should prevent from improper increase() results when a part of label values are changed
        // without counter reset.

        let first_value = values[0];
        let d = if values.len() > 1 {
            values[1] - first_value
        } else if !rfa.real_next_value.is_nan() {
            rfa.real_next_value - first_value
        } else {
            0.0
        };

        if first_value.abs() < 10.0 * (d.abs() + 1.0) {
            prev_value = 0.0;
        } else {
            prev_value = first_value;
            values = &values[1..]
        }
    }
    if values.is_empty() {
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }

    values[values.len() - 1] - prev_value
}

pub(super) fn rollup_delta_prometheus(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let count = rfa.values.len();
    // Just return the difference between the last and the first sample like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1962
    if count < 2 {
        return f64::NAN;
    }
    rfa.values[count - 1] - rfa.values[0]
}

pub(super) fn rollup_idelta(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    if values.is_empty() {
        if rfa.prev_value.is_nan() {
            return f64::NAN;
        }
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    let last_value = rfa.values[rfa.values.len() - 1];
    let values = &values[0..values.len() - 1];
    if values.is_empty() {
        let prev_value = rfa.prev_value;
        if prev_value.is_nan() {
            // Assume that the previous non-existing value was 0.
            return last_value;
        }
        return last_value - prev_value;
    }
    last_value - values[values.len() - 1]
}
