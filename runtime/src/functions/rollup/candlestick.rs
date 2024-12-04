use crate::functions::rollup::RollupFuncArg;

/// get_candlestick_values returns a subset of rfa.values suitable for rollup_candlestick
///
/// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309 for details.
fn get_candlestick_values<'a>(rfa: &'a RollupFuncArg<'a>) -> &'a [f64] {
    let curr_timestamp = rfa.curr_timestamp;

    for i in (0..rfa.timestamps.len()).rev() {
        if rfa.timestamps[i] < curr_timestamp {
            return &rfa.values[0..i];
        }
    }

    &[]
}

fn get_first_value_for_candlestick(rfa: &RollupFuncArg) -> f64 {
    if rfa.prev_timestamp + rfa.window >= rfa.curr_timestamp {
        return rfa.prev_value;
    }
    f64::NAN
}

pub(super) fn rollup_open(rfa: &RollupFuncArg) -> f64 {
    let v = get_first_value_for_candlestick(rfa);
    if !v.is_nan() {
        return v;
    }
    let values = get_candlestick_values(rfa);
    if values.is_empty() {
        return f64::NAN;
    }
    values[0]
}

pub(super) fn rollup_close(rfa: &RollupFuncArg) -> f64 {
    let values = get_candlestick_values(rfa);
    if values.is_empty() {
        return get_first_value_for_candlestick(rfa);
    }
    values[values.len() - 1]
}

pub(super) fn rollup_high(rfa: &RollupFuncArg) -> f64 {
    let mut max = get_first_value_for_candlestick(rfa);
    let values = get_candlestick_values(rfa);
    let mut start = 0;
    if max.is_nan() {
        if values.is_empty() {
            return f64::NAN;
        }
        max = values[0];
        start = 1;
    }

    for v in values.iter().skip(start).copied() {
        if v > max {
            max = v;
        }
    }

    max
}

pub(super) fn rollup_low(rfa: &RollupFuncArg) -> f64 {
    let mut min = get_first_value_for_candlestick(rfa);
    let values = get_candlestick_values(rfa);
    let mut start = 0;
    if min.is_nan() {
        if values.is_empty() {
            return f64::NAN;
        }
        min = values[0];
        start = 1;
    }
    for v in values.iter().skip(start).copied() {
        if v < min {
            min = v
        }
    }
    min
}
