use crate::functions::arg_parse::get_scalar_param_value;
use crate::functions::rollup::{RollupFuncArg, RollupHandler, RollupHandlerFloat};
use crate::{RuntimeResult};
use crate::types::QueryValue;

pub(super) fn new_rollup_duration_over_time(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    let max_interval = get_scalar_param_value(args, 0, "duration_over_time", "max_interval")?;

    let handler =
        RollupHandlerFloat::new(max_interval, |rfa: &RollupFuncArg, interval: &f64| -> f64 {
            // There is no need in handling NaNs here, since they must be cleaned up
            // before calling rollup fns.
            duration_over_time(rfa.timestamps, *interval)
        });

    Ok(RollupHandler::FloatArg(handler))
}

fn duration_over_time(timestamps: &[i64], max_interval: f64) -> f64 {
    if timestamps.is_empty() {
        return f64::NAN;
    }
    let mut t_prev = timestamps[0];
    let mut d_sum: i64 = 0;
    let d_max = (max_interval * 1000_f64) as i64;
    for t in timestamps.iter() {
        let d = t - t_prev;
        if d <= d_max {
            d_sum += d;
        }
        t_prev = *t
    }

    d_sum as f64 / 1000_f64
}
