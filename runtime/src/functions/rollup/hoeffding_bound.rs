use crate::functions::arg_parse::get_scalar_param_value;
use crate::functions::rollup::{RollupFuncArg, RollupHandler, RollupHandlerFloat};
use crate::{RuntimeResult};
use crate::types::QueryValue;

pub(super) fn new_rollup_hoeffding_bound_lower(
    args: &[QueryValue],
) -> RuntimeResult<RollupHandler> {
    let phi = get_scalar_param_value(args, 0, "hoeffding_bound_lower", "phi")?;

    let f = RollupHandlerFloat::new(phi, |rfa: &RollupFuncArg, phi: &f64| -> f64 {
        let (bound, avg) = hoeffding_bound_internal(rfa.values, *phi);
        avg - bound
    });

    Ok(RollupHandler::FloatArg(f))
}

pub(super) fn new_rollup_hoeffding_bound_upper(
    args: &[QueryValue],
) -> RuntimeResult<RollupHandler> {
    let phi = get_scalar_param_value(args, 0, "hoeffding_bound_upper", "phi")?;

    let f = RollupHandlerFloat::new(phi, move |rfa: &RollupFuncArg, phi: &f64| -> f64 {
        let (bound, avg) = hoeffding_bound_internal(rfa.values, *phi);
        avg + bound
    });

    Ok(RollupHandler::FloatArg(f))
}

fn hoeffding_bound_internal(values: &[f64], phi: f64) -> (f64, f64) {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    if values.len() == 1 {
        return (0.0, values[0]);
    }

    let (v_avg, v_range) = {
        let mut v_min = values[0];
        let mut v_max = v_min;
        let mut v_sum = 0.0;
        for v in values.iter().copied() {
            if v < v_min {
                v_min = v;
            }
            if v > v_max {
                v_max = v;
            }
            v_sum += v;
        }
        let v_avg = v_sum / values.len() as f64;
        let v_range = v_max - v_min;
        (v_avg, v_range)
    };

    if v_range <= 0.0 {
        return (0.0, v_avg);
    }

    if phi >= 1.0 {
        return (f64::INFINITY, v_avg);
    }

    if phi <= 0.0 {
        return (0.0, v_avg);
    }
    // See https://en.wikipedia.org/wiki/Hoeffding%27s_inequality
    // and https://www.youtube.com/watch?v=6UwcqiNsZ8U&feature=youtu.be&t=1237

    // let bound = v_range * math.Sqrt(math.Log(1 / (1 - phi)) / (2 * values.len()));
    let bound = v_range * ((1.0 / (1.0 - phi)).ln() / (2 * values.len()) as f64).sqrt();
    (bound, v_avg)
}
