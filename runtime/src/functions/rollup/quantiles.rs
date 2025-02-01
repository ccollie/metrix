use std::ops::DerefMut;

use metricsql_common::pool::get_pooled_vec_f64_filled;

use crate::common::math::{quantile, quantiles};
use crate::functions::arg_parse::{get_scalar_param_value, get_string_param_value};
use crate::functions::rollup::{RollupFuncArg, RollupHandler, RollupHandlerFloat};
use crate::{RuntimeResult};
use crate::types::QueryValue;


pub(super) fn new_rollup_quantiles(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    let phi_label = get_string_param_value(args, 0, "quantiles", "phi_label")?;
    let cap = args.len() - 1;

    let mut phis = Vec::with_capacity(cap);
    // todo: smallvec ??
    let mut phi_labels: Vec<String> = Vec::with_capacity(cap);

    for i in 1..args.len() {
        // unwrap should be safe, since parameters types are checked before calling the function
        let v = get_scalar_param_value(args, i, "quantiles", "phi")?;
        phis.push(v);
        phi_labels.push(format!("{}", v));
    }

    let f = Box::new(move |rfa: &RollupFuncArg| -> f64 {
        quantiles_impl(rfa, &phi_label, &phis, &phi_labels)
    });

    Ok(RollupHandler::General(f))
}

pub(super) fn new_rollup_quantile(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    let phi = get_scalar_param_value(args, 0, "quantile_over_time", "phi")?;

    let rf = RollupHandlerFloat::new(phi, |rfa: &RollupFuncArg, phi: &f64| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        quantile(*phi, rfa.values)
    });

    Ok(RollupHandler::FloatArg(rf))
}

#[inline]
fn quantiles_impl(rfa: &RollupFuncArg, label: &str, phis: &[f64], phi_labels: &[String]) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return rfa.prev_value;
    }
    if rfa.values.len() == 1 {
        // Fast path - only a single value.
        return rfa.values[0];
    }
    // tinyvec ?
    let mut qs = get_pooled_vec_f64_filled(phis.len(), 0f64);
    quantiles(qs.deref_mut(), phis, rfa.values);
    let map = rfa.get_tsm();
    map.set_timeseries_values(label, phi_labels, &qs, rfa.idx);

    f64::NAN
}
