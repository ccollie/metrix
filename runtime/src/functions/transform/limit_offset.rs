use crate::execution::remove_empty_series;
use crate::functions::arg_parse::{get_int_arg, get_series_arg};
use crate::functions::transform::TransformFuncArg;
use crate::{types::Timeseries, RuntimeError, RuntimeResult};

pub(crate) fn limit_offset(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let limit = tfa.args[0].get_int()? as usize;
    let offset = match get_int_arg(&tfa.args, 1) {
        Err(_) => {
            return Err(RuntimeError::from("cannot obtain offset arg"));
        }
        Ok(v) => v as usize,
    };

    let mut rvs = get_series_arg(&tfa.args, 2, tfa.ec)?;
    // remove_empty_series so offset will be calculated after empty series
    // were filtered out.
    remove_empty_series(&mut rvs);

    if rvs.len() >= offset {
        if offset > 0 {
            rvs.drain(0..offset);
        }
    } else {
        return Ok(vec![])
    }
    if rvs.len() > limit {
        rvs.truncate(limit);
    }

    Ok(std::mem::take(&mut rvs))
}
