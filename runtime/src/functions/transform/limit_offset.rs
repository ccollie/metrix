use tracing::debug;
use crate::execution::remove_empty_series;
use crate::functions::arg_parse::{get_int_arg, get_series_arg};
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult, types::Timeseries};

pub(crate) fn limit_offset(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let limit = match tfa.args[0].get_int() {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "cannot obtain limit arg: {e:?}"
            )));
        }
        Ok(l) => l as usize,
    };

    let offset = match get_int_arg(&tfa.args, 1) {
        Err(_) => {
            return Err(RuntimeError::from("cannot obtain offset arg"));
        }
        Ok(v) => v as usize,
    };

    let mut rvs = get_series_arg(&tfa.args, 2, tfa.ec)?;
    println!("limit_offset: limit={}, offset={}, rvs.len()={}", limit, offset, rvs.len());
    println!("rvs={:?}", rvs);
    // remove_empty_series so offset will be calculated after empty series
    // were filtered out.
    remove_empty_series(&mut rvs);

    let slice = if rvs.len() > offset {
        &mut rvs[offset..]
    } else {
        &mut []
    };
    if rvs.len() >= offset {
        rvs.drain(0..offset);
        println!("drained rvs={:?}", rvs);
    } else {
        return Ok(vec![])
    }
    if rvs.len() > limit {
        rvs.truncate(limit);
        println!("truncated rvs={:?}", rvs);
    }

    Ok(std::mem::take(&mut rvs))
}
