use metricsql_parser::parse_number;

use crate::execution::eval_number;
use crate::functions::transform::utils::expect_transform_args_num;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeResult};
use crate::types::{QueryValue, Timeseries};

pub(crate) fn scalar(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    let mut arg = tfa.args.remove(0);
    match arg {
        // Verify whether the arg is a string.
        // Then try converting the string to number.
        QueryValue::String(s) => {
            let n = parse_number(&s).unwrap_or(f64::NAN);
            eval_number(tfa.ec, n)
        }
        QueryValue::Scalar(f) => eval_number(tfa.ec, f),
        QueryValue::InstantVector(ref mut iv) => {
            if iv.len() != 1 {
                eval_number(tfa.ec, f64::NAN)
            } else {
                let arg = iv.get_mut(0).unwrap();
                arg.metric_name.reset();
                Ok(std::mem::take(iv))
            }
        }
        _ => eval_number(tfa.ec, f64::NAN),
    }
}
