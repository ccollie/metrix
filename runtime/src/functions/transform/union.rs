use ahash::AHashSet;
use crate::execution::{eval_number, EvalConfig};
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult};
use crate::types::{QueryValue, Timeseries };

pub(crate) fn union(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    // we don't use args after this
    let args = std::mem::take(&mut tfa.args);
    handle_union(args, tfa.ec)
}

pub(crate) fn handle_union(
    args: Vec<QueryValue>,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<Timeseries>> {
    if args.is_empty() {
        return eval_number(ec, f64::NAN);
    }

    let len = args[0].len();
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(len);

    if are_all_args_scalar(&args) {
        for mut arg in args.into_iter() {
            match arg {
                QueryValue::Scalar(v) => {
                    let mut ts = eval_number(ec, v)?;
                    rvs.append(&mut ts);
                }
                QueryValue::InstantVector(ref mut v) => {
                    rvs.append(v);
                }
                _ => {
                    return Err(RuntimeError::ArgumentError(
                        "expected scalar".to_string(),
                    ));
                }
            }
        }
        return Ok(rvs);
    }

    let mut m: AHashSet<String> = AHashSet::with_capacity(len);

    fn process_vector(
        v: &mut [Timeseries],
        m: &mut AHashSet<String>,
        rvs: &mut Vec<Timeseries>,
    ) {
        for ts in v.iter_mut() {
            let key = ts.metric_name.to_string();
            if m.insert(key) {
                rvs.push(std::mem::take(ts));
            }
        }
    }

    let mut args = args;
    for arg in args.iter_mut() {
        // done this way to avoid allocating a new vector in the case of a InstantVector
        match arg {
            QueryValue::Scalar(v) => {
                let mut ts = eval_number(ec, *v)?;
                process_vector(&mut ts, &mut m, &mut rvs);
            }
            QueryValue::InstantVector(v) => process_vector(v, &mut m, &mut rvs),
            QueryValue::RangeVector(v) => process_vector(v, &mut m, &mut rvs),
            _ => {
                return Err(RuntimeError::ArgumentError(
                    "expected instant or range vector".to_string(),
                ));
            }
        }
    }

    Ok(rvs)
}

fn are_all_args_scalar(args: &Vec<QueryValue>) -> bool {
    args.iter().all(|arg| {
        match arg {
            QueryValue::Scalar(_) => true,
            QueryValue::InstantVector(v) => {
                if v.len() != 1 {
                    return false;
                }
                let mn = &v[0].metric_name;
                return mn.is_empty()
            },
            _ => false,
        }
    })
}

fn is_scalar_ts(tss: &Vec<Timeseries>) -> bool {
    if tss.len() != 1 {
        return false;
    }
    let mn = &tss[0].metric_name;
    mn.is_empty()
}