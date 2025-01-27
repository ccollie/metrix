use rand::distributions::Standard;
use rand::prelude::{Rng, StdRng, SeedableRng, Distribution};
use rand::rngs::ThreadRng;
use rand_distr::{Exp1, StandardNormal};
use crate::execution::eval_number;
use crate::functions::transform::TransformFuncArg;
use crate::types::Timeseries;
use crate::{RuntimeError, RuntimeResult};

fn create_rng(tfa: &mut TransformFuncArg) -> RuntimeResult<StdRng> {
    if tfa.args.len() == 1 {
        return match tfa.args[0].get_int() {
            Err(e) => Err(e),
            Ok(val) => match u64::try_from(val) {
                Err(_) => Err(RuntimeError::ArgumentError(
                    format!("invalid rand seed {}", val).to_string(),
                )),
                Ok(seed) => Ok(StdRng::seed_from_u64(seed)),
            },
        };
    }
    let rng = ThreadRng::default();
    StdRng::from_rng(rng)
        .map_err(|_| RuntimeError::ArgumentError("unable to create rng".to_string()))
}

fn rand_fn_inner<D>(tfa: &mut TransformFuncArg, distro: D) -> RuntimeResult<Vec<Timeseries>>
where
    D: Distribution<f64>,
{
    let rng: StdRng = create_rng(tfa)?;
    let mut tss = eval_number(tfa.ec, 0.0)?;
    let randos = rng.sample_iter(distro);
    for (value, rand_num) in tss[0].values.iter_mut().zip(randos) {
        *value = rand_num;
    }
    Ok(tss)
}

pub(crate) fn rand(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    rand_fn_inner(tfa, Standard)
}

pub(crate) fn rand_norm(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    rand_fn_inner(tfa, StandardNormal)
}

pub(crate) fn rand_exp(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    rand_fn_inner(tfa, Exp1)
}
