use num_traits::FloatConst;

use crate::execution::eval_number;
use crate::functions::transform::{transform_series, TransformFuncArg};
use crate::{RuntimeResult};
use crate::types::Timeseries;

macro_rules! math_fn {
    ($name: ident, $func: expr) => {
        pub(crate) fn $name(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
            math_func_impl(tfa, $func)
        }
    };
}

math_fn!(abs, |x: f64| x.abs());
math_fn!(acos, |x: f64| x.acos());
math_fn!(acosh, |x: f64| x.acosh());
math_fn!(asin, |x: f64| x.asin());
math_fn!(asinh, |x: f64| x.asinh());
math_fn!(atan, |x: f64| x.atan());
math_fn!(atanh, |x: f64| x.atanh());
math_fn!(ceil, |x: f64| x.ceil());
math_fn!(cos, |x: f64| x.cos());
math_fn!(cosh, |x: f64| x.cosh());
math_fn!(deg, |x: f64| x.to_degrees());
math_fn!(exp, |x: f64| x.exp());
math_fn!(floor, |x: f64| x.floor());
math_fn!(ln, |x: f64| x.ln());
math_fn!(log2, |x: f64| x.log2());
math_fn!(log10, |x: f64| x.log10());
math_fn!(rad, |x: f64| x.to_radians());
math_fn!(sin, |x: f64| x.sin());
math_fn!(sinh, |x: f64| x.sinh());
math_fn!(sqrt, |x: f64| x.sqrt());
math_fn!(tan, |x: f64| x.tan());
math_fn!(tanh, |x: f64| x.tanh());

fn math_func_impl(
    tfa: &mut TransformFuncArg,
    op: fn(f64) -> f64,
) -> RuntimeResult<Vec<Timeseries>> {
    let tfe = |values: &mut [f64]| {
        for value in values.iter_mut() {
            *value = op(*value)
        }
    };
    transform_series(tfa, tfe)
}

pub(crate) fn transform_pi(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    eval_number(tfa.ec, f64::PI())
}

pub(crate) fn sgn(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tf = |values: &mut [f64]| {
        let mut zero = 0.0f64;
        for v in values {
            match v.total_cmp(&&mut zero) {
                std::cmp::Ordering::Less => *v = -1.0f64,
                std::cmp::Ordering::Equal => *v = 0.0f64,
                std::cmp::Ordering::Greater => *v = 1.0f64,
            }
        }
    };

    transform_series(tfa, tf)
}
