use crate::functions::arg_parse::get_float_arg;
use crate::functions::rollup::{RollupFuncArg, RollupHandler, RollupHandlerFloat};
use crate::{RuntimeError, RuntimeResult};
use std::io::Cursor;
use std::io::Write;
use crate::types::QueryValue;

type FloatPredicate = fn(f64, f64) -> bool;

#[inline]
fn less_or_equal(x: f64, y: f64) -> bool {
    x.le(&y)
}

#[inline]
fn greater(x: f64, y: f64) -> bool {
    x > y
}

#[inline]
fn equal(x: f64, y: f64) -> bool {
    x == y
}

#[inline]
fn not_equal(x: f64, y: f64) -> bool {
    x != y
}

fn count_filtered(values: &[f64], limit: f64, pred: FloatPredicate) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let res = values.iter().filter(|v| pred(**v, limit)).count() as f64;
    res
}

fn share_filtered(values: &[f64], limit: f64, pred: FloatPredicate) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let len = values.len();
    let n = count_filtered(values, limit, pred);
    n / len as f64
}

fn sum_filtered(values: &[f64], limit: f64, pred: FloatPredicate) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().filter(|x| pred(**x, limit)).sum()
}

fn get_limit(args: &[QueryValue], func_name: &str, param_name: &str) -> RuntimeResult<f64> {
    get_float_arg(args, 1, None).map_err(|_| {
        RuntimeError::ArgumentError(format!("expecting scalar as {param_name} arg to {func_name}()"))
    })
}

macro_rules! make_count_fn {
    ( $name: ident, $func_name: tt, $param_name: tt, $predicate_fn: expr ) => {
        pub(super) fn $name(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
            let limit = get_limit(args, $func_name, $param_name)?;
            let handler =
                RollupHandlerFloat::new(limit, |rfa: &RollupFuncArg, limit: &f64| -> f64 {
                    count_filtered(rfa.values, *limit, $predicate_fn)
                });
            Ok(RollupHandler::FloatArg(handler))
        }
    };
}

make_count_fn!(new_rollup_count_le, "count_le_over_time", "le", less_or_equal);
make_count_fn!(new_rollup_count_gt, "count_gt_over_time", "gt", greater);
make_count_fn!(new_rollup_count_eq, "count_eq_over_time", "eq", equal);
make_count_fn!(new_rollup_count_ne, "count_ne_over_time", "ne", not_equal);

macro_rules! make_share_fn {
    ( $name: ident, $func_name: tt, $param_name: tt, $predicate_fn: expr ) => {
        pub(super) fn $name(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
            let limit = get_limit(args, $func_name, $param_name)?;
            let handler = |rfa: &RollupFuncArg, limit: &f64| -> f64 {
               share_filtered(rfa.values, *limit, $predicate_fn)
            };
            Ok(RollupHandler::float_arg(limit, handler))
        }
    };
}

make_share_fn!(new_rollup_share_le, "share_le_over_time", "le",less_or_equal);
make_share_fn!(new_rollup_share_gt, "share_gt_over_time", "gt", greater);
make_share_fn!(new_rollup_share_eq, "share_eq_over_time", "eq", equal);

macro_rules! make_sum_fn {
    ( $name: ident, $func_name: tt, $param_name: tt, $predicate_fn: expr ) => {
        pub(super) fn $name(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
            let limit = get_limit(args, $func_name, $param_name)?;
            let handler = |rfa: &RollupFuncArg, limit: &f64| -> f64 {
                sum_filtered(rfa.values, *limit, $predicate_fn)
            };
            Ok(RollupHandler::float_arg(limit, handler))
        }
    };
}

make_sum_fn!(new_rollup_sum_eq, "sum_eq_over_time", "eq", equal);
make_sum_fn!(new_rollup_sum_gt, "sum_gt_over_time", "gt", greater);
make_sum_fn!(new_rollup_sum_le, "sum_le_over_time", "le", less_or_equal);

pub(super) fn new_rollup_count_values(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    if args.len() != 2 {
        return Err(RuntimeError::ArgumentError(
            "expecting 2 args to count_values_over_time()".to_string(),
        ));
    }

    match args[1] {
        QueryValue::InstantVector(ref tss) => tss,
        _ => {
            return Err(RuntimeError::ArgumentError(
                "expecting instant vector as the second arg to count_values_over_time()"
                    .to_string(),
            ));
        }
    };

    let label_name = match args[0] {
        QueryValue::String(ref s) => s.clone(),
        _ => {
            return Err(RuntimeError::ArgumentError(
                "expecting string as the first arg to count_values_over_time()".to_string(),
            ));
        }
    };

    let f = move |rfa: &RollupFuncArg| -> f64 {
        let binding = rfa.get_tsm();
        let tsm = binding.as_ref();
        let idx = rfa.idx;

        // https://stackoverflow.com/questions/73117416/how-do-i-print-u64-to-a-buffer-on-stack-in-rust
        let cursor = Cursor::new([0u8; 40]);
        let mut buf: Vec<u8> = Vec::with_capacity(40);

        // Note: the code below may create very big number of time series
        // if the number of unique values in rfa.values is big.
        let mut label_value = String::with_capacity(40);
        for v in rfa.values {
            // Note: write! does not panic, so unwrap is safe here
            write!(buf, "{v}").unwrap();
            let pos = cursor.position() as usize;
            let buffer = &cursor.get_ref()[..pos];

            label_value.push_str(std::str::from_utf8(buffer).unwrap());
            tsm.with_series(label_name.as_str(), label_value.as_str(), |ts| {
                let mut count = ts.values[idx];
                if count.is_nan() {
                    count = 1.0;
                } else {
                    count += 1.0;
                }
                ts.values[idx] = count;
            });

            label_value.clear();
        }
        f64::NAN
    };

    let handler = RollupHandler::General(Box::new(f));
    Ok(handler)
}
