use std::cmp::Ordering;
use rayon::prelude::ParallelSliceMut;
use crate::common::strings::compare_str_alphanumeric;
use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult};
use crate::types::Timeseries;

/// The threshold for switching to a parallel sort implementation.
const THREAD_SORT_THRESHOLD: usize = 4;

pub(crate) fn sort(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_sort_impl(tfa, false)
}

pub(crate) fn sort_desc(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_sort_impl(tfa, true)
}

pub(crate) fn sort_alpha_numeric(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    label_alpha_numeric_sort_impl(tfa, false)
}

pub(crate) fn sort_alpha_numeric_desc(
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    label_alpha_numeric_sort_impl(tfa, true)
}

pub(crate) fn sort_by_label(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    sort_by_label_impl(tfa, false)
}

pub(crate) fn sort_by_label_desc(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    sort_by_label_impl(tfa, true)
}


fn transform_sort_impl(tfa: &TransformFuncArg, is_desc: bool) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

    fn sort(a: &Timeseries, b: &Timeseries, is_desc: bool) -> Ordering {
        let iter_a = a.values.iter().rev().copied();
        let iter_b = b.values.iter().rev().copied();
        if is_desc {
            iter_cmp_f64_desc(iter_a, iter_b)
        } else {
            iter_cmp_f64(iter_a, iter_b)
        }
    }

    if series.len() >= THREAD_SORT_THRESHOLD {
        series.par_sort_by(|a, b| sort(a, b, is_desc));
    } else {
        series.sort_by(|a, b| sort(a, b, is_desc));
    }

    // println!("Sorted {:?}", series);

    Ok(std::mem::take(&mut series))
}

const EMPTY_STRING: String = String::new();
const EMPTY_STRING_REF: &String = &EMPTY_STRING;

fn sort_by_label_impl(tfa: &mut TransformFuncArg, is_desc: bool) -> RuntimeResult<Vec<Timeseries>> {
    let mut labels: Vec<String> = Vec::with_capacity(tfa.args.len() - 1);
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

    for arg in tfa.args.iter().skip(1) {
        let label = arg.get_string()?;
        labels.push(label);
    }

    fn sort(a: &Timeseries, b: &Timeseries, labels: &[String], is_desc: bool) -> Ordering {
        for label in labels.iter() {
            let a = a.metric_name.label_value(label);
            let b = b.metric_name.label_value(label);
            let order = if is_desc {
                compare_string(b, a)
            } else {
                compare_string(a, b)
            };
            if order != Ordering::Equal {
                return order;
            }
        }
        Ordering::Equal
    }

    if series.len() >= THREAD_SORT_THRESHOLD {
        series.par_sort_by(|first, second| sort(first, second, &labels, is_desc));
    } else {
        series.sort_by(|first, second| sort(first, second, &labels, is_desc));
    }

    Ok(series)
}

fn label_alpha_numeric_sort_impl(
    tfa: &mut TransformFuncArg,
    is_desc: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut labels: Vec<String> = Vec::with_capacity(tfa.args.len() - 1);
    for (i, arg) in tfa.args.iter().skip(1).enumerate() {
        let label = arg.get_string().map_err(|err| {
            RuntimeError::ArgumentError(format!(
                "cannot parse label {} for sorting: {:?}",
                i + 1,
                err
            ))
        })?;
        labels.push(label);
    }

    fn sort(a: &Timeseries, b: &Timeseries, labels: &[String], is_desc: bool) -> Ordering {
        let comparator = if is_desc {
            |a: &String, b: &String| compare_str_alphanumeric(b, a)
        } else {
            |a: &String, b: &String| compare_str_alphanumeric(a, b)
        };

        for label in labels.iter() {
            let a = a.metric_name.label_value(label);
            let b = b.metric_name.label_value(label);
            let order = comparator(a.unwrap_or(EMPTY_STRING_REF), b.unwrap_or(EMPTY_STRING_REF));
            if order != Ordering::Equal {
                return order;
            }
        }
        Ordering::Equal
    }

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

    if series.len() >= THREAD_SORT_THRESHOLD {
        series.par_sort_by(|first, second| sort(first, second, &labels, is_desc));
    } else {
        series.sort_by(|first, second| sort(first, second, &labels, is_desc));
    }

    Ok(series)
}

/// Order `a` and `b` lexicographically using `Ord`
#[inline]
pub fn iter_cmp<A, L, R>(mut a: L, mut b: R) -> Ordering
where
    A: Ord,
    L: Iterator<Item = A>,
    R: Iterator<Item = A>,
{
    loop {
        match (a.next(), b.next()) {
            (None, None) => return Ordering::Equal,
            (None, _) => return Ordering::Less,
            (_, None) => return Ordering::Greater,
            (Some(x), Some(y)) => match x.cmp(&y) {
                Ordering::Equal => (),
                non_eq => return non_eq,
            },
        }
    }
}

#[inline]
fn reverse_ordering(order: Ordering) -> Ordering {
    match order {
        Ordering::Less => Ordering::Greater,
        Ordering::Equal => Ordering::Equal,
        Ordering::Greater => Ordering::Less,
    }
}

#[inline]
pub fn iter_cmp_f64<L, R>(mut a: L, mut b: R) -> Ordering
where
    L: Iterator<Item = f64>,
    R: Iterator<Item = f64>,
{
    loop {
        match (a.next(), b.next()) {
            (None, None) => return Ordering::Equal,
            (None, _) => return Ordering::Less,
            (_, None) => return Ordering::Greater,
            (Some(x), Some(y)) => match compare_float(x, y) {
                Ordering::Equal => (),
                non_eq => return non_eq,
            },
        }
    }
}

#[inline]
pub fn iter_cmp_f64_desc<L, R>(a: L, b: R) -> Ordering
where
    L: Iterator<Item = f64>,
    R: Iterator<Item = f64>,
{
    let order = iter_cmp_f64(a, b);
    reverse_ordering(order)
}

#[inline]
fn compare_float(a: f64, b: f64) -> Ordering {
    if !a.is_nan() {
        if b.is_nan() {
            return Ordering::Greater;
        }
        let order = a.total_cmp(&b);
        if order != Ordering::Equal {
            return order
        }
    } else if !b.is_nan() {
        return Ordering::Less;
    }

    Ordering::Equal
}

#[inline]
fn compare_string(a: Option<&String>, b: Option<&String>) -> Ordering {
    let a = a.unwrap_or(EMPTY_STRING_REF);
    let b = b.unwrap_or(EMPTY_STRING_REF);
    a.cmp(b)
}
