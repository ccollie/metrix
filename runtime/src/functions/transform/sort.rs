use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::types::Timeseries;
use crate::{RuntimeError, RuntimeResult};
use rayon::prelude::ParallelSliceMut;
use std::cmp::Ordering;

/// The threshold for switching to a parallel sort implementation.
const THREAD_SORT_THRESHOLD: usize = 6;

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
            |a: &String, b: &String| compare_string_alphanumeric(b, a)
        } else {
            |a: &String, b: &String| compare_string_alphanumeric(a, b)
        };

        for label in labels.iter() {
            let a = a.metric_name.label_value(label).unwrap_or(EMPTY_STRING_REF);
            let b = b.metric_name.label_value(label).unwrap_or(EMPTY_STRING_REF);
            if a == b {
                continue;
            }
            let order = comparator(a, b);
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

fn compare_string_alphanumeric(a: &str, b: &str) -> Ordering {
    let mut a = a;
    let mut b = b;

    loop {
        if a == b {
            return Ordering::Equal;
        }
        if a.is_empty() || b.is_empty() {
            return a.len().cmp(&b.len());
        }

        let mut a_prefix = get_num_prefix(a);
        let mut b_prefix = get_num_prefix(b);

        a = &a[a_prefix.len()..];
        b = &b[b_prefix.len()..];

        match (a_prefix.is_empty(), b_prefix.is_empty()) {
            (true, true) => {},
            (true, false) => return Ordering::Greater,
            (false, true) => return Ordering::Less,
            (false, false) => {
                let a_num = a_prefix.parse::<f64>().unwrap_or(0.0);
                let b_num = b_prefix.parse::<f64>().unwrap_or(0.0);
                if a_num != b_num {
                    return a_num.partial_cmp(&b_num).unwrap();
                }
            }
        }

        a_prefix = get_non_num_prefix(a);
        b_prefix = get_non_num_prefix(b);
        if a_prefix != b_prefix {
            return a_prefix.cmp(b_prefix);
        }

        a = &a[a_prefix.len()..];
        b = &b[b_prefix.len()..];
    }
}

fn get_num_prefix(s: &str) -> &str {
    let mut i = 0;
    let mut iter = s.chars().peekable();
    match iter.peek() {
        Some(&'-') | Some(&'+') => {
            iter.next();
            i += 1
        },
        _ => {}
    }

    let mut has_num = false;
    let mut has_dot = false;
    for ch in iter {
        if !ch.is_digit(10) {
            if !has_dot && ch == '.' {
                has_dot = true;
                i += 1;
                continue;
            }
            if !has_num {
                return "";
            }
            return &s[..i];
        }
        has_num = true;
        i += 1;
    }
    if !has_num {
        return "";
    }
    &s[..i]
}

fn get_non_num_prefix(s: &str) -> &str {
    for (i, ch) in s.chars().enumerate() {
        if ch.is_digit(10) {
            return &s[..i];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_get_num_prefix(s: &str, prefix_expected: &str) {
        let prefix = get_num_prefix(s);
        assert_eq!(prefix, prefix_expected, "unexpected get_num_prefix({:?}): got {:?}; want {:?}", s, prefix, prefix_expected);
        if !prefix.is_empty() {
            assert!(prefix.parse::<f64>().is_ok(), "cannot parse num {:?}", prefix);
        }
    }

    #[test]
    fn test_get_num_prefix_cases() {
        test_get_num_prefix("", "");
        test_get_num_prefix("foo", "");
        test_get_num_prefix("-", "");
        test_get_num_prefix(".", "");
        test_get_num_prefix("-.", "");
        test_get_num_prefix("+..", "");
        test_get_num_prefix("1", "1");
        test_get_num_prefix("12", "12");
        test_get_num_prefix("1foo", "1");
        test_get_num_prefix("-123", "-123");
        test_get_num_prefix("-123bar", "-123");
        test_get_num_prefix("+123", "+123");
        test_get_num_prefix("+123.", "+123.");
        test_get_num_prefix("+123..", "+123.");
        test_get_num_prefix("+123.-", "+123.");
        test_get_num_prefix("12.34..", "12.34");
        test_get_num_prefix("-12.34..", "-12.34");
        test_get_num_prefix("-12.-34..", "-12.");
    }


    #[test]
    fn test_numeric_cmp() {
        fn test_ordering(a: &str, b: &str, want: Ordering) {
            let got = compare_string_alphanumeric(a, b);
            assert_eq!(got, want, "unexpected numeric_less({:?}, {:?}): got {:?}; want {:?}", a, b, got, want);
        }

        // empty strings
        test_ordering("", "", Ordering::Equal);
        test_ordering("", "321", Ordering::Less);
        test_ordering("321", "", Ordering::Greater);
        test_ordering("", "abc", Ordering::Less);
        test_ordering("abc", "", Ordering::Greater);
        test_ordering("foo", "123", Ordering::Greater);
        test_ordering("123", "foo", Ordering::Less);
        // same length numbers
        test_ordering("123", "321", Ordering::Less);
        test_ordering("321", "123", Ordering::Greater);
        test_ordering("123", "123", Ordering::Equal);
        // same length strings
        test_ordering("a", "b", Ordering::Less);
        test_ordering("b", "a", Ordering::Greater);
        test_ordering("a", "a", Ordering::Equal);
        // identical string prefix
        test_ordering("foo123", "foo", Ordering::Greater);
        test_ordering("foo", "foo123", Ordering::Less);
        test_ordering("foo", "foo", Ordering::Equal);
        // identical num prefix
        test_ordering("123foo", "123bar", Ordering::Greater);
        test_ordering("123bar", "123foo", Ordering::Less);
        test_ordering("123bar", "123bar", Ordering::Equal);
        // numbers with special chars
        test_ordering("1:0:0", "1:0:2", Ordering::Less);
        // numbers with special chars and different number rank
        test_ordering("1:0:15", "1:0:2", Ordering::Greater);
        // multiple zeroes
        test_ordering("0", "00", Ordering::Equal);
        // only chars
        test_ordering("aa", "ab", Ordering::Less);
        // strings with different lengths
        test_ordering("ab", "abc", Ordering::Less);
        // multiple zeroes after equal char
        test_ordering("a0001", "a0000001", Ordering::Equal);
        // short first string with numbers and highest rank
        test_ordering("a10", "abcdefgh2", Ordering::Less);
        // less as second string
        test_ordering("a1b", "a01b", Ordering::Equal);
        // equal strings by length with different number rank
        test_ordering("a001b01", "a01b001", Ordering::Equal);
        // different numbers rank
        test_ordering("a01b001", "a001b01", Ordering::Equal);
        // highest char and number
        test_ordering("a1", "a1x", Ordering::Less);
        // highest number reverse chars
        test_ordering("1b", "1ax", Ordering::Greater);
        // numbers with leading zero
        test_ordering("082", "83", Ordering::Less);
        // numbers with leading zero and chars
        test_ordering("083a", "9a", Ordering::Greater);
        test_ordering("083a", "94a", Ordering::Less);
        // negative number
        test_ordering("-123", "123", Ordering::Less);
        test_ordering("-123", "+123", Ordering::Less);
        test_ordering("-123", "-123", Ordering::Equal);
        test_ordering("123", "-123", Ordering::Greater);
        // fractional number
        test_ordering("12.9", "12.56", Ordering::Greater);
        test_ordering("12.56", "12.9", Ordering::Less);
        test_ordering("12.9", "12.9", Ordering::Equal);
    }
}