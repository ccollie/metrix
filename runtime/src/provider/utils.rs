use std::borrow::Cow;

use metricsql_parser::ast::Expr;
use metricsql_parser::label::Matchers;
use metricsql_parser::parse;

use crate::runtime_error::{RuntimeError, RuntimeResult};

pub(crate) fn is_empty_extra_matchers(matchers: &Option<Matchers>) -> bool {
    if let Some(matchers) = matchers {
        return matchers.is_empty();
    }
    true
}

pub(crate) fn join_matchers_with_extra_filters<'a>(
    src: &'a Matchers,
    etfs: &'a Option<Matchers>,
) -> Cow<'a, Matchers> {
    if src.is_empty() {
        if let Some(etfs) = etfs {
            return Cow::Borrowed::<'a>(etfs);
        }
        return Cow::Borrowed::<'a>(src);
    }

    if let Some(etfs) = etfs {
        if etfs.is_empty() {
            return Cow::Borrowed::<'a>(src);
        }
        let mut dst = src.clone();

        if !etfs.or_matchers.is_empty() {
            if !dst.matchers.is_empty() {
                dst.or_matchers.push(std::mem::take(&mut dst.matchers));
            }
            etfs.or_matchers.iter().for_each(|m| {
                dst.or_matchers.push(m.clone());
            });
        }
        if !etfs.matchers.is_empty() {
            if !dst.matchers.is_empty() {
                dst.or_matchers.push(std::mem::take(&mut dst.matchers));
            }
            dst.or_matchers.push(etfs.matchers.clone());
        }
        return Cow::Owned::<'a>(dst);
    }
    Cow::Borrowed::<'a>(src)
}

pub(crate) fn join_matchers_with_extra_filters_owned(
    src: &Matchers,
    etfs: &Option<Matchers>,
) -> Matchers {
    if src.is_empty() {
        if let Some(etfs) = etfs {
            return etfs.clone();
        }
        return src.clone();
    }

    if let Some(etfs) = etfs {
        if etfs.is_empty() {
            return src.clone();
        }
        let mut dst = src.clone();

        if !etfs.or_matchers.is_empty() {
            if !dst.matchers.is_empty() {
                dst.or_matchers.push(std::mem::take(&mut dst.matchers));
            }
            etfs.or_matchers.iter().for_each(|m| {
                dst.or_matchers.push(m.clone());
            });
        }
        if !etfs.matchers.is_empty() {
            if !dst.matchers.is_empty() {
                dst.or_matchers.push(std::mem::take(&mut dst.matchers));
            }
            dst.or_matchers.push(etfs.matchers.clone());
        }
        return dst;
    }
    src.clone()
}

/// parse_metric_selector parses s containing PromQL metric selector and returns the corresponding
/// LabelFilters.
pub fn parse_metric_selector(s: &str) -> RuntimeResult<Matchers> {
    match parse(s) {
        Ok(expr) => match expr {
            Expr::MetricExpression(me) => {
                if me.is_empty() {
                    let msg = "labelFilters cannot be empty";
                    return Err(RuntimeError::from(msg));
                }
                Ok(me.matchers)
            }
            _ => {
                let msg = format!("expecting metric selector; got {}", expr);
                Err(RuntimeError::from(msg))
            }
        },
        Err(err) => Err(RuntimeError::ParseError(err)),
    }
}
