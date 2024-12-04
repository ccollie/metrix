use chrono::Utc;

use metricsql_common::time::{datetime_part, timestamp_secs_to_utc_datetime, DateTimePart};

use crate::execution::{eval_number, eval_time};
use crate::functions::arg_parse::{get_series_arg, get_string_arg};
use crate::functions::transform::{do_transform_values, get_timezone_offset, TransformFuncArg};
use crate::functions::utils::parse_timezone;
use crate::{RuntimeError, RuntimeResult};
use crate::types::{MetricName, Timeseries};

pub(crate) fn hour(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::Hour)
}

pub(crate) fn day_of_month(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::DayOfMonth)
}
pub(crate) fn day_of_week(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::DayOfWeek)
}

pub(crate) fn day_of_year(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::DayOfYear)
}

pub(crate) fn days_in_month(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::DaysInMonth)
}

pub(crate) fn minute(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::Minute)
}

pub(crate) fn month(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::Month)
}

pub(crate) fn year(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::Year)
}

fn extract_datetime_part(epoch_secs: f64, part: DateTimePart) -> f64 {
    if !epoch_secs.is_nan() {
        if let Some(utc) = timestamp_secs_to_utc_datetime(epoch_secs as i64) {
            if let Some(value) = datetime_part(utc, part) {
                return value as f64;
            }
        }
    }
    f64::NAN
}

fn transform_datetime_impl(
    tfa: &mut TransformFuncArg,
    part: DateTimePart,
) -> RuntimeResult<Vec<Timeseries>> {
    let tf = |values: &mut [f64]| {
        for v in values.iter_mut() {
            *v = extract_datetime_part(*v, part)
        }
    };

    let mut arg = if tfa.args.is_empty() {
        eval_time(tfa.ec)?
    } else {
        get_series_arg(&tfa.args, 0, tfa.ec)?
    };

    do_transform_values(&mut arg, tf, tfa.fe)
}

pub(crate) fn timezone_offset(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tz_name = match get_string_arg(&tfa.args, 0) {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "cannot get timezone name from arg: {:?}",
                e
            )))
        }
        Ok(s) => s,
    };

    let zone = match parse_timezone(&tz_name) {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "cannot load timezone {tz_name}: {:?}",
                e
            )))
        }
        Ok(res) => res,
    };

    let timestamps = tfa.ec.get_timestamps()?;
    let mut values: Vec<f64> = Vec::with_capacity(timestamps.len());
    for v in timestamps.iter() {
        // todo(perf) construct a DateTime in the tz and update timestamp
        let ofs = if let Some(val) = get_timezone_offset(&zone, *v) {
            val as f64
        } else {
            f64::NAN
        };
        values.push(ofs);
    }
    let ts = Timeseries {
        metric_name: MetricName::default(),
        values,
        timestamps: timestamps.clone(),
    };

    Ok(vec![ts])
}

pub(crate) fn now(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let now: f64 = Utc::now().timestamp() as f64 / 1e9_f64;
    eval_number(tfa.ec, now)
}

pub(crate) fn time(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    eval_time(tfa.ec)
}
