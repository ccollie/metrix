use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// the majority of the functionality in this file is Licensed to the Apache Software Foundation (ASF)
/// under the APACHE License 2.0
/// http://www.apache.org/licenses/LICENSE-2.0
/// https://docs.rs/arrow-array/29.0.0/src/arrow_array/lib.rs.html
use chrono::{
    DateTime, Datelike, Duration, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Timelike, Utc,
    Weekday,
};
use chrono_tz::Tz;

/// Number of seconds in a day
pub const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
pub const MILLISECONDS: i64 = 1_000;
/// Number of microseconds in a second
pub const MICROSECONDS: i64 = 1_000_000;
/// Number of nanoseconds in a second
pub const NANOSECONDS: i64 = 1_000_000_000;
const DAYS_IN_MONTH: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

pub type Time = Instant;

pub fn now() -> Instant {
    Instant::now()
}

/// Converts a nanosecond UTC timestamp into a DateTime structure
/// (which can have year, month, etc. extracted)
///
/// This is roughly equivalent to ConvertTime
/// from <https://github.com/influxdata/flux/blob/1e9bfd49f21c0e679b42acf6fc515ce05c6dec2b/values/time.go#L35-L37>
pub fn timestamp_to_datetime(ts: i64) -> Option<DateTime<Utc>> {
    timestamp_ns_to_datetime(ts).map(|naive| Utc.from_utc_datetime(&naive))
}

pub fn systemtime_to_timestamp(time: SystemTime) -> u64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs() * 1000 + u64::from(duration.subsec_nanos()) / 1_000_000,
        Err(e) => panic!(
            "SystemTime before UNIX EPOCH! Difference: {:?}",
            e.duration()
        ),
    }
}

pub fn round_to_seconds(ms: i64) -> i64 {
    ms - &ms % 1000
}

/// converts a `i64` representing a `time64(us)` to [`NaiveDateTime`]
#[inline]
pub fn time64us_to_time(v: i64) -> Option<NaiveTime> {
    NaiveTime::from_num_seconds_from_midnight_opt(
        // extract seconds from microseconds
        (v / MICROSECONDS) as u32,
        // discard extracted seconds and convert microseconds to
        // nanoseconds
        (v % MICROSECONDS * MILLISECONDS) as u32,
    )
}

/// converts a `i64` representing a `time64(ns)` to [`NaiveDateTime`]
#[inline]
pub fn time64ns_to_time(v: i64) -> Option<NaiveTime> {
    NaiveTime::from_num_seconds_from_midnight_opt(
        // extract seconds from nanoseconds
        (v / NANOSECONDS) as u32,
        // discard extracted seconds
        (v % NANOSECONDS) as u32,
    )
}

/// converts [`NaiveTime`] to a `i64` representing a `time64(us)`
#[inline]
pub fn time_to_time64us(v: NaiveTime) -> i64 {
    v.num_seconds_from_midnight() as i64 * MICROSECONDS
        + v.nanosecond() as i64 * MICROSECONDS / NANOSECONDS
}

/// converts [`NaiveTime`] to a `i64` representing a `time64(ns)`
#[inline]
pub fn time_to_time64ns(v: NaiveTime) -> i64 {
    v.num_seconds_from_midnight() as i64 * NANOSECONDS + v.nanosecond() as i64
}

/// converts a `i64` representing a `timestamp(s)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_s_to_datetime(v: i64) -> Option<NaiveDateTime> {
    Some(DateTime::from_timestamp(v, 0)?.naive_utc())
}

/// converts a `i64` representing a `timestamp(ms)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ms_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, milli_sec) = split_second(v, MILLISECONDS);

    let datetime = DateTime::from_timestamp(
        // extract seconds from milliseconds
        sec,
        // discard extracted seconds and convert milliseconds to nanoseconds
        milli_sec * MICROSECONDS as u32,
    )?;
    Some(datetime.naive_utc())
}

/// converts a `i64` representing a `time64(s)` to [`NaiveDateTime`]
pub fn timestamp_secs_to_datetime(secs: i64) -> Option<NaiveDateTime> {
    Some(DateTime::from_timestamp(secs, 0)?.naive_utc())
}

/// converts a `i64` representing a `time64(s)` to [`DateTime<Utc>`]
pub fn timestamp_secs_to_utc_datetime(secs: i64) -> Option<DateTime<Utc>> {
    timestamp_secs_to_datetime(secs).map(|naive| Utc.from_utc_datetime(&naive))
}

/// converts a `i64` representing a `timestamp(us)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_us_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, micro_sec) = split_second(v, MICROSECONDS);

    let datetime = DateTime::from_timestamp(
        // extract seconds from microseconds
        sec,
        // discard extracted seconds and convert microseconds to nanoseconds
        micro_sec * MILLISECONDS as u32,
    )?;
    Some(datetime.naive_utc())
}

/// converts a `i64` representing a `timestamp(ns)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ns_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, nano_sec) = split_second(v, NANOSECONDS);

    let datetime = DateTime::from_timestamp(
        // extract seconds from nanoseconds
        sec, // discard extracted seconds
        nano_sec,
    )?;
    Some(datetime.naive_utc())
}

/// Returns the time duration since UNIX_EPOCH in milliseconds.
pub fn current_time_millis() -> i64 {
    Utc::now().timestamp_millis()
}

#[inline]
pub(crate) fn split_second(v: i64, base: i64) -> (i64, u32) {
    (v.div_euclid(base), v.rem_euclid(base) as u32)
}

/// converts a `i64` representing a `duration(s)` to [`Duration`]
#[inline]
pub fn duration_s_to_duration(v: i64) -> Duration {
    Duration::seconds(v)
}

/// converts a `i64` representing a `duration(ms)` to [`Duration`]
#[inline]
pub fn duration_ms_to_duration(v: i64) -> Duration {
    Duration::milliseconds(v)
}

/// converts a `i64` representing a `duration(us)` to [`Duration`]
#[inline]
pub fn duration_us_to_duration(v: i64) -> Duration {
    Duration::microseconds(v)
}

/// converts a `i64` representing a `duration(ns)` to [`Duration`]
#[inline]
pub fn duration_ns_to_duration(v: i64) -> Duration {
    Duration::nanoseconds(v)
}

pub fn is_leap_year(y: u32) -> bool {
    if y % 4 != 0 {
        return false;
    }
    if y % 100 != 0 {
        return true;
    }
    y % 400 == 0
}

pub fn days_in_month<Tz: TimeZone>(t: DateTime<Tz>) -> u8 {
    let m = t.month() as usize;
    let y = t.year() as u32;
    if m == 2 && is_leap_year(y) {
        return 29_u8;
    }
    DAYS_IN_MONTH[m - 1]
}

pub fn int_day_of_week<Tz: TimeZone>(t: DateTime<Tz>) -> u8 {
    match t.weekday() {
        Weekday::Sun => 0_u8,
        Weekday::Mon => 1_u8,
        Weekday::Tue => 2_u8,
        Weekday::Wed => 3_u8,
        Weekday::Thu => 4_u8,
        Weekday::Fri => 5_u8,
        Weekday::Sat => 6_u8,
    }
}

#[derive(Clone, Copy)]
pub enum DateTimePart {
    DayOfMonth,
    DayOfWeek,
    DayOfYear,
    DaysInMonth,
    Hour,
    Minute,
    Month,
    Second,
    Year,
}

impl DateTimePart {
    pub fn get_part_from_datetime<Tz: TimeZone>(self, datetime: DateTime<Tz>) -> Option<u32> {
        datetime_part(datetime, self)
    }

    pub fn get_part_from_naive_datetime(&self, datetime: NaiveDateTime) -> u32 {
        match self {
            DateTimePart::DayOfMonth => datetime.day(),
            DateTimePart::DayOfWeek => datetime.weekday().num_days_from_sunday(),
            DateTimePart::DayOfYear => datetime.ordinal(),
            DateTimePart::DaysInMonth => {
                let cur_month = datetime.month();
                let cur_year = datetime.year();
                let naive_date = if cur_month == 12 {
                    NaiveDate::from_ymd_opt(cur_year + 1, 1, 1)
                } else {
                    NaiveDate::from_ymd_opt(cur_year, cur_month + 1, 1)
                };
                naive_date
                    .unwrap()
                    .signed_duration_since(NaiveDate::from_ymd_opt(cur_year, cur_month, 1).unwrap())
                    .num_days() as u32
            }
            DateTimePart::Hour => datetime.hour(),
            DateTimePart::Minute => datetime.minute(),
            DateTimePart::Month => datetime.month(),
            DateTimePart::Second => datetime.second(),
            DateTimePart::Year => datetime.year() as u32,
        }
    }

    pub fn get_part_from_timestamp_ms(&self, timestamp: i64) -> Option<u32> {
        if let Some(datetime) = timestamp_ms_to_datetime(timestamp) {
            return Some(self.get_part_from_naive_datetime(datetime));
        }
        None
    }
}
pub fn datetime_part<Tz: TimeZone>(datetime: DateTime<Tz>, part: DateTimePart) -> Option<u32> {
    match part {
        DateTimePart::DayOfMonth => Some(datetime.day()),
        DateTimePart::DayOfWeek => Some(int_day_of_week(datetime) as u32),
        DateTimePart::DayOfYear => Some(datetime.ordinal()),
        DateTimePart::DaysInMonth => Some(days_in_month(datetime) as u32),
        DateTimePart::Hour => Some(datetime.hour()),
        DateTimePart::Minute => Some(datetime.minute()),
        DateTimePart::Month => Some(datetime.month()),
        DateTimePart::Second => Some(datetime.second()),
        DateTimePart::Year => {
            let year = datetime.year();
            // we don't handle negative years currently
            if year < 0 {
                None
            } else {
                Some(year as u32)
            }
        }
    }
}

pub fn find_tz_from_env() -> Option<Tz> {
    // Windows does not support "TZ" env variable, which is used in the `Local` timezone under Unix.
    // However, we are used to set "TZ" env as the default timezone without actually providing a
    // timezone argument (especially in tests), and it's very convenient to do so, we decide to make
    // it work under Windows as well.
    std::env::var("TZ")
        .ok()
        .and_then(|tz| Tz::from_str(&tz).ok())
}

pub fn get_local_tz() -> Option<Tz> {
    find_tz_from_env().or_else(|| Tz::from_str("UTC").ok())
}

#[cfg(test)]
mod tests {
    use chrono::DateTime;

    use crate::prelude::{
        timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime, NANOSECONDS,
    };
    use crate::time::split_second;

    #[test]
    fn negative_input_timestamp_ns_to_datetime() {
        assert_eq!(
            timestamp_ns_to_datetime(-1),
            DateTime::from_timestamp(-1, 999_999_999).map(|x| x.naive_utc())
        );

        assert_eq!(
            timestamp_ns_to_datetime(-1_000_000_001),
            DateTime::from_timestamp(-2, 999_999_999).map(|x| x.naive_utc())
        );
    }

    #[test]
    fn negative_input_timestamp_us_to_datetime() {
        assert_eq!(
            timestamp_us_to_datetime(-1),
            DateTime::from_timestamp(-1, 999_999_000).map(|x| x.naive_utc())
        );

        assert_eq!(
            timestamp_us_to_datetime(-1_000_001),
            DateTime::from_timestamp(-2, 999_999_000).map(|x| x.naive_utc())
        );
    }

    #[test]
    fn negative_input_timestamp_ms_to_datetime() {
        assert_eq!(
            timestamp_ms_to_datetime(-1),
            DateTime::from_timestamp(-1, 999_000_000).map(|x| x.naive_utc())
        );

        assert_eq!(
            timestamp_ms_to_datetime(-1_001),
            DateTime::from_timestamp(-2, 999_000_000).map(|x| x.naive_utc())
        );
    }

    // #[test]
    // fn negative_input_date64_to_datetime() {
    //     assert_eq!(
    //         date64_to_datetime(-1),
    //         NaiveDateTime::from_timestamp_opt(-1, 999_000_000)
    //     );
    //
    //     assert_eq!(
    //         date64_to_datetime(-1_001),
    //         NaiveDateTime::from_timestamp_opt(-2, 999_000_000)
    //     );
    // }

    #[test]
    fn test_split_seconds() {
        let (sec, nano_sec) = split_second(100, NANOSECONDS);
        assert_eq!(sec, 0);
        assert_eq!(nano_sec, 100);

        let (sec, nano_sec) = split_second(123_000_000_456, NANOSECONDS);
        assert_eq!(sec, 123);
        assert_eq!(nano_sec, 456);

        let (sec, nano_sec) = split_second(-1, NANOSECONDS);
        assert_eq!(sec, -1);
        assert_eq!(nano_sec, 999_999_999);

        let (sec, nano_sec) = split_second(-123_000_000_001, NANOSECONDS);
        assert_eq!(sec, -124);
        assert_eq!(nano_sec, 999_999_999);
    }
}
