use std::str;

use crate::parser::{ParseError, ParseResult};

const MILLIS_PER_SECOND: f64 = 1e3;
const MILLIS_PER_MINUTE: f64 = 60.0 * MILLIS_PER_SECOND;
const MILLIS_PER_HOUR: f64 = 60.0 * MILLIS_PER_MINUTE;
const MILLIS_PER_DAY: f64 = 24.0 * MILLIS_PER_HOUR;
const MILLIS_PER_WEEK: f64 = 7.0 * MILLIS_PER_DAY;
const MILLIS_PER_YEAR: f64 = 365.0 * MILLIS_PER_DAY;


/// `positive_duration_value` returns positive duration in milliseconds for the given s
/// and the given step.
///
/// Duration in s may be combined, i.e. 2h5m or 2h-5m.
///
/// Error is returned if the duration in s is negative.
pub fn parse_positive_duration_value(s: &str, step: i64) -> Result<i64, ParseError> {
    let d = parse_duration_value(s, step)?;
    if d < 0 {
        return Err(ParseError::InvalidDuration(format!(
            "duration cannot be negative; got {}",
            s
        )));
    }
    Ok(d)
}

/// `parse_duration_value` returns the duration in milliseconds for the given s
/// and the given step.
///
/// Duration in s may be combined, i.e. 2h5m, -2h5m or 2h-5m.
///
/// The returned duration value can be negative.
pub fn parse_duration_value(s: &str, step: i64) -> ParseResult<i64> {
    fn scan_value(s: &str, step: i64) -> ParseResult<i64> {
        let mut is_minus = false;
        let mut cursor: &str = s;
        let mut d = 0.0;

        while !cursor.is_empty() {
            let n = scan_single_duration(cursor, true);
            if n <= 0 {
                return Err(ParseError::InvalidDuration(s.to_string()));
            }
            let ds = &cursor[0..n as usize];
            let mut d_local = parse_single_duration(ds, step)?;
            if is_minus && (d_local > 0.0) {
                d_local = -d_local
            }
            d += d_local;
            if d_local < 0f64 {
                is_minus = true
            }

            if n > cursor.len() as i32 {
                break;
            }
            cursor = &cursor[n as usize..];
        }
        if d.abs() > (1_i64 << (62 - 1)) as f64 {
            let msg = format!("duration {} is too large", s);
            return Err(ParseError::General(msg));
        }
        Ok(d as i64)
    }

    if s.is_empty() {
        return Err(ParseError::General("duration cannot be empty".to_string()));
    }
    // Try parsing floating-point duration
    match s.parse::<f64>() {
        // Convert the duration to milliseconds.
        Ok(d) => {
            let val = (d * 1000_f64) as i64;
            Ok(val)
        }
        Err(_) => scan_value(s, step),
    }
}

fn parse_single_duration(s: &str, step: i64) -> Result<f64, ParseError> {
    let mut num_part = &s[0..s.len() - 1];
    if num_part.ends_with('m') {
        // Duration in ms
        num_part = &num_part[0..num_part.len() - 1]
    }
    let f: f64 = match num_part.parse() {
        Ok(f) => f,
        Err(_) => return Err(ParseError::InvalidDuration(s.to_string())),
    };
    let mp: f64;
    let unit = &s[num_part.len()..];
    match unit {
        "ms" => mp = 1.0,
        "s" => mp = MILLIS_PER_SECOND,
        "m" => mp = MILLIS_PER_MINUTE,
        "h" => mp = MILLIS_PER_HOUR,
        "d" => mp = MILLIS_PER_DAY,
        "w" => mp = MILLIS_PER_WEEK,
        "y" => mp = MILLIS_PER_YEAR,
        "i" => mp = step as f64,
        _ => {
            return Err(ParseError::General(format!("invalid duration suffix in {s}")))
        }
    }
    Ok(mp * f)
}

/// scan_duration scans duration, which must start with positive num.
///
/// I.e. 123h, 3h5m or 3.4d-35.66s
#[allow(dead_code)]
pub fn scan_duration(s: &str) -> i32 {
    // The first part must be non-negative
    let mut n = scan_single_duration(s, false);
    if n <= 0 {
        return -1;
    }

    // todo: duration segments should go from courser to finer grain
    // i.e. we should forbid something like 25s2d
    let mut cursor: &str = &s[n as usize..];
    let mut i = n;
    loop {
        // Other parts may be negative
        n = scan_single_duration(s, true);
        if n <= 0 {
            return i;
        }
        cursor = &cursor[n as usize..];
        i += n
    }
}

fn scan_single_duration(s: &str, can_be_negative: bool) -> i32 {
    if s.is_empty() {
        return -1;
    }
    let mut i = 0;

    let ch = s.chars().next().unwrap();
    if ch == '-' && can_be_negative {
        i += 1;
    }

    let mut cursor = &s[i..];
    let mut curr: char = ch;

    for ch in cursor.chars() {
        if !ch.is_ascii_digit() {
            curr = ch;
            break;
        }
        i += 1;
    }

    if i == 0 || i == s.len() {
        return -1;
    }

    if curr == '.' {
        let j = i;
        i += 1;
        cursor = &s[i..];
        for c in cursor.chars() {
            if !c.is_ascii_digit() {
                curr = c;
                break;
            }
            i += 1;
        }
        if i == j || i == s.len() {
            return -1;
        }
    }
    match curr {
        'm' => {
            if i + 1 < s.len() {
                cursor = &s[i..];
                curr = cursor.chars().nth(1).unwrap();
                if curr == 's' {
                    // duration in ms
                    i += 1;
                }
            }
            // duration in minutes
            (i + 1) as i32
        }
        's' | 'h' | 'd' | 'w' | 'y' | 'i' => (i + 1) as i32,
        _ => -1,
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::duration::parse_positive_duration_value;
    use crate::parser::parse_duration_value;

    const MS: i64 = 1;
    const SECOND: i64 = 1000 * MS;
    const MINUTE: i64 = 60 * SECOND;
    const HOUR: i64 = 60 * MINUTE;
    const DAY: i64 = 24 * HOUR;
    const WEEK: i64 = 7 * DAY;
    const YEAR: i64 = 365 * DAY;

    #[test]
    fn test_duration_success() {
        fn f(s: &str, step: i64, expected: i64) {
            let d = parse_duration_value(s, step).unwrap();
            assert_eq!(
                d, expected,
                "unexpected duration; got {}; want {}; expr {}",
                d, expected, s
            )
        }

        // Integer durations
        f("123ms", 42, 123);
        f("-123ms", 42, -123);
        f("123s", 42, 123 * 1000);
        f("-123s", 42, -123 * 1000);
        f("4236579305ms", 42, 4236579305);
        f("123m", 42, 123 * MINUTE);
        f("1h", 42, HOUR);
        f("2d", 42, 2 * DAY);
        f("3w", 42, 3 * WEEK);
        f("4y", 42, 4 * YEAR);
        f("1i", 42 * 1000, 42 * 1000);
        f("3i", 42, 3 * 42);
        f("-3i", 42, -3 * 42);
        f("1m34s24ms", 42, 94024);
        f("1m-34s24ms", 42, 25976);
        f("-1m34s24ms", 42, -94024);
        f("-1m-34s24ms", 42, -94024);

        // Float durations
        f("34.54ms", 42, 34);
        f("-34.34ms", 42, -34);
        f("0.234s", 42, 234);
        f("-0.234s", 42, -234);
        f("1.5s", 42, (1.5 * SECOND as f64) as i64);
        f("1.5m", 42, (1.5 * MINUTE as f64) as i64);
        f("1.2h", 42, (1.2 * HOUR as f64) as i64);
        f("1.1d", 42, (1.1 * DAY as f64) as i64);
        f("1.1w", 42, (1.1 * WEEK as f64) as i64);
        f("1.3y", 42, (1.3 * YEAR as f64).floor() as i64);
        f("-1.3y", 42, (-1.3 * YEAR as f64).floor() as i64);
        f("0.1i", 12340, (0.1 * 12340.0) as i64);
        f("1.5m3.4s2.4ms", 42, 93402);
        f("-1.5m3.4s2.4ms", 42, -93402);

        // Floating-point durations without suffix.
        f("123", 45, 123000);
        f("1.23", 45, 1230);
        f("-0.56", 12, -560);
        f("-.523e2", 21, -52300)
    }

    #[test]
    fn test_complex() {
        fn f(s: &str) {
            let _ = parse_duration_value(s, 1).unwrap();
        }

        f("5w4h-3.4m13.4ms");
    }

    #[test]
    fn test_duration_error() {
        fn f(s: &str) {
            if let Ok(d) = parse_duration_value(s, 42) {
                panic!("Expected error, got {} for expr {}", d, s)
            }
        }

        f("");
        f("foo");
        f("m");
        f("1.23mm");
        f("123q");
        f("-123q");

        // With uppercase duration
        f("1M");
        f("1Ms");
        f("1MS");
        f("1Y");
        f("2W");
        f("3D");
        f("3H");
        f("3S")
    }

    #[test]
    fn test_positive_duration_error() {
        fn f(s: &str) {
            if parse_positive_duration_value(s, 42).is_ok() {
                panic!("Expecting an error for duration {}", s)
            }
        }
        f("");
        f("foo");
        f("m");
        f("1.23mm");
        f("123q");
        f("-123s");

        // Too big duration
        f("10000000000y")
    }
}
