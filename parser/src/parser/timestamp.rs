use chrono::DateTime;
use crate::parser::{ParseError, ParseResult};

/// Parses a string into a unix timestamp (milliseconds). Accepts a positive integer or an RFC3339 timestamp.
/// Included here only to avoid having to include chrono in the public API
pub fn parse_timestamp(s: &str) -> ParseResult<i64> {
    let value = if let Ok(dt) = parse_numeric_timestamp(s) {
        dt
    } else {
        let value = DateTime::parse_from_rfc3339(s)
            .map_err(|_| ParseError::InvalidTimestamp(s.to_string()))?;
        value.timestamp_millis()
    };
    if value < 0 {
        return Err(ParseError::InvalidTimestamp(s.to_string()));
    }
    Ok(value)
}


/// parse_numeric_timestamp parses timestamp at s in seconds, milliseconds, microseconds or nanoseconds.
///
/// It returns milliseconds for the parsed timestamp.
pub fn parse_numeric_timestamp(s: &str) -> Result<i64, Box<dyn std::error::Error>> {
    const CHARS_TO_CHECK: &[char] = &['.', 'e', 'E'];
    const NANOSECONDS_THRESHOLD: i64 = u32::MAX.saturating_mul(1_000_000) as i64;
    const MICROSECONDS_THRESHOLD: i64 = u32::MAX.saturating_mul(1_000) as i64;

    if s.contains(CHARS_TO_CHECK) {
        // The timestamp is a floating-point number
        let ts: f64 = s.parse()?;
        if ts >= u32::MAX as f64 {
            // The timestamp is in milliseconds
            return Ok(ts as i64);
        }
        return Ok(ts.round() as i64);
    }
    // The timestamp is an integer number
    let ts: i64 = s.parse()?;
    match ts {
        ts if ts >= NANOSECONDS_THRESHOLD => {
            // The timestamp is in nanoseconds
            Ok(ts / 1_000_000)
        }
        ts if ts >= MICROSECONDS_THRESHOLD => {
            // The timestamp is in microseconds
            Ok(ts / 1_000)
        }
        ts if ts >= u32::MAX as i64 => {
            // The timestamp is in milliseconds
            Ok(ts)
        }
        _ => Ok(ts * 1_000),
    }
}



#[cfg(test)]
mod tests {

}