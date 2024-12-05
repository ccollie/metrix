use std::fmt;
use std::fmt::Display;

use crate::types::{Timestamp, TimestampTrait};
use crate::{RuntimeError, RuntimeResult};
use metricsql_common::prelude::humanize_duration;
use std::time::Duration;

/// These values prevent from overflow when storing ms-precision time in i64.
pub const MAX_DURATION_MSECS: u64 = 100 * 365 * 24 * 3600 * 1000;

pub const MAX_DURATION: Duration = Duration::from_millis(MAX_DURATION_MSECS);

/// Deadline contains deadline with the corresponding timeout for pretty error messages.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Deadline {
    /// deadline in unix timestamp seconds.
    pub deadline: Timestamp,
    pub timeout: Duration,
}

impl Deadline {
    /// Returns a deadline for the given timeout.
    pub fn new(timeout: Duration) -> RuntimeResult<Self> {
        Deadline::with_start_time(Timestamp::now(), timeout)
    }

    pub fn from_now(timeout: Duration) -> RuntimeResult<Self> {
        Deadline::with_start_time(Timestamp::now(), timeout)
    }

    /// Returns a deadline for the given start time and timeout.
    pub fn with_start_time<T>(start_time: T, timeout: Duration) -> RuntimeResult<Self>
    where
        T: Into<Timestamp>,
    {
        let millis = timeout.as_millis() as u64;
        if timeout > MAX_DURATION {
            return Err(RuntimeError::ArgumentError(format!(
                "Timeout value too large: {}", humanize_duration(&timeout),
            )));
        }
        let start = start_time.into();
        Ok(Deadline {
            deadline: start + millis as i64,
            timeout,
        })
    }

    /// returns true if deadline is exceeded.
    pub fn exceeded(&self) -> bool {
        Timestamp::now() > self.deadline
    }
}

impl Default for Deadline {
    fn default() -> Self {
        let start = Timestamp::now();
        let timeout = Duration::from_secs(10); // todo: constant
        let deadline = start + timeout.as_millis() as i64;
        Deadline {
            deadline,
            timeout,
        }
    }
}

impl TryFrom<Duration> for Deadline {
    type Error = RuntimeError;

    fn try_from(timeout: Duration) -> Result<Self, Self::Error> {
        Deadline::new(timeout)
    }
}

impl TryFrom<i64> for Deadline {
    type Error = RuntimeError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        let timeout = Duration::from_millis(value as u64);
        Deadline::new(timeout)
    }
}

impl Display for Deadline {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let start_time = self.deadline.sub(self.timeout);
        let elapsed = (Timestamp::now() - start_time) / 1000_i64;
        write!(
            f,
            "{:.3} seconds (elapsed {:.3} seconds);",
            self.timeout.as_secs(),
            elapsed
        )?;
        Ok(())
    }
}
