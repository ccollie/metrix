use std::{error, fmt, io};
pub mod encoders;
pub mod marshal;
pub mod utils;

// https://github.com/influxdata/influxdb/tree/b745a180a40fc62418e719ce531bc3a356503e0b/tsm/src
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodingError {
    pub description: String,
}

impl EncodingError {
    pub fn new(description: String) -> Self {
        Self { description }
    }
}

impl fmt::Display for EncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl error::Error for EncodingError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}

impl From<io::Error> for EncodingError {
    fn from(e: io::Error) -> Self {
        Self {
            description: format!("TODO - io error: {e} ({e:?})"),
        }
    }
}

impl From<std::str::Utf8Error> for EncodingError {
    fn from(e: std::str::Utf8Error) -> Self {
        Self {
            description: format!("TODO - utf8 error: {e} ({e:?})"),
        }
    }
}

pub type EncodingResult<T> = Result<T, EncodingError>;
