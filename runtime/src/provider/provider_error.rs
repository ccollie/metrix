use std::fmt::Display;
use thiserror::Error;


pub type ProviderResult<T> = Result<T, ProviderError>;

#[derive(Debug, PartialEq, Clone, Error)]
pub enum ProviderError {
    #[error("Invalid matcher: `{0}`")]
    InvalidMatcher(String),
    #[error("Missing matching filter")]
    MissingMatcher,
    #[error("Error fetching postings")]
    PostingFetchError,
    #[error("{0}")]
    General(String),
    #[error("Deadline exceeded: {0}")]
    DeadlineExceededError(String),
    #[error("Task cancelled: {0}")]
    TaskCancelledError(String),
    #[error("Duplicate output series: {0}")]
    DuplicateOutputSeries(String),
    #[error("The response contains more than {max_series} series: found {found_series};")]
    MaxSeriesExceeded {
        found_series: usize,
        max_series: usize,
    },
}

impl ProviderError {
    pub fn deadline_exceeded(s: &str) -> Self {
        ProviderError::DeadlineExceededError(s.to_string())
    }
}

impl From<&str> for ProviderError {
    fn from(message: &str) -> Self {
        ProviderError::General(String::from(message))
    }
}

impl From<String> for ProviderError {
    fn from(message: String) -> Self {
        ProviderError::General(message)
    }
}

impl<E: std::error::Error + 'static> From<(String, E)> for ProviderError {
    fn from((message, err): (String, E)) -> Self {
        let msg = format!("{}: {}", message, err);
        ProviderError::General(msg)
    }
}

impl<E: std::error::Error + 'static> From<(&str, E)> for ProviderError {
    fn from((message, err): (&str, E)) -> Self {
        let msg = format!("{}: {}", message, err);
        ProviderError::General(msg)
    }
}


