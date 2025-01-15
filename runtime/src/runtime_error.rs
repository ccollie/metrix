use std::fmt;
use std::fmt::Display;

use thiserror::Error;

use metricsql_parser::parser::ParseError;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Debug, PartialEq, Clone, Error)]
pub enum RuntimeError {
    #[error("Expression execution error `{0}`")]
    ExecutionError(String),
    #[error("Expected number: found `{0}`")]
    InvalidNumber(String),
    #[error("Argument error: {0}")]
    ArgumentError(String),
    #[error(transparent)]
    InvalidArgCount(ArgCountError),
    #[error("{0}")]
    General(String),
    #[error("Invalid regex: {0}")]
    InvalidRegex(String),
    #[error("Unknown function `{0}`")]
    UnknownFunction(String),
    #[error(transparent)]
    ParseError(ParseError),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Type casting error: {0}")]
    TypeCastError(String),
    #[error("Deadline exceeded: {0}")]
    DeadlineExceededError(String),
    #[error("Task cancelled: {0}")]
    TaskCancelledError(String),
    #[error("Invalid State: {0}")]
    InvalidState(String),
    #[error("Internal Error: {0}")]
    Internal(String),
    #[error("{0}")]
    ResourcesExhausted(String),
    #[error("{0}")]
    NotImplemented(String),
    #[error("Cannot optimize expression: {0}")]
    OptimizerError(String),
    #[error("Duplicate metric labels: {0}")]
    DuplicateMetricLabels(String),
}

impl RuntimeError {
    pub fn deadline_exceeded(s: &str) -> Self {
        RuntimeError::DeadlineExceededError(s.to_string())
    }
}

impl From<&str> for RuntimeError {
    fn from(message: &str) -> Self {
        RuntimeError::General(String::from(message))
    }
}

impl From<String> for RuntimeError {
    fn from(message: String) -> Self {
        RuntimeError::General(message)
    }
}

impl<E: std::error::Error + 'static> From<(String, E)> for RuntimeError {
    fn from((message, err): (String, E)) -> Self {
        let msg = format!("{}: {}", message, err);
        RuntimeError::General(msg)
    }
}

impl<E: std::error::Error + 'static> From<(&str, E)> for RuntimeError {
    fn from((message, err): (&str, E)) -> Self {
        let msg = format!("{}: {}", message, err);
        RuntimeError::General(msg)
    }
}

/// Occurs when a function is called with the wrong number of arguments
#[derive(Debug, PartialEq, Clone, Error)]
pub struct ArgCountError {
    pos: Option<usize>,
    min: usize,
    max: usize,
    signature: String,
}

impl ArgCountError {
    /// Create a new instance of the error
    ///
    /// # Arguments
    /// * `signature` - Function call signature
    /// * `min` - Smallest allowed number of arguments
    /// * `max` - Largest allowed number of arguments
    pub fn new(signature: &str, min: usize, max: usize) -> Self {
        Self::new_with_index(None, signature, min, max)
    }

    /// Create a new instance of the error at a specific position
    ///
    /// # Arguments
    /// * `pos` - Index at which the error occured
    /// * `signature` - Function call signature
    /// * `min` - Smallest allowed number of arguments
    /// * `max` - Largest allowed number of arguments
    pub fn new_with_index(pos: Option<usize>, signature: &str, min: usize, max: usize) -> Self {
        Self {
            pos,
            min,
            max,
            signature: signature.to_string(),
        }
    }

    /// Function call signature
    pub fn signature(&self) -> &str {
        &self.signature
    }

    /// Smallest allowed number of arguments
    pub fn min(&self) -> usize {
        self.min
    }

    /// Largest allowed number of arguments
    pub fn max(&self) -> usize {
        self.max
    }

    /// Return the location at which the error occurred
    pub fn pos(&self) -> Option<usize> {
        self.pos
    }
}

impl Display for ArgCountError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.min == self.max {
            write!(f, "{}: expected {} args", self.signature, self.min)?;
        } else {
            write!(
                f,
                "{}: expected {}-{} args",
                self.signature, self.min, self.max
            )?;
        }

        if let Some(pos) = self.pos {
            write!(f, " at position {}", pos)?;
        }

        Ok(())
    }
}
