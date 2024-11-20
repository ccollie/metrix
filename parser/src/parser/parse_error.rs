use std::fmt;
use std::fmt::{Display, Formatter};

use logos::Span;
use thiserror::Error;

use crate::parser::tokens::Token;
use crate::parser::TokenWithLocation;

pub type ParseResult<T> = Result<T, ParseError>;

#[derive(Default, Debug, PartialEq, Clone, Error)]
pub enum ParseError {
    #[error("{0}")]
    ArgumentError(String),
    #[error(transparent)]
    Unexpected(ParseErr), // TODO !!!!!!
    #[error("Duplicate argument `{0}`")]
    DuplicateArgument(String),
    #[error("Unexpected end of text")]
    UnexpectedEOF,
    #[error("Invalid aggregation function `{0}`")]
    InvalidAggregateFunction(String),
    #[error("Expected positive duration: found `{0}`")]
    InvalidDuration(String),
    #[error("Expected positive unix millis or rfc3339 timestamp: found `{0}`")]
    InvalidTimestamp(String),
    #[error("Expected number: found `{0}`")]
    InvalidNumber(String),
    #[error(transparent)]
    InvalidArgCount(ArgCountError),
    #[error("Error expanding WITH expression: `{0}`")]
    WithExprExpansionError(String),
    #[error("Syntax Error: `{0}`")]
    SyntaxError(String),
    #[error("{0}")]
    General(String),
    #[error("Invalid regex: {0}")]
    InvalidRegex(String),
    #[error("Unknown function {0}")]
    InvalidFunction(String),
    #[error("Division by zero")]
    DivisionByZero,
    #[error("{0}")]
    Unsupported(String),
    #[default]
    #[error("Parse error")]
    Other,
}

/// ParseErr wraps a parsing error with line and position context.
#[derive(Debug, PartialEq, Clone, Error)]
pub struct ParseErr {
    pub range: Span,
    pub err: String,
    /// line_offset is an additional line offset to be added. Only used inside unit tests.
    pub line_offset: usize,
}

impl ParseErr {
    pub fn new<S: Into<Span>>(msg: &str, range: S) -> Self {
        Self {
            range: range.into(),
            err: msg.to_string(),
            line_offset: 0,
        }
    }
}

impl Display for ParseErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let pos: usize = self.range.start;
        let last_line_break = 0;
        let line = self.line_offset + 1;

        let col = pos - last_line_break;
        let position_str = format!("{}:{}:", line, col).to_string();
        write!(f, "{} parse error: {}", position_str, self.err)?;
        Ok(())
    }
}

pub(crate) fn invalid_token_error(
    expected: &[Token],
    found: Option<Token>,
    range: &Span,
    context: String,
) -> ParseError {
    let mut res = String::with_capacity(100);
    if !context.is_empty() {
        res.push_str(format!("{} :", context).as_str())
    }
    if found.is_none() {
        res.push_str("unexpected end of stream");
        if range.start > 0 {
            res.push_str(format!(" at {}..{}: ", range.start, range.end).as_str());
        }
    } else {
        res.push_str(format!("error at {}..{}: ", range.start, range.end).as_str());
    }

    let num_expected = expected.len();
    let is_first = |idx| idx == 0;
    let is_last = |idx| idx == num_expected - 1;

    res.push_str("expected ");

    for (idx, expected) in expected.iter().enumerate() {
        let item = format!("\"{}\"", expected);
        if is_first(idx) {
            res.push_str(item.as_str());
        } else if is_last(idx) {
            res.push_str(format!(" or {}", item).as_str());
        } else {
            res.push_str(format!(", {}", item).as_str());
        }
    }

    if let Some(found) = found {
        res.push_str(format!(", but found {}", found).as_str());
    }

    ParseError::SyntaxError(res)
}

pub(crate) fn syntax_error(msg: &str, range: &Span, context: String) -> ParseError {
    let mut res = String::with_capacity(100);
    if !context.is_empty() {
        res.push_str(format!("{} :", context).as_str())
    }

    res.push_str(format!("error at {}..{}: {}", range.start, range.end, msg).as_str());

    ParseError::SyntaxError(res)
}

/// unexpected creates a parser error complaining about an unexpected lexer item.
/// The item that is presented as unexpected is always the last item produced
/// by the lexer.
pub(super) fn unexpected(
    context: &str,
    actual: &str,
    expected: &str,
    span: Option<&Span>,
) -> ParseError {
    let mut err_msg: String = String::with_capacity(25 + context.len() + expected.len());

    err_msg.push_str("unexpected ");

    err_msg.push_str(actual);

    if !context.is_empty() {
        err_msg.push_str(" in ");
        err_msg.push_str(context)
    }

    if !expected.is_empty() {
        err_msg.push_str(", expected ");
        err_msg.push_str(expected)
    }

    let span = if let Some(sp) = span {
        sp.clone()
    } else {
        Span::default()
    };
    ParseError::Unexpected(ParseErr::new(&err_msg, span))
}

/// Occurs when a function is called with the wrong number of arguments
#[derive(Debug, Clone, PartialEq, Eq, Error)]
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

    /// Create a new instance of the error caused by a token
    ///
    /// # Arguments
    /// * `token` - Token causing the error
    /// * `signature` - Function call signature
    /// * `min` - Smallest allowed number of arguments
    /// * `max` - Largest allowed number of arguments
    pub fn new_with_token(
        token: &TokenWithLocation,
        signature: &str,
        min: usize,
        max: usize,
    ) -> Self {
        Self::new_with_index(Some(token.span.start), signature, min, max)
    }

    /// Create a new instance of the error at a specific position
    ///
    /// # Arguments
    /// * `pos` - Index at which the error occurred
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

    /// Return the location at which the error occured
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
