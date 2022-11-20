use std::{fmt};
use std::fmt::{Display, Formatter};
use thiserror::Error;

use crate::lexer::{TextSpan, Token, TokenKind};

pub type ParseResult<T> = Result<T, ParseError>;

#[derive(Debug, PartialEq, Clone, Error)]
pub enum ParseError {
    #[error("{0}")]
    ArgumentError(String),
    #[error(transparent)]
    InvalidToken(InvalidTokenError),
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
    #[error("Expected number: found `{0}`")]
    InvalidNumber(String),
    #[error(transparent)]
    InvalidArgCount(ArgCountError),
    #[error("Error expanding WITH expression: `{0}`")]
    WithExprExpansionError(String),
    #[error("{0}")]
    General(String),
    #[error("Invalid regex: {0}")]
    InvalidRegex(String),
    #[error("{0}")]
    InvalidFunction(String),
    #[error("{0}")]
    InvalidExpression(String),
}

/// ParseErr wraps a parsing error with line and position context.
#[derive(Debug, PartialEq, Clone, Error)]
pub struct ParseErr {
    pub range: TextSpan,
    pub err: String,
    pub query: String,
    /// line_offset is an additional line offset to be added. Only used inside unit tests.
    pub line_offset: usize
}

impl ParseErr {
    pub fn new<S: Into<TextSpan>>(msg: &str, query: &str, range: S) -> Self {
        Self {
            range: range.into(),
            err: msg.to_string(),
            query: query.to_string(),
            line_offset: 0
        }
    }
}

impl Display for ParseErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let pos: usize = self.range.start;
        let mut last_line_break = 0;
        let mut line = self.line_offset + 1;

        let position_str: String;

        if pos > self.query.len() {
            position_str = "invalid position:".to_string()
        } else {
            for (i, c) in self.query[0..pos].chars().enumerate() {
                if c == '\n' {
                    last_line_break = i;
                    line += 1;
                }
            }
            let col = pos - last_line_break;
            position_str = format!("{}:{}:", line, col).to_string()
        }
        write!(f, "{} parse error: {}", position_str, self.err)?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone, Error)]
pub struct InvalidTokenError {
    pub(super) expected: Vec<TokenKind>,
    pub(super) found: Option<TokenKind>,
    pub(super) range: TextSpan,
    pub(super) context: String,
}

impl InvalidTokenError {
    pub fn new<S: Into<TextSpan>>(expected: &[TokenKind], found: Option<TokenKind>, range: S) -> Self {
        Self {
            expected: Vec::from(expected),
            found,
            range: range.into(),
            context: "".to_string(),
        }
    }

    pub fn with_context(&mut self, context: &str) {
        self.context = context.to_string();
    }
}

impl Display for InvalidTokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.context.is_empty() {
            write!(f, "{} :", self.context)?;
        }
        if self.found.is_none() {
            write!(f,"unexpected end of stream")?;
            if self.range.start > 0 {
                write!(
                    f,
                    " at {}..{}",
                    self.range.start,
                    self.range.end,
                )?;
            }
        } else {
            write!(
                f,
                "error at {}..{}",
                self.range.start,
                self.range.end,
            )?;
        }

        write!(f, ": expected ")?;

        let num_expected = self.expected.len();
        let is_first = |idx| idx == 0;
        let is_last = |idx| idx == num_expected - 1;

        for (idx, expected_kind) in self.expected.iter().enumerate() {
            let expected = expected_kind.to_string();
            if is_first(idx) {
                write!(f, "{}", expected)?;
            } else if is_last(idx) {
                write!(f, " or {}", expected)?;
            } else {
                write!(f, ", {}", expected)?;
            }
        }

        if let Some(found) = self.found {
            write!(f, ", but found {}", found)?;
        }

        Ok(())
    }
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
    pub fn new_with_token(token: &Token, signature: &str, min: usize, max: usize) -> Self {
        Self::new_with_index(Some(usize::from(token.span.start)), signature, min, max)
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

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use crate::lexer::{TextSpan, TokenKind};

    use super::*;

    fn check(
        expected: Vec<TokenKind>,
        found: Option<TokenKind>,
        range: Range<usize>,
        output: &str,
    ) {
        let error = InvalidTokenError {
            expected,
            found,
            range: TextSpan::new(range.start, range.end),
            context: "".to_string(),
        };

        assert_eq!(format!("{}", error), output);
    }

    #[test]
    fn one_expected_did_find() {
        check(
            vec![TokenKind::Equal],
            Some(TokenKind::Ident),
            10..20,
            "error at 10..20: expected ‘=’, but found identifier",
        );
    }

    #[test]
    fn one_expected_did_not_find() {
        check(
            vec![TokenKind::RightParen],
            None,
            5..6,
            "error at 5..6: expected ‘)’",
        );
    }

    #[test]
    fn two_expected_did_find() {
        check(
            vec![TokenKind::OpPlus, TokenKind::OpMinus],
            Some(TokenKind::Equal),
            0..1,
            "error at 0..1: expected ‘+’ or ‘-’, but found ‘=’",
        );
    }

    #[test]
    fn multiple_expected_did_find() {
        check(
            vec![
                TokenKind::Number,
                TokenKind::Ident,
                TokenKind::OpMinus,
                TokenKind::LeftParen,
            ],
            Some(TokenKind::With),
            100..105,
            "error at 100..105: expected number, identifier, ‘-’ or ‘(’, but found ‘with’",
        );
    }
}
