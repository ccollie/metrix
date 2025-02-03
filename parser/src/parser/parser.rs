use logos::{Logos, Span};
use std::borrow::BorrowMut;
use std::time::Duration;

use crate::ast::{DurationExpr, Expr, ParensExpr, StringExpr};
use crate::parser::expr::parse_expression;
use crate::parser::tokens::{Token, IDENT_LIKE_TOKENS};
use crate::parser::{
    extract_string_value, invalid_token_error, parse_duration_value,
    parse_number, syntax_error,
    ParseErr, ParseError, ParseResult,
};
use crate::parser::tokens::Token::Identifier;
use crate::prelude::unescape_ident;

/// A token of MetricSql source.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenWithLocation<'source> {
    /// The kind of token.
    pub kind: Token,
    pub text: &'source str,
    pub span: Span,
}

impl TokenWithLocation<'_> {
    pub fn len(&self) -> usize {
        self.text.len()
    }
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// parser parses MetricsQL expression.
///
/// preconditions for all parser.parse* funcs:
/// - self.tokens[self.cursor] should point to the first token to parse.
///
/// post-conditions for all parser.parse* funcs:
/// - self.token[self.cursor] should point to the next token after the parsed token.
pub struct Parser<'a> {
    tokens: Vec<TokenWithLocation<'a>>,
    pub(crate) cursor: usize,
    pub(super) needs_expansion: bool,
}

impl<'a> Parser<'a> {
    pub fn tokenize(input: &'a str) -> ParseResult<Vec<TokenWithLocation<'a>>> {
        let mut lexer: logos::Lexer<'a, Token> = Token::lexer(input);

        let mut tokens = Vec::with_capacity(64); // todo: pre-size
        while let Some(tok) = lexer.next() {
            match tok {
                Ok(tok) => {
                    tokens.push(TokenWithLocation {
                        kind: tok,
                        text: lexer.slice(),
                        span: lexer.span(),
                    });
                }
                Err(_) => {
                    let span = lexer.span();
                    let inner = ParseErr::new(
                        format!("unexpected token \"{}\"", lexer.slice().trim()).as_str(),
                        span,
                    );
                    return Err(ParseError::Unexpected(inner));
                }
            }
        }

        Ok(tokens)
    }

    pub fn new(input: &'a str) -> ParseResult<Self> {
        let tokens = Self::tokenize(input)?;

        Ok(Self {
            cursor: 0,
            tokens,
            needs_expansion: false,
        })
    }

    pub(super) fn next_token(&mut self) -> Option<&TokenWithLocation<'a>> {
        if self.is_eof() {
            return None;
        }
        self.cursor += 1;
        self.tokens.get(self.cursor)
    }

    pub fn prev_token(&mut self) -> Option<&TokenWithLocation<'a>> {
        self.cursor = self.cursor.saturating_sub(1);
        self.tokens.get(self.cursor)
    }

    pub fn peek_token(&self) -> Option<&TokenWithLocation<'a>> {
        self.tokens.get(self.cursor)
    }

    pub(crate) fn current_token(&self) -> ParseResult<&TokenWithLocation<'a>> {
        match self.tokens.get(self.cursor) {
            Some(t) => Ok(t),
            None => Err(ParseError::UnexpectedEOF),
        }
    }

    pub fn is_eof(&self) -> bool {
        self.cursor >= self.tokens.len()
    }

    pub(super) fn last_token_range(&self) -> Option<Span> {
        let index = if self.is_eof() {
            if !self.tokens.is_empty() {
                self.tokens.len() - 1
            } else {
                0
            }
        } else {
            self.cursor
        };
        self.tokens
            .get(index)
            .map(|TokenWithLocation { span, .. }| span.clone())
    }

    pub(super) fn expect(&mut self, kind: &Token) -> ParseResult<()> {
        if self.at(kind) {
            self.bump();
            Ok(())
        } else {
            Err(self.token_error(&[*kind]))
        }
    }

    pub(crate) fn expect_one_of(&mut self, kinds: &[Token]) -> ParseResult<&TokenWithLocation<'a>> {
        if self.at_set(kinds) {
            let prev = self.cursor;
            self.cursor += 1;
            let tok = &self.tokens[prev];
            Ok(tok)
        } else {
            Err(self.token_error(kinds))
        }
    }

    pub(super) fn expect_identifier(&mut self) -> ParseResult<String> {
        let tok = self.expect_one_of(&[Identifier])?;
        Ok(unescape_ident(tok.text)?.to_string())
    }

    pub(super) fn expect_identifier_ex(&mut self) -> ParseResult<String> {
        let tok = self.expect_one_of(IDENT_LIKE_TOKENS);
        match tok {
            Ok(tok) => Ok(unescape_ident(tok.text)?.to_string()),
            Err(_) => {
                let span = self.last_token_range().unwrap_or_default();
                Err(invalid_token_error(&[Identifier], None, &span, "".to_string()))
            }
        }
    }

    pub(super) fn token_error(&self, expected: &[Token]) -> ParseError {
        let current_token = self.peek_token();

        if let Some(TokenWithLocation { kind, span, .. }) = current_token {
            invalid_token_error(expected, Some(*kind), span, "".to_string())
        } else {
            // If we’re at the end of the input we use the range of the very last token in the input.
            let span = self.last_token_range().unwrap_or_default();
            invalid_token_error(expected, None, &span, "".to_string())
        }
    }

    pub(super) fn syntax_error(&self, msg: &str) -> ParseError {
        let span = self.last_token_range().unwrap_or_default();
        syntax_error(msg, &span, "".to_string())
    }

    pub(super) fn parse_number(&mut self) -> ParseResult<f64> {
        let token = self.current_token()?;
        let value = parse_number(token.text).map_err(|_| {
            let msg = format!("expect number, got {}", token.text);
            syntax_error(&msg, &token.span, "".to_string())
        })?;

        self.bump();

        Ok(value)
    }

    pub fn parse_duration(&mut self) -> ParseResult<DurationExpr> {
        use Token::*;

        let token = self.expect_one_of(&[Number, Duration, Interval, RateInterval])?;
        if token.kind == Interval || token.kind == RateInterval {
            return Ok(DurationExpr::StepValue(1.0));
        }
        let last_ch = token.text.chars().last().unwrap();

        if token.kind == Number {
            let millis = if last_ch == 'm' || last_ch == 'M' {
                parse_duration_value(token.text, 1)?
            } else {
                let secs = parse_number(token.text)? as i64;
                secs * 1000
            };
            return Ok(DurationExpr::new(millis));
        }

        if last_ch == 'i' || last_ch == 'I' {
            let step_text = &token.text[..token.text.len() - 1];
            return if let Ok(value) = step_text.parse::<f64>() {
                Ok(DurationExpr::StepValue(value))
            } else {
                Err(ParseError::InvalidDuration(token.text.to_string()))
            }
        }

        let millis = parse_duration_value(token.text, 1)?;
        Ok(DurationExpr::new(millis))
    }

    /// returns positive duration in milliseconds for the given s
    /// and the given step.
    ///
    /// Duration in s may be combined, i.e. 2h5m or 2h-5m.
    ///
    /// Error is returned if the duration in s is negative.
    pub fn parse_positive_duration(&mut self) -> ParseResult<DurationExpr> {
        // Verify the duration in seconds without explicit suffix.
        let duration = self.parse_duration()?;
        let val = duration.value(Duration::from_millis(1));
        if val < 0 {
            Err(ParseError::InvalidDuration(duration.to_string()))
        } else {
            Ok(duration)
        }
    }

    pub fn parse_ident_list(&mut self) -> ParseResult<Vec<String>> {
        use Token::*;

        self.expect(&LeftParen)?;
        self.parse_comma_separated(&[RightParen], |parser| parser.expect_identifier())
    }

    pub(super) fn parse_arg_list(&mut self) -> ParseResult<Vec<Expr>> {
        use Token::*;
        self.expect(&LeftParen)?;
        self.parse_comma_separated(&[RightParen], parse_expression)
    }

    /// Parse a comma-separated list of 1+ items accepted by `f`
    pub fn parse_comma_separated<T, F>(
        &mut self,
        stop_tokens: &[Token],
        mut f: F,
    ) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Parser<'a>) -> ParseResult<T>,
    {
        let mut values = Vec::with_capacity(4);
        loop {
            if self.at_set(stop_tokens) {
                self.bump();
                break;
            }
            let item = f(self)?;
            values.push(item);

            let kind = self.peek_kind();
            if kind == Token::Comma {
                self.bump();
                continue;
            } else if stop_tokens.contains(&kind) {
                self.bump();
                break;
            } else {
                let mut expected = Vec::from(stop_tokens);
                expected.insert(0, Token::Comma);
                return Err(self.token_error(&expected));
            }
        }
        Ok(values)
    }

    pub(super) fn parse_string_expression(&mut self, accept_identifiers: bool) -> ParseResult<StringExpr> {
        use Token::*;

        let mut tok = self.current_token()?;
        let mut result = StringExpr::default();

        loop {
            match &tok.kind {
                StringLiteral => {
                    let value = extract_string_value(tok.text)?;
                    if !value.is_empty() {
                        result.push_str(&value);
                    }
                }
                Identifier => {
                    if !accept_identifiers {
                        let msg = "identifiers are not allowed in this context";
                        return Err(self.syntax_error(msg));
                    }
                    let ident = unescape_ident(tok.text)?;
                    result.push_ident(&ident);
                    self.needs_expansion = true;
                }
                _ => {
                    return if accept_identifiers {
                        Err(self.token_error(&[StringLiteral, Identifier]))
                    } else {
                        Err(self.token_error(&[StringLiteral]))
                    };
                }
            }

            if let Some(token) = self.next_token() {
                tok = token;
            } else {
                break;
            }

            if tok.kind != OpPlus {
                break;
            }

            // composite StringExpr like `"s1" + "s2"`, `"s" + m()` or `"s" + m{}` or `"s" + unknownToken`.
            if let Some(token) = self.next_token() {
                tok = token;
            } else {
                break;
            }

            if tok.kind == StringLiteral {
                // "s1" + "s2"
                continue;
            }

            if tok.kind != Identifier {
                // "s" + unknownToken
                self.back();
                break;
            }

            // Look after ident
            match self.next_token() {
                None => break,
                Some(t) => {
                    tok = t;
                    if tok.kind == LeftParen || tok.kind == LeftBrace {
                        self.back();
                        self.back();
                        // `"s" + m(` or `"s" + m{`
                        break;
                    }
                }
            }

            // "s" + ident
            tok = self.prev_token().unwrap();
        }

        Ok(result)
    }

    pub(super) fn parse_parens_expr(&mut self) -> ParseResult<Expr> {
        let mut list = self.parse_arg_list()?;
        if list.len() == 1 {
            if let Expr::BinaryOperator(be) = list[0].borrow_mut() {
                if self.at(&Token::KeepMetricNames) {
                    self.bump();
                    be.set_keep_metric_names();
                }
            }
        }
        Ok(Expr::Parens(ParensExpr::new(list)))
    }

    pub(super) fn bump(&mut self) {
        if self.cursor < self.tokens.len() {
            self.cursor += 1;
        }
    }

    pub(super) fn back(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub(super) fn at(&self, kind: &Token) -> bool {
        let actual = self.peek_kind();
        &actual == kind
    }

    pub(super) fn at_set(&self, set: &[Token]) -> bool {
        let kind = self.peek_kind();
        set.contains(&kind)
    }

    pub(crate) fn at_end(&self) -> bool {
        self.cursor >= self.tokens.len()
    }

    pub(super) fn peek_kind(&self) -> Token {
        if self.at_end() {
            return Token::Eof;
        }
        // self.kind
        let tok = self.tokens.get(self.cursor);
        tok.expect("BUG: invalid index out of bounds").kind
    }
}
