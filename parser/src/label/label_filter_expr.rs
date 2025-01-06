use serde::{Deserialize, Serialize};
use std::fmt;

use crate::ast::StringExpr;
use crate::label::{Matcher, MatchOp, LabelName, NAME_LABEL};
use crate::parser::{compile_regexp, escape_ident, is_empty_regex, ParseError, ParseResult};

/// LabelFilterExpr represents `foo <op> ident + "bar"` expression, where <op> is `=`, `!=`, `=~` or `!~`.
/// For internal use only, in the context of WITH expressions
#[derive(Default, Debug, Clone, Eq, Serialize, Deserialize)]
pub struct LabelFilterExpr {
    pub op: MatchOp,

    /// Label contains label name for the filter.
    pub label: String,

    /// Value contains unquoted value for the filter.
    pub value: StringExpr,

    is_variable: bool,
}

impl LabelFilterExpr {
    pub fn new<N>(label: N, match_op: MatchOp, value: StringExpr) -> ParseResult<Self>
    where
        N: Into<LabelName>,
    {
        let label = label.into();

        assert!(!label.is_empty());

        if (match_op == MatchOp::RegexEqual || match_op == MatchOp::RegexNotEqual)
            && value.is_expanded()
        {
            let resolved_value = value.to_string();
            if compile_regexp(&resolved_value).is_err() {
                return Err(ParseError::InvalidRegex(resolved_value));
            }
        }

        Ok(Self {
            label,
            op: match_op,
            value,
            is_variable: false,
        })
    }

    pub fn named(name: &str) -> Self {
        Self {
            label: NAME_LABEL.to_string(),
            op: MatchOp::Equal,
            value: StringExpr::from(name),
            is_variable: false,
        }
    }

    pub(crate) fn variable(name: &str) -> Self {
        Self {
            label: NAME_LABEL.to_string(),
            op: MatchOp::Equal,
            value: StringExpr::from(name),
            is_variable: true,
        }
    }

    pub fn equal<S: Into<String>>(key: S, value: StringExpr) -> ParseResult<LabelFilterExpr> {
        LabelFilterExpr::new(key, MatchOp::Equal, value)
    }

    pub fn not_equal<S: Into<String>>(key: S, value: StringExpr) -> ParseResult<LabelFilterExpr> {
        LabelFilterExpr::new(key, MatchOp::NotEqual, value)
    }

    /// is_negative represents whether the filter is negative, i.e. '!=' or '!~'.
    pub fn is_negative(&self) -> bool {
        self.op.is_negative()
    }

    /// is_regexp represents whether the filter is regexp, i.e. `=~` or `!~`.
    pub fn is_regexp(&self) -> bool {
        self.op.is_regex()
    }

    pub fn is_metric_name_filter(&self) -> bool {
        self.label == NAME_LABEL && self.op == MatchOp::Equal
    }

    pub fn is_name_label(&self) -> bool {
        self.label == NAME_LABEL && self.op == MatchOp::Equal
    }

    pub fn set_value<S: Into<String>>(&mut self, value: S) {
        self.value = StringExpr::from(value.into())
    }

    pub fn as_string(&self) -> String {
        format!("{}{}{}", escape_ident(&self.label), self.op, self.value)
    }

    pub fn is_resolved(&self) -> bool {
        self.value.is_literal_only() && !self.is_variable
    }

    pub fn is_variable(&self) -> bool {
        self.is_variable
    }

    pub fn name(&self) -> String {
        if self.label == NAME_LABEL {
            if self.value.is_literal_only() {
                // todo: better error handling
                if let Some(value) = self.value.get_literal().unwrap() {
                    return value.to_string();
                }
            }
            if let Some(ident) = self.value.as_identifier() {
                return ident.to_string();
            }
            // todo: panic
            return self.value.to_string();
        }
        self.label.clone()
    }

    /// Vector selectors must either specify a name or at least one label
    /// matcher that does not match the empty string.
    ///
    /// The following expression is illegal:
    /// {job=~".*"} # Bad!
    pub fn is_empty_matcher(&self) -> bool {
        use MatchOp::*;
        match self.op {
            Equal => self.value.is_empty(),
            NotEqual => !self.value.is_empty(),
            RegexEqual => {
                let str = self.value.to_string();
                is_empty_regex(&str)
            }
            RegexNotEqual => {
                let str = self.value.to_string();
                is_empty_regex(&str)
            }
        }
    }

    pub fn is_match(&self, str: &str) -> bool {
        use MatchOp::*;
        let haystack = self.value.to_string();
        match self.op {
            Equal => haystack.eq(str),
            NotEqual => haystack.ne(str),
            RegexEqual => compile_regexp(&haystack)
                .map(|re| re.is_match(str))
                .unwrap_or(false),
            RegexNotEqual => {
                let str = self.value.to_string();
                compile_regexp(&haystack)
                    .map(|re| !re.is_match(&str))
                    .unwrap_or(false)
            }
        }
    }

    pub fn to_matcher(&self) -> ParseResult<Matcher> {
        let empty_str = "".to_string();
        let value = self.value.get_literal()?.unwrap_or(&empty_str).to_string();
        Matcher::new(self.op, &self.label, value)
    }

    pub fn into_matcher(self) -> ParseResult<Matcher> {
        let label = self.label;
        let value = self.value.into_literal()?;
        Matcher::new(self.op, label, value)
    }

    //
    pub fn is_raw_ident(&self) -> bool {
        self.value.is_empty()
            && self.op == MatchOp::Equal
            && !self.label.is_empty()
            && self.label != NAME_LABEL
    }
}

impl PartialOrd for LabelFilterExpr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let cmp = self.label.cmp(&other.label);
        if cmp != std::cmp::Ordering::Equal {
            return Some(cmp);
        }
        if let Some(cmp) = self.value.partial_cmp(&other.value) {
            if cmp != std::cmp::Ordering::Equal {
                return Some(cmp);
            }
        }
        Some(std::cmp::Ordering::Equal)
    }
}

impl PartialEq for LabelFilterExpr {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.op == other.op && self.value == other.value
    }
}

impl fmt::Display for LabelFilterExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // todo: this is a mess. refactor
        if self.is_variable || self.is_metric_name_filter() {
            write!(f, "{}", escape_ident(&self.value.to_string()))?;
            return Ok(());
        }
        if self.is_raw_ident() {
            write!(f, "{}", escape_ident(&self.label))?;
            return Ok(());
        }
        write!(f, "{}{}{}", escape_ident(&self.label), self.op, &self.value)?;
        Ok(())
    }
}
