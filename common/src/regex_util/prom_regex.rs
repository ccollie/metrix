use super::{compile_regexp, StringMatchHandler};
use crate::prelude::compile_regexp_anchored;
use regex::Error as RegexError;

/// PromRegex implements an optimized string matching for Prometheus-like regex.
///
/// The following regexes are optimized:
///
/// - plain string such as "foobar"
/// - alternate strings such as "foo|bar|baz"
/// - prefix match such as "foo.*" or "foo.+"
/// - substring match such as ".*foo.*" or ".+bar.+"
///
/// The rest of regexps are also optimized by returning cached match results for the same input strings.
#[derive(Clone, Debug)]
pub struct PromRegex {
    pub matcher: StringMatchHandler,
}

impl Default for PromRegex {
    fn default() -> Self {
        Self::new(".*").unwrap()
    }
}

impl PromRegex {
    pub fn new(expr: &str) -> Result<PromRegex, RegexError> {
        let (matcher, _) = compile_regexp_anchored(expr)
            .map_err(RegexError::Syntax)?;
        let pr = PromRegex {
            matcher,
        };
        Ok(pr)
    }

    /// is_match returns true if s matches pr.
    ///
    /// The pattern is automatically anchored to the beginning and to the end
    /// of the matching string with '^' and '$'.
    pub fn is_match(&self, s: &str) -> bool {
        self.matcher.matches(s)
    }
}