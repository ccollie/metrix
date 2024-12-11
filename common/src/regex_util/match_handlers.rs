use std::any::Any;
use crate::bytes_util::FastRegexMatcher;
use regex::Regex;
use std::fmt::{Display, Formatter};
use crate::regex_util::fast_matcher::StringMatcher;

pub type MatchFn = fn(pattern: &str, candidate: &str) -> bool;

#[derive(Copy, Clone)]
pub enum Quantifier {
    ZeroOrOne, // ?
    ZeroOrMore, // *
    OneOrMore, // +
}

#[derive(Default)]
pub struct StringMatchOptions {
    pub anchor_end: bool,
    pub anchor_start: bool,
    pub prefix_quantifier: Option<Quantifier>,
    pub suffix_quantifier: Option<Quantifier>
}

impl StringMatchOptions {
    pub fn is_default(&self) -> bool {
        self.anchor_end == false &&
        self.anchor_start == false &&
        self.prefix_quantifier.is_none() &&
        self.suffix_quantifier.is_none()
    }
}

#[derive(Clone, Debug)]
pub struct ZeroOrOneCharsMatcher {
    match_nl: bool,
}

#[derive(Clone, Debug)]
pub struct LiteralPrefixStringMatcher {
    prefix: String,
    right: Option<Box<StringMatchHandler>>,
}

impl LiteralPrefixStringMatcher {
    fn matches(&self, s: &str) -> bool {
        if !s.starts_with(&self.prefix) {
            return false;
        }
        if let Some(right) = &self.right {
            right.matches(&s[self.prefix.len()..])
        } else {
            true
        }
    }
}

#[derive(Clone, Debug)]
pub struct LiteralSuffixStringMatcher {
    left: Option<Box<StringMatchHandler>>,
    suffix: String,
}

impl LiteralSuffixStringMatcher {
    fn matches(&self, s: &str) -> bool {
        if !s.ends_with(&self.suffix) {
            return false;
        }
        if let Some(left) = &self.left {
            left.matches(&s[..s.len() - self.suffix.len()])
        } else {
            true
        }
    }
}

#[derive(Clone, Debug)]
pub enum StringMatchHandler {
    MatchAll,
    MatchNone,
    Empty,
    NotEmpty,
    EqualMultiString(Vec<String>),
    Literal(String),
    Contains(String),
    StartsWith(String),
    EndsWith(String),
    FastRegex(FastRegexMatcher),
    OrderedAlternates(Vec<String>),
    MatchFn(MatchFnHandler),
    Alternates(Vec<String>, MatchFn),
    And(Box<StringMatchHandler>, Box<StringMatchHandler>),
    Or(Vec<Box<StringMatchHandler>>),
    ZeroOrOneChars(ZeroOrOneCharsMatcher),
}

impl Default for StringMatchHandler {
    fn default() -> Self {
        Self::MatchAll
    }
}

impl StringMatchHandler {
    #[allow(dead_code)]
    pub fn match_fn(pattern: String, match_fn: MatchFn) -> Self {
        Self::MatchFn(MatchFnHandler::new(pattern, match_fn))
    }

    pub fn empty_string_match() -> Self {
        Self::Empty
    }

    pub fn fast_regex(regex: Regex) -> Self {
        Self::FastRegex(FastRegexMatcher::new(regex))
    }

    pub fn alternates(alts: Vec<String>, options: &StringMatchOptions) -> Self {
        let mut alts = alts;
        if options.is_default() {
            if alts.len() == 1 {
                return Self::Literal(alts.pop().unwrap());
            }
            return Self::Alternates(alts, contains_fn);
        }
        let match_fn = get_literal_match_fn(options);
        if alts.len() == 1 {
            let pattern = alts.pop().unwrap();
            return Self::MatchFn(MatchFnHandler{
                pattern,
                match_fn
            });
        }
        Self::Alternates(alts, match_fn)
    }

    pub fn literal_alternates(alts: Vec<String>) -> Self {
        if alts.len() == 1 {
            let mut alts = alts;
            Self::Literal(alts.pop().unwrap())
        } else {
            Self::Alternates(alts, equals_fn)
        }
    }

    pub fn literal(value: String, options: &StringMatchOptions) -> Self {
        get_optimized_literal_matcher(value, options)
    }

    pub fn equals(value: String) -> Self {
        StringMatchHandler::Literal(value)
    }

    pub fn and(self, b: StringMatchHandler) -> Self {
        Self::And(Box::new(self), Box::new(b))
    }

    #[allow(dead_code)]
    pub fn matches(&self, s: &str) -> bool {
        match self {
            StringMatchHandler::MatchAll => true,
            StringMatchHandler::MatchNone => false,
            StringMatchHandler::Alternates(alts, match_fn) => matches_alternates(alts, s, match_fn),
            StringMatchHandler::MatchFn(m) => m.matches(s),
            StringMatchHandler::FastRegex(r) => r.matches(s),
            StringMatchHandler::OrderedAlternates(m) => match_ordered_alternates(m, s),
            StringMatchHandler::And(a, b) => a.matches(s) && b.matches(s),
            StringMatchHandler::Contains(value) => s.contains(value),
            StringMatchHandler::StartsWith(prefix) => s.starts_with(prefix),
            StringMatchHandler::EndsWith(suffix) => s.ends_with(suffix),
            StringMatchHandler::Literal(val) => s == val,
            StringMatchHandler::Empty => s.is_empty(),
            StringMatchHandler::NotEmpty => !s.is_empty(),
            StringMatchHandler::Or(matchers) => {
                matchers.iter().any(|m| m.matches(s))
            }
            StringMatchHandler::ZeroOrOneChars(m) => {
                if m.match_nl {
                    s.is_empty() || s.chars().count() == 1
                } else {
                    s.is_empty() || (s.chars().count() == 1 && s.chars().next().unwrap() != '\n')
                }
            }
            StringMatchHandler::EqualMultiString(values) => {
                values.iter().any(|v| s == *v)
            }
        }
    }
}


#[derive(Clone, Debug)]
pub struct MatchFnHandler {
    pattern: String,
    pub(super) match_fn: MatchFn,
}

impl MatchFnHandler {
    pub(super) fn new<T: Into<String>>(pattern: T, match_fn: MatchFn) -> Self {
        Self {
            pattern: pattern.into(),
            match_fn,
        }
    }

    #[allow(dead_code)]
    pub(super) fn matches(&self, s: &str) -> bool {
        (self.match_fn)(&self.pattern, s)
    }
}

impl Display for StringMatchHandler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[inline]
fn matches_alternates(alternates: &[String], haystack: &str, match_fn: &MatchFn) -> bool {
    alternates.iter().any(|v| match_fn(v, haystack))
}

const fn get_literal_match_fn(options: &StringMatchOptions) -> MatchFn {

    let StringMatchOptions {
        anchor_start,
        anchor_end,
        prefix_quantifier,
        suffix_quantifier,
    } = options;

    // ^foobar.+
    fn start_with_dot_plus_fn(needle: &str, haystack: &str) -> bool {
        haystack.len() > needle.len() && haystack.starts_with(needle)
    }

    // ^.?foo
    fn start_with_zero_or_one_chars_fn(needle: &str, haystack: &str) -> bool {
        if haystack.is_empty() || haystack.chars().count() == 1 {
            if needle.is_empty() {
                return true;
            }
            if let Some(pos) = haystack.find(needle) {
                return pos == 0 || pos == 1;
            }
        }
        false
    }

    fn ends_with_zero_or_one_chars_fn(needle: &str, haystack: &str) -> bool {
        if haystack.len() < needle.len() {
            return false;
        }
        if let Some(pos) = haystack.rfind(needle) {
            let end = pos + needle.len();
            end == haystack.len() || end == haystack.len() - 1
        } else {
            false
        }
    }

    // .?foo.?$
    fn zero_or_one_chars_anchors_fn(needle: &str, haystack: &str) -> bool {
        if haystack.len() < needle.len() {
            return false;
        }
        if let Some(pos) = haystack.find(needle) {
            let end = pos + needle.len();
            end == haystack.len() || end == haystack.len() - 1
        } else {
            false
        }
    }

    // xxx.?foo
    fn zero_or_one_contains_left_fn(needle: &str, haystack: &str) -> bool {
        let mut haystack = haystack;
        while let Some(pos) = haystack.find(needle) {
            if pos >= 0 {
                return true;
            }
            haystack = &haystack[pos + 1..];
        }
        if let Some(pos) = haystack.find(needle) {
            pos == 0 || pos == 1
        } else {
            false
        }
    }

    // foo.?
    fn zero_or_one_contains_right_fn(needle: &str, haystack: &str) -> bool {
        if let Some(pos) = haystack.rfind(needle) {
            let end = pos + needle.len();
            end <= haystack.len() - 1
        } else {
            false
        }
    }

    // something like .*foo.+$
    fn contains_dot_plus_fn(needle: &str, haystack: &str) -> bool {
        if let Some(pos) = haystack.find(needle) {
            let end = pos + needle.len();
            end < haystack.len()
        } else {
            false
        }
    }

    fn dot_plus_fn(needle: &str, haystack: &str) -> bool {
        if let Some(pos) = haystack.find(needle) {
            pos > 0
        } else {
            false
        }
    }

    if *anchor_start && *anchor_end {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrOne), Some(Quantifier::ZeroOrOne)) => {
                // ^.?foo.?$
                zero_or_one_chars_anchors_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::OneOrMore)) => {
                // ^.*foo.+$
                contains_dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.+foo.*$
                dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::OneOrMore)) => {
                // ^.+foo.+$
                dot_plus_dot_plus_fn
            }
            (Some(Quantifier::ZeroOrOne), None) => {
                // ^.?foo$
                start_with_zero_or_one_chars_fn
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // ^.*foo$
                ends_with_fn
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // ^foo.*$
                starts_with_fn
            }
            (Some(Quantifier::OneOrMore), None) => {
                // ^.+foo$
                dot_plus_ends_with_fn
            }
            (None, Some(Quantifier::OneOrMore)) => {
                // ^foo.+$
                start_with_dot_plus_fn
            }
            (None, Some(Quantifier::ZeroOrOne)) => {
                // ^foo.?$
                ends_with_zero_or_one_chars_fn
            }
            _ => {
                // ^foobar$
                equals_fn
            }
        }
    } else if *anchor_start {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrOne), Some(Quantifier::ZeroOrOne)) => {
                // ^.?foo.?
                start_with_zero_or_one_chars_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.*foo.*
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::OneOrMore)) => {
                // ^.*foo.+
                contains_dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.+foo.*
                dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::OneOrMore)) => {
                // ^.+foo.+
                dot_plus_dot_plus_fn
            }
            (Some(Quantifier::ZeroOrOne), None) => {
                start_with_zero_or_one_chars_fn
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // ^.*foo
                contains_fn
            }
            (None, Some(Quantifier::ZeroOrOne)) => {
                ends_with_zero_or_one_chars_fn
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // ^foo.*
                starts_with_fn
            }
            (Some(Quantifier::OneOrMore), None) => {
                // ^.+foo
                dot_plus_ends_with_fn
            }
            (None, Some(Quantifier::OneOrMore)) => {
                // ^foo.+
                start_with_dot_plus_fn
            }
            _ => {
                // ^foobar
                starts_with_fn
            }
        }
    } else if *anchor_end {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrOne), Some(Quantifier::ZeroOrOne)) => {
                // .?foo.?$
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .*foo.*$
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::OneOrMore)) => {
                // .*foo.+$
                contains_dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .+foo.*$
                dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::OneOrMore)) => {
                // .+foo.+$
                dot_plus_dot_plus_fn
            }
            (Some(Quantifier::ZeroOrOne), None) => {
                // .?foo$
                ends_with_fn
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // .*foo$
                ends_with_fn
            }
            (None, Some(Quantifier::ZeroOrOne)) => {
                // foo.?$
                ends_with_zero_or_one_chars_fn
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // foo.*$
                contains_fn
            }
            (Some(Quantifier::OneOrMore), None) => {
                // .+foo$
                dot_plus_ends_with_fn
            }
            (None, Some(Quantifier::OneOrMore)) => {
                // foo.+$
                prefix_dot_plus_fn
            }
            _ => {
                // foobar$
                ends_with_fn
            }
        }
    } else {
        // no anchors
        match(prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrOne), Some(Quantifier::ZeroOrOne)) => {
                // .?foo.?
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .*foo.*
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::OneOrMore)) => {
                // .*foo.+
                contains_dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .+foo.*
                dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::OneOrMore)) => {
                // .+foo.+
                dot_plus_dot_plus_fn
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // .*foo
                contains_fn
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // foo.*
                contains_fn
            }
            (Some(Quantifier::OneOrMore), None) => {
                // .+foo
                dot_plus_fn
            }
            (None, Some(Quantifier::OneOrMore)) => {
                // foo.+
                contains_dot_plus_fn
            }
            _ => {
                // foobar
                contains_fn
            }
        }
    }
}

fn get_optimized_literal_matcher(value: String, options: &StringMatchOptions) -> StringMatchHandler {
    let StringMatchOptions {
        anchor_start,
        anchor_end,
        prefix_quantifier,
        suffix_quantifier,
    } = options;

    fn handle_default(options: &StringMatchOptions, value: String) -> StringMatchHandler {
        let match_fn = get_literal_match_fn(options);
        StringMatchHandler::MatchFn(MatchFnHandler::new(value, match_fn))
    }

    if *anchor_start && *anchor_end {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.*foo.*$
                StringMatchHandler::Contains(value)
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // ^.*foo$
                StringMatchHandler::EndsWith(value)
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // ^foo.*$
                StringMatchHandler::StartsWith(value)
            }
            (None, None) => {
                // ^foobar$
                StringMatchHandler::Literal(value)
            }
            _ => {
                handle_default(options, value)
            }
        }
    } else if *anchor_start {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.*foo.*
                StringMatchHandler::Contains(value)
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // ^.*foo
                StringMatchHandler::Contains(value)
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // ^foo.*
                StringMatchHandler::StartsWith(value)
            }
            (None, None) => {
                // ^foobar
                StringMatchHandler::StartsWith(value)
            }
            _ => {
                handle_default(options, value)
            }
        }
    } else if *anchor_end {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .*foo.*$
                StringMatchHandler::Contains(value)
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // .*foo$
                StringMatchHandler::EndsWith(value)
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // foo.*$
                StringMatchHandler::Contains(value)
            }
            (None, None) => {
                // foobar$
                StringMatchHandler::EndsWith(value)
            }
            _ => {
                // foobar$
                handle_default(options, value)
            }
        }
    } else {
        // no anchors
        match(prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .*foo.*
                StringMatchHandler::Contains(value)
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // .*foo
                StringMatchHandler::Contains(value)
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // foo.*
                StringMatchHandler::Contains(value)
            }
            (None, None) => {
                // foobar
                StringMatchHandler::Contains(value)
            }
            _ => {
                // foobar
                handle_default(options, value)
            }
        }
    }
}

fn equals_fn(needle: &str, haystack: &str) -> bool {
    haystack == needle
}

fn contains_fn(needle: &str, haystack: &str) -> bool {
    haystack.contains(needle)
}

fn starts_with_fn(needle: &str, haystack: &str) -> bool {
    haystack.starts_with(needle)
}

fn ends_with_fn(needle: &str, haystack: &str) -> bool {
    haystack.ends_with(needle)
}

// foobar.+
fn prefix_dot_plus_fn(needle: &str, haystack: &str) -> bool {
    if let Some(pos) = haystack.find(needle) {
        pos + needle.len() < haystack.len() - 1
    } else {
        false
    }
}


// ^.+(foo|bar)$ / .+(foo|bar)$
fn dot_plus_ends_with_fn(needle: &str, haystack: &str) -> bool {
    haystack.len() > needle.len() && haystack.ends_with(needle)
}


// ^.+(foo|bar).+
fn dot_plus_dot_plus_fn(needle: &str, haystack: &str) -> bool {
    if let Some(pos) = haystack.find(needle) {
        pos > 0 && pos + needle.len() < haystack.len()
    } else {
        false
    }
}

fn match_ordered_alternates(or_values: &[String], s: &str) -> bool {
    if or_values.is_empty() {
        return false;
    }
    let mut cursor = &s[0..];
    for literal in or_values.iter() {
        if let Some(pos) = cursor.find(literal) {
            cursor = &cursor[pos + 1..];
        } else {
            return false;
        }
    }
    true
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains() {
        let handler = StringMatchHandler::Contains("a".to_string());
        assert!(handler.matches("a"));
        assert!(handler.matches("ba"));
        assert!(!handler.matches("b"));
    }
}