use crate::regex_util::string_pattern::StringPattern;
use get_size::GetSize;
use regex::Regex;
use std::collections::HashSet;
use std::fmt::{Display, Formatter};

const MAX_SET_MATCHES: usize = 256;

pub type MatchFn = fn(pattern: &str, candidate: &str) -> bool;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct MatchAnyMatcher {
    ignore_nl: bool,
}

impl MatchAnyMatcher {
    pub fn new(ignore_nl: bool) -> Self {
        Self {
            ignore_nl
        }
    }

    fn matches(&self, s: &str) -> bool {
        if !self.ignore_nl {
            !s.contains('\n')
        } else {
            true
        }
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct NonEmptyStringMatcher {
    match_nl: bool,
}

impl NonEmptyStringMatcher {
    pub fn new(match_nl: bool) -> Self {
        Self {
            match_nl
        }
    }

    fn matches(&self, s: &str) -> bool {
        if self.match_nl {
            !s.is_empty()
        } else {
            !s.is_empty() && !s.contains('\n')
        }
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct ZeroOrOneCharsMatcher {
    match_nl: bool,
}

impl ZeroOrOneCharsMatcher {
    fn matches(&self, s: &str) -> bool {
        if self.match_nl {
            s.is_empty() || s.chars().count() == 1
        } else {
            s.is_empty() || (s.chars().count() == 1 && s.chars().next().unwrap() != '\n')
        }
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct EqualMultiStringMatcher {
    pub values: Vec<String>,
    is_ascii: bool,
    case_sensitive: bool,
}

impl EqualMultiStringMatcher {
    pub(crate) fn new(case_sensitive: bool, estimated_size: usize) -> Self {
        EqualMultiStringMatcher {
            values: Vec::with_capacity(estimated_size),
            case_sensitive,
            is_ascii: true,
        }
    }

    pub fn push(&mut self, s: String) {
        self.is_ascii = self.is_ascii && s.is_ascii();
        self.values.push(s);
    }

    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    pub fn matches(&self, s: &str) -> bool {
        if self.case_sensitive {
            self.values.iter().any(|v| v == s)
        } else if self.is_ascii {
            self.values.iter().any(|v| v.eq_ignore_ascii_case(s))
        } else {
            let needle = s.to_lowercase();
            self.values.iter().any(|v| v.to_lowercase() == needle)
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
}


#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct LiteralPrefixMatcher {
    pub prefix: StringPattern,
    pub right: Option<Box<StringMatchHandler>>,
}

impl LiteralPrefixMatcher {
    pub fn new<S: Into<String>>(prefix: S, right: Option<Box<StringMatchHandler>>, case_sensitive: bool) -> Self {
        let prefix = StringPattern::new(prefix.into(), case_sensitive);
        Self {
            prefix,
            right,
        }
    }

    fn is_case_sensitive(&self) -> bool {
        self.prefix.is_case_sensitive()
    }
}

impl LiteralPrefixMatcher {
    fn matches(&self, s: &str) -> bool {
        if !self.prefix.starts_with(s) {
            return false;
        }
        if let Some(right) = &self.right {
            let right_part = &s[self.prefix.len()..];
            right.matches(right_part)
        } else {
            true
        }
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct LiteralSuffixMatcher {
    pub left: Option<Box<StringMatchHandler>>,
    pub suffix: StringPattern,
}

impl LiteralSuffixMatcher {
    pub fn new<S: Into<String>>(left: Option<Box<StringMatchHandler>>, suffix: S, case_sensitive: bool) -> Self {
        let suffix = StringPattern::new(suffix.into(), case_sensitive);
        Self {
            left,
            suffix,
        }
    }

    fn is_case_sensitive(&self) -> bool {
        self.suffix.is_case_sensitive()
    }
}

impl LiteralSuffixMatcher {
    fn matches(&self, s: &str) -> bool {
        if !self.suffix.ends_with(s) {
            return false;
        }
        if let Some(left) = &self.left {
            if s.len() < self.suffix.len() {
                return false;
            }
            let left_part = &s[..s.len() - self.suffix.len()];
            left.matches(left_part)
        } else {
            true
        }
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct ContainsMultiStringMatcher {
    pub substrings: Vec<String>,
    pub left: Option<Box<StringMatchHandler>>,
    pub right: Option<Box<StringMatchHandler>>,
}

impl ContainsMultiStringMatcher {
    pub(crate) fn new(substrings: Vec<String>, left: Option<StringMatchHandler>, right: Option<StringMatchHandler>) -> Self {
        let left = left.map(|l| Box::new(l));
        let right = right.map(|r| Box::new(r));
        Self {
            substrings,
            left,
            right,
        }
    }

    fn matches(&self, s: &str) -> bool {
        for substr in &self.substrings {
            match (self.left.as_ref(), self.right.as_ref()) {
                (Some(left), Some(right)) => {
                    let mut search_start_pos = 0;
                    while let Some(pos) = s[search_start_pos..].find(substr) {
                        let pos = pos + search_start_pos;
                        if left.matches(&s[..pos]) && right.matches(&s[pos + substr.len()..]) {
                            return true;
                        }
                        search_start_pos = pos + 1;
                    }
                }
                (Some(left), None) => {
                    if s.ends_with(substr) && left.matches(&s[..s.len() - substr.len()]) {
                        return true;
                    }
                }
                (None, Some(right)) => {
                    if s.starts_with(substr) && right.matches(&s[substr.len()..]) {
                        return true;
                    }
                }
                (None, None) => {
                    if s.contains(substr) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct LiteralMapMatcher {
    pub values: HashSet<String>,
    pub is_case_sensitive: bool,
}

impl LiteralMapMatcher {
    pub(crate) fn new() -> Self {
        Self {
            values: Default::default(),
            is_case_sensitive: true,
        }
    }

    pub(crate) fn push(&mut self, s: String) {
        if self.is_case_sensitive {
            self.values.insert(s.to_lowercase());
        } else {
            self.values.insert(s);
        }
    }

    pub fn set_matches(&self) -> Vec<String> {
        if self.values.len() >= MAX_SET_MATCHES {
            return Vec::new();
        }

        self.values.iter().cloned().collect::<Vec<String>>()
    }

    fn matches(&self, s: &str) -> bool {
        if self.is_case_sensitive {
            self.values.contains(&s.to_lowercase())
        } else {
            self.values.contains(s)
        }
    }
    pub fn is_case_sensitive(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Eq, PartialEq, GetSize)]
pub struct RepetitionMatcher {
    pub sub: String,
    pub min: u32,
    pub max: Option<u32>,
}

impl RepetitionMatcher {
    pub fn new(sub: String, min: u32, max: Option<u32>) -> Self {
        Self {
            sub,
            min,
            max
        }
    }

    pub fn matches(&self, s: &str) -> bool {
        if self.min == 0 && s.is_empty(){
            return true;
        }
        if self.min == 1 && s == self.sub {
            return true;
        }
        if let Some(max) = &self.max {
            let pat_len = self.sub.len();
            let mut cursor = &s[..];
            let mut i = 0;

            while i <= (*max + 1) {
                if !cursor.starts_with(&self.sub) {
                    return i >= self.min;
                }
                i += 1;
                if i > *max {
                    return false;
                }
                cursor = &cursor[pat_len..];
                if cursor.len() < pat_len {
                    return i >= self.min;
                }
            }
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct RegexMatcher {
    pub regex: Regex,
    pub prefix: String,
    pub suffix: String,
    pub set_matches: Vec<String>,
    pub contains: Vec<String>,
}

impl GetSize for RegexMatcher {
    fn get_size(&self) -> usize {
        // TODO: properly calculate a value for the bookkeeping overhead of the regex object
        const REGEX_OVERHEAD: usize = 256;
        REGEX_OVERHEAD + self.regex.as_str().get_size() + self.prefix.get_size() + self.suffix.get_size()
    }
}

impl PartialEq for RegexMatcher {
    fn eq(&self, other: &Self) -> bool {
        self.regex.as_str() == other.regex.as_str() &&
            self.prefix == other.prefix &&
            self.suffix == other.suffix &&
            self.set_matches == other.set_matches &&
            self.contains == other.contains
    }
}

impl Eq for RegexMatcher {}

impl RegexMatcher {
    pub(crate) fn new(regex: Regex, prefix: String, suffix: String) -> Self {
        Self {
            regex,
            prefix,
            suffix,
            set_matches: vec![],
            contains: vec![],
        }
    }

    fn matches(&self, s: &str) -> bool {
        if !self.set_matches.is_empty() && !self.set_matches.iter().any(|x| x.as_str() == s) {
            return false;
        }
        if !self.prefix.is_empty() && !s.starts_with(&self.prefix) {
            return false;
        }
        if !self.suffix.is_empty() && !s.ends_with(&self.suffix) {
            return false;
        }
        if !self.contains.is_empty() && !contains_in_order(s, &self.contains) {
            return false;
        }
        self.regex.is_match(s)
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub enum StringMatchHandler {
    MatchAny(MatchAnyMatcher),
    MatchNone,
    Empty,
    NotEmpty(NonEmptyStringMatcher),
    Literal(StringPattern),
    /// Alteration of literals
    Alternates(EqualMultiStringMatcher),
    LiteralMap(LiteralMapMatcher),
    ContainsMulti(ContainsMultiStringMatcher),
    Prefix(LiteralPrefixMatcher),
    Suffix(LiteralSuffixMatcher),
    Regex(RegexMatcher),
    Repetition(RepetitionMatcher),
    MatchFn(MatchFnHandler),
    Or(Vec<Box<StringMatchHandler>>),
    ZeroOrOneChars(ZeroOrOneCharsMatcher),
}

impl Default for StringMatchHandler {
    fn default() -> Self {
        Self::MatchAny(MatchAnyMatcher { ignore_nl: false })
    }
}

impl StringMatchHandler {
    pub fn any(ignore_nl: bool) -> Self {
        Self::MatchAny(MatchAnyMatcher::new(ignore_nl))
    }

    #[allow(dead_code)]
    pub fn match_fn(pattern: String, match_fn: MatchFn) -> Self {
        Self::MatchFn(MatchFnHandler::new(pattern, match_fn))
    }

    pub fn empty_string_match() -> Self {
        Self::Empty
    }

    pub fn fast_regex(regex: Regex) -> Self {
        Self::Regex(RegexMatcher::new(regex, String::new(), String::new()))
    }

    pub fn literal_alternates(alts: Vec<String>, case_sensitive: bool) -> Self {
        if alts.len() == 1 {
            let mut alts = alts;
            let pattern = alts.pop().unwrap();
            if case_sensitive {
                Self::Literal(StringPattern::case_sensitive(pattern))
            } else {
                Self::Literal(StringPattern::case_insensitive(pattern))
            }
        } else {
            let mut matcher = EqualMultiStringMatcher::new(true, alts.len());
            matcher.case_sensitive = case_sensitive;
            for alt in alts {
                matcher.push(alt);
            }
            Self::Alternates(matcher)
        }
    }

    pub fn zero_or_one_chars(match_nl: bool) -> Self {
        StringMatchHandler::ZeroOrOneChars(ZeroOrOneCharsMatcher { match_nl })
    }

    pub fn not_empty(match_nl: bool) -> Self {
        StringMatchHandler::NotEmpty(NonEmptyStringMatcher::new(match_nl))
    }

    pub fn literal(value: String, case_sensitive: bool) -> Self {
        let pattern = if case_sensitive {
            StringPattern::case_sensitive(value)
        } else {
            StringPattern::case_insensitive(value)
        };
        StringMatchHandler::Literal(pattern)
    }

    pub fn literal_fn(value: String, options: &StringMatchOptions) -> Self {
        get_optimized_literal_matcher(value, options)
    }
    pub fn equals(value: String) -> Self {
        StringMatchHandler::literal(value.into(), true)
    }

    pub fn prefix(value: String, right: Option<StringMatchHandler>, case_sensitive: bool) -> Self {
        StringMatchHandler::Prefix(LiteralPrefixMatcher::new(value, right.map(Box::new), case_sensitive))
    }

    pub fn suffix(left: Option<StringMatchHandler>, value: String, case_sensitive: bool) -> Self {
        StringMatchHandler::Suffix(LiteralSuffixMatcher::new(left.map(Box::new), value, case_sensitive))
    }

    pub fn is_case_sensitive(&self) -> bool {
        match self {
            StringMatchHandler::Literal(p) => p.is_case_sensitive(),
            StringMatchHandler::Prefix(p) => p.is_case_sensitive(),
            StringMatchHandler::Suffix(p) => p.is_case_sensitive(),
            StringMatchHandler::LiteralMap(p) => p.is_case_sensitive(),
            StringMatchHandler::Alternates(p) => p.is_case_sensitive(),
            _ => true,
        }
    }

    #[allow(dead_code)]
    pub fn matches(&self, s: &str) -> bool {
        match self {
            StringMatchHandler::MatchAny(m) => m.matches(s),
            StringMatchHandler::MatchNone => false,
            StringMatchHandler::MatchFn(m) => m.matches(s),
            StringMatchHandler::Regex(r) => r.matches(s),
            StringMatchHandler::Literal(m) => m.matches(s),
            StringMatchHandler::Empty => s.is_empty(),
            StringMatchHandler::NotEmpty(opts) => opts.matches(s),
            StringMatchHandler::Or(matchers) => {
                matchers.iter().any(|m| m.matches(s))
            }
            StringMatchHandler::ZeroOrOneChars(m) => m.matches(s),
            StringMatchHandler::Alternates(m) => m.matches(s),
            StringMatchHandler::ContainsMulti(m) => m.matches(s),
            StringMatchHandler::Prefix(m) => m.matches(s),
            StringMatchHandler::Suffix(m) => m.matches(s),
            StringMatchHandler::LiteralMap(m) => m.matches(s),
            StringMatchHandler::Repetition(m) => m.matches(s),
        }
    }
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchFnHandler {
    pattern: String,
    pub(super) match_fn: MatchFn,
}

impl GetSize for MatchFnHandler {
    fn get_size(&self) -> usize {
        self.pattern.get_size() + size_of::<MatchFn>()
    }
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
                StringMatchHandler::MatchFn(
                    MatchFnHandler::new(value, contains_fn)
                )
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // ^.*foo$
                StringMatchHandler::suffix(None, value, true)
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // ^foo.*$
                StringMatchHandler::prefix(value, None, true)
            }
            (None, None) => {
                // ^foobar$
                StringMatchHandler::literal(value.into(), true)
            }
            _ => {
                handle_default(options, value)
            }
        }
    } else if *anchor_start {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.*foo.*
                StringMatchHandler::MatchFn(
                    MatchFnHandler::new(value, contains_fn)
                )
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // ^.*foo
                StringMatchHandler::MatchFn(
                    MatchFnHandler::new(value, contains_fn)
                )
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // ^foo.*
                StringMatchHandler::prefix(value, None, true)
            }
            (None, None) => {
                // ^foobar
                StringMatchHandler::suffix(None, value, true)
            }
            _ => {
                handle_default(options, value)
            }
        }
    } else if *anchor_end {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .*foo.*$
                StringMatchHandler::MatchFn(
                    MatchFnHandler::new(value, contains_fn)
                )
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // .*foo$
                StringMatchHandler::suffix(None, value, true)
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // foo.*$
                StringMatchHandler::prefix(value, None, true)
            }
            (None, None) => {
                // foobar$
                StringMatchHandler::suffix(None, value, true)
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
                StringMatchHandler::MatchFn(
                    MatchFnHandler::new(value, contains_fn)
                )
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // .*foo
                StringMatchHandler::Repetition(
                    RepetitionMatcher::new(value, 0, None)
                )
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // foo.*
                StringMatchHandler::prefix(value, None, true)
            }
            (None, None) => {
                // foobar
                StringMatchHandler::literal(value.into(), true)
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

pub fn contains_in_order(s: &str, contains: &[String]) -> bool {
    if contains.len() == 1 {
        return s.contains(&contains[0]);
    }
    contains_in_order_multi(s, contains)
}

fn contains_in_order_multi(s: &str, contains: &[String]) -> bool {
    let mut offset = 0;
    for substr in contains {
        if let Some(pos) = s[offset..].find(substr) {
            offset += pos + substr.len();
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
    fn test_zero_or_one_character_string_matcher() {
        // Test case: match newline
        let matcher_match_nl = ZeroOrOneCharsMatcher { match_nl: true };
        assert!(matcher_match_nl.matches(""));
        assert!(matcher_match_nl.matches("x"));
        assert!(matcher_match_nl.matches("\n"));
        assert!(!matcher_match_nl.matches("xx"));
        assert!(!matcher_match_nl.matches("\n\n"));

        // Test case: do not match newline
        let matcher_no_match_nl = ZeroOrOneCharsMatcher { match_nl: false };
        assert!(matcher_no_match_nl.matches(""));
        assert!(matcher_no_match_nl.matches("x"));
        assert!(!matcher_no_match_nl.matches("\n"));
        assert!(!matcher_no_match_nl.matches("xx"));
        assert!(!matcher_no_match_nl.matches("\n\n"));

        // Test case: Unicode
        let emoji1 = "😀"; // 1 rune
        let emoji2 = "❤️"; // 2 runes
        assert_eq!(emoji1.chars().count(), 1);
        assert_eq!(emoji2.chars().count(), 2);

        let matcher_unicode = ZeroOrOneCharsMatcher { match_nl: true };
        assert!(matcher_unicode.matches(emoji1));
        assert!(!matcher_unicode.matches(emoji2));
        assert!(!matcher_unicode.matches(&format!("{}{}", emoji1, emoji1)));
        assert!(!matcher_unicode.matches(&format!("x{}", emoji1)));
        assert!(!matcher_unicode.matches(&format!("{}{}", emoji1, "x")));
        assert!(!matcher_unicode.matches(&format!("{}{}", emoji1, emoji2)));

        // Test case: invalid Unicode
        let re = Regex::new(r"^.?$").unwrap();
        let matcher_invalid_unicode = ZeroOrOneCharsMatcher { match_nl: true };

        let require_matches = |s: &str, expected: bool| {
            assert_eq!(matcher_invalid_unicode.matches(s), expected, "String: {}", s);
            assert_eq!(re.is_match(s), matcher_invalid_unicode.matches(s), "String: {}", s);
        };

        require_matches("\u{FF}", true);
        let value = "x\u{FF}";
        require_matches(value, false);
        require_matches("x\u{FF}x", false);
        require_matches("\u{FF}\u{FE}", false);
    }

    #[test]
    fn test_repetition_matcher_min_zero_empty_string() {
        let matcher = RepetitionMatcher::new("sub".to_string(), 0, Some(3));
        assert!(matcher.matches(""));
    }

    #[test]
    fn test_repetition_exact_match() {
        let matcher = RepetitionMatcher::new("abc".to_string(), 1, None);
        assert!(matcher.matches("abc"));
    }

    #[test]
    fn test_repetition_min_zero() {
        let matcher = RepetitionMatcher::new("abc".to_string(), 0, None);
        assert!(matcher.matches(""));
        assert!(matcher.matches("abc"));
        assert!(matcher.matches("abcabc"));
    }

    #[test]
    fn test_repetition_zero_to_n() {
        let matcher = RepetitionMatcher::new("abc".to_string(), 0, Some(3));
        assert!(matcher.matches(""));
        assert!(matcher.matches("abc"));
        assert!(matcher.matches("abcabc"));
        assert!(matcher.matches("abcabcabc"));

        assert_eq!(false, matcher.matches("abcabcabcabc"));
    }

}