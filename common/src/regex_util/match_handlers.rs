use get_size::GetSize;
use regex::Regex;
use std::collections::{HashMap, HashSet};
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
pub struct LiteralPrefixMatcher {
    pub prefix: String,
    pub right: Option<Box<StringMatchHandler>>,
}

impl LiteralPrefixMatcher {
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

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct LiteralSuffixMatcher {
    pub left: Option<Box<StringMatchHandler>>,
    pub suffix: String,
}

impl LiteralSuffixMatcher {
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
pub struct EqualMultiStringMapMatcher {
    pub values: HashSet<String>,
    pub prefixes: HashMap<String, Vec<Box<StringMatchHandler>>>,
    pub min_prefix_len: usize,
}

impl EqualMultiStringMapMatcher {
    pub(crate) fn new(min_prefix_len: usize) -> Self {
        Self {
            values: Default::default(),
            prefixes: Default::default(),
            min_prefix_len,
        }
    }

    pub(crate) fn add(&mut self, s: String) {
        self.values.insert(s);
    }

    pub(crate) fn add_prefix(&mut self, prefix: String, matcher: Box<StringMatchHandler>) {
        if self.min_prefix_len == 0 {
            panic!("add_prefix called when no prefix length defined");
        }
        if prefix.len() < self.min_prefix_len {
            panic!("add_prefix called with a too short prefix");
        }

        let s = get_prefix(&prefix, self.min_prefix_len);

        self.prefixes
            .entry(s)
            .or_insert_with(Vec::new)
            .push(matcher);
    }

    pub fn set_matches(&self) -> Vec<String> {
        if self.values.len() >= MAX_SET_MATCHES || !self.prefixes.is_empty() {
            return Vec::new();
        }

        self.values.iter().cloned().collect::<Vec<String>>()
    }

    fn matches(&self, s: &str) -> bool {
        if self.values.contains(s) {
            return true;
        }

        if self.min_prefix_len > 0 && s.len() >= self.min_prefix_len {
            let prefix = &s[..self.min_prefix_len];
            if let Some(matchers) = self.prefixes.get(prefix) {
                if matchers.iter().any(|m| m.matches(s)) {
                    return true;
                }
            }
        }
        false
    }
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AlternatesMatcher {
    pub alts: Vec<String>,
    match_fn: MatchFn,
}

impl GetSize for AlternatesMatcher {
    fn get_size(&self) -> usize {
        self.alts.get_size() + size_of::<MatchFn>()
    }
}
impl AlternatesMatcher {
    pub fn new(alts: Vec<String>, match_fn: MatchFn) -> Self {
        Self {
            alts,
            match_fn,
        }
    }

    pub fn matches(&self, s: &str) -> bool {
        self.alts.iter().any(|alt| (self.match_fn)(alt, s))
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
            let mut cursor = &s[..];
            let mut i = 0;
            while cursor.len() >= s.len() {
                if !cursor.starts_with(&self.sub) {
                    break;
                }
                cursor = &cursor[self.sub.len()..];
                i += 1;

                if i > *max {
                    return false;
                }
            }
            return i >= self.min && i <= *max
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
    MatchAll,
    MatchNone,
    Empty,
    AnyWithoutNewline,
    NotEmpty(NonEmptyStringMatcher),
    EqualsMulti(Vec<String>),
    EqualMultiMap(EqualMultiStringMapMatcher),
    ContainsMulti(ContainsMultiStringMatcher),
    Literal(String),
    Contains(String),
    StartsWith(String),
    Prefix(LiteralPrefixMatcher),
    Suffix(LiteralSuffixMatcher),
    EndsWith(String),
    Regex(RegexMatcher),
    Repetition(RepetitionMatcher),
    OrderedAlternates(Vec<String>),
    MatchFn(MatchFnHandler),
    Alternates(AlternatesMatcher),
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
        Self::Regex(RegexMatcher::new(regex, String::new(), String::new()))
    }

    pub fn alternates(alts: Vec<String>, options: &StringMatchOptions) -> Self {
        let mut alts = alts;
        let is_default = options.is_default();
        if is_default {
            if alts.len() == 1 {
                return Self::Literal(alts.pop().unwrap());
            }
            return Self::Alternates(AlternatesMatcher { alts, match_fn: contains_fn });
        }
        let match_fn = get_literal_match_fn(options);
        if alts.len() == 1 {
            let pattern = alts.pop().unwrap();
            return Self::MatchFn(MatchFnHandler{
                pattern,
                match_fn
            });
        }
        Self::Alternates(AlternatesMatcher{ alts, match_fn })
    }

    pub fn literal_alternates(alts: Vec<String>) -> Self {
        if alts.len() == 1 {
            let mut alts = alts;
            Self::Literal(alts.pop().unwrap())
        } else {
            Self::Alternates(AlternatesMatcher { alts, match_fn: equals_fn })
        }
    }

    pub fn zero_or_one_chars(match_nl: bool) -> Self {
        StringMatchHandler::ZeroOrOneChars(ZeroOrOneCharsMatcher { match_nl })
    }

    pub fn not_empty(match_nl: bool) -> Self {
        StringMatchHandler::NotEmpty(NonEmptyStringMatcher::new(match_nl))
    }

    pub fn literal(value: String, options: &StringMatchOptions) -> Self {
        get_optimized_literal_matcher(value, options)
    }

    pub fn equals(value: String) -> Self {
        StringMatchHandler::Literal(value)
    }

    pub fn prefix(value: String, right: Option<StringMatchHandler>) -> Self {
        StringMatchHandler::Prefix(LiteralPrefixMatcher {
            prefix: value,
            right: right.map(Box::new),
        })
    }

    pub fn suffix(left: Option<StringMatchHandler>, value: String) -> Self {
        StringMatchHandler::Suffix(LiteralSuffixMatcher {
            left: left.map(Box::new),
            suffix: value,
        })
    }

    pub fn and(self, b: StringMatchHandler) -> Self {
        Self::And(Box::new(self), Box::new(b))
    }

    #[allow(dead_code)]
    pub fn matches(&self, s: &str) -> bool {
        match self {
            StringMatchHandler::MatchAll => true,
            StringMatchHandler::MatchNone => false,
            StringMatchHandler::Alternates(alts) => alts.matches(s),
            StringMatchHandler::MatchFn(m) => m.matches(s),
            StringMatchHandler::Regex(r) => r.matches(s),
            StringMatchHandler::OrderedAlternates(m) => match_ordered_alternates(m, s),
            StringMatchHandler::And(a, b) => a.matches(s) && b.matches(s),
            StringMatchHandler::Contains(value) => s.contains(value),
            StringMatchHandler::StartsWith(prefix) => s.starts_with(prefix),
            StringMatchHandler::EndsWith(suffix) => s.ends_with(suffix),
            StringMatchHandler::Literal(val) => s == val,
            StringMatchHandler::Empty => s.is_empty(),
            StringMatchHandler::NotEmpty(opts) => opts.matches(s),
            StringMatchHandler::Or(matchers) => {
                matchers.iter().any(|m| m.matches(s))
            }
            StringMatchHandler::ZeroOrOneChars(m) => m.matches(s),
            StringMatchHandler::EqualsMulti(values) => {
                values.iter().any(|v| s == *v)
            }
            StringMatchHandler::ContainsMulti(m) => m.matches(s),
            StringMatchHandler::Prefix(m) => m.matches(s),
            StringMatchHandler::Suffix(m) => m.matches(s),
            StringMatchHandler::AnyWithoutNewline => !s.contains('\n'),
            StringMatchHandler::EqualMultiMap(m) => m.matches(s),
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

fn get_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
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
    fn test_contains() {
        let handler = StringMatchHandler::Contains("a".to_string());
        assert!(handler.matches("a"));
        assert!(handler.matches("ba"));
        assert!(!handler.matches("b"));
    }

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

}