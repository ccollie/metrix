use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::common::join_vector;
use crate::parser::{escape_ident, is_empty_regex, quote, ParseError, ParseResult};
use ahash::AHashMap;
use metricsql_common::regex_util::FastRegexMatcher;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

pub const NAME_LABEL: &str = "__name__";
pub type LabelName = String;

pub type LabelValue = String;

// NOTE: https://github.com/rust-lang/regex/issues/668

#[derive(
    Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Copy, Hash, Serialize, Deserialize,
)]
pub enum LabelFilterOp {
    #[default]
    Equal,
    NotEqual,
    RegexEqual,
    RegexNotEqual,
}

impl LabelFilterOp {
    pub fn is_negative(&self) -> bool {
        matches!(self, LabelFilterOp::NotEqual | LabelFilterOp::RegexNotEqual)
    }

    pub fn is_regex(&self) -> bool {
        matches!(
            self,
            LabelFilterOp::RegexEqual | LabelFilterOp::RegexNotEqual
        )
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            LabelFilterOp::Equal => "=",
            LabelFilterOp::NotEqual => "!=",
            LabelFilterOp::RegexEqual => "=~",
            LabelFilterOp::RegexNotEqual => "!~",
        }
    }
}

impl TryFrom<&str> for LabelFilterOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        match op {
            "=" => Ok(LabelFilterOp::Equal),
            "!=" => Ok(LabelFilterOp::NotEqual),
            "=~" => Ok(LabelFilterOp::RegexEqual),
            "!~" => Ok(LabelFilterOp::RegexNotEqual),
            _ => Err(ParseError::General(format!(
                "Unexpected match op literal: {op}"
            ))),
        }
    }
}

impl fmt::Display for LabelFilterOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// LabelFilter represents MetricsQL label filter like `foo="bar"`.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct LabelFilter {
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub op: LabelFilterOp,

    /// label contains label name for the filter.
    pub label: String,

    /// value contains unquoted value for the filter.
    pub value: String,

    #[serde(skip)]
    re: Option<Box<FastRegexMatcher>> // boxed to reduce struct size
}

impl LabelFilter {
    pub fn new<N, V>(match_op: LabelFilterOp, label: N, value: V) -> Result<Self, ParseError>
    where
        N: Into<LabelName>,
        V: Into<LabelValue>,
    {
        let label = label.into();
        let value = value.into();

        let re: Option<Box<FastRegexMatcher>> = match match_op {
            LabelFilterOp::RegexEqual | LabelFilterOp::RegexNotEqual => {
                let fre = FastRegexMatcher::new(&value)
                    .map_err(|_e| ParseError::InvalidRegex(value.clone()))?;
                Some(Box::new(fre))
            }
            _ => None
        };

        Ok(Self {
            label,
            op: match_op,
            value,
            re,
        })
    }

    pub fn equal<S: Into<String>>(key: S, value: S) -> Self {
        LabelFilter {
            op: LabelFilterOp::Equal,
            label: key.into(),
            value: value.into(),
            re: None,
        }
    }

    pub fn not_equal<S: Into<String>>(key: S, value: S) -> Self {
        LabelFilter {
            op: LabelFilterOp::NotEqual,
            label: key.into(),
            value: value.into(),
            re: None,
        }
    }

    pub fn regex_equal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter, ParseError> {
        LabelFilter::new(LabelFilterOp::RegexEqual, key, value)
    }

    pub fn regex_notequal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter, ParseError> {
        LabelFilter::new(LabelFilterOp::RegexNotEqual, key, value)
    }

    /// is_regexp represents whether the filter is regexp, i.e. `=~` or `!~`.
    pub fn is_regexp(&self) -> bool {
        self.op.is_regex()
    }

    /// is_negative represents whether the filter is negative, i.e. '!=' or '!~'.
    pub fn is_negative(&self) -> bool {
        self.op.is_negative()
    }

    pub fn is_metric_name_filter(&self) -> bool {
        self.label == NAME_LABEL && self.op == LabelFilterOp::Equal
    }

    pub fn is_name_label(&self) -> bool {
        self.label == NAME_LABEL && self.op == LabelFilterOp::Equal
    }

    /// Vector selectors must either specify a name or at least one label
    /// matcher that does not match the empty string.
    ///
    /// The following expression is illegal:
    /// {job=~".*"} # Bad!
    pub fn is_empty_matcher(&self) -> bool {
        use LabelFilterOp::*;
        // if we're matching against __name__, a negative comparison against the empty
        // string is valid
        let is_name_label = self.label == NAME_LABEL;
        match self.op {
            Equal => self.value.is_empty(),
            NotEqual => !self.value.is_empty() && !is_name_label,
            RegexEqual => is_empty_regex(&self.value),
            RegexNotEqual => is_empty_regex(&self.value) && !is_name_label,
        }
    }

    pub fn is_match(&self, str: &str) -> bool {
        match self.op {
            LabelFilterOp::Equal => self.value.eq(str),
            LabelFilterOp::NotEqual => self.value.ne(str),
            LabelFilterOp::RegexEqual => {
                // slight optimization for frequent case
                if str.is_empty() {
                    return is_empty_regex(&self.value);
                }
                if let Some(re) = &self.re {
                    re.matches(str)
                } else {
                    unreachable!("regex_equal without compiled regex");
                }
            }
            LabelFilterOp::RegexNotEqual => {
                if let Some(re) = &self.re {
                    !re.matches(str)
                } else {
                    unreachable!("regex_not_equal without compiled regex");
                }
            }
        }
    }

    pub fn inverse(&self) -> ParseResult<Self> {
        use LabelFilterOp::*;
        let op = match self.op {
            Equal => NotEqual,
            NotEqual => Equal,
            RegexEqual => RegexNotEqual,
            RegexNotEqual => RegexEqual,
        };
        Self::new(op, &self.label, &self.value)
    }


    pub fn as_string(&self) -> String {
        format!(
            "{}{}{}",
            escape_ident(&self.label),
            self.op,
            quote(&self.value)
        )
    }

    pub fn name(&self) -> String {
        if self.label == NAME_LABEL {
            return self.value.to_string();
        }
        self.label.clone()
    }

    pub fn is_optimized(&self) -> bool {
        if let Some(re) = &self.re {
            re.is_optimized()
        } else { false }
    }

    /// `prefix()` returns the required prefix of the value to match, if possible.
    /// It will be empty if it's an equality matcher or if the prefix can't be determined.
    pub fn prefix(&self) -> &str {
        if let Some(re) = &self.re {
            re.prefix.as_str()
        } else {
            EMPTY_STRING
        }
    }

    /// set_matches returns a set of equality matchers for the current regex matchers if possible.
    /// For examples the regexp `a(b|f)` will returns "ab" and "af".
    /// Returns nil if we can't replace the regexp by only equality matchers.
    pub fn set_matches(&self) -> Vec<String> {
        if let Some(matcher) = &self.re {
            return matcher.set_matches();
        }
        vec![]
    }
}

const EMPTY_STRING: &str = "";

impl PartialEq<LabelFilter> for LabelFilter {
    fn eq(&self, other: &Self) -> bool {
        self.op.eq(&other.op) && self.label.eq(&other.label) && self.value.eq(&other.value)
    }
}

impl PartialOrd for LabelFilter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for LabelFilter {}

impl Ord for LabelFilter {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut cmp = self.label.cmp(&other.label);
        if cmp == Ordering::Equal {
            cmp = self.value.cmp(&other.value);
            if cmp == Ordering::Equal {
                cmp = self.op.cmp(&other.op);
            }
        }
        cmp
    }
}

impl fmt::Display for LabelFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}{}{}",
            escape_ident(&self.label),
            self.op,
            quote(&self.value)
        )?;
        Ok(())
    }
}

impl Hash for LabelFilter {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.op.hash(state);
        self.label.hash(state);
        self.value.hash(state);
    }
}

pub type Matcher = LabelFilter;

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Matchers {
    pub matchers: Vec<LabelFilter>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "<[_]>::is_empty"))]
    pub or_matchers: Vec<Vec<LabelFilter>>,
}

impl Matchers {
    pub fn new(filters: Vec<LabelFilter>) -> Self {
        Matchers {
            matchers: filters,
            or_matchers: vec![],
        }
    }

    pub fn empty() -> Self {
        Self {
            matchers: vec![],
            or_matchers: vec![],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.matchers.is_empty() && self.or_matchers.is_empty()
    }

    pub fn with_or_matchers(or_matchers: Vec<Vec<LabelFilter>>) -> Self {
        Matchers {
            matchers: vec![],
            or_matchers,
        }
    }

    pub fn append(mut self, matcher: LabelFilter) -> Self {
        // Check the latest or_matcher group. If it is not empty,
        // we need to add the current matcher to this group.
        if let Some(last_or_matcher) = self.or_matchers.last_mut() {
            last_or_matcher.push(matcher);
        } else {
            self.matchers.push(matcher);
        }
        self
    }

    pub fn append_or(mut self, matcher: LabelFilter) -> Self {
        if !self.matchers.is_empty() {
            // Be careful not to move ownership here, because it
            // will be used by the subsequent append method.
            let last_matchers = std::mem::take(&mut self.matchers);
            self.or_matchers.push(last_matchers);
        }
        let new_or_matchers = vec![matcher];
        self.or_matchers.push(new_or_matchers);
        self
    }

    pub fn merge(mut self, other: Matchers) -> Self {
        if !other.or_matchers.is_empty() {
            if !self.matchers.is_empty() {
                self.or_matchers.push(std::mem::take(&mut self.matchers));
            }
            self.or_matchers.extend(other.or_matchers);
        } else {
            self.matchers.extend(other.matchers);
        }
        self
    }

    /// Vector selectors must either specify a name or at least one label
    /// matcher that does not match the empty string.
    ///
    /// The following expression is illegal:
    /// {job=~".*"} -- Bad!
    pub fn is_empty_matchers(&self) -> bool {
        (self.matchers.is_empty() && self.or_matchers.is_empty())
            || self
                .matchers
                .iter()
                .chain(self.or_matchers.iter().flatten())
                .all(|m| m.is_match(""))
    }

    /// find the matcher's value whose name equals the specified name. This function
    /// is designed to prepare error message of invalid promql expression.
    #[allow(dead_code)]
    pub(crate) fn find_matcher_value(&self, name: &str) -> Option<&String> {
        self.matchers
            .iter()
            .chain(self.or_matchers.iter().flatten())
            .find(|m| m.label.eq(name))
            .map(|m| &m.value)
    }

    /// find matchers whose name equals the specified name
    pub fn find_matchers(&self, name: &str) -> Vec<&LabelFilter> {
        self.matchers
            .iter()
            .chain(self.or_matchers.iter().flatten())
            .filter(|m| m.label.eq(name))
            .collect()
    }

    pub fn sort_filters(&mut self) {
        if !self.matchers.is_empty() {
            self.matchers.sort();
        }
        if !self.or_matchers.is_empty() {
            for filter_list in self.or_matchers.iter_mut() {
                filter_list.sort();
            }
        }
    }

    pub fn is_only_metric_name(&self) -> bool {
        if !self.matchers.is_empty() {
            return self.matchers.len() == 1 && self.matchers[0].is_metric_name_filter();
        }
        if !self.or_matchers.is_empty() {
            if self.metric_name().is_none() {
                return false;
            }
            return self.or_matchers.iter().all(|lfs| lfs.len() <= 1);
        }
        true
    }

    pub fn metric_name(&self) -> Option<&str> {
        if !self.matchers.is_empty() {
            let found = self
                .matchers
                .iter()
                .find(|m| m.is_metric_name_filter())
                .map(|m| m.value.as_str());

            // todo: make sure only 1 is specified
            return found;
        }
        if !self.or_matchers.is_empty() {
            let lfs = self.or_matchers.first().unwrap();
            if lfs.is_empty() {
                return None;
            }
            let head = &lfs[0];
            if !head.is_metric_name_filter() {
                return None;
            }
            let metric_name = head.value.as_str();
            for or_matchers in &self.or_matchers[1..] {
                if or_matchers.is_empty() {
                    return None;
                }
                let first = &or_matchers[0];
                if !first.is_metric_name_filter() || first.value.as_str() != metric_name {
                    return None;
                }
            }
            return Some(metric_name);
        }
        None
    }

    pub fn dedup(&mut self) {
        if !self.matchers.is_empty() {
            remove_duplicate_label_filters(&mut self.matchers);
        }
        if !self.or_matchers.is_empty() {
            for or_matchers in &mut self.or_matchers {
                remove_duplicate_label_filters(or_matchers);
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vec<LabelFilter>> {
        OrIter::new(&self.matchers, &self.or_matchers)
    }

    pub fn filter_iter(&self) -> impl Iterator<Item = &LabelFilter> {
        self.matchers
            .iter()
            .chain(self.or_matchers.iter().flatten())
    }
}

fn hash_filters(hasher: &mut impl Hasher, filters: &[LabelFilter]) {
    for filter in filters {
        filter.hash(hasher);
    }
}

impl Hash for Matchers {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if !self.matchers.is_empty() {
            hash_filters(state, &self.matchers);
        }
        for filters in &self.or_matchers {
            hash_filters(state, filters);
        }
    }
}

impl fmt::Display for Matchers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let simple_matchers = &self.matchers;
        let or_matchers = &self.or_matchers;
        if or_matchers.is_empty() {
            write!(f, "{}", join_vector(simple_matchers, ",", true))
        } else {
            let or_matchers_string =
                self.or_matchers
                    .iter()
                    .fold(String::new(), |or_matchers_str, pair| {
                        format!("{} or {}", or_matchers_str, join_vector(pair, ", ", false))
                    });
            let or_matchers_string = or_matchers_string.trim_start_matches(" or").trim();
            write!(f, "{}", or_matchers_string)
        }
    }
}

struct OrIter<'a> {
    matchers: &'a Vec<LabelFilter>,
    or_matchers: &'a Vec<Vec<LabelFilter>>,
    index: usize,
    first: bool,
}

impl<'a> OrIter<'a> {
    fn new(matchers: &'a Vec<LabelFilter>, or_matchers: &'a Vec<Vec<LabelFilter>>) -> Self {
        Self {
            matchers,
            or_matchers,
            index: 0,
            first: true,
        }
    }
}
impl<'a> Iterator for OrIter<'a> {
    type Item = &'a Vec<LabelFilter>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.first {
            self.first = false;
            if !self.matchers.is_empty() {
                return Some(self.matchers);
            }
        }
        if self.index < self.or_matchers.len() {
            let index = self.index;
            self.index += 1;
            return Some(&self.or_matchers[index]);
        }
        None
    }
}

pub(crate) fn remove_duplicate_label_filters(filters: &mut Vec<LabelFilter>) {
    fn get_hash(hasher: &mut Xxh3, filter: &LabelFilter) -> u64 {
        hasher.reset();
        hasher.write(filter.label.as_bytes());
        hasher.write(filter.op.as_str().as_bytes());
        hasher.write(filter.value.as_bytes());
        hasher.finish()
    }

    let mut hasher = Xxh3::new();
    let mut hash_map: AHashMap<u64, bool> = AHashMap::with_capacity(filters.len());

    for i in (0..filters.len()).rev() {
        let hash = get_hash(&mut hasher, &filters[i]);
        if let std::collections::hash_map::Entry::Vacant(e) = hash_map.entry(hash) {
            e.insert(true);
        } else {
            filters.remove(i);
        }
    }
}

/// Go and Rust handle the repeat pattern differently
/// in Go the following is valid: `aaa{bbb}ccc`
/// in Rust {bbb} is seen as an invalid repeat and must be escaped \{bbb}
/// This escapes the opening "{" if it's not followed by valid repeat pattern (e.g. 4,6).
pub fn try_escape_for_repeat_re(re: &str) -> String {
    fn is_repeat(chars: &mut std::str::Chars<'_>) -> (bool, String) {
        let mut buf = String::new();
        let mut comma_seen = false;
        for c in chars.by_ref() {
            buf.push(c);
            match c {
                ',' if comma_seen => {
                    return (false, buf); // ,, is invalid
                }
                ',' if buf == "," => {
                    return (false, buf); // {, is invalid
                }
                ',' if !comma_seen => comma_seen = true,
                '}' if buf == "}" => {
                    return (false, buf); // {} is invalid
                }
                '}' => {
                    return (true, buf);
                }
                _ if c.is_ascii_digit() => continue,
                _ => {
                    return (false, buf); // false if visit non-digit char
                }
            }
        }
        (false, buf) // not ended with "}"
    }

    let mut result = String::with_capacity(re.len() + 1);
    let mut chars = re.chars();

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(cc) = chars.next() {
                    result.push(c);
                    result.push(cc);
                }
            }
            '{' => {
                let (is, s) = is_repeat(&mut chars);
                if !is {
                    result.push('\\');
                }
                result.push(c);
                result.push_str(&s);
            }
            _ => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::label::{LabelFilter, LabelFilterOp, Matchers};

    use super::try_escape_for_repeat_re;

    #[test]
    fn test_matcher_eq_ne() {
        let op = LabelFilterOp::Equal;
        let matcher = LabelFilter::new(op, "name", "up").unwrap();
        assert!(matcher.is_match("up"));
        assert!(!matcher.is_match("down"));

        let op = LabelFilterOp::NotEqual;
        let matcher = LabelFilter::new(op, "name", "up").unwrap();
        assert!(matcher.is_match("foo"));
        assert!(matcher.is_match("bar"));
        assert!(!matcher.is_match("up"));
    }

    #[test]
    fn test_matcher_re() {
        let value = "api/v1/.*";
        let matcher = LabelFilter::new(LabelFilterOp::RegexEqual, "name", value).unwrap();
        assert!(matcher.is_match("api/v1/query"));
        assert!(matcher.is_match("api/v1/range_query"));
        assert!(!matcher.is_match("api/v2"));
    }

    #[test]
    fn test_eq_matcher_equality() {
        assert_eq!(
            LabelFilter::equal("code", "200"),
            LabelFilter::equal("code", "200")
        );

        assert_ne!(
            LabelFilter::equal("code", "200"),
            LabelFilter::equal("code", "201")
        );

        assert_ne!(
            LabelFilter::equal("code", "200"),
            LabelFilter::not_equal("code", "200")
        );
    }

    #[test]
    fn test_ne_matcher_equality() {
        assert_eq!(
            LabelFilter::not_equal("code", "200"),
            LabelFilter::not_equal("code", "200")
        );

        assert_ne!(
            LabelFilter::not_equal("code", "200"),
            LabelFilter::not_equal("code", "201")
        );

        assert_ne!(
            LabelFilter::not_equal("code", "200"),
            LabelFilter::equal("code", "200")
        );
    }

    #[test]
    fn test_re_matcher_equality() {
        assert_eq!(
            LabelFilter::regex_equal("code", "2??"),
            LabelFilter::regex_equal("code", "2??")
        );

        assert_ne!(
            LabelFilter::regex_equal("code", "2??",),
            LabelFilter::regex_equal("code", "2*?",)
        );

        assert_ne!(
            LabelFilter::new(LabelFilterOp::RegexEqual, "code", "2??",),
            LabelFilter::new(LabelFilterOp::Equal, "code", "2??")
        );
    }

    #[test]
    fn test_not_re_matcher_equality() {
        assert_eq!(
            LabelFilter::regex_notequal("code", "2??",),
            LabelFilter::regex_notequal("code", "2??",)
        );

        assert_ne!(
            LabelFilter::regex_notequal("code", "2??"),
            LabelFilter::regex_notequal("code", "2*?",)
        );

        assert_ne!(
            LabelFilter::regex_equal("code", "2??").unwrap(),
            LabelFilter::equal("code", "2??")
        );
    }

    #[test]
    fn test_matchers_equality() {
        assert_eq!(
            Matchers::empty()
                .append(LabelFilter::equal("name1", "val1"))
                .append(LabelFilter::equal("name2", "val2")),
            Matchers::empty()
                .append(LabelFilter::equal("name1", "val1"))
                .append(LabelFilter::equal("name2", "val2"))
        );

        assert_ne!(
            Matchers::empty().append(LabelFilter::equal("name1", "val1")),
            Matchers::empty().append(LabelFilter::equal("name2", "val2"))
        );

        assert_ne!(
            Matchers::empty().append(LabelFilter::equal("name1", "val1")),
            Matchers::empty().append(LabelFilter::not_equal("name1", "val1"))
        );

        assert_eq!(
            Matchers::empty()
                .append(LabelFilter::equal("name1", "val1"))
                .append(LabelFilter::not_equal("name2", "val2"))
                .append(LabelFilter::regex_equal("name2", "\\d+").unwrap())
                .append(LabelFilter::regex_notequal("name2", "\\d+").unwrap()),
            Matchers::empty()
                .append(LabelFilter::equal("name1", "val1"))
                .append(LabelFilter::not_equal("name2", "val2"))
                .append(LabelFilter::regex_equal("name2", "\\d+").unwrap())
                .append(LabelFilter::regex_notequal("name2", "\\d+").unwrap())
        );
    }

    #[test]
    fn test_find_matchers() {
        let matchers = Matchers::empty()
            .append(LabelFilter::equal("foo", "bar"))
            .append(LabelFilter::not_equal("foo", "bar"))
            .append(LabelFilter::equal("FOO", "bar"))
            .append(LabelFilter::not_equal("bar", "bar"));

        let ms = matchers.find_matchers("foo");
        assert_eq!(4, ms.len());
    }

    #[test]
    fn test_convert_re() {
        assert_eq!(try_escape_for_repeat_re("abc{}"), r"abc\{}");
        assert_eq!(try_escape_for_repeat_re("abc{def}"), r"abc\{def}");
        assert_eq!(try_escape_for_repeat_re("abc{def"), r"abc\{def");
        assert_eq!(try_escape_for_repeat_re("abc{1}"), "abc{1}");
        assert_eq!(try_escape_for_repeat_re("abc{1,}"), "abc{1,}");
        assert_eq!(try_escape_for_repeat_re("abc{1,2}"), "abc{1,2}");
        assert_eq!(try_escape_for_repeat_re("abc{,2}"), r"abc\{,2}");
        assert_eq!(try_escape_for_repeat_re("abc{{1,2}}"), r"abc\{{1,2}}");
        assert_eq!(try_escape_for_repeat_re(r"abc\{abc"), r"abc\{abc");
        assert_eq!(try_escape_for_repeat_re("abc{1a}"), r"abc\{1a}");
        assert_eq!(try_escape_for_repeat_re("abc{1,a}"), r"abc\{1,a}");
        assert_eq!(try_escape_for_repeat_re("abc{1,2a}"), r"abc\{1,2a}");
        assert_eq!(try_escape_for_repeat_re("abc{1,2,3}"), r"abc\{1,2,3}");
        assert_eq!(try_escape_for_repeat_re("abc{1,,2}"), r"abc\{1,,2}");
    }
}
