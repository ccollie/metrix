use std::fmt::Display;
use get_size::GetSize;
use serde_derive::{Deserialize, Serialize};

#[derive(Clone, GetSize, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StringPattern {
    CaseSensitive(CaseSensitivePattern),
    CaseInsensitive(CaseInsensitivePattern),
    AsciiCaseInsensitive(AsciiCaseInsensitivePattern),
}

impl StringPattern {
    pub fn new(pattern: String, case_sensitive: bool) -> Self {
        match (pattern.is_ascii(), case_sensitive) {
            (true, false) => Self::AsciiCaseInsensitive(AsciiCaseInsensitivePattern::new(pattern)),
            (true, true) => Self::CaseSensitive(CaseSensitivePattern::new(pattern)),
            (false, false) => Self::CaseInsensitive(CaseInsensitivePattern::new(pattern)),
            (false, true) => Self::CaseSensitive(CaseSensitivePattern::new(pattern)),
        }
    }

    pub fn case_sensitive(pattern: String) -> Self {
        Self::CaseSensitive(CaseSensitivePattern::new(pattern))
    }

    pub fn case_insensitive(pattern: String) -> Self {
        Self::CaseInsensitive(CaseInsensitivePattern::new(pattern))
    }

    pub fn matches(&self, s: &str) -> bool {
        match self {
            Self::CaseSensitive(p) => p.matches(s),
            Self::CaseInsensitive(p) => p.matches(s),
            Self::AsciiCaseInsensitive(p) => p.matches(s),
        }
    }

    pub fn starts_with(&self, s: &str) -> bool {
        match self {
            Self::CaseSensitive(p) => p.starts_with(s),
            Self::CaseInsensitive(p) => p.starts_with(s),
            Self::AsciiCaseInsensitive(p) => p.starts_with(s),
        }
    }

    pub fn ends_with(&self, s: &str) -> bool {
        match self {
            Self::CaseSensitive(p) => p.ends_with(s),
            Self::CaseInsensitive(p) => p.ends_with(s),
            Self::AsciiCaseInsensitive(p) => p.ends_with(s),
        }
    }

    pub fn is_case_sensitive(&self) -> bool {
        match self {
            Self::CaseSensitive(_) => true,
            Self::CaseInsensitive(_) => false,
            Self::AsciiCaseInsensitive(_) => false,
        }
    }

    pub fn is_ascii(&self) -> bool {
        match self {
            Self::CaseSensitive(_) => false,
            Self::CaseInsensitive(_) => false,
            Self::AsciiCaseInsensitive(_) => true,
        }
    }

    pub fn pattern(&self) -> &str {
        match self {
            Self::CaseSensitive(p) => &p.pattern,
            Self::CaseInsensitive(p) => &p.pattern,
            Self::AsciiCaseInsensitive(p) => &p.pattern,
        }
    }

    pub fn len(&self) -> usize {
        self.pattern().len()
    }
}

impl From<&str> for StringPattern {
    fn from(s: &str) -> Self {
        Self::new(s.to_string(), true)
    }
}

impl From<String> for StringPattern {
    fn from(s: String) -> Self {
        Self::new(s, true)
    }
}

impl From<&String> for StringPattern {
    fn from(s: &String) -> Self {
        Self::new(s.clone(), true)
    }
}

impl From<&StringPattern> for String {
    fn from(p: &StringPattern) -> Self {
        p.pattern().to_string()
    }
}

/// Take ownership of the pattern and return a String
impl Into<String> for StringPattern {
    fn into(self) -> String {
        match self {
            Self::CaseSensitive(p) => p.pattern,
            Self::CaseInsensitive(p) => p.pattern,
            Self::AsciiCaseInsensitive(p) => p.pattern,
        }
    }
}

impl Display for StringPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CaseSensitive(p) => write!(f, "CaseSensitive{}", p.pattern),
            Self::CaseInsensitive(p) => write!(f, "CaseInsensitive{}", p.pattern),
            Self::AsciiCaseInsensitive(p) => write!(f, "AsciiCaseInsensitive{}", p.pattern),
        }
    }
}

/// Specialized pattern for matching ASCII strings. This is a non-allocation optimization for the common case
/// where the pattern is ASCII and case-insensitive
#[derive(Clone, GetSize, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsciiCaseInsensitivePattern {
    pattern: String,
}

impl AsciiCaseInsensitivePattern {
    fn new(pattern: String) -> Self {
        Self {
            pattern,
        }
    }

    fn matches(&self, s: &str) -> bool {
        self.pattern.eq_ignore_ascii_case(s)
    }
    
    fn starts_with(&self, s: &str) -> bool {
        let pattern = self.pattern.as_str();
        let len = pattern.len();
        if len > s.len() {
            return false;
        }
        let prefix = &s[..len];
        prefix.eq_ignore_ascii_case(pattern)
    }
    
    fn ends_with(&self, s: &str) -> bool {
        if self.pattern.len() > s.len() {
            return false;
        }
        let this = &self.pattern[s.len() - self.pattern.len()..];
        this.eq_ignore_ascii_case(s)
    }
}

#[derive(Clone, GetSize, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseSensitivePattern {
    pattern: String,
}

impl CaseSensitivePattern {
    fn new(pattern: String) -> Self {
        Self {
            pattern,
        }
    }

    fn matches(&self, s: &str) -> bool {
        self.pattern == s
    }

    fn starts_with(&self, s: &str) -> bool {
        s.starts_with(&self.pattern)
    }

    fn ends_with(&self, s: &str) -> bool {
        s.ends_with(&self.pattern)
    }
}

impl From<&str> for CaseSensitivePattern {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl From<String> for CaseSensitivePattern {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

#[derive(Clone, GetSize, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseInsensitivePattern {
    pub(crate) pattern: String,
    lowercase_pattern: String,
}

impl CaseInsensitivePattern {
    fn new(pattern: String) -> Self {
        let lowercase_pattern = pattern.to_lowercase();
        Self {
            pattern,
            lowercase_pattern,
        }
    }
    
    fn matches(&self, s: &str) -> bool {
        if s.len() != self.pattern.len() {
            return false;
        }
        self.lowercase_pattern == s.to_lowercase()
    }
    
    fn starts_with(&self, s: &str) -> bool {
        if self.pattern.len() > s.len() {
            return false;
        }
        s.to_lowercase().starts_with(&self.lowercase_pattern)
    }
    
    fn ends_with(&self, s: &str) -> bool {
        if self.pattern.len() > s.len() {
            return false;
        }
        s.to_lowercase().ends_with(&self.lowercase_pattern)
    }
}