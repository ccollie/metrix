use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::{fmt, ops};

use serde::{Deserialize, Serialize};

use crate::ast::Prettier;
use crate::common::ValueType;
use crate::parser::ParseError;
use crate::prelude::ParseResult;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum StringSegment {
    Literal(String),
    Ident(String),
}

impl Default for StringSegment {
    fn default() -> Self {
        StringSegment::Literal(String::new())
    }
}

impl Display for StringSegment {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            StringSegment::Literal(lit) => write!(f, "{}", enquote::enquote('"', lit))?,
            StringSegment::Ident(ident) => write!(f, "{}", ident)?,
        }
        Ok(())
    }
}

impl PartialOrd for StringSegment {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (StringSegment::Literal(s), StringSegment::Literal(o)) => s.partial_cmp(o),
            (StringSegment::Ident(s), StringSegment::Ident(o)) => s.partial_cmp(o),
            (StringSegment::Literal(s), StringSegment::Ident(o)) => s.partial_cmp(o),
            (StringSegment::Ident(s), StringSegment::Literal(o)) => s.partial_cmp(o),
        }
    }
}

/// StringExpr represents a string expression which may be composed of multiple segments.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct StringExpr(Vec<StringSegment>, bool);

impl StringExpr {
    pub fn new<S: Into<String>>(s: S) -> Self {
        let segments = vec![StringSegment::Literal(s.into())];
        StringExpr(segments, true)
    }

    pub fn new_identifier<S: Into<String>>(ident: S) -> Self {
        let segments = vec![StringSegment::Ident(ident.into())];
        StringExpr(segments, false)
    }

    pub fn with_segment_capacity(initial_capacity: usize) -> Self {
        StringExpr(Vec::with_capacity(initial_capacity), true)
    }

    pub fn set_from_string<S: Into<String>>(&mut self, s: S) {
        if self.0.len() == 1 {
            if let Some(StringSegment::Literal(elem)) = self.0.first_mut() {
                elem.clear();
                elem.push_str(s.into().as_str())
            }
            return;
        }
        self.0.clear();
        self.0.push(StringSegment::Literal(s.into()));
        self.1 = true
    }

    pub fn push_str(&mut self, tok: &str) {
        if let Some(StringSegment::Literal(value)) = self.0.last_mut() {
            value.push_str(tok);
            self.1 = self.0.len() == 1;
        } else {
            self.0.push(StringSegment::Literal(tok.to_string()));
        }
    }

    pub fn push_ident(&mut self, tok: &str) {
        self.0.push(StringSegment::Ident(tok.to_string()));
        self.1 = false;
    }

    pub fn push(&mut self, segment: &StringSegment) {
        match segment {
            StringSegment::Literal(s) => self.push_str(s),
            StringSegment::Ident(ident) => self.push_ident(ident),
        }
    }

    pub fn clear(&mut self) {
        self.0.clear();
        self.1 = true;
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn segment_count(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn is_expanded(&self) -> bool {
        self.is_literal_only()
    }

    pub fn is_literal_only(&self) -> bool {
        self.1
    }

    pub fn is_identifier(&self) -> bool {
        if self.0.len() == 1 {
            let first = self.0.first().unwrap();
            return matches!(first, StringSegment::Ident(_));
        }
        false
    }

    pub(crate) fn estimate_result_capacity(&self) -> usize {
        self.0.iter().fold(0, |acc, s| {
            let res = match s {
                StringSegment::Literal(lit) => lit.len(),
                StringSegment::Ident(_) => 4, // todo: proper named constant
            };
            acc + res
        })
    }

    pub fn resolve<F>(&self, resolve_fn: F) -> ParseResult<String>
    where
        F: Fn(&str) -> ParseResult<Option<String>>,
    {
        if self.is_literal_only() {
            if let Some(first) = self.0.first() {
                match first {
                    StringSegment::Literal(lit) => return Ok(lit.clone()),
                    _ => panic!("BUG: string segment should be all literal"),
                }
            } else {
                return Ok("".to_string());
            }
        }
        let min_capacity = self.estimate_result_capacity();

        let mut res = String::with_capacity(min_capacity);
        for s in self.0.iter() {
            match s {
                StringSegment::Literal(lit) => res.push_str(lit),
                StringSegment::Ident(ident) => {
                    if let Some(ident_value) = resolve_fn(ident)? {
                        res.push_str(&ident_value);
                    } else {
                        let msg = format!(
                            "unknown identifier {:?} in string expression of {:?}",
                            ident, self
                        );
                        return Err(ParseError::General(msg));
                    }
                }
            };
        }

        Ok(res)
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::String
    }

    pub fn iter(&self) -> impl Iterator<Item = &StringSegment> + '_ {
        self.0.iter()
    }

    pub fn get_literal(&self) -> ParseResult<Option<&String>> {
        if self.is_literal_only() {
            if let Some(StringSegment::Literal(lit)) = self.0.first() {
                return Ok(Some(lit));
            }
            return Err(ParseError::General(
                "BUG: string segment should be all literal".to_string(),
            ));
        }
        Ok(None)
    }

    pub fn into_literal(self) -> ParseResult<String> {
        if self.is_literal_only() {
            if let Some(StringSegment::Literal(lit)) = self.0.into_iter().next() {
                return Ok(lit);
            }
            return Ok("".to_string())
        }
        let msg = "BUG: string expression is not a literal".to_string();
        Err(ParseError::General(msg))
    }

    pub fn as_identifier(&self) -> Option<&String> {
        if self.is_identifier() {
            if let StringSegment::Ident(ident) = self.0.first().unwrap() {
                return Some(ident);
            }
        }
        None
    }
}

impl PartialOrd for StringExpr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        for (left, right) in self.0.iter().zip(other.0.iter()) {
            if let Some(cmp) = left.partial_cmp(right) {
                if cmp != Ordering::Equal {
                    break;
                }
            }
        }
        Some(Ordering::Equal)
    }
}

impl Default for StringExpr {
    fn default() -> Self {
        StringExpr(vec![], true)
    }
}

impl From<String> for StringExpr {
    fn from(s: String) -> Self {
        let segment = StringSegment::Literal(s);
        StringExpr(vec![segment], true)
    }
}

impl From<&str> for StringExpr {
    fn from(s: &str) -> Self {
        Self::from(s.to_string())
    }
}

impl Display for StringExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for (i, segment) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}", segment)?;
        }
        Ok(())
    }
}

impl Prettier for StringExpr {
    fn format(&self, level: usize, _max: usize) -> String {
        let mut s = String::with_capacity(32);
        let spaces = " ".repeat(level * 2);
        for (i, segment) in self.0.iter().enumerate() {
            if i > 0 {
                s.push_str(&spaces);
                s.push_str(" + \n");
            }
            s.push_str(&format!("{spaces}{segment}\n"));
        }
        s
    }

    fn needs_split(&self, max: usize) -> bool {
        if self.is_literal_only() {
            return false;
        }
        let mut len = 0;
        for s in self.iter() {
            let segment_len = match s {
                StringSegment::Literal(lit) => lit.len(),
                StringSegment::Ident(id) => id.len(),
            };
            len += segment_len;
        }
        len += self.segment_count() * 3;
        len > max
    }
}

impl ops::Add for StringExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut result = self.clone();
        for item in rhs.0.iter() {
            result.0.push(item.clone())
        }
        result
    }
}
