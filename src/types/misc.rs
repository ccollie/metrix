// (C) Copyright 2019-2020 Hewlett Packard Enterprise Development LP

use std::{cmp::Ordering, fmt};

/// A byte-index tuple representing a span of characters in a string
///
/// Note that spans refer to the position in the input string as read by the
/// parser rather than the output of an expression's `Display` impl.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct Span {
    pub start: usize,
    pub end: usize
}

impl From<(usize, usize)> for Span {
    fn from(tup: (usize, usize)) -> Span {
        Span::new(tup.0, tup.1)
    }
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }

    pub(crate) fn from_node(node: &crate::parser::Node) -> Self {
        let span = node.as_span();
        Span {
            start: span.start(),
            end: span.end()
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.start, self.end)
    }
}

/// A Prometheus duration
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PromDuration {
    Seconds(u64),
    Minutes(u64),
    Hours(u64),
    Days(u64),
    Weeks(u64),
    Years(u64)
}

type StringResult<T> = std::result::Result<T, String>;

impl PromDuration {
    pub fn from_pair(unit: &str, value: u64) -> StringResult<PromDuration> {
        Ok(match unit {
            "s" => PromDuration::Seconds(value),
            "m" => PromDuration::Minutes(value),
            "h" => PromDuration::Hours(value),
            "d" => PromDuration::Days(value),
            "w" => PromDuration::Weeks(value),
            "y" => PromDuration::Years(value),
            u => return Err(format!("invalid duration unit: {:?}", u))
        })
    }

    pub fn as_char(self) -> char {
        match self {
            PromDuration::Seconds(_) => 's',
            PromDuration::Minutes(_) => 'm',
            PromDuration::Hours(_) => 'h',
            PromDuration::Days(_) => 'd',
            PromDuration::Weeks(_) => 'w',
            PromDuration::Years(_) => 'y'
        }
    }
}

impl fmt::Display for PromDuration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let v = match self {
            PromDuration::Seconds(v) => v,
            PromDuration::Minutes(v) => v,
            PromDuration::Hours(v) => v,
            PromDuration::Days(v) => v,
            PromDuration::Weeks(v) => v,
            PromDuration::Years(v) => v
        };

        write!(f, "{}{}", v, self.as_char())
    }
}

impl PartialOrd for PromDuration {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PromDuration {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (PromDuration::Seconds(left), PromDuration::Seconds(right)) => left.cmp(right),
            (PromDuration::Seconds(_), PromDuration::Minutes(_)) => Ordering::Less,
            (PromDuration::Seconds(_), PromDuration::Hours(_)) => Ordering::Less,
            (PromDuration::Seconds(_), PromDuration::Days(_)) => Ordering::Less,
            (PromDuration::Seconds(_), PromDuration::Weeks(_)) => Ordering::Less,
            (PromDuration::Seconds(_), PromDuration::Years(_)) => Ordering::Less,
            (PromDuration::Minutes(_), PromDuration::Seconds(_)) => Ordering::Greater,
            (PromDuration::Minutes(left), PromDuration::Minutes(right)) => left.cmp(right),
            (PromDuration::Minutes(_), PromDuration::Hours(_)) => Ordering::Less,
            (PromDuration::Minutes(_), PromDuration::Days(_)) => Ordering::Less,
            (PromDuration::Minutes(_), PromDuration::Weeks(_)) => Ordering::Less,
            (PromDuration::Minutes(_), PromDuration::Years(_)) => Ordering::Less,
            (PromDuration::Hours(_), PromDuration::Seconds(_)) => Ordering::Greater,
            (PromDuration::Hours(_), PromDuration::Minutes(_)) => Ordering::Greater,
            (PromDuration::Hours(left), PromDuration::Hours(right)) => left.cmp(right),
            (PromDuration::Hours(_), PromDuration::Days(_)) => Ordering::Less,
            (PromDuration::Hours(_), PromDuration::Weeks(_)) => Ordering::Less,
            (PromDuration::Hours(_), PromDuration::Years(_)) => Ordering::Less,
            (PromDuration::Days(_), PromDuration::Seconds(_)) => Ordering::Greater,
            (PromDuration::Days(_), PromDuration::Minutes(_)) => Ordering::Greater,
            (PromDuration::Days(_), PromDuration::Hours(_)) => Ordering::Greater,
            (PromDuration::Days(left), PromDuration::Days(right)) => left.cmp(right),
            (PromDuration::Days(_), PromDuration::Weeks(_)) => Ordering::Less,
            (PromDuration::Days(_), PromDuration::Years(_)) => Ordering::Less,
            (PromDuration::Weeks(_), PromDuration::Seconds(_)) => Ordering::Greater,
            (PromDuration::Weeks(_), PromDuration::Minutes(_)) => Ordering::Greater,
            (PromDuration::Weeks(_), PromDuration::Hours(_)) => Ordering::Greater,
            (PromDuration::Weeks(_), PromDuration::Days(_)) => Ordering::Greater,
            (PromDuration::Weeks(left), PromDuration::Weeks(right)) => left.cmp(right),
            (PromDuration::Weeks(_), PromDuration::Years(_)) => Ordering::Less,
            (PromDuration::Years(_), PromDuration::Seconds(_)) => Ordering::Greater,
            (PromDuration::Years(_), PromDuration::Minutes(_)) => Ordering::Greater,
            (PromDuration::Years(_), PromDuration::Hours(_)) => Ordering::Greater,
            (PromDuration::Years(_), PromDuration::Days(_)) => Ordering::Greater,
            (PromDuration::Years(_), PromDuration::Weeks(_)) => Ordering::Greater,
            (PromDuration::Years(left), PromDuration::Years(right)) => left.cmp(right)
        }
    }
}

/// A Subquery which converts an instant vector to a range vector by repeatedly
/// evaluating it at set intervals into the relative past
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Subquery {
    /// Duration back in time to begin the subquery
    pub range: PromDuration,

    /// Optional step size. If unset, uses the global/query default at runtime.
    pub resolution: Option<PromDuration>,

    pub span: Option<Span>
}

impl Subquery {
    pub fn new(range: PromDuration) -> Self {
        Subquery {
            range,
            resolution: None,
            span: None
        }
    }

    pub fn resolution(mut self, res: PromDuration) -> Self {
        self.resolution = Some(res);
        self
    }

    pub fn clear_resolution(mut self) -> Self {
        self.resolution = None;
        self
    }

    pub fn span<S: Into<Span>>(mut self, span: S) -> Self {
        self.span = Some(span.into());
        self
    }
}

impl fmt::Display for Subquery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(res) = self.resolution {
            write!(f, "[{}:{}]", self.range, res)
        } else {
            write!(f, "[{}:]", self.range)
        }
    }
}