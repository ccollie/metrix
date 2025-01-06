use std::fmt;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::ast::{Prettier, StringExpr};
use crate::common::{Value, ValueType};
use crate::label::{LabelFilterExpr, Matchers, NAME_LABEL};
use crate::parser::ParseResult;

/// InterpolatedSelector represents a Vector Selector in the context of a `WITH` expression.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpolatedSelector {
    /// a list of label filter expressions from WITH clause.
    /// This is transformed into label_filters during compilation.
    pub(crate) matchers: Vec<Vec<LabelFilterExpr>>,
}

impl InterpolatedSelector {
    pub fn new<S: Into<String>>(name: S) -> InterpolatedSelector {
        let name_filter = LabelFilterExpr::equal(NAME_LABEL, StringExpr::new(name)).unwrap();
        InterpolatedSelector {
            matchers: vec![vec![name_filter]],
        }
    }

    pub fn with_filters(filters: Vec<LabelFilterExpr>) -> Self {
        InterpolatedSelector {
            matchers: vec![filters],
        }
    }

    pub fn with_or_filters(filters: Vec<Vec<LabelFilterExpr>>) -> Self {
        InterpolatedSelector { matchers: filters }
    }

    pub fn is_empty(&self) -> bool {
        self.matchers.is_empty()
    }

    pub fn is_resolved(&self) -> bool {
        self.matchers.is_empty()
            || self
                .matchers
                .iter()
                .all(|x| x.iter().all(|label| label.is_resolved()))
    }

    pub fn metric_name(&self) -> Option<&str> {
        let lfss = &self.matchers;
        if lfss.is_empty() {
            return None;
        }

        fn get_name(lf: &LabelFilterExpr) -> Option<&str> {
            if lf.is_metric_name_filter() {
                if let Some(literal) = lf.value.get_literal().unwrap_or(None) {
                    return Some(literal.as_str());
                }
            }
            None
        }

        if let Some((first, rest)) = lfss.split_first() {
            if let Some(lf) = first.first() {
                if let Some(literal) = get_name(lf) {
                    let metric_name = literal;
                    for lf in rest {
                        if let Some(head) = lf.first() {
                            if let Some(literal) = get_name(head) {
                                if literal != metric_name {
                                    return None;
                                }
                            }
                        } else {
                            return None;
                        }
                    }
                    return Some(metric_name);
                }
            }
        }
        None
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }

    pub fn to_matchers(&self) -> ParseResult<Matchers> {
        if !self.is_resolved() {
            // todo: err
        }

        let mut or_matchers = vec![];
        for m in &self.matchers {
            let mut and_matchers = vec![];
            for l in m {
                and_matchers.push(l.to_matcher()?);
            }
            or_matchers.push(and_matchers);
        }

        Ok(Matchers::with_or_matchers(or_matchers))
    }

    pub fn is_empty_matchers(&self) -> bool {
        self.matchers.is_empty()
            || self
                .matchers
                .iter()
                .all(|x| x.iter().all(|y| y.is_empty_matcher()))
    }

    /// find all the matchers whose name equals the specified name.
    pub fn find_matchers(&self, name: &str) -> Vec<&LabelFilterExpr> {
        self.matchers
            .iter()
            .flatten()
            .filter(|m| m.label.eq_ignore_ascii_case(name))
            .collect()
    }
}

impl Value for InterpolatedSelector {
    fn value_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

impl Display for InterpolatedSelector {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{{")?;

        let exprs = &self.matchers[0..];

        for (i, filter_list) in exprs.iter().enumerate() {
            for (j, filter) in filter_list.iter().enumerate() {
                write!(f, "{}", filter)?;
                if j + 1 < filter_list.len() {
                    write!(f, ", ")?;
                }
            }
            if i + 1 < exprs.len() {
                write!(f, " or ")?;
            }
        }

        write!(f, "}}")?;

        Ok(())
    }
}

impl Prettier for InterpolatedSelector {
    fn needs_split(&self, _max: usize) -> bool {
        false
    }
}
