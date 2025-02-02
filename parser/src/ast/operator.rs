use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

use crate::parser::{ParseError, Token};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, EnumIter, Serialize, Deserialize)]
pub enum Operator {
    Add,
    And,
    Atan2,
    Default,
    Div,
    #[default]
    Eql,
    Mod,
    Mul,
    Pow,
    Sub,
    Gt,
    Gte,
    If,
    IfNot,
    Lt,
    Lte,
    NotEq,
    Or,
    Unless,
}

fn lookup_operator_internal(key: &[u8]) -> Option<Operator> {
    hashify::tiny_map! { key,
        "+" => Operator::Add,
        "-" => Operator::Sub,
        "*" => Operator::Mul,
        "/" => Operator::Div,
        "%" => Operator::Mod,
        "^" => Operator::Pow,

        // See https://github.com/prometheus/prometheus/pull/9248
        "atan2" => Operator::Atan2,

        // cmp ops
        "==" => Operator::Eql,
        "!=" => Operator::NotEq,
        "<" => Operator::Lt,
        ">" => Operator::Gt,
        "<=" => Operator::Lte,
        ">=" => Operator::Gte,

        // logic set ops
        "and" => Operator::And,
        "or" => Operator::Or,
        "unless" => Operator::Unless,

        // New ops for MetricsQL
        "if" => Operator::If,
        "ifnot" => Operator::IfNot,
        "default" => Operator::Default,
    }
}

pub fn lookup_operator(op: &str) -> Option<Operator> {
    if let Some(ch) = op.chars().next() {
        if !ch.is_alphabetic() {
            lookup_operator_internal(op.as_bytes())
        } else {
            // slight optimization - don't lowercase if not needed (save allocation)
            lookup_operator_internal(op.as_bytes()).or_else(|| {
                let lower = op.to_ascii_lowercase();
                lookup_operator_internal(lower.as_bytes())
            })
        }
    } else {
        None
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum BinaryOpKind {
    Arithmetic,
    Comparison,
    Logical,
}

pub type Precedence = usize;

impl Operator {
    #[inline]
    pub const fn precedence(&self) -> Precedence {
        use Operator::*;

        match self {
            Default => 0,
            If | IfNot => 1,
            // See https://prometheus.io/docs/prometheus/latest/querying/operators/#binary-operator-precedence
            Or => 10,
            And | Unless => 20,
            Eql | Gte | Gt | Lt | Lte | NotEq => 30,
            Add | Sub => 40,
            Mul | Div | Mod | Atan2 => 50,
            Pow => 60,
        }
    }

    #[inline]
    pub const fn kind(&self) -> BinaryOpKind {
        use BinaryOpKind::*;
        use Operator::*;

        match self {
            Add | Sub | Mul | Div | Mod | Pow | Atan2 => Arithmetic,
            Eql | Gte | Gt | Lt | Lte | NotEq => Comparison,
            And | Unless | Or | If | IfNot | Default => Logical,
        }
    }

    // See https://prometheus.io/docs/prometheus/latest/querying/operators/#binary-operator-precedence
    pub const fn is_right_associative(self) -> bool {
        matches!(self, Operator::Pow)
    }

    pub const fn is_arithmetic_op(&self) -> bool {
        use Operator::*;
        matches!(self, Add | Sub | Mul | Div | Mod | Pow | Atan2)
    }

    pub const fn is_logical_op(&self) -> bool {
        use Operator::*;
        matches!(self, And | Or | Unless | If | IfNot | Default)
    }

    pub const fn is_comparison(&self) -> bool {
        use Operator::*;
        matches!(self, Eql | Gte | Gt | Lt | Lte | NotEq)
    }

    pub const fn is_valid_string_op(&self) -> bool {
        use Operator::*;
        matches!(self, Add | Eql | Gte | Gt | Lt | Lte | NotEq)
    }

    pub const fn get_reverse_cmp(&self) -> Operator {
        match self {
            Operator::Gt => Operator::Lt,
            Operator::Lt => Operator::Gt,
            Operator::Gte => Operator::Lte,
            Operator::Lte => Operator::Gte,
            // there is no need in changing `==` and `!=`.
            _ => *self,
        }
    }

    #[inline]
    pub const fn is_set_operator(&self) -> bool {
        use Operator::*;
        matches!(self, And | Or | Unless)
    }

    pub const fn as_str(&self) -> &'static str {
        use Operator::*;
        match self {
            Add => "+",
            And => "and",
            Atan2 => "atan2",
            Default => "default",
            Div => "/",
            Eql => "==",
            Gt => ">",
            Gte => ">=",
            If => "if",
            IfNot => "ifNot",
            Mod => "%",
            Mul => "*",
            Lt => "<",
            Lte => "<=",
            NotEq => "!=",
            Or => "or",
            Pow => "^",
            Sub => "-",
            Unless => "unless",
        }
    }
}

impl FromStr for Operator {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Operator::try_from(s)
    }
}

impl TryFrom<&str> for Operator {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        if let Some(operator) = lookup_operator(op) {
            return Ok(operator);
        }
        Err(ParseError::General(format!("Unknown binary op {}", op)))
    }
}

impl TryFrom<Token> for Operator {
    type Error = ParseError;

    fn try_from(token: Token) -> Result<Self, Self::Error> {
        match token {
            Token::OpAnd => Ok(Operator::And),
            Token::OpAtan2 => Ok(Operator::Atan2),
            Token::OpDefault => Ok(Operator::Default),
            Token::OpDiv => Ok(Operator::Div),
            Token::OpEqual => Ok(Operator::Eql),
            Token::OpGreaterThan => Ok(Operator::Gt),
            Token::OpGreaterThanOrEqual => Ok(Operator::Gte),
            Token::OpIf => Ok(Operator::If),
            Token::OpIfNot => Ok(Operator::IfNot),
            Token::OpMod => Ok(Operator::Mod),
            Token::OpMul => Ok(Operator::Mul),
            Token::OpMinus => Ok(Operator::Sub),
            Token::OpLessThan => Ok(Operator::Lt),
            Token::OpLessThanOrEqual => Ok(Operator::Lte),
            Token::OpNotEqual => Ok(Operator::NotEq),
            Token::OpOr => Ok(Operator::Or),
            Token::OpPow => Ok(Operator::Pow),
            Token::OpUnless => Ok(Operator::Unless),
            Token::OpPlus => Ok(Operator::Add),
            _ => Err(ParseError::General(format!(
                "Unknown binary op {:?}",
                token
            ))),
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

pub fn is_binary_op(op: &str) -> bool {
    lookup_operator(op).is_some()
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use strum::IntoEnumIterator;
    use crate::ast::is_binary_op;
    use crate::prelude::Operator;

    #[test]
    fn test_is_binary_op_success() {
        let f = |s: &str| assert!(is_binary_op(s), "expecting valid binaryOp: {}", s);

        f("and");
        f("AND");
        f("unless");
        f("unleSS");
        f("==");
        f("!=");
        f(">=");
        f("<=");
        f("or");
        f("Or");
        f("+");
        f("-");
        f("*");
        f("/");
        f("%");
        f("atan2");
        f("^");
        f(">");
        f("<");
    }

    #[test]
    fn test_is_binary_op_error() {
        let f = |s: &str| {
            assert!(!is_binary_op(s), "unexpected valid binaryOp: {}", s);
        };

        f("foobar");
        f("=~");
        f("!~");
        f("=");
        f("<==");
        f("234");
    }
    
    #[test]
    fn test_operator_from_str() {
        for op in Operator::iter() {
            let op_str = op.as_str();
            assert_eq!(Operator::from_str(op.as_str()), Ok(op));
            
            let upper = op_str.to_ascii_uppercase();
            assert_eq!(Operator::from_str(&upper), Ok(op));
        }
    }
}
