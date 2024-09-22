pub use aggregate::*;
use metricsql_common::hash::FastHashMap;
pub use rollup::*;
use serde::{Deserialize, Serialize};
pub use signature::*;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use std::sync::{LazyLock};
use strum::{IntoEnumIterator};
pub use transform::*;

use crate::ast::Expr;
use crate::common::ValueType;
use crate::parser::{validate_function_args, ParseError, ParseResult};

mod aggregate;
mod rollup;
mod signature;
mod transform;

/// Maximum number of arguments permitted in a rollup function. This really only applies
/// to variadic functions like `aggr_over_time` and `quantiles_over_time`
const MAX_ARG_COUNT: usize = 32;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinFunction {
    Aggregate(AggregateFunction),
    Rollup(RollupFunction),
    Transform(TransformFunction),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinFunctionType {
    Aggregate,
    Rollup,
    Transform,
}

impl BuiltinFunctionType {
    pub const fn to_str(&self) -> &'static str {
        match self {
            BuiltinFunctionType::Aggregate => "aggregate",
            BuiltinFunctionType::Rollup => "rollup",
            BuiltinFunctionType::Transform => "transform",
        }
    }
}

impl FromStr for BuiltinFunctionType {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, ParseError> {
        match s {
            s if s.eq_ignore_ascii_case("aggregate") => Ok(BuiltinFunctionType::Aggregate),
            s if s.eq_ignore_ascii_case("rollup") => Ok(BuiltinFunctionType::Rollup),
            s if s.eq_ignore_ascii_case("transform") => Ok(BuiltinFunctionType::Transform),
            _ => Err(ParseError::InvalidFunction(s.to_string())),
        }
    }
}

impl Display for BuiltinFunctionType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FunctionMeta {
    pub name: &'static str,
    pub function: BuiltinFunction,
    pub signature: Signature,
}

impl FunctionMeta {
    pub fn lookup(name: &str) -> Option<&'static FunctionMeta> {
        FUNCTION_REGISTRY
            .get(name)
            .or_else(|| FUNCTION_REGISTRY.get(name.to_lowercase().as_str()))
    }

    pub fn get_rollup_function(name: &str) -> ParseResult<&'static FunctionMeta> {
        if let Some(meta) = FunctionMeta::lookup(name) {
            if let BuiltinFunction::Rollup(_) = &meta.function {
                return Ok(meta);
            }
        }
        Err(ParseError::InvalidFunction(format!("rollup::{name}")))
    }

    pub fn get_aggregate_function(name: &str) -> ParseResult<&'static FunctionMeta> {
        FunctionMeta::lookup(name)
            .filter(|meta| matches!(meta.function, BuiltinFunction::Aggregate(_)))
            .ok_or_else(|| ParseError::InvalidFunction(format!("aggregate::{name}")))
    }

    pub fn get_type(&self) -> BuiltinFunctionType {
        self.function.get_type()
    }

    pub fn validate_arg_count(&self, name: &str, arg_len: usize) -> ParseResult<()> {
        self.signature.validate_arg_count(name, arg_len)
    }

    pub fn validate_args(&self, args: &[Expr]) -> ParseResult<()> {
        validate_function_args(&self.function, args)
    }

    pub fn is_aggregation(&self) -> bool {
        matches!(self.function, BuiltinFunction::Aggregate(_))
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self.function, BuiltinFunction::Transform(func) if func.return_type() == ValueType::Scalar)
    }

    pub fn is_rollup_function(&self, func: RollupFunction) -> bool {
        matches!(self.function, BuiltinFunction::Rollup(rf) if rf == func)
    }

    pub fn is_variadic(&self) -> bool {
        self.signature.is_variadic()
    }

    pub fn args_iter(&self) -> TypeIterator<'_> {
        self.signature.args_iter()
    }
}

impl Display for FunctionMeta {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        let mut min = self.signature.min_args();
        if min == 0 {
            min = MAX_ARG_COUNT;
        }
        let is_variadic = self.is_variadic();
        let mut bracket_written = false;
        for (i, arg) in self.args_iter().enumerate() {
            if i > 0 {
                if i == min {
                    if is_variadic {
                        write!(f, ", ...")?;
                        break;
                    }
                    if !bracket_written {
                        bracket_written = true;
                        write!(f, "[")?;
                    }
                }
                write!(f, ", ")?;
            }
            write!(f, "{}", arg)?;
        }

        if bracket_written {
            write!(f, "]")?;
        }

        write!(f, ")")
    }
}

// TODO: use blart
pub type FunctionRegistry = FastHashMap<&'static str, FunctionMeta>;

pub static FUNCTION_REGISTRY: LazyLock<FunctionRegistry> = LazyLock::new(init_registry);


fn init_registry() -> FunctionRegistry {
    let mut registry = FunctionRegistry::default();

    for af in AggregateFunction::iter() {
        let name = af.name();
        let function = BuiltinFunction::Aggregate(af);
        let signature = af.signature();
        registry.insert(
            name,
            FunctionMeta {
                name,
                function,
                signature,
            },
        );
    }

    for rf in RollupFunction::iter() {
        let name = rf.name();
        let function = BuiltinFunction::Rollup(rf);
        let signature = rf.signature();
        registry.insert(
            name,
            FunctionMeta {
                name,
                function,
                signature,
            },
        );
    }

    for tf in TransformFunction::iter() {
        let name = tf.name();
        let function = BuiltinFunction::Transform(tf);
        let signature = tf.signature();
        registry.insert(
            name,
            FunctionMeta {
                name,
                function,
                signature,
            },
        );
    }

    registry
}

impl BuiltinFunction {
    pub fn new(name: &str) -> ParseResult<Self> {
        if let Some(meta) = FunctionMeta::lookup(name) {
            return Ok(meta.function.clone());
        }
        Err(ParseError::InvalidFunction(format!("built-in::{name}")))
    }

    pub fn is_supported(name: &str) -> bool {
        Self::new(name).is_ok()
    }

    pub fn name(&self) -> &'static str {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => af.name(),
            Rollup(rf) => rf.name(),
            Transform(tf) => tf.name(),
        }
    }

    pub fn signature(&self) -> Signature {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => af.signature(),
            Rollup(rf) => rf.signature(),
            Transform(tf) => tf.signature(),
        }
    }

    pub fn validate_args(&self, args: &[Expr]) -> ParseResult<()> {
        validate_function_args(self, args)
    }

    pub fn type_name(&self) -> &'static str {
        self.get_type().to_str()
    }

    pub fn get_type(&self) -> BuiltinFunctionType {
        use BuiltinFunction::*;
        match self {
            Aggregate(_) => BuiltinFunctionType::Aggregate,
            Rollup(_) => BuiltinFunctionType::Rollup,
            Transform(_) => BuiltinFunctionType::Transform,
        }
    }

    pub fn is_type(&self, other: BuiltinFunction) -> bool {
        self.type_name() == other.type_name()
    }

    pub fn is_aggregate_func(name: &str) -> bool {
        matches!(FunctionMeta::lookup(name), Some(meta) if meta.is_aggregation())
    }

    pub fn is_aggregation(&self) -> bool {
        matches!(self, BuiltinFunction::Aggregate(_))
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, BuiltinFunction::Transform(func) if func.return_type() == ValueType::Scalar)
    }

    pub fn is_rollup_function(&self, func: RollupFunction) -> bool {
        matches!(self, BuiltinFunction::Rollup(rf) if rf == &func)
    }

    pub fn may_sort_results(&self) -> bool {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => af.may_sort_results(),
            Rollup(_) => false, // todo
            Transform(tf) => tf.may_sort_results(),
        }
    }

     pub fn get_arg_for_optimization<'a>(&'a self, args: &'a [Expr]) -> Option<&'a Expr> {
        self.get_arg_idx_for_optimization(args.len()).and_then(|idx| args.get(idx))
    }

    pub fn get_arg_idx_for_optimization(&self, args_len: usize) -> Option<usize> {
        match self {
            BuiltinFunction::Aggregate(af) => get_aggregate_arg_idx_for_optimization(*af, args_len),
            BuiltinFunction::Rollup(rf) => get_rollup_arg_idx_for_optimization(*rf, args_len),
            BuiltinFunction::Transform(tf) => get_transform_arg_idx_for_optimization(*tf, args_len),
        }
    }

    pub fn return_type(&self, args: &[Expr]) -> ParseResult<ValueType> {
        if self.is_scalar() {
            return Ok(ValueType::Scalar);
        }

        // determine the arg to pass through
        let arg = self.get_arg_for_optimization(args);

        let kind = if let Some(exp) = arg {
            exp.return_type()
        } else {
            // todo: does this depend on the function type (rollup, transform, aggregation)
            ValueType::InstantVector
        };

        match self {
            BuiltinFunction::Rollup(rf) => {
                if is_rollup_aggregation_over_time(*rf) {
                    match kind {
                        ValueType::RangeVector => Ok(ValueType::InstantVector),
                        ValueType::Scalar | ValueType::InstantVector => {
                            Ok(ValueType::InstantVector)
                        }
                        _ => {
                            // invalid arg
                            Err(ParseError::General(format!(
                                "aggregation over time is not valid with Expr returning {:?}",
                                kind
                            )))
                        }
                    }
                } else {
                    Ok(kind)
                }
            }
            _ => Ok(kind),
        }
    }
}

impl Display for BuiltinFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => write!(f, "{}", af),
            Rollup(rf) => write!(f, "{}", rf),
            Transform(tf) => write!(f, "{}", tf),
        }
    }
}

impl FromStr for BuiltinFunction {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl TryFrom<&str> for BuiltinFunction {
    type Error = ParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_is_supported() {
        assert!(super::BuiltinFunction::is_supported("geomean"));
        assert!(super::BuiltinFunction::is_supported("Predict_Linear"));
        assert!(super::BuiltinFunction::is_supported("minute"));
        assert!(!super::BuiltinFunction::is_supported("foo"));
    }
}
