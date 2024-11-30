// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Signature module contains foundational types that are used to represent signatures, types,
//! and return types of functions.

use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::common::ValueType;
use crate::functions::MAX_ARG_COUNT;
use crate::parser::{ParseError, ParseResult};

///A function's volatility, which defines the functions eligibility for certain optimizations
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum Volatility {
    /// Immutable - An immutable function will always return the same output when given the same
    /// input. An example of this is [super::BuiltinScalarFunction::Cos].
    Immutable,
    /// Stable - A stable function may return different values given the same input across different
    /// queries but must return the same value for a given input within a query. An example of
    /// this is [super::BuiltinScalarFunction::Now].
    Stable,
    /// Volatile - A volatile function may change the return value from evaluation to evaluation.
    /// Multiple invocations of a volatile function may return different results when used in the
    /// same query. An example of this is [super::BuiltinScalarFunction::Random].
    Volatile,
}

/// A function's type signature, which defines the function's supported argument types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeSignature {
    /// arbitrary number of arguments of any common type out of a list of valid types
    /// second element is the min number of arguments
    /// A function such as `concat` is `Variadic(vec![ValueType::String, ValueType::Float], 1)`
    Variadic(Vec<ValueType>, usize),
    /// arbitrary number of arguments of an arbitrary but equal type, with possible minimum
    /// A function such as `array` is `VariadicEqual`
    VariadicEqual(ValueType, usize),
    /// arbitrary number of arguments of any type, with possible minimum
    VariadicAny(usize),
    /// fixed number of arguments of an arbitrary type out of a list of valid types
    /// A function of one argument of f64 is `Uniform(1, ValueType::Scalar)`
    Uniform(ValueType, usize),
    /// arguments of an exact type with an optional minimum
    Exact(Vec<ValueType>, Option<usize>),
    /// fixed number of arguments of arbitrary types
    Any(usize),
}

impl TypeSignature {
    /// Validate argument counts matches the `signature`.
    pub fn validate_arg_count(&self, name: &str, arg_len: usize) -> ParseResult<()> {
        fn expect_arg_count(name: &str, arg_len: usize, expected: usize) -> ParseResult<()> {
            if arg_len != expected {
                return Err(ParseError::ArgumentError(format!(
                    "The function {name}() expected {expected} arguments but received {arg_len}",
                )));
            }
            Ok(())
        }

        fn expect_min_args(name: &str, args_len: usize, min: usize) -> ParseResult<()> {
            if args_len < min {
                return Err(ParseError::ArgumentError(format!(
                    "The function {name}() expected a minimum of {min} arguments but received {args_len}",
                )));
            }
            Ok(())
        }

        match self {
            TypeSignature::Exact(valid_types, min) => {
                let max = valid_types.len();
                if let Some(min) = min {
                    if !(*min..=max).contains(&arg_len) {
                        return Err(ParseError::ArgumentError(format!(
                            "The function {name}() expected between {min} and {max} arguments but received {arg_len}",
                        )));
                    }
                    return Ok(());
                }
                expect_arg_count(name, arg_len, max)
            }
            TypeSignature::Any(min)
            | TypeSignature::Uniform(_, min)
            | TypeSignature::VariadicEqual(_, min)
            | TypeSignature::Variadic(_, min)
            | TypeSignature::VariadicAny(min) => expect_min_args(name, arg_len, *min),
        }
    }

    pub fn is_variadic(&self) -> bool {
        matches!(
            self,
            TypeSignature::Variadic(_, _)
                | TypeSignature::VariadicEqual(_, _)
                | TypeSignature::VariadicAny(_)
        )
    }
}

/// The Signature of a function defines its supported input types as well as its volatility.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Signature {
    /// type_signature - The types that the function accepts. See [TypeSignature] for more information.
    pub type_signature: TypeSignature,
    /// volatility - The volatility of the function. See [Volatility] for more information.
    pub volatility: Volatility,

    pub return_type: ValueType,
}

impl Signature {
    /// new - Creates a new Signature from any type signature and the volatility.
    pub fn new(type_signature: TypeSignature, volatility: Volatility) -> Self {
        Signature {
            type_signature,
            volatility,
            return_type: ValueType::InstantVector,
        }
    }

    /// variadic - Creates a variadic signature that represents an arbitrary number of arguments all from a type in common_types.
    pub fn variadic(common_types: Vec<ValueType>, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::Variadic(common_types, 0),
            volatility,
            return_type: ValueType::InstantVector,
        }
    }

    /// variadic - Creates a variadic signature that represents an arbitrary number of arguments all from a type in common_types.
    pub fn variadic_min(common_types: Vec<ValueType>, min: usize, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::Variadic(common_types, min),
            volatility,
            return_type: ValueType::InstantVector,
        }
    }

    /// variadic_equal - Creates a variadic signature that represents an arbitrary number of arguments of the same type.
    pub fn variadic_equal(valid_type: ValueType, min: usize, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::VariadicEqual(valid_type, min),
            volatility,
            return_type: ValueType::InstantVector,
        }
    }

    /// uniform - Creates a signature with a fixed number of arguments of the same type, which must be from valid_types.
    pub fn uniform(arg_count: usize, valid_type: ValueType, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::Uniform(valid_type, arg_count),
            volatility,
            return_type: ValueType::InstantVector,
        }
    }

    /// exact - Creates a signature which must match the types in exact_types in order.
    pub fn exact(exact_types: Vec<ValueType>, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::Exact(exact_types, None),
            volatility,
            return_type: ValueType::InstantVector,
        }
    }

    /// exact - Creates a signature which must match the types in exact_types in order, but with
    /// a minimum number of args.
    pub fn exact_with_min_args(
        exact_types: Vec<ValueType>,
        min: usize,
        volatility: Volatility,
    ) -> Self {
        // todo: panic if out of range
        let min_arg = min.clamp(0, exact_types.len());
        Signature {
            type_signature: TypeSignature::Exact(exact_types, Some(min_arg)),
            volatility,
            return_type: ValueType::InstantVector,
        }
    }

    /// any - Creates a signature which can be made of any type but of a fixed number
    pub fn any(arg_count: usize, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::Any(arg_count),
            volatility,
            return_type: ValueType::InstantVector,
        }
    }

    /// variadic_any - creates a variadic signature that represents an arbitrary number of arguments of any type.
    pub fn variadic_any(min_arg_count: usize, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::VariadicAny(min_arg_count),
            volatility,
            return_type: ValueType::InstantVector,
        }
    }

    /// Validate argument counts matches the `signature`.
    pub fn validate_arg_count(&self, name: &str, arg_len: usize) -> Result<(), ParseError> {
        self.type_signature.validate_arg_count(name, arg_len)
    }

    pub fn expand_types(&self) -> (Vec<ValueType>, usize) {
        // todo: also return min count
        match &self.type_signature {
            TypeSignature::Variadic(types, min) => (types.clone(), *min),
            TypeSignature::VariadicEqual(data_type, min) => (vec![*data_type; MAX_ARG_COUNT], *min),
            TypeSignature::Uniform(data_type, count) => (vec![*data_type; MAX_ARG_COUNT], *count),
            TypeSignature::Exact(types, min) => (types.clone(), min.unwrap_or(types.len())),
            TypeSignature::VariadicAny(min) => {
                (vec![ValueType::InstantVector; MAX_ARG_COUNT], *min)
            }
            TypeSignature::Any(count) => {
                (vec![ValueType::InstantVector; *count], *count) // TODO:: !!!! have a ValueType::Any
            }
        }
    }

    pub fn types(&self) -> TypeIterator<'_> {
        TypeIterator::new(self)
    }

    pub fn min_args(&self) -> usize {
        match &self.type_signature {
            TypeSignature::Exact(_, min) => min.unwrap_or(0),
            TypeSignature::Any(count) => *count,
            TypeSignature::Uniform(_, count) => *count,
            TypeSignature::Variadic(_, min) => *min,
            TypeSignature::VariadicEqual(_, min) => *min,
            TypeSignature::VariadicAny(min) => *min,
        }
    }

    pub fn is_variadic(&self) -> bool {
        self.type_signature.is_variadic()
    }

    pub fn args_iter(&self) -> TypeIterator<'_> {
        TypeIterator::new(self)
    }
}

impl Display for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (types, min) = self.expand_types();
        write!(f, "Signature({:?}, min: {})", types, min)
    }
}

pub struct TypeIterator<'a> {
    signature: &'a Signature,
    arg_index: usize,
}

impl Iterator for TypeIterator<'_> {
    type Item = ValueType;

    fn next(&mut self) -> Option<ValueType> {
        use TypeSignature::*;

        match &self.signature.type_signature {
            Variadic(types, _min) => {
                if self.arg_index < types.len() {
                    self.arg_index += 1;
                    Some(types[self.arg_index - 1])
                } else {
                    Some(types[types.len() - 1])
                }
            }
            VariadicEqual(data_type, _) => {
                if self.arg_index < MAX_ARG_COUNT {
                    self.arg_index += 1;
                    Some(*data_type)
                } else {
                    None
                }
            }
            Uniform(data_type, count) => {
                if self.arg_index < *count {
                    self.arg_index += 1;
                    Some(*data_type)
                } else {
                    None
                }
            }
            Exact(types, _) => {
                if self.arg_index < types.len() {
                    self.arg_index += 1;
                    Some(types[self.arg_index - 1])
                } else {
                    None
                }
            }
            VariadicAny(count) | Any(count) => {
                if self.arg_index < *count {
                    self.arg_index += 1;
                    // ?? TODO: !!!! have a ValueType::Any
                    Some(ValueType::InstantVector)
                } else {
                    None
                }
            }
        }
    }
}

impl<'a> TypeIterator<'a> {
    pub fn new(signature: &'a Signature) -> Self {
        Self {
            signature,
            arg_index: 0,
        }
    }
}
