use rayon::iter::IntoParallelIterator;

use metricsql::ast::BExpression;
use metricsql::functions::{DataType, Signature, TypeSignature, Volatility};

use crate::{EvalConfig, Timeseries};
use crate::context::Context;
use crate::eval::{create_evaluators, Evaluator, ExprEvaluator};
use crate::eval::eval::eval_volatility;
use crate::functions::types::ParameterValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};

pub(crate) struct ArgList {
    args: Vec<ExprEvaluator>,
    parallel: bool,
    pub volatility: Volatility,
}

impl ArgList {
    pub fn new(signature: &Signature, args: &[BExpression]) -> RuntimeResult<Self> {
        let _args = create_evaluators(args)?;
        let volatility = eval_volatility(signature, &_args);
        Ok(Self {
            args: _args,
            volatility,
            parallel: should_parallelize_param_parsing(signature),
        })
    }

    pub fn from(signature: &Signature, args: Vec<ExprEvaluator>) -> Self {
        let volatility = eval_volatility(signature, &args);
        Self {
            args,
            volatility,
            parallel: should_parallelize_param_parsing(signature),
        }
    }

    pub fn eval(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<ParameterValue>> {
        // todo: use tinyvec and pass in as &mut vec
        if self.parallel {
            return self.eval_parallel(ctx, ec)
        } else {
            let mut res: Vec<ParameterValue> = Vec::with_capacity(self.args.len());
            for expr in self.args {
                let val = expr.eval(ctx, ec)?;
                res.push(val);
            }
            Ok(res)
        }
    }

    fn eval_parallel(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<ParameterValue>> {
        let mut err: Option<RuntimeError> = None;
        let res: Vec<Vec<Timeseries>> = self.args.into_par_iter().map(|expr| {
            match expr.eval(ctx, ec) {
                Err(e) => {
                    if err.is_none() {
                        err = Some(e)
                    }
                    vec![]
                }
                Ok(val) => {
                    val
                }
            }
        }).collect::<Vec<_>>();

        return match err {
            Some(e) => Err(e),
            None => Ok(res)
        };
    }
}


#[inline]
fn should_parallelize(t: &DataType) -> bool {
    !matches!(t, DataType::String | DataType::Float | DataType::Int | DataType::Vector)
}

/// Determines if we should parallelize parameter evaluation. We ignore "lightweight"
/// parameter types like `String` or `Int`
pub(crate) fn should_parallelize_param_parsing(signature: &Signature) -> bool {
    let types = &signature.type_signature;

    fn check_args(valid_types: &[DataType]) -> bool {
        valid_types.iter().filter(|x| should_parallelize(*x)).count() > 1
    }

    return match types {
        TypeSignature::Variadic(valid_types, min) => {
            check_args(valid_types)
        },
        TypeSignature::Uniform(number, valid_type) => {
            let types = &[*valid_type];
            return *number >= 2 && check_args(types)
        },
        TypeSignature::VariadicEqual(data_type, _) => {
            let types = &[*data_type];
            check_args(types)
        }
        TypeSignature::Exact(valid_types) => {
            check_args(valid_types)
        },
        TypeSignature::Any(number) => {
            if *number < 2 {
                return false
            }
            true
        }
    }
}
