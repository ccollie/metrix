use metricsql_parser::prelude::{string_compare, Operator};

use crate::types::QueryValue;
use crate::{RuntimeError, RuntimeResult};

// move to parser binop module ?
pub(crate) fn eval_string_string_binop(
    op: Operator,
    left: &str,
    right: &str,
    is_bool: bool,
) -> RuntimeResult<QueryValue> {
    match op {
        Operator::Add => {
            if left.is_empty() {
                Ok(right.into())
            } else if right.is_empty() {
                Ok(left.into())
            } else {
                let mut res = String::with_capacity(left.len() + right.len());
                res += left;
                res += right;
                Ok(QueryValue::String(res))
            }
        }
        _ => {
            let cmp = string_compare(left, right, op, is_bool).map_err(|_| {
                RuntimeError::Internal(format!("Invalid operator {op} in string comparison"))
            })?;
            Ok(QueryValue::Scalar(cmp))
        }
    }
}
