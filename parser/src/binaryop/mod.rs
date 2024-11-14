use crate::ast::Operator;
use crate::parser::{ParseError, ParseResult};

pub type BinopFunc = fn(left: f64, right: f64) -> f64;

/// eq returns true if left == right.
#[inline]
fn op_eq(left: f64, right: f64) -> bool {
    // Special handling for nan == nan.
    if left.is_nan() {
        return right.is_nan();
    }
    left == right
}

/// neq returns true if left != right.
#[inline]
fn op_neq(left: f64, right: f64) -> bool {
    // Special handling for comparison with nan.
    if left.is_nan() {
        return !right.is_nan();
    }
    if right.is_nan() {
        return true;
    }
    left != right
}

fn op_and(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else {
        left
    }
}

// return the first non-NaN item. If both left and right are NaN, it returns NaN.
fn op_or(left: f64, right: f64) -> f64 {
    if !left.is_nan() {
        return left
    }
    right
}

/// gt returns true if left > right
#[inline]
fn op_gt(left: f64, right: f64) -> bool {
    left > right
}

/// lt returns true if left < right
#[inline]
fn op_lt(left: f64, right: f64) -> bool {
    left < right
}

/// Gte returns true if left >= right
#[inline]
fn op_gte(left: f64, right: f64) -> bool {
    left >= right
}

/// Lte returns true if left <= right
#[inline]
fn op_lte(left: f64, right: f64) -> bool {
    left <= right
}

/// Plus returns left + right
#[inline]
fn op_plus(left: f64, right: f64) -> f64 {
    left + right
}

/// Minus returns left - right
#[inline]
fn op_minus(left: f64, right: f64) -> f64 {
    left - right
}

/// Mul returns left * right
#[inline]
fn op_mul(left: f64, right: f64) -> f64 {
    left * right
}

/// Div returns left / right
/// Todo: protect against div by zero
#[inline]
fn op_div(left: f64, right: f64) -> f64 {
    if right == 0.0 {
        return f64::NAN;
    }
    left / right
}

/// returns left % right
#[inline]
fn op_mod(left: f64, right: f64) -> f64 {
    left % right
}

/// pow returns pow(left, right)
#[inline]
fn op_pow(left: f64, right: f64) -> f64 {
    left.powf(right)
}

/// returns atan2(left, right)
#[inline]
fn op_atan2(left: f64, right: f64) -> f64 {
    left.atan2(right)
}

/// returns left or right if left is NaN.
#[inline]
fn op_default(left: f64, right: f64) -> f64 {
    if left.is_nan() {
        return right;
    }
    left
}

/// returns left if right is not NaN. Otherwise, NaN is returned.
#[inline]
fn op_if(left: f64, right: f64) -> f64 {
    if right.is_nan() {
        return f64::NAN;
    }
    left
}

/// if_not returns left if right is NaN. Otherwise, NaN is returned.
#[inline]
pub fn op_if_not(left: f64, right: f64) -> f64 {
    if right.is_nan() {
        return left;
    }
    f64::NAN
}

#[inline]
pub fn op_unless(left: f64, right: f64) -> f64 {
    if right != left {
        return f64::NAN;
    }
    left
}

/// convert true to x, false to NaN.
#[inline]
pub const fn to_comparison_value(b: bool, x: f64) -> f64 {
    if b {
        x
    } else {
        f64::NAN
    }
}

macro_rules! make_comparison_func {
    ($name: ident, $func: expr) => {
        pub fn $name(left: f64, right: f64) -> f64 {
            to_comparison_value($func(left, right), left)
        }
    };
}

macro_rules! make_comparison_func_bool {
    ($name: ident, $func: expr) => {
        pub fn $name(left: f64, right: f64) -> f64 {
            if left.is_nan() {
                return f64::NAN;
            }
            if $func(left, right) { 1_f64 } else { 0_f64 }
        }
    };
}

make_comparison_func!(compare_eq, op_eq);
make_comparison_func!(compare_neq, op_neq);
make_comparison_func!(compare_gt, op_gt);
make_comparison_func!(compare_lt, op_lt);
make_comparison_func!(compare_gte, op_gte);
make_comparison_func!(compare_lte, op_lte);

make_comparison_func_bool!(compare_eq_bool, op_eq);
make_comparison_func_bool!(compare_neq_bool, op_neq);
make_comparison_func_bool!(compare_gt_bool, op_gt);
make_comparison_func_bool!(compare_lt_bool, op_lt);
make_comparison_func_bool!(compare_gte_bool, op_gte);
make_comparison_func_bool!(compare_lte_bool, op_lte);

pub const fn get_scalar_comparison_handler(op: Operator, is_bool: bool) -> BinopFunc {
    if is_bool {
        match op {
            Operator::Eql => compare_eq_bool,
            Operator::NotEq => compare_neq_bool,
            Operator::Gt => compare_gt_bool,
            Operator::Lt => compare_lt_bool,
            Operator::Gte => compare_gte_bool,
            Operator::Lte => compare_lte_bool,
            _ => unreachable!(),
        }
    } else {
        match op {
            Operator::Eql => compare_eq,
            Operator::NotEq => compare_neq,
            Operator::Gt => compare_gt,
            Operator::Lt => compare_lt,
            Operator::Gte => compare_gte,
            Operator::Lte => compare_lte,
            _ => unreachable!(),
        }
    }
}

pub const fn get_scalar_binop_handler(op: Operator, is_bool: bool) -> BinopFunc {
    match op {
        Operator::Add => op_plus,
        Operator::Atan2 => op_atan2,
        Operator::Default => op_default,
        Operator::Div => op_div,
        Operator::Mod => op_mod,
        Operator::Mul => op_mul,
        Operator::Pow => op_pow,
        Operator::Sub => op_minus,
        Operator::If => op_if,
        Operator::IfNot => op_if_not,
        Operator::Unless => op_unless,
        Operator::And => op_and,
        Operator::Or => op_or,
        Operator::Eql => get_scalar_comparison_handler(Operator::Eql, is_bool),
        Operator::NotEq => get_scalar_comparison_handler(Operator::NotEq, is_bool),
        Operator::Gt => get_scalar_comparison_handler(Operator::Gt, is_bool),
        Operator::Lt => get_scalar_comparison_handler(Operator::Lt, is_bool),
        Operator::Gte => get_scalar_comparison_handler(Operator::Gte, is_bool),
        Operator::Lte => get_scalar_comparison_handler(Operator::Lte, is_bool),
    }
}

pub fn eval_binary_op(left: f64, right: f64, op: Operator, is_bool: bool) -> f64 {
    let handler = get_scalar_binop_handler(op, is_bool);
    handler(left, right)
}

pub fn string_compare(a: &str, b: &str, op: Operator, is_bool: bool) -> ParseResult<f64> {
    let res = match op {
        Operator::Eql => a == b,
        Operator::NotEq => a != b,
        Operator::Lt => a < b,
        Operator::Gt => a > b,
        Operator::Lte => a <= b,
        Operator::Gte => a >= b,
        _ => {
            return Err(ParseError::Unsupported(format!(
                "unexpected operator {op} in string comparison"
            )))
        }
    };
    Ok(if res {
        1_f64
    } else if is_bool {
        0_f64
    } else {
        f64::NAN
    })
}

/// Supported operation between two float type values.
/// For one-off operations. This differs from the `get_scalar_binop_handler` in that it
/// is optimized for a single operation. The `get_scalar_binop_handler` is optimized for
/// a single operation that is applied to many values (it minimizes the number of branches).
pub fn scalar_binary_operation(
    lhs: f64,
    rhs: f64,
    op: Operator,
    return_bool: bool,
) -> ParseResult<f64> {
    use Operator::*;

    let value = if op.is_comparison() {
        let val = match op {
            Eql => op_eq(lhs, rhs),
            NotEq => op_neq(lhs, rhs),
            Gt => lhs > rhs,
            Lt => lhs < rhs,
            Gte => lhs >= rhs,
            Lte => lhs <= rhs,
            _ => {
                unreachable!("Unsupported scalar comparison operation: {lhs} {op} {rhs}",)
            }
        };
        if return_bool {
            val as u32 as f64
        } else {
            // if the return value was true, that means our element
            // satisfies the comparison, hence return it
            if val {
                lhs
            } else {
                f64::NAN
            }
        }
    } else {
        match op {
            Add => lhs + rhs,
            Sub => lhs - rhs,
            Mul => lhs * rhs,
            Div => op_div(lhs, rhs),
            Pow => lhs.powf(rhs),
            Mod => lhs % rhs,
            Atan2 => lhs.atan2(rhs),
            Default => op_default(lhs, rhs),
            If => op_if(lhs, rhs),
            IfNot => op_if_not(lhs, rhs),
            And => op_and(lhs, rhs),
            Or => op_or(lhs, rhs),
            Unless => f64::NAN,
            _ => {
                return Err(ParseError::Unsupported(format!(
                    "Unsupported scalar operation: {lhs} {op} {rhs}",
                )))
            }
        }
    };
    Ok(value)
}
