use std::time::Duration;
use num_traits::FloatConst;

use metricsql_common::prelude::{datetime_part, timestamp_secs_to_utc_datetime, DateTimePart};

use crate::ast::{
    AggregationExpr, BinaryExpr, DurationExpr, Expr, FunctionExpr, NumberLiteral, Operator,
    ParensExpr, RollupExpr, UnaryExpr, WithArgExpr, WithExpr,
};
use crate::binaryop::{scalar_binary_operation, string_compare};
use crate::common::{RewriteRecursion, TreeNodeRewriter};
use crate::functions::{BuiltinFunction, TransformFunction};
use crate::parser::{parse_number, ParseError, ParseResult};

#[allow(rustdoc::private_intra_doc_links)]
/// Partially evaluate `Expr`s so constant subtrees are evaluated at plan time.
///
/// Note it does not handle algebraic rewrites such as `(a or false)`
/// --> `a`, which is handled by [`Simplifier`]
#[derive(Default)]
pub struct ConstEvaluator {
    /// `can_evaluate` is used during the depth-first-provider of the
    /// `Expr` tree to track if any siblings (or their descendants) were
    /// non-evaluatable (e.g. had a column reference or volatile
    /// function)
    ///
    /// Specifically, `can_evaluate[N]` represents the state of
    /// traversal when we are N levels deep in the tree, one entry for
    /// this Expr and each of its parents.
    ///
    /// After visiting all siblings if `can_evaluate.top()`` is true, that
    /// means there were no non-evaluatable siblings (or their
    /// descendants) so this `Expr` can be evaluated
    can_evaluate: Vec<bool>,
}

impl TreeNodeRewriter for ConstEvaluator {
    type N = Expr;

    fn pre_visit(&mut self, expr: &Expr) -> ParseResult<RewriteRecursion> {
        // Default to being able to evaluate this node
        self.can_evaluate.push(true);

        // if this expr is not ok to evaluate, mark entire parent
        // stack as not ok (as all parents have at least one child or
        // descendant that can not be evaluated

        if !can_evaluate(expr) {
            // walk back up stack, marking first parent that is not mutable
            let parent_iter = self.can_evaluate.iter_mut().rev();
            for p in parent_iter {
                if !*p {
                    // optimization: if we find an element on the
                    // stack already marked, know all elements above are also marked
                    break;
                }
                *p = false;
            }
        }

        // NB: do not short circuit recursion even if we find a non
        // evaluatable node (so we can fold other children, args to
        // functions, etc.)
        Ok(RewriteRecursion::Continue)
    }

    fn mutate(&mut self, expr: Expr) -> ParseResult<Expr> {
        match self.can_evaluate.pop() {
            Some(true) => evaluate_to_scalar(expr),
            Some(false) => Ok(expr),
            // todo: specific optimize error
            _ => Err(ParseError::General(
                "Failed to pop can_evaluate".to_string(),
            )),
        }
    }
}

impl ConstEvaluator {
    /// Create a new `ConstantEvaluator`. Session constants such as
    /// the time for `now()` are taken from the passed
    /// `execution_props`.
    pub fn new() -> Self {
        Self {
            can_evaluate: vec![],
        }
    }
}

/// Can the expression be evaluated at plan time, (assuming all of
/// its children can also be evaluated)?
pub(super) fn can_evaluate(expr: &Expr) -> bool {
    // check for reasons we can't evaluate this node
    //
    // NOTE all expr types are listed here so when new ones are
    // added they can be checked for their ability to be evaluated
    // at plan time
    match expr {
        Expr::Aggregation(_) | Expr::Rollup(_) => false,
        Expr::Parens(_) => true,
        // only handle immutable scalar functions
        Expr::Function(_) => true,
        Expr::NumberLiteral(_)
        | Expr::Duration(_)
        | Expr::StringLiteral(_)
        | Expr::UnaryOperator(_)
        | Expr::BinaryOperator(_) => true,
        Expr::StringExpr(se) => !se.is_expanded(),
        Expr::With(_) => false,
        Expr::WithSelector(_) => false,
        Expr::MetricExpression(_) => false,
    }
}

pub fn const_simplify(expr: Expr) -> ParseResult<Expr> {
    match expr {
        Expr::UnaryOperator(uo) => handle_unary_expr(uo),
        Expr::BinaryOperator(be) => handle_binary_expr(be),
        Expr::Function(fe) => handle_function_expr(fe),
        Expr::With(we) => handle_with_expr(we),
        Expr::Aggregation(ae) => handle_aggregation_expr(ae),
        Expr::Rollup(re) => handle_rollup_expr(re),
        Expr::Parens(p) => {
            let expressions = handle_expr_vecs(p.expressions)?;
            Ok(Expr::Parens(ParensExpr { expressions }))
        }
        _ => Ok(expr),
    }
}

/// Internal helper to evaluate an Expr
fn evaluate_to_scalar(expr: Expr) -> ParseResult<Expr> {
    match expr {
        Expr::UnaryOperator(ue) => handle_unary_expr(ue),
        Expr::BinaryOperator(be) => handle_binary_expr(be),
        Expr::Function(fe) => handle_function_expr(fe),
        _ => Ok(expr),
    }
}

fn handle_unary_expr(ue: UnaryExpr) -> ParseResult<Expr> {
    let mut ue = ue;
    match ue.expr.as_mut() {
        Expr::NumberLiteral(n) => {
            return Ok(Expr::from(n.value * -1.0));
        }
        Expr::Duration(d) => {
            return match d {
                DurationExpr::Millis(left_val) => {
                    let dur = DurationExpr::new(*left_val * -1);
                    Ok(Expr::Duration(dur))
                }
                DurationExpr::StepValue(left_val) => {
                    let n = *left_val * -1.0;
                    let dur = DurationExpr::new_step(n);
                    Ok(Expr::Duration(dur))
                }
            }
        }
        Expr::UnaryOperator(ue2) => {
            return Ok(std::mem::take(&mut ue2.expr));
        }
        _ => {}
    }
    Ok(Expr::UnaryOperator(ue))
}

fn handle_binary_expr(be: BinaryExpr) -> ParseResult<Expr> {
    let is_bool = be.returns_bool();

    match (be.left.as_ref(), be.right.as_ref(), be.op) {
        (Expr::Duration(ln), Expr::Duration(rn), op) if op == Operator::Add || op == Operator::Sub => {
            handle_duration_duration(ln, rn, op, is_bool)
        }
        (Expr::Duration(ln), Expr::NumberLiteral(NumberLiteral { value }), op) if !ln.requires_step() && (op == Operator::Add || op == Operator::Sub) => {
            handle_duration_number(ln, *value, op, is_bool)
        }
        (Expr::Duration(ln), Expr::NumberLiteral(NumberLiteral { value }), op) if ln.requires_step() && (op == Operator::Mul || op == Operator::Div) => {
            handle_duration_step(ln, *value, op, is_bool)
        }
        (Expr::NumberLiteral(ln), Expr::NumberLiteral(rn), op) => {
            handle_number_number(ln.value, rn.value, op, is_bool)
        }
        (Expr::StringLiteral(left), Expr::StringLiteral(right), op) => {
            handle_string_string(left, right, op, is_bool)
        }
        _ => Ok(Expr::BinaryOperator(be)),
    }
}

fn handle_duration_duration(ln: &DurationExpr, rn: &DurationExpr, op: Operator, is_bool: bool) -> ParseResult<Expr> {
    match (ln, rn) {
        (DurationExpr::Millis(left_val), DurationExpr::Millis(right_val)) => {
            let n = scalar_binary_operation(*left_val as f64, *right_val as f64, op, is_bool)? as i64;
            Ok(Expr::Duration(DurationExpr::new(n)))
        }
        (DurationExpr::StepValue(left_val), DurationExpr::StepValue(right_val)) => {
            let n = scalar_binary_operation(*left_val, *right_val, op, is_bool)?;
            Ok(Expr::Duration(DurationExpr::new_step(n)))
        }
        _ => Ok(Expr::BinaryOperator(BinaryExpr { left: Box::new(Expr::Duration(ln.clone())), right: Box::new(Expr::Duration(rn.clone())), op, modifier: None })),
    }
}

fn handle_duration_number(ln: &DurationExpr, value: f64, op: Operator, is_bool: bool) -> ParseResult<Expr> {
    let secs = value * 1e3_f64;
    let n = scalar_binary_operation(ln.value(Duration::from_millis(1)) as f64, secs, op, is_bool)? as i64;
    Ok(Expr::Duration(DurationExpr::new(n)))
}

fn handle_duration_step(ln: &DurationExpr, value: f64, op: Operator, is_bool: bool) -> ParseResult<Expr> {
    if let DurationExpr::StepValue(step_value) = ln {
        let n = scalar_binary_operation(*step_value, value, op, is_bool)?;
        Ok(Expr::Duration(DurationExpr::new_step(n)))
    } else {
        Ok(Expr::BinaryOperator(BinaryExpr { left: Box::new(Expr::Duration(ln.clone())), right: Box::new(Expr::NumberLiteral(NumberLiteral { value })), op, modifier: None }))
    }
}

fn handle_number_number(ln: f64, rn: f64, op: Operator, is_bool: bool) -> ParseResult<Expr> {
    let n = scalar_binary_operation(ln, rn, op, is_bool)?;
    Ok(Expr::from(n))
}

fn handle_string_string(left: &str, right: &str, op: Operator, is_bool: bool) -> ParseResult<Expr> {
    if op == Operator::Add {
        if left.is_empty() {
            return Ok(Expr::from(right));
        } else if right.is_empty() {
            return Ok(Expr::from(left));
        }
        let mut res = String::with_capacity(left.len() + right.len());
        res += left;
        res += right;
        Ok(Expr::from(res))
    } else if op.is_comparison() {
        let n = string_compare(left, right, op, is_bool)?;
        Ok(Expr::from(n))
    } else {
        Ok(
            Expr::BinaryOperator(
                BinaryExpr { 
                    left: Box::new(Expr::from(left)),
                    right: Box::new(Expr::from(right)),
                    op, 
                    modifier: None 
                }
            )
        )
    }
}

fn get_single_scalar_arg(fe: &FunctionExpr) -> Option<f64> {
    if fe.args.len() == 1 {
        if let Expr::NumberLiteral(val) = &fe.args[0] {
            return Some(val.value);
        }
    }
    None
}

fn handle_function_expr(fe: FunctionExpr) -> ParseResult<Expr> {
    let arg_count = fe.args.len();
    match fe.function {
        BuiltinFunction::Transform(func) if arg_count == 1 && func == TransformFunction::Scalar => {
            if let Some(value) = handle_scalar_fn(&fe.args[0]) {
                return Ok(Expr::from(value));
            }
            Ok(Expr::Function(fe))
        }
        BuiltinFunction::Transform(func) if arg_count == 1 && func == TransformFunction::Vector => {
            let mut fe = fe;
            let arg = fe.args.remove(0);
            Ok(arg)
        }
        BuiltinFunction::Transform(func) => {
            use TransformFunction::*;

            if func == Pi && arg_count == 0 {
                return Ok(Expr::from(f64::PI()));
            }
            let arg = get_single_scalar_arg(&fe);
            if arg.is_none() {
                return Ok(Expr::Function(fe));
            }
            let arg = arg.unwrap();
            let mut valid = true;
            let value = match func {
                Abs => arg.abs(),
                Acos => arg.acos(),
                Acosh => arg.acosh(),
                Asin => arg.asin(),
                Asinh => arg.asinh(),
                Atan => arg.atan(),
                Atanh => arg.atanh(),
                Ceil => arg.ceil(),
                Cos => arg.cos(),
                Cosh => arg.cosh(),
                DayOfMonth => extract_datetime_part(arg, DateTimePart::DayOfMonth),
                DayOfWeek => extract_datetime_part(arg, DateTimePart::DayOfWeek),
                DayOfYear => extract_datetime_part(arg, DateTimePart::DayOfYear),
                DaysInMonth => extract_datetime_part(arg, DateTimePart::DaysInMonth),
                Deg => arg.to_degrees(),
                Exp => arg.exp(),
                Floor => arg.floor(),
                Hour => extract_datetime_part(arg, DateTimePart::Hour),
                Ln => arg.ln(),
                Log2 => arg.log2(),
                Log10 => arg.log10(),
                Minute => extract_datetime_part(arg, DateTimePart::Minute),
                Month => extract_datetime_part(arg, DateTimePart::Month),
                Rad => arg.to_radians(),
                Sgn => arg.signum(),
                Sin => arg.sin(),
                Sinh => arg.sinh(),
                Sqrt => arg.sqrt(),
                Tan => arg.tan(),
                Tanh => arg.tanh(),
                Year => extract_datetime_part(arg, DateTimePart::Year),
                _ => {
                    valid = false;
                    f64::NAN
                }
            };
            if valid {
                Ok(Expr::from(value))
            } else {
                Ok(Expr::Function(fe.clone()))
            }
        }
        _ => Ok(Expr::Function(fe)),
    }
}

pub(crate) fn extract_datetime_part(epoch_secs: f64, part: DateTimePart) -> f64 {
    if epoch_secs.is_nan() {
        return f64::NAN;
    }
    if let Some(utc) = timestamp_secs_to_utc_datetime(epoch_secs as i64) {
        if let Some(value) = datetime_part(utc, part) {
            return value as f64;
        }
    }
    f64::NAN
}

fn handle_scalar_fn(arg: &Expr) -> Option<f64> {
    match arg {
        // Verify whether the arg is a string.
        // Then try converting the string to number.
        Expr::StringLiteral(s) => {
            let n = parse_number(s).unwrap_or(f64::NAN);
            Some(n)
        }
        Expr::NumberLiteral(n) => Some(n.value),
        _ => None,
    }
}

fn handle_aggregation_expr(ae: AggregationExpr) -> ParseResult<Expr> {
    let args = handle_expr_vecs(ae.args)?;
    let new_aggregation = AggregationExpr { args, ..ae };
    Ok(Expr::Aggregation(new_aggregation))
}

fn handle_rollup_expr(re: RollupExpr) -> ParseResult<Expr> {
    let expr = const_simplify(*re.expr)?;
    let at = if let Some(at) = re.at {
        if can_evaluate(&at) {
            let simplified = const_simplify(*at)?;
            Some(Box::new(simplified))
        } else {
            None
        }
    } else {
        None
    };
    let new_expr = RollupExpr {
        expr: Box::new(expr),
        at,
        ..re
    };
    Ok(Expr::Rollup(new_expr))
}

fn handle_with_expr(we: WithExpr) -> ParseResult<Expr> {
    let expr = const_simplify(*we.expr)?;
    let was = we
        .was
        .into_iter()
        .map(|wa| {
            Ok(WithArgExpr {
                name: wa.name,
                args: wa.args,
                expr: const_simplify(wa.expr)?,
                token_range: wa.token_range,
            })
        })
        .collect::<ParseResult<Vec<WithArgExpr>>>()?;
    Ok(Expr::With(WithExpr {
        expr: Box::new(expr),
        was,
    }))
}

fn handle_expr_vecs(args: Vec<Expr>) -> ParseResult<Vec<Expr>> {
    args.into_iter()
        .map(const_simplify)
        .collect::<ParseResult<Vec<Expr>>>()
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use crate::ast::binary_expr;
    use crate::ast::utils::{expr_equals, lit, number, selector};
    use crate::common::TreeNode;
    use crate::parser::parse;

    use super::*;

    // ------------------------------
    // --- ConstEvaluator tests -----
    // ------------------------------
    fn test_const_simplify(input_expr: Expr, expected_expr: Expr) {
        let mut const_evaluator = ConstEvaluator::new();
        let evaluated_expr = input_expr
            .clone()
            .rewrite(&mut const_evaluator)
            .expect("successfully evaluated");

        assert!(
            expr_equals(&evaluated_expr, &expected_expr),
            "Mismatch evaluating {input_expr}\n  Expected:{expected_expr}\n  Got:{evaluated_expr}"
        );
    }

    #[test]
    fn test_const_evaluator() {
        // true --> true
        test_const_simplify(number(1.0), number(1.0));
        // true or true --> true
        test_const_simplify(number(1.0).or(number(1.0)), number(1.0));
        // true or false --> true
        test_const_simplify(number(1.0).or(number(0.0)), number(1.0));

        // "foo" == "foo" --> true
        test_const_simplify(lit("foo").eq(lit("foo")), number(1.0));
        // "foo" != "foo" --> false
        test_const_simplify(lit("foo").not_eq(lit("foo")), number(0.0));

        // c = 1 --> c = 1
        test_const_simplify(selector("c").eq(number(1.0)), selector("c").eq(number(1.0)));
        // c = 1 + 2 --> c + 3
        test_const_simplify(
            selector("c").eq(number(1.0) + number(2.0)),
            selector("c").eq(number(3.0)),
        );
        // (foo != foo) OR (c = 1) --> false OR (c = 1)
        test_const_simplify(
            lit("foo")
                .not_eq(lit("foo"))
                .or(selector("c").eq(number(1.0))),
            number(0.0).or(selector("c").eq(number(1.0))),
        );
    }

    #[test]
    fn test_const_evaluator_strings() {
        // "foo" + "bar" --> "foobar"
        test_const_simplify(lit("foo") + lit("bar"), lit("foobar"));

        // "foo" == "foo" --> true
        test_const_simplify(lit("foo").eq(lit("foo")), number(1.0));

        // "foo" != "foo" --> false
        test_const_simplify(lit("foo").not_eq(lit("foo")), number(0.0));

        // "foo" != "bar" --> false
        test_const_simplify(lit("foo").not_eq(lit("bar")), number(1.0));

        // "foo" > "bar" --> true
        test_const_simplify(lit("foo").gt(lit("bar")), number(1.0));

        // "foo" < "bar" --> false
        test_const_simplify(lit("foo").lt(lit("bar")), number(0.0));

        // "foo" >= "foo" --> true
        test_const_simplify(lit("foo").gt_eq(lit("foo")), number(1.0));

        // "foo_99" >= "foo" --> true
        test_const_simplify(lit("foo_99").gt_eq(lit("foo")), number(1.0));

        // "foo" <= "foo1" --> true
        test_const_simplify(lit("foo").lt_eq(lit("foo1")), number(1.0));

        // "foo" <= "foo" --> true
        test_const_simplify(lit("foo").lt_eq(lit("foo")), number(1.0));
    }

    #[test]
    fn test_const_evaluator_scalar_functions() {
        let rand = Expr::call("rand", vec![]).expect("invalid function call");
        let expr = rand.clone() + (number(1.0) + number(2.0));
        let expected = rand + number(3.0);
        test_const_simplify(expr, expected);

        // parenthesization matters: can't rewrite
        // (rand() + 1) + 2 --> (rand() + 1) + 2)
        let rand = Expr::call("rand", vec![]).expect("invalid function call");
        let expr = (rand + number(1.0)) + number(2.0);
        test_const_simplify(expr.clone(), expr);
    }

    fn test_math_fn(name: &str, arg: f64, expected: f64) {
        let expr = Expr::call(name, vec![number(arg)]).expect("invalid function call");
        test_const_simplify(expr, number(expected));
    }

    #[test]
    fn test_const_evaluator_math_function() {
        test_math_fn("abs", -1.0, 1.0);
        test_math_fn("abs", 1.0, 1.0);

        test_math_fn("acos", 2.0, 2_f64.acos());

        // acosh
        test_math_fn("acosh", 2.0, 2_f64.acosh());

        // asin
        test_math_fn("asin", 1.0, 1_f64.asin());

        // asinh
        test_math_fn("asinh", 1.0, 1_f64.asinh());

        test_math_fn("atan", 1.0, 1_f64.atan());

        // atanh
        test_math_fn("atanh", 0.5, 0.5_f64.atanh());

        // ceil
        test_math_fn("ceil", 0.0, 0.0);
        test_math_fn("ceil", 1.1, 2.0);

        // cos
        test_math_fn("cos", 0.5, 0.5_f64.cos());

        // cosh
        test_math_fn("cosh", 1.0, 1_f64.cosh());

        // deg
        test_math_fn("deg", std::f64::consts::FRAC_PI_2, 90.0);

        // exp
        test_math_fn("exp", 1.0, 1_f64.exp());

        // floor
        test_math_fn("floor", 0.0, 0.0);
        test_math_fn("floor", 1.1, 1.0);

        // ln
        test_math_fn("ln", 1.0, 0.0);
        test_math_fn("ln", std::f64::consts::E, 1.0);

        // log10
        test_math_fn("log10", 1.0, 0.0);
        test_math_fn("log10", 10.0, 1.0);

        // log2
        test_math_fn("log2", 1.0, 0.0);

        // rad
        test_math_fn("rad", 90.0, std::f64::consts::FRAC_PI_2);

        // sgn
        test_math_fn("sgn", -4.5, -1.0);

        // sin
        test_math_fn("sin", std::f64::consts::FRAC_PI_2, 1.0);

        // sinh
        test_math_fn("sinh", 1.0, 1_f64.sinh());

        // sqrt
        test_math_fn("sqrt", 4.0, 2.0);

        // tan
        test_math_fn("tan", 0.75, 0.75_f64.tan());

        // tanh
        test_math_fn("tanh", 1.0, 1_f64.tanh());
    }

    fn test_date_part_fn(name: &str, epoch_secs: f64, part: DateTimePart) {
        let expr = Expr::call(name, vec![number(epoch_secs)]).expect("invalid function call");
        let value = extract_datetime_part(epoch_secs, part);
        test_const_simplify(expr, number(value));
    }

    #[test]
    fn test_const_evaluator_date_parts() {
        let now = Utc::now();
        let epoch = now.timestamp() as f64;

        test_date_part_fn("day_of_month", epoch, DateTimePart::DayOfMonth);
        test_date_part_fn("days_in_month", epoch, DateTimePart::DaysInMonth);
        test_date_part_fn("day_of_week", epoch, DateTimePart::DayOfWeek);
        test_date_part_fn("day_of_year", epoch, DateTimePart::DayOfYear);

        test_date_part_fn("hour", epoch, DateTimePart::Hour);
        test_date_part_fn("minute", epoch, DateTimePart::Minute);
        test_date_part_fn("month", epoch, DateTimePart::Month);

        test_date_part_fn("year", epoch, DateTimePart::Year);
    }

    fn duration_millis(ms: i64) -> Expr {
        Expr::Duration(DurationExpr::new(ms))
    }

    fn duration_step(value: f64) -> Expr {
        Expr::Duration(DurationExpr::new_step(value))
    }

    #[test]
    fn test_const_evaluator_durations() {
        // handle durations with steps
        let left = duration_step(1.0);
        let right = duration_step(2.5);
        let expr = binary_expr(left, Operator::Add, right);
        let expected = duration_step(3.5);
        test_const_simplify(expr, expected);

        let left = duration_step(1.0);
        let right = duration_step(2.5);
        let expr = binary_expr(left, Operator::Sub, right);
        let expected = duration_step(-1.5);
        test_const_simplify(expr, expected);

        // 2.5i * 2
        let left = duration_step(2.5);
        let right = number(2.0);
        let expr = binary_expr(left, Operator::Mul, right);
        let expected = duration_step(5.0);
        test_const_simplify(expr, expected);

        // 5i / 2
        let left = duration_step(5.0);
        let right = number(2.0);
        let expr = binary_expr(left, Operator::Div, right);
        let expected = duration_step(2.5);
        test_const_simplify(expr, expected);

        // duration(millis) + duration(millis)
        let left = duration_millis(1000);
        let right = duration_millis(2500);
        let expr = binary_expr(left, Operator::Add, right);
        let expected = duration_millis(3500);
        test_const_simplify(expr, expected);

        // duration(millis) - duration(millis)
        let left = duration_millis(1000);
        let right = duration_millis(2500);
        let expr = binary_expr(left, Operator::Sub, right);
        let expected = duration_millis(-1500);
        test_const_simplify(expr, expected);

        // duration(millis) + number
        let left = duration_millis(1000);
        let right = number(2.0);
        let expr = binary_expr(left, Operator::Add, right);
        let expected = duration_millis(3000);
        test_const_simplify(expr, expected);

        // duration(millis) - number
        let left = duration_millis(1000);
        let right = number(2.0);
        let expr = binary_expr(left, Operator::Sub, right);
        let expected = duration_millis(-1000);
        test_const_simplify(expr, expected);
    }

    #[test]
    fn test_scalar_vector() {
        struct TestCase {
            expr: &'static str,
            expected: f64,
        }

        let tests = vec![
            TestCase {
                expr: "(9+scalar(vector(-10)))",
                expected: -1.0,
            },
            TestCase {
                expr: r#"(scalar("12.90"))"#,
                expected: 12.90,
            },
            TestCase {
                expr: "scalar(9+vector(4)) / 2",
                expected: 6.5,
            },
            TestCase {
                expr: r#"scalar(
                scalar(
                    scalar(
                        vector( 20 - 4 ) * 0.5 - 2
                    ) - vector( 2 )
                ) + vector(2)
            ) * 9"#,
                expected: 54.0,
            },
            TestCase {
                expr: "5 - scalar(
                scalar(
                    scalar(
                        vector( 20 - 4 ) * vector(0.5) - vector(2)
                    ) - vector( 2 )
                ) + vector(2)
            )",
                expected: -1.0,
            },
            TestCase {
                expr: "scalar(vector(1) + vector(2))",
                expected: 3.0,
            },
            TestCase {
                expr: "scalar(vector(1) + scalar(vector(1) + vector(2)))",
                expected: 4.0,
            },
            TestCase {
                expr: "scalar(vector(1) + scalar(vector(1) + scalar(vector(1) + vector(2))))",
                expected: 5.0,
            },
            TestCase {
                expr: "(scalar(9+vector(4)) * 4 - 9+scalar(vector(3)))",
                expected: 46.0,
            },
            TestCase {
                expr: "scalar(1 +vector(2 != bool 1))",
                expected: 2.0,
            },
            TestCase {
                expr: "scalar(1 +vector(1 != bool 1))",
                expected: 1.0,
            },
            TestCase {
                expr: "1 >= bool 1",
                expected: 1.0,
            },
            TestCase {
                expr: "1 >= bool 2",
                expected: 0.0,
            },
        ];

        for tt in tests {
            let expr = parse(tt.expr).expect("parse failed");
            test_const_simplify(expr, number(tt.expected));
        }
    }
}
