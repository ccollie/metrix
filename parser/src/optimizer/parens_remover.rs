use crate::ast::{
    AggregationExpr, BinaryExpr, Expr, FunctionExpr, ParensExpr, RollupExpr, UnaryExpr
};
use crate::common::{RewriteRecursion, TreeNodeRewriter};
use crate::parser::ParseResult;

#[allow(rustdoc::private_intra_doc_links)]
#[derive(Debug, Clone, PartialEq)]
/// Remove unnecessary parentheses from an `Expr`s
pub struct ParensRemover {}

impl TreeNodeRewriter for ParensRemover {
    type N = Expr;

    /// Invoked before (Preorder) any children of `node` are rewritten /
    /// visited. Default implementation returns `Ok(Recursion::Continue)`
    fn pre_visit(&mut self, node: &Self::N) -> ParseResult<RewriteRecursion> {
        if !should_remove_parens(node) {
            Ok(RewriteRecursion::Stop)
        } else {
            Ok(RewriteRecursion::Continue)
        }
    }

    fn mutate(&mut self, node: Self::N) -> ParseResult<Self::N> {
        Ok(remove_parens_expr(node))
    }
}

impl Default for ParensRemover {
    fn default() -> Self {
        Self::new()
    }
}

impl ParensRemover {
    pub fn new() -> Self {
        ParensRemover {}
    }

    pub fn remove_parens(&self, e: Expr) -> Expr {
        remove_parens_expr(e)
    }
}

fn unnest_parens(pe: &mut ParensExpr) {
    while pe.expressions.len() == 1 {
        if let Some(Expr::Parens(pe2)) = pe.expressions.get_mut(0) {
            *pe = pe2.clone(); // todo: take/swap
        } else {
            break;
        }
    }
}

fn remove_parens_args(args: Vec<Expr>) -> Vec<Expr> {
    args.into_iter().map(remove_parens_expr).collect()
}

/// remove_parens_expr removes parensExpr for (Expr) case.
pub fn remove_parens_expr(e: Expr) -> Expr {
    if !should_remove_parens(&e) {
        return e;
    }
    match e {
        Expr::Rollup(re) => Expr::Rollup(RollupExpr {
            expr: Box::new(remove_parens_expr(*re.expr)),
            at: re
                .at
                .map(|at| Box::new(remove_parens_expr(*at))),
            window: re.window,
            step: re.step,
            offset: re.offset,
            inherit_step: re.inherit_step,
        }),
        Expr::BinaryOperator(be) => Expr::BinaryOperator(BinaryExpr {
            left: Box::new(remove_parens_expr(*be.left)),
            right: Box::new(remove_parens_expr(*be.right)),
            op: be.op,
            modifier: be.modifier,
        }),
        Expr::UnaryOperator(ue) => Expr::UnaryOperator(UnaryExpr {
            expr: Box::new(remove_parens_expr(*ue.expr)),
        }),
        Expr::Aggregation(ae) => Expr::Aggregation(AggregationExpr {
            function: ae.function,
            args: ae.args.into_iter().map(remove_parens_expr).collect(),
            modifier: ae.modifier,
            limit: ae.limit,
            keep_metric_names: ae.keep_metric_names,
        }),
        Expr::Function(fe) => Expr::Function(FunctionExpr {
            args: remove_parens_args(fe.args),
            keep_metric_names: fe.keep_metric_names,
            function: fe.function,
        }),
        Expr::Parens(pe) => {
            let mut pe = pe;
            unnest_parens(&mut pe);
            let len = pe.len();
            let mut args = remove_parens_args(pe.expressions);
            if len == 1 {
                return args.remove(0);
            }
            Expr::Parens(ParensExpr { expressions: args })
        }
        _ => e,
    }
}

pub fn should_remove_parens(e: &Expr) -> bool {
    match e {
        Expr::Rollup(re) => {
            let mut should_remove = should_remove_parens(&re.expr);
            if let Some(at) = &re.at {
                should_remove |= should_remove_parens(at);
            }
            should_remove
        }
        Expr::BinaryOperator(be) => {
            should_remove_parens(&be.left) || should_remove_parens(&be.right)
        }
        Expr::UnaryOperator(ue) => should_remove_parens(&ue.expr),
        Expr::Aggregation(ae) => ae.args.iter().any(should_remove_parens),
        Expr::Function(fe) => fe.args.iter().any(should_remove_parens),
        Expr::Parens(_) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{Expr, ParensExpr};
    use crate::optimizer::parens_remover::remove_parens_expr;

    #[test]
    fn test_remove_parens_expr() {
        let empty_parens = Expr::Parens(ParensExpr::new(vec![]));
        let actual = Expr::Parens(ParensExpr::new(vec![empty_parens.clone()]));

        let result = remove_parens_expr(actual);
        assert_expr_eq(&empty_parens, &result);
    }

    fn assert_expr_eq(expected: &Expr, actual: &Expr) {
        assert_eq!(
            expected, actual,
            "expected: \n{}\n but got: \n{}",
            expected, actual
        );
    }
}
