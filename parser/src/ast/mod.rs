pub use adjust_comparison_ops::*;
pub use check_ast::*;
pub use expr::*;
pub use expr_tree_node::*;
pub use operator::*;
pub use string_expr::*;
pub use utils::*;

mod adjust_comparison_ops;
mod check_ast;
mod expr;
mod expr_tree_node;
pub mod operator;
mod string_expr;
pub mod utils;
mod visitor;

const INDENT_STR: &str = "  ";
const MAX_CHARACTERS_PER_LINE: usize = 100;

/// Approach
/// --------
/// When a PromQL query is parsed, it is converted into PromQL AST,
/// which is a nested structure of nodes. Each node has a depth/level
/// (distance from the root), that is passed by its parent.
///
/// While prettifying, a Node considers 2 things:
/// 1. Did the current Node's parent add a new line?
/// 2. Does the current Node needs to be prettified?
///
/// The level of a Node determines if it should be indented or not.
/// The answer to the 1 is NO if the level passed is 0. This means, the
/// parent Node did not apply a new line, so the current Node must not
/// apply any indentation as prefix.
/// If level > 1, a new line is applied by the parent. So, the current Node
/// should prefix an indentation before writing any of its content. This indentation
/// will be ([level/depth of current Node] * "  ").
///
/// The answer to 2 is YES if the normalized length of the current Node exceeds
/// the [MAX_CHARACTERS_PER_LINE] limit. Hence, it applies the indentation equal to
/// its depth and increments the level by 1 before passing down the child.
/// If the answer is NO, the current Node returns the normalized string value of itself.
pub trait Prettier: std::fmt::Display {
    /// max param is short for max_characters_per_line.
    fn pretty(&self, level: usize, max: usize) -> String {
        if self.needs_split(max) {
            self.format(level, max)
        } else {
            format!("{}{self}", indent(level))
        }
    }

    /// override format if expr needs to be split into multiple lines
    fn format(&self, level: usize, _max: usize) -> String {
        format!("{}{self}", indent(level))
    }

    /// override needs_split to return false, in order not to split multiple lines
    fn needs_split(&self, max: usize) -> bool {
        self.to_string().len() > max
    }
}

fn indent(n: usize) -> String {
    INDENT_STR.repeat(n)
}

pub(super) fn prettify_args(args: &[Expr], level: usize, max: usize) -> String {
    if args.is_empty() {
        return "".to_string();
    }
    let mut v = Vec::with_capacity(args.len());
    for ex in args {
        v.push(ex.pretty(level, max));
    }
    v.join(",\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Pretty(String);

    impl std::fmt::Display for Pretty {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl Prettier for Pretty {}

    #[test]
    fn test_prettier_trait() {
        let max = 10;
        let level = 1;

        let p = Pretty("demo".into());
        assert!(!p.needs_split(max));
        assert_eq!(p.format(level, max), p.pretty(level, max));

        let p = Pretty("demo_again.".into());
        assert!(p.needs_split(max));
        assert_eq!(p.format(level, max), p.pretty(level, max));
    }
}
