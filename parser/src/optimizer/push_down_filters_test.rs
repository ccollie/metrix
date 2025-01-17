#[cfg(test)]
mod tests {
    use crate::ast::utils::expr_equals;
    use crate::ast::{Expr, MetricExpr};
    use crate::optimizer::{get_common_label_filters, optimize, pushdown_binary_op_filters};
    use crate::parser::parse;

    fn parse_selector(q: &str) -> Expr {
        parse(q).unwrap_or_else(|e| panic!("unexpected error in parse({}): {:?}", q, e))
    }

    #[test]
    fn test_pushdown_binary_op_filters() {
        let f = |q: &str, filters: &str, result_expected: &str| {
            let expr = parse_selector(q);
            let orig = expr.to_string();
            let filters_expr = parse_selector(filters);
            match filters_expr {
                Expr::MetricExpression(me) => {
                    if me.matchers.or_matchers.len() > 1 {
                        panic!("filters={} mustn't contain 'or'", filters)
                    }

                    let mut lfs = vec![];
                    if me.matchers.matchers.len() == 1 {
                        lfs = me.matchers.matchers;
                    }

                    let result_expr = pushdown_binary_op_filters(&expr, lfs);
                    let expected_expr = parse(result_expected).expect("parse error in test");
                    let result = result_expr.to_string();
                    assert!(
                        expr_equals(&result_expr, &expected_expr),
                        "pushdown_binary_op_filters({}, {});\nwant: {},\ngot: {}",
                        q,
                        filters,
                        result_expected,
                        result
                    );
                    // Verify that the original e didn't change after pushdown_binary_op_filters() call
                    let s = expr.to_string();
                    assert_eq!(
                        s, orig,
                        "the original expression has been changed;\ngot\n{}\nwant\n{}",
                        s, orig
                    )
                }
                _ => {
                    panic!(
                        "filters={} must be a metrics expression; got {}",
                        filters, filters_expr
                    )
                }
            }
        };

        f("foo", "{}", "foo");
        f("foo", r#"{a="b"}"#, r#"foo{a="b"}"#);

        f(
            r#"round(rate(x[5m] offset -1h)) + 123 / {a="b"}"#,
            r#"{x="y"}"#,
            r#"round(rate(x{x="y"}[5m] offset -1h)) + (123 / {a="b", x="y"})"#,
        );

        // f(
        //     r#"foo + bar{x="y"}"#,
        //     r#"{c="d",a="b"}"#,
        //     r#"foo{a="b", c="d"} + bar{a="b", c="d", x="y"}"#,
        // );
        f("sum(x)", r#"{a="b"}"#, "sum(x)");
        f(r#"foo or bar"#, r#"{a="b"}"#, r#"foo{a="b"} or bar{a="b"}"#);
        f(r#"foo or on(x) bar"#, r#"{a="b"}"#, r#"foo or on (x) bar"#);
        f(
            r#"foo == on(x) group_LEft bar"#,
            r#"{a="b"}"#,
            r#"foo == on (x) group_left () bar"#,
        );
        f(
            r#"foo{x="y"} > ignoRIng(x) group_left(abc) bar"#,
            r#"{a="b"}"#,
            r#"foo{a="b", x="y"} > ignoring (x) group_left (abc) bar{a="b"}"#,
        );
        f(
            r#"foo{x="y"} >bool ignoring(x) group_right(abc,def) bar"#,
            r#"{a="b"}"#,
            r#"foo{a="b", x="y"} > bool ignoring (x) group_right (abc, def) bar{a="b"}"#,
        );
        f(
            r#"foo * ignoring(x) bar"#,
            r#"{a="b"}"#,
            r#"foo{a="b"} * ignoring (x) bar{a="b"}"#,
        );
        // f(
        //     r#"foo{f1!~"x"} UNLEss bar{f2=~"y.+"}"#,
        //     r#"{a="b",x=~"y"}"#,
        //     r#"foo{a="b", f1!~"x", x=~"y"} unless bar{a="b", f2=~"y.+", x=~"y"}"#,
        // );
        // f(
        //     r#"a / sum(x)"#,
        //     r#"{a="b",c=~"foo|bar"}"#,
        //     r#"a{a="b", c=~"foo|bar"} / sum(x)"#,
        // );
        f(
            r#"scalar(foo)+bar"#,
            r#"{a="b"}"#,
            r#"scalar(foo) + bar{a="b"}"#,
        );

        f("vector(foo)", r#"{a="b"}"#, "vector(foo)");

        // f(r#"vector(foo{x="y"} + a) + bar{a="b"}"#,
        //   r#"vector(foo{a="b",x="y"} + a{a="b",x="y"}) + bar{a="b",x="y"}"#);

        f(
            r#"{a="b"} + on() group_left() {c="d"}"#,
            r#"{a="b"}"#,
            r#"{a="b"} + on () group_left () {c="d"}"#,
        );
        f(
            r#"round(rate(x[5m] offset -1h)) + 123 / {a="b"}"#,
            r#"{x="y"}"#,
            r#"round(rate(x{x="y"}[5m] offset -1h)) + (123 / {a="b", x="y"})"#,
        );
    }

    #[test]
    fn test_label_set() {
        validate_optimized(r#"label_set(foo, "a", "bar") + x{__name__="y"}"#, r#"label_set(foo, "a", "bar") + x{__name__="y",a="bar"}"#);
        // label_set
        validate_optimized(r#"label_set(foo, "__name__", "bar") + x"#, r#"label_set(foo, "__name__", "bar") + x"#);
        validate_optimized(r#"label_set(foo, "a", "bar") + x{__name__="y"}"#, r#"label_set(foo, "a", "bar") + x{__name__="y",a="bar"}"#);
        validate_optimized(r#"label_set(foo{bar="baz"}, "xx", "y") + a{x="y"}"#, r#"label_set(foo{bar="baz",x="y"}, "xx", "y") + a{bar="baz",x="y",xx="y"}"#);
        validate_optimized(r#"label_set(foo{x="y"}, "q", "b", "x", "qwe") + label_set(bar{q="w"}, "x", "a", "q", "w")"#, r#"label_set(foo{x="y"}, "q", "b", "x", "qwe") + label_set(bar{q="w"}, "x", "a", "q", "w")"#);
        validate_optimized(r#"label_set(foo{a="b"}, "a", "qwe") + bar{a="x"}"#, r#"label_set(foo{a="b"}, "a", "qwe") + bar{a="qwe",a="x"}"#);
    }

    #[test]
    fn test_get_common_label_filters() {
        let get_filters = |q: &str| -> String {
            let e = parse_selector(q);
            let expr = optimize(e).expect("unexpected error in optimize()");

            let mut lfs = get_common_label_filters(&expr);
            lfs.sort();
            let mut me = MetricExpr::with_filters(lfs);
            me.to_string()
        };

        let f = |q, result_expected: &str| {
            let result = get_filters(q);
            assert_eq!(result, result_expected, "get_common_label_filters({});", q);
        };

        f("{}", "{}");
        f("foo", "{}");
        f(r#"{__name__="foo"}"#, "{}");
        f(r#"{__name__=~"bar"}"#, "{}");
        f(r#"{__name__=~"a|b",x="y"}"#, r#"{x="y"}"#);
        f(r#"foo{c!="d",a="b"}"#, r#"{a="b", c!="d"}"#);
        f(r#"1+foo"#, "{}");
        f(r#"foo + bar{a="b"}"#, r#"{a="b"}"#);
        f(r#"foo + bar / baz{a="b"}"#, r#"{a="b"}"#);
        f(r#"foo{x!="y"} + bar / baz{a="b"}"#, r#"{a="b", x!="y"}"#);
        f(
            r#"foo{x!="y"} + bar{x=~"a|b",q!~"we|rt"} / baz{a="b"}"#,
            r#"{a="b", q!~"we|rt", x=~"a|b", x!="y"}"#,
        );
        f(r#"{a="b"} + on() {c="d"}"#, "{}");
        f(r#"{a="b"} + on() group_left() {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} + on(a) group_left() {c="d"}"#, r#"{a="b"}"#);
        f(
            r#"{a="b"} + on(c) group_left() {c="d"}"#,
            r#"{a="b", c="d"}"#,
        );
        f(
            r#"{a="b"} + on(a,c) group_left() {c="d"}"#,
            r#"{a="b", c="d"}"#,
        );
        f(r#"{a="b"} + on(d) group_left() {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} + on() group_right(s) {c="d"}"#, r#"{c="d"}"#);
        f(
            r#"{a="b"} + On(a) groUp_right() {c="d"}"#,
            r#"{a="b", c="d"}"#,
        );
        f(r#"{a="b"} + on(c) group_right() {c="d"}"#, r#"{c="d"}"#);
        f(
            r#"{a="b"} + on(a,c) group_right() {c="d"}"#,
            r#"{a="b", c="d"}"#,
        );
        f(r#"{a="b"} + on(d) group_right() {c="d"}"#, r#"{c="d"}"#);
        f(r#"{a="b"} or {c="d"}"#, "{}");
        f(r#"{a="b",x="y"} or {x="y",c="d"}"#, r#"{x="y"}"#);
        f(r#"{a="b",x="y"} Or on() {x="y",c="d"}"#, "{}");
        f(r#"{a="b",x="y"} Or on(a) {x="y",c="d"}"#, "{}");
        f(r#"{a="b",x="y"} Or on(x) {x="y",c="d"}"#, r#"{x="y"}"#);
        f(r#"{a="b",x="y"} Or oN(x,y) {x="y",c="d"}"#, r#"{x="y"}"#);
        f(r#"{a="b",x="y"} Or on(y) {x="y",c="d"}"#, "{}");
        f(
            r#"(foo{a="b"} + bar{c="d"}) or (baz{x="y"} <= x{a="b"})"#,
            r#"{a="b"}"#,
        );
        f(r#"{a="b"} unless {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} unless on() {c="d"}"#, "{}");
        f(r#"{a="b"} unLess on(a) {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} unLEss on(c) {c="d"}"#, "{}");
        f(r#"{a="b"} unless on(a,c) {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} Unless on(x) {c="d"}"#, "{}");

        // common filters for 'or' filters
        f(r#"{a="b" or c="d",a="b"}"#, r#"{a="b"}"#);
        f(r#"{a="b",c="d" or c="d",a="b"}"#, r#"{a="b", c="d"}"#);
        f(r#"foo{x="y",a="b",c="d" or c="d",a="b"}"#, r#"{a="b", c="d"}"#);
    }

    #[test]
    fn test_reserved_words() {
        // reserved words. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/4422
        validate_optimized("1 + (on)", "1 + (on)");
        validate_optimized(r#"{a="b"} + (group_left)"#, r#"{a="b"} + (group_left{a="b"})"#);
        validate_optimized(r#"bool{a="b"} + (ignoring{c="d"})"#, r#"bool{a="b",c="d"} + (ignoring{a="b",c="d"})"#);
    }

    #[test]
    fn test_common_binary_expressions() {
        // common binary expressions
        validate_optimized("a + b", "a + b");
        validate_optimized(
            r#"foo{label1="value1"} == bar"#,
            r#"foo{label1="value1"} == bar{label1="value1"}"#,
        );
        validate_optimized(
            r#"foo{label1="value1"} == bar{label2="value2"}"#,
            r#"foo{label1="value1", label2="value2"} == bar{label1="value1", label2="value2"}"#,
        );
        validate_optimized(
            r#"foo + bar{b=~"a.*", a!="ss"}"#,
            r#"foo{a!="ss", b=~"a.*"} + bar{a!="ss", b=~"a.*"}"#,
        );
        validate_optimized(r#"foo{bar="1"} / 234"#, r#"foo{bar="1"} / 234"#);
        validate_optimized(
            r#"foo{bar="1"} / foo{bar="1"}"#,
            r#"foo{bar="1"} / foo{bar="1"}"#,
        );
        validate_optimized(r#"123 + foo{bar!~"xx"}"#, r#"123 + foo{bar!~"xx"}"#);
        validate_optimized(r#"foo or bar{x="y"}"#, r#"foo or bar{x="y"}"#);
        validate_optimized(
            r#"foo{x="y"} * on() baz{a="b"}"#,
            r#"foo{x="y"} * on () baz{a="b"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * on(a) baz{a="b"}"#,
            r#"foo{a="b", x="y"} * on (a) baz{a="b"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * on(bar) baz{a="b"}"#,
            r#"foo{x="y"} * on (bar) baz{a="b"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * on(x,a,bar) baz{a="b"}"#,
            r#"foo{a="b", x="y"} * on (a, bar, x) baz{a="b", x="y"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * ignoring() baz{a="b"}"#,
            r#"foo{a="b", x="y"} * ignoring () baz{a="b", x="y"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * ignoring(a) baz{a="b"}"#,
            r#"foo{x="y"} * ignoring (a) baz{a="b", x="y"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * ignoring(bar) baz{a="b"}"#,
            r#"foo{a="b", x="y"} * ignoring (bar) baz{a="b", x="y"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * ignoring(x,a,bar) baz{a="b"}"#,
            r#"foo{x="y"} * ignoring (x, a, bar) baz{a="b"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * ignoring() group_left(foo,bar) baz{a="b"}"#,
            r#"foo{a="b", x="y"} * ignoring () group_left (foo, bar) baz{a="b", x="y"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * on(a) group_left baz{a="b"}"#,
            r#"foo{a="b", x="y"} * on (a) group_left () baz{a="b"}"#,
        );
        validate_optimized(
            r#"foo{x="y"} * on(a) group_right(x, y) baz{a="b"}"#,
            r#"foo{a="b", x="y"} * on (a) group_right (x, y) baz{a="b"}"#,
        );
        validate_optimized(
            r#"sum(foo, bar{baz=~"sdf"} + aa{baz=~"axx", aa="b"})"#,
            r#"sum(foo, bar{aa="b", baz=~"axx", baz=~"sdf"} + aa{aa="b", baz=~"axx", baz=~"sdf"})"#,
        );
        validate_optimized(
            r#"sum(foo, bar{baz=~"sdf"} + aa{baz=~"axx", aa="b"})"#,
            r#"sum(foo, bar{aa="b", baz=~"axx", baz=~"sdf"} + aa{aa="b", baz=~"axx", baz=~"sdf"})"#,
        );
        validate_optimized(
            r#"foo AND bar{baz="aa"}"#,
            r#"foo{baz="aa"} and bar{baz="aa"}"#,
        );
        validate_optimized(
            r#"{x="y",__name__="a"} + {a="b"}"#,
            r#"a{a="b", x="y"} + {a="b", x="y"}"#,
        );
        validate_optimized(
            r#"{x="y",__name__=~"a|b"} + {a="b"}"#,
            r#"{__name__=~"a|b", a="b", x="y"} + {a="b", x="y"}"#,
        );
        validate_optimized(
            r#"a{x="y",__name__=~"a|b"} + {a="b"}"#,
            r#"a{__name__=~"a|b", a="b", x="y"} + {a="b", x="y"}"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on() group_left() {e="f"})"#,
            r#"{a="b", c="d"} + ({c="d"} * on () group_left () {e="f"})"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on(a) group_left() {e="f"})"#,
            r#"{a="b", c="d"} + ({a="b", c="d"} * on (a) group_left () {a="b", e="f"})"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on(c) group_left() {e="f"})"#,
            r#"{a="b", c="d"} + ({c="d"} * on (c) group_left () {c="d", e="f"})"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on(e) group_left() {e="f"})"#,
            r#"{a="b", c="d", e="f"} + ({c="d", e="f"} * on (e) group_left () {e="f"})"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on(x) group_left() {e="f"})"#,
            r#"{a="b", c="d"} + ({c="d"} * on (x) group_left () {e="f"})"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on() group_right() {e="f"})"#,
            r#"{a="b", e="f"} + ({c="d"} * on () group_right () {e="f"})"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on(a) group_right() {e="f"})"#,
            r#"{a="b", e="f"} + ({a="b", c="d"} * on (a) group_right () {a="b", e="f"})"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on(c) group_right() {e="f"})"#,
            r#"{a="b", c="d", e="f"} + ({c="d"} * on (c) group_right () {c="d", e="f"})"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on(e) group_right() {e="f"})"#,
            r#"{a="b", e="f"} + ({c="d", e="f"} * on (e) group_right () {e="f"})"#,
        );
        validate_optimized(
            r#"{a="b"} + ({c="d"} * on(x) group_right() {e="f"})"#,
            r#"{a="b", e="f"} + ({c="d"} * on (x) group_right () {e="f"})"#,
        );
    }

    #[test]
    fn test_specially_handled_binary_expressions() {
        // specially handled binary expressions
        validate_optimized(r#"foo{a="b"} or bar{x="y"}"#, r#"foo{a="b"} or bar{x="y"}"#);
        validate_optimized(
            r#"(foo{a="b"} + bar{c="d"}) or (baz{x="y"} <= x{a="b"})"#,
            r#"(foo{a="b", c="d"} + bar{a="b", c="d"}) or (baz{a="b", x="y"} <= x{a="b", x="y"})"#,
        );
        validate_optimized(
            r#"(foo{a="b"} + bar{c="d"}) or on(x) (baz{x="y"} <= x{a="b"})"#,
            r#"(foo{a="b", c="d"} + bar{a="b", c="d"}) or on (x) (baz{a="b", x="y"} <= x{a="b", x="y"})"#,
        );
        validate_optimized(
            r#"foo + (bar or baz{a="b"})"#,
            r#"foo + (bar or baz{a="b"})"#,
        );
        validate_optimized(
            r#"foo + (bar{a="b"} or baz{a="b"})"#,
            r#"foo{a="b"} + (bar{a="b"} or baz{a="b"})"#,
        );
        validate_optimized(
            r#"foo + (bar{a="b",c="d"} or baz{a="b"})"#,
            r#"foo{a="b"} + (bar{a="b", c="d"} or baz{a="b"})"#,
        );
        validate_optimized(
            r#"foo{a="b"} + (bar OR baz{x="y"})"#,
            r#"foo{a="b"} + (bar{a="b"} or baz{a="b", x="y"})"#,
        );
        validate_optimized(
            r#"foo{a="b"} + (bar{x="y",z="456"} OR baz{x="y",z="123"})"#,
            r#"foo{a="b", x="y"} + (bar{a="b", x="y", z="456"} or baz{a="b", x="y", z="123"})"#,
        );
        validate_optimized(
            r#"foo{a="b"} unless bar{c="d"}"#,
            r#"foo{a="b"} unless bar{a="b", c="d"}"#,
        );
        validate_optimized(
            r#"foo{a="b"} unless on() bar{c="d"}"#,
            r#"foo{a="b"} unless on () bar{c="d"}"#,
        );
        validate_optimized(
            r#"foo + (bar{x="y"} unless baz{a="b"})"#,
            r#"foo{x="y"} + (bar{x="y"} unless baz{a="b", x="y"})"#,
        );
        validate_optimized(
            r#"foo + (bar{x="y"} unless on() baz{a="b"})"#,
            r#"foo + (bar{x="y"} unless on () baz{a="b"})"#,
        );
        validate_optimized(
            r#"foo{a="b"} + (bar UNLESS baz{x="y"})"#,
            r#"foo{a="b"} + (bar{a="b"} unless baz{a="b", x="y"})"#,
        );
        validate_optimized(
            r#"foo{a="b"} + (bar{x="y"} unLESS baz)"#,
            r#"foo{a="b", x="y"} + (bar{a="b", x="y"} unless baz{a="b", x="y"})"#,
        );
    }

    #[test]
    fn test_optimize_aggregate_funcs() {
        // aggregate funcs
        validate_optimized(
            r#"sum(foo{bar="baz"}) / a{b="c"}"#,
            r#"sum(foo{bar="baz"}) / a{b="c"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) by () / a{b="c"}"#,
            r#"sum(foo{bar="baz"}) by () / a{b="c"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) by (bar) / a{b="c"}"#,
            r#"sum(foo{bar="baz"}) by (bar) / a{b="c", bar="baz"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) by (b) / a{b="c"}"#,
            r#"sum(foo{b="c", bar="baz"}) by (b) / a{b="c"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) by (x) / a{b="c"}"#,
            r#"sum(foo{bar="baz"}) by (x) / a{b="c"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) by (bar,b) / a{b="c"}"#,
            r#"sum(foo{b="c", bar="baz"}) by (bar, b) / a{b="c", bar="baz"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) without () / a{b="c"}"#,
            r#"sum(foo{b="c", bar="baz"}) without () / a{b="c", bar="baz"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) without (bar) / a{b="c"}"#,
            r#"sum(foo{b="c", bar="baz"}) without (bar) / a{b="c"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) without (b) / a{b="c"}"#,
            r#"sum(foo{bar="baz"}) without (b) / a{b="c", bar="baz"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) without (x) / a{b="c"}"#,
            r#"sum(foo{b="c", bar="baz"}) without (x) / a{b="c", bar="baz"}"#,
        );
        validate_optimized(
            r#"sum(foo{bar="baz"}) without (bar,b) / a{b="c"}"#,
            r#"sum(foo{bar="baz"}) without (bar, b) / a{b="c"}"#,
        );
        validate_optimized(
            r#"sum(foo, bar) by (a) + baz{a="b"}"#,
            r#"sum(foo{a="b"}, bar{a="b"}) by (a) + baz{a="b"}"#,
        );
        validate_optimized(
            r#"topk(3, foo) by (baz,x) + bar{baz="a"}"#,
            r#"topk(3, foo{baz="a"}) by (baz, x) + bar{baz="a"}"#,
        );
        validate_optimized(
            r#"topk(a, foo) without (x,y) + bar{baz="a"}"#,
            r#"topk(a, foo{baz="a"}) without (x, y) + bar{baz="a"}"#,
        );
        validate_optimized(
            r#"a{b="c"} + quantiles("foo", 0.1, 0.2, bar{x="y"}) by (b, x, y)"#,
            r#"a{b="c", x="y"} + quantiles("foo", 0.1, 0.2, bar{b="c", x="y"}) by (b, x, y)"#,
        );

        validate_optimized(
            r#"sum(
                avg(foo{bar="one"}) by (bar),
                avg(foo{bar="two"}[1i]) by (bar)
            ) by(bar)
            + avg(foo{bar="three"}) by(bar)"#,
            r#"sum(avg(foo{bar="one", bar="three"}) by(bar), avg(foo{bar="three", bar="two"}[1i]) by(bar)) by(bar) + avg(foo{bar="three"}) by(bar)"#,
        );

        validate_optimized(
            r#"sum(
                foo{bar="one"},
                avg(foo{bar="two"}[1i]) by (bar)
            ) by(bar)
                + avg(foo{bar="three"}) by(bar)"#,
            r#"sum(foo{bar="one",bar="three"}, avg(foo{bar="three",bar="two"}[1i]) by(bar)) by(bar) + avg(foo{bar="three"}) by(bar)"#,
        );

        validate_optimized(
            r#"count_values("foo", bar{baz="a"}) by (bar,b) + a{b="c"}"#,
            r#"count_values("foo", bar{baz="a"}) by (bar, b) + a{b="c"}"#,
        );
    }

    #[test]
    fn test_count_values() {
        // count_values
        validate_optimized(
            r#"count_values("foo", bar{a="b",c="d"}) by (a,x,y) + baz{foo="c",x="q",z="r"}"#,
            r#"count_values("foo", bar{a="b",c="d",x="q"}) by(a,x,y) + baz{a="b",foo="c",x="q",z="r"}"#,
        );
        validate_optimized(
            r#"count_values("foo", bar{a="b",c="d"}) by (a) + baz{foo="c",x="q",z="r"}"#,
            r#"count_values("foo", bar{a="b",c="d"}) by(a) + baz{a="b",foo="c",x="q",z="r"}"#,
        );
        validate_optimized(
            r#"count_values("foo", bar{a="b",c="d"}) + baz{foo="c",x="q",z="r"}"#,
            r#"count_values("foo", bar{a="b",c="d"}) + baz{foo="c",x="q",z="r"}"#,
        );
    }

    #[test]
    fn test_label_replace() {
        // Label_replace
        validate_optimized(r#"label_replace(foo, "a", "b", "c", "d") + bar{x="y"}"#, r#"label_replace(foo{x="y"}, "a", "b", "c", "d") + bar{x="y"}"#);
        validate_optimized(r#"label_replace(foo, "a", "b", "c", "d") + bar{a="y"}"#, r#"label_replace(foo, "a", "b", "c", "d") + bar{a="y"}"#);
        validate_optimized(r#"label_replace(foo{x="qwe"}, "a", "b", "c", "d") + bar{a="y"}"#, r#"label_replace(foo{x="qwe"}, "a", "b", "c", "d") + bar{a="y",x="qwe"}"#);
        validate_optimized(r#"label_replace(foo{x="qwe"}, "a", "b", "c", "d") + bar{x="y"}"#, r#"label_replace(foo{x="qwe",x="y"}, "a", "b", "c", "d") + bar{x="qwe",x="y"}"#);
        validate_optimized(r#"label_replace(foo{aa!="qwe"}, "a", "b", "c", "d") + bar{x="y"}"#, r#"label_replace(foo{aa!="qwe",x="y"}, "a", "b", "c", "d") + bar{aa!="qwe",x="y"}"#);
    }

    #[test]
    fn test_label_join() {
        // Label_join
        validate_optimized(r#"label_join(foo, "a", "b", "c") + bar{x="y"}"#, r#"label_join(foo{x="y"}, "a", "b", "c") + bar{x="y"}"#);
        validate_optimized(r#"label_join(foo, "a", "b", "c") + bar{a="y"}"#, r#"label_join(foo, "a", "b", "c") + bar{a="y"}"#);
        validate_optimized(r#"label_join(foo{a="qwe"}, "a", "b", "c") + bar{x="y"}"#, r#"label_join(foo{a="qwe",x="y"}, "a", "b", "c") + bar{x="y"}"#);
        validate_optimized(r#"label_join(foo{q="z"}, "a", "b", "c") + bar{a="y"}"#, r#"label_join(foo{q="z"}, "a", "b", "c") + bar{a="y",q="z"}"#);
        validate_optimized(r#"label_join(foo{q="z"}, "a", "b", "c") + bar{w="y"}"#, r#"label_join(foo{q="z",w="y"}, "a", "b", "c") + bar{q="z",w="y"}"#);
    }

    #[test]
    fn test_label_map() {
        // Label_map
        validate_optimized(r#"label_map(foo, "a", "x", "y") + bar{x="y"}"#, r#"label_map(foo{x="y"}, "a", "x", "y") + bar{x="y"}"#);
        validate_optimized(r#"label_map(foo{a="qwe",b="c"}, "a", "x", "y") + bar{a="rt",x="y"}"#, r#"label_map(foo{a="qwe",b="c",x="y"}, "a", "x", "y") + bar{a="rt",b="c",x="y"}"#);
    }

    #[test]
    fn test_label_match() {
        // Label_match
        validate_optimized(r#"label_match(foo, "a", "x", "y") + bar{x="y"}"#, r#"label_match(foo{x="y"}, "a", "x", "y") + bar{x="y"}"#);
        validate_optimized(r#"label_match(foo{a="qwe",b="c"}, "a", "x", "y") + bar{a="rt",x="y"}"#, r#"label_match(foo{a="qwe",b="c",x="y"}, "a", "x", "y") + bar{a="rt",b="c",x="y"}"#);
    }

    #[test]
    fn test_label_mismatch() {
        // Label_mismatch
        validate_optimized(r#"label_mismatch(foo, "a", "x", "y") + bar{x="y"}"#, r#"label_mismatch(foo{x="y"}, "a", "x", "y") + bar{x="y"}"#);
        validate_optimized(r#"label_mismatch(foo{a="qwe",b="c"}, "a", "x", "y") + bar{a="rt",x="y"}"#, r#"label_mismatch(foo{a="qwe",b="c",x="y"}, "a", "x", "y") + bar{a="rt",b="c",x="y"}"#);
    }

    #[test]
    fn test_label_transform() {
        // Label_transform
        validate_optimized(r#"label_transform(foo, "a", "x", "y") + bar{x="y"}"#, r#"label_transform(foo{x="y"}, "a", "x", "y") + bar{x="y"}"#);
        validate_optimized(r#"label_transform(foo{a="qwe",b="c"}, "a", "x", "y") + bar{a="rt",x="y"}"#, r#"label_transform(foo{a="qwe",b="c",x="y"}, "a", "x", "y") + bar{a="rt",b="c",x="y"}"#);
    }

    #[test]
    fn test_optimize_label_copy() {
        // Label_copy
        validate_optimized(r#"label_copy(foo, "a", "b") + bar{x="y"}"#, r#"label_copy(foo{x="y"}, "a", "b") + bar{x="y"}"#);
        validate_optimized(r#"label_copy(foo, "a", "b", "c", "d") + bar{a="y",b="z"}"#, r#"label_copy(foo{a="y"}, "a", "b", "c", "d") + bar{a="y",b="z"}"#);
        validate_optimized(r#"label_copy(foo{q="w"}, "a", "b") + bar{a="y",b="z"}"#, r#"label_copy(foo{a="y",q="w"}, "a", "b") + bar{a="y",b="z",q="w"}"#);
        validate_optimized(r#"label_copy(foo{b="w"}, "a", "b") + bar{a="y",b="z"}"#, r#"label_copy(foo{a="y",b="w"}, "a", "b") + bar{a="y",b="z"}"#);
    }

    #[test]
    fn test_label_del() {
        // Label_del
        validate_optimized(r#"label_del(foo, "a", "b") + bar{x="y"}"#, r#"label_del(foo{x="y"}, "a", "b") + bar{x="y"}"#);
        validate_optimized(r#"label_del(foo{a="q",b="w",z="d"}, "a", "b") + bar{a="y",b="z",x="y"}"#, r#"label_del(foo{a="q",b="w",x="y",z="d"}, "a", "b") + bar{a="y",b="z",x="y",z="d"}"#);
    }

    #[test]
    fn test_label_keep() {
        // Label_keep
        validate_optimized(r#"label_keep(foo, "a", "b") + bar{x="y"}"#, r#"label_keep(foo, "a", "b") + bar{x="y"}"#);
        validate_optimized(r#"label_keep(foo{a="q",c="d"}, "a", "b") + bar{x="y",b="z"}"#, r#"label_keep(foo{a="q",b="z",c="d"}, "a", "b") + bar{a="q",b="z",x="y"}"#);
    }

    #[test]
    fn test_label_uppercase() {
        // Label_uppercase
        validate_optimized(r#"label_uppercase(foo, "a", "b") + bar{x="y"}"#, r#"label_uppercase(foo{x="y"}, "a", "b") + bar{x="y"}"#);
        validate_optimized(r#"label_uppercase(foo{a="q",b="w",z="d"}, "a", "b") + bar{a="y",b="z",x="y"}"#, r#"label_uppercase(foo{a="q",b="w",x="y",z="d"}, "a", "b") + bar{a="y",b="z",x="y",z="d"}"#);
    }

    #[test]
    fn test_label_lowercase() {
        // Label_lowercase
        validate_optimized(r#"label_lowercase(foo, "a", "b") + bar{x="y"}"#, r#"label_lowercase(foo{x="y"}, "a", "b") + bar{x="y"}"#);
        validate_optimized(r#"label_lowercase(foo{a="q",b="w",z="d"}, "a", "b") + bar{a="y",b="z",x="y"}"#, r#"label_lowercase(foo{a="q",b="w",x="y",z="d"}, "a", "b") + bar{a="y",b="z",x="y",z="d"}"#);
    }

    #[test]
    fn test_labels_equal() {
        // Labels_equal
        validate_optimized(r#"labels_equal(foo, "a", "b") + bar{x="y"}"#, r#"labels_equal(foo{x="y"}, "a", "b") + bar{x="y"}"#);
        validate_optimized(r#"labels_equal(foo{a="q",b="w",z="d"}, "a", "b") + bar{a="y",b="z",x="y"}"#, r#"labels_equal(foo{a="q",b="w",x="y",z="d"}, "a", "b") + bar{a="y",b="z",x="y",z="d"}"#);
    }

    #[test]
    fn test_label_graphite_group() {
        validate_optimized(r#"label_graphite_group(foo, 1, 2) + bar{x="y"}"#, r#"label_graphite_group(foo{x="y"}, 1, 2) + bar{x="y"}"#);
        validate_optimized(r#"label_graphite_group({a="b",__name__="qwe"}, 1, 2) + {__name__="abc",x="y"}"#, r#"label_graphite_group(qwe{a="b",x="y"}, 1, 2) + abc{a="b",x="y"}"#);
    }

    #[test]
    fn test_range_normalize() {
        // range_normalize
        validate_optimized(
            r#"range_normalize(foo{a="b",c="d"},bar{a="b",x="y"}) + baz{z="w"}"#,
            r#"range_normalize(foo{a="b",c="d",z="w"}, bar{a="b",x="y",z="w"}) + baz{a="b",z="w"}"#,
        );
    }

    #[test]
    fn test_union() {
        // union
        validate_optimized(
            r#"union(foo{a="b",c="d"},bar{a="b",x="y"}) + baz{z="w"}"#,
            r#"union(foo{a="b",c="d",z="w"}, bar{a="b",x="y",z="w"}) + baz{a="b",z="w"}"#,
        );
    }

    #[test]
    fn test_optimize_transform_funcs() {
        // transform funcs
        validate_optimized(
            r#"round(foo{bar="baz"}) + sqrt(a{z=~"c"})"#,
            r#"round(foo{bar="baz", z=~"c"}) + sqrt(a{bar="baz", z=~"c"})"#,
        );
        validate_optimized(
            r#"foo{bar="baz"} + SQRT(a{z=~"c"})"#,
            r#"foo{bar="baz", z=~"c"} + SQRT(a{bar="baz", z=~"c"})"#,
        );
        validate_optimized(r#"round({__name__="foo"}) + bar"#, r#"round(foo) + bar"#);
        validate_optimized(
            r#"round({__name__=~"foo|bar"}) + baz"#,
            r#"round({__name__=~"foo|bar"}) + baz"#,
        );
        validate_optimized(
            r#"round({__name__=~"foo|bar",a="b"}) + baz"#,
            r#"round({__name__=~"foo|bar", a="b"}) + baz{a="b"}"#,
        );
        validate_optimized(
            r#"round({__name__=~"foo|bar",a="b"}) + sqrt(baz)"#,
            r#"round({__name__=~"foo|bar", a="b"}) + sqrt(baz{a="b"})"#,
        );
        validate_optimized(
            r#"round(foo) + {__name__="bar",x="y"}"#,
            r#"round(foo{x="y"}) + bar{x="y"}"#,
        );
        validate_optimized(
            r#"absent(foo{bar="baz"}) + sqrt(a{z=~"c"})"#,
            r#"absent(foo{bar="baz"}) + sqrt(a{z=~"c"})"#,
        );
        validate_optimized(
            r#"ABSENT(foo{bar="baz"}) + sqrt(a{z=~"c"})"#,
            r#"ABSENT(foo{bar="baz"}) + sqrt(a{z=~"c"})"#,
        );
        validate_optimized(
            r#"label_set(foo{bar="baz"}, "xx", "y") + a{x="y"}"#,
            r#"label_set(foo{bar="baz"}, "xx", "y") + a{x="y"}"#,
        );
        validate_optimized(
            r#"now() + foo{bar="baz"} + x{y="x"}"#,
            r#"(now() + foo{bar="baz", y="x"}) + x{bar="baz", y="x"}"#,
        );
        validate_optimized(
            r#"limit_offset(5, 10, {x="y"}) if {a="b"}"#,
            r#"limit_offset(5, 10, {a="b", x="y"}) if {a="b", x="y"}"#,
        );
        validate_optimized(
            r#"buckets_limit(aa, {x="y"}) if {a="b"}"#,
            r#"buckets_limit(aa, {a="b", x="y"}) if {a="b", x="y"}"#,
        );
        validate_optimized(
            r#"histogram_quantiles("q", 0.1, 0.9, {x="y"}) - {a="b"}"#,
            r#"histogram_quantiles("q", 0.1, 0.9, {a="b", x="y"}) - {a="b", x="y"}"#,
        );
        validate_optimized(
            r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le)) - {a="b"}"#,
            r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le)) - {a="b"}"#,
        );
        validate_optimized(
            r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le,x)) - {a="b"}"#,
            r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le, x)) - {a="b", x="y"}"#,
        );
        validate_optimized(
            r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le,x,a)) - {a="b"}"#,
            r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({a="b", x="y"}[5m])) by (le, x, a)) - {a="b", x="y"}"#,
        );
        validate_optimized(r#"vector(foo) + bar{a="b"}"#, r#"vector(foo) + bar{a="b"}"#);
        validate_optimized(
            r#"vector(foo{x="y"} + a) + bar{a="b"}"#,
            r#"vector(foo{x="y"} + a{x="y"}) + bar{a="b"}"#,
        );

        validate_optimized(
            r#"(foo{a="b",c="d"},bar{a="b",x="y"}) + baz{z="w"}"#,
            r#"(foo{a="b",c="d",z="w"}, bar{a="b",x="y",z="w"}) + baz{a="b",z="w"}"#,
        );
    }

    #[test]
    fn test_optimize_multi_level_transform_funcs() {
        // multilevel transform funcs
        validate_optimized(r#"round(sqrt(foo)) + bar"#, r#"round(sqrt(foo)) + bar"#);
        validate_optimized(
            r#"round(sqrt(foo)) + bar{b="a"}"#,
            r#"round(sqrt(foo{b="a"})) + bar{b="a"}"#,
        );
        validate_optimized(
            r#"round(sqrt(foo{a="b"})) + bar{x="y"}"#,
            r#"round(sqrt(foo{a="b", x="y"})) + bar{a="b", x="y"}"#,
        );
    }

    #[test]
    fn test_optimize_rollup_funcs() {
        // rollup funcs
        validate_optimized(
            r#"RATE(foo[5m]) / rate(baz{a="b"}) + increase(x{y="z"} offset 5i)"#,
            r#"(RATE(foo{a="b", y="z"}[5m]) / rate(baz{a="b", y="z"})) + increase(x{a="b", y="z"} offset 5i)"#,
        );
        validate_optimized(
            r#"sum(rate(foo[5m])) / rate(baz{a="b"})"#,
            r#"sum(rate(foo[5m])) / rate(baz{a="b"})"#,
        );
        validate_optimized(
            r#"sum(rate(foo[5m])) by (a) / rate(baz{a="b"})"#,
            r#"sum(rate(foo{a="b"}[5m])) by (a) / rate(baz{a="b"})"#,
        );
        validate_optimized(
            r#"rate({__name__="foo"}) + rate({__name__="bar",x="y"}) - rate({__name__=~"baz"})"#,
            r#"(rate(foo{x="y"}) + rate(bar{x="y"})) - rate({__name__=~"baz", x="y"})"#,
        );
        validate_optimized(
            r#"rate({__name__=~"foo|bar", x="y"}) + rate(baz)"#,
            r#"rate({__name__=~"foo|bar", x="y"}) + rate(baz{x="y"})"#,
        );
        // validate_optimized(
        //     r#"absent_over_time(foo{x="y"}[5m]) + bar{a="b"}"#,
        //     r#"absent_over_time(foo{x="y"}[5m]) + bar{a="b"}"#,
        // );
        validate_optimized(
            r#"{x="y"} + quantile_over_time(0.5, {a="b"})"#,
            r#"{a="b", x="y"} + quantile_over_time(0.5, {a="b", x="y"})"#,
        );
        validate_optimized(
            r#"quantiles_over_time("quantile", 0.1, 0.9, foo{x="y"}[5m] offset 4h) + bar{a!="b"}"#,
            r#"quantiles_over_time("quantile", 0.1, 0.9, foo{a!="b", x="y"}[5m] offset 4h) + bar{a!="b", x="y"}"#,
        );

        // count_values_over_time
        validate_optimized(
            r#"count_values_over_time("a", foo{a="x",b="c"}[5m]) + bar{a="y",d="e"}"#,
            r#"count_values_over_time("a", foo{a="x",b="c",d="e"}[5m]) + bar{a="y",b="c",d="e"}"#,
        );
    }

    #[test]
    fn test_optimize_at_modifier() {
        // @ modifier
        validate_optimized(
            r#"(foo @ end()) + bar{baz="a"}"#,
            r#"foo{baz="a"} @ end() + bar{baz="a"}"#,
        );
        validate_optimized(
            r#"sum(foo @ end()) + bar{baz="a"}"#,
            r#"sum(foo @ end()) + bar{baz="a"}"#,
        );
        validate_optimized(
            r#"foo @ (bar{a="b"} + baz{x="y"})"#,
            r#"foo @ (bar{a="b", x="y"} + baz{a="b", x="y"})"#,
        );
    }

    #[test]
    fn test_optimize_subqueries() {
        validate_optimized(
            r#"rate(avg_over_time(foo[5m:])) + bar{baz="a"}"#,
            r#"rate(avg_over_time(foo{baz="a"}[5m:])) + bar{baz="a"}"#,
        );
        // currently aggregation functions don't accept range vectors like VM does (as yet)
        /*
        validate_optimized(
            r#"rate(sum(foo[5m:])) + bar{baz="a"}"#,
            r#"rate(sum(foo[5m:])) + bar{baz="a"}"#,
        );
        validate_optimized(
            r#"rate(sum(foo[5m:]) by (baz)) + bar{baz="a"}"#,
            r#"rate(sum(foo{baz="a"}[5m:]) by (baz)) + bar{baz="a"}"#,
        );
         */
    }

    #[test]
    fn test_optimize_binop_with_consts_or_scalars() {
        // binary ops with constants or scalars
        validate_optimized(
            r#"100 * foo / bar{baz="a"}"#,
            r#"(100 * foo{baz="a"}) / bar{baz="a"}"#,
        );
        validate_optimized(
            r#"foo * 100 / bar{baz="a"}"#,
            r#"(foo{baz="a"} * 100) / bar{baz="a"}"#,
        );
        validate_optimized(
            r#"foo / bar{baz="a"} * 100"#,
            r#"(foo{baz="a"} / bar{baz="a"}) * 100"#,
        );
        validate_optimized(
            r#"scalar(x) * foo / bar{baz="a"}"#,
            r#"(scalar(x) * foo{baz="a"}) / bar{baz="a"}"#,
        );
        // validate_optimized(r#"SCALAR(x) * foo / bar{baz="a"}"#,  r#"(SCALAR(x) * foo{baz="a"}) / bar{baz="a"}"#);
        validate_optimized(
            r#"100 * on(foo) bar{baz="z"} + a"#,
            r#"(100 * on (foo) bar{baz="z"}) + a"#,
        );
    }

    #[test]
    fn test_optimize() {
        validate_optimized("foo", "foo");
    }

    fn validate_optimized(q: &str, expected: &str) {
        let e = parse_selector(q);
        let _orig = e.to_string();
        let e_optimized = optimize(e.clone()).expect("unexpected error in optimize()");
        let e_expected = parse_selector(expected);

        assert!(
            expr_equals(&e_optimized, &e_expected),
            "optimize() returned unexpected result;\ngot\n{}\nexpected\n{}",
            e_optimized,
            e_expected
        );

        // assert_eq!(q_optimized, expected, "\nquery: {}", q);
    }
}
