#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use strum::IntoEnumIterator;

    use crate::ast::Expr;
    use crate::label::{Matcher, Matchers};
    use crate::optimizer::optimize;
    use crate::parser::{parse, ParseError, ParseResult};
    use crate::prelude::Operator;

    struct Case {
        input: String,
        expected: ParseResult<Expr>,
    }

    impl Case {
        fn new(input: &str, expected: ParseResult<Expr>) -> Self {
            Case {
                input: String::from(input),
                expected,
            }
        }

        fn new_result_cases(cases: Vec<(&str, Expr)>) -> Vec<Case> {
            cases
                .into_iter()
                .map(|(input, expected)| Case::new(input, Ok(expected)))
                .collect()
        }

        fn new_expr_cases(cases: Vec<(&str, Expr)>) -> Vec<Case> {
            cases
                .into_iter()
                .map(|(input, expected)| Case::new(input, Ok(expected)))
                .collect()
        }

        fn new_fail_cases(cases: Vec<(&str, &str)>) -> Vec<Case> {
            cases
                .into_iter()
                .map(|(input, expected)| {
                    Case::new(input, Err(ParseError::General(expected.to_string())))
                })
                .collect()
        }
    }

    fn assert_cases(cases: Vec<Case>) {
        for Case { input, expected } in cases {
            assert_eq!(expected, crate::parser::parse(&input));
        }
    }

    fn parse_expr(s: &str) -> Expr {
        parse(s).unwrap_or_else(|e| panic!("Error parsing expression {s}: {:?}", e))
    }

    fn another(s: &str, expected: &str) {
        let expr = parse_expr(s);
        let optimized = optimize(expr).expect("Error optimizing expression");
        let expected_expr = parse_expr(expected);
        assert_eq!(expected_expr, optimized);
        // assert_eq_expr(&expected_expr, &optimized);

        // let res = optimized.to_string();
        // assert_eq!(&res, expected, "\nquery: {}", s)
    }

    fn same(s: &str) {
        another(s, s)
    }

    #[test]
    fn test_parse_number_expr() {
        fn another(s: &str, expected: &str) {
            let expected_val: f64 = expected.parse::<f64>().expect("parse f64");

            let expr = parse_expr(s);
            match expr {
                Expr::NumberLiteral(ne) => {
                    let actual = ne.value;
                    let valid = if actual.is_nan() {
                        expected_val.is_nan()
                    } else {
                        actual == expected_val
                    };

                    assert!(
                        valid,
                        "error parsing number \"{s}\", got {actual}, expected {expected_val}",
                    )
                }
                _ => {
                    panic!(
                        "Expected a number Expr. Got {}\nq: {}",
                        expr.variant_name(),
                        s
                    )
                }
            }
        }

        fn same(s: &str) {
            another(s, s)
        }

        // numberExpr
        same("1");
        same("1.23");
        same("0.23");
        same("1.2e+45");
        same("1.2e-45");
        same("-1");
        same("-1.23");
        same("-0.23");
        same("-1.2e+45");
        same("-1.2e-45");
        same("-1.2e-45");
        another("12.5E34", "1.25e+35");
        same("NaN");
        another("nan", "NaN");
        another("NAN", "NaN");
        another("nAN", "NaN");
        another("Inf", "+Inf");
        another("INF", "+Inf");
        another("inf", "+Inf");
        another("+Inf", "+Inf");
        another("-Inf", "-Inf");
        another("-inF", "-Inf");
        another("0x12", "18");
        another("0x3b", "59");
        another("-0x3b", "-59");
        another("+0X3B", "59");
        another("0b1011", "11");
        another("073", "59");
        another("-0o12", "-10");
        another("-.2", "-0.2");
        another("-.2E-2", "-0.002");
    }

    #[test]
    fn test_parse_metric_expr() {
        same(r#"{foo="bar"}[5m]"#);
        same("foo{}");
        same("foo[5m:]");
        same("bar[:]");
        another("some_metric[: ]", "some_metric[:]");
        another("other_metric[: 3s ]", "other_metric[:3s]");
        same("test[5m:3s]");
        another("errors[ 5m : 3s ]", "errors[5m:3s]");
        same("foo offset 5m");
        same("bar offset -5m");
        same(r#"{__name__="baz"}[5m] offset 10y"#);
        same("latency[5.3m:3.4s] offset 10y");
        same("cache_size[:3.4s] offset 10y");
        same("cache_size[:3.4s] offset -10y");
        same(r#"{Foo="bAR"}"#);
        same(r#"{foo="bar"}"#);
        same(r#"{foo="bar"}[5m]"#);
        same(r#"{foo="bar"}[5m:]"#);
        same(r#"{foo="bar"}[5m:3s]"#);
        same(r#"{foo="bar"} offset 13.4ms"#);
        same(r#"{foo="bar"}[5w4h-3.4m13.4ms]"#);
        same(r#"{foo="bar"} offset 10y"#);
        same(r#"{foo="bar"} offset -10y"#);
        same(r#"{foo="bar"}[5m] offset 10y"#);
        same(r#"{foo="bar"}[5m:3s] offset 10y"#);
        another(
            r#"{foo="bar"}[5m] oFFSEt 10y"#,
            r#"{foo="bar"}[5m] offset 10y"#,
        );
        same("METRIC");
        same("metric");
        same("m_e:tri44:_c123");
        another("-metric", "-metric");
        same("metric offset 10h");
        same("metric[5m]");
        same("metric[5m:3s]");
        same("metric[5m] offset 10h");
        same("metric[5m:3s] offset 10h");
        same("metric[5i:3i] offset 10i");
        same(r#"metric{foo="bar"}"#);
        same(r#"metric{foo="bar"} offset 10h"#);
        same(r#"metric{foo!="bar"}[2d]"#);
        same(r#"metric{foo="bar"}[2d] offset 10h"#);
        same(r#"metric{foo="bar", b="sdfsdf"}[2d:3h] offset 10h"#);
        same(r#"metric{foo="bar", b="sdfsdf"}[2d:3h] offset 10"#);
        same(r#"metric{foo="bar", b="sdfsdf"}[2d:3] offset 10h"#);
        same(r#"metric{foo="bar", b="sdfsdf"}[2:3h] offset 10h"#);
        same(r#"metric{foo="bar", b="sdfsdf"}[2.34:5.6] offset 3600.5"#);
        same(r#"metric{foo="bar", b="sdfsdf"}[234:56] offset -3600"#);
        another(
            r#"  metric  {  foo  = "bar"  }  [  2d ]   offset   10h  "#,
            r#"metric{foo="bar"}[2d] offset 10h"#,
        );
    }

    #[test]
    fn test_1() {
        another(
            r#"{__name__="a",bar="baz" or __name__="a"}"#,
            r#"a{bar="baz"}"#,
        ); 
    }
    
    #[test]
    fn test_parse_metric_expr_with_or() {
        // metricExpr with 'or'
        same(r#"metric{foo="bar" or baz="a"}"#);
        same(r#"metric{foo="bar",x="y" or baz="a",z="q" or a="b"}"#);
        same(r#"{foo="bar",x="y" or baz="a",z="q" or a="b"}"#);
        another(
            r#"metric{foo="bar" OR baz="a"}"#,
            r#"metric{foo="bar" or baz="a"}"#,
        );
        another(r#"{foo="bar" OR baz="a"}"#, r#"{foo="bar" or baz="a"}"#);

        another(
            r#"{__name__="a",bar="baz" or __name__="a"}"#,
            r#"a{bar="baz"}"#,
        );
        another(
            r#"{__name__="a",bar="baz" or __name__="a" or __name__="a"}"#,
            r#"a{bar="baz"}"#,
        );
        another(
            r#"{__name__="a",bar="baz" or __name__="a",bar="abc"}"#,
            r#"{bar="baz" or bar="abc"}"#,
        );
        another(
            r#"{__name__="a" or __name__="a",bar="abc",x!="y"}"#,
            r#"a{bar="abc",x!="y"}"#,
        );
    }

    #[test]
    fn test_parse_at_modifier() {
        // @ modifier
        // See https://prometheus.io/docs/prometheus/latest/querying/basics/#modifier
        same(r#"foo @ 123.45"#);
        // same(r#"foo\@ @ 123.45"#);
        same(r#"{foo=~"bar"} @ end()"#);
        same(r#"foo{bar="baz"} @ start()"#);
        same(r#"foo{bar="baz"}[5m] @ 12345"#);
        same(r#"foo{bar="baz"}[5m:4s] offset 5m @ (end() - 3.5m)"#);
        another(
            r#"foo{bar="baz"}[5m:4s] @ (end() - 3.5m) offset 2.4h"#,
            r#"foo{bar="baz"}[5m:4s] offset 2.4h @ (end() - 3.5m)"#,
        );
        another(
            r#"foo @ start() + (bar offset 3m @ end()) / baz OFFSET -5m"#,
            r#"(foo @ start()) + ((bar offset 3m @ end()) / (baz offset -5m))"#,
        );
        same("sum(foo) @ start() + rate(bar @ (end() - 5m))");
        another("time() @ (start())", "time() @ start()");
        another("time() @ (start()+(1+1))", "time() @ (start() + 2)");
        same("time() @ (end() - 10m)");
        another("a + b offset 5m @ 1235", "a + (b offset 5m @ 1235)");
        another("a + b @ 1235 offset 5m", "a + (b offset 5m @ 1235)");
    }

    #[test]
    fn test_parse_duplicate_filters() {
        // Duplicate filters
        assert_invalid(r#"foo{__name__="bar"}"#);
        assert_invalid(r#"foo{a="b", a="c", __name__="aaa", b="d"}"#);
    }

    #[test]
    fn test_parse_filter_ending_in_comma() {
        // Metric filters ending with comma
        another(r#"m{foo="bar",}"#, r#"m{foo="bar"}"#);
    }

    #[test]
    fn test_valid_regexp_filters() {
        // Valid regexp
        same(r#"foo{bar=~"x"}"#);
        same(r#"foo{bar=~"^x"}"#);
        same(r#"foo{bar=~"^x$"}"#);
        same(r#"foo{bar=~"^(a[bc]|d)$"}"#);
        same(r#"foo{bar!~"x"}"#);
        same(r#"foo{bar!~"^x"}"#);
        same(r#"foo{bar!~"^x$"}"#);
        same(r#"foo{bar!~"^(a[bc]|d)$"}"#);
    }

    #[test]
    fn test_parse_string_concat() {
        // string concat
        another(r#""foo"+'bar'"#, r#""foobar""#);
        // String concat in tag value
        another(r#"m{foo="bar" + "baz"}"#, r#"m{foo="barbaz"}"#);
    }

    #[test]
    fn test_parse_duration_expr() {
        // durationExpr
        same("1h");
        another("-1h", "-1h");
        same("0.34h4m5s");
        another("-0.34h4m5s", "-0.34h4m5s");
        same("sum_over_time(m[1h]) / 1h");
        same("sum_over_time(m[3600]) / 3600");
    }

    #[test]
    fn test_parse_parens_expr() {
        // parensExpr
        another("(-foo + ((bar) / (baz))) + ((23))", "-foo + bar / baz + 23");
        another("(FOO + ((Bar) / (baZ))) + ((23))", "FOO + Bar / baZ + 23");
        another("(foo, bar)", "(foo, bar)");
        another("((foo, bar),(baz))", "((foo, bar), baz)");
        another(
            "(foo, (bar, baz), ((x, y), (z, y), xx))",
            "(foo, (bar, baz), ((x, y), (z, y), xx))",
        );
        another("1+(foo, bar,)", "1 + (foo, bar)");
        another(
            "((avg(bar,baz)), (1+(2)+(3,4)+()))",
            "(avg(bar, baz), (3 + (3, 4)) + ())",
        );
        another("()", "()");
    }
    
    #[test]
    fn test_parse_aggr_func_expr() {
        // aggrFuncExpr
        same("sum(http_server_request) by ()");
        same("sum(http_server_request) by (job)");
        same("sum(http_server_request) without (job, foo)");
        another("sum(x,y,) without (a,b,)", "sum(x, y) without (a, b)");
        another("sum by () (xx)", "sum(xx) by ()");
        another("sum by (s) (xx)[5s]", "sum(xx) by (s)[5s]");
        another("SUM BY (ZZ, aa) (XX)", "sum(XX) by (ZZ, aa)");
        another("sum without (a, b) (xx,2+2)", "sum(xx, 4) without (a, b)");
        another("Sum without (a, B) (XX,2+2)", "sum(XX, 4) without (a, B)");
        same("sum(a) or sum(b)");
        same("sum(a) by () or sum(b) without (x, y)");
        same("sum(a) + sum(b)");
        same("avg(x) limit 10");
        same("avg(x) without (z, b) limit 1");
        another("avg by(x) (z) limit 20", "avg(z) by (x) limit 20");

        // All the above
        another(
            r#"Sum(abs(M) * M{X=""}[5m] Offset 7m - 123, 35) BY (X, y) * scalar("Test")"#,
            r#"sum((abs(M) * (M{X=""}[5m] offset 7m)) - 123, 35) by(X,y) * scalar("Test")"#,
        );
        another(
            r##"# comment
                Sum(abs(M) * M{X=""}[5m] Offset 7m - 123, 35) BY (X, y) # yet another comment
                    * scalar("Test")"##,
            r#"sum((abs(M) * (M{X=""}[5m] offset 7m)) - 123, 35) by(X,y) * scalar("Test")"#,
        );
    }

    #[test]
    fn testing() {
        another("1/0", "+Inf");
    }

    #[test]
    fn test_parse_binary_op_expr() {
        // binaryOpExpr
        // another("nan == nan", "NaN");
        another("nan ==bool nan", "1");
        another("nan !=bool nan", "0");
        another("nan !=bool 2", "1");
        another("2 !=bool nan", "1");
        another("nan >bool nan", "0");
        another("nan <bool nan", "0");
        another("1 ==bool nan", "0");
        another("NaN !=bool 1", "1");
        another("inf >=bool 2", "1");
        another("-1 >bool -inf", "1");
        another("-1 <bool -inf", "0");
        another("nan + 2 *3 * inf", "NaN");
        another("INF - Inf", "NaN");
        another("Inf + inf", "+Inf");
        another("1/0", "+Inf");
        another("0/0", "NaN");
        another("-m", "-m");
        //	same("m + ignoring () n[5m]");
        //	another("M + IGNORING () N[5m]", "M + ignoring () N[5m]");
        same("m + on (foo) n");
        same("m + ignoring (a, b) n");
        another("1 or 2", "1");
        another("1 or NaN", "1");
        another("NaN or 1", "1");
        another("(1 > 0) or 2", "1");
        another("(1 < 0) or 2", "2");
        another("(1 < 0) or (2 < 0)", "NaN");
        another("NaN or NaN", "NaN");
        another("1 and 2", "1");
        another("1 and (1 > 0)", "1");
        another("1 and (1 < 0)","NaN");
        another("1 and NaN", "NaN");
        another("1 unless 2", "NaN");
        another("1 default 2", "1");
        another("1 default NaN", "1");
        another("NaN default 2", "2");
        another("1 > 2", "NaN");
        another("1 > bool 2", "0");
        another("3 >= 2", "3");
        another("3 <= bool 2", "0");
        another("1 + -2 - 3", "-4");
        another("1 / 0 + 2", "+Inf");
        another("2 + -1 / 0", "-Inf");
        another("512.5 - (1 + 3) * (2 ^ 2) ^ 3", "256.5");
        another("1 == bool 1 != bool 24 < bool 4 > bool -1", "1");
        another("1 == bOOl 1 != BOOL 24 < Bool 4 > booL -1", "1");
        another("m1+on(foo)group_left m2", "m1 + on (foo) group_left () m2");
        another("M1+ON(FOO)GROUP_left M2", "M1 + on (FOO) group_left () M2");
        same("m1 + on (foo) group_right () m2");
        same("m1 + on (foo, bar) group_right (x, y) m2");
        another(
            "m1 + on (foo, bar,) group_right (x, y,) m2",
            "m1 + on (foo, bar) group_right (x, y) m2",
        );
        same("m1 == bool on (foo, bar) group_right (x, y) m2");
        another(
            r#"5 - 1 + 3 * 2 ^ 2 ^ 3 - 2  OR Metric {Bar= "Baz", aaa!="bb",cc=~"dd" ,zz !~"ff" } "#,
            r#"770 or Metric{Bar="Baz", aaa!="bb", cc=~"dd", zz!~"ff"}"#,
        );
        same(r#"("foo"[3s] + bar{x="y"})[5m:3s] offset 10s"#);
        same(r#"("foo"[3s] + bar{x="y"})[5i:3i] offset 10i"#);
        same(r#"bar + "foo" offset 3s"#);
        same(r#"bar + "foo" offset 3i"#);
        another("1+2 if 2>3", "NaN");
        another("1+4 if 2<3", "5");
        another("2+6 default 3 if 2>3", "8");
        another("2+6 if 2>3 default NaN", "NaN");
        another("42 if 3>2 if 2+2<5", "42");
        another("42 if 3>2 if 2+2>=5", "NaN");
        another("1+2 ifNot 2>3", "3");
        another("1+4 ifNot 2<3", "NaN");
        another("2+6 default 3 ifNot 2>3", "8");
        another("2+6 ifNot 2>3 default NaN", "8");
        another("42 if 3>2 ifNot 2+2<5", "NaN");
        another("42 if 3>2 ifNot 2+2>=5", "42");
        another(r#""foo" + "bar""#, r#""foobar""#);
        another(r#""foo"=="bar""#, "NaN");
        another(r#""foo"=="foo""#, "1");
        another(r#""foo"!="bar""#, "1");
        another(r#""foo"+"bar"+"baz""#, r#""foobarbaz""#);
        another(r#""a">"b""#, "NaN");
        another(r#""a">bool"b""#, "0");
        another(r#""a"<"b""#, "1");
        another(r#""a">="b""#, "NaN");
        another(r#""a">=bool"b""#, "0");
        another(r#""a"<="b""#, "1");
        another("a / b keep_metric_names", "(a / b) keep_metric_names");
        another("a / 1 keep_metric_names", "a");

        same("(a + b) keep_metric_names");
        another("((a) + (b)) keep_metric_names", "(a + b) keep_metric_names");
        another(
            "a + on(x) group_left(y) b offset 5m @ 1235 keep_metric_names",
            "(a + on(x) group_left(y) (b offset 5m @ 1235)) keep_metric_names",
        );
        another(
            "(a + on(x) group_left(y) b offset 5m keep_metric_names) @ 1235",
            "((a + on(x) group_left(y) (b offset 5m)) keep_metric_names) @ 1235",
        );
        another(
            "(a + on(x) group_left(y) b keep_metric_names) offset 5m @ 1235",
            "((a + on(x) group_left(y) b) keep_metric_names) offset 5m @ 1235",
        );
        another(
            "(a + on (x) group_left (y) b keep_metric_names) @ 1235 offset 5m",
            "((a + on(x) group_left(y) b) keep_metric_names) offset 5m @ 1235",
        );
        another(
            "rate(x) keep_metric_names + (abs(y) keep_metric_names) keep_metric_names",
            "(rate(x) keep_metric_names + (abs(y) keep_metric_names)) keep_metric_names",
        );

        another("(-1) ^ 0.5", "NaN");
        another("-1 ^ 0.5", "-1");
    }

    #[test]
    fn test_or_filters() {
        let cases = vec![
            (r#"foo{label1="1" or label1="2"}"#, {
                let matchers = Matchers::with_or_matchers(vec![
                    vec![Matcher::equal("label1", "1")],
                    vec![Matcher::equal("label1", "2")],
                ]);
                Expr::new_vector_selector(Some(String::from("foo")), matchers)
            }),
            (r#"foo{label1="1" OR label1="2"}"#, {
                let matchers = Matchers::with_or_matchers(vec![
                    vec![Matcher::equal("label1", "1")],
                    vec![Matcher::equal("label1", "2")],
                ]);
                Expr::new_vector_selector(Some(String::from("foo")), matchers)
            }),
            (r#"foo{label1="1" Or label1="2"}"#, {
                let matchers = Matchers::with_or_matchers(vec![
                    vec![Matcher::equal("label1", "1")],
                    vec![Matcher::equal("label1", "2")],
                ]);
                Expr::new_vector_selector(Some(String::from("foo")), matchers)
            }),
            (r#"foo{label1="1" oR label1="2"}"#, {
                let matchers = Matchers::with_or_matchers(vec![
                    vec![Matcher::equal("label1", "1")],
                    vec![Matcher::equal("label1", "2")],
                ]);
                Expr::new_vector_selector(Some(String::from("foo")), matchers)
            }),
            (r#"foo{label1="1" or or="or"}"#, {
                let matchers = Matchers::with_or_matchers(vec![
                    vec![Matcher::equal("label1", "1")],
                    vec![Matcher::equal("or", "or")],
                ]);
                Expr::new_vector_selector(Some(String::from("foo")), matchers)
            }),
            (
                r#"foo{label1="1" or label1="2" or label1="3" or label1="4"}"#,
                {
                    let matchers = Matchers::with_or_matchers(vec![
                        vec![Matcher::equal("label1", "1")],
                        vec![Matcher::equal("label1", "2")],
                        vec![Matcher::equal("label1", "3")],
                        vec![Matcher::equal("label1", "4")],
                    ]);
                    Expr::new_vector_selector(Some(String::from("foo")), matchers)
                },
            ),
            (
                r#"foo{label1="1" or label1="2" or label1="3", label2="4"}"#,
                {
                    let matchers = Matchers::with_or_matchers(vec![
                        vec![Matcher::equal("label1", "1")],
                        vec![Matcher::equal("label1", "2")],
                        vec![Matcher::equal("label1", "3"), Matcher::equal("label2", "4")],
                    ]);
                    Expr::new_vector_selector(Some(String::from("foo")), matchers)
                },
            ),
            (
                r#"foo{label1="1", label2="2" or label1="3" or label1="4"}"#,
                {
                    let matchers = Matchers::with_or_matchers(vec![
                        vec![Matcher::equal("label1", "1"), Matcher::equal("label2", "2")],
                        vec![Matcher::equal("label1", "3")],
                        vec![Matcher::equal("label1", "4")],
                    ]);
                    Expr::new_vector_selector(Some(String::from("foo")), matchers)
                },
            ),
        ];
        assert_cases(Case::new_result_cases(cases));

        let display_cases = [
            r#"a{label1="1"}"#,
            r#"a{label1="1" or label2="2"}"#,
            r#"a{label1="1" or label2="2" or label3="3" or label4="4"}"#,
            r#"a{label1="1", label2="2" or label3="3" or label4="4"}"#,
            r#"a{label1="1", label2="2" or label3="3", label4="4"}"#,
        ];
        display_cases
            .iter()
            .for_each(|expr| assert_eq!(parse(expr).unwrap().to_string(), *expr));

        let or_insensitive_cases = [
            r#"a{label1="1" or label2="2"}"#,
            r#"a{label1="1" OR label2="2"}"#,
            r#"a{label1="1" Or label2="2"}"#,
            r#"a{label1="1" oR label2="2"}"#,
        ];

        or_insensitive_cases.iter().for_each(|expr| {
            assert_eq!(
                parse(expr).unwrap().to_string(),
                r#"a{label1="1" or label2="2"}"#
            )
        });

        let fail_cases = vec![
            (
                r#"foo{or}"#,
                r#"invalid label matcher, expected label matching operator after 'or'"#,
            ),
            // (r#"foo{label1="1" or}"#, INVALID_QUERY_INFO),
            // (r#"foo{or label1="1"}"#, INVALID_QUERY_INFO),
            // (r#"foo{label1="1" or or label2="2"}"#, INVALID_QUERY_INFO),
        ];
        assert_cases(Case::new_fail_cases(fail_cases));
    }

    #[test]
    fn test_parse_func_expr() {
        // funcExpr
        same("now()");
        another("avg(x,)", "avg(x)");
        another("-now()-pi()", "-now() - 3.141592653589793");
        same("now()");
        another("+pi()", "3.141592653589793");
        another("++now()", "now()");
        another("--now()", "now()");
        same("avg(http_server_request)");
        same("floor(http_server_request)[4s:5m] offset 10m");
        same("ceil(http_server_request)[4i:5i] offset 10i");
        same("irate(HttpServerRequest)");
        same("avg(job, foo)");
        same("max(Job, Foo)");
        another(
            r#" sin(bar) + avg (  pi  (  ), sin(1 + (  2.5)) ,M[5m ]  , "ff"  )"#,
            r#"sin(bar) + avg(3.141592653589793, -0.35078322768961984, M[5m], "ff")"#,
        );
        same("rate(foo[5m]) keep_metric_names");
        another("log2(foo) KEEP_metric_names + 1 / increase(bar[5m]) keep_metric_names offset 1h @ 435",
                "log2(foo) keep_metric_names + (1 / (increase(bar[5m]) keep_metric_names offset 1h @ 435))")
    }

    #[test]
    fn func_name_matching_keywords() {
        // funcName matching keywords
        same("rate(rate(m))");
        same("rate(rate(m[5m]))");
        same("rate(rate(m[5m])[1h:])");
        same("rate(rate(m[5m])[1h:3s])");
    }

    #[test]
    fn test_with() {
        another("with (x(foo) = foo+1) x(a)", "a + 1");
        // another(
        //     r#"with (
        // 			x = {foo="bar"},
        // 			q = m{x, y="1"},
        // 			f(x) = with ( z(y) = x + y * q ) z(foo) / count(x) )
        // 			f(a)"#,
        //     r#"(a + (foo * m{foo="bar", y="1"})) / count(a)"#,
        // );
    }

    #[test]
    fn with_expr() {
        // withExpr
        another("with () x", "x");
        another("with (x=1,) x", "1");
        another("with (x = m offset 5h) x + x", "m offset 5h * 2");
        another("with (x = m offset 5i) x + x", "m offset 5i * 2");
        another(r#"with (foo = bar{x="x"}) 1"#, "1");
        another(r#"with (foo = bar{x="x"}) "x""#, r#""x""#);
        another(r#"with (f="x") f"#, r#""x""#);
        another(r#"with (foo = bar{x="x"}) x{x="y"}"#, r#"x{x="y"}"#);
        another(r#"with (foo = bar{x="x"}) 1+1"#, "2");
        another(r#"with (foo = bar{x="x"}) start()"#, "start()");
        another(r#"with (foo = bar{x="x"}) sum(x)"#, "sum(x)");
        another(
            r#"with (foo = bar{x="x"}) baz{foo="bar"}"#,
            r#"baz{foo="bar"}"#,
        );
        another(r#"with (foo = bar) baz"#, "baz");
        // another(
        //    r#"with (foo = bar) foo + foo{a="b"}"#,
        //    r#"bar{a="b"} + bar{a="b"}"#,
        // );
        another(r#"with (foo = bar, bar=baz + now()) test"#, "test");
        // another(
        //     r#"with (ct={job="test"}) a{ct} + ct() + sum({ct="x"})"#,
        //     r#"(a{job="test"} + {job="test"}) + sum({ct="x"})"#,
        // );
        // another(
        //     r#"with (ct={job="test", i="bar"}) ct + {ct, x="d"} + foo{ct, ct} + cos(1)"#,
        //     r#"(({job="test", i="bar"} + {job="test", i="bar", x="d"}) + foo{job="test", i="bar"}) + cos(1)"#,
        // );
        another(
            r#"with (foo = bar) {__name__=~"foo"}"#,
            r#"{__name__=~"foo"}"#,
        );
        // another(r#"with (foo = bar) foo{__name__= "foo"}"#, "bar");
        // another(
        //    r#"with (foo = bar) {__name__="foo", x="y"}"#,
        //    r#"bar{x="y"}"#,
        //);
        // another(
        //     r#"with (foo(bar) = {__name__!="bar"}) foo(x)"#,
        //     r#"{__name__!="bar"}"#,
        // );
        another(r#"with (foo(bar) = bar{__name__="bar"}) foo(x)"#, "x");
        another(
            r#"with (foo\-bar(baz) = baz + baz) foo\-bar((x,y))"#,
            "(x, y) + (x, y)",
        );
        another(
            r#"with (foo\-bar(baz) = baz + baz) foo\-bar(x*y)"#,
            r#"(x * y) + (x * y)"#,
        );
        another(
            r#"with (foo\-bar(baz) = baz + baz) foo\-bar(x\*y)"#,
            r#"x\*y + x\*y"#,
        );
        another(
            r#"with (foo\-bar(b\ az) = b\ az + b\ az) foo\-bar(x\*y)"#,
            r#"x\*y + x\*y"#,
        );
        // override ttf to something new.
        another("with (ttf = a) ttf + b", "a + b");
        // override ttf to ru
        another(
            "with (ttf = ru(m, n)) ttf",
            "(clamp_min(n - clamp_min(m, 0), 0) / clamp_min(n, 0)) * 100",
        );

        // Verify withExpr recursion and forward reference
        another("with (x = x+y, y = x+x) y ^ 2", "((x + y) + (x + y)) ^ 2");
        another("with (f1(x)=f2(x), f2(x)=f1(x)^2) f1(foobar)", "f2(foobar)");
        another(
            "with (f1(x)=f2(x), f2(x)=f1(x)^2) f2(foobar)",
            "f2(foobar) ^ 2",
        );

        // Verify withExpr for aggr fn modifiers
        another("with (f(x) = x, y = sum(m) by (f)) y", "sum(m) by (f)");
        another(
            "with (f(x) = x, y = sum(m) by (f) limit 20) y",
            "sum(m) by (f) limit 20",
        );
        another("with (f(x) = sum(m) by (x)) f(foo)", "sum(m) by (foo)");
        another(
            "with (f(x) = sum(m) by (x) limit 42) f(foo)",
            "sum(m) by (foo) limit 42",
        );
        another(
            "with (f(x) = sum(m) by (x)) f((foo, bar, foo))",
            "sum(m) by (foo, bar)",
        );
        another(
            "with (f(x) = sum(m) without (x,y)) f((a, b))",
            "sum(m) without (a, b, y)",
        );
        another(
            "with (f(x) = sum(m) without (y,x)) f((a, y))",
            "sum(m) without (y, a)",
        );
        another(
            "with (f(x,y) = a + on (x,y) group_left (y,bar) b) f(foo,())",
            "a + on (foo) group_left (bar) b",
        );
        another(
            "with (f(x,y) = a + on (x,y) group_left (y,bar) b) f((foo),())",
            "a + on (foo) group_left (bar) b",
        );
        another(
            "with (f(x,y) = a + on (x,y) group_left (y,bar) b) f((foo,xx),())",
            "a + on (foo, xx) group_left (bar) b",
        );
    }

    #[test]
    fn with_expr_funcs() {
        // Verify withExpr funcs
        another("with (x() = y+1) x", "y + 1");
        another("with (x(foo) = foo+1) x(a)", "a + 1");
        another("with (x(a, b) = a + b) x(foo, bar)", "foo + bar");
        another("with (x(a, b) = a + b) x(foo, x(1, 2))", "foo + 3");
        another(
            "with (x(a) = sum(a) by (b)) x(xx) / x(y)",
            "sum(xx) by (b) / sum(y) by (b)",
        );
        another(
            "with (f(a,f,x)=avg(x,f,a)) f(f(x,y,z),1,2)",
            "avg(2, 1, avg(z, y, x))",
        );
        another(
            r#"with (f(x)=1+ceil(x)) f(foo{bar="baz"})"#,
            r#"1 + ceil(foo{bar="baz"})"#,
        );
        another("with (a=foo, y=bar, f(a)= a+a+y) f(x)", "x * 2 + bar");
        // another(
        //     r#"with (f(a, b) = m{a, b}) f({a="x", b="y"}, {c="d"})"#,
        //     r#"m{a="x", b="y", c="d"}"#,
        // );
        another(
            r#"with (xx={a="x"}, f(a, b) = m{a, b}) f({xx, b="y"}, {c="d"})"#,
            r#"m{a="x", b="y", c="d"}"#,
        );
        another(r#"with (x() = {b="c"}) foo{x}"#, r#"foo{b="c"}"#);
        another(
            r#"with (f(x)=x{foo="bar"} offset 5m) f(m offset 10m)"#,
            r#"(m{foo="bar"} offset 10m) offset 5m"#,
        );
        another(
            r#"with (f(x)=x{foo="bar",bas="a"}[5m]) f(m[10m] offset 3s)"#,
            r#"(m{foo="bar", bas="a"}[10m] offset 3s)[5m]"#,
        );
        another(
            r#"with (f(x)=x{foo="bar"}[5m] offset 10m) f(m{x="y"})"#,
            r#"m{x="y", foo="bar"}[5m] offset 10m"#,
        );
        another(
            r#"with (f(x)=x{foo="bar"}[5m] offset 10m) f({x="y", foo="bar", foo="bar"})"#,
            r#"{x="y", foo="bar"}[5m] offset 10m"#,
        );
        another(
            "with (f(m, x)=m{x}[5m] offset 10m) f(foo, {})",
            "foo[5m] offset 10m",
        );
        another(
            r#"with (f(m, x)=m{x, bar="baz"}[5m] offset 10m) f(foo, {})"#,
            r#"foo{bar="baz"}[5m] offset 10m"#,
        );
        another(
            "with (f(x)=x[5m] offset 3s) f(foo[3m]+bar)",
            "(foo[3m] + bar)[5m] offset 3s",
        );
        another(
            "with (f(x)=x[5m:3s] oFFsEt 1.5m) f(sum(s) by (a,b))",
            "(sum(s) by (a, b))[5m:3s] offset 1.5m",
        );
        another(r#"with (x="a", y=x) y+"bc""#, r#""abc""#);
        another(
            r#"with (x="a", y="b"+x) "we"+y+"z"+now()"#,
            r#""webaz" + now()"#,
        );
        another(
            r#"with (f(x) = m{foo=x+"y", bar="y"+x, baz=x} + x) f("qwe")"#,
            r#"m{foo="qwey", bar="yqwe", baz="qwe"} + "qwe""#,
        );
        another("with (f(a)=a) f", "f");
    }

    #[test]
    fn nested_with_expressions() {
        // Verify nested with expressions
        another("with (f(x) = (with(x=y) x) + x) f(z)", "y + z");
        another("with (x=foo) max(a, with (y=x) y)", "max(a, foo)");
        another(
            "with (x=foo) a * x + (with (y=x) y) / y",
            "(a * foo) + (foo / y)",
        );
        another("with (x = with (y = foo) y + x) x/x", "x/x");
        another(
            r#"with (
					x = {foo="bar"},
					q = m{x, y="1"},
					f(x) = with ( z(y) = x + y * q ) z(foo) / count(x) )
					f(a)"#,
            r#"(a + (foo * m{foo="bar", y="1"})) / count(a)"#,
        );
    }

    #[test]
    fn complex_with_expressions() {
        // complex withExpr
        another(
            r#"WITH (
			treshold = (0.9),
			commonFilters = {job="cacher", instance=~"1.2.3.4"},
			hits = rate(cache{type="hit", commonFilters}[5m]),
			miss = rate(cache{type="miss", commonFilters}[5m]),
			sumByInstance(arg) = sum(arg) by (instance),
			hitRatio = sumByInstance(hits) / sumByInstance(hits + miss)
			)
			hitRatio < treshold"#,
            r#"(sum(rate(cache{type="hit", job="cacher", instance=~"1.2.3.4"}[5m])) by (instance) / sum(rate(cache{type="hit", job="cacher", instance=~"1.2.3.4"}[5m]) + rate(cache{type="miss", job="cacher", instance=~"1.2.3.4"}[5m])) by (instance)) < 0.9"#,
        );
        another(
            r#"WITH (
			x2(x) = x^2,
			f(x, y) = x2(x) + x*y + x2(y)
			)
			f(a, 3)
			"#,
            r#"((a ^ 2) + (a * 3)) + 9"#,
        );
        another(
            r#"WITH (
			x2(x) = x^2,
			f(x, y) = x2(x) + x*y + x2(y)
			)
			f(2, 3)
			"#,
            "19",
        );
        another(
            r#"WITH (
				commonFilters = {instance="foo"},
				timeToFuckup(currv, maxv) = (maxv - currv) / rate(currv)
				)
				timeToFuckup(diskUsage{commonFilters}, maxDiskSize{commonFilters})"#,
            r#"(maxDiskSize{instance="foo"} - diskUsage{instance="foo"}) / rate(diskUsage{instance="foo"})"#,
        );
        another(
            r#"WITH (
				commonFilters = {job="foo", instance="bar"},
				sumRate(m, cf) = sum(rate(m{cf})) by (job, instance),
				hitRate(hits, misses) = sumRate(hits, commonFilters) / (sumRate(hits, commonFilters) + sumRate(misses, commonFilters))
				)
				hitRate(cacheHits, cacheMisses)"#,
            r#"sum(rate(cacheHits{job="foo", instance="bar"})) by (job, instance) / (sum(rate(cacheHits{job="foo", instance="bar"})) by (job, instance) + sum(rate(cacheMisses{job="foo", instance="bar"})) by (job, instance))"#,
        );

        another(
            "with(y=123,z=5) union(with(y=3,f(x)=x*y) f(2) + f(3), with(x=5,y=2) x*y*z)",
            "union(15, 50)",
        );

        another("with(sum=123,now=5) union(with(sum=3,f(x)=x*sum) f(2) + f(3), with(x=5,sum=2) x*sum*now)", "union(15, 50)");
        another("WITH(now = sum(rate(my_metric_total)), before = sum(rate(my_metric_total) offset 1h)) now/before*100", "(sum(rate(my_metric_total)) / sum(rate(my_metric_total) offset 1h)) * 100");
        another("with (sum = x) sum", "x");
        another("with (clamp_min=x) clamp_min", "x");
        another("with (now=now(), sum=sum()) now", "now()");
        another("with (now=now(), sum=sum()) now()", "now()");
        another("with (now(a)=now()+a) now(1)", "now() + 1");
        another("with (rate(a,b)=a+b) rate(1,2)", "3");
        another("with (now=now(), sum=sum()) x", "x");
        another("with (rate(a) = b) c", "c");
        another(
            "rate(x) + with (rate(a,b)=a*b) rate(2,b)",
            "rate(x) + (2 * b)",
        );
        another("with (sum(a,b)=a+b) sum(c,d)", "c + d")
    }

    fn assert_invalid_ex(s: &str, msg_to_check: Option<&str>) {
        match parse(s) {
            Ok(_) => panic!("expecting error expr when parsing {s}"),
            Err(e) => {
                if let Some(msg) = msg_to_check {
                    assert!(e.to_string().contains(msg));
                }
            }
        }
    }

    fn assert_invalid(s: &str) {
        assert_invalid_ex(s, None);
    }

    #[test]
    fn mismatched_operand_error() {
        fn f(s: &str) {
            assert_invalid_ex(s, Some("mismatched operand types in binary expression"));
        }

        f(r#""foo" + PI()"#);
        f(r#""foo" + bar{x="y"}"#);
    }

    #[test]
    fn invalid_string_operator() {
        for op in Operator::iter() {
            if !op.is_valid_string_op() {
                let expr = format!(r#""foo" {op} "bar""#);
                let expected_msg = "not allowed in string string operations";
                assert_invalid_ex(&expr, Some(expected_msg))
            }
        }
    }

    #[test]
    fn invalid_metric_expr() {
        // invalid metricExpr
        assert_invalid("{}");
        assert_invalid("{}[5m]");
        assert_invalid("foo[-55]");
        assert_invalid("m[-5m]");
        assert_invalid("{");
        assert_invalid("foo{");
        assert_invalid("foo{bar");
        assert_invalid("foo{bar=");
        assert_invalid(r#"foo{bar="baz"#);
        assert_invalid(r#"foo{bar="baz",  "#);
        assert_invalid(r#"foo{123="23"}"#);
        assert_invalid("foo{foo}");
        assert_invalid("foo{,}");
        assert_invalid(r#"foo{,foo="bar"}"#);
        assert_invalid("foo{foo=}");
        assert_invalid(r#"foo{foo="ba}"#);
        assert_invalid(r#"foo{"foo"="bar"}"#);
        assert_invalid("foo{$");
        assert_invalid("foo{a $");
        assert_invalid(r#"foo{a="b",$"#);
        assert_invalid(r#"foo{a="b"}$"#);
        assert_invalid("[");
        assert_invalid("[]");
        assert_invalid("f[5m]$");
        assert_invalid("[5m]");
        assert_invalid("[5m] offset 4h");
        assert_invalid("m[5m] offset $");
        assert_invalid("m[5m] offset 5h $");
        assert_invalid("m[]");
        assert_invalid("m[-5m]");
        assert_invalid("m[5m:");
        assert_invalid("m[5m:-");
        assert_invalid("m[5m:-1");
        assert_invalid("m[5m:-1]");
        assert_invalid("m[5m:-1s]");
        assert_invalid("m[-5m:1s]");
        assert_invalid("m[-5m:-1s]");
        assert_invalid("m[:");
        assert_invalid("m[:-");
        assert_invalid("m[:-1]");
        assert_invalid("m[:-1m]");
        assert_invalid("m[-5]");
        assert_invalid("m[[5m]]");
        assert_invalid("m[foo]");
        assert_invalid(r#"m["ff"]"#);
        assert_invalid("m[10m");
        assert_invalid("m[123");
        assert_invalid(r#"m["ff"#);
        assert_invalid("m[(f");
        assert_invalid("fd}");
        assert_invalid("]");
        assert_invalid("m $");
        assert_invalid("m{,}");
        assert_invalid("m{x=y}");
        assert_invalid("m{x=y/5}");
        assert_invalid("m{x=y+5}");
        assert_invalid("m keep_metric_names"); // keep_metric_names cannot be used with metric Expr
    }

    #[test]
    fn invalid_at_modifier() {
        // Invalid @ modifier
        assert_invalid("@");
        assert_invalid("foo @");
        assert_invalid("foo @ ! ");
        assert_invalid("foo @ @");
        assert_invalid("foo @ offset 5m");
        assert_invalid("foo @ [5m]");
        assert_invalid("foo offset @ 5m");
        assert_invalid("foo @ 123 offset 5m @ 456");
        assert_invalid("foo offset 5m @");
    }

    #[test]
    fn invalid_regexp() {
        // Invalid regexp
        assert_invalid(r#"foo{bar=~"x["}"#);
        assert_invalid(r#"foo{bar=~"x("}"#);
        assert_invalid(r#"foo{bar=~"x)"}"#);
        assert_invalid(r#"foo{bar!~"x["}"#);
        assert_invalid(r#"foo{bar!~"x("}"#);
        assert_invalid(r#"foo{bar!~"x)"}"#);
    }

    #[test]
    fn invalid_string_expr() {
        // invalid stringExpr
        assert_invalid("'");
        assert_invalid("\"");
        assert_invalid("`");
        assert_invalid(r#""foo"#);
        assert_invalid(r"'foo");
        assert_invalid("`foo");
        assert_invalid(r#""foo\"bar"#);
        assert_invalid(r#"'foo\'bar"#);
        assert_invalid("`foo\\`bar");
        assert_invalid(r#"" $"#);
        assert_invalid(r#"foo" +"#);
        assert_invalid(r#"n{"foo" + m"#);
        assert_invalid(r#""foo" keep_metric_names"#);
        assert_invalid(r#"keep_metric_names "foo""#);
    }

    #[test]
    fn invalid_number_expr() {
        // invalid numberExpr
        assert_invalid("12.");
        assert_invalid("1.2e");
        assert_invalid("23e-");
        assert_invalid("23E+");
        assert_invalid(".");
        assert_invalid("-12.");
        assert_invalid("-1.2e");
        assert_invalid("-23e-");
        assert_invalid("-23E+");
        assert_invalid("-.");
        assert_invalid("-1$$");
        assert_invalid("-$$");
        assert_invalid("+$$");
        assert_invalid("23 $$");
        assert_invalid("1 keep_metric_names");
        assert_invalid("keep_metric_names 1");
    }

    #[test]
    fn invalid_binary_op_expr() {
        // invalid binaryOpExpr
        assert_invalid("+");
        assert_invalid("1 +");
        assert_invalid("1 + 2.");
        assert_invalid("3 unless");
        assert_invalid("23 + on (foo)");
        assert_invalid("m + on (,) m");
        assert_invalid("3 * ignoring");
        assert_invalid("m * on (");
        assert_invalid("m * on (foo");
        assert_invalid("m * on (foo,");
        assert_invalid("m * on (foo,)");
        assert_invalid("m * on (,foo)");
        assert_invalid("m * on (,)");
        assert_invalid("m == bool (bar) baz");
        assert_invalid("m == bool () baz");
        assert_invalid("m * by (baz) n");
        assert_invalid("m + bool group_left m2");
        assert_invalid("m + on () group_left (");
        assert_invalid("m + on () group_left (,");
        assert_invalid("m + on () group_left (,foo");
        assert_invalid("m + on () group_left (foo,)");
        assert_invalid("m + on () group_left (,foo)");
        assert_invalid("m + on () group_left (foo)");
        assert_invalid("m + on () group_right (foo) (m");
        assert_invalid("m or ignoring () group_left () n");
        assert_invalid("1 + bool 2");
        assert_invalid("m % bool n");
        assert_invalid("m * bool baz");
        assert_invalid("M * BOoL BaZ");
        assert_invalid("foo unless ignoring (bar) group_left xxx");
        assert_invalid("foo or bool bar");
        assert_invalid("foo == bool $$");
        assert_invalid(r#""foo" + bar"#);
    }

    #[test]
    fn invalid_empty_string() {
        // an empty string
        assert_invalid("");
        assert_invalid(r#"  \t\b\r\n  "#);
    }

    #[test]
    fn invalid_parens_expr() {
        // invalid parensExpr
        assert_invalid("(");
        assert_invalid("($");
        assert_invalid("(+");
        assert_invalid("(1");
        assert_invalid("(m+");
        assert_invalid("1)");
        assert_invalid("(,)");
        assert_invalid("(1)$");
        assert_invalid("(foo) keep_metric_names");
    }

    #[test]
    fn invalid_func_expr() {
        // invalid funcExpr
        assert_invalid("f $");
        assert_invalid("f($)");
        assert_invalid("f[");
        assert_invalid("f()$");
        assert_invalid("f(");
        assert_invalid("f(foo");
        assert_invalid("f(f,");
        assert_invalid("f(,");
        assert_invalid("f(,)");
        assert_invalid("f(,foo)");
        assert_invalid("f(,foo");
        assert_invalid("f(foo,$");
        assert_invalid("f() by (a)");
        assert_invalid("f without (x) (y)");
        assert_invalid("f() foo (a)");
        assert_invalid("f bar (x) (b)");
        assert_invalid("f bar (x)");
        assert_invalid("keep_metric_names f()");
        assert_invalid("f() abc");
    }

    #[test]
    fn invalid_aggr_expr() {
        // invalid aggrFuncExpr
        assert_invalid("sum(");
        assert_invalid("sum $");
        assert_invalid("sum [");
        assert_invalid("sum($)");
        assert_invalid("sum()$");
        assert_invalid("sum(foo) ba");
        assert_invalid("sum(foo) ba()");
        assert_invalid("sum(foo) by");
        assert_invalid("sum(foo) without x");
        assert_invalid("sum(foo) aaa");
        assert_invalid("sum(foo) aaa x");
        assert_invalid("sum() by $");
        assert_invalid("sum() by (");
        assert_invalid("sum() by ($");
        assert_invalid("sum() by (a");
        assert_invalid("sum() by (a $");
        assert_invalid("sum() by (a ]");
        assert_invalid("sum() by (a)$");
        assert_invalid("sum() by (,");
        assert_invalid("sum() by (a,$");
        assert_invalid("sum() by (,)");
        assert_invalid("sum() by (,a");
        assert_invalid("sum() by (,a)");
        assert_invalid("sum() on (b)");
        assert_invalid("sum() bool");
        assert_invalid("sum() group_left");
        assert_invalid("sum() group_right(x)");
        assert_invalid("sum ba");
        assert_invalid("sum ba ()");
        assert_invalid("sum by (");
        assert_invalid("sum by (a");
        assert_invalid("sum by (,");
        assert_invalid("sum by (,)");
        assert_invalid("sum by (,a");
        assert_invalid("sum by (,a)");
        assert_invalid("sum by (a)");
        assert_invalid("sum by (a) (");
        assert_invalid("sum by (a) [");
        assert_invalid("sum by (a) {");
        assert_invalid("sum by (a) (b");
        assert_invalid("sum by (a) (b,");
        assert_invalid("sum by (a) (,)");
        assert_invalid("avg by (a) (,b)");
        assert_invalid("sum by (x) (y) by (z)");
        assert_invalid("sum(m) by (1)");
        assert_invalid("sum(m) keep_metric_names"); // keep_metric_names cannot be used for aggregate functions
    }

    #[test]
    fn invalid_with_expr() {
        assert_invalid("with $");
        assert_invalid("with a");
        assert_invalid("with a=b c");
        assert_invalid("with (");
        assert_invalid("with (x=b)$");
        assert_invalid("with ($");
        assert_invalid("with (foo");
        assert_invalid("with (foo $");
        assert_invalid("with (x y");
        assert_invalid("with (x =");
        assert_invalid("with (x = $");
        assert_invalid("with (x= y");
        assert_invalid("with (x= y $");
        assert_invalid("with (x= y)");
        assert_invalid("with (x=(");
        assert_invalid("with (x=[)");
        assert_invalid("with (x=() x)");
        assert_invalid("with(x)");
        assert_invalid("with ($$)");
        assert_invalid("with (x $$");
        assert_invalid("with (x = $$)");
        assert_invalid("with (x = foo) bar{x}");
        assert_invalid(r#"with (x = {foo="bar"}[5m]) bar{x}"#);
        assert_invalid(r#"with (x = {foo="bar"} offset 5m) bar{x}"#);
        assert_invalid("with (x = a, x = b) c");
        assert_invalid("with (x(a, a) = b) c");
        assert_invalid(r#"with (x=m{f="x"}) foo{x}"#);
        assert_invalid("with (f()");
        assert_invalid("with (a=b c=d) e");
        assert_invalid("with (f(x)=x^2) m{x}");
        assert_invalid("with (f(x)=ff()) m{x}");
        assert_invalid("with (f(x");
        assert_invalid("with (x=m) a{x} + b");
        assert_invalid("with (x=m) b + a{x}");
        assert_invalid("with (x=m) f(b, a{x})");
        assert_invalid("with (x=m) sum(a{x})");
        assert_invalid("with (x=m) (a{x})");
        assert_invalid("with (f(a)=a) f(1, 2)");
     //   assert_invalid(r#"with (f(x)=x{foo="bar"}) f(1)"#);
     //   assert_invalid(r#"with (f(x)=x{foo="bar"}) f(m + n)"#);
        assert_invalid("with (f = with");
        assert_invalid("with (,)");
        assert_invalid("with (1) 2");
        assert_invalid("with (f(1)=2) 3");
        assert_invalid("with (f(,)=x) x");
        assert_invalid(r#"with (x(a) = {b="c"}) foo{x}"#);
        assert_invalid(r#"with (f(x) = m{foo=xx}) f("qwe")"#);
        assert_invalid("a + with(f(x)=x) assert_invalid(1,2)");
        assert_invalid(r#"with (f(x) = sum(m) by (x)) f({foo="bar"})"#);
        assert_invalid(r#"with (f(x) = sum(m) by (x)) f((xx(), {foo="bar"}))"#);
        assert_invalid("with (f(x) = m + on (x) n) f(xx())");
        assert_invalid("with (f(x) = m + on (a) group_right (x) n) f(xx())");
        assert_invalid("with (f(x) = m keep_metric_names)");
        assert_invalid("with (now)");
        assert_invalid("with (sum)");
        assert_invalid("with (now=now()) now(1)");
        assert_invalid("with (f())");
        assert_invalid("with (sum(a,b)=a+b) sum(x)");
        assert_invalid("with (rate()=foobar) rate(x)")
    }
}
