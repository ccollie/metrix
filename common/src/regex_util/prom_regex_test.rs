#[cfg(test)]
mod test {
    use regex::Regex;
    use crate::regex_util::PromRegex;

    #[test]
    fn test_prom_regex_parse_failure() {
        fn f(expr: &str) {
            let _ = PromRegex::new(expr).expect("unexpected success for expr={expr}");
        }

        f("fo[bar");
        f("foo(bar")
    }

    fn test_regex(expr: &str, s: &str, result_expected: bool) {
        let pr = PromRegex::new(expr).expect("unexpected failure");
        let result = pr.is_match(s);
        assert_eq!(
            result, result_expected,
            "unexpected result when matching \"{expr}\" against \"{s}\"; got {result}; want {result_expected}"
        );

        // Make sure the result is the same for regular regexp
        let expr_anchored = "^(?:".to_owned() + expr + ")$";
        let re = Regex::new(&*expr_anchored).expect("unexpected failure");
        let result = re.is_match(s);
        assert_eq!(
            result, result_expected,
            "unexpected result when matching {expr_anchored} against {s}; got {result}; want {result_expected}"
        );
    }

    #[test]
    fn test_prom_regex() {
        fn f(expr: &str, s: &str, result_expected: bool) {
            test_regex(expr, s, result_expected);
        }


        f("^foo|b(ar)$", "foo", true);
        f("", "foo", false);
        f("", "", true);
        f("", "foo", false);
        f("foo", "", false);
        f(".*", "", true);
        f(".*", "foo", true);
        f(".+", "", false);
        f(".+", "foo", true);
        f("foo.*", "bar", false);
        f("foo.*", "foo", true);
        f("foo.*", "foobar", true);
        f("foo.+", "bar", false);
        f("foo.+", "foo", false);
        f("foo.+", "foobar", true);
        f("foo|bar", "", false);
        f("foo|bar", "a", false);
        f("foo|bar", "foo", true);
        f("foo|bar", "bar", true);
   //     f("foo|bar", "foobar", false);
        f("foo(bar|baz)", "a", false);
        f("foo(bar|baz)", "foobar", true);
        f("foo(bar|baz)", "foobaz", true);
        f("foo(bar|baz)", "foobaza", false);
        f("foo(bar|baz)", "foobal", false);
        f("^foo|b(ar)$", "foo", true);
        f("^foo|b(ar)$", "bar", true);
        f("^foo|b(ar)$", "ar", false);
        f(".*foo.*", "foo", true);
        f(".*foo.*", "afoobar", true);
        f(".*foo.*", "abc", false);
        f("foo.*bar.*", "foobar", true);
        f("foo.*bar.*", "foo_bar_", true);
        f("foo.*bar.*", "foobaz", false);
        f(".+foo.+", "foo", false);
        f(".+foo.+", "afoobar", true);
        f(".+foo.+", "afoo", false);
        f(".+foo.+", "abc", false);
        f("foo.+bar.+", "foobar", false);
        f("foo.+bar.+", "foo_bar_", true);
        f("foo.+bar.+", "foobaz", false);
        f(".+foo.*", "foo", false);
        f(".+foo.*", "afoo", true);
        f(".+foo.*", "afoobar", true);
        f(".*(a|b).*", "a", true);
        f(".*(a|b).*", "ax", true);
        f(".*(a|b).*", "xa", true);
        f(".*(a|b).*", "xay", true);
        f(".*(a|b).*", "xzy", false);
        f("^(?:true)$", "true", true);
        f("^(?:true)$", "false", false)
    }

    #[test]
    fn test_prom_regex_1() {
        fn f(expr: &str, s: &str, result_expected: bool) {
            test_regex(expr, s, result_expected);
        }

        f(".+foo|bar|baz.+", "afooa", false);

        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/5297
        f(".+;|;.+", ";", false);
        f(".+;|;.+", "foo", false);
//        f(".+;|;.+", "foo;bar", false);
        f(".+;|;.+", "foo;", true);
        f(".+;|;.+", ";foo", true);
        f(".+foo|bar|baz.+", "foo", false);
        f(".+foo|bar|baz.+", "afoo", true);
        f(".+foo|bar|baz.+", "fooa", false);
   //     f(".+foo|bar|baz.+", "afooa", false);
        f(".+foo|bar|baz.+", "bar", true);
    //    f(".+foo|bar|baz.+", "abar", false);
    //    f(".+foo|bar|baz.+", "abara", false);
    //    f(".+foo|bar|baz.+", "bara", false);
        f(".+foo|bar|baz.+", "baz", false);
        f(".+foo|bar|baz.+", "baza", true);
        f(".+foo|bar|baz.+", "abaz", false);
    //    f(".+foo|bar|baz.+", "abaza", false);
    //    f(".+foo|bar|baz.+", "afoo|bar|baza", false);
        f(".+(foo|bar|baz).+", "abara", true);
        f(".+(foo|bar|baz).+", "afooa", true);
        f(".+(foo|bar|baz).+", "abaza", true);

        f(".*;|;.*", ";", true);
        f(".*;|;.*", "foo", false);
   //     f(".*;|;.*", "foo;bar", false);
        f(".*;|;.*", "foo;", true);
        f(".*;|;.*", ";foo", true);

        f(".*foo(bar|baz)", "fooxfoobaz", true);
        f(".*foo(bar|baz)", "fooxfooban", false);
        f(".*foo(bar|baz)", "fooxfooban foobar", true)
    }
}