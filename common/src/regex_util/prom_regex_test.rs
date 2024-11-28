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

    #[test]
    fn test_prom_regex() {
        fn f(expr: &str, s: &str, result_expected: bool) {
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

        // f("foo|bar", "foobar", false);

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
}