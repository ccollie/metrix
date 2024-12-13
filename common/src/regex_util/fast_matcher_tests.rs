use rand::Rng;
use regex::Regex;

// Define a trait for StringMatcher
trait StringMatcher {
    fn matches(&self, s: &str) -> bool;
}


// Helper function to generate random values
fn generate_random_values() -> Vec<String> {
    let mut rng = rand::thread_rng();
    let mut texts = Vec::new();
    for _ in 0..10 {
        texts.push(rng.gen_range(0..10).to_string());
    }
    texts
}

// Test function for FastRegexMatcher
#[cfg(test)]
mod tests {
    use super::*;
    use crate::regex_util::fast_matcher::{find_set_matches, optimize_concat_regex, string_matcher_from_regex, FastRegexMatcher};
    use crate::regex_util::hir_utils::build_hir;
    use crate::regex_util::{ContainsMultiStringMatcher, StringMatchHandler};

    #[test]
    fn test_fast_regex_matcher_match_string() {
        let test_values = generate_random_values();
        let regexes = vec!["foo", "bar", "baz"];

        for r in regexes {
            let matcher = FastRegexMatcher::new(r).unwrap();
            for v in &test_values {
                let re = Regex::new(&format!("^(?s:{})$", r)).unwrap();
                assert_eq!(re.is_match(v), matcher.matches(v));
            }
        }
    }

    #[test]
    fn test_optimize_concat_regex() {
        let cases = vec![
            ("foo(hello|bar)", "foo", "", vec![]),
            ("foo(hello|bar)world", "foo", "world", vec![]),
            ("foo.*", "foo", "", vec![]),
            ("foo.*hello.*bar", "foo", "bar", vec!["hello"]),
            (".*foo", "", "foo", vec![]),
            ("^.*foo$", "", "foo", vec![]),
            (".*foo.*", "", "", vec!["foo"]),
            (".*foo.*bar.*", "", "", vec!["foo", "bar"]),
            (".*(foo|bar).*", "", "", vec![]),
            (".*[abc].*", "", "", vec![]),
            (".*((?i)abc).*", "", "", vec![]),
            (".*(?i:abc).*", "", "", vec![]),
            ("(?i:abc).*", "", "", vec![]),
            (".*(?i:abc)", "", "", vec![]),
            (".*(?i:abc)def.*", "", "", vec!["def"]),
            ("(?i).*(?-i:abc)def", "", "", vec!["abc"]),
            (".*(?msU:abc).*", "", "", vec!["abc"]),
            ("[aA]bc.*", "", "", vec!["bc"]),
            ("^5..$", "5", "", vec![]),
            ("^release.*", "release", "", vec![]),
            ("^env-[0-9]+laio[1]?[^0-9].*", "env-", "", vec!["laio"]),
        ];

        for (regex, prefix, suffix, contains) in cases {
            let parsed = regex::Regex::new(&format!("^(?s:{})$", regex)).unwrap();
            let (actual_prefix, actual_suffix, actual_contains) = optimize_concat_regex(&parsed);
            assert_eq!(prefix, actual_prefix);
            assert_eq!(suffix, actual_suffix);
            assert_eq!(contains, actual_contains);
        }
    }

    #[test]
    fn test_fast_regex_matcher_match_string_inner(r: &str, v: &str, test_name: &str) {
        let m = FastRegexMatcher::new(r).unwrap();
        let re = Regex::new(&format!("^(?s:{})$", r)).unwrap();
        assert_eq!(re.is_match(v), m.matches(v), "{}", test_name);
    }


    #[test]
    fn test_find_set_matches() {
        let cases = vec![
            ("foo", vec!["foo"], true),
            ("^foo", vec!["foo"], true),
            ("^foo$", vec!["foo"], true),
            ("foo|bar|zz", vec!["foo", "bar", "zz"], true),
            ("foo|bar|baz", vec!["foo", "bar", "baz"], true),
            ("foo|bar|baz|(zz)", vec!["foo", "bar", "baz", "zz"], true),
            ("bar|b|buzz", vec!["bar", "b", "buzz"], true),
            ("^((bar|b|buzz))$", vec!["bar", "b", "buzz"], true),
            ("^(bar|b|buzz)$", vec!["bar", "b", "buzz"], true),
            ("^(?:prod|production)$", vec!["prod", "production"], true),
            ("fo\\.o|bar\\?|\\^baz", vec!["fo.o", "bar?", "^baz"], true),
            ("[abc]d", vec!["ad", "bd", "cd"], true),
            ("ABC|ABD|AEF|BCX|BCY", vec!["ABC", "ABD", "AEF", "BCX", "BCY"], true),
            ("api_(v1|prom)_push", vec!["api_v1_push", "api_prom_push"], true),
            ("(api|rpc)_(v1|prom)_push", vec!["api_v1_push", "api_prom_push", "rpc_v1_push", "rpc_prom_push"], true),
            ("(api|rpc)_(v1|prom)_(push|query)", vec!["api_v1_push", "api_v1_query", "api_prom_push", "api_prom_query", "rpc_v1_push", "rpc_v1_query", "rpc_prom_push", "rpc_prom_query"], true),
            ("[-1-2][a-c]", vec!["-a", "-b", "-c", "1a", "1b", "1c", "2a", "2b", "2c"], true),
            ("[1^3]", vec!["1", "3", "^"], true),
            ("(?i)foo", vec!["FOO"], false),
            ("(?i)foo|bar|baz", vec!["FOO", "BAR", "BAZ", "BAr", "BAz"], false),
            ("rpc|(?i)(?-i)api", vec!["rpc", "api"], true),
            ("(?i)((?-i)api|(?-i)rpc)", vec!["api", "rpc"], true),
        ];

        for (pattern, exp_matches, exp_case_sensitive) in cases {
            let mut parsed = build_hir(&format!("^(?s:{})$", pattern)).unwrap();
            let (matches) = find_set_matches(&mut parsed).unwrap();
            assert_eq!(exp_matches, matches);

            if exp_case_sensitive {
                let r = FastRegexMatcher::new(pattern).unwrap();
                assert_eq!(exp_matches, r.set_matches());
            }
        }
    }


    #[test]
    fn test_fast_regex_matcher_set_matches_should_return_a_copy() {
        let m = FastRegexMatcher::new("a|b").unwrap();
        assert_eq!(vec!["a", "b"], m.set_matches());

        // Manipulate the returned slice.
        let mut matches = m.set_matches();
        matches[0] = "xxx".to_string();
        matches[1] = "yyy".to_string();

        // Ensure that if we call set_matches() again we get the original one.
        assert_eq!(vec!["a", "b"], m.set_matches());
    }

    #[test]
    fn test_fast_regex_matcher_set_matches_should_return_acopy() {
        let mut m = FastRegexMatcher::new("a|b").unwrap();
        let expected = vec!["a", "b"];
        assert_eq!(expected, m.set_matches());

        // Manipulate the returned slice.
        let mut matches = m.set_matches();
        matches[0] = "xxx".to_string();
        matches[1] = "yyy".to_string();

        // Ensure that if we call set_matches() again we get the original one.
        assert_eq!(expected, m.set_matches());
    }

    #[test]
    fn test_string_matcher_from_regexp() {
        fn literal(s: &str) -> StringMatchHandler {
            StringMatchHandler::Literal(s.to_owned())
        }

        fn boxed_literal(s: &str) -> Box<StringMatchHandler> {
            boxed_literal(s)
        }

        fn true_matcher() -> StringMatchHandler {
            StringMatchHandler::MatchAll
        }
        
        fn some_true_matcher() -> Option<StringMatchHandler> {
            Some(true_matcher())
        }

        fn suffix(s: &str, left: Option<StringMatchHandler>) -> StringMatchHandler {
            StringMatchHandler::suffix(left, s)
        }

        fn prefix(s: &str, right: Option<StringMatchHandler>) -> StringMatchHandler {
            StringMatchHandler::prefix(s, right)
        }

        fn contains_multi(substrings: &[&str], left: Option<StringMatchHandler>, right: Option<StringMatchHandler>) -> StringMatchHandler {
            let matches = substrings.iter().map(|s| s.to_string()).collect();
            let matcher = ContainsMultiStringMatcher::new(matches, left, right);
            StringMatchHandler::ContainsMulti(matcher)
        }
        fn not_empty(b: bool) -> StringMatchHandler {
            StringMatchHandler::not_empty(b)
        }

        fn empty() -> StringMatchHandler {
            StringMatchHandler::Empty
        }

        fn or_matcher(matchers: &[StringMatchHandler]) -> StringMatchHandler {
            let handlers = matchers.iter().map(|m| Box::new(m.clone())).collect();
            StringMatchHandler::Or(handlers)
        }

        fn zero_or_one_chars(b: bool) -> StringMatchHandler {
            StringMatchHandler::zero_or_one_chars(b)
        }

        let cases = vec![
            (".*", some_true_matcher()),
            (".*?", some_true_matcher()),
            ("(?s:.*)", some_true_matcher()),
            ("(.*)", some_true_matcher()),
            ("^.*$", some_true_matcher()),
            (".+", Some(not_empty(true))),
            ("(?s:.+)", Some(not_empty(true))),
            ("^.+$", Some(not_empty(true))),
            ("(.+)", Some(not_empty(true))),
            ("", Some(StringMatchHandler::Empty)),
            ("^$", Some(StringMatchHandler::Empty)),
            ("^foo$", Some(literal("foo"))),
             ("^(?i:foo)$", Some(literal("FOO"))), // todo: wrong - just to compile
             ("^((?i:foo)|(bar))$", Some(
                 or_matcher(&[literal("FOO"), literal("bar")])
             )),
            ("(?i:((foo|bar)))", Some(
               or_matcher(&[literal("FOO"), literal("BAR")])
            )),
            ("(?i:((foo1|foo2|bar)))", Some(
               or_matcher(&[
                   or_matcher(&[literal("FOO1"), literal("FOO2"), literal("BAR")])
               ])
            )),
              ("^((?i:foo|oo)|(bar))$", Some(
                   or_matcher(&[literal("FOO"), literal("OO"), literal("bar")])
              )),
              ("(?i:(foo1|foo2|bar))", Some(
                   or_matcher(
                       &[
                           or_matcher(&[literal("FOO1"), literal("FOO2"), literal("BAR")])
                       ]
                   )
              )),
              (".*foo.*", Some(
                contains_multi(&["foo"], Some(true_matcher()), Some(true_matcher()))
              )),
               ("(.*)foo.*", Some(
                   contains_multi(&["foo"], Some(true_matcher()), Some(true_matcher()))
               )),
               ("(.*)foo(.*)", Some(
                   contains_multi(&["foo"], Some(true_matcher()), Some(true_matcher()))
               )),
               ("(.+)foo(.*)", Some(
                   contains_multi(&["foo"], Some(not_empty(true)), Some(true_matcher()))
               )),
               ("^.+foo.+", Some(
                   contains_multi(&["foo"], Some(not_empty(true)), Some(not_empty(true)))
               )),
               ("^(.*)(foo)(.*)$", Some(
                   contains_multi(&["foo"], Some(true_matcher()), Some(true_matcher()))
               )),
               ("^(.*)(foo|foobar)(.*)$", Some(
                   contains_multi(&["foo", "foobar"], Some(true_matcher()), Some(true_matcher()))
               )),
              //("^(.*)(bar|b|buzz)(.*)$",  ))),
              ("^(.*)(foo|foobar)(.+)$", Some(contains_multi(&["foo", "foobar"], Some(true_matcher()), Some(not_empty(true))))),
              ("^(.*)(bar|b|buzz)(.+)$", Some(contains_multi(&["bar", "b", "buzz"], Some(true_matcher()), Some(not_empty(true))))),
              ("10\\.0\\.(1|2)\\.+", None),
              ("10\\.0\\.(1|2).+", Some(
                  contains_multi(&["10.0.1", "10.0.2"], None, Some(not_empty(true)))
              )),
              ("^.+foo", Some(
                  suffix("foo", Some(not_empty(true)))
              )),
              ("foo-.*$", Some(prefix("foo-", None))),
              ("(prometheus|api_prom)_api_v1_.+", Some(
                  contains_multi(&["prometheus_api_v1_", "api_prom_api_v1_"], None, Some(not_empty(true)))
              )),
              ("^((.*)(bar|b|buzz)(.+)|foo)$", Some(
                  or_matcher(&[
                      contains_multi(&["bar", "b", "buzz"], Some(true_matcher()), Some(not_empty(true))),
                      literal("foo")
                  ])
              )),
              ("((fo(bar))|.+foo)", Some(
                  or_matcher(&[or_matcher(&[literal("fobar"), suffix("foo", Some(not_empty(true)))])])
              )),
              ("(.+)/(gateway|cortex-gw|cortex-gw-internal)", Some(
                  contains_multi(&["/gateway", "/cortex-gw", "/cortex-gw-internal"], Some(not_empty(true)), Some(true_matcher()))
              )),
               // we don't support case insensitive matching for contains.
               // This is because there's no strings.IndexOfFold function.
               // We can revisit later if this is really popular by using strings.ToUpper.
               ("^(.*)((?i)foo|foobar)(.*)$", None),
               ("(api|rpc)_(v1|prom)_((?i)push|query)", None),
               ("[a-z][a-z]", None),
               ("[1^3]", None),
               (".*foo.*bar.*", None),
               ("\\d*", None),
               (".", None),
               ("/|/bar.*", Some(
                   prefix("/",
                          Some(or_matcher(&[empty(), prefix("bar", Some(true_matcher()))]))
                   )
               )),
               // This one is not supported because  `stringMatcherFromRegexp` is not reentrant for syntax.OpConcat.
               // It would make the code too complex to handle it.
               ("(.+)/(foo.*|bar$)", None),
               // Case sensitive alternate with same literal prefix and .* suffix.
               ("(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*)", Some(
                   prefix("xyz-016a-ixb-",
                          Some(or_matcher(&[prefix("dp", Some(true_matcher())), prefix("op", Some(true_matcher()))]))
                   )
               )),
                // Case insensitive alternate with same literal prefix and .* suffix.
                ("(?i:(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*))",
                 Some(
                     prefix("XYZ-016A-IXB-",
                            Some(
                                or_matcher(&[prefix("DP", Some(true_matcher())), prefix("OP", Some(true_matcher()))])
                            )
                     )
                 )),
                ("(?i)(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*)", Some(
                     prefix("XYZ-016A-IXB-",
                            Some(or_matcher(&[
                                prefix("DP", Some(true_matcher())),
                                prefix("OP", Some(true_matcher()))
                            ]))
                     )
                )),
                // Concatenated variable length selectors are not supported.
                ("foo.*.*", None),
                ("foo.+.+", None),
                (".*.*foo", None),
                (".+.+foo", None),
                ("aaa.?.?", None),
                ("aaa.?.*", None),
                // Regexps with ".?".
                ("ext.?|xfs", Some(
                    or_matcher(&[
                        prefix("ext", Some(zero_or_one_chars(true))),
                        literal("xfs"),
                    ])
                )
                ),
                ("(?s)(ext.?|xfs)", Some(
                    or_matcher(&[prefix("ext", Some(zero_or_one_chars(true))), literal("xfs")])
                )),
                ("foo.?", Some(prefix("foo", Some(zero_or_one_chars(true))))),
                ("f.?o", None)
        ];
        for c in cases {
            let mut parsed = build_hir(c.0).unwrap();
            let matches = string_matcher_from_regex(&mut parsed);
            assert_eq!(c.1, matches);
        }
    }

}
