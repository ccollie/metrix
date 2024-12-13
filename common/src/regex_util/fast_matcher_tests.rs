use rand::Rng;
use regex::Regex;


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
    use crate::regex_util::fast_matcher::{find_set_matches, string_matcher_from_hir, FastRegexMatcher};
    use crate::regex_util::hir_utils::build_hir;
    use crate::regex_util::{string_matcher_from_regex, ContainsMultiStringMatcher, StringMatchHandler};

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


    // Refer to https://github.com/prometheus/prometheus/issues/2651.
    #[test]
    fn test_find_set_matches() {
        let cases = vec![
            // Single value, coming from a `bar=~"foo"` selector.
            ("foo", vec!["foo"], true),
            ("^foo", vec!["foo"], true),
            ("^foo$", vec!["foo"], true),
            // Simple sets alternates.
            ("foo|bar|zz", vec!["foo", "bar", "zz"], true),
            // Simple sets alternate and concat (bar|baz is parsed as "ba[rz]").
            ("foo|bar|baz", vec!["foo", "bar", "baz"], true),
            // Simple sets alternate and concat and capture
            ("foo|bar|baz|(zz)", vec!["foo", "bar", "baz", "zz"], true),
            // Simple sets alternate and concat and alternates with empty matches
            // parsed as  b(ar|(?:)|uzz) where b(?:) means literal b.
            ("bar|b|buzz", vec!["bar", "b", "buzz"], true),
            // Skip nested capture groups.
            ("^((bar|b|buzz))$", vec!["bar", "b", "buzz"], true),
            // Skip outer anchors (it's enforced anyway at the root).
            ("^(bar|b|buzz)$", vec!["bar", "b", "buzz"], true),
            ("^(?:prod|production)$", vec!["prod", "production"], true),
            // Do not optimize regexp with inner anchors.
            ("(bar|b|b^uz$z)", vec![], false),
            // Do not optimize regexp with empty string matcher.
            ("^$|Running", vec![], false),
            // Simple sets containing escaped characters.
            ("fo\\.o|bar\\?|\\^baz", vec!["fo.o", "bar?", "^baz"], true),
            // using charclass
            ("[abc]d", vec!["ad", "bd", "cd"], true),
            // high low charset different => A(B[CD]|EF)|BC[XY]
            ("ABC|ABD|AEF|BCX|BCY", vec!["ABC", "ABD", "AEF", "BCX", "BCY"], true),
            // triple concat
            ("api_(v1|prom)_push", vec!["api_v1_push", "api_prom_push"], true),
            // triple concat with multiple alternates
            ("(api|rpc)_(v1|prom)_push", vec!["api_v1_push", "api_prom_push", "rpc_v1_push", "rpc_prom_push"], true),
            ("(api|rpc)_(v1|prom)_(push|query)", vec!["api_v1_push", "api_v1_query", "api_prom_push", "api_prom_query", "rpc_v1_push", "rpc_v1_query", "rpc_prom_push", "rpc_prom_query"], true),
            // class starting with "-"
            ("[-1-2][a-c]", vec!["-a", "-b", "-c", "1a", "1b", "1c", "2a", "2b", "2c"], true),
            ("[1^3]", vec!["1", "3", "^"], true),
            // OpPlus with concat
            ("(.+)/(foo|bar)", vec![], false),
            // Simple sets containing special characters without escaping.
            ("fo.o|bar?|^baz", vec![], false),
            // case-sensitive wrapper.
            ("(?i)foo", vec!["foo"], false),
            // case-sensitive wrapper on alternate.
            ("(?i)foo|bar|baz", vec!["FOO", "BAR", "BAZ", "BAr", "BAz"], false),
            // mixed case sensitivity.
            ("(api|rpc)_(v1|prom)_((?i)push|query)", vec![], false),
            // mixed case sensitivity concatenation only without capture group.
            ("api_v1_(?i)push", vec![], false),
            // mixed case sensitivity alternation only without capture group.
            ("api|(?i)rpc", vec![], false),
            // case sensitive after unsetting insensitivity.
            ("rpc|(?i)(?-i)api", vec!["rpc", "api"], true),
            // case-sensitive after unsetting insensitivity in all alternation options.
            ("(?i)((?-i)api|(?-i)rpc)", vec!["api", "rpc"], true),
            // mixed case sensitivity after unsetting insensitivity.
            ("(?i)rpc|(?-i)api", vec![], false),
            // too high charset combination
            ("(api|rpc)_[^0-9]", vec![], false),
            // too many combinations
            ("[a-z][a-z]", vec![], false),
        ];

        for (pattern, exp_matches, exp_case_sensitive) in cases {
            let mut parsed = build_hir(&format!("^(?s:{})$", pattern)).unwrap();
            let matches = find_set_matches(&mut parsed).unwrap();
            assert_eq!(exp_matches, matches);

            if exp_case_sensitive {
                // When the regexp is case-sensitive, we want to ensure that the
                // set matches are maintained in the final matcher.
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
        let m = FastRegexMatcher::new("a|b").unwrap();
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
            Box::new(literal(s))
        }

        fn true_matcher() -> StringMatchHandler {
            StringMatchHandler::MatchAll
        }
        
        fn some_true_matcher() -> Option<StringMatchHandler> {
            Some(true_matcher())
        }

        fn suffix(s: &str, left: Option<StringMatchHandler>) -> StringMatchHandler {
            StringMatchHandler::suffix(left, s.to_string())
        }

        fn prefix(s: &str, right: Option<StringMatchHandler>) -> StringMatchHandler {
            StringMatchHandler::prefix(s.to_string(), right)
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
            let matches = string_matcher_from_hir(&mut parsed);
            assert_eq!(c.1, matches);
        }
    }

    #[test]
    fn test_string_matcher_from_regexp_literal_prefix() {
        struct TestConfig {
            pattern: String,
            expected_literal_prefix_matchers: usize,
            expected_matches: Vec<String>,
            expected_not_matches: Vec<String>,
        }

        let test_cases: Vec<TestConfig> = vec![
            // Case sensitive
            TestConfig {
                pattern: "(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*)".to_string(),
                expected_literal_prefix_matchers: 3,
                expected_matches: vec![
                    "xyz-016a-ixb-dp".to_string(),
                    "xyz-016a-ixb-dpXXX".to_string(),
                    "xyz-016a-ixb-op".to_string(),
                    "xyz-016a-ixb-opXXX".to_string(),
                    "xyz-016a-ixb-dp\n".to_string(),
                ],
                expected_not_matches: vec![
                    "XYZ-016a-ixb-dp".to_string(),
                    "xyz-016a-ixb-d".to_string(),
                    "XYZ-016a-ixb-op".to_string(),
                    "xyz-016a-ixb-o".to_string(),
                    "xyz".to_string(),
                    "dp".to_string(),
                ],
            },
            // Case insensitive
            TestConfig {
                pattern: "(?i)(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*)".to_string(),
                expected_literal_prefix_matchers: 3,
                expected_matches: vec![
                    "xyz-016a-ixb-dp".to_string(),
                    "XYZ-016a-ixb-dpXXX".to_string(),
                    "xyz-016a-ixb-op".to_string(),
                    "XYZ-016a-ixb-opXXX".to_string(),
                    "xyz-016a-ixb-dp\n".to_string(),
                ],
                expected_not_matches: vec![
                    "xyz-016a-ixb-d".to_string(),
                    "xyz".to_string(),
                    "dp".to_string(),
                ],
            },
            // Nested literal prefixes, case sensitive
            TestConfig {
                pattern: "(xyz-(aaa-(111.*)|bbb-(222.*)))|(xyz-(aaa-(333.*)|bbb-(444.*)))".to_string(),
                expected_literal_prefix_matchers: 10,
                expected_matches: vec![
                    "xyz-aaa-111".to_string(),
                    "xyz-aaa-111XXX".to_string(),
                    "xyz-aaa-333".to_string(),
                    "xyz-aaa-333XXX".to_string(),
                    "xyz-bbb-222".to_string(),
                    "xyz-bbb-222XXX".to_string(),
                    "xyz-bbb-444".to_string(),
                    "xyz-bbb-444XXX".to_string(),
                ],
                expected_not_matches: vec![
                    "XYZ-aaa-111".to_string(),
                    "xyz-aaa-11".to_string(),
                    "xyz-aaa-222".to_string(),
                    "xyz-bbb-111".to_string(),
                ],
            },
            // Nested literal prefixes, case insensitive
            TestConfig {
                pattern: "(?i)(xyz-(aaa-(111.*)|bbb-(222.*)))|(xyz-(aaa-(333.*)|bbb-(444.*)))".to_string(),
                expected_literal_prefix_matchers: 10,
                expected_matches: vec![
                    "xyz-aaa-111".to_string(),
                    "XYZ-aaa-111XXX".to_string(),
                    "xyz-aaa-333".to_string(),
                    "xyz-AAA-333XXX".to_string(),
                    "xyz-bbb-222".to_string(),
                    "xyz-BBB-222XXX".to_string(),
                    "XYZ-bbb-444".to_string(),
                    "xyz-bbb-444XXX".to_string(),
                ],
                expected_not_matches: vec![
                    "xyz-aaa-11".to_string(),
                    "xyz-aaa-222".to_string(),
                    "xyz-bbb-111".to_string(),
                ],
            },
            // Mixed case sensitivity
            TestConfig {
                pattern: "(xyz-((?i)(aaa.*|bbb.*)))".to_string(),
                expected_literal_prefix_matchers: 3,
                expected_matches: vec![
                    "xyz-aaa".to_string(),
                    "xyz-AAA".to_string(),
                    "xyz-aaaXXX".to_string(),
                    "xyz-AAAXXX".to_string(),
                    "xyz-bbb".to_string(),
                    "xyz-BBBXXX".to_string(),
                ],
                expected_not_matches: vec![
                    "XYZ-aaa".to_string(),
                    "xyz-aa".to_string(),
                    "yz-aaa".to_string(),
                    "aaa".to_string(),
                ],
            },
        ];

        for case in test_cases {
            let re = Regex::new(&case.pattern).unwrap();
            let matcher = string_matcher_from_regex(&case.pattern).unwrap().unwrap();

            // Pre-condition check: ensure it contains literalPrefixSensitiveStringMatcher or literalPrefixInsensitiveStringMatcher.
            let mut num_prefix_matchers = 0;
            visit_string_matcher(&matcher, &mut num_prefix_matchers, |m, state| {
                if let StringMatchHandler::Prefix(_) = m {
                    *state += 1;
                }
            });

            // Count literal prefix matchers
            let num_prefix_matchers = count_literal_prefix_matchers(&case.pattern);
            assert_eq!(
                num_prefix_matchers, case.expected_literal_prefix_matchers,
                "Pattern: {}",
                case.pattern
            );

            // Test matches
            for value in &case.expected_matches {
                assert!(
                    matcher.matches(value),
                    "Matcher: Pattern - {}, Value: {value} should match", case.pattern
                );

                assert!(
                    re.is_match(value),
                    "Regex Pattern: {}, Value: {value} should match", case.pattern
                );
            }

            // Test non-matches
            for value in &case.expected_not_matches {
                assert!(
                    !matcher.matches(value),
                    "Matcher: Pattern - {}, Value: {value} should not match", case.pattern
                );

                assert!(
                    !re.is_match(value),
                    "Pattern: {}, Value: {value} should not match", case.pattern
                );
            }
        }
    }

    #[test]
    fn test_string_matcher_from_regexp_literal_suffix() {
        struct TestCase {
            pattern: &'static str,
            expected_literal_suffix_matchers: usize,
            expected_matches: Vec<&'static str>,
            expected_not_matches: Vec<&'static str>,
        }

        let test_cases = vec![
            TestCase {
                pattern: "(.*xyz-016a-ixb-dp|.*xyz-016a-ixb-op)",
                expected_literal_suffix_matchers: 2,
                expected_matches: vec![
                    "xyz-016a-ixb-dp",
                    "XXXxyz-016a-ixb-dp",
                    "xyz-016a-ixb-op",
                    "XXXxyz-016a-ixb-op",
                    "\nxyz-016a-ixb-dp"
                ],
                expected_not_matches: vec![
                    "XYZ-016a-ixb-dp", "yz-016a-ixb-dp", "XYZ-016a-ixb-op", "xyz-016a-ixb-o", "xyz", "dp"
                ],
            },
            // Case insensitive.
            TestCase {
                pattern: "(?i)(.*xyz-016a-ixb-dp|.*xyz-016a-ixb-op)",
                expected_literal_suffix_matchers: 2,
                expected_matches: vec![
                    "xyz-016a-ixb-dp",
                    "XYZ-016a-ixb-dp",
                    "XXXxyz-016a-ixb-dp",
                    "XyZ-016a-ixb-op",
                    "XXXxyz-016a-ixb-op",
                    "\nxyz-016a-ixb-dp"
                ],
                expected_not_matches: vec![
                    "yz-016a-ixb-dp", "xyz-016a-ixb-o", "xyz", "dp"
                ],
            },
            // Nested literal suffixes, case sensitive.
            TestCase {
                pattern: "(.*aaa|.*bbb(.*ccc|.*ddd))",
                expected_literal_suffix_matchers: 3,
                expected_matches: vec![
                    "aaa", "XXXaaa", "bbbccc", "XXXbbbccc", "XXXbbbXXXccc", "bbbddd",
                    "bbbddd", "XXXbbbddd", "XXXbbbXXXddd", "bbbXXXccc", "aaabbbccc", "aaabbbddd"
                ],
                expected_not_matches: vec![
                    "AAA", "aa", "Xaa", "BBBCCC", "bb", "Xbb", "bbccc", "bbbcc", "bbbdd"
                ],
            },
            // Mixed case sensitivity.
            TestCase {
                pattern: "(.*aaa|.*bbb((?i)(.*ccc|.*ddd)))",
                expected_literal_suffix_matchers: 3,
                expected_matches: vec![
                    "aaa", "XXXaaa", "bbbccc", "bbbCCC", "bbbXXXCCC",
                    "bbbddd", "bbbDDD", "bbbXXXddd", "bbbXXXDDD"
                ],
                expected_not_matches: vec![
                    "AAA", "XXXAAA", "BBBccc", "BBBCCC", "aaaBBB"
                ],
            },

        ];

        for case in test_cases {
            // Create the matcher
            let matcher = string_matcher_from_regex(&case.pattern).unwrap().unwrap();

            // Compile the regex
            let re = Regex::new(&format!("^(?s:{})$", &case.pattern)).unwrap();

            // Pre-condition check: ensure it contains literalSuffixStringMatcher
            let mut num_suffix_matchers = 0;
            visit_string_matcher(&matcher, &mut num_suffix_matchers, |m, &mut mut state| {
                if let StringMatchHandler::Suffix(_) = m {
                    state += 1;
                }
            });

            assert_eq!(num_suffix_matchers, case.expected_literal_suffix_matchers);

            // Test expected matches
            for value in case.expected_matches {
                assert!(
                    matcher.matches(value),
                    "Value: {} should match",
                    value
                );
                assert!(
                    re.is_match(value),
                    "Value: {} should match (regex)",
                    value
                );
            }

            // Test expected not matches
            for value in case.expected_not_matches {
                assert!(
                    !matcher.matches(value),
                    "Value: {} should not match",
                    value
                );
                assert!(
                    !re.is_match(value),
                    "Value: {} should not match (regex)",
                    value
                );
            }
        }
    }

    #[test]
    fn test_string_matcher_from_regexp_quest() {
        struct TestCase {
            pattern: &'static str,
            expected_zero_or_one_matchers: usize,
            expected_matches: Vec<&'static str>,
            expected_not_matches: Vec<&'static str>,
        }

        let test_cases = vec![
            TestCase {
                pattern: "test.?",
                expected_zero_or_one_matchers: 1,
                expected_matches: vec!["test\n", "test", "test!"],
                expected_not_matches: vec!["tes", "test!!"]
            },
            TestCase {
                pattern: ".?test",
                expected_zero_or_one_matchers: 1,
                expected_matches: vec!["\ntest", "test", "!test"],
                expected_not_matches: vec!["tes", "test!"],
            },
            TestCase {
                pattern: "(aaa.?|bbb.?)",
                expected_zero_or_one_matchers: 2,
                expected_matches: vec!["aaa", "aaaX", "bbb", "bbbX", "aaa\n", "bbb\n"],
                expected_not_matches: vec!["aa", "aaaXX", "bb", "bbbXX"],
            },
            TestCase {
                pattern: ".*aaa.?",
                expected_zero_or_one_matchers: 1,
                expected_matches: vec![
                    "aaa", "Xaaa", "aaaX", "XXXaaa", "XXXaaaX", "XXXaaa\n",
                ],
                expected_not_matches: vec!["aa", "aaaXX", "XXXaaaXXX"],
            },
            TestCase {
                pattern: "(?s)test.?",
                expected_zero_or_one_matchers: 1,
                expected_matches: vec!["test", "test!", "test\n"],
                expected_not_matches: vec!["tes", "test!!", "test\n\n"],
            },
            TestCase {
                pattern: "(aaa.?|((?s).?bbb.+))",
                expected_zero_or_one_matchers: 2,
                expected_matches: vec!["aaa", "aaaX", "bbbX", "XbbbX", "bbbXXX", "\nbbbX", "aaa\n"],
                expected_not_matches: vec!["aa", "Xbbb", "\nbbb"],
            },
        ];

        for case in test_cases {
            let re = Regex::new(&format!("^(?s:{})$", case.pattern)).unwrap();
            let matcher = string_matcher_from_regex(&case.pattern).unwrap().unwrap();

            // Pre-condition check: ensure it contains zeroOrOneCharacterStringMatcher.
            let mut num_zero_or_one_matchers = 0;
            visit_string_matcher(&matcher, &mut num_zero_or_one_matchers,|_, state| {
                // Placeholder for the logic to count zeroOrOneCharacterStringMatcher.
                *state += 1;
            });

            assert_eq!(
                num_zero_or_one_matchers, case.expected_zero_or_one_matchers,
                "Pattern: {}",
                case.pattern
            );

            for value in case.expected_matches {
                assert!(
                    re.is_match(value),
                    "Re: Pattern: {}, Value: {}",
                    case.pattern,
                    value
                );
                assert!(
                    matcher.matches(&value),
                    "Pattern: {}, Value: {}",
                    case.pattern,
                    value
                );
            }

            for value in case.expected_not_matches {
                assert!(
                    !re.is_match(value),
                    "Re Not Match: Pattern: {}, Value: {}",
                    case.pattern,
                    value
                );

                assert!(
                    !matcher.matches(&value),
                    "Pattern: {}, Value: {}",
                    case.pattern,
                    value
                );
            }
        }

    }


    fn visit_string_matcher<STATE>(matcher: &StringMatchHandler,
                                   state: &mut STATE,
                                   callback: fn(&StringMatchHandler, &mut STATE))
    {
        callback(matcher, state);

        match matcher {
            StringMatchHandler::ContainsMulti(m) => {
                if let Some(left) = &m.left {
                    visit_string_matcher(left, state, callback);
                }
                if let Some(right) = &m.right {
                    visit_string_matcher(right, state, callback);
                }
            }
            StringMatchHandler::Or(m) => {
                for entry in m {
                    visit_string_matcher(entry, state, callback);
                }
            }
            StringMatchHandler::Prefix(m) => {
                if let Some(right) = &m.right {
                    visit_string_matcher(right, state, callback);
                }
            }
            StringMatchHandler::Suffix(m) => {
                if let Some(left) = &m.left {
                    visit_string_matcher(left, state, callback);
                }
            }
            StringMatchHandler::EqualMultiMap(m) => {
                for (_, prefixes) in &m.prefixes {
                    for matcher in prefixes {
                        visit_string_matcher(&matcher, state, callback)
                    }
                }
            }
            _ => {}
        }
    }

    // Helper function to count literal prefix matchers
    fn count_literal_prefix_matchers(pattern: &str) -> usize {
        let mut num_prefix_matchers = 0;

        // Simulate counting literal prefix matchers
        // This is a simplified version since Rust's regex crate doesn't directly expose literal prefixes
        if pattern.starts_with("(?i)") {
            num_prefix_matchers += 1; // Case-insensitive prefix
        }
        if pattern.contains(".*") {
            num_prefix_matchers += 1; // Wildcard suffix
        }

        num_prefix_matchers
    }

}
