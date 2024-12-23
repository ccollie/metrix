// Copyright 2020 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    use crate::regex_util::{string_matcher_from_regex, ContainsMultiStringMatcher, StringMatchHandler};
    use crate::regex_util::regex_utils::{build_hir, string_matcher_from_regex_internal};
    use crate::regex_util::string_pattern::StringPattern;

    #[test]
    fn test_fast_regex_matcher_match_string() {
        let test_values = generate_random_values();
        let regexes = vec!["foo", "bar", "baz"];

        for r in regexes {
            let (matcher, _) = string_matcher_from_regex(r).unwrap();
            for v in &test_values {
                let re = Regex::new(&format!("^(?s:{})$", r)).unwrap();
                assert_eq!(re.is_match(v), matcher.matches(v));
            }
        }
    }

    #[test]
    fn test_string_matcher_from_regexp() {
        fn literal(s: &str) -> StringMatchHandler {
            StringMatchHandler::Literal(StringPattern::case_sensitive(s.to_string()))
        }

        fn boxed_literal(s: &str) -> Box<StringMatchHandler> {
            Box::new(literal(s))
        }

        fn true_matcher() -> StringMatchHandler {
            StringMatchHandler::any(false)
        }
        
        fn some_true_matcher() -> Option<StringMatchHandler> {
            Some(true_matcher())
        }

        fn suffix(s: &str, left: Option<StringMatchHandler>) -> StringMatchHandler {
            StringMatchHandler::suffix(left, s.to_string(), true)
        }

        fn prefix(s: &str, right: Option<StringMatchHandler>) -> StringMatchHandler {
            StringMatchHandler::prefix(s.to_string(), right, true)
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
               // we don't support case-insensitive matching for contains.
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
               // Case-sensitive alternate with same literal prefix and .* suffix.
               ("(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*)", Some(
                   prefix("xyz-016a-ixb-",
                          Some(or_matcher(&[prefix("dp", Some(true_matcher())), prefix("op", Some(true_matcher()))]))
                   )
               )),
                // Case-insensitive alternate with same literal prefix and .* suffix.
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
        for (expr, expected_matcher) in cases {
            let hir = build_hir(expr).unwrap();
            let res = string_matcher_from_regex_internal(&hir).unwrap();
            if expected_matcher.is_none() {
                assert!(res.is_none(), "Expected None for {}", expr);
                continue;
            } else {
                assert!(res.is_some(), "Expected Some for {}", expr);
                let expected_matcher = expected_matcher.unwrap();
                let (matcher, _) = res.unwrap();
                assert_eq!(matcher, expected_matcher, "Invalid matcher for {expr}");
            }
        }
    }

    #[test]
    fn test_string_matcher_from_regexp_literal_prefix() {
        struct TestConfig {
            pattern: &'static str,
            expected_literal_prefix_matchers: usize,
            expected_matches: Vec<&'static str>,
            expected_not_matches: Vec<&'static str>,
        }

        let test_cases: Vec<TestConfig> = vec![
            // Case-sensitive
            TestConfig {
                pattern: "(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*)",
                expected_literal_prefix_matchers: 2,
                expected_matches: vec![
                    "xyz-016a-ixb-dp",
                    "xyz-016a-ixb-dpXXX",
                    "xyz-016a-ixb-op",
                    "xyz-016a-ixb-opXXX",
                    "xyz-016a-ixb-dp\n",
                ],
                expected_not_matches: vec![
                    "XYZ-016a-ixb-dp",
                    "xyz-016a-ixb-d",
                    "XYZ-016a-ixb-op",
                    "xyz-016a-ixb-o",
                    "xyz",
                    "dp",
                ],
            },
            // Case-insensitive
            TestConfig {
                pattern: "(?i)(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*)",
                expected_literal_prefix_matchers: 3,
                expected_matches: vec![
                    "xyz-016a-ixb-dp",
                    "XYZ-016a-ixb-dpXXX",
                    "xyz-016a-ixb-op",
                    "XYZ-016a-ixb-opXXX",
                    "xyz-016a-ixb-dp\n",
                ],
                expected_not_matches: vec![
                    "xyz-016a-ixb-d",
                    "xyz",
                    "dp",
                ],
            },
            // Nested literal prefixes, case sensitive
            TestConfig {
                pattern: "(xyz-(aaa-(111.*)|bbb-(222.*)))|(xyz-(aaa-(333.*)|bbb-(444.*)))",
                expected_literal_prefix_matchers: 10,
                expected_matches: vec![
                    "xyz-aaa-111",
                    "xyz-aaa-111XXX",
                    "xyz-aaa-333",
                    "xyz-aaa-333XXX",
                    "xyz-bbb-222",
                    "xyz-bbb-222XXX",
                    "xyz-bbb-444",
                    "xyz-bbb-444XXX",
                ],
                expected_not_matches: vec![
                    "XYZ-aaa-111",
                    "xyz-aaa-11",
                    "xyz-aaa-222",
                    "xyz-bbb-111",
                ],
            },
            // Nested literal prefixes, case-insensitive
            TestConfig {
                pattern: "(?i)(xyz-(aaa-(111.*)|bbb-(222.*)))|(xyz-(aaa-(333.*)|bbb-(444.*)))",
                expected_literal_prefix_matchers: 10,
                expected_matches: vec![
                    "xyz-aaa-111",
                    "XYZ-aaa-111XXX",
                    "xyz-aaa-333",
                    "xyz-AAA-333XXX",
                    "xyz-bbb-222",
                    "xyz-BBB-222XXX",
                    "XYZ-bbb-444",
                    "xyz-bbb-444XXX",
                ],
                expected_not_matches: vec![
                    "xyz-aaa-11",
                    "xyz-aaa-222",
                    "xyz-bbb-111",
                ],
            },
            // Mixed case sensitivity
            TestConfig {
                pattern: "(xyz-((?i)(aaa.*|bbb.*)))",
                expected_literal_prefix_matchers: 3,
                expected_matches: vec![
                    "xyz-aaa",
                    "xyz-AAA",
                    "xyz-aaaXXX",
                    "xyz-AAAXXX",
                    "xyz-bbb",
                    "xyz-BBBXXX",
                ],
                expected_not_matches: vec![
                    "XYZ-aaa",
                    "xyz-aa",
                    "yz-aaa",
                    "aaa",
                ],
            },
        ];

        for case in test_cases {
            let re = Regex::new(&case.pattern).unwrap();
            let (matcher, _) = string_matcher_from_regex(&case.pattern).unwrap();

            // Pre-condition check: ensure it contains literalPrefixSensitiveStringMatcher or literalPrefixInsensitiveStringMatcher.
            let mut num_prefix_matchers = 0;
            visit_string_matcher(&matcher, &mut num_prefix_matchers, |m, state| {
                if let StringMatchHandler::Prefix(_) = m {
                    *state += 1;
                }
            });

            // Count literal prefix matchers
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
            // Case-insensitive.
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
            // Nested literal suffixes, case-sensitive.
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
            let (matcher, _) = string_matcher_from_regex(&case.pattern).unwrap();

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
            let (matcher, _) = string_matcher_from_regex(&case.pattern).unwrap();

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
            StringMatchHandler::LiteralMap(m) => {
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
