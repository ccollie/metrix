use crate::regex_util::fast_matcher::{find_set_matches, EmptyStringMatcher, AnyNonEmptyStringMatcher, FastRegexMatcher, EqualStringMatcher, OrStringMatcher};
use crate::regex_util::hir_utils::build_hir;

// Refer to https://github.com/prometheus/prometheus/issues/2651.
#[test]
fn test_find_set_matches() {

    type Case = (&'static str, Vec<&'static str>, bool);
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
        let parsed = build_hir(&format!("^(?s:{})$", pattern)).unwrap();
        let (matches, actual_case_sensitive) = find_set_matches(&parsed);
        assert_eq!(exp_matches, matches);
        assert_eq!(exp_case_sensitive, actual_case_sensitive);

        if exp_case_sensitive {
            // When the regexp is case sensitive, we want to ensure that the
            // set matches are maintained in the final matcher.
            let r = FastRegexMatcher::new(pattern).unwrap();
            assert_eq!(exp_matches, r.set_matches());
        }
    }
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

    // Ensure that if we call SetMatches() again we get the original one.
    assert_eq!(expected, m.set_matches());
}

struct TrueMatcher {}

#[test]
fn test_string_matcher_from_regexp() {
    let cases = vec![
        (".*", Some(TrueMatcher {})),
        (".*?", Some(TrueMatcher {})),
        ("(?s:.*)", Some(TrueMatcher {})),
        ("(.*)", Some(TrueMatcher {})),
        ("^.*$", Some(TrueMatcher {})),
        (".+", Some(AnyNonEmptyStringMatcher{match_nl: true} )),
        ("(?s:.+)", Some( AnyNonEmptyStringMatcher{match_nl: true} )),
        ("^.+$", Some(AnyNonEmptyStringMatcher{match_nl: true} )),
        ("(.+)", Some(AnyNonEmptyStringMatcher{match_nl: true} )),
        ("", Some(EmptyStringMatcher {})),
        ("^$", Some(EmptyStringMatcher {})),
        ("^foo$", Some(EqualStringMatcher{s: "foo", case_sensitive: true})),
        ("^(?i:foo)$", Some(EqualStringMatcher{s: "FOO".to_string(), case_sensitive: false})),
        ("^((?i:foo)|(bar))$", OrStringMatcher([]StringMatcher{EqualStringMatcher{s: "FOO", case_sensitive: false}, EqualStringMatcher{s: "bar", case_sensitive: true}})),
        ("(?i:((foo|bar)))",
            Some(
                OrStringMatcher([]StringMatcher{EqualStringMatcher{s: "FOO", case_sensitive: false}, EqualStringMatcher{s: "BAR", case_sensitive: false}})
            )
        ),
        ("(?i:((foo1|foo2|bar)))", OrStringMatcher([]StringMatcher{OrStringMatcher([]StringMatcher{EqualStringMatcher{s: "FOO1", case_sensitive: false}, EqualStringMatcher{s: "FOO2", case_sensitive: false}}), EqualStringMatcher{s: "BAR", case_sensitive: false}})},
        ("^((?i:foo|oo)|(bar))$", OrStringMatcher([]StringMatcher{EqualStringMatcher{s: "FOO", case_sensitive: false}, EqualStringMatcher{s: "OO", case_sensitive: false}, EqualStringMatcher{s: "bar", case_sensitive: true}})},
        ("(?i:(foo1|foo2|bar))", OrStringMatcher([]StringMatcher{OrStringMatcher([]StringMatcher{EqualStringMatcher{s: "FOO1", case_sensitive: false}, EqualStringMatcher{s: "FOO2", case_sensitive: false}}), EqualStringMatcher{s: "BAR", case_sensitive: false}})},
        (".*foo.*", Some(ContainsStringMatcher{substrings: vec!["foo".to_string()], left: Some(TrueMatcher{}), right: Some(TrueMatcher{}))),
        ("(.*)foo.*", Some(ContainsStringMatcher{substrings: vec!["foo".to_string()], left: Some(TrueMatcher{}), right: Some(TrueMatcher{})}},
        ("(.*)foo(.*)", Some(ContainsStringMatcher{substrings: vec!["foo".to_string()], left: Some(TrueMatcher{}), right: Some(TrueMatcher{})}},
        ("(.+)foo(.*)", Some(ContainsStringMatcher{substrings: vec!["foo".to_string()], left: Some(AnyNonEmptyStringMatcher{match_nl: true}), right: Some(TrueMatcher{})}},
        ("^.+foo.+", Some(ContainsStringMatcher{substrings: vec!["foo".to_string()], left: Some(AnyNonEmptyStringMatcher{match_nl: true}), right: Some(AnyNonEmptyStringMatcher{match_nl: true})}},
        ("^(.*)(foo)(.*)$", Some(ContainsStringMatcher{substrings: vec!["foo".to_string()], left: Some(TrueMatcher{}), right: Some(TrueMatcher{})}},
        ("^(.*)(foo|foobar)(.*)$",
                Some(ContainsStringMatcher{
                        substrings: vec!["foo".to_string(), "foobar".to_string()],
                        left: Some(TrueMatcher{}),
                        right: Some(TrueMatcher{})
                })
        ),
        ("^(.*)(bar|b|buzz)(.*)$",  ))),
        ("^(.*)(foo|foobar)(.+)$", Some(ContainsStringMatcher{substrings: vec!["foo".to_string(), "foobar".to_string()], left: Some(TrueMatcher{}), right: Some(AnyNonEmptyStringMatcher{match_nl: true})}},
        ("^(.*)(bar|b|buzz)(.+)$", Some(ContainsStringMatcher{substrings: vec!["bar".to_string(), "b".to_string(), "buzz".to_string()], left: Some(TrueMatcher{}), right: Some(AnyNonEmptyStringMatcher{match_nl: true})}},
        ("10\\.0\\.(1|2)\\.+", None),
        ("10\\.0\\.(1|2).+", Some(ContainsStringMatcher{substrings: vec!["10.0.1", "10.0.2"], left: None, right: Some(AnyNonEmptyStringMatcher{match_nl: true})}},
        ("^.+foo", Some(
            LiteralSuffixStringMatcher{ left: Some(AnyNonEmptyStringMatcher{match_nl: true}), 
            suffix: "foo"}}),
        ("foo-.*$", &literalPrefixSensitiveStringMatcher{prefix: "foo-", right: Some(TrueMatcher{})}},
        ("(prometheus|api_prom)_api_v1_.+", Some(ContainsStringMatcher{substrings: vec!["prometheus_api_v1_", "api_prom_api_v1_"], left: None, right: Some(AnyNonEmptyStringMatcher{match_nl: true})}},
        ("^((.*)(bar|b|buzz)(.+)|foo)$", OrStringMatcher([]StringMatcher{Some(ContainsStringMatcher{substrings: vec!["bar", "b", "buzz"], left: trueMatcher{}, right: Some(AnyNonEmptyStringMatcher{match_nl: true)), EqualStringMatcher{s: "foo", case_sensitive: true}})},
        ("((fo(bar))|.+foo)", OrStringMatcher([]StringMatcher{OrStringMatcher([]StringMatcher{EqualStringMatcher{s: "fobar", case_sensitive: true}}), &literalSuffixStringMatcher{suffix: "foo", suffixcase_sensitive: true, left: Some(AnyNonEmptyStringMatcher{match_nl: true})}})},
        ("(.+)/(gateway|cortex-gw|cortex-gw-internal)", Some(
                ContainsStringMatcher{
                    substrings: vec!["/gateway".to_string(), "/cortex-gw".to_string(), "/cortex-gw-internal".to_string()],
                    left: Some(AnyNonEmptyStringMatcher{match_nl: true}),
                    right: None
                })
        ),
        // we don't support case insensitive matching for contains.
        // This is because there's no strings.IndexOfFold function.
        // We can revisit later if this is really popular by using strings.ToUpper.
        ("^(.*)((?i)foo|foobar)(.*)$", None),
        ("(api|rpc)_(v1|prom)_((?i)push|query)", None),
        ("[a-z][a-z]", None),
        ("[1^3]", None),
        (".*foo.*bar.*", None),
        {`\d*`, None),
        (".", None),
        ("/|/bar.*", &literalPrefixSensitiveStringMatcher{prefix: "/", right: orStringMatcher{EmptyStringMatcher{}, &literalPrefixSensitiveStringMatcher{prefix: "bar", right: Some(TrueMatcher{})}}}},
        // This one is not supported because  `stringMatcherFromRegexp` is not reentrant for syntax.OpConcat.
        // It would make the code too complex to handle it.
        ("(.+)/(foo.*|bar$)", None),
        // Case sensitive alternate with same literal prefix and .* suffix.
        ("(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*)", &literalPrefixSensitiveStringMatcher{prefix: "xyz-016a-ixb-", right: orStringMatcher{&literalPrefixSensitiveStringMatcher{prefix: "dp", right: Some(TrueMatcher {})), &literalPrefixSensitiveStringMatcher{prefix: "op", right: Some(TrueMatcher{})}}}},
        // Case insensitive alternate with same literal prefix and .* suffix.
        ("(?i:(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*))", &literalPrefixInsensitiveStringMatcher{prefix: "XYZ-016A-IXB-", right: orStringMatcher{&literalPrefixInsensitiveStringMatcher{prefix: "DP", right: Some(TrueMatcher {})), &literalPrefixInsensitiveStringMatcher{prefix: "OP", right: Some(TrueMatcher{})}}}},
        ("(?i)(xyz-016a-ixb-dp.*|xyz-016a-ixb-op.*)", &literalPrefixInsensitiveStringMatcher{prefix: "XYZ-016A-IXB-", right: orStringMatcher{&literalPrefixInsensitiveStringMatcher{prefix: "DP", right: Some(TrueMatcher {})), &literalPrefixInsensitiveStringMatcher{prefix: "OP", right: Some(TrueMatcher{})}}}},
        // Concatenated variable length selectors are not supported.
        ("foo.*.*", None),
        ("foo.+.+", None),
        (".*.*foo", None),
        (".+.+foo", None),
        ("aaa.?.?", None),
        ("aaa.?.*", None),
        // Regexps with ".?".
        ("ext.?|xfs", orStringMatcher{&literalPrefixSensitiveStringMatcher{prefix: "ext", right: &zeroOrOneCharacterStringMatcher{match_nl: true}}, EqualStringMatcher{s: "xfs", case_sensitive: true}}},
        ("(?s)(ext.?|xfs)", orStringMatcher{&literalPrefixSensitiveStringMatcher{prefix: "ext", right: &zeroOrOneCharacterStringMatcher{match_nl: true}}, EqualStringMatcher{s: "xfs", case_sensitive: true}}},
        ("foo.?", &literalPrefixSensitiveStringMatcher{prefix: "foo", right: &zeroOrOneCharacterStringMatcher{match_nl: true}}},
        ("f.?o", None)
    ];
    for c in cases {
        let parsed = syntax.Parse(c.pattern, syntax.Perl|syntax.DotNL)
        let matches = string_matcher_from_regexp(parsed);
    }
}
