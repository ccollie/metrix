use crate::prelude::StringMatchHandler;
use crate::regex_util::fast_matcher::{find_set_matches, FastRegexMatcher};
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
        let (matches) = find_set_matches(&parsed);
        assert_eq!(exp_matches, matches);

        if exp_case_sensitive {
            // When the regexp is case sensitive, we want to ensure that the
            // set matches are maintained in the final matcher.
            let r = FastRegexMatcher::new(pattern).unwrap();
            assert_eq!(exp_matches, r.set_matches());
        }
    }
}
