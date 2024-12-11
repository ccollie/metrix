use rand::Rng;
use regex::Regex;
use std::sync::Arc;
use std::sync::Mutex;

// Define a trait for StringMatcher
trait StringMatcher {
    fn matches(&self, s: &str) -> bool;
}

// Implement the trait for different matchers
struct TrueMatcher;
impl StringMatcher for TrueMatcher {
    fn matches(&self, _s: &str) -> bool {
        true
    }
}

struct EqualStringMatcher {
    s: String,
    case_sensitive: bool,
}
impl StringMatcher for EqualStringMatcher {
    fn matches(&self, s: &str) -> bool {
        if self.case_sensitive {
            s == self.s
        } else {
            s.to_lowercase() == self.s.to_lowercase()
        }
    }
}

struct ContainsStringMatcher {
    substrings: Vec<String>,
    left: Option<Box<dyn StringMatcher>>,
    right: Option<Box<dyn StringMatcher>>,
}
impl StringMatcher for ContainsStringMatcher {
    fn matches(&self, s: &str) -> bool {
        self.substrings.iter().all(|sub| s.contains(sub))
            && self.left.as_ref().map_or(true, |m| m.matches(s))
            && self.right.as_ref().map_or(true, |m| m.matches(s))
    }
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
    use crate::regex_util::fast_matcher::{find_set_matches, optimize_concat_regex, FastRegexMatcher};
    use super::*;

    #[test]
    fn test_fast_regex_matcher_match_string() {
        let test_values = generate_random_values();
        let regexes = vec!["foo", "bar", "baz"];

        for r in regexes {
            let matcher = FastRegexMatcher::new(r).unwrap();
            for v in &test_values {
                let re = Regex::new(&format!("^(?s:{})$", r)).unwrap();
                assert_eq!(re.is_match(v), matcher.match_string(v));
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
        assert_eq!(re.is_match(v), m.match_string(v), "{}", test_name);
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
            let parsed = Regex::new(&format!("^(?s:{})$", pattern)).unwrap();
            let (matches) = find_set_matches(&parsed);
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

        // Ensure that if we call SetMatches() again we get the original one.
        assert_eq!(vec!["a", "b"], m.set_matches());
    }
}
