use super::hir_utils::*;
use super::match_handlers::{StringMatchHandler, StringMatchOptions};
use crate::regex_util::Quantifier;
use regex::{Error as RegexError, Regex};
use regex_syntax::hir::{Hir, HirKind};
use regex_syntax::parse as parse_regex;

const META_CHARS: &str = ".^$*+?{}[]|()\\/%~";
pub fn contains_regex_meta_chars(s: &str) -> bool {
    s.chars().any(|c| META_CHARS.contains(c))
}

/// remove_start_end_anchors removes '^' at the start of expr and '$' at the end of the expr.
pub fn remove_start_end_anchors(expr: &str) -> &str {
    let mut cursor = expr;
    while let Some(t) = cursor.strip_prefix('^') {
        cursor = t;
    }
    while cursor.ends_with("$") && !cursor.ends_with("\\$") {
        if let Some(t) = cursor.strip_suffix("$") {
            cursor = t;
        } else {
            break;
        }
    }
    cursor
}

pub fn is_valid_regexp(expr: &str) -> bool {
    if expr == ".*" || expr == ".+" || expr.is_empty() {
        return true;
    }
    parse_regex(expr).is_ok()
}

/// These cost values are used for sorting tag filters in ascending order or the required CPU
/// time for execution.
///
/// These values are obtained from BenchmarkOptimizedRematch_cost benchmark.
pub const EMPTY_MATCH_COST: usize = 0;
pub const FULL_MATCH_COST: usize = 1;
pub const PREFIX_MATCH_COST: usize = 2;
pub const LITERAL_MATCH_COST: usize = 3;
pub const SUFFIX_MATCH_COST: usize = 4;
pub const MIDDLE_MATCH_COST: usize = 6;
pub const RE_MATCH_COST: usize = 100;

/// get_optimized_re_match_func tries returning optimized function for matching the given expr.
///
///    '.*'
///    '.+'
///    'literal.*'
///    'literal.+'
///    '.*literal'
///    '.+literal
///    '.*literal.*'
///    '.*literal.+'
///    '.+literal.*'
///    '.+literal.+'
///     'foo|bar|baz|quux'
///     '(foo|bar|baz)quux'
///     'foo(bar|baz)'
///
/// It returns re_match if it cannot find optimized function.
///
/// It also returns literal suffix from the expr.
pub fn get_optimized_re_match_func(expr: &str) -> Result<(StringMatchHandler, usize), RegexError> {
    fn create_re_match_fn(expr: &str) -> Result<(StringMatchHandler, usize), RegexError> {
        let re = Regex::new(expr)?;
        let re_match = StringMatchHandler::fast_regex(re);
        Ok((re_match, RE_MATCH_COST))
    }

    if expr.is_empty() {
        return Ok((StringMatchHandler::MatchAll, FULL_MATCH_COST))
    }

    if expr == ".*" {
        return Ok((StringMatchHandler::MatchAll, FULL_MATCH_COST));
    }

    if expr == ".+" {
        return Ok((StringMatchHandler::not_empty(false), FULL_MATCH_COST));
    }

    let mut sre = match build_hir(expr) {
        Ok(sre) => sre,
        Err(err) => {
            panic!(
                "BUG: unexpected error when parsing verified expr={expr}: {:?}",
                err
            );
        }
    };

    let mut anchor_start = false;
    let mut anchor_end = false;

    // let debug_str = format!("{:?}, {}", sre, hir_to_string(&sre));
    //
    // println!("expr {}", debug_str);

    if let HirKind::Concat(subs) = sre.kind() {
        let mut concat = &subs[..];

        if !subs.is_empty() {
            if let HirKind::Look(_) = subs[0].kind() {
                concat = &concat[1..];
                anchor_start = true;
            }
            if let HirKind::Look(_) = subs[subs.len() - 1].kind() {
                concat = &concat[..concat.len() - 1];
                anchor_end = true;
            }
        }

        if concat.len() != subs.len() {
            if concat.len() == 1 {
                sre = concat[0].clone();
            } else {
                sre = Hir::concat(Vec::from(concat));
            }
        }
    }

    // Prepare fast string matcher for re_match.
    if let Some(match_func) = get_optimized_re_match_func_ext(expr, &sre, anchor_start, anchor_end)?
    {
        // Found optimized function for matching the expr.
        return Ok(match_func);
    }

    // Fall back to re_match_fast.
    create_re_match_fn(expr)
}

fn get_optimized_re_match_func_ext(
    expr: &str,
    sre: &Hir,
    anchor_start: bool,
    anchor_end: bool,
) -> Result<Option<(StringMatchHandler, usize)>, RegexError> {
    fn handle_alternates(
        node: &Hir,
        prefix_quantifier: Option<Quantifier>,
        suffix_quantifier: Option<Quantifier>,
        anchor_start: bool,
        anchor_end: bool,
    ) -> Option<(StringMatchHandler, usize)> {
        let mut alternates = Vec::new();
        if get_or_values_ext(node, &mut alternates) {
            let match_options = StringMatchOptions {
                anchor_start,
                prefix_quantifier,
                anchor_end,
                suffix_quantifier,
            };
            let cost = alternates.len() * LITERAL_MATCH_COST;
            let matcher = StringMatchHandler::alternates(alternates, &match_options);
            return Some((matcher, cost));
        }
        None
    }

    fn handle_literal(
        literal: String,
        prefix_quantifier: Option<Quantifier>,
        suffix_quantifier: Option<Quantifier>,
        anchor_start: bool,
        anchor_end: bool,
    ) -> Option<(StringMatchHandler, usize)> {
        let match_options = StringMatchOptions {
            anchor_start,
            prefix_quantifier,
            anchor_end,
            suffix_quantifier,
        };
        let matcher = StringMatchHandler::literal(literal, &match_options);
        Some((matcher, LITERAL_MATCH_COST))
    }

    if is_dot_star(sre) {
        // '.*'
        return Ok(Some((StringMatchHandler::MatchAll, FULL_MATCH_COST)));
    }

    if is_dot_plus(sre) {
        // '.+'
        return Ok(Some((StringMatchHandler::not_empty(true), FULL_MATCH_COST)));
    }

    match sre.kind() {
        HirKind::Alternation(alts) => {
            let len = alts.len();

            let mut items = &alts[..];
            let prefix = &items[0];
            let suffix = &items[len - 1];

            let mut has_quantifier = false;
            let prefix_quantifier = get_quantifier(prefix);
            let suffix_quantifier = get_quantifier(suffix);

            if len >= 2 {
                // possible .+foo|bar|baz|quux.+ or .+foo|bar|baz|quux or foo|bar|baz|quux.+
                if prefix_quantifier.is_some() {
                    has_quantifier = true;
                    items = &items[1..];
                }
                if suffix_quantifier.is_some() {
                    has_quantifier = true;
                    items = &items[..items.len() - 1];
                }
                if has_quantifier {
                    let res = match items.len() {
                        0 => {
                            // should not happen
                            None
                        }
                        1 => handle_alternates(
                            &items[0],
                            prefix_quantifier,
                            suffix_quantifier,
                            anchor_start,
                            anchor_end,
                        ),
                        _ => {
                            let node = Hir::alternation(Vec::from(items));
                            handle_alternates(
                                &node,
                                prefix_quantifier,
                                suffix_quantifier,
                                anchor_start,
                                anchor_end,
                            )
                        }
                    };
                    return Ok(res);
                }
            }

            Ok(handle_alternates(
                sre,
                prefix_quantifier,
                suffix_quantifier,
                anchor_start,
                anchor_end,
            ))
        }
        HirKind::Capture(cap) => {
            // Remove parenthesis from expr, i.e. '(expr) -> expr'
            get_optimized_re_match_func_ext(expr, cap.sub.as_ref(), anchor_start, anchor_end)
        }
        HirKind::Class(_class) => Ok(handle_alternates(sre, None, None, anchor_start, anchor_end)),
        HirKind::Literal(_lit) => {
            if let Some(s) = get_literal(sre) {
                // Literal match
                let res = handle_literal(s, None, None, anchor_start, anchor_end);
                return Ok(res);
            }
            Ok(None)
        }
        HirKind::Concat(subs) => {
            if subs.len() == 2 {
                let first = &subs[0];
                let second = &subs[1];

                let prefix_quantifier = get_quantifier(first);
                let suffix_quantifier = get_quantifier(second);

                if prefix_quantifier.is_some() {
                    if let Some(literal) = get_literal(second) {
                        let res = handle_literal(
                            literal,
                            prefix_quantifier,
                            suffix_quantifier,
                            anchor_start,
                            anchor_end,
                        );
                        return Ok(res);
                    }
                    // try foo(bar).+ or some such
                    if let Some(res) = handle_alternates(
                        second,
                        prefix_quantifier,
                        suffix_quantifier,
                        anchor_start,
                        anchor_end,
                    ) {
                        return Ok(Some(res));
                    }
                } else if suffix_quantifier.is_some() {
                    if let Some(literal) = get_literal(first) {
                        let res = handle_literal(
                            literal,
                            prefix_quantifier,
                            suffix_quantifier,
                            anchor_start,
                            anchor_end,
                        );
                        return Ok(res);
                    }
                    if let Some(res) = handle_alternates(
                        first,
                        prefix_quantifier,
                        suffix_quantifier,
                        anchor_start,
                        anchor_end,
                    ) {
                        return Ok(Some(res));
                    }
                } else if let Some(res) =
                    handle_alternates(sre, None, None, anchor_start, anchor_end)
                {
                    return Ok(Some(res));
                }
            }

            if subs.len() >= 3 {
                let len = subs.len();
                let prefix = &subs[0];
                let suffix = &subs[len - 1];

                // Note: at this point, .* has been removed from both ends of the regexp.
                let prefix_quantifier = get_quantifier(prefix);
                let suffix_quantifier = get_quantifier(suffix);

                let mut middle = &subs[0..];
                if prefix_quantifier.is_some() {
                    middle = &middle[1..];
                }
                if suffix_quantifier.is_some() {
                    middle = &middle[..middle.len() - 1];
                }

                if middle.len() == 1 {
                    let middle = &middle[0];
                    // handle something like '*.middle.*' or '*.middle.+' or '.+middle.*' or '.+middle.+'
                    if let Some(literal) = get_literal(middle) {
                        let options: StringMatchOptions = StringMatchOptions {
                            anchor_start,
                            anchor_end,
                            prefix_quantifier,
                            suffix_quantifier,
                        };
                        let matcher = StringMatchHandler::literal(literal, &options);
                        let cost = match matcher {
                            StringMatchHandler::StartsWith(_) => PREFIX_MATCH_COST,
                            StringMatchHandler::EndsWith(_) => SUFFIX_MATCH_COST,
                            _ => MIDDLE_MATCH_COST,
                        };
                        return Ok(Some((matcher, cost)));
                    }

                    // handle something like '.+(foo|bar)' or '.+foo(bar|baz).+' etc
                    if let Some(res) = handle_alternates(
                        middle,
                        prefix_quantifier,
                        suffix_quantifier,
                        anchor_start,
                        anchor_end,
                    ) {
                        return Ok(Some(res));
                    }
                }

                let concat = Hir::concat(Vec::from(middle));
                if let Some(res) = handle_alternates(
                    &concat,
                    prefix_quantifier,
                    suffix_quantifier,
                    anchor_start,
                    anchor_end,
                ) {
                    return Ok(Some(res));
                }
            }

            let re = Regex::new(expr)?;
            let re_match = StringMatchHandler::fast_regex(re);

            // Verify that the string matches all the literals found in the regexp
            // before applying the regexp.
            // This should optimize the case when the regexp doesn't match the string.
            let mut literals = subs
                .iter()
                .filter(|x| is_literal(x))
                .map(literal_to_string)
                .collect::<Vec<_>>();

            if literals.is_empty() {
                return Ok(Some((re_match, RE_MATCH_COST)));
            }

            let suffix: String = if is_literal(&subs[subs.len() - 1]) {
                literals.pop().unwrap_or("".to_string())
            } else {
                "".to_string()
            };

            let first = if literals.len() == 1 {
                let literal = literals.pop().unwrap();
                StringMatchHandler::Contains(literal)
            } else {
                StringMatchHandler::OrderedAlternates(literals)
            };

            if !suffix.is_empty() {
                let ends_with = StringMatchHandler::match_fn(suffix, |needle, haystack| {
                    !needle.is_empty() && haystack.contains(needle)
                });
                let pred = ends_with.and(first).and(re_match);
                Ok(Some((pred, RE_MATCH_COST)))
            } else {
                let pred = first.and(re_match);
                Ok(Some((pred, RE_MATCH_COST)))
            }
        }
        _ => {
            // todo!()
            Ok(None)
        }
    }
}

fn literal_to_string(sre: &Hir) -> String {
    if let HirKind::Literal(lit) = sre.kind() {
        return String::from_utf8(lit.0.to_vec()).unwrap_or_default();
    }
    "".to_string()
}

fn get_quantifier(sre: &Hir) -> Option<Quantifier> {
    match sre.kind() {
        HirKind::Capture(cap) => get_quantifier(cap.sub.as_ref()),
        HirKind::Repetition(repetition) if repetition.max.is_none() && repetition.greedy => {
            if let HirKind::Class(clazz) = repetition.sub.kind() {
                if is_empty_class(clazz) {
                    return match repetition.min {
                        0 => {
                            if repetition.max == Some(1) {
                                Some(Quantifier::ZeroOrOne)
                            } else {
                                Some(Quantifier::ZeroOrMore)
                            }
                        },
                        1 => Some(Quantifier::OneOrMore),
                        _ => None,
                    };
                }
            }
            None
        }
        _ => None,
    }
}

pub(super) fn optimize_concat_regex(subs: &Vec<Hir>) -> (String, String, Vec<String>, Vec<Hir>) {
    if subs.is_empty() {
        return (String::new(), String::new(), Vec::new(), Vec::new());
    }

    let mut new_subs = subs.iter().cloned().collect::<Vec<_>>();

    while let Some(first) = new_subs.first() {
        if is_start_anchor(first) {
            new_subs.remove(0);
        } else {
            break;
        }
    }

    while let Some(last) = new_subs.last() {
        if is_end_anchor(last) {
            new_subs.pop();
        } else {
            break;
        }
    }

    let mut prefix = String::new();
    let mut suffix = String::new();
    let mut contains = Vec::new();

    let mut start = 0;
    let mut end = new_subs.len();

    if let Some(first) = new_subs.first() {
        if is_literal(first) {
            prefix = literal_to_string(first);
            start = 1;
        }
    }

    if !prefix.is_empty() && new_subs.len()  == 1 {
        return (prefix, suffix, contains, new_subs);
    }

    if let Some(last) = new_subs.last() {
        if is_literal(last) {
            suffix = literal_to_string(last);
            end -= 1;
        }
    }

    for hir in &new_subs[start..end] {
        if is_literal(hir) {
            contains.push(literal_to_string(hir));
        }
    }

    (prefix, suffix, contains, new_subs)
}

#[cfg(test)]
mod test {
    use regex_syntax::hir::HirKind;
    use super::remove_start_end_anchors;
    use crate::prelude::get_optimized_re_match_func;
    use crate::prelude::regex_utils::optimize_concat_regex;
    use crate::regex_util::hir_utils::build_hir;

    #[test]
    fn test_remove_start_end_anchors() {
        fn f(s: &str, result_expected: &str) {
            let result = remove_start_end_anchors(s);
            assert_eq!(
                result, result_expected,
                "unexpected result for remove_start_end_anchors({s}); got {result}; want {}",
                result_expected
            );
        }

        f("", "");
        f("a", "a");
        f("^^abc", "abc");
        f("a^b$c", "a^b$c");
        f("$$abc^", "$$abc^");
        f("^abc|de$", "abc|de");
        f("abc\\$", "abc\\$");
        f("^abc\\$$$", "abc\\$");
        f("^a\\$b\\$$", "a\\$b\\$")
    }

    #[test]
    fn test_regex_failure() {
        let s = "a(";
        let got = build_hir(s);
        assert!(got.is_err());
    }

    fn test_optimized_regex(expr: &str, s: &str, result_expected: bool) {
        let (matcher, _) = get_optimized_re_match_func(expr).unwrap();
        let result = matcher.matches(s);
        assert_eq!(
            result, result_expected,
            "unexpected result when matching {s} against regex={expr}; got {result}; want {result_expected}"
        );
    }

    #[test]
    fn test_simple() {
        let expr = ".+";
        let s = "foobaza";
        let result_expected = true;
        test_optimized_regex(expr, s, result_expected);
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
            let parsed = build_hir(&format!("^(?s:{})$", regex)).unwrap();
            if let HirKind::Concat(hirs) = &parsed.kind() {
                let (actual_prefix, actual_suffix, actual_contains, _) = optimize_concat_regex(hirs);
                assert_eq!(prefix, actual_prefix, "unexpected prefix for regex={regex}. Expected {prefix}, got {actual_prefix}");
                assert_eq!(suffix, actual_suffix, "unexpected suffix for regex={regex}. Expected {suffix}, got {actual_suffix}");
                assert_eq!(contains, actual_contains);
            } else {
                panic!("Expected HirKind::Concat, got {:?}", parsed.kind());
            }
        }
    }

    #[test]
    fn test_regex_match() {
        fn f(expr: &str, s: &str, result_expected: bool) {
            test_optimized_regex(expr, s, result_expected);
        }

        f("", "foo", true);
        f("", "", true);
        f("foo", "", false);
        f(".*", "", true);
        f(".*", "foo", true);
        f(".+", "", false);
        f(".+", "foo", true);
        f("foo.*", "bar", false);
        f("foo.*", "foo", true);
        f("foo.*", "foobar", true);
        f("foo.*", "a foobar", true);
        f("foo.+", "bar", false);
        f("foo.+", "foo", false);
        f("foo.+", "a foo", false);
        f("foo.+", "foobar", true);
        f("foo.+", "a foobar", true);
        f("foo|bar", "", false);
        f("foo|bar", "a", false);
        f("foo|bar", "foo", true);
        f("foo|bar", "foo a", true);
        f("foo|bar", "a foo a", true);
        f("foo|bar", "bar", true);
        f("foo|bar", "foobar", true);
        f("foo(bar|baz)", "a", false);
        f("foo(bar|baz)", "foobar", true);
        f("foo(bar|baz)", "foobaz", true);
        f("foo(bar|baz)", "foobaza", true);
        f("foo(bar|baz)", "a foobaz a", true);
        f("foo(bar|baz)", "foobal", false);
        f("^foo|b(ar)$", "foo", true);
        f("^foo|b(ar)$", "foo a", true);
        f("^foo|b(ar)$", "a foo", false);
        f("^foo|b(ar)$", "bar", true);
        f("^foo|b(ar)$", "a bar", true);
        f("^foo|b(ar)$", "barz", false);
        f("^foo|b(ar)$", "ar", false);
        f(".*foo.*", "foo", true);
        f(".*foo.*", "afoobar", true);
        f(".*foo.*", "abc", false);
        f("foo.*bar.*", "foobar", true);
        f("foo.*bar.*", "foo_bar_", true);
        f("foo.*bar.*", "a foo bar baz", true);
        f("foo.*bar.*", "foobaz", false);
        f("foo.*bar.*", "baz foo", false);
        f(".+foo.+", "foo", false);
        f(".+foo.+", "afoobar", true);
        f(".+foo.+", "afoo", false);
        f(".+foo.+", "abc", false);
        f("foo.+bar.+", "foobar", false);
        f("foo.+bar.+", "foo_bar_", true);
        f("foo.+bar.+", "a foo_bar_", true);
        f("foo.+bar.+", "foobaz", false);
        f("foo.+bar.+", "abc", false);
        f(".+foo.*", "foo", false);
        f(".+foo.*", "afoo", true);
        f(".+foo.*", "afoobar", true);
        f(".*(a|b).*", "a", true);
        f(".*(a|b).*", "ax", true);
        f(".*(a|b).*", "xa", true);
        f(".*(a|b).*", "xay", true);
        f(".*(a|b).*", "xzy", false);
        f("^(?:true)$", "true", true);
        f("^(?:true)$", "false", false);

        f(".+;|;.+", ";", false);
        f(".+;|;.+", "foo", false);
        f(".+;|;.+", "foo;bar", true);
        f(".+;|;.+", "foo;", true);
        f(".+;|;.+", ";foo", true);
        f(".+foo|bar|baz.+", "foo", false);
        f(".+foo|bar|baz.+", "afoo", true);
        f(".+foo|bar|baz.+", "fooa", false);
        f(".+foo|bar|baz.+", "afooa", true);
        f(".+foo|bar|baz.+", "bar", true);
        f(".+foo|bar|baz.+", "abar", true);
        f(".+foo|bar|baz.+", "abara", true);
        f(".+foo|bar|baz.+", "bara", true);
        f(".+foo|bar|baz.+", "baz", false);
        f(".+foo|bar|baz.+", "baza", true);
        f(".+foo|bar|baz.+", "abaz", false);
        f(".+foo|bar|baz.+", "abaza", true);
        f(".+foo|bar|baz.+", "afoo|bar|baza", true);
        f(".+(foo|bar|baz).+", "bar", false);
        f(".+(foo|bar|baz).+", "bara", false);
        f(".+(foo|bar|baz).+", "abar", false);
        f(".+(foo|bar|baz).+", "abara", true);
        f(".+(foo|bar|baz).+", "afooa", true);
        f(".+(foo|bar|baz).+", "abaza", true);

        f(".*;|;.*", ";", true);
        f(".*;|;.*", "foo", false);
        f(".*;|;.*", "foo;bar", true);
        f(".*;|;.*", "foo;", true);
        f(".*;|;.*", ";foo", true);

        f("^bar", "foobarbaz", false);
        f("^foo", "foobarbaz", true);
        f("bar$", "foobarbaz", false);
        f("baz$", "foobarbaz", true);
        f("(bar$|^foo)", "foobarbaz", true);
        f("(bar$^boo)", "foobarbaz", false);
        f("foo(bar|baz)", "a fooxfoobaz a", true);
        f("foo(bar|baz)", "a fooxfooban a", false);
        f("foo(bar|baz)", "a fooxfooban foobar a", true);
    }
}
