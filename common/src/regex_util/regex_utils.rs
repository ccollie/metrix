use regex_syntax::hir::Class::{Unicode, Bytes};
use super::match_handlers::StringMatchHandler;
use crate::prelude::{ContainsMultiStringMatcher, EqualMultiStringMatcher, RegexMatcher, RepetitionMatcher};
use crate::regex_util::{LiteralMapMatcher, Quantifier};
use regex::{Error as RegexError, Regex};
use regex_syntax::hir::{Class, Dot, Hir, HirKind, Look, Repetition};
use regex_syntax::parse as parse_regex;
use smallvec::SmallVec;
use std::sync::LazyLock;
use crate::regex_util::string_pattern::StringPattern;

const ANY_CHAR_EXCEPT_LF: LazyLock<Hir> = LazyLock::new(|| Hir::dot(Dot::AnyCharExceptLF));
const ANY_CHAR: LazyLock<Hir> = LazyLock::new(|| Hir::dot(Dot::AnyChar));

const MAX_SET_MATCHES: usize = 256;
// Beyond this, it's better to use regexp.
const MAX_OR_VALUES: usize = 256;

/// The minimum number of alternate values a regex should have to trigger
/// the optimization done by `optimize_equal_or_prefix_string_matchers()` and so use a map
/// to match values instead of iterating over a list.
const MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD: usize = 16;

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

pub(super) fn build_hir(pattern: &str) -> Result<Hir, RegexError> {
    parse_regex(pattern).map_err(|err| RegexError::Syntax(err.to_string()))
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
pub const FN_MATCH_COST: usize = 20;

/// get_optimized_match_func tries returning optimized function for matching the given expr.
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
pub fn string_matcher_from_regex(expr: &str) -> Result<(StringMatchHandler, usize), RegexError> {
    fn create_re_match_fn(expr: &str, sre: &Hir) -> Result<(StringMatchHandler, usize), RegexError> {
        let re_match = handle_regex(expr, sre)?;
        Ok((re_match, RE_MATCH_COST))
    }

    if expr.is_empty() {
        return Ok((StringMatchHandler::Empty, EMPTY_MATCH_COST))
    }

    if expr == ".*" {
        return Ok((StringMatchHandler::any(false), FULL_MATCH_COST));
    }

    if expr == ".+" {
        return Ok((StringMatchHandler::not_empty(false), FULL_MATCH_COST));
    }

    if let Some(res) = optimize_alternating_literals(expr) {
        return Ok(res);
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

    // let debug_str = format!("{:?}, {}", sre, hir_to_string(&sre));
    //
    // println!("expr {}", debug_str);

    if let HirKind::Concat(subs) = sre.kind() {
        let mut concat = &subs[..];

        if !subs.is_empty() {
            if let HirKind::Look(_) = subs[0].kind() {
                concat = &concat[1..];
            }
            if let HirKind::Look(_) = subs[subs.len() - 1].kind() {
                concat = &concat[..concat.len() - 1];
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
    if let Some(match_func) = string_matcher_from_regex_internal(&sre)? {
        // Found optimized function for matching the expr.
        return Ok(match_func);
    }

    // Fall back to re_match_fast.
    create_re_match_fn(expr, &sre)
}

pub(super) fn string_matcher_from_regex_internal(
    sre: &Hir,
) -> Result<Option<(StringMatchHandler, usize)>, RegexError> {

    // Correctly handling anchors inside a regex is tricky,
    // so in this case we fall back to the regex engine.
    if is_start_anchor(sre) || is_end_anchor(sre) {
        return Ok(None);
    }

    match sre.kind() {
        HirKind::Empty => Ok(Some((StringMatchHandler::Empty, EMPTY_MATCH_COST))),
        HirKind::Repetition(rep) => {
            if rep_is_dot_star(rep) {
                // '.*'
                return Ok(Some((StringMatchHandler::any(true), FULL_MATCH_COST)));
            } else if rep_is_dot_plus(rep) {
                // '.+'
                return Ok(Some((StringMatchHandler::not_empty(true), FULL_MATCH_COST)));
            }
            Ok(get_repetition_matcher(sre, rep))
        },
        HirKind::Alternation(alts) => Ok(get_alternation_matcher(&alts)?),
        HirKind::Capture(cap) => {
            // Remove parenthesis from expr, i.e. '(expr) -> expr'
            string_matcher_from_regex_internal(cap.sub.as_ref())
        }
        HirKind::Class(_) => {
            if let Some((alternatives, case_sensitive)) = find_set_matches_internal(sre, "") {
                let cost = alternatives.len() * LITERAL_MATCH_COST;
                let matcher = StringMatchHandler::literal_alternates(alternatives, case_sensitive);
                Ok(Some((matcher, cost)))
            } else {
                Ok(None)
            }
        }
        HirKind::Literal(_lit) => {
            let literal = literal_to_string(sre);
            Ok(Some((StringMatchHandler::equals(literal), LITERAL_MATCH_COST)))
        }
        HirKind::Concat(subs) => get_concat_matcher(&subs),
        _ => {
            // todo!()
            Ok(None)
        }
    }
}

fn get_alternation_matcher(hirs: &[Hir]) -> Result<Option<(StringMatchHandler, usize)>, RegexError> {
    let mut is_all_literal = true;
    let mut num_values: usize = 0;

    let mut matchers: SmallVec<(StringMatchHandler, usize), 6> = SmallVec::new();
    let mut total_cost: usize = 0;
    let mut matches_case_sensitive: Option<bool> = None;

    for sub_hir in hirs {
        if let Some((matcher, cost)) = string_matcher_from_regex_internal(sub_hir)? {
            total_cost += cost;
            let case_sensitive = matcher.is_case_sensitive();

            if let Some(sensitive) = matches_case_sensitive {
                if sensitive != case_sensitive {
                    return Ok(None);
                }
            } else {
                matches_case_sensitive = Some(case_sensitive);
            }

            match &matcher {
                StringMatchHandler::Literal(_) => {
                    num_values += 1;
                },
                StringMatchHandler::Alternates(values) => {
                    num_values += values.len();
                },
                _ => is_all_literal = false,
            }

            matchers.push((matcher, cost));
        } else {
            return Ok(None);
        }
    }

    matchers.sort_by_key(|(_, cost)| *cost);

    let case_sensitive = matches_case_sensitive.unwrap_or_default();

    // optimize the case where all the alternatives are literals
    if is_all_literal {
        if num_values >= MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD {
            let mut res = LiteralMapMatcher::new();
            res.is_case_sensitive = case_sensitive;

            for (matcher, _) in matchers.into_iter() {
                match matcher {
                    StringMatchHandler::Literal(lit) => {
                        res.values.insert(lit.into());
                    },
                    StringMatchHandler::Alternates(matcher) => {
                        for value in matcher.values {
                            res.values.insert(value);
                        }
                    },
                    _ => unreachable!("BUG: unexpected matcher (check is_literal)"),
                }
            }
            return Ok(Some((StringMatchHandler::LiteralMap(res), total_cost)));
        }

        let mut result = EqualMultiStringMatcher::new(case_sensitive, num_values);
        for (matcher, _) in matchers.into_iter() {
            match matcher {
                StringMatchHandler::Literal(lit) => {
                    result.push(lit.into());
                },
                StringMatchHandler::Alternates(matcher) => {
                    for value in matcher.values.into_iter() {
                        result.push(value);
                    }
                }
                _ => unreachable!("BUG: unexpected matcher (check is_literal)"),
            }
        }
        return Ok(Some((StringMatchHandler::Alternates(result), total_cost)));
    }

    let or_matchers = matchers
        .into_iter()
        .map(|(matcher, _)| Box::new(matcher))
        .collect::<Vec<_>>();

    Ok(Some((StringMatchHandler::Or(or_matchers), total_cost)))
}

fn get_repetition_matcher(hir: &Hir, rep: &Repetition) -> Option<(StringMatchHandler, usize)> {
    fn validate_repetition(rep: &Repetition) -> bool {
        // if re.sub.Op != syntax.OpAnyChar && re.sub.Op != syntax.OpAnyCharNotNL {
        //     return nil
        // }
        matches_any_char(&rep.sub) || matches_any_character_except_newline(&rep.sub)
    }

    // .?
    if is_dot_question(hir) {
        let match_nl = matches_any_char(&rep.sub);
        Some((StringMatchHandler::zero_or_one_chars(match_nl), FULL_MATCH_COST))
    } else if rep_is_dot_plus(rep) {
        let matches_any = matches_any_char(&rep.sub);
        let matches_non_newline = matches_any_character_except_newline(&rep.sub);
        if !matches_any && !matches_non_newline {
            return None;
        }
        Some((StringMatchHandler::not_empty(matches_any), FULL_MATCH_COST))
    } else if rep_is_dot_star(rep) {
        if !validate_repetition(rep) {
            return None;
        } else if matches_any_char(&rep.sub) {
            // If the newline is valid, then this matcher literally match any string (even empty).
            Some((StringMatchHandler::any(false), FULL_MATCH_COST))
        } else {
            // Any string is fine (including an empty one), as far as it doesn't contain any newline.
            Some((StringMatchHandler::any(true), FULL_MATCH_COST))
        }
    } else if is_literal(&rep.sub) {
        let literal = literal_to_string(&rep.sub);
        let repetition = RepetitionMatcher::new(literal, rep.min, rep.max);
        // TODO: Fix this !
        let cost = LITERAL_MATCH_COST * rep.min as usize;
        return Some((StringMatchHandler::Repetition(repetition), cost));
    } else {
        None
    }
}

fn get_concat_matcher(hirs: &[Hir]) -> Result<Option<(StringMatchHandler, usize)>, RegexError> {
    if hirs.is_empty() {
        return Ok(Some((StringMatchHandler::Empty, EMPTY_MATCH_COST)));
    }

    if hirs.len() == 1 {
        return string_matcher_from_regex_internal(&hirs[0]);
    }

    fn get_literal_matcher(sre: &Hir) -> (StringMatchHandler, usize) {
        let literal = literal_to_string(sre);
        (StringMatchHandler::Literal(StringPattern::case_sensitive(literal)), LITERAL_MATCH_COST)
    }

    fn get_insensitive_literal_matcher(hirs: &[Hir]) -> Option<(StringMatchHandler, usize)> {
        if let Some((value, len)) = get_case_folded_string(hirs) {
            let matcher = StringMatchHandler::literal(value, false);
            Some((matcher, len))
        } else {
            None
        }
    }

    fn get_repetition_matcher(hir: &Hir) -> Result<Option<(StringMatchHandler, usize)>, RegexError> {
        if get_quantifier(hir).is_some() {
            return Ok(string_matcher_from_regex_internal(hir)?);
        }
        Ok(None)
    }

    let mut left = None;
    let mut right = None;

    let mut first_is_literal = false;
    let mut left_is_case_sensitive = true;
    let mut last_is_literal = false;
    let mut match_len = hirs.len();

    let first = &hirs[0];
    let mut hirs_new = &hirs[0..];
    match first.kind() {
        HirKind::Class(_) => {
            if let Some((matcher, len)) = get_insensitive_literal_matcher(hirs) {
                hirs_new = &hirs[len..];
                left = Some((matcher, LITERAL_MATCH_COST));
                first_is_literal = true;
                match_len = hirs_new.len() + 1;
                if hirs_new.is_empty() {
                    return Ok(left);
                }
            } else {
                return Ok(None)
            }
        }
        HirKind::Repetition(_) => {
            left = get_repetition_matcher(first)?;
            if left.is_some() {
                hirs_new = &hirs[1..];
            }
        }
        HirKind::Literal(_) => {
            let matcher = get_literal_matcher(first);
            left_is_case_sensitive = true;
            left = Some(matcher);
            hirs_new = &hirs[1..];
            first_is_literal = true;
        }
        _ => {}
    }

    if !hirs_new.is_empty() {
        let last = &hirs_new[hirs_new.len() - 1];
        right = string_matcher_from_regex_internal(last)?;
        if let Some(ref r) = right {
            if matches!(r.0, StringMatchHandler::Literal(_)) {
                last_is_literal = true;
            }
            hirs_new = &hirs_new[0..hirs_new.len() - 1];
        }
    }

    if match_len == 2 {
        // NOTE: the parser optimizes concats of successive literals into a single literal, so if we have only two nodes,
        // then the second node is guaranteed to NOT be a literal. IOW, there is no need to check for
        // left_is_literal && right_is_literal
        if first_is_literal && right.is_some() {
            if let Some((matcher, cost)) = right {
                match left {
                    Some((StringMatchHandler::Literal(lit), _)) => {
                        let case_sensitive = lit.is_case_sensitive();
                        let literal: String = lit.into();
                        let handler = StringMatchHandler::prefix(literal, Some(matcher), case_sensitive);
                        return Ok(Some((handler, cost)));
                    },
                    _ => unreachable!("BUG: Literal value should have been set"),
                }
            } else {
                // protected by if last.is_some()
                unreachable!("BUG: None matcher");
            }
        }
        if last_is_literal && left.is_some() {
            if let Some((matcher, cost)) = left {
                match right {
                    Some((StringMatchHandler::Literal(lit), _)) => {
                        let case_sensitive = lit.is_case_sensitive();
                        let literal: String = lit.into();
                        let handler = StringMatchHandler::suffix(Some(matcher), literal, case_sensitive);
                        return Ok(Some((handler, cost)));
                    },
                    _ => unreachable!("BUG: Literal value should have been set"),
                }
            } else {
                // protected by if first.is_some()
                unreachable!("BUG: None matcher");
            }
        }
    }

    let new_len = hirs_new.len();
    let hir = Hir::concat(hirs_new.to_vec());
    let res = find_set_matches_internal(&hir, "");

    if res.is_none() && new_len == 2 {
        // We have not found fixed set matches. We look for other known cases that
        // we can optimize.
        let first = &hirs_new[0];
        let second = &hirs_new[1];
        // prefix is literal.
        if right.is_none() {
            if let HirKind::Literal(_) = first.kind() {
                right = Some(get_literal_matcher(second));
            }
        }
        // Suffix is literal.
        if left.is_none() {
            if let HirKind::Literal(_) = second.kind() {
                left = Some(get_literal_matcher(first));
            }
        }
    }

    let tmp = hir_to_string(&hir);
    println!("{}", tmp);

    let (mut matches, matches_case_sensitive) = res.unwrap_or_default();

    // Ensure we've found some literals to match (optionally with a left and/or right matcher).
    // If not, then this optimization doesn't trigger.
    if matches.is_empty() {
        return Ok(None);
    }

    let mut cost = matches.len() * MIDDLE_MATCH_COST;
    let mut resolved_left = None;
    let mut resolved_right = None;

    // Use the right (and best) matcher based on what we've found.
    match (left, right) {
        (Some((left, left_cost)), Some((right, right_cost))) => {
            resolved_left = Some(left);
            resolved_right = Some(right);
            cost += left_cost + right_cost;
        }
        (Some((left, left_cost)), None) => {
            resolved_left = Some(left);
            cost += left_cost;
            if matches.len() == 1 {
                let case_sensitive = left_is_case_sensitive;
                let suffix = matches.pop().expect("BUG: Invariant failed. Matches is not empty");
                let matcher = StringMatchHandler::suffix(resolved_left, suffix, case_sensitive);
                return Ok(Some((matcher, cost)));
            }
        }
        (None, Some((right, right_cost))) => {
            resolved_right = Some(right);
            cost += right_cost;
            if matches.len() == 1 {
                let case_sensitive = left_is_case_sensitive;
                let prefix = matches.pop().expect("BUG: Invariant failed. Matches is not empty");
                let matcher = StringMatchHandler::prefix(prefix, resolved_right, case_sensitive);
                return Ok(Some((matcher, cost)));
            }
        }
        _ => {
            // No left and right matchers (only fixed set matches).
            let cost = matches.len() * MIDDLE_MATCH_COST;
            let matcher = StringMatchHandler::literal_alternates(matches, matches_case_sensitive);
            return Ok(Some((matcher, cost)));
        }
    }

    // We found literals in the middle. We can trigger the fast path only if
    // the matches are case-sensitive because ContainsMultiStringMatcher doesn't
    // support case-insensitive.
    if matches_case_sensitive {
        let matcher = ContainsMultiStringMatcher::new(matches, resolved_left, resolved_right);
        return Ok(Some((StringMatchHandler::ContainsMulti(matcher), cost)));
    }

    Ok(None)
}

fn is_case_insensitive_class(hir: &Hir) -> Option<char> {
    if let HirKind::Class(class) = hir.kind() {
        match class {
            Unicode(ranges) => {
                match ranges.ranges() {
                    [first, second] => {
                        if first.start() == first.end() && second.start() == second.end() {
                            return Some(first.start());
                        }
                    }
                    _ => {}
                }
            }
            Bytes(ranges) => {
                match ranges.ranges() {
                    [first, second] => {
                        if first.start() == first.end() && second.start() == second.end() {
                            return Some(first.start() as char);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    None
}

/// In HIR, casing is represented by individual Char classes per Unicode case folding. E.g.
/// 'a' is represented by [aA] and 'A' is represented by [aA]. This function returns the coalesced
/// casing for the given string.
pub(super) fn get_case_folded_string(hirs: &[Hir]) -> Option<(String, usize)> {
    let mut res: String = String::with_capacity(16); // todo: calculate

    let mut count: usize = 0;
    for hir in hirs.iter() {
        if let Some(ch) = is_case_insensitive_class(hir) {
            res.push(ch);
            count += 1;
        } else if let HirKind::Literal(lit) = hir.kind() {
            // We may have character classes followed by a non-alphanumeric literal (eg. (?i)xyz-abc).
            // Here we'll have classes for xyz, then a literal for '-' and then classes for abc.
            // We coalesce these literals and character classes together. Note that we are not being
            // exhaustive here and only handling common cases.

            // we do this only if we've already seen a case-insensitive class
            if !res.is_empty() {
                let mut cancelled = false;
                let saved_len = res.len();

                let value = String::from_utf8(lit.0.to_vec()).unwrap_or_default();
                for ch in value.chars() {
                    if ch.is_alphabetic() {
                        res.truncate(saved_len);
                        cancelled = true;
                        break;
                    }
                }

                if cancelled {
                    break;
                }

                res.extend(value.chars());
                count += 1;
            } else {
                break
            }
        } else {
            break;
        }
    }

    if res.is_empty() {
        return None;
    }

    Some((res, count))
}


pub(super) fn is_literal(sre: &Hir) -> bool {
    match sre.kind() {
        HirKind::Literal(_) => true,
        HirKind::Capture(cap) => is_literal(cap.sub.as_ref()),
        _ => false,
    }
}

pub(super) fn is_dot_star(sre: &Hir) -> bool {
    match sre.kind() {
        HirKind::Capture(cap) => is_dot_star(cap.sub.as_ref()),
        HirKind::Alternation(alternate) => {
            alternate.iter().any(is_dot_star)
        }
        HirKind::Repetition(repetition) => {
            repetition.min == 0 &&
                repetition.max.is_none() &&
                repetition.greedy &&
                !sre.properties().is_literal()
        }
        _ => false,
    }
}

pub(super) fn is_dot_plus(sre: &Hir) -> bool {
    match sre.kind() {
        HirKind::Capture(cap) => is_dot_plus(cap.sub.as_ref()),
        HirKind::Alternation(alternate) => {
            alternate.iter().any(is_dot_plus)
        }
        HirKind::Repetition(repetition) => {
            repetition.min == 1 &&
                repetition.max.is_none() &&
                repetition.greedy &&
                !sre.properties().is_literal()
        }
        _ => false,
    }
}

pub(super) fn is_empty_class(class: &Class) -> bool {
    if class.is_empty() {
        return true;
    }
    match class {
        Unicode(uni) => {
            let ranges = uni.ranges();
            if ranges.len() == 2 {
                let first = ranges.first().unwrap();
                let last = ranges.last().unwrap();
                if first.start() == '\0' && last.end() == '\u{10ffff}' {
                    return true;
                }
            }
        }
        Bytes(bytes) => {
            let ranges = bytes.ranges();
            if ranges.len() == 2 {
                let first = ranges.first().unwrap();
                let last = ranges.last().unwrap();
                if first.start() == 0 && last.end() == 255 {
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn is_dot_question(sre: &Hir) -> bool {
    if let HirKind::Repetition(repetition) = sre.kind() {
        return !repetition.greedy
            && repetition.min == 0
            && repetition.max == Some(1)
            && (ANY_CHAR.eq(&repetition.sub) || ANY_CHAR_EXCEPT_LF.eq(&repetition.sub));
    }
    false
}

pub(super) fn matches_any_char(hir: &Hir) -> bool {
    if let HirKind::Class(class) = hir.kind() {
        return is_empty_class(class)
    }
    false
}

pub(super) fn matches_any_character_except_newline(hir: &Hir) -> bool {
    match hir.kind() {
        HirKind::Literal(lit) => {
            // Check if the literal is not a newline
            !lit.0.contains(&b'\n')
        },
        HirKind::Class(class) => {
            match class {
                // Check if the class does not include newline
                Unicode(class) => {
                    let nl = '\n';
                    class.ranges().iter()
                        .all(|range| !(range.start() .. range.end()).contains(&nl))
                },
                Bytes(class) => {
                    let nl = b'\n';
                    class.ranges().iter()
                        .all(|range| !(range.start() .. range.end()).contains(&nl))
                },
            }
        },
        HirKind::Repetition(repetition) => {
            // Check the sub-expression of repetition
            matches_any_character_except_newline(&repetition.sub)
        },
        _ => false, // Other node types do not match any character except newlines
    }
}

pub(super) fn is_anchor(sre: &Hir, look: Look) -> bool {
    matches!(sre.kind(), HirKind::Look(l) if look == *l)
}

pub(super) fn is_start_anchor(sre: &Hir) -> bool {
    is_anchor(sre, Look::Start)
}

pub(super) fn is_end_anchor(sre: &Hir) -> bool {
    is_anchor(sre, Look::End)
}


fn rep_is_dot_star(rep: &Repetition) -> bool {
    rep.min == 0 && rep.max.is_none() && rep.greedy
    // && !sre.properties().is_literal()
}

fn rep_is_dot_plus(repetition: &Repetition) -> bool {
    repetition.min == 1 &&
        repetition.max.is_none()
    // rep.min == 1 && rep.max() // && rep.greedy
}

pub(super) fn literal_to_string(sre: &Hir) -> String {
    if let HirKind::Literal(lit) = sre.kind() {
        return String::from_utf8(lit.0.to_vec()).unwrap_or_default();
    }
    "".to_string()
}

pub(super) fn get_literal(sre: &Hir) -> Option<String> {
    match sre.kind() {
        HirKind::Capture(cap) => get_literal(cap.sub.as_ref()),
        HirKind::Literal(lit) => {
            let s = String::from_utf8(lit.0.to_vec()).unwrap_or_default();
            Some(s)
        }
        _ => None,
    }
}

pub(super) fn hir_to_string(sre: &Hir) -> String {
    match sre.kind() {
        HirKind::Literal(lit) => {
            String::from_utf8(lit.0.to_vec()).unwrap_or_default()
        }
        HirKind::Concat(concat) => {
            let mut s = String::new();
            for hir in concat.iter() {
                s.push_str(&hir_to_string(hir));
            }
            s
        }
        HirKind::Alternation(alternate) => {
            // avoid extra allocation if it's all literal
            if alternate.iter().all(is_literal) {
                return alternate
                    .iter()
                    .map(hir_to_string)
                    .collect::<Vec<_>>()
                    .join("|")
            }
            let mut s = Vec::with_capacity(alternate.len());
            for hir in alternate.iter() {
                s.push(hir_to_string(hir));
            }
            s.join("|")
        }
        HirKind::Repetition(_repetition) => {
            if is_dot_star(sre) {
                return ".*".to_string();
            } else if is_dot_plus(sre) {
                return ".+".to_string();
            }
            sre.to_string()
        }
        _ => {
            sre.to_string()
        }
    }
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

/// optimize_alternating_literals optimizes a regex of the form
///
///	`literal1|literal2|literal3|...`
///
/// this function returns an optimized StringMatcher or None if the regex
/// cannot be optimized in this way
pub(super) fn optimize_alternating_literals(s: &str) -> Option<(StringMatchHandler, usize)> {
    if s.is_empty() {
        return Some((StringMatchHandler::Empty, EMPTY_MATCH_COST));
    }

    let estimated_alternates = s.matches('|').count() + 1;

    if estimated_alternates == 1 {
        if regex::escape(s) == s {
            return Some((StringMatchHandler::equals(s.into()), LITERAL_MATCH_COST));
        }
        return None;
    }

    let use_map = estimated_alternates >= MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD;
    if use_map {
        let mut map = LiteralMapMatcher::new();
        for sub_match in s.split('|') {
            if regex::escape(sub_match) != sub_match {
                return None;
            }
            map.values.insert(sub_match.to_string());
        }

        Some((StringMatchHandler::LiteralMap(map), estimated_alternates * LITERAL_MATCH_COST))
    } else {
        let mut matcher = EqualMultiStringMatcher::new(true, estimated_alternates);
        for sub_match in s.split('|') {
            if regex::escape(sub_match) != sub_match {
                return None;
            }
            matcher.push(sub_match.to_string());
        }

        let multi = StringMatchHandler::Alternates(matcher);
        Some((multi, estimated_alternates * LITERAL_MATCH_COST))
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

// findSetMatches extract equality matches from a regexp.
// Returns nil if we can't replace the regexp by only equality matchers or the regexp contains
// a mix of case-sensitive and case-insensitive matchers.
pub(super) fn find_set_matches(hir: &mut Hir) -> Option<(Vec<String>, bool)> {
    clear_begin_end_text(hir);
    find_set_matches_internal(hir, "")
}

pub(super) fn find_set_matches_internal(hir: &Hir, base: &str) -> Option<(Vec<String>, bool)> {
    match hir.kind() {
        HirKind::Look(Look::Start) | HirKind::Look(Look::End) => None,
        HirKind::Literal(_) => {
            let literal = format!("{}{}", base, literal_to_string(hir));
            Some((vec![literal], true))
        },
        HirKind::Empty => {
            if !base.is_empty() {
                Some((vec![base.to_string()], true))
            } else {
                None
            }
        }
        HirKind::Alternation(_) => find_set_matches_from_alternate(hir, base),
        HirKind::Capture(hir) => find_set_matches_internal(&hir.sub, base),
        HirKind::Concat(_) => find_set_matches_from_concat(hir, base),
        HirKind::Class(class) => {
            match class {
                Unicode(ranges) => {
                    let total_set = ranges.iter()
                        .map(|r| 1 + (r.end() as usize - r.start() as usize))
                        .sum::<usize>();

                    if total_set > MAX_SET_MATCHES {
                        return None;
                    }

                    let mut matches = Vec::new();
                    for range in ranges.iter().flat_map(|r| r.start()..=r.end()) {
                        matches.push(format!("{base}{range}"));
                    }

                    Some((matches, true))
                }
                Bytes(ranges) => {
                    let total_set = ranges.iter()
                        .map(|r| 1 + (r.end() as usize - r.start() as usize))
                        .sum::<usize>();

                    if total_set > MAX_SET_MATCHES {
                        return None;
                    }

                    let mut matches = Vec::new();

                    for ch in ranges.iter().flat_map(|r| r.start()..=r.end()) {
                        matches.push(format!("{base}{ch}"));
                    }

                    Some((matches, true))
                }
            }
        }
        _ => None,
    }
}

fn find_set_matches_from_concat(hir: &Hir, base: &str) -> Option<(Vec<String>, bool)> {

    if let HirKind::Concat(hirs) = hir.kind() {
        let mut matches = vec![base.to_string()];
        let mut matches_case_sensitive: Option<bool> = None;

        let cursor = hirs;
        let mut i: usize = 0;

        let len = cursor.len();
        while i < len {
            let mut hir = &cursor[i]; // todo: get_unchecked
            if let Some((val, len)) = get_case_folded_string(&cursor[i..]) {
                if let Some(sensitive) = matches_case_sensitive {
                    if sensitive {
                        return None;
                    }
                } else {
                    matches_case_sensitive = Some(false);
                }

                matches.push(format!("{base}{val}"));

                i += len;
                if i < len {
                    hir = &cursor[i];
                } else {
                    break;
                }
            }

            let mut new_matches = Vec::new();

            for b in matches.iter() {
                if let Some((items, sensitive)) = find_set_matches_internal(hir, b) {
                    if matches.len() + items.len() > MAX_SET_MATCHES {
                        return None;
                    }

                    if let Some(sensitive) = matches_case_sensitive {
                        if sensitive != sensitive {
                            return None;
                        }
                    } else {
                        matches_case_sensitive = Some(sensitive);
                    }

                    new_matches.extend(items);
                } else {
                    return None;
                }

                i += 1;
                if i < len {
                    hir = &cursor[i];
                } else {
                    break;
                }

            }
            matches = new_matches;
        }

        return Some((matches, matches_case_sensitive.unwrap_or_default()))
    }

    None
}

fn find_set_matches_from_alternate(hir: &Hir, base: &str) -> Option<(Vec<String>, bool)> {
    let mut matches = Vec::new();
    let mut matches_case_sensitive = true;

    match hir.kind() {
        HirKind::Alternation(alternates) => {
            for (i, sub) in alternates.iter().enumerate() {
                if let Some((found, sensitive)) = find_set_matches_internal(sub, base) {
                    if found.is_empty() {
                        return None;
                    }
                    if i == 0 {
                        matches_case_sensitive = sensitive;
                    } else if matches_case_sensitive != sensitive {
                        return None;
                    }
                    if matches.len() + found.len() > MAX_SET_MATCHES {
                        return None;
                    }
                    matches.extend(found);
                } else {
                    return None;
                }
            }
        }
        _ => return None,
    }

    Some((matches, matches_case_sensitive))
}


pub(super) fn clear_begin_end_text(hir: &mut Hir) {

    fn handle_concat(items: &Vec<Hir>) -> Option<Hir> {
        if !items.is_empty() {
            let mut start: usize = 0;
            let mut end: usize = items.len() - 1;

            if is_start_anchor(&items[0]) {
                start += 1;
            }

            if is_end_anchor(&items[end]) {
                end -= 1;
            }
            let slice = &items[start..=end];
            if slice.is_empty() {
                return Some(Hir::empty());
            }
            let hirs: Vec<Hir> = slice.iter().cloned().collect();
            return Some(Hir::concat(hirs))
        }
        None
    }

    match hir.kind() {
        HirKind::Alternation(_) => return,
        HirKind::Concat(hirs) => {
            if let Some(modified) = handle_concat(hirs) {
                *hir = modified;
            }
        },
        HirKind::Capture(capture) => {
            if let HirKind::Concat(hirs) = capture.sub.kind() {
                if let Some(modified) = handle_concat(hirs) {
                    *hir = modified;
                }
            }
        },
        _ => return,
    }
}

fn handle_regex(expr: &str, hir: &Hir) -> Result<StringMatchHandler, RegexError> {
    // todo: ensure anchor
    let regex = Regex::new(&format!("^(?s:{expr})$"))?;

    let mut matches = Vec::new();

    if let HirKind::Concat(hirs) = hir.kind() {
        let (prefix, suffix, contains, subs) = optimize_concat_regex(&hirs);
        let sub = Hir::concat(subs);
        if let Some((sub_matches, case_sensitive)) = find_set_matches_internal(&sub, "") {
            if case_sensitive {
                matches = sub_matches;
            }
        }
        let matcher = RegexMatcher{
            regex,
            prefix,
            suffix,
            contains,
            set_matches: matches
        };
        Ok(StringMatchHandler::Regex(matcher))
    } else {
        if let Some((sub_matches, _)) = find_set_matches_internal(hir, "") {
            matches = sub_matches;
        }
        let matcher = RegexMatcher{
            regex,
            prefix: "".to_string(),
            suffix: "".to_string(),
            contains: Vec::new(),
            set_matches: matches
        };
        Ok(StringMatchHandler::Regex(matcher))
    }
}


pub fn get_or_values(pattern: &str) -> Result<Vec<String>, RegexError> {
    let mut values = Vec::new();
    let sre = build_hir(pattern)?;
    get_or_values_ext(&sre, &mut values);
    Ok(values)
}

pub fn get_or_values_ext(sre: &Hir, dest: &mut Vec<String>) -> bool {
    use HirKind::*;
    match sre.kind() {
        Empty => {
            dest.push("".to_string());
            true
        }
        Capture(cap) => get_or_values_ext(cap.sub.as_ref(), dest),
        Literal(literal) => {
            if let Ok(s) = String::from_utf8(literal.0.to_vec()) {
                dest.push(s);
                true
            } else {
                false
            }
        }
        Alternation(alt) => {
            dest.reserve(alt.len());
            for sub in alt.iter() {
                let start_count = dest.len();
                if let Some(literal) = get_literal(sub) {
                    dest.push(literal);
                } else if !get_or_values_ext(sub, dest) {
                    return false;
                }
                if dest.len() - start_count > MAX_OR_VALUES {
                    return false;
                }
            }
            true
        }
        Concat(concat) => {
            let mut prefixes = Vec::with_capacity(MAX_OR_VALUES);
            if !get_or_values_ext(&concat[0], &mut prefixes) {
                return false;
            }
            let subs = Vec::from(&concat[1..]);
            let concat = Hir::concat(subs);
            let prefix_count = prefixes.len();
            if !get_or_values_ext(&concat, &mut prefixes) {
                return false;
            }
            let suffix_count = prefixes.len() - prefix_count;
            let additional_capacity = prefix_count * suffix_count;
            if additional_capacity > MAX_OR_VALUES {
                // It is cheaper to use regexp here.
                return false;
            }
            dest.reserve(additional_capacity);
            let (pre, suffixes) = prefixes.split_at(prefix_count);
            for prefix in pre.iter() {
                for suffix in suffixes.iter() {
                    dest.push(format!("{prefix}{suffix}"));
                }
            }
            true
        }
        Class(class) => {
            if let Some(literal) = class.literal() {
                return if let Ok(s) = String::from_utf8(literal.to_vec()) {
                    dest.push(s);
                    true
                } else {
                    false
                };
            }

            match class {
                Unicode(uni) => {
                    for range in uni.iter().flat_map(|r| r.start()..=r.end()) {
                        dest.push(format!("{range}"));
                        if dest.len() > MAX_OR_VALUES {
                            // It is cheaper to use regexp here.
                            return false;
                        }
                    }
                    true
                }
                Bytes(bytes) => {
                    for range in bytes.iter().flat_map(|r| r.start()..=r.end()) {
                        dest.push(format!("{range}"));
                        if dest.len() > MAX_OR_VALUES {
                            return false;
                        }
                    }
                    true
                }
            }
        }
        _ => false,
    }
}

// too_many_matches guards against creating too many set matches.
fn too_many_matches(matches: &[String], added: &[String]) -> bool {
    matches.len() + added.len() > MAX_SET_MATCHES
}

#[cfg(test)]
mod test {
    use super::remove_start_end_anchors;
    use crate::prelude::regex_utils::{find_set_matches, optimize_concat_regex};
    use crate::prelude::string_matcher_from_regex;
    use crate::regex_util::{build_hir, get_or_values, is_dot_plus, is_dot_star};
    use regex_syntax::hir::HirKind;

    #[test]
    fn test_is_dot_star() {
        fn check(s: &str, expected: bool) {
            let sre = build_hir(s).unwrap();
            let got = is_dot_star(&sre);
            assert_eq!(
                got, expected,
                "unexpected is_dot_star for s={:?}; got {:?}; want {:?}",
                s, got, expected
            );
        }

        check(".*", true);
        check(".+", false);
        check("foo.*", false);
        check(".*foo", false);
        check("foo.*bar", false);
        check(".*foo.*", false);
        check(".*foo.*bar", false);
        check(".*foo.*bar.*", false);
        check(".*foo.*bar.*baz", false);
        check(".*foo.*bar.*baz.*", false);
        check(".*foo.*bar.*baz.*qux.*", false);
        check(".*foo.*bar.*baz.*qux.*quux.*quuz.*corge.*grault", false);
        check(".*foo.*bar.*baz.*qux.*quux.*quuz.*corge.*grault.*", false);
    }

    #[test]
    fn test_is_dot_plus() {
        fn check(s: &str, expected: bool) {
            let sre = build_hir(s).unwrap();
            let got = is_dot_plus(&sre);
            assert_eq!(
                got, expected,
                "unexpected is_dot_plus for s={:?}; got {:?}; want {:?}",
                s, got, expected
            );
        }

        check(".*", false);
        check(".+", true);
        check("foo.*", false);
        check(".*foo", false);
        check("foo.*bar", false);
        check(".*foo.*", false);
        check(".*foo.*bar", false);
        check(".*foo.*bar.*", false);
        check(".*foo.*bar.*baz.*qux", false);
        check(".*foo.*bar.*baz.*qux.*", false);
        check(".*foo.*bar.*baz.*qux.*quux.*quuz.*corge.*grault", false);
        check(".*foo.*bar.*baz.*qux.*quux.*quuz.*corge.*grault.*", false);
    }

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
        let (matcher, _) = string_matcher_from_regex(expr).unwrap();
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
            let (matches, _) = find_set_matches(&mut parsed).unwrap();
            assert_eq!(exp_matches, matches);

            // TODO:
            // if exp_case_sensitive {
            //     // When the regexp is case-sensitive, we want to ensure that the
            //     // set matches are maintained in the final matcher.
            //     let r = FastRegexMatcher::new(pattern).unwrap();
            //     assert_eq!(exp_matches, r.set_matches());
            // }
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

    #[test]
    fn test_get_or_values_regex() {
        let test_cases = vec![
            ("", vec![""]),
            ("foo", vec!["foo"]),
            ("^foo$", vec![]),
            ("|foo", vec!["", "foo"]),
            ("|foo|", vec!["", "", "foo"]),
            ("foo.+", vec![]),
            ("foo.*", vec![]),
            (".*", vec![]),
            ("foo|.*", vec![]),
            ("(fo((o)))|(bar)", vec!["bar", "foo"]),
            ("foobar", vec!["foobar"]),
            ("z|x|c", vec!["c", "x", "z"]),
            ("foo|bar", vec!["bar", "foo"]),
            ("(foo|bar)", vec!["bar", "foo"]),
            ("(foo|bar)baz", vec!["barbaz", "foobaz"]),
            ("[a-z][a-z]", vec![]),
            ("[a-d]", vec!["a", "b", "c", "d"]),
            ("x[a-d]we", vec!["xawe", "xbwe", "xcwe", "xdwe"]),
            ("foo(bar|baz)", vec!["foobar", "foobaz"]),
            ("foo(ba[rz]|(xx|o))", vec!["foobar", "foobaz", "fooo", "fooxx"]),
            ("foo(?:bar|baz)x(qwe|rt)", vec!["foobarxqwe", "foobarxrt", "foobazxqwe", "foobazxrt"]),
            ("foo(bar||baz)", vec!["foo", "foobar", "foobaz"]),
            ("(a|b|c)(d|e|f|0|1|2)(g|h|k|x|y|z)", vec![]),
            ("(?i)foo", vec![]),
            ("(?i)(foo|bar)", vec![]),
            ("^foo|bar$", vec![]),
            ("^(foo|bar)$", vec![]),
            ("^a(foo|b(?:a|r))$", vec![]),
            ("^a(foo$|b(?:a$|r))$", vec![]),
            ("^a(^foo|bar$)z$", vec![]),
        ];

        for (s, expected) in test_cases {
            let result = get_or_values(s).unwrap();
            assert_eq!(result, expected, "unexpected values for s={}", s);
        }
    }
}
