use super::hir_utils::*;
use super::match_handlers::{StringMatchHandler, StringMatchOptions};
use crate::regex_util::{EqualMultiStringMapMatcher, Quantifier};
use regex::{Error as RegexError, Regex};
use regex_syntax::hir::{Class, Hir, HirKind, Look, Repetition};
use regex_syntax::parse as parse_regex;
use crate::prelude::{RegexMatcher, RepetitionMatcher};

const MAX_SET_MATCHES: usize = 256;

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
    fn create_re_match_fn(expr: &str, sre: &Hir) -> Result<(StringMatchHandler, usize), RegexError> {
        let re_match = handle_regex(expr, sre)?;
        Ok((re_match, RE_MATCH_COST))
    }

    if expr.is_empty() {
        return Ok((StringMatchHandler::Empty, EMPTY_MATCH_COST))
    }

    if expr == ".*" {
        return Ok((StringMatchHandler::MatchAll, FULL_MATCH_COST));
    }

    if expr == ".+" {
        return Ok((StringMatchHandler::not_empty(false), FULL_MATCH_COST));
    }

    if let Some(string_matcher) = optimize_alternating_literals(expr) {
        let cost = calc_match_cost(&string_matcher);
        return Ok((string_matcher, cost));
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
    create_re_match_fn(expr, &sre)
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
        HirKind::Empty => Ok(Some((StringMatchHandler::Empty, EMPTY_MATCH_COST))),
        HirKind::Repetition(rep) => {
            if let Some(matcher) = get_repetition_matcher(sre, rep) {
                let cost = calc_match_cost(&matcher);
                return Ok(Some((matcher, cost)));
            }
            Ok(None)
        }
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
                        let cost =  calc_match_cost(&matcher);
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

fn get_repetition_matcher(hir: &Hir, rep: &Repetition) -> Option<StringMatchHandler> {
    fn validate_repetition(rep: &Repetition) -> bool {
        // if re.sub.Op != syntax.OpAnyChar && re.sub.Op != syntax.OpAnyCharNotNL {
        //     return nil
        // }
        matches_any_char(&rep.sub) || matches_any_character_except_newline(&rep.sub)
    }

    // .?
    if is_dot_question(hir) {
        let match_nl = matches_any_char(&rep.sub);
        Some(StringMatchHandler::zero_or_one_chars(match_nl))
    } else if rep_is_dot_plus(rep) {
        let matches_any = matches_any_char(&rep.sub);
        let matches_non_newline = matches_any_character_except_newline(&rep.sub);
        if !matches_any && !matches_non_newline {
            return None;
        }
        Some(StringMatchHandler::not_empty(matches_any))
    } else if rep_is_dot_star(rep) {
        if !validate_repetition(rep) {
            return None;
        } else if matches_any_char(&rep.sub) {
            // If the newline is valid, then this matcher literally match any string (even empty).
            Some(StringMatchHandler::MatchAll)
        } else {
            // Any string is fine (including an empty one), as far as it doesn't contain any newline.
            Some(StringMatchHandler::AnyWithoutNewline)
        }
    } else if is_literal(&rep.sub) {
        let literal = literal_to_string(&rep.sub);
        let repetition = RepetitionMatcher::new(literal, rep.min, rep.max);
        return Some(StringMatchHandler::Repetition(repetition));
    } else {
        None
    }
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

pub(super) fn optimize_alternating_literals(s: &str) -> Option<StringMatchHandler> {
    if s.is_empty() {
        return Some(StringMatchHandler::Empty);
    }

    let estimated_alternates = s.matches('|').count() + 1;

    if estimated_alternates == 1 {
        if regex::escape(s) == s {
            return Some(StringMatchHandler::Literal(s.to_string()));
        }
        return None;
    }

    let use_map = estimated_alternates >= MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD;
    if use_map {
        let mut map = EqualMultiStringMapMatcher::new(0);
        for sub_match in s.split('|') {
            if regex::escape(sub_match) != sub_match {
                return None;
            }
            map.values.insert(sub_match.to_string());
        }

        Some(StringMatchHandler::EqualMultiMap(map))
    } else {
        let mut sub_matches = Vec::with_capacity(estimated_alternates);
        for sub_match in s.split('|') {
            if regex::escape(sub_match) != sub_match {
                return None;
            }
            sub_matches.push(sub_match.to_string());
        }

        let multi = StringMatchHandler::EqualsMulti(sub_matches);
        Some(multi)
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

pub(super) fn find_set_matches(hir: &mut Hir) -> Option<Vec<String>> {
    clear_begin_end_text(hir);
    find_set_matches_internal(hir, "")
}

pub(super) fn find_set_matches_internal(hir: &Hir, base: &str) -> Option<Vec<String>> {
    match hir.kind() {
        HirKind::Look(Look::Start) | HirKind::Look(Look::End) => None,
        HirKind::Literal(_) => {
            let literal = format!("{}{}", base, crate::regex_util::hir_utils::literal_to_string(hir));
            Some(vec![literal])
        },
        HirKind::Empty => {
            if !base.is_empty() {
                Some(vec![base.to_string()])
            } else {
                None
            }
        }
        HirKind::Alternation(_) => find_set_matches_from_alternate(hir, base),
        HirKind::Capture(hir) => {
            find_set_matches_internal(&hir.sub, base)
        }
        HirKind::Concat(_) => find_set_matches_from_concat(hir, base),
        HirKind::Class(class) => {
            match class {
                Class::Unicode(ranges) => {
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

                    Some(matches)
                }
                Class::Bytes(ranges) => {
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

                    Some(matches)
                }
            }
        }
        _ => None,
    }
}

fn find_set_matches_from_concat(hir: &Hir, base: &str) -> Option<Vec<String>> {

    if let HirKind::Concat(hirs) = hir.kind() {
        let mut matches = vec![base.to_string()];

        for hir in hirs.iter() {
            let mut new_matches = Vec::new();
            for b in &matches {
                if let Some(items) = find_set_matches_internal(hir, b) {
                    if matches.len() + items.len() > MAX_SET_MATCHES {
                        return None;
                    }
                    new_matches.extend(items);
                } else {
                    return None;
                }
            }
            matches = new_matches;
        }

        return Some(matches)
    }

    None
}

fn find_set_matches_from_alternate(hir: &Hir, base: &str) -> Option<Vec<String>> {
    let mut matches = Vec::new();

    match hir.kind() {
        HirKind::Alternation(alternates) => {
            for sub in alternates.iter() {
                if let Some(found) = find_set_matches_internal(sub, base) {
                    if found.is_empty() {
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

    Some(matches)
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
        if let Some(sub_matches) = find_set_matches_internal(&sub, "") {
            matches = sub_matches;
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
        if let Some(sub_matches) = find_set_matches_internal(hir, "") {
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


fn calc_match_cost(matcher: &StringMatchHandler) -> usize {
    match matcher {
        StringMatchHandler::MatchAll => FULL_MATCH_COST,
        StringMatchHandler::NotEmpty(_) => FULL_MATCH_COST,
        StringMatchHandler::Literal(_) => LITERAL_MATCH_COST,
        StringMatchHandler::StartsWith(_) => PREFIX_MATCH_COST,
        StringMatchHandler::EndsWith(_) => SUFFIX_MATCH_COST,
        StringMatchHandler::Contains(_) => MIDDLE_MATCH_COST,
        StringMatchHandler::OrderedAlternates(m) => m.len() * MIDDLE_MATCH_COST,
        StringMatchHandler::EqualsMulti(m) => m.len() * LITERAL_MATCH_COST,
        StringMatchHandler::EqualMultiMap(_) => LITERAL_MATCH_COST,
        StringMatchHandler::Alternates(m) => {
            m.alts.len() * LITERAL_MATCH_COST
        },
        StringMatchHandler::Regex(_) => RE_MATCH_COST,
        _ => todo!(),
        StringMatchHandler::MatchNone => 0,
        StringMatchHandler::Empty => FULL_MATCH_COST,
        StringMatchHandler::AnyWithoutNewline => FULL_MATCH_COST,
        StringMatchHandler::ContainsMulti(m) => {
            let mut base = m.substrings.len() * MIDDLE_MATCH_COST;
            if let Some(l) = &m.left {
                base += calc_match_cost(l);
            }
            if let Some(r) = &m.right {
                base += calc_match_cost(r);
            }
            base
        }
        StringMatchHandler::Prefix(_) => PREFIX_MATCH_COST,
        StringMatchHandler::Suffix(_) => SUFFIX_MATCH_COST,
        StringMatchHandler::Repetition(r) => {
            LITERAL_MATCH_COST * r.min as usize // ??
        }
        StringMatchHandler::MatchFn(_) => FN_MATCH_COST,
        StringMatchHandler::And(a, b) => {
            calc_match_cost(a) + calc_match_cost(b)
        }
        StringMatchHandler::Or(items) => {
            items.iter().map(|x| calc_match_cost(x)).sum()
        }
        StringMatchHandler::ZeroOrOneChars(_) => LITERAL_MATCH_COST
    }
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
