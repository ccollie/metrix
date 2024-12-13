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

use super::optimize_concat_regex;
use super::{contains_in_order, ContainsMultiStringMatcher, EqualMultiStringMapMatcher, StringMatchHandler};
use crate::regex_util::hir_utils::{build_hir, is_dot_question, is_end_anchor, is_start_anchor, literal_to_string, matches_any_char, matches_any_character_except_newline};
use regex::{
    Regex,
    Error as RegexError,
};
use regex_syntax::hir::{
    Class,
    Hir,
    HirKind,
    Look,
    Repetition
};

const MAX_SET_MATCHES: usize = 256;

/// The minimum number of alternate values a regex should have to trigger
/// the optimization done by `optimize_equal_or_prefix_string_matchers()` and so use a map
/// to match values instead of iterating over a list.
const MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD: usize = 16;

// #[derive(GetSize)]
#[derive(Clone, Debug)]
pub struct FastRegexMatcher {
    _optimized: bool,
    re_string: String,
    re: Option<Regex>,
    set_matches: Vec<String>,
    pub string_matcher: Option<StringMatchHandler>,
    pub prefix: String,
    pub suffix: String,
    pub contains: Vec<String>,
}

impl FastRegexMatcher {
    pub fn new(v: &str) -> Result<Self, regex::Error> {
        let mut matcher = FastRegexMatcher {
            re_string: v.to_string(),
            re: None,
            set_matches: vec![],
            string_matcher: None,
            prefix: String::new(),
            suffix: String::new(),
            contains: vec![],
            _optimized: false,
        };

        if let Some((string_matcher, set_matches)) = optimize_alternating_literals(v) {
            matcher.string_matcher = Some(string_matcher);
            matcher.set_matches = set_matches;
        } else {
            let mut parsed = build_hir(v)?;
            let re = Regex::new(&format!("^(?s:{v})$"))?;
            matcher.re = Some(re);

            if let HirKind::Concat(hirs) = parsed.kind() {
                let (prefix, suffix, contains, subs) = optimize_concat_regex(&hirs);
                matcher.prefix = prefix;
                matcher.suffix = suffix;
                matcher.contains = contains;
                parsed = Hir::concat(subs);
            }

            if let Some(matches) = find_set_matches(&mut parsed) {
                matcher.set_matches = matches;
            }

            matcher.string_matcher = string_matcher_from_hir(&mut parsed);
            matcher._optimized = matcher.is_optimized();
        }

        Ok(matcher)
    }

    pub fn is_optimized(&self) -> bool {
        !self.set_matches.is_empty()
            || self.string_matcher.is_some()
            || !self.prefix.is_empty()
            || !self.suffix.is_empty()
            || !self.contains.is_empty()
    }

    pub fn set_matches(&self) -> Vec<String> {
        self.set_matches.clone()
    }

    pub fn get_regex_string(&self) -> &str {
        &self.re_string
    }

    pub fn matches(&self, s: &str) -> bool {
        if !self._optimized && self.string_matcher.is_some() {
            return self.string_matcher.as_ref().unwrap().matches(s);
        }
        if !self.set_matches.is_empty() {
            return self.set_matches.iter().any(|match_str| match_str == s);
        }
        if !self.prefix.is_empty() && !s.starts_with(&self.prefix) {
            return false;
        }
        if !self.suffix.is_empty() && !s.ends_with(&self.suffix) {
            return false;
        }
        if !self.contains.is_empty() && !contains_in_order(&s, &self.contains) {
            return false;
        }
        if let Some(matcher) = &self.string_matcher {
            return matcher.matches(&s);
        }
        if let Some(re) = &self.re {
            return re.is_match(&s);
        }
        false
    }
}

fn optimize_alternating_literals(s: &str) -> Option<(StringMatchHandler, Vec<String>)> {
    if s.is_empty() {
        return Some((StringMatchHandler::Empty, Vec::new()));
    }

    let estimated_alternates = s.matches('|').count() + 1;

    if estimated_alternates == 1 {
        if regex::escape(s) == s {
            return Some((StringMatchHandler::Literal(s.to_string()), vec![s.to_string()]));
        }
        return None;
    }

    let mut sub_matches = Vec::with_capacity(estimated_alternates);
    for sub_match in s.split('|') {
        if regex::escape(sub_match) != sub_match {
            return None;
        }
        sub_matches.push(sub_match.to_string());
    }


    let multi = StringMatchHandler::EqualsMulti(sub_matches.clone()); // todo: avoid clone below
    Some((multi, sub_matches))
}


pub(super) fn find_set_matches(hir: &mut Hir) -> Option<Vec<String>> {
    clear_begin_end_text(hir);
    find_set_matches_internal(hir, "")
}

fn find_set_matches_internal(hir: &Hir, base: &str) -> Option<Vec<String>> {
    match hir.kind() {
        HirKind::Look(Look::Start) | HirKind::Look(Look::End) => None,
        HirKind::Literal(_) => {
            let literal = format!("{}{}", base, literal_to_string(hir));
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
                    for urange in ranges.iter().flat_map(|r| r.start()..=r.end()) {
                        matches.push(format!("{base}{urange}"));
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
                if let Some(m) = find_set_matches_internal(hir, b) {
                    if m.is_empty() {
                        return None;
                    }
                    if matches.len() + m.len() > MAX_SET_MATCHES {
                        return None;
                    }
                    new_matches.extend(m);
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
        HirKind::Alternation(hirs) => {
            for sub in hirs.iter() {
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
        },
        _ => return None,
    }

    Some(matches)
}


fn clear_begin_end_text(hir: &mut Hir) {

    fn handle_concat(hirs: &Vec<Hir>) -> Option<Hir> {
        if !hirs.is_empty() {
            let mut start: usize = 0;
            let mut end: usize = hirs.len() - 1;

            if is_start_anchor(&hirs[0]) {
                start += 1;
            }

            if is_end_anchor(&hirs[end]) {
                end -= 1;
            }
            let slice = &hirs[start..=end];
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

fn too_many_matches(matches: &[String], added: &[String]) -> bool {
    matches.len() + added.len() > MAX_SET_MATCHES
}

const EMPTY_SET_MATCHES: Vec<String> = Vec::new();

pub fn string_matcher_from_regex(pattern: &str) -> Result<Option<StringMatchHandler>, RegexError> {
    let mut hir = build_hir(pattern)?;
    Ok(string_matcher_from_hir(&mut hir))
}

pub(super) fn string_matcher_from_hir(hir: &mut Hir) -> Option<StringMatchHandler> {
    clear_begin_end_text(hir);
    let matcher = string_matcher_from_regex_internal(hir);
    if let Some(matcher) = matcher {
        Some(optimize_equal_or_prefix_string_matchers(matcher, MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD))
    } else {
        None
    }
}

fn rep_is_zero_or_one(rep: &Repetition) -> bool {
    rep.min == 0 && rep.max == Some(1)
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

fn is_quantifier(hir: &Hir) -> bool {
    match hir.kind() {
        HirKind::Repetition(rep) => {
            rep_is_zero_or_one(rep) || rep_is_dot_plus(rep) || rep_is_dot_star(rep)
        },
        _ => false,
    }
}


fn string_matcher_from_regex_internal(hir: &Hir) -> Option<StringMatchHandler> {
    // Correctly handling anchors inside a regex is tricky,
    // so in this case we fallback to the regex engine.
    if is_start_anchor(hir) || is_end_anchor(hir) {
        return None;
    }

    fn validate_repetition(rep: &Repetition) -> bool {
        // if re.sub.Op != syntax.OpAnyChar && re.sub.Op != syntax.OpAnyCharNotNL {
        //     return nil
        // }
        matches_any_char(&rep.sub) || matches_any_character_except_newline(&rep.sub)
    }

    match hir.kind() {
        HirKind::Capture(captures) => {
            string_matcher_from_regex_internal(&captures.sub)
        }
        HirKind::Repetition(rep) => {
            // .?
            if rep_is_zero_or_one(rep) {
                if !is_dot_question(&rep.sub) && !validate_repetition(rep) {
                    return None;
                }
                let match_nl = matches_any_char(&rep.sub);
                Some(StringMatchHandler::zero_or_one_chars(match_nl))
            } else if rep_is_dot_plus(rep) {
                if !validate_repetition(rep) {
                    return None;
                }
                let match_nl = matches_any_char(&rep.sub);
                Some(StringMatchHandler::not_empty(match_nl))
            } else if rep_is_dot_star(rep) {
                if !validate_repetition(rep) {
                    return None;
                }
                // If the newline is valid, then this matcher literally match any string (even empty).
                if matches_any_char(&rep.sub) {
                    Some(StringMatchHandler::MatchAll)
                } else {
                    // Any string is fine (including an empty one), as far as it doesn't contain any newline.
                    return Some(StringMatchHandler::AnyWithoutNewline)
                }
            } else {
                None
            }
        }
        HirKind::Empty => Some(StringMatchHandler::Empty),
        HirKind::Literal(_) => {
            Some(StringMatchHandler::Literal(literal_to_string(hir)))
        }
        HirKind::Alternation(hirs ) => {
            let mut or_matchers = Vec::new();
            let mut is_literal = true;
            let mut num_values: usize = 0;
            let mut num_prefixes: usize = 0;
            let mut min_prefix_length: usize = 0;

            for sub_hir in hirs {
                if let Some(matcher) = string_matcher_from_regex_internal(sub_hir) {
                    match &matcher {
                        StringMatchHandler::Literal(_) => {
                            num_values += 1;
                        },
                        StringMatchHandler::EqualsMulti(values) => {
                            num_values += values.len();
                        },
                        StringMatchHandler::StartsWith(prefix) => {
                            num_prefixes += 1;
                            is_literal = false;
                            min_prefix_length = min_prefix_length.min(prefix.len());
                        },
                        StringMatchHandler::Prefix(prefix_matcher) => {
                            num_prefixes += 1;
                            is_literal = false;
                            min_prefix_length = min_prefix_length.min(prefix_matcher.prefix.len());
                        },
                        _ => is_literal = false,
                    }
                    or_matchers.push(Box::new(matcher));
                } else {
                    return None;
                }
            }
            // optimize the case where all the alternatives are literals
            if is_literal {
                if num_values >= MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD {
                    let mut res = EqualMultiStringMapMatcher::new(min_prefix_length);
                    for matcher in or_matchers.into_iter() {
                        match *matcher {
                            StringMatchHandler::Literal(lit) => {
                                res.values.insert(lit);
                            },
                            StringMatchHandler::EqualsMulti(values) => {
                                for value in values {
                                    res.values.insert(value);
                                }
                            },
                            _ => unreachable!(),
                        }
                    }
                    return Some(StringMatchHandler::EqualMultiMap(res));
                }
                let mut values = Vec::with_capacity(num_values);
                for matcher in or_matchers.into_iter() {
                    match *matcher {
                        StringMatchHandler::Literal(lit) => {
                            values.push(lit);
                        },
                        StringMatchHandler::EqualsMulti(_values) => {
                            for value in _values {
                                values.push(value);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                return Some(StringMatchHandler::EqualsMulti(values))
            }
            Some(StringMatchHandler::Or(or_matchers))
        }
        HirKind::Concat(hirs) => {
            if hirs.is_empty() {
                return Some(StringMatchHandler::Empty);
            }
            if hirs.len() == 1 {
                return string_matcher_from_regex_internal(&hirs[0]);
            }

            let mut left = None;
            let mut right = None;

            let first = &hirs[0];
            let mut hirs_new = &hirs[1..];
            if let HirKind::Repetition(rep) = first.kind() {
                if is_quantifier(first) {
                    left = string_matcher_from_regex_internal(&rep.sub);
                    if left.is_none() {
                        return None;
                    }
                    hirs_new = &hirs[1..];
                }
            }

            let last = &hirs[hirs.len() - 1];
            if let HirKind::Repetition(rep) = last.kind() {
                if is_quantifier(last) {
                    right = string_matcher_from_regex_internal(&rep.sub);
                    if right.is_none() {
                        return None;
                    }
                    hirs_new = &hirs_new[0..hirs.len() - 1];
                }
            }

            let hir = Hir::concat(hirs_new.to_vec());
            let matches= find_set_matches_internal(&hir, "")?;

            if matches.is_empty() && hirs.len() == 2 {
                // We have not found fixed set matches. We look for other known cases that
                // we can optimize.
                let first = &hirs[0];
                let second = &hirs[1];
                if let HirKind::Literal(_) = first.kind() {
                    if right.is_none() {
                        right = string_matcher_from_regex_internal(second);
                        if right.is_some() {
                            return Some(StringMatchHandler::prefix(literal_to_string(first), right));
                        }
                    }
                }
                // Suffix is literal.
                if let HirKind::Literal(_chars) = second.kind() {
                    if left.is_none() {
                        left = string_matcher_from_regex_internal(second);
                        if left.is_some() {
                            return Some(StringMatchHandler::prefix(literal_to_string(second), left));
                        }
                    }
                }
            }

            if matches.is_empty() {
                return None;
            }

            if matches.len() == 1 {

            }

            let matcher = ContainsMultiStringMatcher::new(matches, left, right);

            Some(StringMatchHandler::ContainsMulti(matcher))
        }
        _ => None,
    }
}

fn optimize_equal_or_prefix_string_matchers(
    input: StringMatchHandler,
    threshold: usize,
) -> StringMatchHandler {
    let mut num_values = 0;
    let mut num_prefixes = 0;
    let mut min_prefix_length = 0;

    get_equal_or_prefix_string_matchers_counts(&input, &mut num_values, &mut num_prefixes, &mut min_prefix_length);

    if (num_values + num_prefixes) < threshold {
        return input;
    }

    let mut multi_matcher = EqualMultiStringMapMatcher::new(min_prefix_length);

    find_equal_or_prefix_string_matchers(&input, &mut multi_matcher);

    StringMatchHandler::EqualMultiMap(multi_matcher)
}

fn get_equal_or_prefix_string_matchers_counts(
    input: &StringMatchHandler,
    num_values: &mut usize,
    num_prefixes: &mut usize,
    min_prefix_length: &mut usize,
) {
    match input {
        StringMatchHandler::Or(or_matchers) => {
            for matcher in or_matchers {
                get_equal_or_prefix_string_matchers_counts(&matcher, num_values, num_prefixes, min_prefix_length);
            }
        },
        StringMatchHandler::EqualsMulti(values) => {
            *num_values += values.len();
        },
        StringMatchHandler::Literal(_) => {
            *num_values += 1;
        },
        StringMatchHandler::StartsWith(prefix) => {
            *min_prefix_length = *min_prefix_length.min(&mut prefix.len());
            *num_prefixes += 1;
        },
        StringMatchHandler::Prefix(prefix_matcher) => {
            *min_prefix_length = *min_prefix_length.min(&mut prefix_matcher.prefix.len());
            *num_prefixes += 1;
        },
        _ => (),
    }
}

fn find_equal_or_prefix_string_matchers(
    input: &StringMatchHandler,
    res: &mut EqualMultiStringMapMatcher,
) {
    match input {  // TODO: optimize this recursion with tail call optimization.
        StringMatchHandler::Or(or_matchers) => {
            for matcher in or_matchers {
                find_equal_or_prefix_string_matchers(&matcher, res)
            }
        },
        StringMatchHandler::EqualsMulti(values) => {
            for value in values {
                res.values.insert(value.clone());
            }
        },
        StringMatchHandler::Literal(s) => {
            res.values.insert(s.to_string());
        },
        StringMatchHandler::StartsWith(prefix) => {
            res.add_prefix(prefix.to_string(), Box::new(input.clone()));
        },
        StringMatchHandler::Prefix(prefix_matcher) => {
            res.add_prefix(prefix_matcher.prefix.clone(), Box::new(input.clone()));
        },
        _ => (),
    }
}