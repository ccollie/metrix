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

use crate::hash::{FastHashMap, FastHashSet};
use crate::regex_util::hir_utils::{build_hir, is_end_anchor, is_literal, is_start_anchor, literal_to_string};
use crate::regex_util::hir_utils::{is_dot_question, matches_any_char};
use regex::Regex;
use regex_syntax::hir::{Class, Hir, HirKind, Look, Repetition};
use std::any::Any;

const MAX_SET_MATCHES: usize = 256;

/// The minimum number of alternate values a regex should have to trigger
/// the optimization done by `optimize_equal_or_prefix_string_matchers()` and so use a map
/// to match values instead of iterating over a list.
const MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD: usize = 16;

pub trait MultiStringMatcherBuilder: StringMatcher {
    fn add(&mut self, s: String);
    fn add_prefix(&mut self, prefix: String, prefix_case_sensitive: bool, matcher: Box<dyn StringMatcher>);
    fn set_matches(&self) -> Vec<String>;
}

pub struct FastRegexMatcher {
    _optimized: bool,
    re_string: String,
    re: Option<Regex>,
    set_matches: Vec<String>,
    string_matcher: Option<Box<dyn StringMatcher>>,
    prefix: String,
    suffix: String,
    contains: Vec<String>,
    match_string: Box<dyn Fn(&str) -> bool>,
}

impl FastRegexMatcher {
    pub fn new(v: &str) -> Result<Self, regex::Error> {
        let mut matcher = FastRegexMatcher {
            re_string: v.to_string(),
            re: None,
            set_matches: Vec::new(),
            string_matcher: None,
            prefix: String::new(),
            suffix: String::new(),
            contains: Vec::new(),
            match_string: Box::new(|_| false),
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

            matcher.string_matcher = string_matcher_from_regex(&mut parsed);
            matcher.match_string = matcher.compile_match_string_function();
            matcher._optimized = matcher.is_optimized();
        }

        Ok(matcher)
    }

    fn compile_match_string_function(&self) -> Box<dyn Fn(String) -> bool + '_> {
        if self.set_matches.is_empty()
            && self.prefix.is_empty()
            && self.suffix.is_empty()
            && self.contains.is_empty()
            && self.string_matcher.is_some()
        {
            return Box::new(|s| self.string_matcher.as_ref().unwrap().matches(&s));
        }

        Box::new(|s| { self.match_string(&s) })
    }

    pub fn is_optimized(&self) -> bool {
        !self.set_matches.is_empty()
            || self.string_matcher.is_some()
            || !self.prefix.is_empty()
            || !self.suffix.is_empty()
            || !self.contains.is_empty()
    }

    pub fn match_string(&self, s: &str) -> bool {
        (self.match_string)(s)
    }

    pub fn set_matches(&self) -> Vec<String> {
        self.set_matches.clone()
    }

    pub fn get_regex_string(&self) -> &str {
        &self.re_string
    }

    pub fn matches(&self, s: &str) -> bool {
        if !self._optimized && self.string_matcher.is_some() {
            //return self.string_matcher.as_ref().unwrap().is_match(s);
        }
        if !self.set_matches.is_empty() {
            for match_str in &self.set_matches {
                if match_str == &s {
                    return true;
                }
            }
            return false;
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

fn optimize_alternating_literals(s: &str) -> Option<(Box<dyn StringMatcher>, Vec<String>)> {
    if s.is_empty() {
        return Some((Box::new(EmptyStringMatcher {}), Vec::new()));
    }

    let estimated_alternates = s.matches('|').count() + 1;

    if estimated_alternates == 1 {
        if Regex::new(s).is_ok() {
            return Some((
                Box::new(EqualStringMatcher {
                    s: s.to_string(),
                }),
                vec![s.to_string()],
            ));
        }
        return None;
    }

    let mut multi_matcher = EqualMultiStringMatcher::new(estimated_alternates);

    for sub_match in s.split('|') {
        if Regex::new(sub_match).is_err() {
            return None;
        }
        multi_matcher.add(sub_match.to_string());
    }

    let multi = multi_matcher.set_matches().clone(); // todo: avoid clone below
    Some((Box::new(multi_matcher), multi))
}

pub(super) fn optimize_concat_regex(subs: &Vec<Hir>) -> (String, String, Vec<String>, Vec<Hir>) {
    let mut new_subs = subs.clone();

    if new_subs.is_empty() {
        return (String::new(), String::new(), Vec::new(), new_subs);
    }

    if is_start_anchor(&new_subs[0])  {
        new_subs.remove(0);
    }

    if let Some(last) = new_subs.last() {
        if is_end_anchor(&last) {
            new_subs.pop();
        }
    }

    let mut prefix = String::new();
    let mut suffix = String::new();
    let mut contains = Vec::new();

    if let Some(first) = new_subs.first() {
        if is_literal(&first) {
            prefix = literal_to_string(&first);
        }
    }

    if let Some(last) = new_subs.last() {
        if is_literal(last) {
            suffix = literal_to_string(&last);
        }
    }

    for hir in new_subs.iter().skip(1).take(new_subs.len() - 2) {
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

pub trait StringMatcher: Any {
    fn matches(&self, s: &str) -> bool;
    fn as_any(&self) -> &dyn Any;
}

pub struct EmptyStringMatcher;

impl StringMatcher for EmptyStringMatcher {
    fn matches(&self, s: &str) -> bool {
        s.is_empty()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(super) struct OrStringMatcher {
    matchers: Vec<Box<dyn StringMatcher>>,
}

impl StringMatcher for OrStringMatcher {
    fn matches(&self, s: &str) -> bool {
        self.matchers.iter().any(|matcher| matcher.matches(s))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct EqualStringMatcher {
    s: String,
}

impl StringMatcher for EqualStringMatcher {
    fn matches(&self, s: &str) -> bool {
        self.s == s
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

const EMPTY_SET_MATCHES: Vec<String> = Vec::new();

pub struct EqualMultiStringMatcher {
    values: Vec<String>,
}

impl EqualMultiStringMatcher {
    fn new(estimated_size: usize) -> Self {
        Self {
            values: Vec::with_capacity(estimated_size),
        }
    }

    fn add(&mut self, s: String) {
        self.values.push(s);
    }

    pub fn set_matches(&self) -> &Vec<String> {
        &self.values
    }
}

impl StringMatcher for EqualMultiStringMatcher {
    fn matches(&self, s: &str) -> bool {
        self.values.iter().any(|v| v == s)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct EqualMultiStringMapMatcher {
    values: FastHashSet<String>,
    prefixes: FastHashMap<String, Vec<Box<dyn StringMatcher>>>,
    min_prefix_len: usize,
}

impl EqualMultiStringMapMatcher {
    fn add(&mut self, s: String) {
        self.values.insert(s);
    }

    fn add_prefix(&mut self, prefix: String, matcher: Box<dyn StringMatcher>) {
        if self.min_prefix_len == 0 {
            panic!("add_prefix called when no prefix length defined");
        }
        if prefix.len() < self.min_prefix_len {
            panic!("add_prefix called with a too short prefix");
        }

        let s = get_prefix(&prefix, self.min_prefix_len);

        self.prefixes
            .entry(s)
            .or_insert_with(Vec::new)
            .push(matcher);
    }

    fn set_matches(&self) -> Vec<String> {
        if self.values.len() >= MAX_SET_MATCHES || !self.prefixes.is_empty() {
            return Vec::new();
        }

        self.values.iter().collect()
    }
}

impl StringMatcher for EqualMultiStringMapMatcher {
    fn matches(&self, s: &str) -> bool {
        if !self.values.is_empty() {
            if self.values.contains(&s) {
                return true;
            }
        }

        if self.min_prefix_len > 0 && s.len() >= self.min_prefix_len {
            let prefix = &s[..self.min_prefix_len];
            if let Some(matchers) = self.prefixes.get(prefix) {
                for matcher in matchers {
                    if matcher.matches(s) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn get_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

impl MultiStringMatcherBuilder for EqualMultiStringMapMatcher {
    fn add(&mut self, s: String) {
        self.add(s);
    }

    fn add_prefix(&mut self, prefix: String, prefix_case_sensitive: bool, matcher: Box<dyn StringMatcher>) {
        todo!()
    }

    fn set_matches(&self) -> Vec<String> {
        self.set_matches()
    }
}

pub struct ContainsStringMatcher {
    substrings: Vec<String>,
    left: Option<Box<dyn StringMatcher>>,
    right: Option<Box<dyn StringMatcher>>,
}

impl StringMatcher for ContainsStringMatcher {
    fn matches(&self, s: &str) -> bool {
        for substr in &self.substrings {
            match (self.left.as_ref(), self.right.as_ref()) {
                (Some(left), Some(right)) => {
                    let mut search_start_pos = 0;
                    while let Some(pos) = s[search_start_pos..].find(substr) {
                        let pos = pos + search_start_pos;
                        if left.matches(&s[..pos]) && right.matches(&s[pos + substr.len()..]) {
                            return true;
                        }
                        search_start_pos = pos + 1;
                    }
                }
                (Some(left), None) => {
                    if s.ends_with(substr) && left.matches(&s[..s.len() - substr.len()]) {
                        return true;
                    }
                }
                (None, Some(right)) => {
                    if s.starts_with(substr) && right.matches(&s[substr.len()..]) {
                        return true;
                    }
                }
                (None, None) => {
                    if s.contains(substr) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct LiteralPrefixStringMatcher {
    prefix: String,
    right: Option<Box<dyn StringMatcher>>,
}

impl StringMatcher for LiteralPrefixStringMatcher {
    fn matches(&self, s: &str) -> bool {
        if !s.starts_with(&self.prefix) {
            return false;
        }
        if let Some(right) = &self.right {
            right.matches(&s[self.prefix.len()..])
        } else {
            true
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct LiteralSuffixStringMatcher {
    left: Option<Box<dyn StringMatcher>>,
    suffix: String,
}

impl StringMatcher for LiteralSuffixStringMatcher {
    fn matches(&self, s: &str) -> bool {
        if !s.ends_with(&self.suffix) {
            return false;
        }
        if let Some(left) = &self.left {
            left.matches(&s[..s.len() - self.suffix.len()])
        } else {
            true
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct AnyStringWithoutNewlineMatcher;

impl StringMatcher for AnyStringWithoutNewlineMatcher {
    fn matches(&self, s: &str) -> bool {
        !s.contains('\n')
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct AnyNonEmptyStringMatcher {
    match_nl: bool,
}

impl StringMatcher for AnyNonEmptyStringMatcher {
    fn matches(&self, s: &str) -> bool {
        if self.match_nl {
            !s.is_empty()
        } else {
            !s.is_empty() && !s.contains('\n')
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct ZeroOrOneCharacterStringMatcher {
    match_nl: bool,
}

impl StringMatcher for ZeroOrOneCharacterStringMatcher {
    fn matches(&self, s: &str) -> bool {
        if self.match_nl {
            s.is_empty() || s.chars().count() == 1
        } else {
            s.is_empty() || (s.chars().count() == 1 && s.chars().next().unwrap() != '\n')
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct TrueMatcher;

impl StringMatcher for TrueMatcher {
    fn matches(&self, _s: &str) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn string_matcher_from_regex(hir: &mut Hir) -> Option<Box<dyn StringMatcher>> {
    clear_begin_end_text(hir);
    let matcher = string_matcher_from_regex_internal(hir);
    optimize_equal_or_prefix_string_matchers(matcher, MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD)
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

fn matches_any_character_except_newline(hir: &Hir) -> bool {
    match hir.kind() {
        HirKind::Literal(lit) => {
            // Check if the literal is not a newline
            !lit.0.contains(&b'\n')
        },
        HirKind::Class(class) => {
            match class {
                // Check if the class does not include newline
                Class::Unicode(class) => {
                    let nl = '\n';
                    class.ranges().iter()
                        .all(|range| !(range.start() .. range.end()).contains(&nl))
                },
                Class::Bytes(class) => {
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

fn string_matcher_from_regex_internal(hir: &Hir) -> Option<Box<dyn StringMatcher>> {
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
                Some(Box::new(ZeroOrOneCharacterStringMatcher { match_nl }))
            } else if rep_is_dot_plus(rep) {
                if !validate_repetition(rep) {
                    return None;
                }
                let match_nl = matches_any_char(&rep.sub);
                Some(Box::new(AnyNonEmptyStringMatcher { match_nl }))
            } else if rep_is_dot_star(rep) {
                if !validate_repetition(rep) {
                    return None;
                }
                // If the newline is valid, then this matcher literally match any string (even empty).
                if matches_any_char(&rep.sub) {
                    Some(Box::new(TrueMatcher))
                } else {
                    // Any string is fine (including an empty one), as far as it doesn't contain any newline.
                    return Some(Box::new(AnyStringWithoutNewlineMatcher{}))
                }
            } else {
                None
            }
        }
        HirKind::Empty => Some(Box::new(EmptyStringMatcher)),
        HirKind::Literal(_) => {
            Some(Box::new(EqualStringMatcher { s: literal_to_string(hir) }))
        }
        HirKind::Alternation(hirs ) => {
            let mut or_matchers = Vec::new();
            for sub_hir in hirs {
                if let Some(matcher) = string_matcher_from_regex_internal(sub_hir) {
                    or_matchers.push(matcher);
                } else {
                    return None;
                }
            }
            Some(Box::new(OrStringMatcher { matchers: or_matchers }))
        }
        HirKind::Concat(hirs) => {
            if hirs.is_empty() {
                return Some(Box::new(EmptyStringMatcher));
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
                            return Some(Box::new(LiteralPrefixStringMatcher {
                                prefix: literal_to_string(first),
                                right,
                            }));
                        }
                    }
                }
                // Suffix is literal.
                if let HirKind::Literal(_chars) = second.kind() {
                    if left.is_none() {
                        left = string_matcher_from_regex_internal(second);
                        if left.is_some() {
                            return Some(Box::new(LiteralSuffixStringMatcher {
                                left,
                                suffix: literal_to_string(second),
                            }));
                        }
                    }
                }
            }

            if matches.is_empty() {
                return None;
            }

            Some(Box::new(ContainsStringMatcher {
                substrings: matches,
                left,
                right,
            }))
        }
        _ => None,
    }
}

fn optimize_equal_or_prefix_string_matchers(
    input: Option<Box<dyn StringMatcher>>,
    threshold: usize,
) -> Option<Box<dyn StringMatcher>> {
    let mut num_values = 0;
    let mut num_prefixes = 0;
    let mut min_prefix_length = 0;

    let analyse_prefix_matcher_callback = |prefix: &str, _matcher: &dyn StringMatcher| -> bool {
        if num_prefixes == 0 || prefix.len() < min_prefix_length {
            min_prefix_length = prefix.len();
        }
        num_prefixes += 1;
        true
    };

    if !find_equal_or_prefix_string_matchers(&input,
                                             analyse_equal_matcher_callback,
                                             analyse_prefix_matcher_callback) {
        return input;
    }

    if (num_values + num_prefixes) < threshold {
        return input;
    }

    let mut multi_matcher = EqualMultiStringMatcher::new(num_values);

    let add_equal_matcher_callback = |matcher: &EqualStringMatcher| {
        multi_matcher.add(matcher.s.clone());
    };

    let add_prefix_matcher_callback = |prefix: &str, prefix_case_sensitive: bool, matcher: &dyn StringMatcher| {
        multi_matcher.add_prefix(prefix.to_string(), prefix_case_sensitive, Box::new(matcher.clone()));
    };

    find_equal_or_prefix_string_matchers(&input,
                                         add_equal_matcher_callback,
                                         add_prefix_matcher_callback);

    Some(Box::new(multi_matcher))
}

fn find_equal_or_prefix_string_matchers(
    input: &impl StringMatcher,
    equal_matcher_callback: fn(&EqualStringMatcher) -> bool,
    prefix_matcher_callback: fn(&str, bool, &dyn StringMatcher) -> bool,
) -> bool {
    if let Some(or_matcher) = input.as_any().downcast_ref::<OrStringMatcher>() {
        for matcher in &or_matcher.matchers {
            if !find_equal_or_prefix_string_matchers(&matcher, equal_matcher_callback, prefix_matcher_callback) {
                return false;
            }
        }
        return true;
    }

    if let Some(equal_matcher) = input.as_any().downcast_ref::<EqualStringMatcher>() {
        return equal_matcher_callback(equal_matcher);
    }

    if let Some(prefix_matcher) = input.as_any().downcast_ref::<LiteralPrefixStringMatcher>() {
        return prefix_matcher_callback(&prefix_matcher.prefix, true, prefix_matcher);
    }

    false
}

fn has_prefix_case_insensitive(s: &str, prefix: &str) -> bool {
    s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix)
}

fn has_suffix_case_insensitive(s: &str, suffix: &str) -> bool {
    s.len() >= suffix.len() && s[s.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
}

fn has_suffix(s: &str, suffix: &str) -> bool {
    s.len() >= suffix.len() && s.ends_with(suffix)
}

fn contains_in_order(s: &str, contains: &[String]) -> bool {
    if contains.len() == 1 {
        return s.contains(&contains[0]);
    }
    contains_in_order_multi(s, contains)
}

fn contains_in_order_multi(s: &str, contains: &[String]) -> bool {
    let mut offset = 0;
    for substr in contains {
        if let Some(pos) = s[offset..].find(substr) {
            offset += pos + substr.len();
        } else {
            return false;
        }
    }
    true
}