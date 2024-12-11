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
use regex_syntax::hir::{self, Hir, Look};
use regex_syntax::hir::Class::{Bytes, Unicode};
use regex_syntax::Parser;
use crate::regex_util::hir_utils::{is_dot_star, is_empty_match, is_end_anchor, is_start_anchor};

// RemoveStartEndAnchors removes '^' at the start of expr and '$' at the end of the expr.
pub fn remove_start_end_anchors(expr: &str) -> String {
    let mut expr = expr.to_string();
    while expr.starts_with('^') {
        expr.remove(0);
    }
    while expr.ends_with('$') && !expr.ends_with("\\$") {
        expr.pop();
    }
    expr
}

// `get_or_values_regex` returns "or" values from the given regexp expr.
pub fn get_or_values_regex(expr: &str) -> Vec<String> {
    get_or_values_regex_internal(expr, true)
}

// `get_or_values_prom_regex` returns "or" values from the given Prometheus-like regexp expr.
pub fn get_or_values_prom_regex(expr: &str) -> Vec<String> {
    let expr = remove_start_end_anchors(expr);
    get_or_values_regex_internal(&expr, false)
}

fn get_or_values_regex_internal(expr: &str, keep_anchors: bool) -> Vec<String> {
    let (prefix, tail_expr) = simplify_regex_internal(expr, keep_anchors);
    if tail_expr.is_empty() {
        return vec![prefix];
    }
    let hir = match parse_regex(&tail_expr) {
        Ok(hir) => hir,
        Err(_) => return vec![],
    };
    let or_values = get_or_values(&hir);
    if !prefix.is_empty() {
        return or_values.into_iter().map(|v| format!("{}{}", prefix, v)).collect();
    }
    or_values
}

fn get_or_values(hir: &Hir) -> Vec<String> {
    use hir::HirKind::*;
    match hir.kind() {
        Capture(sub_hir) => get_or_values(&sub_hir.sub),
        Literal(lit) => vec![
            String::from_utf8(lit.0.to_vec()).unwrap_or_default()
        ],
        Empty => vec![String::new()],
        Alternation(hirs) => {
            let mut all_values = Vec::new();
            for sub_hir in hirs {
                let sub_values = get_or_values(sub_hir);
                if sub_values.is_empty() {
                    return vec![];
                }
                all_values.extend(sub_values);
                if all_values.len() > MAX_OR_VALUES {
                    return vec![];
                }
            }
            all_values
        }
        Class(class) => {
            let mut dest = Vec::new();
            if let Some(literal) = class.literal() {
                if let Ok(s) = String::from_utf8(literal.to_vec()) {
                    dest.push(s);
                };
                return dest;
            }

            match class {
                Unicode(uni) => {
                    for urange in uni.iter().flat_map(|r| r.start()..=r.end()) {
                        dest.push(format!("{urange}"));
                        if dest.len() > MAX_OR_VALUES {
                            // It is cheaper to use regexp here.
                            return vec![];
                        }
                    }
                    dest
                }
                Bytes(bytes) => {
                    for range in bytes.iter().flat_map(|r| r.start()..=r.end()) {
                        dest.push(format!("{range}"));
                        if dest.len() > MAX_OR_VALUES {
                            return vec![];
                        }
                    }
                    dest
                }
            }
        }
        Concat(hirs ) => {
            if hirs.is_empty() {
                return vec![String::new()];
            }
            let prefixes = get_or_values(&hirs[0]);
            if prefixes.is_empty() {
                return vec![];
            }
            if hirs.len() == 1 {
                return prefixes;
            }
            let suffixes = get_or_values(&Hir::concat(hirs[1..].to_vec()));
            if suffixes.is_empty() {
                return vec![];
            }
            if prefixes.len() * suffixes.len() > MAX_OR_VALUES {
                return vec![];
            }
            let mut values = Vec::new();
            for prefix in prefixes {
                for suffix in &suffixes {
                    values.push(format!("{}{}", prefix, suffix));
                }
            }
            values
        }
        _ => vec![],
    }
}

const MAX_OR_VALUES: usize = 100;

/// simplifies the given regexp expr.
///
/// It returns plaintext prefix and the remaining regular expression without capturing parens.
pub fn simplify_regex(expr: &str) -> (String, String) {
    let (prefix, suffix) = simplify_regex_internal(expr, true);
    let hir = must_parse_regex(&suffix);
    if is_dot_star(&hir) {
        return (prefix, String::new());
    }
    if let hir::HirKind::Concat(hirs ) = hir.kind() {
        let mut subs = hirs.clone();
        if prefix.is_empty() {
            while !subs.is_empty() && is_dot_star(&subs[0]) {
                subs.remove(0);
            }
        }
        while !subs.is_empty() && is_dot_star(&subs[subs.len() - 1]) {
            subs.pop();
        }
        if subs.is_empty() {
            return (prefix, String::new());
        }
        return (prefix, Hir::concat(subs).to_string());
    }
    (prefix, suffix)
}

// SimplifyPromRegex simplifies the given Prometheus-like expr.
pub fn simplify_prom_regex(expr: &str) -> (String, String) {
    simplify_regex_internal(expr, false)
}

fn simplify_regex_internal(expr: &str, keep_anchors: bool) -> (String, String) {
    let hir = match parse_regex(expr) {
        Ok(hir) => hir,
        Err(_) => return (expr.to_string(), String::new()),
    };
    let hir = simplify_regex_ext(&hir, keep_anchors, keep_anchors);
    if is_empty_match(&hir) {
        return (String::new(), String::new());
    }
    let prefix = match hir.kind() {
        hir::HirKind::Concat(hirs ) => {
            if hirs.is_empty() {
                String::new()
            } else {
                match hirs[0].kind() {
                    hir::HirKind::Literal(lit ) => {
                        String::from_utf8(lit.0.to_vec()).unwrap_or_default()
                    },
                    _ => String::new(),
                }
            }
        }
        _ => String::new(),
    };
    let suffix = hir.to_string();
    (prefix, suffix)
}

fn simplify_regex_ext(hir: &Hir, keep_begin_op: bool, keep_end_op: bool) -> Hir {
    let mut hir = hir.clone();
    loop {
        hir = simplify_regex_ext_internal(&hir, keep_begin_op, keep_end_op);
        if !keep_begin_op && matches!(hir.kind(), hir::HirKind::Look(Look::Start)) {
            return Hir::empty();
        }
        if !keep_end_op && matches!(hir.kind(), hir::HirKind::Look(Look::End)) {
            return Hir::empty();
        }
        let new_hir = hir.to_string();
        if new_hir == hir.to_string() {
            return hir;
        }
        hir = must_parse_regex(&new_hir);
    }
}

fn simplify_regex_ext_internal(hir: &Hir, keep_begin_op: bool, keep_end_op: bool) -> Hir {
    match hir.kind() {
        hir::HirKind::Capture(capture) => {
            simplify_regex_ext_internal(&capture.sub, keep_begin_op, keep_end_op)
        }
        hir::HirKind::Alternation(hirs) => {
            let mut new_hirs = Vec::new();
            for sub_hir in hirs {
                let sub_hir = simplify_regex_ext_internal(sub_hir, keep_begin_op, keep_end_op);
                if is_empty_match(&sub_hir) {
                    new_hirs.push(sub_hir);
                }
            }
            Hir::alternation(new_hirs)
        }
        hir::HirKind::Concat( hirs ) => {
            let mut new_hirs = Vec::new();
            for (i, sub_hir) in hirs.iter().enumerate() {
                let sub_hir = simplify_regex_ext_internal(sub_hir, keep_begin_op || !new_hirs.is_empty(), keep_end_op || i + 1 < hirs.len());
                if is_empty_match(&sub_hir) {
                    new_hirs.push(sub_hir);
                }
            }
            if !keep_begin_op {
                while !new_hirs.is_empty() && is_start_anchor(&new_hirs[0]) {
                    new_hirs.remove(0);
                }
            }
            if !keep_end_op {
                while !new_hirs.is_empty() && is_end_anchor(&new_hirs[new_hirs.len() - 1]) {
                    new_hirs.pop();
                }
            }
            if new_hirs.is_empty() {
                return Hir::empty();
            }
            if new_hirs.len() == 1 {
                return new_hirs[0].clone();
            }
            Hir::concat(new_hirs)
        }
        hir::HirKind::Empty => Hir::empty(),
        _ => hir.clone(),
    }
}


fn parse_regex(expr: &str) -> Result<Hir, regex_syntax::Error> {
    Parser::new().parse(expr)
}

fn must_parse_regex(expr: &str) -> Hir {
    parse_regex(expr).unwrap_or_else(|e| panic!("BUG: cannot parse already verified regexp {}: {}", expr, e))
}