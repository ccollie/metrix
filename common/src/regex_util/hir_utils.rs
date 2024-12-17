use regex::Error as RegexError;
use regex_syntax::hir::Class::{Bytes, Unicode};
use regex_syntax::hir::Dot;
use regex_syntax::{hir::{
    Class,
    Hir,
    HirKind,
    Look
}, parse as parse_regex};
use std::sync::LazyLock;

const ANY_CHAR_EXCEPT_LF: LazyLock<Hir> = LazyLock::new(|| Hir::dot(Dot::AnyCharExceptLF));
const ANY_CHAR: LazyLock<Hir> = LazyLock::new(|| Hir::dot(Dot::AnyChar));

// Beyond this, it's better to use regexp.
const MAX_OR_VALUES: usize = 32;

pub fn hir_to_string(sre: &Hir) -> String {
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

pub fn is_empty_regexp(sre: &Hir) -> bool {
    matches!(sre.kind(), HirKind::Empty)
}

pub fn is_dot_star(sre: &Hir) -> bool {
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

pub fn is_dot_plus(sre: &Hir) -> bool {
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

pub fn is_literal(sre: &Hir) -> bool {
    match sre.kind() {
        HirKind::Literal(_) => true,
        HirKind::Capture(cap) => is_literal(cap.sub.as_ref()),
        _ => false,
    }
}

pub fn is_empty_class(class: &Class) -> bool {
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

pub fn is_dot_question(sre: &Hir) -> bool {
    if let HirKind::Repetition(repetition) = sre.kind() {
        return !repetition.greedy
            && repetition.min == 0
            && repetition.max == Some(1)
            && (ANY_CHAR.eq(&repetition.sub) || ANY_CHAR_EXCEPT_LF.eq(&repetition.sub));
    }
    false
}

pub fn matches_any_char(hir: &Hir) -> bool {
    if let HirKind::Class(class) = hir.kind() {
        return is_empty_class(class)
    }
    false
}

pub fn is_anchor(sre: &Hir, look: Look) -> bool {
    matches!(sre.kind(), HirKind::Look(l) if look == *l)
}

pub fn is_start_anchor(sre: &Hir) -> bool {
    is_anchor(sre, Look::Start)
}

pub fn is_end_anchor(sre: &Hir) -> bool {
    is_anchor(sre, Look::End)
}

pub fn literal_to_string(sre: &Hir) -> String {
    if let HirKind::Literal(lit) = sre.kind() {
        return String::from_utf8(lit.0.to_vec()).unwrap_or_default();
    }
    "".to_string()
}

/// Checks if the given Hir enum variant represents an empty match.
///
/// # Arguments
///
/// * `hir` - A reference to a Hir instance.
///
/// # Returns
///
/// Returns true if the Hir variant is an empty match, false otherwise.
pub fn is_empty_match(hir: &Hir) -> bool {
    // Use the is_empty method on the kind of Hir
    matches!(hir.kind(), HirKind::Empty)
}


pub fn get_literal(sre: &Hir) -> Option<String> {
    match sre.kind() {
        HirKind::Literal(lit) => {
            let s = String::from_utf8(lit.0.to_vec()).unwrap_or_default();
            Some(s)
        }
        _ => None,
    }
}

pub fn build_hir(pattern: &str) -> Result<Hir, RegexError> {
    parse_regex(pattern).map_err(|err| RegexError::Syntax(err.to_string()))
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

#[cfg(test)]
mod tests {
    use super::*;

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
