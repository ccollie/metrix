use super::tokens::Token;
use super::parse_error::unexpected;
use crate::parser::{extract_string_value, ParseErr, ParseError, ParseResult};
use logos::{Lexer, Logos};
use metricsql_common::label::Label;

/// specialized parser for a Prometheus compatible metric name (as opposed to a metric selector).
///
///    <label_set>
///    <metric_identifier> [<label_set>]
///
pub fn parse_metric_name(s: &str) -> ParseResult<Vec<Label>> {
    let mut lex = Token::lexer(s);

    let mut labels: Vec<Label> = Vec::new();

    let measurement = expect_token(&mut lex, &Token::Identifier)?;
    labels.push(Label::new("__name__".to_string(), measurement.to_string()));
    let (tok, _) = get_next(&mut lex)?;
    if tok == Token::Eof {
        return Ok(labels);
    }
    if tok == Token::LeftBrace {
        parse_label_filters(&mut lex, &mut labels)?;
    }
    Ok(labels)
}

/// parse a set of label matchers.
///
/// '{' [ <label_name> <match_op> <match_string>, ... '}'
///
fn parse_label_filters(lex: &mut Lexer<Token>, labels: &mut Vec<Label>) -> ParseResult<()> {
    use Token::*;

    let mut was_ident = false;
    let save_count = labels.len();
    loop {
        let (tok, name) = get_next(lex)?;
        match tok {
            Identifier => {
                let name = name.to_string();
                let _ = expect_token(lex, &Equal)?;
                let value = expect_token(lex, &StringLiteral)?;
                let contents = extract_string_value(value)?;
                labels.push(Label::new(name, contents.to_string()));
                was_ident = true;
            },
            Comma => {
                if !was_ident {
                    return Err(unexpected("metric name", name, "identifier", None));
                }
                was_ident = false;
                continue
            },
            RightBrace => {
                if !was_ident {
                    let count = labels.len() - save_count;
                    if count > 0 {
                        return Err(unexpected("metric name", name, "identifier", None));
                    }
                }
                break
            },
            _ => return Err(unexpected("metric name label", name, ", or }", None)),
        }
    }
    // make sure we're at eof

    Ok(())
}


fn get_next<'a>(lex: &'a mut Lexer<Token>) -> ParseResult<(Token, &'a str)> {
    match lex.next() {
        Some(Ok(tok)) => Ok((tok, lex.slice())),
        Some(Err(_)) => {
            let span = lex.span();
            let inner = ParseErr::new(
                format!("unexpected token \"{}\"", lex.slice().trim()).as_str(),
                span,
            );
            Err(ParseError::Unexpected(inner))
        }
        None => Ok((Token::Eof, "")),
    }
}

fn expect_token<'a>(lex: &'a mut Lexer<Token>, kind: &Token) -> ParseResult<&'a str> {
    let res = get_next(lex)?;
    if res.0 == *kind {
        Ok(res.1)
    } else {
        let actual = res.1.to_string();
        // let span = lex.span();
        Err(unexpected("label name", &actual, "identifier", None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metric_name() {
        let cases = vec![
            ("foo{}", vec![Label::new("__name__".to_string(), "foo".to_string())]),
            ("foo", vec![Label::new("__name__".to_string(), "foo".to_string())]),
            (
                "foo{bar=\"baz\"}",
                vec![
                    Label::new("__name__".to_string(), "foo".to_string()),
                    Label::new("bar".to_string(), "baz".to_string()),
                ],
            ),
            (
                "foo{bar=\"baz\", qux=\"quux\"}",
                vec![
                    Label::new("__name__".to_string(), "foo".to_string()),
                    Label::new("bar".to_string(), "baz".to_string()),
                    Label::new("qux".to_string(), "quux".to_string()),
                ],
            ),
            (
                "metric_name{label1=\"value1\", label2=\"value2\"}",
                vec![
                    Label::new("__name__".to_string(), "metric_name".to_string()),
                    Label::new("label1".to_string(), "value1".to_string()),
                    Label::new("label2".to_string(), "value2".to_string()),
                ],
            ),
            (
                "http_requests_total{method=\"post\", code=\"200\"}",
                vec![
                    Label::new("__name__".to_string(), "http_requests_total".to_string()),
                    Label::new("code".to_string(), "200".to_string()),
                    Label::new("method".to_string(), "post".to_string()),
                ],
            ),
            (
                "up{instance=\"localhost:9090\", job=\"prometheus\"}",
                vec![
                    Label::new("__name__".to_string(), "up".to_string()),
                    Label::new("instance".to_string(), "localhost:9090".to_string()),
                    Label::new("job".to_string(), "prometheus".to_string()),
                ],
            ),
        ];



        for (input, expected) in cases {
            let mut got = parse_metric_name(input).unwrap();
            got.sort();
            assert_eq!(got, expected);
        }
    }
}