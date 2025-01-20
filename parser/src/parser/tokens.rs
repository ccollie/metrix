use std::fmt::{Display, Formatter};

use logos::{Lexer, Logos};

use crate::parser::{ParseError, ParseResult};
use crate::prelude::syntax_error;

fn unterminated_string_literal(_: &mut Lexer<Token>) -> ParseResult<()> {
    Err(ParseError::SyntaxError(
        "unterminated string literal".to_string(),
    ))
}

fn invalid_number(lex: &mut Lexer<Token>) -> ParseResult<()> {
    let span = lex.span();
    Err(syntax_error(
        format!("Invalid number: \"{}\"", lex.slice()).as_str(),
        &span,
        "".to_string(),
    ))
}

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(error = ParseError)]
#[logos(subpattern decimal = r"[0-9][_0-9]*")]
#[logos(subpattern hex = r"-?0[xX][0-9a-fA-F][_0-9a-fA-F]*")]
#[logos(subpattern octal = r"-?0[oO]?[1-7][_0-7]*")]
#[logos(subpattern binary = r"-?0[bB][0-1][_0-1]*")]
#[logos(subpattern float = r"-?(?:([0-9]*[.])?[0-9][0-9_]*)(?:[eE][+-]?[0-9][0-9_]*)?")]
#[logos(subpattern duration = r"-?(?:([0-9]*[.])?[0-9][0-9_]*)(ms|s|m|h|d|w|y|i)")]
#[logos(skip r"[ \t\n\f\r]+")]
#[logos(skip r"#[^\r\n]*(\r\n|\n)?")] // single line comment
pub enum Token {
    #[token("and", ignore(ascii_case))]
    OpAnd,

    #[token("atan2", ignore(ascii_case))]
    OpAtan2,

    #[token("bool", ignore(ascii_case))]
    Bool,

    #[token("by", ignore(ascii_case))]
    By,

    #[token("default", ignore(ascii_case))]
    OpDefault,

    #[token("group_left", ignore(ascii_case))]
    GroupLeft,

    #[token("group_right", ignore(ascii_case))]
    GroupRight,

    #[token("if", ignore(ascii_case))]
    OpIf,

    #[token("ifNot", ignore(ascii_case))]
    OpIfNot,

    #[token("ignoring", ignore(ascii_case))]
    Ignoring,

    #[token("keep_metric_names", ignore(ascii_case))]
    KeepMetricNames,

    #[token("limit", ignore(ascii_case))]
    Limit,

    #[token("on", ignore(ascii_case))]
    On,

    #[token("offset", ignore(ascii_case))]
    Offset,

    #[token("or", ignore(ascii_case))]
    OpOr,

    #[token("unless", ignore(ascii_case))]
    OpUnless,

    #[token("without", ignore(ascii_case))]
    Without,

    #[token("$__interval")]
    Interval,

    #[token("$__rate_interval")]
    RateInterval,

    #[regex("(?&duration)+", priority = 5)]
    Duration,

    #[regex("-?[Nn][Aa][Nn]")]
    #[regex("-?[Ii][Nn][Ff]")]
    #[regex("(?&binary)")]
    #[regex("(?&hex)")]
    #[regex("(?&octal)")]
    #[regex("(?&float)((?i)(kib|ki|kb|k|mib|mi|mb|m|gib|gi|gb|g|tib|ti|tb|t))*")]
    Number,

    #[regex(r"[_a-zA-Z][_a-zA-Z0-9:\.]*")]
    Identifier,

    #[regex("'(?s:[^'\\\\]|\\\\.)*'")]
    #[regex("`(?s:[^`\\\\]|\\\\.)*`")]
    #[regex("\"(?s:[^\"\\\\]|\\\\.)*\"")]
    StringLiteral,

    #[token("@")]
    At,

    #[token("{")]
    LeftBrace,

    #[token("}")]
    RightBrace,

    #[token("[")]
    LeftBracket,

    #[token("]")]
    RightBracket,

    #[token(",")]
    Comma,

    #[token(":")]
    Colon,

    #[token("(")]
    LeftParen,

    #[token(")")]
    RightParen,

    #[token(";")]
    SemiColon,

    #[token("=")]
    Equal,

    #[token("==")]
    OpEqual,

    #[token("!=")]
    OpNotEqual,

    #[token("<")]
    OpLessThan,

    #[token("<=")]
    OpLessThanOrEqual,

    #[token(">")]
    OpGreaterThan,

    #[token(">=")]
    OpGreaterThanOrEqual,

    #[token("+")]
    OpPlus,

    #[token("-")]
    OpMinus,

    #[token("/")]
    OpDiv,

    #[token("*")]
    OpMul,

    #[token("^")]
    OpPow,

    #[token("%")]
    OpMod,

    #[token("=~")]
    RegexEqual,

    #[token("!~")]
    RegexNotEqual,

    #[regex("\"(?s:[^\"\\\\]|\\\\.)*", unterminated_string_literal)]
    #[regex("'(?s:[^'\\\\]|\\\\.)*", unterminated_string_literal)]
    #[regex("@[^\"'\\s]\\S+", unterminated_string_literal)]
    ErrorStringUnterminated,

    #[regex(r"(?:0|[1-9][0-9]*)\.[^0-9]", invalid_number)]
    #[regex(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][^+\-0-9]", invalid_number)]
    #[regex(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][+-][^0-9]", invalid_number)]
    ErrorInvalidNumber,

    /// Marker for end of stream.
    Eof,
}

impl Token {
    pub const fn is_operator(&self) -> bool {
        use Token::*;

        matches!(
            self,
            OpAtan2
                | OpMul
                | OpDiv
                | OpDefault
                | OpIf
                | OpIfNot
                | OpMod
                | OpPlus
                | OpMinus
                | OpLessThan
                | OpGreaterThan
                | OpLessThanOrEqual
                | OpGreaterThanOrEqual
                | OpEqual
                | OpNotEqual
                | OpPow
                | OpAnd
                | OpOr
                | OpUnless
        )
    }

    #[inline]
    pub const fn is_comparison_op(&self) -> bool {
        use Token::*;
        matches!(
            self,
            OpEqual
                | OpNotEqual
                | OpGreaterThanOrEqual
                | OpGreaterThan
                | OpLessThanOrEqual
                | OpLessThan
        )
    }

    #[inline]
    pub const fn is_rollup_start(&self) -> bool {
        use Token::*;
        matches!(self, Offset | At | LeftBracket)
    }

    #[inline]
    pub const fn is_group_modifier(&self) -> bool {
        use Token::*;
        matches!(self, On | Ignoring)
    }

    #[inline]
    pub const fn is_join_modifier(&self) -> bool {
        use Token::*;
        matches!(self, GroupLeft | GroupRight)
    }

    pub const fn is_aggregate_modifier(&self) -> bool {
        use Token::*;
        matches!(self, By | Without)
    }

    /// tokens that can function as identifiers in certain constructions (functions/metric names)
    /// not a good idea, but we're matching the original
    pub const fn is_ident_like(&self) -> bool {
        use Token::*;
        // keywords
        matches!(
            self,
            By | Bool
                | OpDefault
                | GroupLeft
                | GroupRight
                | Identifier
                | Ignoring
                | KeepMetricNames
                | Limit
                | On
                | Offset
                | Without
                | OpAnd
                | OpAtan2
                | OpIf
                | OpIfNot
                | OpOr
                | OpUnless
        )
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            // keywords
            Self::By => "by",
            Self::Bool => "bool",
            Self::OpDefault => "default",
            Self::GroupLeft => "group_left",
            Self::GroupRight => "group_right",
            Self::Ignoring => "ignoring",
            Self::KeepMetricNames => "keep_metric_names",
            Self::Limit => "limit",
            Self::On => "on",
            Self::Offset => "offset",
            Self::StringLiteral => "<string literal>",
            Self::Without => "without",

            Self::Identifier => "<identifier>",

            // symbols
            Self::At => "@",
            Self::LeftBrace => "{{",
            Self::RightBrace => "}}",
            Self::LeftBracket => "[",
            Self::RightBracket => "]",
            Self::Colon => ":",
            Self::Comma => ",",
            Self::LeftParen => "(",
            Self::RightParen => ")",
            Self::SemiColon => ";",
            Self::Equal => "=",

            // operators
            Self::OpAnd => "and",
            Self::OpAtan2 => "atan2",
            Self::OpIf => "if",
            Self::OpIfNot => "ifnot",
            Self::OpOr => "or",
            Self::OpUnless => "unless",
            Self::OpMul => "*",
            Self::OpDiv => "/",
            Self::OpMod => "%",
            Self::OpPlus => "+",
            Self::OpMinus => "-",
            Self::OpLessThan => "<",
            Self::OpGreaterThan => ">",
            Self::OpLessThanOrEqual => "<=",
            Self::OpGreaterThanOrEqual => ">=",
            Self::OpEqual => "==",
            Self::OpNotEqual => "!=",
            Self::OpPow => "^",
            Self::RegexEqual => "=~",
            Self::RegexNotEqual => "!~",

            Self::Duration => "<duration>",
            Self::Number => "<number>",

            Self::Eof => "<eof>",

            // other
            Self::ErrorInvalidNumber => "<invalid number>",
            Self::ErrorStringUnterminated => "<unterminated string literal>",
            Self::Interval => "$__interval",
            Self::RateInterval => "$__rate_interval",
        }
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub(crate) const IDENT_LIKE_TOKENS: &'static [Token] =
    &[
        Token::By,
        Token::Bool,
        Token::OpDefault,
        Token::GroupLeft,
        Token::GroupRight,
        Token::Ignoring,
        Token::KeepMetricNames,
        Token::Identifier,
        Token::Limit,
        Token::On,
        Token::Offset,
        Token::Without,
        Token::OpAnd,
        Token::OpAtan2,
        Token::OpIf,
        Token::OpIfNot,
        Token::OpOr,
        Token::OpUnless,
    ];

#[cfg(test)]
mod tests {
    use logos::Logos;
    use pretty_assertions::assert_eq;
    use test_case::test_case;

    use super::Token;
    use super::Token::*;

    macro_rules! test_tokens {
    ($src:expr, [$(
      $tok:expr
      $(=> $val:expr)?
    ),*$(,)?]) => {#[allow(unused)] {
      let src: &str = $src;
      let mut lex = Token::lexer(src);
      let mut index = 0;
      let mut offset = 0;
      $({
        let actual = lex.next().expect(&format!("Expected token {}", index + 1)).unwrap();
        let expected = $tok;
        let text = lex.slice();
        assert_eq!(actual, expected);
        $(
          assert_eq!(text, $val, "index: {}", index);
        )?

        index += 1;
        offset += text.len();
      })*

      match lex.next() {
        None => (),
        Some(t) => panic!("Expected exactly {} tokens, but got {:#?} when expecting EOF", index, t),
      }
    }};
  }

    #[test]
    fn empty() {
        test_tokens!("", []);
    }

    #[test]
    fn whitespace() {
        // whitespace is skipped
        test_tokens!("  \t\n\r\r\n", []);
    }

    #[test]
    fn negative_ints() {
        test_tokens!("-1", [Number]);
    }

    #[test_case("@", At)]
    #[test_case("{", LeftBrace)]
    #[test_case("}", RightBrace)]
    #[test_case("[", LeftBracket)]
    #[test_case("]", RightBracket)]
    #[test_case("(", LeftParen)]
    #[test_case(")", RightParen)]
    #[test_case(",", Comma)]
    #[test_case(";", SemiColon)]
    #[test_case(":", Colon)]
    #[test_case("=", Equal)]
    fn symbol(src: &str, tok: Token) {
        test_tokens!(src, [tok]);
    }

    #[test_case("atan2", OpAtan2)]
    #[test_case("and", OpAnd)]
    #[test_case("or", OpOr)]
    #[test_case("if", OpIf)]
    #[test_case("ifNot", OpIfNot)]
    #[test_case("unless", OpUnless)]
    #[test_case("==", OpEqual)]
    #[test_case("!=", OpNotEqual)]
    #[test_case("=~", RegexEqual)]
    #[test_case("!~", RegexNotEqual)]
    #[test_case("+", OpPlus)]
    #[test_case("-", OpMinus)]
    #[test_case("*", OpMul)]
    #[test_case("/", OpDiv)]
    #[test_case("%", OpMod)]
    #[test_case("^", OpPow)]
    #[test_case("<", OpLessThan)]
    #[test_case(">", OpGreaterThan)]
    #[test_case("<=", OpLessThanOrEqual)]
    #[test_case(">=", OpGreaterThanOrEqual)]
    fn operator(src: &str, tok: Token) {
        test_tokens!(src, [tok]);
    }

    #[test_case("1")]
    #[test_case("73")]
    #[test_case("65535")]
    #[test_case("1_000_334")]
    fn number_int(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test_case("1.0")]
    #[test_case("0.10")]
    #[test_case("0e100")]
    #[test_case("1e100")]
    #[test_case("123.45")]
    #[test_case("0.234")]
    #[test_case("1.1e100")]
    #[test_case("1.2e-100")]
    #[test_case("1.3e+100")]
    #[test_case("0.49641")]
    fn float_literal(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test_case("0b1110")]
    #[test_case("0b0")]
    fn binary_literal(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test_case("0x1fffa")]
    #[test_case("0x4")]
    fn hex_literal(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test_case("03")]
    #[test_case("0o1")]
    #[test_case("0O12")]
    fn octal_literal(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test]
    fn number_inf() {
        let infs = ["inf", "Inf", "INf", "InF", "iNf", "INF"];
        for inf in &infs {
            test_tokens!(inf, [Number]);
        }
    }

    #[test]
    fn number_nan() {
        let nans = ["nan", "NaN", "Nan", "NAN"];
        for nan in &nans {
            test_tokens!(nan, [Number]);
        }
    }

    #[test]
    fn misc_numbers() {
        // Numbers
        let s = r"3+1.2-.23+4.5e5-78e-6+1.24e+45-NaN+Inf";
        let expected = vec![
            "3", "+", "1.2", "-", ".23", "+", "4.5e5", "-", "78e-6", "+", "1.24e+45", "-", "NaN",
            "+", "Inf",
        ];
        test_success(s, &expected);

        let s = "12.34 * 0X34 + 0b11 + 0O77";
        let expected = vec!["12.34", "*", "0X34", "+", "0b11", "+", "0O77"];
        test_success(s, &expected);
    }

    fn expect_error_containing(s: &str, needle: &str) {
        let mut lex = Token::lexer(s);
        let actual = lex.next().unwrap();
        assert_eq!(actual.is_err(), true);
        let msg = format!("{}", actual.unwrap_err());
        assert!(msg.contains(needle));
    }

    #[test]
    fn test_invalid_number() {
        fn check(s: &str) {
            expect_error_containing(s, "Invalid number");
        }

        check("1.+");
        check("1e!");
        check("1e+!");
    }

    #[test_case("\"hi\"", StringLiteral; "double_1")]
    #[test_case("\"hi\n\"", StringLiteral; "double_2")]
    #[test_case("\"hi\\\"\"", StringLiteral; "double_3")]
    #[test_case("'hi'", StringLiteral; "single_1")]
    #[test_case("'hi\n'", StringLiteral; "single_2")]
    #[test_case("'hi\\''", StringLiteral; "single_3")]
    fn string(src: &str, tok: Token) {
        test_tokens!(src, [tok]);
    }

    #[test]
    fn string_unterminated() {
        fn check(s: &str) {
            expect_error_containing(s, "unterminated string literal");
        }

        check("\"hi");
        check("\'hi");
    }

    #[test_case("by", By)]
    #[test_case("bool", Bool)]
    #[test_case("default", OpDefault)]
    #[test_case("group_left", GroupLeft)]
    #[test_case("group_right", GroupRight)]
    #[test_case("ignoring", Ignoring)]
    #[test_case("keep_metric_names", KeepMetricNames)]
    #[test_case("limit", Limit)]
    #[test_case("offset", Offset)]
    #[test_case("on", On)]
    #[test_case("without", Without)]
    fn keyword(src: &str, tok: Token) {
        test_tokens!(src, [tok]);
    }

    #[test]
    fn identifier() {
        test_tokens!("foobar123", [Identifier]);
    }

    #[test_case("$__interval", Interval)]
    #[test_case("$__rate_interval", RateInterval)]
    fn interval(src: &str, tok: Token) {
        test_tokens!(src, [tok]);
    }

    #[test_case("123ms")]
    #[test_case("123s")]
    #[test_case("123m")]
    #[test_case("1h")]
    #[test_case("2d")]
    #[test_case("3w")]
    #[test_case("4y")]
    #[test_case("1i")]
    #[test_case("3i")]
    fn integer_duration(src: &str) {
        test_tokens!(src, [Duration]);
    }

    #[test_case("123.45ms")]
    #[test_case("0.234s")]
    #[test_case("1.5s")]
    #[test_case("1.5h")]
    #[test_case("1.2h")]
    #[test_case("1.1d")]
    #[test_case("2.3w")]
    #[test_case("1.4y")]
    #[test_case("0.1i")]
    #[test_case("1.5m3.4s2.4ms")]
    #[test_case("-1.5m3.4s2.9ms")]
    fn float_duration(src: &str) {
        test_tokens!(src, [Duration]);
    }

    #[test]
    fn various_durations() {
        // Various durations
        test_success("m offset 123h", &["m", "offset", "123h"]);

        let s = "m offset -1.23w-5h34.5m - 123";
        let expected = vec!["m", "offset", "-1.23w-5h34.5m", "-", "123"];
        test_success(s, &expected);
    }

    #[test_case("2k")]
    #[test_case("2.3kb")]
    #[test_case("3ki")]
    #[test_case("4.5kib")]
    #[test_case("2M")]
    #[test_case("2.3MB")]
    #[test_case("3mi")]
    #[test_case("4.5Mib")]
    #[test_case("2G")]
    #[test_case("2.3gB")]
    #[test_case("3Gi")]
    #[test_case("4.5GiB")]
    #[test_case("2T")]
    #[test_case("2.3tb")]
    #[test_case("3ti")]
    #[test_case("-4.5TIB")]
    fn number_with_unit(src: &str) {
        test_tokens!(src, [Number]);
    }

    fn get_single_token(src: &str) -> Token {
        let mut lexer = Token::lexer(src);
        if let Some(first) = lexer.next() {
            let token = first.unwrap();
            if lexer.next().is_some() {
                panic!("unexpected token after {:?} : {}", token, lexer.slice());
            }
            return token;
        }
        panic!("unexpected EOF")
    }

    #[test_case("1_2_334")]
    #[test_case("1_2.3_34_5")]
    #[test_case("1_2.3_34_5e8")]
    fn number_with_underscore(src: &str) {
        let token = get_single_token(src);
        assert_eq!(token, Number);
    }

    #[test]
    fn identifiers() {
        // short
        test_tokens!(
          "m y f i h d s",
          [
            Identifier=>"m",
            Identifier=>"y",
            Identifier=>"f",
            Identifier=>"i",
            Identifier=>"h",
            Identifier=>"d",
            Identifier=>"s"
          ]
        );

        test_tokens!(
          "foo bar123",
          [
            Identifier=>"foo",
            Identifier=>"bar123",
          ]
        );
    }

    #[test]
    fn py_comment() {
        // comments are skipped
        test_tokens!("# hi", []);
    }

    #[test]
    fn junk() {
        let src = "💩";
        let mut lex = Token::lexer(src);
        let actual = lex.next().unwrap();
        assert_eq!(actual.is_err(), true);
    }

    #[test]
    fn metric_name() {
        // Just metric name
        test_success("metric", &["metric"]);

        // Metric name with spec chars
        test_success("foo.bar_", &["foo.bar_"]);
    }

    #[test]
    fn metric_name_with_window() {
        // Metric name with window
        let s = "metric[5m]  ";
        test_success(s, &["metric", "[", "5m", "]"]);
    }

    #[test]
    fn metric_name_with_offset() {
        // Metric name with offset
        let s = "   metric offset 10d   ";
        test_success(s, &["metric", "offset", "10d"]);
    }

    #[test]
    fn metric_name_with_tag_filters() {
        // Metric name with tag filters
        let s = r#"  metric:12.34{a="foo", b != "bar", c=~ "x.+y", d !~ "zzz"}"#;
        let expected = vec![
            "metric:12.34",
            "{",
            "a",
            "=",
            r#""foo""#,
            ",",
            "b",
            "!=",
            r#""bar""#,
            ",",
            "c",
            "=~",
            r#""x.+y""#,
            ",",
            "d",
            "!~",
            r#""zzz""#,
            "}",
        ];

        test_success(s, &expected);
    }

    #[test]
    fn function_call() {
        // fn call
        let s = r#"sum  (  metric{x="y"  }  [5m] offset 10h)"#;
        let expected = vec![
            "sum", "(", "metric", "{", "x", "=", r#""y""#, "}", "[", "5m", "]", "offset", "10h",
            ")",
        ];
        test_success(s, &expected);
    }



    #[test]
    fn binary_op() {
        // Binary op
        let s = "a+b or c % d and e unless f";
        let expected = vec![
            "a", "+", "b", "or", "c", "%", "d", "and", "e", "unless", "f",
        ];
        test_success(s, &expected);
    }

    #[test]
    fn comments() {
        let s = r"# comment # sdf
		foobar # comment
		baz
		# yet another comment";
        test_success(s, &["foobar", "baz"])
    }

    #[test]
    fn strings() {
        // An empty string
        test_success("", &[]);

        // String with whitespace
        let s = "  \n\t\r ";
        test_success(s, &[]);

        // Strings
        let s = r#""''"#.to_owned() + "``" + r#"\\"  '\\'  "\"" '\''"\\\"\\""#;
        let expected = vec![
            r#""""#,
            "''",
            "``",
            r#"\\"", `'\\'", `"\""", `'\''" "\\\"\\"#,
        ];
        test_success(&s, &expected);
    }

    #[test]
    fn miscellaneous() {
        let s = "   `foo\\\\\\`бар`  ";
        let expected = vec!["`foo\\\\\\`бар`"];
        test_success(s, &expected);
    }

    fn test_success(s: &str, expected_tokens: &[&str]) {
        let mut lexer = Token::lexer(s);
        let mut tokens = vec![];
        while let Some(tok) = lexer.next() {
            match tok {
                Ok(_) => {
                    tokens.push(lexer.slice());
                }
                Err(e) => {
                    panic!("unexpected error {:?}\n {}", e, lexer.slice())
                }
            }
        }

        assert_eq!(
            tokens.len(),
            expected_tokens.len(),
            "expected {:?} tokens, got {:?}",
            expected_tokens.len(),
            tokens.len()
        );

        for i in 0..tokens.len() {
            let actual = tokens[i];
            let expected = expected_tokens[i];
            assert_eq!(
                actual, expected,
                "expected {:?} at index {}, got {:?}",
                expected, i, actual
            );
        }
    }
}
