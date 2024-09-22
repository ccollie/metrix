#[cfg(test)]
mod test {
	use crate::IfExpression;
	use crate::utils::{new_labels_from_string, parse_metric_name, parse_metric_selector};

	#[test]
	fn test_if_expression_parse_failure() {
		fn f(s: &str) {
			let ie = IfExpression::parse(s);
			assert!(ie.is_err(), "expecting non-nil error when parsing {}", s);
		}

		f("{");
		f("{foo");
		f("foo{");
		f(r#"foo{bar="a" or}"#);
	}

	#[test]
	fn test_if_expression_parse_success() {
		fn check(s: &str) {
			IfExpression::parse(s).unwrap();
		}

		check("foo");
		check(r#"{foo="bar"}"#);
		check(r#"foo{bar=~"baz", x!="y"}"#);
		check(r#"{a="b" or c="d",e="x"}"#);
		check(r#"foo{
        bar="a",x="y" or
        x="a",a="b" or
        a="x"
        }"#);
	}

	#[test]
	fn test_if_expression_unmarshal_failure() {
		fn check(s: &str) {
			assert!(IfExpression::parse(s).is_err(), "expecting non-nil error");
		}

		check("{");
		check("{x:y}");
		check("[1]");
		check(r#""{""#);
		check(r#"'{'"#);
		check("foo{bar");
		check("foo{bar}");
		check("foo{bar=");
		check(r#"foo{bar=""#);
		check(r#"foo{bar='"#);
		check(r#"foo{bar=~"("}"#);
		check(r#"foo{bar!~"("}"#);
		check("foo{bar==aaa}");
		check(r#"foo{bar=="b"}"#);
		check(r#"'foo+bar'"#);
		check(r#"'foo{bar=~"a[b"}'"#);
	}

	#[test]
	fn test_if_expression_match() {
		fn check(if_expr: &str, metric_with_labels: &str) {
			let ie = match IfExpression::parse(if_expr) {
				Err(err) => panic!("unexpected error during unmarshal: {:?}", err),
				Ok(val) => val
			};
			let labels = parse_metric_name(metric_with_labels).unwrap();
			if !ie.is_match(&labels) {
				panic!("unexpected mismatch of if_expr={} for {}", if_expr, metric_with_labels)
			}
		}

		check("foo", "foo");
		check("foo", r#"foo{bar="baz",a="b"}"#);
		check(r#"foo{bar="a"}"#, r#"foo{bar="a"}"#);
		check(r#"foo{bar="a" or baz="x"}"#, r#"foo{bar="a"}"#);
		check(r#"foo{baz="x" or bar="a"}"#, r#"foo{bar="a"}"#);
		check(r#"foo{bar="a"}"#, r#"foo{x="y",bar="a",baz="b"}"#);
		check(r#"{a=~"x|abc",y!="z"}"#, r#"m{x="aa",a="abc"}"#);
		check(r#"{a=~"x|abc",y!="z"}"#, r#"m{x="aa",a="abc",y="qwe"}"#);
		check(r#"{__name__="foo"}"#, r#"foo{bar="baz"}"#);
		check(r#"{__name__=~"foo|bar"}"#, "bar");
		check(r#"{__name__!=""}"#, "foo");
		check(r#"{__name__!=""}"#, r#"bar{baz="aa",b="c"}"#);
		check(r#"{__name__!~"a.+"}"#, r#"bar{baz="aa",b="c"}"#);
		check(r#"foo{a!~"a.+"}"#, r#"foo{a="baa"}"#);
		check(r#"{foo=""}"#, "bar");
		check(r#"{foo!=""}"#, r#"aa{foo="b"}"#);
		check(r#"{foo=~".*"}"#, "abc");
		check(r#"{foo=~".*"}"#, r#"abc{foo="bar"}"#);
		check(r#"{foo!~".+"}"#, "abc");
		check(r#"{foo=~"bar|"}"#, "abc");
		check(r#"{foo=~"bar|"}"#, r#"abc{foo="bar"}"#);
		check(r#"{foo!~"bar|"}"#, r#"abc{foo="baz"}"#);
	}

	#[test]
	fn test_if_expression_mismatch() {
		fn check(if_expr: &str, metric_with_labels: &str) {
			let ie: IfExpression = IfExpression::parse(if_expr).unwrap();
			let labels = new_labels_from_string(metric_with_labels).unwrap();
			if ie.is_match(&labels[..]) {
				panic!("unexpected match of if_expr={} for {}", if_expr, metric_with_labels)
			}
		}

		check("foo", "bar");
		check("foo", r#"a{foo="bar"}"#);
		check(r#"foo{bar="a"}"#, "foo");
		check(r#"foo{bar="a" or baz="a"}"#, "foo");
		check(r#"foo{bar="a"}"#, r#"foo{bar="b"}"#);
		check(r#"foo{bar="a"}"#, r#"foo{baz="b",a="b"}"#);
		check(r#"{a=~"x|abc",y!="z"}"#, r#"m{x="aa",a="xabc"}"#);
		check(r#"{a=~"x|abc",y!="z"}"#, r#"m{x="aa",a="abc",y="z"}"#);
		check(r#"{__name__!~".+"}"#, "foo");
		check(r#"{a!~"a.+"}"#, r#"foo{a="abc"}"#);
		check(r#"{foo=""}"#, r#"bar{foo="aa"}"#);
		check(r#"{foo!=""}"#, "aa");
		check(r#"{foo=~".+"}"#, "abc");
		check(r#"{foo!~".+"}"#, r#"abc{foo="x"}"#);
		check(r#"{foo=~"bar|"}"#, r#"abc{foo="baz"}"#);
		check(r#"{foo!~"bar|"}"#, "abc");
		check(r#"{foo!~"bar|"}"#, r#"abc{foo="bar"}"#);
	}
}