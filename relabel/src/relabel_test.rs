#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use metricsql_common::prelude::Label;
    use crate::relabel::{fill_label_references};
    use crate::relabel::{labels_to_string, sanitize_metric_name, DebugStep, ParsedRelabelConfig};
    use crate::relabel_config::RelabelConfig;
    use crate::utils::new_labels_from_string;
    use crate::{parse_relabel_config, ParsedConfigs, RelabelAction, METRIC_NAME_LABEL};

    /// removes labels with "__" in the beginning (except "__name__").
    pub(crate) fn finalize_labels(dst: &mut Vec<Label>) {
        dst.retain(|label| !label.name.starts_with("__") || label.name == METRIC_NAME_LABEL);
    }

    #[test]
    fn test_sanitize_metric_name() {
        fn f(s: &str, result_expected: &str) {
            for i in 0..5 {
                let result = sanitize_metric_name(s);
                assert_eq!(result, result_expected,
                           "unexpected result for SanitizeMetricName({s}) at iteration {i}; got {}; want {}", result, result_expected)
            }
        }

        f("", "");
        f("a", "a");
        f("foo.bar/baz:a", "foo_bar_baz:a");
        f("foo...bar", "foo___bar")
    }

    #[test]
    fn test_sanitize_label_name() {
        fn f(s: &str, result_expected: &str) {
            for i in 0..5 {
                let result = sanitize_metric_name(s);
                assert_eq!(result, result_expected,
                           "unexpected result for SanitizeLabelName({s}) at iteration {i}; got {}; want {}", result, result_expected)
            }
        }

        f("", "");
        f("a", "a");
        f("foo.bar/baz:a", "foo_bar_baz_a");
        f("foo...bar", "foo___bar")
    }

    #[test]
    fn test_labels_to_string() {
        fn check(labels: Vec<Label>, expected: &str) {
            let s = labels_to_string(&labels);
            assert_eq!(s, expected,
                       "unexpected result;\ngot\n{}\nwant\n{}", s, expected)
        }

        check(vec![], "{}");
        check(vec![Label { name: "__name__".to_string(), value: "foo".to_string() }], "foo");
        check(vec![Label { name: "foo".to_string(), value: "bar".to_string() }], r#"{foo="bar"}"#);
        check(vec![
            Label { name: "foo".to_string(), value: "bar".to_string() },
            Label { name: "a".to_string(), value: "bc".to_string() },
        ], r#"{a="bc",foo="bar"}"#);
        check(vec![
            Label { name: "foo".to_string(), value: "bar".to_string() },
            Label { name: "__name__".to_string(), value: "xxx".to_string() },
            Label { name: "a".to_string(), value: "bc".to_string() },
        ], r#"xxx{a="bc",foo="bar"}"#)
    }

    fn parse_config(config: &str) -> ParsedConfigs {
        todo!("parse_config")
        // parse_relabel_configs_data(config)
        //     .map_err(|err| format!("cannot parse \"{}\": {}", config, err))
        //     .unwrap()
    }

    fn check_apply_debug(config: &str, metric: &str, dss_expected: Vec<DebugStep>) {
        let pcs = parse_config(config);
        let mut labels = new_labels_from_string(metric).unwrap();
        let dss = pcs.apply_debug(&mut labels);
        assert_eq!(dss, dss_expected,
            "unexpected result; got\n{:?}\nwant\n{:?}", dss, dss_expected);
    }

    #[test]
    fn parsed_relabel_configs_apply_debug() {

// empty relabel config
        check_apply_debug("", "foo", vec![]);
// add label
        check_apply_debug(r#"
- target_label: abc
replacement: xyz
"#, r#"foo{bar="baz"}"#, vec![DebugStep {
            rule: "target_label: abc\nreplacement: xyz\n".to_string(),
            r#in: r#"foo{bar="baz"}"#.to_string(),
            out: r#"foo{abc="xyz",bar="baz"}"#.to_string(),
        }]);
// drop label
        check_apply_debug(r#"
- target_label: bar
replacement: ''
"#, r#"foo{bar="baz"}"#,
                          vec![
                              DebugStep {
                                  rule: "target_label: bar\nreplacement: \"\"\n".to_string(),
                                  r#in: r#"foo{bar="baz"}"#.to_string(),
                                  out: r#"foo{bar=""}"#.to_string(),
                              },
                              DebugStep {
                                  rule: "remove empty labels".to_string(),
                                  r#in: r#"foo{bar=""}"#.to_string(),
                                  out: "foo".to_string(),
                              },
                          ]);
// drop metric
        check_apply_debug(r#"
- action: drop
source_labels: [bar]
regex: baz
"#, r#"foo{bar="baz",abc="def"}"#, vec![
            DebugStep {
                rule: "action: drop\nsource_labels: [bar]\nregex: baz\n".to_string(),
                r#in: r#"foo{abc="def",bar="baz"}"#.to_string(),
                out: "{}".to_string(),
            }]);
// Multiple steps
        check_apply_debug(r#"
- action: labeldrop
regex: "foo.*"
- target_label: foobar
replacement: "abc"
"#, r#"m{foo="x",foobc="123",a="b"}"#,
                          vec![
                              DebugStep {
                                  rule: "action: labeldrop\nregex: foo.*\n".to_string(),
                                  r#in: r#"m{a="b",foo="x",foobc="123"}"#.to_string(),
                                  out: r#"m{a="b"}"#.to_string(),
                              },
                              DebugStep {
                                  rule: "target_label: foobar\nreplacement: abc\n".to_string(),
                                  r#in: r#"m{a="b"}"#.to_string(),
                                  out: r#"m{a="b",foobar="abc"}"#.to_string(),
                              },
                          ]);
    }


    fn check_apply(config: &str, metric: &str, is_finalize: bool, result_expected: &str) {
        let pcs = parse_config(config);
        let mut labels = new_labels_from_string(metric).unwrap();
        pcs.apply(&mut labels, 0);
        if is_finalize {
            finalize_labels(&mut labels)
        }
        labels.sort();
        let result = labels_to_string(&labels);
        assert_eq!(result, result_expected, "unexpected result; got\n{}\nwant\n{}",
                   result, result_expected)
    }

    fn check(config: RelabelConfig, metric: &str, is_finalize: bool, result_expected: &str) {
        let pcs = parse_relabel_config(config).unwrap();
        let mut labels = new_labels_from_string(metric).unwrap();
        pcs.apply(&mut labels, 0);
        if is_finalize {
            finalize_labels(&mut labels)
        }
        labels.sort();
        let result = labels_to_string(&labels);
        assert_eq!(result, result_expected, "unexpected result; got\n{}\nwant\n{}",
                   result, result_expected)
    }


    #[test]
    fn apply_empty_relabel_configs() {
        check_apply("", "{}", false, "{}");
        check_apply("", "{}", true, "{}");
        check_apply("", r#"{foo="bar"}"#, false, r#"{foo="bar"}"#);
        check_apply("", r#"xxx{foo="bar",__aaa="yyy"}"#, false, r#"xxx{__aaa="yyy",foo="bar"}"#);
        check_apply("", r#"xxx{foo="bar",__aaa="yyy"}"#, true, r#"xxx{foo="bar"}"#);
    }

    #[test]
    fn apply_replace_miss() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["foo".to_string()],
            target_label: "bar".to_string(),
            ..Default::default()
        };
        check(config.clone(), "{}", false, "{}");
        check(config.clone(), r#"{xxx="yyy"}"#, false, r#"{xxx="yyy"}"#);

        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["foo".to_string()],
            target_label: "xxx".to_string(),
            regex: Some(".+".to_string()),
            ..Default::default()
        };

        check(config, r#"{xxx="yyy"}"#, false, r#"{xxx="yyy"}"#);
    }

    #[test]
    fn apply_replace_if_miss() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            if_expr: Some(r#"{foo="bar"}"#.to_string()),
            source_labels: vec!["xxx".to_string(), "foo".to_string()],
            target_label: "bar".to_string(),
            replacement: "a-$1-b".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="yyy"}"#, false, r#"{xxx="yyy"}"#);
    }

    #[test]
    fn apply_replace_hit() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["xxx".to_string(), "foo".to_string()],
            target_label: "bar".to_string(),
            replacement: "a-$1-b".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="yyy"}"#, false, r#"{bar="a-yyy;-b",xxx="yyy"}"#);

        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["xxx".to_string(), "foo".to_string()],
            target_label: "xxx".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="yyy"}"#, false, r#"{xxx="yyy;"}"#);

        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["foo".to_string()],
            target_label: "xxx".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="yyy"}"#, false, "{}")
    }

    #[test]
    fn apply_replace_if_hit() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            if_expr: Some(r#"{xxx=~".y."}"#.to_string()),
            source_labels: vec!["xxx".to_string(), "foo".to_string()],
            target_label: "bar".to_string(),
            replacement: "a-$1-b".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="yyy"}"#, false, r#"{bar="a-yyy;-b",xxx="yyy"}"#);
    }

    #[test]
    fn apply_replace_remove_label_value_hit() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["foo".to_string()],
            target_label: "foo".to_string(),
            regex: Some("xxx".to_string()),
            replacement: "".to_string(),
            ..Default::default()
        };
        check(config, r#"{foo="xxx",bar="baz"}"#, false, r#"{bar="baz"}"#);
    }

    #[test]
    fn apply_replace_remove_label_value_miss() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["foo".to_string()],
            target_label: "foo".to_string(),
            regex: Some("xxx".to_string()),
            replacement: "".to_string(),
            ..Default::default()
        };

        check(config, r#"{foo="yyy",bar="baz"}"#, false, r#"{bar="baz",foo="yyy"}"#);
    }

    #[test]
    fn apply_replace_hit_remove_label() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["xxx".to_string(), "foo".to_string()],
            target_label: "foo".to_string(),
            regex: Some("yyy;.+".to_string()),
            replacement: "".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="yyy",foo="bar"}"#, false, r#"{xxx="yyy"}"#);
    }

    #[test]
    fn apply_replace_miss_remove_label() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["xxx".to_string(), "foo".to_string()],
            target_label: "foo".to_string(),
            regex: Some("yyy;.+".to_string()),
            replacement: "".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="yyyz",foo="bar"}"#, false, r#"{foo="bar",xxx="yyyz"}"#);
    }

    #[test]
    fn apply_replace_hit_target_label_with_capture_group() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["xxx".to_string(), "foo".to_string()],
            target_label: "bar-$1".to_string(),
            replacement: "a-$1-b".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="yyy"}"#, false, r#"{bar-yyy;="a-yyy;-b",xxx="yyy"}"#)
    }

    #[test]
    fn apply_replace_all_miss() {
        let config = RelabelConfig {
            action: RelabelAction::ReplaceAll,
            source_labels: vec!["foo".to_string()],
            target_label: "bar".to_string(),
            ..Default::default()
        };

        check(config.clone(), "{}", false, "{}");
        check(config.clone(), r#"{xxx="yyy"}"#, false, r#"{xxx="yyy"}"#);

        let config = RelabelConfig {
            action: RelabelAction::ReplaceAll,
            source_labels: vec!["foo".to_string()],
            target_label: "bar".to_string(),
            regex: Some(".+".to_string()),
            ..Default::default()
        };
        check(config, r#"{xxx="yyy"}"#, false, r#"{xxx="yyy"}"#);
    }

    #[test]
    fn apply_replace_all_if_miss() {
        let config = RelabelConfig {
            action: RelabelAction::ReplaceAll,
            if_expr: Some("foo".to_string()),
            source_labels: vec!["xxx".to_string()],
            target_label: "xxx".to_string(),
            regex: Some("-".to_string()),
            replacement: ".".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="a-b-c"}"#, false, r#"{xxx="a-b-c"}"#);
    }

    #[test]
    fn apply_replace_all_hit() {
        let config = RelabelConfig {
            action: RelabelAction::ReplaceAll,
            source_labels: vec!["xxx".to_string()],
            target_label: "xxx".to_string(),
            regex: Some("-".to_string()),
            replacement: ".".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="a-b-c"}"#, false, r#"{xxx="a.b.c"}"#);
    }

    #[test]
    fn apply_replace_all_if_hit() {
        let config = RelabelConfig {
            action: RelabelAction::ReplaceAll,
            if_expr: Some(r#"{non_existing_label=~".*"}"#.to_string()),
            source_labels: vec!["xxx".to_string()],
            target_label: "xxx".to_string(),
            regex: Some("-".to_string()),
            replacement: ".".to_string(),
            ..Default::default()
        };
        check(config, r#"{xxx="a-b-c"}"#, false, r#"{xxx="a.b.c"}"#);
    }

    #[test]
    fn apply_replace_all_regex_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::ReplaceAll,
            source_labels: vec!["xxx".to_string(), "foo".to_string()],
            target_label: "xxx".to_string(),
            regex: Some("(;)".to_string()),
            replacement: "-$1-".to_string(),
            ..Default::default()
        };

        check(cfg, r#"{xxx="y;y"}"#, false, r#"{xxx="y-;-y-;-"}"#)
    }

    #[test]
    fn apply_replace_add_multi_labels() {
        check_apply(r#"
- action: replace
source_labels: ["xxx"]
target_label: "bar"
replacement: "a-$1"
- action: replace
source_labels: ["bar"]
target_label: "zar"
replacement: "b-$1"
"#, r#"{xxx="yyy",instance="a.bc"}"#, true, r#"{bar="a-yyy",instance="a.bc",xxx="yyy",zar="b-a-yyy"}"#)
    }

    #[test]
    fn apply_replace_self() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            source_labels: vec!["foo".to_string()],
            target_label: "foo".to_string(),
            replacement: "a-$1".to_string(),
            ..Default::default()
        };
        check(config, r#"{foo="aaxx"}"#, true, r#"{foo="a-aaxx"}"#);
    }

    #[test]
    fn apply_replace_missing_source() {
        let config = RelabelConfig {
            action: RelabelAction::Replace,
            target_label: "foo".to_string(),
            replacement: "foobar".to_string(),
            ..Default::default()
        };
        check(config, "{}", true, r#"{foo="foobar"}"#);
    }

    #[test]
    fn keep_if_equal_miss() {
        check_apply(r#"
- action: keep_if_equal
source_labels: ["foo", "bar"]
", "{}", true, "{}");
f(r#"
- action: keep_if_equal
source_labels: ["xxx", "bar"]
"#, r#"{xxx="yyy"}"#, true, "{}")
    }

    #[test]
    fn keep_if_equal_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::KeepIfEqual,
            source_labels: vec!["xxx".to_string(), "bar".to_string()],
            ..Default::default()
        };
        check(cfg, r#"{xxx="yyy",bar="yyy"}"#, true, r#"{bar="yyy",xxx="yyy"}"#);
    }

    #[test]
    fn drop_if_equal_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::DropIfEqual,
            source_labels: vec!["foo".to_string(), "bar".to_string()],
            ..Default::default()
        };
        check(cfg.clone(), "{}", true, "{}");

        check(cfg, r#"{xxx="yyy"}"#, true, r#"{xxx="yyy"}"#);
    }

    #[test]
    fn drop_if_equal_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::DropIfEqual,
            source_labels: vec!["xxx".to_string(), "bar".to_string()],
            ..Default::default()
        };
        check(cfg, r#"{xxx="yyy",bar="yyy"}"#, true, "{}")
    }

    #[test]
    fn keepequal_hit() {
        let config = RelabelConfig {
            action: RelabelAction::KeepEqual,
            source_labels: vec!["foo".to_string()],
            target_label: "bar".to_string(),
            ..Default::default()
        };
        check(config, r#"{foo="a",bar="a"}"#, true, r#"{bar="a",foo="a"}"#)
    }

    #[test]
    fn keepequal_miss() {
        let config = RelabelConfig {
            action: RelabelAction::KeepEqual,
            source_labels: vec!["foo".to_string()],
            target_label: "bar".to_string(),
            ..Default::default()
        };
        check(config, r#"{foo="a",bar="x"}"#, true, "{}")
    }

    #[test]
    fn dropequal_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::DropEqual,
            source_labels: vec!["foo".to_string()],
            target_label: "bar".to_string(),
            ..Default::default()
        };
        check(cfg, r#"{foo="a",bar="a"}"#, true, "{}")
    }

    #[test]
    fn dropequal_miss() {
        let config = RelabelConfig {
            action: RelabelAction::DropEqual,
            source_labels: vec!["foo".to_string()],
            target_label: "bar".to_string(),
            ..Default::default()
        };
        check(config, r#"{foo="a",bar="x"}"#, true, r#"{bar="x",foo="a"}"#)
    }

    #[test]
    fn keep_miss() {
        check_apply(r#"
- action: keep
source_labels: [foo]
regex: ".+"
"#, "{}", true, "{}");
        check_apply(r#"
- action: keep
source_labels: [foo]
regex: ".+"
"#, r#"{xxx="yyy"}"#, true, "{}")
    }

    #[test]
    fn keep_if_miss() {
        let config = RelabelConfig {
            action: RelabelAction::Keep,
            if_expr: Some(r#"{foo="bar"}"#.to_string()),
            ..Default::default()
        };
        check(config, r#"{foo="yyy"}"#, false, "{}")
    }

    #[test]
    fn keep_if_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::Keep,
            if_expr: Some(r#"['foobar', '{foo="yyy"}', '{a="b"}']"#.to_string()),
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy"}"#, false, r#"{foo="yyy"}"#)
    }

    #[test]
    fn keep_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::Keep,
            source_labels: vec!["foo".to_string()],
            regex: Some("yyy".to_string()),
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy"}"#, false, r#"{foo="yyy"}"#)
    }

    #[test]
    fn keep_hit_regexp() {
        let cfg = RelabelConfig {
            action: RelabelAction::Keep,
            source_labels: vec!["foo".to_string()],
            regex: Some(".+".to_string()),
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy"}"#, false, r#"{foo="yyy"}"#)
    }

    #[test]
    fn keep_metrics_miss() {
        check_apply(r#"
- action: keep_metrics
regex:
- foo
- bar
"#, "xxx", true, "{}")
    }

    #[test]
    fn keep_metrics_if_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::KeepMetrics,
            if_expr: Some("bar".to_string()),
            ..Default::default()
        };
        check(cfg, "foo", true, "{}")
    }

    #[test]
    fn keep_metrics_if_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::KeepMetrics,
            if_expr: Some("foo".to_string()),
            ..Default::default()
        };
        check(cfg, "foo", true, "foo")
    }

    #[test]
    fn keep_metrics_hit() {
        check_apply(r#"
- action: keep_metrics
regex:
- foo
- bar
"#, "foo", true, "foo")
    }

    #[test]
    fn drop_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::Drop,
            source_labels: vec!["foo".to_string()],
            regex: Some(".+".to_string()),
            ..Default::default()
        };
        check(cfg.clone(), "{}", false, "{}");
        check(cfg, r#"{xxx="yyy"}"#, true, r#"{xxx="yyy"}"#)
    }

    #[test]
    fn drop_if_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::Drop,
            if_expr: Some(r#"{foo="bar"}"#.to_string()),
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy"}"#, true, r#"{foo="yyy"}"#)
    }

    #[test]
    fn drop_if_hit() {
        let config = RelabelConfig {
            action: RelabelAction::Drop,
            if_expr: Some(r#"{foo="yyy}"#.to_string()),
            ..Default::default()
        };
        check(config, r#"{foo="yyy"}"#, true, "{}")
    }

    #[test]
    fn drop_hit() {
        let config = RelabelConfig {
            action: RelabelAction::Drop,
            source_labels: vec!["foo".to_string()],
            regex: Some("yyy".to_string()),
            ..Default::default()
        };
        check(config, r#"{foo="yyy"}"#, true, "{}")
    }

    #[test]
    fn drop_hit_regexp() {
        let config = RelabelConfig {
            action: RelabelAction::Drop,
            source_labels: vec!["foo".to_string()],
            regex: Some(".+".to_string()),
            ..Default::default()
        };
        check(config, r#"{foo="yyy"}"#, true, "{}")
    }

    #[test]
    fn drop_metrics_miss() {
        check_apply(r#"
- action: drop_metrics
regex:
- foo
- bar
"#, "xxx", true, "xxx")
}

    #[test]
    fn drop_metrics_if_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::DropMetrics,
            if_expr: Some("bar".to_string()),
            ..Default::default()
        };
        check(cfg, "foo", true, "foo")
    }

    #[test]
    fn drop_metrics_if_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::DropMetrics,
            if_expr: Some("foo".to_string()),
            ..Default::default()
        };
        check(cfg, "foo", true, "{}")
    }

    #[test]
    fn drop_metrics_hit() {
        check_apply(r#"
- action: drop_metrics
regex:
- foo
- bar
"#, "foo", true, "{}")
    }

    #[test]
    fn hashmod_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::HashMod,
            source_labels: vec!["foo".to_string()],
            target_label: "aaa".to_string(),
            modulus: 123,
            ..Default::default()
        };
        check(cfg, r#"{xxx="yyy"}"#, false, r#"{aaa="81",xxx="yyy"}"#)
    }

    #[test]
    fn hashmod_if_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::HashMod,
            if_expr: Some(r#"{foo="bar"}"#.to_string()),
            source_labels: vec!["foo".to_string()],
            target_label: "aaa".to_string(),
            modulus: 123,
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy"}"#, true, r#"{foo="yyy"}"#)
    }

    #[test]
    fn hashmod_if_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::HashMod,
            source_labels: vec!["foo".to_string()],
            target_label: "aaa".to_string(),
            if_expr: Some(r#"{foo="yyy"}"#.to_string()),
            modulus: 123,
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy"}"#, true, r#"{aaa="73",foo="yyy"}"#)
    }

    #[test]
    fn hashmod_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::HashMod,
            source_labels: vec!["foo".to_string()],
            target_label: "aaa".to_string(),
            modulus: 123,
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy"}"#, true, r#"{aaa="73",foo="yyy"}"#)
    }

    #[test]
    fn labelmap_copy_label_if_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMap,
            if_expr: Some(r#"{foo="yyy",foobar="aab"}"#.to_string()),
            regex: Some("foo".to_string()),
            replacement: "bar".to_string(),
            ..Default::default()
        };

        check(cfg, r#"{foo="yyy",foobar="aaa"}"#, true, r#"{foo="yyy",foobar="aaa"}"#)
    }

    #[test]
    fn labelmap_copy_label_if_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMap,
            if_expr: Some(r#"{foo="yyy",foobar="aaa"}"#.to_string()),
            regex: Some("foo".to_string()),
            replacement: "bar".to_string(),
            ..Default::default()
        };

        check(cfg, r#"{foo="yyy",foobar="aaa"}"#, true, r#"{bar="yyy",foo="yyy",foobar="aaa"}"#)
    }

    #[test]
    fn labelmap_copy_label() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMap,
            regex: Some("foo".to_string()),
            replacement: "bar".to_string(),
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy",foobar="aaa"}"#, true, r#"{bar="yyy",foo="yyy",foobar="aaa"}"#)
    }

    #[test]
    fn labelmap_remove_prefix_dot_star() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMap,
            regex: Some("foo(.*)".to_string()),
            ..Default::default()
        };
        check(cfg, r#"{xoo="yyy",foobar="aaa"}"#, true, r#"{bar="aaa",foobar="aaa",xoo="yyy"}"#)
    }

    #[test]
    fn labelmap_remove_prefix_dot_plus() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMap,
            regex: Some("foo(.+)".to_string()),
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy",foobar="aaa"}"#, true, r#"{bar="aaa",foo="yyy",foobar="aaa"}"#)
    }

    #[test]
    fn labelmap_regex() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMap,
            regex: Some("foo(.+)".to_string()),
            replacement: "$1-x".to_string(),
            ..Default::default()
        };
        check(cfg, r#"{foo="yyy",foobar="aaa"}"#, true, r#"{bar-x="aaa",foo="yyy",foobar="aaa"}"#)
    }

    #[test]
    fn labelmap_all_if_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMapAll,
            if_expr: Some("foobar".to_string()),
            regex: Some("\\.".to_string()),
            replacement: "_".to_string(),
            ..Default::default()
        };
        check(cfg, r#"{foo.bar.baz="yyy",foobar="aaa"}"#, true, r#"{foo.bar.baz="yyy",foobar="aaa"}"#)
    }

    #[test]
    fn labelmap_all_if_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMapAll,
            if_expr: Some(r#"{foo.bar.baz="yyy"}"#.to_string()),
            regex: Some("\\.".to_string()),
            replacement: "_".to_string(),
            ..Default::default()
        };
        check(cfg, r#"{foo.bar.baz="yyy",foobar="aaa"}"#, true, r#"{foo-bar-baz="yyy",foobar="aaa"}"#)
    }

    #[test]
    fn labelmap_all() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMapAll,
            regex: Some("\\.".to_string()),
            replacement: "_".to_string(),
            ..Default::default()
        };

        check(cfg, r#"{foo.bar.baz="yyy",foobar="aaa"}"#, true, r#"{foo-bar-baz="yyy",foobar="aaa"}"#)
    }

    #[test]
    fn labelmap_all_regexp() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelMapAll,
            regex: Some("ba(.)".to_string()),
            replacement: "${1}ss".to_string(),
            ..Default::default()
        };

        check(cfg, r#"{foo.bar.baz="yyy",foozar="aaa"}"#, true, r#"{foo.rss.zss="yyy",foozar="aaa"}"#)
    }

    #[test]
    fn label_drop() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelDrop,
            regex: Some("dropme".to_string()),
            ..Default::default()
        };
        check(cfg.clone(), r#"{aaa="bbb"}"#, true, r#"{aaa="bbb"}"#);

        check(cfg.clone(), r#"{xxx="yyy",dropme="aaa",foo="bar"}"#, false, r#"{foo="bar",xxx="yyy"}"#);
        // regex in single quotes
        check(cfg.clone(), r#"{xxx="yyy",dropme="aaa"}"#, false, r#"{xxx="yyy"}"#);
        // regex in double quotes
        check(cfg, r#"{xxx="yyy",dropme="aaa"}"#, false, r#"{xxx="yyy"}"#)
    }

    #[test]
    fn if_hit() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelDrop,
            if_expr: Some("foo".to_string()),
            regex: Some("dropme".to_string()),
            ..Default::default()
        };
        check(cfg, r#"{xxx="yyy",dropme="aaa",foo="bar"}"#, false, r#"{dropme="aaa",foo="bar",xxx="yyy"}"#);
    }

    #[test]
    fn if_miss() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelDrop,
            if_expr: Some(r#"{xxx="yyy"}"#.to_string()),
            regex: Some("dropme".to_string()),
            ..Default::default()
        };
        check(cfg, r#"{xxx="yyy",dropme="aaa",foo="bar"}"#, false, r#"{foo="bar",xxx="yyy"}"#);
    }

    #[test]
    fn labeldrop_prefix() {
        check_apply(r#"
- action: labeldrop
regex: "dropme.*"
"#, r#"{aaa="bbb"}"#, true, r#"{aaa="bbb"}"#);
        check_apply(r#"
- action: labeldrop
regex: "dropme(.+)"
"#, r#"{xxx="yyy",dropme-please="aaa",foo="bar"}"#, false, r#"{foo="bar",xxx="yyy"}"#)
    }

    #[test]
    fn labeldrop_regexp() {
        check_apply(r#"
- action: labeldrop
regex: ".*dropme.*"
"#, r#"{aaa="bbb"}"#, true, r#"{aaa="bbb"}"#);
        check_apply(r#"
- action: labeldrop
regex: ".*dropme.*"
"#, r#"{xxx="yyy",dropme-please="aaa",foo="bar"}"#, false, r#"{foo="bar",xxx="yyy"}"#);
    }

    #[test]
    fn labelkeep() {
        check_apply(r#"
- action: labelkeep
regex: "keepme"
"#, r#"{keepme="aaa"}"#, true, r#"{keepme="aaa"}"#);
// if_miss
        check_apply(r#"
- action: labelkeep
if: '{aaaa="awefx"}'
regex: keepme
"#, r#"{keepme="aaa",aaaa="awef",keepme-aaa="234"}"#, false, r#"{aaaa="awef",keepme="aaa",keepme-aaa="234"}"#);
// if-hit
        check_apply(r#"
- action: labelkeep
if: '{aaaa="awef"}'
regex: keepme
"#, r#"{keepme="aaa",aaaa="awef",keepme-aaa="234"}"#, false, r#"{keepme="aaa"}"#);
        check_apply(r#"
- action: labelkeep
regex: keepme
"#, r#"{keepme="aaa",aaaa="awef",keepme-aaa="234"}"#, false, r#"{keepme="aaa"}"#);
    }

    #[test]
    fn labelkeep_regexp() {
        let cfg = RelabelConfig {
            action: RelabelAction::LabelKeep,
            regex: Some("keepme.*".to_string()),
            ..Default::default()
        };

        check(cfg.clone(), r#"{keepme="aaa"}"#, true, r#"{keepme="aaa"}"#);
        check(cfg, r#"{keepme="aaa",aaaa="awef",keepme-aaa="234"}"#, false, r#"{keepme="aaa",keepme-aaa="234"}"#);
    }

    #[test]
    fn upper_lower_case() {
        let config1 = RelabelConfig {
            action: RelabelAction::Uppercase,
            source_labels: vec!["foo".to_string()],
            target_label: "foo".to_string(),
            ..Default::default()
        };
        check(config1, r#"{foo="bar"}"#, true, r#"{foo="BAR"}"#);
        check_apply(r#"
- action: lowercase
source_labels: ["foo", "bar"]
target_label: baz
- action: labeldrop
regex: foo|bar
"#, r#"{foo="BaR",bar="fOO"}"#, true, r#"{baz="bar;foo"}"#);
        check_apply(r#"
- action: lowercase
source_labels: ["foo"]
target_label: baz
- action: uppercase
source_labels: ["bar"]
target_label: baz
"#, r#"{qux="quux"}"#, true, r#"{qux="quux"}"#);
    }

    #[test]
    fn graphite_match() {
        let cfg = RelabelConfig {
            action: RelabelAction::Graphite,
            r#match: "foo.*.baz".to_string(),
            labels: HashMap::from([
                ("name".to_string(), "aaa".to_string()),
                ("job".to_string(), "${1}-zz".to_string())
            ]),
            ..Default::default()
        };
        check(cfg, r#"foo.bar.baz"#, true, r#"aaa{job="bar-zz"}"#);
    }

    #[test]
    fn graphite_mismatch() {
        let config = RelabelConfig {
            action: RelabelAction::Graphite,
            r#match: "foo.*.baz".to_string(),
            labels: HashMap::from([
                ("name".to_string(), "aaa".to_string()),
                ("job".to_string(), "${1}-zz".to_string())
            ]),
            ..Default::default()
        };
        check(config, r#"foo.bar.bazz"#, true, r#"foo.bar.bazz"#);
    }

    #[test]
    fn replacement_with_label_refs() {
// no regex
        check_apply(r#"
- target_label: abc
replacement: "{{__name__}}.{{foo}}"
"#, r#"qwe{foo="bar",baz="aaa"}"#, true, r#"qwe{abc="qwe.bar",baz="aaa",foo="bar"}"#);
// with regex
        check_apply(r#"
- target_label: abc
replacement: "{{__name__}}.{{foo}}.$1"
source_labels: [baz]
regex: "a(.+)"
"#, r#"qwe{foo="bar",baz="aaa"}"#, true, r#"qwe{abc="qwe.bar.aa",baz="aaa",foo="bar"}"#);
    }

    // Check $ at the end of regex - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3131
    #[test]
    fn replacement_with_dollar_sign_at_the_end_of_regex() {
        check_apply(r#"
- target_label: xyz
regex: "foo\\$$"
replacement: bar
source_labels: [xyz]
"#, r#"metric{xyz="foo$",a="b"}"#, true, r#"metric{a="b",xyz="bar"}"#);
    }

    #[test]
    fn issue_3251() {
        check_apply(r#"
- source_labels: [instance, container_label_com_docker_swarm_task_name]
separator: ';'
#  regex: '(.*?)\..*;(.*?)\..*'
regex: '([^.]+).[^;]+;([^.]+).+'
replacement: '$2:$1'
target_label: container_label_com_docker_swarm_task_name
action: replace
"#, r#"{instance="subdomain.domain.com",container_label_com_docker_swarm_task_name="myservice.h408nlaxmv8oqkn1pjjtd71to.nv987lz99rb27lkjjnfiay0g4"}"#, true,
                    r#"{container_label_com_docker_swarm_task_name="myservice:subdomain",instance="subdomain.domain.com"}"#);
    }

    #[test]
    fn test_finalize_labels() {
        fn f(metric: &str, result_expected: &str) {
            let mut result_labels = new_labels_from_string(metric).unwrap();
            finalize_labels(&mut result_labels);
            let result = labels_to_string(&result_labels);
            assert_eq!(result, result_expected,
                       "unexpected result; got\n{}\nwant\n{}", result, result_expected)
        }
        f(r#"{}"#, "{}");
        f(r#"{foo="bar",__aaa="ass",instance="foo.com"}"#, r#"{foo="bar",instance="foo.com"}"#);
        f(r#"{foo="bar",instance="ass",__address__="foo.com"}"#, r#"{foo="bar",instance="ass"}"#);
        f(r#"{foo="bar",abc="def",__address__="foo.com"}"#, r#"{abc="def",foo="bar"}"#);
    }

    #[test]
    fn test_fill_label_references() {
        fn f(replacement: &str, metric: &str, result_expected: &str) {
            let labels = new_labels_from_string(metric).unwrap();
            let mut result: String = String::with_capacity(32);
            fill_label_references(&mut result, replacement, &labels);
            assert_eq!(result, result_expected, "unexpected result; got\n{}\nwant\n{}", result, result_expected)
        }

        f("", r#"foo{bar="baz"}"#, "");
        f("abc", r#"foo{bar="baz"}"#, "abc");
        f(r#"foo{{bar"#, r#"foo{bar="baz"}"#, r#"foo{{bar"#);
        f(r#"foo-$1"#, r#"foo{bar="baz"}"#, r#"foo-$1"#);
        f(r#"foo{{bar}}"#, r#"foo{bar="baz"}"#, r#"foobaz"#);
        f(r#"{{bar}}"#, r#"foo{bar="baz"}"#, r#"baz"#);
        f(r#"{{bar}}-aa"#, r#"foo{bar="baz"}"#, r#"baz-aa"#);
        f(r#"{{bar}}-aa{{__name__}}.{{bar}}{{non-existing-label}}"#, r#"foo{bar="baz"}"#, r#"baz-aafoo.baz"#);
    }

    #[test]
    fn test_regex_match_string_success() {
        fn f(pattern: &str, s: &str) {
            let prc = new_test_regex_relabel_config(pattern);
            if !prc.regex.is_match(s) {
                panic!("unexpected match_string(%{s}) result; got false; want true")
            }
        }

        f("", "");
        f("foo", "foo");
        f(".*", "");
        f(".*", "foo");
        f("foo.*", "foobar");
        f("foo.+", "foobar");
        f("f.+o", "foo");
        f("foo|bar", "bar");
        f("^(foo|bar)$", "foo");
        f("foo.+", "foobar");
        f("^foo$", "foo");
    }

    #[test]
    fn test_regexp_match_string_failure() {
        fn f(pattern: &str, s: &str) {
            let prc = new_test_regex_relabel_config(pattern);
            if prc.regex.is_match(s) {
                panic!("unexpected match_string({}) result; got true; want false", s)
            }
        }

        f("", "foo");
        f("foo", "");
        f("foo.*", "foa");
        f("foo.+", "foo");
        f("f.+o", "foor");
        f("foo|bar", "barz");
        f("^(foo|bar)$", "xfoo");
        f("foo.+", "foo");
        f("^foo$", "foobar")
    }

    fn new_test_regex_relabel_config(pattern: &str) -> ParsedRelabelConfig {
        let mut rc: RelabelConfig = Default::default();
        rc.action = RelabelAction::LabelDrop;
        rc.regex = Some(pattern.to_string());
        parse_relabel_config(rc)
            .map_err(|err| format!("cannot parse pattern {:?}: {}", pattern, err))
            .unwrap()
    }

    fn test_parsed_relabel_configs_apply_for_multiple_series() {
        fn f(config: &str, metrics: Vec<String>, result_expected: Vec<String>) {
            let pcs = parse_config(config);

            let mut total_labels = 0;
            let mut labels = vec![];
            for metric in metrics {
                labels.extend(new_labels_from_string(&metric).unwrap());
                let _ = pcs.apply(&mut labels, total_labels);
                labels.sort();
                total_labels += labels.len();
            }

            let mut result: Vec<String> = vec![];
            for i in 0..labels.len() {
                result.push(labels_to_string(&labels[i .. i + 1]))
            }

            assert_eq!(result.len(), result_expected.len(),
                       "unexpected number of results; got\n{}\nwant\n{}", result.len(), result_expected.len());

            result.iter().zip(result_expected.iter()).for_each(|(r, e)| {
                assert_eq!(r, e, "unexpected result; got\n{}\nwant\n{}", r, e)
            });
        }

        fn drops_one_of_series() {
            f(r#"
- action: drop
if: '{__name__!~"smth"}' 
"#, vec!["smth".to_string(), "notthis".to_string()], vec!["smth".to_string()]);
            f(r#"
- action: drop
if: '{__name__!~"smth"}'
"#, vec!["notthis".to_string(), "smth".to_string()], vec!["smth".to_string()])
        }
    }

}