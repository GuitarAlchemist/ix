//! Harness adapter — `cargo clippy --message-format=json` output →
//! `SessionEvent::ObservationAdded` stream.
//!
//! Third evidential-shape adapter (after tars/cargo). Focuses on
//! the static-analysis signal axis — "is the code written
//! correctly" rather than "does the code behave correctly."
//!
//! Spec: `demerzel/logic/harness-clippy.md`.

use ix_agent_core::SessionEvent;
use ix_types::Hexavalent;
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const SOURCE: &str = "clippy";

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("input is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

#[derive(Debug, Clone, Deserialize)]
struct ClippyMessage {
    #[serde(default)]
    reason: String,
    #[serde(default)]
    message: Option<Diagnostic>,
}

#[derive(Debug, Clone, Deserialize)]
struct Diagnostic {
    #[serde(default)]
    level: String,
    #[serde(default)]
    code: Option<LintCode>,
    #[serde(default)]
    message: String,
}

#[derive(Debug, Clone, Deserialize)]
struct LintCode {
    #[serde(default)]
    code: String,
}

/// Project a clippy JSON stream into
/// [`SessionEvent::ObservationAdded`] records.
pub fn clippy_to_observations(input: &[u8], round: u32) -> Result<Vec<SessionEvent>, AdapterError> {
    let text = std::str::from_utf8(input)?;
    let diagnosis_id = sha256_hex(input);

    // Collect only clippy diagnostics.
    let diagnostics: Vec<Diagnostic> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<ClippyMessage>(line).ok())
        .filter(|m| m.reason == "compiler-message")
        .filter_map(|m| m.message)
        .filter(|d| {
            d.code
                .as_ref()
                .map(|c| c.code.starts_with("clippy::"))
                .unwrap_or(false)
        })
        .collect();

    let error_count = diagnostics.iter().filter(|d| d.level == "error").count();
    let warning_count = diagnostics.iter().filter(|d| d.level == "warning").count();

    let mut out: Vec<SessionEvent> = Vec::new();
    let mut ordinal: u64 = 0;

    // Aggregate reliability observation first.
    let (variant, weight, evidence) = classify_run(error_count, warning_count);
    out.push(emit(
        &mut ordinal,
        &diagnosis_id,
        round,
        "clippy_run::reliable",
        variant,
        weight,
        evidence,
    ));

    // Per-lint observations.
    for d in &diagnostics {
        let lint_name = d.code.as_ref().map(|c| c.code.clone()).unwrap_or_default();
        if let Some((aspect, variant, weight)) = classify_diagnostic(&d.level, &lint_name) {
            let claim_key = format!("clippy:{lint_name}::{aspect}");
            let evidence = format!("{}: {}", d.level, d.message);
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                &claim_key,
                variant,
                weight,
                evidence,
            ));
        }
    }

    Ok(out)
}

fn classify_run(errors: usize, warnings: usize) -> (Hexavalent, f64, String) {
    let evidence = format!("{errors} error(s), {warnings} warning(s)");
    if errors > 0 {
        (Hexavalent::False, 1.0, evidence)
    } else if warnings > 20 {
        (Hexavalent::Doubtful, 0.7, evidence)
    } else if warnings > 0 {
        (Hexavalent::Probable, 0.7, evidence)
    } else {
        (Hexavalent::True, 0.9, evidence)
    }
}

const CORRECTNESS_LINTS: &[&str] = &[
    "absurd_extreme_comparisons",
    "approx_constant",
    "async_yields_async",
    "bad_bit_mask",
    "cast_slice_from_raw_parts",
    "clone_on_copy",
    "cmp_nan",
    "deprecated_semver",
    "derive_hash_xor_eq",
    "derive_ord_xor_partial_ord",
    "double_comparisons",
    "drop_copy",
    "drop_ref",
    "duration_subsec",
    "erasing_op",
    "float_cmp",
    "fn_address_comparisons",
    "forget_copy",
    "forget_non_drop",
    "forget_ref",
    "ifs_same_cond",
    "infinite_iter",
    "inherent_to_string_shadow_display",
    "inline_fn_without_body",
    "invalid_null_ptr_usage",
    "invalid_regex",
    "invisible_characters",
    "iter_next_loop",
    "iterator_step_by_zero",
    "let_underscore_lock",
    "logic_bug",
    "mem_replace_with_uninit",
    "min_max",
    "mismatched_target_os",
    "mistyped_array_indexes",
    "mut_from_ref",
    "mut_range_bound",
    "needless_bool",
    "non_octal_unix_permissions",
    "nonsensical_open_options",
    "not_unsafe_ptr_arg_deref",
    "option_env_unwrap",
    "out_of_bounds_indexing",
    "panicking_unwrap",
    "possible_missing_comma",
    "reversed_empty_ranges",
    "self_assignment",
    "serde_api_misuse",
    "size_of_in_element_count",
    "suspicious_splitn",
    "to_string_in_format_args",
    "transmute_undefined_repr",
    "transmuting_null",
    "unaligned_references",
    "undocumented_unsafe_blocks",
    "uninit_assumed_init",
    "unit_cmp",
    "unnecessary_cast",
    "unnecessary_filter_map",
    "unnecessary_fold",
    "unnecessary_lazy_evaluations",
    "unnecessary_mut_passed",
    "unnecessary_operation",
    "unnecessary_to_owned",
    "unnecessary_unwrap",
    "unreachable_code",
    "unused_io_amount",
    "unused_must_use",
    "unused_unit",
    "useless_asref",
    "useless_format",
    "vtable_address_comparisons",
    "while_immutable_condition",
    "wrong_pub_self_convention",
    "zero_divided_by_zero",
];

const SUSPICIOUS_LINTS: &[&str] = &[
    "almost_swapped",
    "arc_with_non_send_sync",
    "await_holding_lock",
    "await_holding_refcell_ref",
    "blanket_clippy_restriction_lints",
    "cast_lossless",
    "cast_possible_truncation",
    "cast_possible_wrap",
    "cast_precision_loss",
    "cast_sign_loss",
    "cognitive_complexity",
    "copy_iterator",
    "crate_in_macro_def",
    "debug_assert_with_mut_call",
    "decimal_literal_representation",
    "declare_interior_mutable_const",
    "default_numeric_fallback",
    "deprecated_cfg_attr",
    "diverging_sub_expression",
    "doc_markdown",
    "empty_loop",
    "eval_order_dependence",
    "float_arithmetic",
    "float_cmp_const",
    "format_in_format_args",
    "implicit_hasher",
    "inconsistent_digits_grouping",
    "index_refutable_slice",
    "integer_arithmetic",
    "items_after_statements",
    "large_digit_groups",
    "large_stack_arrays",
    "large_types_passed_by_value",
    "let_and_return",
    "let_underscore_must_use",
    "let_unit_value",
    "linkedlist",
    "macro_use_imports",
    "maybe_infinite_iter",
    "mem_forget",
    "missing_const_for_fn",
    "missing_enforced_import_renames",
    "missing_errors_doc",
    "missing_inline_in_public_items",
    "missing_panics_doc",
    "missing_safety_doc",
    "missing_spin_loop",
    "modulo_arithmetic",
    "modulo_one",
    "multiple_crate_versions",
    "multiple_inherent_impl",
    "mut_mut",
    "mutex_atomic",
    "mutex_integer",
    "needless_continue",
    "needless_for_each",
    "needless_pass_by_value",
    "non_ascii_literal",
    "nonstandard_macro_attributes",
    "option_if_let_else",
    "path_buf_push_overwrite",
    "print_stderr",
    "print_stdout",
    "rc_buffer",
    "rc_mutex",
    "redundant_feature_names",
    "rest_pat_in_fully_bound_structs",
    "same_item_push",
    "search_is_some",
    "self_named_module_files",
    "semicolon_if_nothing_returned",
    "single_char_pattern",
    "single_match",
    "single_match_else",
    "size_of_ref",
    "suboptimal_flops",
    "suspicious_arithmetic_impl",
    "suspicious_assignment_formatting",
    "suspicious_else_formatting",
    "suspicious_map",
    "suspicious_op_assign_impl",
    "suspicious_to_owned",
    "suspicious_unary_op_formatting",
    "todo",
    "trait_duplication_in_bounds",
    "type_repetition_in_bounds",
    "unimplemented",
    "unnecessary_self_imports",
    "unnecessary_wraps",
    "unpack_nested_variable_matches",
    "unreadable_literal",
    "unsafe_derive_deserialize",
    "unused_async",
    "unused_peekable",
    "unused_rounding",
    "unused_self",
    "use_self",
    "useless_let_if_seq",
    "useless_transmute",
    "verbose_bit_mask",
    "verbose_file_reads",
    "wildcard_dependencies",
    "wildcard_imports",
    "write_stderr",
    "write_stdout",
    "wrong_self_convention",
    "zero_prefixed_literal",
];

const PERF_LINTS: &[&str] = &[
    "box_collection",
    "boxed_local",
    "cmp_owned",
    "expect_fun_call",
    "format_collect",
    "inefficient_to_string",
    "iter_nth",
    "iter_nth_zero",
    "iter_ovf",
    "large_const_arrays",
    "large_enum_variant",
    "manual_memcpy",
    "manual_str_repeat",
    "needless_collect",
    "or_fun_call",
    "redundant_allocation",
    "redundant_clone",
    "slow_vector_initialization",
    "stable_sort_primitive",
    "to_string_in_format_args",
    "unnecessary_to_owned",
    "useless_vec",
    "vec_box",
];

fn get_leaf_lint_name(lint_name: &str) -> &str {
    lint_name.split("::").last().unwrap_or(lint_name)
}

/// Classify one diagnostic into `(aspect, variant, weight)`.
/// Returns `None` for `help` and unknown levels.
fn classify_diagnostic(level: &str, lint_name: &str) -> Option<(&'static str, Hexavalent, f64)> {
    match level {
        "error" => Some(("safe", Hexavalent::False, 1.0)),
        "warning" => {
            let leaf = get_leaf_lint_name(lint_name);

            // Check lookup tables first.
            if CORRECTNESS_LINTS.binary_search(&leaf).is_ok() {
                Some(("safe", Hexavalent::Doubtful, 0.8))
            } else if SUSPICIOUS_LINTS.binary_search(&leaf).is_ok() {
                Some(("safe", Hexavalent::Doubtful, 0.7))
            } else if PERF_LINTS.binary_search(&leaf).is_ok() {
                Some(("timely", Hexavalent::Doubtful, 0.6))
            } else {
                // Fall back to matching explicit substrings (primarily for custom mock prefixes in tests)
                if lint_name.contains("::correctness::") {
                    Some(("safe", Hexavalent::Doubtful, 0.8))
                } else if lint_name.contains("::suspicious::") {
                    Some(("safe", Hexavalent::Doubtful, 0.7))
                } else if lint_name.contains("::perf::") {
                    Some(("timely", Hexavalent::Doubtful, 0.6))
                } else {
                    // Default for style / complexity / pedantic / nursery / other lints
                    Some(("valuable", Hexavalent::Doubtful, 0.5))
                }
            }
        }
        "note" => Some(("valuable", Hexavalent::Unknown, 0.3)),
        _ => None, // help, unknown levels
    }
}

fn emit(
    ordinal: &mut u64,
    diagnosis_id: &str,
    round: u32,
    claim_key: &str,
    variant: Hexavalent,
    weight: f64,
    evidence: String,
) -> SessionEvent {
    let ord = *ordinal;
    *ordinal += 1;
    SessionEvent::ObservationAdded {
        ordinal: ord,
        source: SOURCE.to_string(),
        diagnosis_id: diagnosis_id.to_string(),
        round,
        claim_key: claim_key.to_string(),
        variant,
        weight,
        evidence: Some(evidence),
    }
}

fn sha256_hex(input: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input);
    let hash = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in hash.iter() {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(event: &SessionEvent) -> (&str, &str, Hexavalent, f64) {
        if let SessionEvent::ObservationAdded {
            source,
            claim_key,
            variant,
            weight,
            ..
        } = event
        {
            (source, claim_key, *variant, *weight)
        } else {
            panic!("expected ObservationAdded")
        }
    }

    #[test]
    fn empty_input_emits_clean_baseline() {
        let obs = clippy_to_observations(b"", 0).unwrap();
        assert_eq!(obs.len(), 1);
        let (_, claim, variant, weight) = extract(&obs[0]);
        assert_eq!(claim, "clippy_run::reliable");
        assert_eq!(variant, Hexavalent::True);
        assert!((weight - 0.9).abs() < 1e-9);
    }

    #[test]
    fn error_level_emits_safe_false() {
        let input = r#"{"reason":"compiler-message","message":{"level":"error","code":{"code":"clippy::panicking_unwrap"},"message":"unwrap on const None"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        // Summary + per-lint observation
        assert_eq!(obs.len(), 2);

        // Summary is F (errors > 0)
        let (_, _, sv, _) = extract(&obs[0]);
        assert_eq!(sv, Hexavalent::False);

        // Per-lint is safe F
        let (_, claim, variant, weight) = extract(&obs[1]);
        assert_eq!(claim, "clippy:clippy::panicking_unwrap::safe");
        assert_eq!(variant, Hexavalent::False);
        assert!((weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn style_warning_emits_valuable_doubtful() {
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::style::needless_return"},"message":"needless return"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let lint_obs = obs
            .iter()
            .find(|e| extract(e).1.contains("needless_return"))
            .unwrap();
        let (_, claim, variant, _) = extract(lint_obs);
        assert!(claim.ends_with("::valuable"));
        assert_eq!(variant, Hexavalent::Doubtful);
    }

    #[test]
    fn correctness_warning_emits_safe_doubtful() {
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::correctness::float_cmp"},"message":"strict float comparison"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let lint_obs = obs
            .iter()
            .find(|e| extract(e).1.contains("float_cmp"))
            .unwrap();
        let (_, claim, variant, weight) = extract(lint_obs);
        assert!(claim.ends_with("::safe"));
        assert_eq!(variant, Hexavalent::Doubtful);
        assert!((weight - 0.8).abs() < 1e-9);
    }

    #[test]
    fn perf_warning_emits_timely_aspect() {
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::perf::needless_collect"},"message":"unnecessary collect"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let lint_obs = obs
            .iter()
            .find(|e| extract(e).1.contains("needless_collect"))
            .unwrap();
        let (_, claim, _, _) = extract(lint_obs);
        assert!(claim.ends_with("::timely"), "got {claim}");
    }

    #[test]
    fn non_clippy_diagnostics_are_ignored() {
        // A plain rustc warning (no clippy:: prefix).
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"unused_variables"},"message":"unused variable"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        // Only the summary observation (clean baseline, no
        // clippy diagnostics seen).
        assert_eq!(obs.len(), 1);
    }

    #[test]
    fn help_level_is_ignored() {
        let input = r#"{"reason":"compiler-message","message":{"level":"help","code":{"code":"clippy::style::foo"},"message":"suggestion"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        // Only summary; help is skipped.
        assert_eq!(obs.len(), 1);
    }

    #[test]
    fn warning_count_summary_scales_with_count() {
        // 10 warnings → P variant (1-20 warnings band)
        let lines: Vec<String> = (0..10)
            .map(|i| format!(
                r#"{{"reason":"compiler-message","message":{{"level":"warning","code":{{"code":"clippy::style::w{i}"}},"message":"w"}}}}"#
            ))
            .collect();
        let input = lines.join("\n");
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let (_, _, sv, _) = extract(&obs[0]);
        assert_eq!(sv, Hexavalent::Probable);
    }

    #[test]
    fn many_warnings_emit_doubtful_summary() {
        let lines: Vec<String> = (0..25)
            .map(|i| format!(
                r#"{{"reason":"compiler-message","message":{{"level":"warning","code":{{"code":"clippy::style::w{i}"}},"message":"w"}}}}"#
            ))
            .collect();
        let input = lines.join("\n");
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let (_, _, sv, _) = extract(&obs[0]);
        assert_eq!(sv, Hexavalent::Doubtful);
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let input = concat!(
            "garbage line\n",
            r#"{"reason":"compiler-message","message":{"level":"error","code":{"code":"clippy::panic"},"message":"panic"}}"#,
            "\n",
        );
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        assert_eq!(obs.len(), 2); // summary + 1 diagnostic
    }

    #[test]
    fn round_trip_through_session_event() {
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::style::foo"},"message":"w"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 1).unwrap();
        for event in &obs {
            let json = serde_json::to_string(event).unwrap();
            let back: SessionEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *event);
        }
    }

    #[test]
    fn standard_real_world_clippy_warnings_are_mapped_correctly() {
        // Standard real-world clippy warnings without the artificial category subfolder.
        let correctness_input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::float_cmp"},"message":"strict float comparison"}}"#;
        let suspicious_input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::almost_swapped"},"message":"almost swapped"}}"#;
        let perf_input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::needless_collect"},"message":"unnecessary collect"}}"#;
        let style_input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::needless_return"},"message":"needless return"}}"#;

        // Correctness should map to safe aspect, weight 0.8
        let obs_c = clippy_to_observations(correctness_input.as_bytes(), 0).unwrap();
        let lint_c = obs_c
            .iter()
            .find(|e| extract(e).1.contains("float_cmp"))
            .unwrap();
        let (_, claim_c, variant_c, weight_c) = extract(lint_c);
        assert!(claim_c.ends_with("::safe"), "got claim: {claim_c}");
        assert_eq!(variant_c, Hexavalent::Doubtful);
        assert!((weight_c - 0.8).abs() < 1e-9);

        // Suspicious should map to safe aspect, weight 0.7
        let obs_s = clippy_to_observations(suspicious_input.as_bytes(), 0).unwrap();
        let lint_s = obs_s
            .iter()
            .find(|e| extract(e).1.contains("almost_swapped"))
            .unwrap();
        let (_, claim_s, variant_s, weight_s) = extract(lint_s);
        assert!(claim_s.ends_with("::safe"), "got claim: {claim_s}");
        assert_eq!(variant_s, Hexavalent::Doubtful);
        assert!((weight_s - 0.7).abs() < 1e-9);

        // Perf should map to timely aspect, weight 0.6
        let obs_p = clippy_to_observations(perf_input.as_bytes(), 0).unwrap();
        let lint_p = obs_p
            .iter()
            .find(|e| extract(e).1.contains("needless_collect"))
            .unwrap();
        let (_, claim_p, variant_p, weight_p) = extract(lint_p);
        assert!(claim_p.ends_with("::timely"), "got claim: {claim_p}");
        assert_eq!(variant_p, Hexavalent::Doubtful);
        assert!((weight_p - 0.6).abs() < 1e-9);

        // Style should map to valuable aspect, weight 0.5
        let obs_st = clippy_to_observations(style_input.as_bytes(), 0).unwrap();
        let lint_st = obs_st
            .iter()
            .find(|e| extract(e).1.contains("needless_return"))
            .unwrap();
        let (_, claim_st, variant_st, weight_st) = extract(lint_st);
        assert!(claim_st.ends_with("::valuable"), "got claim: {claim_st}");
        assert_eq!(variant_st, Hexavalent::Doubtful);
        assert!((weight_st - 0.5).abs() < 1e-9);
    }
}
