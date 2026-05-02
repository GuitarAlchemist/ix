//! Triage session support: plan parser + hexavalent priority helper.
//!
//! These are the pure, testable-in-isolation primitives that back
//! `handlers::triage_session_with_ctx`. Keeping them in a dedicated
//! module lets us unit-test the parser and the ranking logic without
//! standing up a `ServerContext` or a `SessionLog`.
//!
//! ## Responsibilities
//!
//! - Parse an LLM-authored JSON plan (strict mode, then lenient
//!   "extract between first `[` and last `]`" mode) into a
//!   `Vec<TriagePlanItem>`.
//! - Validate each item: tool name not blank, not `ix_triage_session`
//!   (recursion guard), confidence is a recognized hexavalent label.
//! - Assign a priority integer per item using the Demerzel tiebreak
//!   order `C > U > D > P > T > F` so sorting by priority descending
//!   matches `hexavalent_argmax`'s tiebreak semantics.
//! - Build a `HexavalentDistribution` from the aggregated plan so the
//!   caller can check `escalation_triggered` before dispatching.
//!
//! ## What this module does NOT do
//!
//! - No dispatching. The handler owns that.
//! - No sampling. The handler owns that.
//! - No session log access. The handler owns that.
//! - No Tokio, no I/O, no globals. Everything is a pure function over
//!   `&str` / owned JSON.

use ix_fuzzy::{FuzzyError, HexavalentDistribution};
use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The reserved tool name that a plan MUST NOT propose — the triage
/// tool itself. Hard-rejected by the parser regardless of what the
/// system prompt told the LLM.
pub const RECURSION_GUARD_TOOL: &str = "ix_triage_session";

/// A single plan item proposed by the LLM. The confidence is carried
/// as a parsed [`Hexavalent`] so downstream code never has to
/// re-interpret the raw label.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TriagePlanItem {
    /// MCP tool name to invoke, e.g. `"ix_stats"`.
    pub tool_name: String,
    /// Params object that will be passed to the tool as-is.
    pub params: Value,
    /// LLM-reported confidence in this step (one of T/P/U/D/F/C).
    pub confidence: Hexavalent,
    /// Short natural-language justification. Not load-bearing; kept
    /// for the final synthesis output.
    pub reason: String,
}

/// Errors surfaced by [`parse_plan`] and [`build_distribution`].
#[derive(Debug, thiserror::Error)]
pub enum TriageError {
    /// The LLM response wasn't valid JSON in either strict or lenient
    /// mode.
    #[error("plan parse: {0}")]
    Parse(String),
    /// A plan item was missing a required field or had the wrong
    /// type.
    #[error("plan item {index}: {reason}")]
    InvalidItem { index: usize, reason: String },
    /// The plan tried to propose the triage tool itself.
    #[error("plan item {index}: recursion guard — refusing to propose {RECURSION_GUARD_TOOL}")]
    Recursion { index: usize },
    /// The LLM emitted an unrecognized confidence label.
    #[error("plan item {index}: unrecognized confidence label {label:?}")]
    BadConfidence { index: usize, label: String },
    /// Building the fuzzy distribution from the plan failed.
    #[error("distribution: {0}")]
    Fuzzy(#[from] FuzzyError),
}

/// Parse an LLM response into a validated plan. Tries strict JSON
/// parsing first, then falls back to extracting the first `[ … ]`
/// block from the response (LLMs love wrapping JSON in prose).
///
/// Returns items in the order the LLM proposed them. Sorting by
/// hexavalent priority is the caller's job — the parser never
/// reorders.
///
/// # Validation
///
/// - `tool_name` must be a non-empty string
/// - `tool_name` must not be [`RECURSION_GUARD_TOOL`]
/// - `params` must be an object (or absent → defaults to `{}`)
/// - `confidence` must parse via [`parse_hexavalent_label`]
/// - `reason` must be a string (absent → defaults to `""`)
pub fn parse_plan(raw: &str) -> Result<Vec<TriagePlanItem>, TriageError> {
    let json_value: Value = match serde_json::from_str::<Value>(raw) {
        Ok(v) => v,
        Err(_) => {
            // Lenient fallback: extract the first `[` to the last `]`.
            let start = raw.find('[');
            let end = raw.rfind(']');
            match (start, end) {
                (Some(s), Some(e)) if e > s => {
                    let slice = &raw[s..=e];
                    serde_json::from_str::<Value>(slice).map_err(|e| {
                        TriageError::Parse(format!("lenient extraction failed: {e}"))
                    })?
                }
                _ => return Err(TriageError::Parse("response contains no JSON array".into())),
            }
        }
    };

    let arr = json_value
        .as_array()
        .ok_or_else(|| TriageError::Parse("top-level JSON is not an array".into()))?;

    let mut items = Vec::with_capacity(arr.len());
    for (index, raw_item) in arr.iter().enumerate() {
        items.push(parse_item(index, raw_item)?);
    }
    Ok(items)
}

fn parse_item(index: usize, value: &Value) -> Result<TriagePlanItem, TriageError> {
    let obj = value.as_object().ok_or_else(|| TriageError::InvalidItem {
        index,
        reason: "not an object".into(),
    })?;

    let tool_name = obj
        .get("tool_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| TriageError::InvalidItem {
            index,
            reason: "missing or non-string `tool_name`".into(),
        })?
        .to_string();

    if tool_name.is_empty() {
        return Err(TriageError::InvalidItem {
            index,
            reason: "tool_name is empty".into(),
        });
    }
    if tool_name == RECURSION_GUARD_TOOL {
        return Err(TriageError::Recursion { index });
    }

    let params = match obj.get("params") {
        Some(p) if p.is_object() => p.clone(),
        Some(p) if p.is_null() => Value::Object(Default::default()),
        None => Value::Object(Default::default()),
        Some(_) => {
            return Err(TriageError::InvalidItem {
                index,
                reason: "`params` must be an object".into(),
            })
        }
    };

    let confidence_label = obj
        .get("confidence")
        .and_then(|v| v.as_str())
        .ok_or_else(|| TriageError::InvalidItem {
            index,
            reason: "missing or non-string `confidence`".into(),
        })?;
    let confidence =
        parse_hexavalent_label(confidence_label).ok_or_else(|| TriageError::BadConfidence {
            index,
            label: confidence_label.to_string(),
        })?;

    let reason = obj
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(TriagePlanItem {
        tool_name,
        params,
        confidence,
        reason,
    })
}

/// Parse a textual hexavalent label into its [`Hexavalent`] variant.
/// Accepts both single-letter forms (`"T"`, `"P"`, …) and full words
/// (`"true"`, `"probable"`, …). Case-insensitive.
pub fn parse_hexavalent_label(label: &str) -> Option<Hexavalent> {
    let normalized = label.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "t" | "true" => Some(Hexavalent::True),
        "p" | "probable" => Some(Hexavalent::Probable),
        "u" | "unknown" => Some(Hexavalent::Unknown),
        "d" | "doubtful" => Some(Hexavalent::Doubtful),
        "f" | "false" => Some(Hexavalent::False),
        "c" | "contradictory" => Some(Hexavalent::Contradictory),
        _ => None,
    }
}

/// Priority integer matching the Demerzel tiebreak order
/// `C > U > D > P > T > F`. Higher returned value = higher priority.
/// Use with a descending sort to order plan items.
///
/// This mirrors `ix_fuzzy::hexavalent_argmax`'s tiebreak semantics so
/// callers that sort a plan here and then call `hexavalent_argmax` on
/// an aggregated distribution get consistent "what goes first" logic.
pub fn hex_priority(h: Hexavalent) -> u8 {
    match h {
        Hexavalent::Contradictory => 5,
        Hexavalent::Unknown => 4,
        Hexavalent::Doubtful => 3,
        Hexavalent::Probable => 2,
        Hexavalent::True => 1,
        Hexavalent::False => 0,
    }
}

/// Sort a plan by [`hex_priority`] descending, stable within ties.
/// The original item order is preserved for ties — the LLM's order is
/// the secondary signal when confidences are equal.
pub fn sort_plan_by_priority(plan: &mut [TriagePlanItem]) {
    plan.sort_by_key(|item| std::cmp::Reverse(hex_priority(item.confidence)));
}

/// Build a [`HexavalentDistribution`] from the plan's confidence
/// labels. Each plan item contributes equal weight `1/N` to its
/// variant; empty plans return the uniform distribution over all six
/// variants.
///
/// The resulting distribution can be passed to
/// `ix_fuzzy::escalation_triggered` to decide whether the plan as a
/// whole should be escalated to a human instead of dispatched.
pub fn build_distribution(plan: &[TriagePlanItem]) -> Result<HexavalentDistribution, TriageError> {
    if plan.is_empty() {
        return Ok(HexavalentDistribution::uniform(vec![
            Hexavalent::True,
            Hexavalent::Probable,
            Hexavalent::Unknown,
            Hexavalent::Doubtful,
            Hexavalent::False,
            Hexavalent::Contradictory,
        ])?);
    }

    let weight = 1.0f64 / plan.len() as f64;
    let mut t = 0.0;
    let mut p = 0.0;
    let mut u = 0.0;
    let mut d = 0.0;
    let mut f = 0.0;
    let mut c = 0.0;
    for item in plan {
        match item.confidence {
            Hexavalent::True => t += weight,
            Hexavalent::Probable => p += weight,
            Hexavalent::Unknown => u += weight,
            Hexavalent::Doubtful => d += weight,
            Hexavalent::False => f += weight,
            Hexavalent::Contradictory => c += weight,
        }
    }

    Ok(HexavalentDistribution::new(vec![
        (Hexavalent::True, t),
        (Hexavalent::Probable, p),
        (Hexavalent::Unknown, u),
        (Hexavalent::Doubtful, d),
        (Hexavalent::False, f),
        (Hexavalent::Contradictory, c),
    ])?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── parse_hexavalent_label ────────────────────────────────────

    #[test]
    fn parses_single_letter_labels() {
        assert_eq!(parse_hexavalent_label("T"), Some(Hexavalent::True));
        assert_eq!(parse_hexavalent_label("p"), Some(Hexavalent::Probable));
        assert_eq!(parse_hexavalent_label(" U "), Some(Hexavalent::Unknown));
        assert_eq!(parse_hexavalent_label("D"), Some(Hexavalent::Doubtful));
        assert_eq!(parse_hexavalent_label("F"), Some(Hexavalent::False));
        assert_eq!(parse_hexavalent_label("C"), Some(Hexavalent::Contradictory));
    }

    #[test]
    fn parses_full_word_labels() {
        assert_eq!(parse_hexavalent_label("true"), Some(Hexavalent::True));
        assert_eq!(
            parse_hexavalent_label("Probable"),
            Some(Hexavalent::Probable)
        );
        assert_eq!(
            parse_hexavalent_label("CONTRADICTORY"),
            Some(Hexavalent::Contradictory)
        );
    }

    #[test]
    fn rejects_bogus_labels() {
        assert_eq!(parse_hexavalent_label("maybe"), None);
        assert_eq!(parse_hexavalent_label(""), None);
        assert_eq!(parse_hexavalent_label("tru"), None);
    }

    // ── hex_priority ──────────────────────────────────────────────

    #[test]
    fn hex_priority_matches_demerzel_tiebreak() {
        // C > U > D > P > T > F
        assert!(hex_priority(Hexavalent::Contradictory) > hex_priority(Hexavalent::Unknown));
        assert!(hex_priority(Hexavalent::Unknown) > hex_priority(Hexavalent::Doubtful));
        assert!(hex_priority(Hexavalent::Doubtful) > hex_priority(Hexavalent::Probable));
        assert!(hex_priority(Hexavalent::Probable) > hex_priority(Hexavalent::True));
        assert!(hex_priority(Hexavalent::True) > hex_priority(Hexavalent::False));
    }

    // ── parse_plan — happy path ──────────────────────────────────

    #[test]
    fn strict_parse_accepts_well_formed_plan() {
        let raw = r#"[
            {"tool_name": "ix_stats", "params": {"data": [1,2,3]}, "confidence": "T", "reason": "baseline"},
            {"tool_name": "ix_fft", "params": {}, "confidence": "probable", "reason": "frequency check"}
        ]"#;
        let plan = parse_plan(raw).expect("parses");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].tool_name, "ix_stats");
        assert_eq!(plan[0].confidence, Hexavalent::True);
        assert_eq!(plan[1].tool_name, "ix_fft");
        assert_eq!(plan[1].confidence, Hexavalent::Probable);
    }

    #[test]
    fn lenient_parse_extracts_from_prose() {
        let raw = "Here is my plan:\n\n```json\n[{\"tool_name\":\"ix_stats\",\"confidence\":\"T\"}]\n```\n\nHope this helps!";
        let plan = parse_plan(raw).expect("lenient extraction works");
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].tool_name, "ix_stats");
    }

    #[test]
    fn parse_defaults_missing_params_to_empty_object() {
        let raw = r#"[{"tool_name": "ix_stats", "confidence": "T"}]"#;
        let plan = parse_plan(raw).expect("parses");
        assert_eq!(plan[0].params, json!({}));
        assert_eq!(plan[0].reason, "");
    }

    // ── parse_plan — rejection paths ─────────────────────────────

    #[test]
    fn rejects_non_array_top_level() {
        let raw = r#"{"plan": []}"#;
        match parse_plan(raw) {
            Err(TriageError::Parse(msg)) => assert!(msg.contains("not an array")),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_tool_name() {
        let raw = r#"[{"tool_name": "", "confidence": "T"}]"#;
        match parse_plan(raw) {
            Err(TriageError::InvalidItem { index, reason }) => {
                assert_eq!(index, 0);
                assert!(reason.contains("empty"));
            }
            other => panic!("expected InvalidItem, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_tool_name() {
        let raw = r#"[{"confidence": "T"}]"#;
        match parse_plan(raw) {
            Err(TriageError::InvalidItem { index, reason }) => {
                assert_eq!(index, 0);
                assert!(reason.contains("tool_name"));
            }
            other => panic!("expected InvalidItem, got {other:?}"),
        }
    }

    #[test]
    fn rejects_recursion_even_if_system_prompt_ignored() {
        let raw = r#"[
            {"tool_name": "ix_stats", "confidence": "T"},
            {"tool_name": "ix_triage_session", "confidence": "T"}
        ]"#;
        match parse_plan(raw) {
            Err(TriageError::Recursion { index }) => assert_eq!(index, 1),
            other => panic!("expected Recursion, got {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_confidence_label() {
        let raw = r#"[{"tool_name": "ix_stats", "confidence": "maybe-ish"}]"#;
        match parse_plan(raw) {
            Err(TriageError::BadConfidence { index, label }) => {
                assert_eq!(index, 0);
                assert_eq!(label, "maybe-ish");
            }
            other => panic!("expected BadConfidence, got {other:?}"),
        }
    }

    #[test]
    fn rejects_params_that_are_not_objects() {
        let raw = r#"[{"tool_name": "ix_stats", "params": [1,2,3], "confidence": "T"}]"#;
        match parse_plan(raw) {
            Err(TriageError::InvalidItem { index, reason }) => {
                assert_eq!(index, 0);
                assert!(reason.contains("object"));
            }
            other => panic!("expected InvalidItem, got {other:?}"),
        }
    }

    // ── sort_plan_by_priority ────────────────────────────────────

    #[test]
    fn sort_places_contradictory_first_then_unknown() {
        let mut plan = vec![
            item("ix_a", Hexavalent::True),
            item("ix_b", Hexavalent::Contradictory),
            item("ix_c", Hexavalent::Unknown),
            item("ix_d", Hexavalent::False),
        ];
        sort_plan_by_priority(&mut plan);
        let names: Vec<&str> = plan.iter().map(|i| i.tool_name.as_str()).collect();
        assert_eq!(names, vec!["ix_b", "ix_c", "ix_a", "ix_d"]);
    }

    #[test]
    fn sort_is_stable_within_ties() {
        let mut plan = vec![
            item("first", Hexavalent::True),
            item("second", Hexavalent::True),
            item("third", Hexavalent::True),
        ];
        sort_plan_by_priority(&mut plan);
        let names: Vec<&str> = plan.iter().map(|i| i.tool_name.as_str()).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
    }

    // ── build_distribution ───────────────────────────────────────

    #[test]
    fn build_distribution_weights_equal_per_item() {
        let plan = vec![
            item("ix_a", Hexavalent::True),
            item("ix_b", Hexavalent::True),
            item("ix_c", Hexavalent::Probable),
            item("ix_d", Hexavalent::Contradictory),
        ];
        let dist = build_distribution(&plan).expect("builds");
        assert!((dist.get(&Hexavalent::True) - 0.5).abs() < 1e-9);
        assert!((dist.get(&Hexavalent::Probable) - 0.25).abs() < 1e-9);
        assert!((dist.get(&Hexavalent::Contradictory) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn build_distribution_triggers_escalation_when_c_dominates() {
        use ix_fuzzy::escalation_triggered;
        let plan = vec![
            item("ix_a", Hexavalent::Contradictory),
            item("ix_b", Hexavalent::Contradictory),
            item("ix_c", Hexavalent::True),
        ];
        let dist = build_distribution(&plan).expect("builds");
        // C mass = 2/3 ≈ 0.667 >> 0.3 threshold
        assert!(escalation_triggered(&dist));
    }

    #[test]
    fn build_distribution_empty_plan_returns_uniform() {
        let dist = build_distribution(&[]).expect("uniform");
        // Six variants, uniform weight = 1/6 each.
        for h in [
            Hexavalent::True,
            Hexavalent::Probable,
            Hexavalent::Unknown,
            Hexavalent::Doubtful,
            Hexavalent::False,
            Hexavalent::Contradictory,
        ] {
            assert!((dist.get(&h) - (1.0 / 6.0)).abs() < 1e-9);
        }
    }

    // ── helpers ──────────────────────────────────────────────────

    fn item(name: &str, h: Hexavalent) -> TriagePlanItem {
        TriagePlanItem {
            tool_name: name.into(),
            params: json!({}),
            confidence: h,
            reason: String::new(),
        }
    }
}
