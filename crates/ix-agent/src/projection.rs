//! Project a stream of [`SessionEvent`]s into a set of
//! [`HexObservation`]s for CRDT merging.
//!
//! This is Step 4 of the Phase 1 Path C implementation. It closes
//! the feedback channel specified in
//! `demerzel/logic/hex-merge.md`: ix's execution outcomes become
//! observations that flow back into the next round's diagnosis via
//! the G-Set merge function in `ix-fuzzy::observations`.
//!
//! # Mapping rules
//!
//! The SessionEvent stream is walked once. A correlation table from
//! `ordinal → AgentAction::InvokeTool` is built from
//! `ActionProposed` events so that terminal events (Completed,
//! Blocked, Failed, Replaced) can recover the tool_name and
//! target_hint they refer to. Events without a corresponding
//! ActionProposed (e.g., synthetic events emitted by the demo
//! scenarios) are skipped.
//!
//! | SessionEvent | Emits | Rationale |
//! |---|---|---|
//! | `ActionProposed` | none | Proposal is not yet an outcome |
//! | `ActionCompleted` | `<tool>::valuable = T (0.9)` | Execution verified value |
//! | `ActionBlocked(ApprovalRequired)` | `<tool>::safe = F (1.0)` | Tier-3 classifier refusal |
//! | `ActionBlocked(LoopDetected)` | `<tool>::valuable = D (0.7)` | Repeated calls suggest doubtful value |
//! | `ActionBlocked(BlastRadiusTooLarge)` | `<tool>::safe = F (1.0)` | Safety refusal |
//! | `ActionBlocked(other)` | `<tool>::valuable = F (0.9)` | Policy refutation |
//! | `ActionReplaced` | `<original_tool>::valuable = D (0.6)` | Middleware thought the original was worse |
//! | `ActionFailed` | `<tool>::valuable = F (0.8)` | Handler refuted the action |
//! | `MetadataMounted` | none | Implementation detail |
//! | `BeliefChanged` | none | Already hexavalent, not a direct observation (future: re-emit) |
//! | `ObservationAdded` | passthrough | Caller-emitted observations are preserved |
//!
//! The weights are specified in `demerzel/logic/hex-merge.md`
//! §"What this module does NOT do" — they are governance decisions,
//! not math. Adjusting them is a Demerzel PR.
//!
//! # Pure function, no I/O
//!
//! `events_to_observations` does not touch the filesystem, the
//! session log slot, or any globals. It is a deterministic pure
//! function from `&[SessionEvent]` to `Vec<HexObservation>` — the
//! caller provides the event slice (typically from
//! `SessionLog::events()` or a cached in-memory vec) and the
//! provenance fields (source name, diagnosis_id, round).

use std::collections::HashMap;

use ix_agent_core::event::BlockCode;
use ix_agent_core::{AgentAction, SessionEvent};
use ix_fuzzy::observations::HexObservation;
use ix_types::Hexavalent;

/// Weight for a `T` observation derived from `ActionCompleted`.
pub const WEIGHT_COMPLETED: f64 = 0.9;
/// Weight for an `F` observation derived from approval / blast-
/// radius blocks. Full weight because the middleware is
/// authoritative on safety.
pub const WEIGHT_SAFETY_BLOCK: f64 = 1.0;
/// Weight for a `D` observation derived from loop-detect blocks.
/// Lower than other blocks because loop detection reflects
/// repeated-call patterns, not a hard safety or policy refusal.
pub const WEIGHT_LOOP_BLOCK: f64 = 0.7;
/// Weight for an `F` observation derived from non-safety policy
/// blocks.
pub const WEIGHT_POLICY_BLOCK: f64 = 0.9;
/// Weight for an `F` observation derived from `ActionFailed`.
/// Lower than a policy block because handler failures can be
/// transient (network glitches, timeouts) rather than true value
/// refutation.
pub const WEIGHT_FAILED: f64 = 0.8;
/// Weight for a `D` observation derived from `ActionReplaced`.
/// Soft doubt — the middleware thought the original was worse but
/// didn't refuse it outright.
pub const WEIGHT_REPLACED: f64 = 0.6;

/// Build a vector of [`HexObservation`]s from a slice of session
/// events. See the module-level docs for the mapping rules.
///
/// `source`, `diagnosis_id`, and `round` are stamped into every
/// emitted observation. They must match the caller's notion of
/// "which run this was."
///
/// The returned observations' `ordinal` field is the SessionEvent's
/// own ordinal — this preserves the G-Set dedup key across
/// re-projections of the same log.
pub fn events_to_observations(
    events: &[SessionEvent],
    source: &str,
    diagnosis_id: &str,
    round: u32,
) -> Vec<HexObservation> {
    // Step 1: build the ordinal → action map from ActionProposed
    // events. Only InvokeTool actions produce observations; other
    // action variants (EmitObservation, RequestApproval, Return)
    // are not tool invocations and don't project into the
    // valuable/safe/etc. claim space.
    let mut proposed: HashMap<u64, (String, Option<String>)> = HashMap::new();
    for event in events {
        if let SessionEvent::ActionProposed {
            ordinal,
            action:
                AgentAction::InvokeTool {
                    tool_name,
                    target_hint,
                    ..
                },
        } = event
        {
            proposed.insert(*ordinal, (tool_name.clone(), target_hint.clone()));
        }
    }

    // Step 2: walk events in order, emitting observations for
    // terminal variants. The ordinal in the emitted observation is
    // cast from u64 to u32 — if it overflows (sessions longer than
    // 4 billion events), we saturate. That's theoretically possible
    // but practically never encountered.
    let mut out = Vec::new();
    for event in events {
        match event {
            SessionEvent::ActionCompleted { ordinal, .. } => {
                if let Some((tool_name, target_hint)) = proposed.get(ordinal) {
                    out.push(obs(
                        source,
                        diagnosis_id,
                        round,
                        *ordinal,
                        tool_name,
                        target_hint.as_deref(),
                        "valuable",
                        Hexavalent::True,
                        WEIGHT_COMPLETED,
                        Some("action_completed".to_string()),
                    ));
                }
            }

            SessionEvent::ActionBlocked {
                ordinal,
                code,
                reason,
                emitted_by,
            } => {
                if let Some((tool_name, target_hint)) = proposed.get(ordinal) {
                    let (aspect, variant, weight) = match code {
                        BlockCode::ApprovalRequired | BlockCode::BlastRadiusTooLarge => {
                            ("safe", Hexavalent::False, WEIGHT_SAFETY_BLOCK)
                        }
                        BlockCode::LoopDetected => {
                            ("valuable", Hexavalent::Doubtful, WEIGHT_LOOP_BLOCK)
                        }
                        _ => ("valuable", Hexavalent::False, WEIGHT_POLICY_BLOCK),
                    };
                    out.push(obs(
                        source,
                        diagnosis_id,
                        round,
                        *ordinal,
                        tool_name,
                        target_hint.as_deref(),
                        aspect,
                        variant,
                        weight,
                        Some(format!(
                            "action_blocked({code:?}) by {emitted_by}: {reason}"
                        )),
                    ));
                }
            }

            SessionEvent::ActionFailed { ordinal, error } => {
                if let Some((tool_name, target_hint)) = proposed.get(ordinal) {
                    out.push(obs(
                        source,
                        diagnosis_id,
                        round,
                        *ordinal,
                        tool_name,
                        target_hint.as_deref(),
                        "valuable",
                        Hexavalent::False,
                        WEIGHT_FAILED,
                        Some(format!("action_failed: {error}")),
                    ));
                }
            }

            SessionEvent::ActionReplaced {
                ordinal,
                original,
                emitted_by,
                ..
            } => {
                // The ORIGINAL action was the one the agent proposed
                // and the middleware doubted. Emit a D observation
                // on it (not on the replacement, which is the
                // middleware's preferred version).
                if let AgentAction::InvokeTool {
                    tool_name,
                    target_hint,
                    ..
                } = original
                {
                    out.push(obs(
                        source,
                        diagnosis_id,
                        round,
                        *ordinal,
                        tool_name,
                        target_hint.as_deref(),
                        "valuable",
                        Hexavalent::Doubtful,
                        WEIGHT_REPLACED,
                        Some(format!("action_replaced by {emitted_by}")),
                    ));
                }
            }

            // Pass through caller-emitted observations unchanged.
            // This is what lets the triage handler merge its own
            // plan observations with prior-round observations in a
            // single merge call: feed both streams through the
            // projection, get one unified list back.
            SessionEvent::ObservationAdded {
                source: o_source,
                diagnosis_id: o_diagnosis_id,
                round: o_round,
                ordinal: o_ordinal,
                claim_key,
                variant,
                weight,
                evidence,
            } => {
                // SessionEvent::ObservationAdded carries a u64 ordinal
                // (matches the rest of SessionEvent for consistency),
                // but HexObservation uses u32 to stay compact. Cast
                // with saturation — 4B events per session is
                // unreachable in practice.
                out.push(HexObservation {
                    source: o_source.clone(),
                    diagnosis_id: o_diagnosis_id.clone(),
                    round: *o_round,
                    ordinal: (*o_ordinal).min(u32::MAX as u64) as u32,
                    claim_key: claim_key.clone(),
                    variant: *variant,
                    weight: *weight,
                    evidence: evidence.clone(),
                });
            }

            // Intentionally ignored variants.
            SessionEvent::ActionProposed { .. } => {} // proposals are not outcomes
            SessionEvent::MetadataMounted { .. } => {} // implementation detail
            SessionEvent::BeliefChanged { .. } => {}  // future: re-emit as observation
        }
    }
    out
}

/// Internal constructor. Builds a claim_key from `action_key` +
/// `aspect`, saturating-casts the ordinal, and fills in the fields
/// consistently.
#[allow(clippy::too_many_arguments)]
fn obs(
    source: &str,
    diagnosis_id: &str,
    round: u32,
    u64_ordinal: u64,
    tool_name: &str,
    target_hint: Option<&str>,
    aspect: &str,
    variant: Hexavalent,
    weight: f64,
    evidence: Option<String>,
) -> HexObservation {
    let action_key = match target_hint {
        Some(t) if !t.is_empty() => format!("{tool_name}:{t}"),
        _ => tool_name.to_string(),
    };
    HexObservation {
        source: source.to_string(),
        diagnosis_id: diagnosis_id.to_string(),
        round,
        ordinal: u64_ordinal.min(u32::MAX as u64) as u32,
        claim_key: format!("{action_key}::{aspect}"),
        variant,
        weight,
        evidence,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ix_agent_core::{ActionError, AgentAction, SessionEvent};
    use serde_json::json;

    fn propose(ordinal: u64, tool: &str, target: Option<&str>) -> SessionEvent {
        SessionEvent::ActionProposed {
            ordinal,
            action: AgentAction::InvokeTool {
                tool_name: tool.to_string(),
                params: json!({}),
                ordinal,
                target_hint: target.map(str::to_string),
            },
        }
    }

    fn completed(ordinal: u64) -> SessionEvent {
        SessionEvent::ActionCompleted {
            ordinal,
            value: json!({}),
        }
    }

    fn blocked(ordinal: u64, code: BlockCode, reason: &str) -> SessionEvent {
        SessionEvent::ActionBlocked {
            ordinal,
            code,
            reason: reason.to_string(),
            emitted_by: "test".to_string(),
        }
    }

    fn failed(ordinal: u64) -> SessionEvent {
        SessionEvent::ActionFailed {
            ordinal,
            error: ActionError::Exec("boom".to_string()),
        }
    }

    fn project(events: &[SessionEvent]) -> Vec<HexObservation> {
        events_to_observations(events, "ix", "test-diagnosis", 1)
    }

    // ── ActionCompleted mapping ────────────────────────────────────

    #[test]
    fn completed_emits_valuable_true() {
        let events = vec![propose(0, "ix_stats", None), completed(0)];
        let obs = project(&events);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].claim_key, "ix_stats::valuable");
        assert_eq!(obs[0].variant, Hexavalent::True);
        assert_eq!(obs[0].weight, WEIGHT_COMPLETED);
        assert_eq!(obs[0].source, "ix");
        assert_eq!(obs[0].diagnosis_id, "test-diagnosis");
        assert_eq!(obs[0].round, 1);
    }

    #[test]
    fn completed_with_target_hint_includes_it_in_action_key() {
        let events = vec![
            propose(5, "ix_context_walk", Some("crates/ix-math")),
            completed(5),
        ];
        let obs = project(&events);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].claim_key, "ix_context_walk:crates/ix-math::valuable");
    }

    #[test]
    fn completed_without_propose_is_dropped() {
        // A completion with no corresponding proposal (corrupt
        // or synthetic log) has no tool info. Drop silently.
        let events = vec![completed(42)];
        let obs = project(&events);
        assert!(obs.is_empty());
    }

    // ── ActionBlocked by approval / blast-radius ──────────────────

    #[test]
    fn approval_required_emits_safe_false() {
        let events = vec![
            propose(0, "ix_delete", None),
            blocked(0, BlockCode::ApprovalRequired, "Tier 3 needs approval"),
        ];
        let obs = project(&events);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].claim_key, "ix_delete::safe");
        assert_eq!(obs[0].variant, Hexavalent::False);
        assert_eq!(obs[0].weight, WEIGHT_SAFETY_BLOCK);
    }

    #[test]
    fn blast_radius_too_large_emits_safe_false() {
        let events = vec![
            propose(0, "ix_rewrite_universe", None),
            blocked(0, BlockCode::BlastRadiusTooLarge, "too big"),
        ];
        let obs = project(&events);
        assert_eq!(obs[0].claim_key, "ix_rewrite_universe::safe");
        assert_eq!(obs[0].variant, Hexavalent::False);
    }

    // ── ActionBlocked by loop-detect ──────────────────────────────

    #[test]
    fn loop_detected_emits_valuable_doubtful() {
        let events = vec![
            propose(0, "ix_stats", None),
            blocked(0, BlockCode::LoopDetected, "too many calls"),
        ];
        let obs = project(&events);
        assert_eq!(obs[0].claim_key, "ix_stats::valuable");
        assert_eq!(obs[0].variant, Hexavalent::Doubtful);
        assert_eq!(obs[0].weight, WEIGHT_LOOP_BLOCK);
    }

    // ── ActionBlocked by other policies ───────────────────────────

    #[test]
    fn other_block_emits_valuable_false() {
        let events = vec![
            propose(0, "ix_stats", None),
            blocked(0, BlockCode::PolicyDenied, "forbidden"),
        ];
        let obs = project(&events);
        assert_eq!(obs[0].claim_key, "ix_stats::valuable");
        assert_eq!(obs[0].variant, Hexavalent::False);
        assert_eq!(obs[0].weight, WEIGHT_POLICY_BLOCK);
    }

    // ── ActionFailed ──────────────────────────────────────────────

    #[test]
    fn failed_emits_valuable_false_at_failed_weight() {
        let events = vec![propose(0, "ix_stats", None), failed(0)];
        let obs = project(&events);
        assert_eq!(obs[0].claim_key, "ix_stats::valuable");
        assert_eq!(obs[0].variant, Hexavalent::False);
        assert_eq!(obs[0].weight, WEIGHT_FAILED);
        assert!(obs[0].weight < WEIGHT_POLICY_BLOCK);
    }

    // ── ActionReplaced ────────────────────────────────────────────

    #[test]
    fn replaced_emits_valuable_doubtful_on_original() {
        let original = AgentAction::InvokeTool {
            tool_name: "ix_naive".to_string(),
            params: json!({}),
            ordinal: 7,
            target_hint: None,
        };
        let replacement = AgentAction::InvokeTool {
            tool_name: "ix_improved".to_string(),
            params: json!({}),
            ordinal: 7,
            target_hint: None,
        };
        let events = vec![SessionEvent::ActionReplaced {
            ordinal: 7,
            original,
            replacement,
            emitted_by: "middleware".to_string(),
        }];
        let obs = project(&events);
        assert_eq!(obs.len(), 1);
        // Observation is on the ORIGINAL tool, not the replacement.
        assert_eq!(obs[0].claim_key, "ix_naive::valuable");
        assert_eq!(obs[0].variant, Hexavalent::Doubtful);
        assert_eq!(obs[0].weight, WEIGHT_REPLACED);
    }

    // ── Passthrough of pre-existing observations ─────────────────

    #[test]
    fn observation_added_passes_through_unchanged() {
        let input_obs = SessionEvent::ObservationAdded {
            ordinal: 99,
            source: "tars".to_string(),
            diagnosis_id: "other-diagnosis".to_string(),
            round: 0,
            claim_key: "git_gc::reversible".to_string(),
            variant: Hexavalent::Probable,
            weight: 0.77,
            evidence: Some("tars diagnosis".to_string()),
        };
        let obs = project(&[input_obs]);
        assert_eq!(obs.len(), 1);
        // Passthrough preserves source/diagnosis/round — the caller's
        // fields take precedence, NOT our projection defaults.
        assert_eq!(obs[0].source, "tars");
        assert_eq!(obs[0].diagnosis_id, "other-diagnosis");
        assert_eq!(obs[0].round, 0);
        assert_eq!(obs[0].ordinal, 99);
        assert_eq!(obs[0].claim_key, "git_gc::reversible");
        assert_eq!(obs[0].variant, Hexavalent::Probable);
        assert!((obs[0].weight - 0.77).abs() < 1e-9);
        assert_eq!(obs[0].evidence, Some("tars diagnosis".to_string()));
    }

    // ── Ignored variants ─────────────────────────────────────────

    #[test]
    fn metadata_mounted_is_ignored() {
        let events = vec![SessionEvent::MetadataMounted {
            ordinal: 0,
            path: "approval/verdict".to_string(),
            value: json!("ok"),
            emitted_by: "ix_approval".to_string(),
        }];
        assert!(project(&events).is_empty());
    }

    #[test]
    fn belief_changed_is_ignored_for_now() {
        let events = vec![SessionEvent::BeliefChanged {
            ordinal: 0,
            proposition: "api_stable".to_string(),
            old: Some(Hexavalent::Unknown),
            new: Hexavalent::Probable,
            evidence: json!({}),
        }];
        assert!(project(&events).is_empty());
    }

    #[test]
    fn bare_action_proposed_is_ignored() {
        // A proposal with no outcome is not an observation — the
        // action hasn't been evaluated yet.
        let events = vec![propose(0, "ix_stats", None)];
        assert!(project(&events).is_empty());
    }

    // ── End-to-end realistic session ─────────────────────────────

    #[test]
    fn realistic_session_projects_mixed_observations() {
        // A triage session that proposed 4 actions: one succeeded,
        // one was approval-blocked, one was loop-detected, one
        // failed. Should emit 4 distinct observations with the
        // right shapes.
        let events = vec![
            propose(0, "ix_stats", Some("baseline")),
            completed(0),
            propose(1, "ix_delete_all", None),
            blocked(1, BlockCode::ApprovalRequired, "Tier 3"),
            propose(2, "ix_stats", Some("baseline")), // same tool+target
            blocked(2, BlockCode::LoopDetected, "4th call"),
            propose(3, "ix_fft", None),
            failed(3),
        ];
        let obs = project(&events);
        assert_eq!(obs.len(), 4);

        // Find each by claim_key for stable assertions.
        let find = |claim: &str, ord: u32| {
            obs.iter()
                .find(|o| o.claim_key == claim && o.ordinal == ord)
                .unwrap_or_else(|| panic!("missing {claim} at ord {ord}"))
        };
        assert_eq!(
            find("ix_stats:baseline::valuable", 0).variant,
            Hexavalent::True
        );
        assert_eq!(find("ix_delete_all::safe", 1).variant, Hexavalent::False);
        assert_eq!(
            find("ix_stats:baseline::valuable", 2).variant,
            Hexavalent::Doubtful
        );
        assert_eq!(find("ix_fft::valuable", 3).variant, Hexavalent::False);
    }

    #[test]
    fn projection_roundtrips_through_merge() {
        // Smoke test: the output of events_to_observations should
        // be a valid input to ix_fuzzy::observations::merge. If the
        // types drift, this test catches it.
        use ix_fuzzy::observations::merge_all;

        let events = vec![
            propose(0, "ix_stats", None),
            completed(0),
            propose(1, "ix_fft", None),
            failed(1),
        ];
        let obs = project(&events);
        let state = merge_all(&obs).expect("merge succeeds on projection output");
        // Two disjoint claim_keys → no contradictions.
        assert_eq!(state.contradictions.len(), 0);
        // Four observations in, four observations out.
        assert_eq!(state.observations.len(), 2);
    }
}
