//! Stub harness adapter — Guitar Alchemist governance events →
//! `SessionEvent::ObservationAdded` stream.
//!
//! This is an intentionally minimal stub crate. The full adapter
//! requires a ga-side shim that converts ga's native event streams
//! (SignalR hub messages, .NET event logs, MCP server output)
//! into a canonical NDJSON input format. The shim is out of scope
//! for this crate — it lives in the ga repo.
//!
//! What this crate does: given canonical NDJSON input, project
//! it into `SessionEvent::ObservationAdded` records per the rules
//! in `demerzel/logic/harness-ga.md`.
//!
//! # Current state
//!
//! - ✅ Canonical NDJSON input parser
//! - ✅ Projection rules for all six event kinds (algedonic,
//!   constitutional, grammar, belief, seldon, compliance)
//! - ✅ Severity → variant mapping
//! - ⏸ ga-side shim (deferred — different repo)
//! - ⏸ Direct SignalR ingestion (deferred)
//! - ⏸ First real end-to-end integration test (deferred until
//!   the shim exists)

use ix_agent_core::SessionEvent;
use ix_types::Hexavalent;
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const SOURCE: &str = "ga";

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("input is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

/// Canonical NDJSON shape consumed by this adapter. Produced by
/// a ga-side shim (TBD) from ga's native event streams.
///
/// Fields match `demerzel/logic/harness-ga.md §"Proposed input shape"`.
#[derive(Debug, Clone, Deserialize)]
pub struct GaGovernanceEvent {
    /// Which governance category this event belongs to.
    pub kind: GaEventKind,
    /// Severity level from ga's emission side.
    pub severity: GaSeverity,
    /// Subject identifier — what this event is about. Used as
    /// part of the claim_key.
    pub subject: String,
    /// Optional evidence payload (kind-specific, pass-through).
    #[serde(default)]
    #[allow(dead_code)]
    pub evidence: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GaEventKind {
    Algedonic,
    Constitutional,
    Grammar,
    Belief,
    Seldon,
    Compliance,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GaSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Project a stream of canonical ga governance events into
/// `SessionEvent::ObservationAdded` records.
pub fn ga_to_observations(input: &[u8], round: u32) -> Result<Vec<SessionEvent>, AdapterError> {
    let text = std::str::from_utf8(input)?;
    let diagnosis_id = sha256_hex(input);

    let events: Vec<GaGovernanceEvent> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<GaGovernanceEvent>(line).ok())
        .collect();

    let out: Vec<SessionEvent> = events
        .iter()
        .enumerate()
        .map(|(i, event)| {
            let (claim_key, variant, weight) = project_event(event);
            SessionEvent::ObservationAdded {
                ordinal: i as u64,
                source: SOURCE.to_string(),
                diagnosis_id: diagnosis_id.clone(),
                round,
                claim_key,
                variant,
                weight,
                evidence: Some(format!(
                    "{:?}/{:?}: {}",
                    event.kind, event.severity, event.subject
                )),
            }
        })
        .collect();

    Ok(out)
}

/// The projection table, implementing the rules from
/// `demerzel/logic/harness-ga.md §"Proposed projection rules"`.
fn project_event(event: &GaGovernanceEvent) -> (String, Hexavalent, f64) {
    use GaEventKind::*;
    use GaSeverity::*;
    let subject = &event.subject;

    match (event.kind, event.severity) {
        // algedonic → ga:<subject>::reliable
        (Algedonic, Info) => (format!("ga:{subject}::reliable"), Hexavalent::True, 0.7),
        (Algedonic, Warning) => (format!("ga:{subject}::reliable"), Hexavalent::Doubtful, 0.6),
        (Algedonic, Error) => (format!("ga:{subject}::reliable"), Hexavalent::False, 0.8),
        (Algedonic, Critical) => (format!("ga:{subject}::reliable"), Hexavalent::False, 1.0),

        // constitutional → ga_constitution:<article>::safe or ::valuable
        (Constitutional, Info) => (
            format!("ga_constitution:{subject}::valuable"),
            Hexavalent::True,
            0.8,
        ),
        (Constitutional, Warning) => (
            format!("ga_constitution:{subject}::safe"),
            Hexavalent::Doubtful,
            0.7,
        ),
        (Constitutional, Error) | (Constitutional, Critical) => (
            format!("ga_constitution:{subject}::safe"),
            Hexavalent::False,
            1.0,
        ),

        // grammar → ga_grammar:<rule>::valuable or ::reliable
        (Grammar, Info) => (
            format!("ga_grammar:{subject}::valuable"),
            Hexavalent::Probable,
            0.6,
        ),
        (Grammar, Warning) => (
            format!("ga_grammar:{subject}::reliable"),
            Hexavalent::Doubtful,
            0.5,
        ),
        (Grammar, Error) | (Grammar, Critical) => (
            format!("ga_grammar:{subject}::reliable"),
            Hexavalent::False,
            0.8,
        ),

        // belief → ga_belief:<proposition>::reliable
        (Belief, Info) => (
            format!("ga_belief:{subject}::reliable"),
            Hexavalent::Probable,
            0.5,
        ),
        (Belief, Warning) => (
            format!("ga_belief:{subject}::reliable"),
            Hexavalent::Unknown,
            0.4,
        ),
        (Belief, Error) | (Belief, Critical) => (
            format!("ga_belief:{subject}::reliable"),
            Hexavalent::Doubtful,
            0.6,
        ),

        // seldon → ga_seldon:<plan>::timely or ::valuable
        (Seldon, Info) => (
            format!("ga_seldon:{subject}::timely"),
            Hexavalent::True,
            0.7,
        ),
        (Seldon, Warning) => (
            format!("ga_seldon:{subject}::timely"),
            Hexavalent::Doubtful,
            0.6,
        ),
        (Seldon, Error) | (Seldon, Critical) => (
            format!("ga_seldon:{subject}::valuable"),
            Hexavalent::False,
            0.8,
        ),

        // compliance → ga_compliance:<check>::reliable
        (Compliance, Info) => (
            format!("ga_compliance:{subject}::reliable"),
            Hexavalent::True,
            0.9,
        ),
        (Compliance, Warning) => (
            format!("ga_compliance:{subject}::reliable"),
            Hexavalent::Doubtful,
            0.7,
        ),
        (Compliance, Error) | (Compliance, Critical) => (
            format!("ga_compliance:{subject}::reliable"),
            Hexavalent::False,
            1.0,
        ),
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

    fn extract(event: &SessionEvent) -> (&str, Hexavalent, f64) {
        if let SessionEvent::ObservationAdded {
            claim_key,
            variant,
            weight,
            ..
        } = event
        {
            (claim_key, *variant, *weight)
        } else {
            panic!("expected ObservationAdded")
        }
    }

    #[test]
    fn algedonic_critical_emits_reliable_false_full_weight() {
        let input =
            r#"{"kind":"algedonic","severity":"critical","subject":"audit_panic","evidence":null}"#;
        let obs = ga_to_observations(input.as_bytes(), 1).unwrap();
        assert_eq!(obs.len(), 1);
        let (claim, variant, weight) = extract(&obs[0]);
        assert_eq!(claim, "ga:audit_panic::reliable");
        assert_eq!(variant, Hexavalent::False);
        assert!((weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn constitutional_info_emits_valuable_true() {
        let input =
            r#"{"kind":"constitutional","severity":"info","subject":"article_1","evidence":null}"#;
        let obs = ga_to_observations(input.as_bytes(), 1).unwrap();
        let (claim, variant, _) = extract(&obs[0]);
        assert_eq!(claim, "ga_constitution:article_1::valuable");
        assert_eq!(variant, Hexavalent::True);
    }

    #[test]
    fn constitutional_error_emits_safe_false() {
        let input =
            r#"{"kind":"constitutional","severity":"error","subject":"article_3","evidence":null}"#;
        let obs = ga_to_observations(input.as_bytes(), 1).unwrap();
        let (claim, variant, _) = extract(&obs[0]);
        assert_eq!(claim, "ga_constitution:article_3::safe");
        assert_eq!(variant, Hexavalent::False);
    }

    #[test]
    fn seldon_info_emits_timely_true() {
        let input = r#"{"kind":"seldon","severity":"info","subject":"plan_42","evidence":null}"#;
        let obs = ga_to_observations(input.as_bytes(), 1).unwrap();
        let (claim, variant, _) = extract(&obs[0]);
        assert_eq!(claim, "ga_seldon:plan_42::timely");
        assert_eq!(variant, Hexavalent::True);
    }

    #[test]
    fn multi_event_stream_preserves_order_and_ordinals() {
        let input = concat!(
            r#"{"kind":"algedonic","severity":"info","subject":"a"}"#,
            "\n",
            r#"{"kind":"belief","severity":"warning","subject":"b"}"#,
            "\n",
            r#"{"kind":"compliance","severity":"critical","subject":"c"}"#,
            "\n",
        );
        let obs = ga_to_observations(input.as_bytes(), 5).unwrap();
        assert_eq!(obs.len(), 3);
        for (i, event) in obs.iter().enumerate() {
            if let SessionEvent::ObservationAdded { ordinal, round, .. } = event {
                assert_eq!(*ordinal, i as u64);
                assert_eq!(*round, 5);
            }
        }
    }

    #[test]
    fn malformed_lines_are_silently_skipped() {
        let input = concat!(
            "not json at all\n",
            r#"{"kind":"compliance","severity":"info","subject":"ok"}"#,
            "\n",
            r#"{"kind":"unknown_kind","severity":"info","subject":"x"}"#,
            "\n",
        );
        let obs = ga_to_observations(input.as_bytes(), 1).unwrap();
        // Only the valid line produces an observation.
        assert_eq!(obs.len(), 1);
    }

    #[test]
    fn round_trip_through_session_event() {
        let input = r#"{"kind":"algedonic","severity":"warning","subject":"x"}"#;
        let obs = ga_to_observations(input.as_bytes(), 1).unwrap();
        for event in &obs {
            let json = serde_json::to_string(event).unwrap();
            let back: SessionEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *event);
        }
    }
}
