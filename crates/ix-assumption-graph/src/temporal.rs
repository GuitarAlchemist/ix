//! Phase 3 — the belief-time axis.
//!
//! An append-only [`BeliefLog`] of [`BeliefEvent`]s records how a claim's fused
//! verdict changes over *belief-time* (the design doc §5 / contract §5 third
//! axis). Point-in-time reconstruction is [`BeliefLog::belief_at`]; what changed
//! between two times is [`BeliefLog::diff`]. This is the "semantic story over a
//! long period" substrate — longitudinal revision, not snapshot storage.
//!
//! [`AssumptionGraph::revise`] is one turn of the loop: fuse current evidence,
//! and append a [`BeliefEvent`] for every claim whose verdict differs from the
//! log's latest belief. The autonomous evidence-gathering loop (scheduled
//! `/deep-research` re-verification, test re-runs) is the orchestration that
//! *drives* `revise` repeatedly over time — it lives outside this crate.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::graph::{normalize_claim, AssumptionGraph};
use crate::opinion::{from_hexavalent, Opinion};

/// Schema version of the `belief_event` record (contract §2.3).
pub const BELIEF_EVENT_SCHEMA_VERSION: u32 = 1;

/// Stable id for a *claim-level* belief — the fused-verdict identity, distinct
/// from a single annotation's node id. `sha256:` over the normalized claim, so
/// the same claim tracks one belief timeline regardless of how many annotations
/// assert it.
pub fn claim_id(claim: &str) -> String {
    let key = format!("claim:{}", normalize_claim(claim));
    format!("sha256:{:x}", Sha256::digest(key.as_bytes()))
}

/// One append-only revision of a claim's epistemic state at a belief-time.
/// Matches the `belief_event` record in `assumption-graph.schema.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefEvent {
    pub schema_version: u32,
    /// Always `"belief_event"` — the JSONL record discriminator.
    pub record: String,
    /// Claim-level id (see [`claim_id`]).
    pub node_id: String,
    /// Belief-time of this revision.
    pub at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub from_truth_value: Option<Hexavalent>,
    pub to_truth_value: Hexavalent,
    /// `None` when the verdict is `C` (Contradictory) — it lies outside the
    /// subjective-logic simplex (see [`Opinion`]).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub to_opinion: Option<Opinion>,
    /// What caused the revision (e.g. `"deep-research-reverify"`, `"test-run"`).
    pub trigger: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub evidence: Option<String>,
}

impl BeliefEvent {
    /// Construct a well-formed event (sets `schema_version` + `record`).
    pub fn new(
        node_id: String,
        at: DateTime<Utc>,
        from_truth_value: Option<Hexavalent>,
        to_truth_value: Hexavalent,
        to_opinion: Option<Opinion>,
        trigger: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: BELIEF_EVENT_SCHEMA_VERSION,
            record: "belief_event".to_string(),
            node_id,
            at,
            from_truth_value,
            to_truth_value,
            to_opinion,
            trigger: trigger.into(),
            evidence: None,
        }
    }
}

/// The epistemic state of one claim at a point in belief-time.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BeliefSnapshot {
    pub truth_value: Hexavalent,
    pub opinion: Option<Opinion>,
    pub at: DateTime<Utc>,
}

/// A change in a claim's verdict between two belief-times.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BeliefChange {
    pub node_id: String,
    pub from: Option<Hexavalent>,
    pub to: Hexavalent,
    pub at: DateTime<Utc>,
}

/// Append-only log of belief revisions. Kept sorted by `(at, node_id)` so
/// replay is deterministic.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BeliefLog {
    events: Vec<BeliefEvent>,
}

impl BeliefLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event, keeping the log sorted for deterministic replay.
    pub fn record(&mut self, event: BeliefEvent) {
        self.events.push(event);
        self.events
            .sort_by(|a, b| a.at.cmp(&b.at).then_with(|| a.node_id.cmp(&b.node_id)));
    }

    pub fn events(&self) -> &[BeliefEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Reconstruct the belief state at time `t`: for each claim, the latest
    /// event with `at <= t` wins. Events are sorted ascending, so iterating in
    /// order and overwriting yields the most recent applicable state.
    pub fn belief_at(&self, t: DateTime<Utc>) -> BTreeMap<String, BeliefSnapshot> {
        let mut state = BTreeMap::new();
        for e in &self.events {
            if e.at <= t {
                state.insert(
                    e.node_id.clone(),
                    BeliefSnapshot {
                        truth_value: e.to_truth_value,
                        opinion: e.to_opinion,
                        at: e.at,
                    },
                );
            }
        }
        state
    }

    /// Claims whose verdict differs between `t1` and `t2` — the "what flipped,
    /// and when" view. Ordered by claim id.
    pub fn diff(&self, t1: DateTime<Utc>, t2: DateTime<Utc>) -> Vec<BeliefChange> {
        let s1 = self.belief_at(t1);
        let s2 = self.belief_at(t2);
        let keys: BTreeSet<&String> = s1.keys().chain(s2.keys()).collect();

        let mut out = Vec::new();
        for k in keys {
            let from = s1.get(k).map(|s| s.truth_value);
            match s2.get(k) {
                Some(snap) if Some(snap.truth_value) != from => out.push(BeliefChange {
                    node_id: k.clone(),
                    from,
                    to: snap.truth_value,
                    at: snap.at,
                }),
                _ => {}
            }
        }
        out
    }

    /// Serialize to JSONL (one event per line).
    pub fn to_jsonl(&self) -> String {
        self.events
            .iter()
            .map(|e| serde_json::to_string(e).expect("BeliefEvent serializes"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Parse from JSONL (blank lines ignored). Re-sorts for deterministic replay.
    pub fn from_jsonl(s: &str) -> Result<Self, serde_json::Error> {
        let mut log = BeliefLog::new();
        for line in s.lines().filter(|l| !l.trim().is_empty()) {
            log.events.push(serde_json::from_str(line)?);
        }
        log.events
            .sort_by(|a, b| a.at.cmp(&b.at).then_with(|| a.node_id.cmp(&b.node_id)));
        Ok(log)
    }
}

impl AssumptionGraph {
    /// One turn of the longitudinal loop. Fuses current evidence and appends a
    /// [`BeliefEvent`] to `log` for every claim whose fused verdict differs from
    /// the log's latest belief as of `at`. Returns the events appended (empty if
    /// nothing changed — so repeated calls on unchanged evidence are no-ops).
    pub fn revise(
        &self,
        log: &mut BeliefLog,
        at: DateTime<Utc>,
        trigger: &str,
    ) -> Result<Vec<BeliefEvent>, ix_fuzzy::FuzzyError> {
        let prior = log.belief_at(at);
        let mut appended = Vec::new();
        for fc in self.fuse()? {
            let id = claim_id(&fc.claim);
            let prior_tv = prior.get(&id).map(|s| s.truth_value);
            if prior_tv != Some(fc.verdict) {
                let opinion = Opinion::from_truth(from_hexavalent(fc.verdict), fc.confidence);
                let event =
                    BeliefEvent::new(id, at, prior_tv, fc.verdict, opinion, trigger.to_string());
                log.record(event.clone());
                appended.push(event);
            }
        }
        Ok(appended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AssumptionGraph;
    use ix_ai_annotations::types::annotation_id;
    use ix_ai_annotations::{
        Annotation, AnnotationKind, Certainty, Location, Source, TruthValue, SCHEMA_VERSION,
    };

    fn ts(s: &str) -> DateTime<Utc> {
        s.parse().expect("rfc3339")
    }

    fn ann(
        author: &str,
        path: &str,
        line: u32,
        claim: &str,
        tv: TruthValue,
        conf: f64,
    ) -> Annotation {
        let kind = AnnotationKind::Assumption;
        Annotation {
            schema_version: SCHEMA_VERSION,
            id: annotation_id(path, line, kind, claim),
            kind,
            claim: claim.to_string(),
            truth_value: tv,
            certainty: Certainty::Assumed,
            confidence: conf,
            source: Source {
                author: author.to_string(),
                model: None,
                evidence: None,
            },
            location: Location {
                path: path.to_string(),
                line_start: line,
                line_end: line,
            },
            created_at: "2026-05-31T00:00:00Z".to_string(),
            updated_at: "2026-05-31T00:00:00Z".to_string(),
            stale: false,
            reconciliation: None,
        }
    }

    #[test]
    fn claim_id_is_stable_and_normalized() {
        assert_eq!(
            claim_id("Caller  Holds  Lock"),
            claim_id("caller holds lock")
        );
        assert!(claim_id("x").starts_with("sha256:"));
        assert_ne!(claim_id("a"), claim_id("b"));
    }

    #[test]
    fn belief_at_returns_latest_event_before_t() {
        let id = claim_id("x");
        let mut log = BeliefLog::new();
        log.record(BeliefEvent::new(
            id.clone(),
            ts("2026-01-01T00:00:00Z"),
            None,
            Hexavalent::Probable,
            None,
            "init",
        ));
        log.record(BeliefEvent::new(
            id.clone(),
            ts("2026-03-01T00:00:00Z"),
            Some(Hexavalent::Probable),
            Hexavalent::False,
            None,
            "test",
        ));

        assert_eq!(
            log.belief_at(ts("2026-02-01T00:00:00Z"))[&id].truth_value,
            Hexavalent::Probable
        );
        assert_eq!(
            log.belief_at(ts("2026-04-01T00:00:00Z"))[&id].truth_value,
            Hexavalent::False
        );
        // Before any event: empty.
        assert!(log.belief_at(ts("2025-01-01T00:00:00Z")).is_empty());
    }

    #[test]
    fn diff_reports_flips_between_times() {
        let id = claim_id("x");
        let mut log = BeliefLog::new();
        log.record(BeliefEvent::new(
            id.clone(),
            ts("2026-01-01T00:00:00Z"),
            None,
            Hexavalent::True,
            None,
            "init",
        ));
        log.record(BeliefEvent::new(
            id.clone(),
            ts("2026-03-01T00:00:00Z"),
            Some(Hexavalent::True),
            Hexavalent::False,
            None,
            "refuted",
        ));

        let changes = log.diff(ts("2026-02-01T00:00:00Z"), ts("2026-04-01T00:00:00Z"));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].from, Some(Hexavalent::True));
        assert_eq!(changes[0].to, Hexavalent::False);
    }

    #[test]
    fn jsonl_round_trips() {
        let id = claim_id("buffer flushed");
        let mut log = BeliefLog::new();
        log.record(BeliefEvent::new(
            id,
            ts("2026-05-31T00:00:00Z"),
            None,
            Hexavalent::True,
            Opinion::from_truth(TruthValue::T, 0.9),
            "test-run",
        ));
        let jsonl = log.to_jsonl();
        assert!(jsonl.contains("\"record\":\"belief_event\""));
        let back = BeliefLog::from_jsonl(&jsonl).unwrap();
        assert_eq!(back.events(), log.events());
    }

    #[test]
    fn revise_records_only_on_change() {
        // t1: cross-source agreement → leans true.
        let g1 = AssumptionGraph::from_annotations(vec![
            ann("claude", "a.rs", 1, "lock held", TruthValue::T, 0.9),
            ann("human", "b.rs", 2, "lock held", TruthValue::P, 0.7),
        ])
        .unwrap();
        let mut log = BeliefLog::new();
        let t1 = ts("2026-01-01T00:00:00Z");
        let first = g1.revise(&mut log, t1, "init").unwrap();
        assert_eq!(first.len(), 1, "first observation records an event");

        // Same evidence again at t1 → no change, no new event (idempotent).
        let repeat = g1.revise(&mut log, t1, "init").unwrap();
        assert!(repeat.is_empty(), "unchanged verdict appends nothing");

        // t2: an independent source now refutes it → escalates to C.
        let g2 = AssumptionGraph::from_annotations(vec![
            ann("claude", "a.rs", 1, "lock held", TruthValue::T, 1.0),
            ann("sentrux", "c.rs", 3, "lock held", TruthValue::F, 1.0),
        ])
        .unwrap();
        let t2 = ts("2026-03-01T00:00:00Z");
        let revised = g2.revise(&mut log, t2, "deep-research-reverify").unwrap();
        assert_eq!(revised.len(), 1);
        assert_eq!(revised[0].to_truth_value, Hexavalent::Contradictory);
        assert!(revised[0].to_opinion.is_none(), "C has no point opinion");

        // The story over time: the claim flipped from a positive lean to C.
        let changes = log.diff(t1, t2);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].to, Hexavalent::Contradictory);
    }
}
