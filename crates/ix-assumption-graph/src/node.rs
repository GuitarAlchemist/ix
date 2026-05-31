//! The assumption-graph node — a superset of an `@ai:` annotation.

use ix_ai_annotations::{Annotation, AnnotationKind, Certainty, Location, Source, TruthValue};
use serde::{Deserialize, Serialize};

/// A node in the temporal assumption graph.
///
/// Phase 1 carries exactly the epistemic state an `@ai:` annotation already
/// provides — no new epistemics. The subjective-logic `opinion`, the
/// `belief_time` axis, and the JSONL wire form defined in
/// `docs/contracts/2026-05-31-assumption-graph.contract.md` arrive in Phase 2/3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionNode {
    /// Deterministic SHA-256 id, identical to the source annotation's id
    /// (`sha256:…` over `path:line_start:kind:claim`) so an annotation and its
    /// graph node share identity.
    pub id: String,
    pub kind: AnnotationKind,
    pub claim: String,
    pub truth_value: TruthValue,
    pub certainty: Certainty,
    pub confidence: f64,
    pub source: Source,
    /// Present for code-anchored assumptions; `None` for research claims (Phase 3).
    pub location: Option<Location>,
}

impl From<Annotation> for AssumptionNode {
    fn from(a: Annotation) -> Self {
        Self {
            id: a.id,
            kind: a.kind,
            claim: a.claim,
            truth_value: a.truth_value,
            certainty: a.certainty,
            confidence: a.confidence,
            source: a.source,
            location: Some(a.location),
        }
    }
}

/// Evidential polarity of a hexavalent truth value — the axis along which two
/// claims about the same thing can contradict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// Leans true: `T` (True) or `P` (Probable).
    Positive,
    /// Leans false: `D` (Doubtful) or `F` (False).
    Negative,
    /// No evidential direction: `U` (Unknown).
    Neutral,
    /// Already contradictory: `C` — flagged at the source, not re-derived pairwise.
    Contradictory,
}

/// Map a hexavalent truth value to its evidential polarity.
///
// @ai:invariant every TruthValue maps to exactly one Polarity (total match) [T:formal-proof conf:0.95 src:exhaustive match, compiler-checked]
pub fn polarity(tv: TruthValue) -> Polarity {
    match tv {
        TruthValue::T | TruthValue::P => Polarity::Positive,
        TruthValue::D | TruthValue::F => Polarity::Negative,
        TruthValue::U => Polarity::Neutral,
        TruthValue::C => Polarity::Contradictory,
    }
}
