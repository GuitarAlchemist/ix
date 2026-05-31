//! The assumption-graph node — a superset of an `@ai:` annotation.

use ix_ai_annotations::{Annotation, AnnotationKind, Certainty, Location, Source, TruthValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

impl AssumptionNode {
    /// Build a research-claim node (no code location) — e.g. a verified finding
    /// from a `/deep-research` run.
    ///
    /// NOTE: the reused annotation vocabulary ([`AnnotationKind`]) has no
    /// dedicated `research-claim` variant, so a research claim is represented as
    /// a [`AnnotationKind::Hypothesis`] with `certainty = Inferred`; the
    /// originating run is identified by `source.author` (conventionally
    /// `"deep-research"`). Splitting the node enums from the annotation enums —
    /// so the contract's `research-claim` kind and `adversarial-panel` certainty
    /// become first-class — is a Phase 4 item. Id is `sha256` over
    /// `research:<author>:<normalized claim>` per the contract.
    pub fn research_claim(
        claim: impl Into<String>,
        truth_value: TruthValue,
        confidence: f64,
        source_author: impl Into<String>,
        evidence: Option<String>,
    ) -> Self {
        let claim = claim.into();
        let author = source_author.into();
        let key = format!("research:{}:{}", author, crate::graph::normalize_claim(&claim));
        let id = format!("sha256:{:x}", Sha256::digest(key.as_bytes()));
        Self {
            id,
            kind: AnnotationKind::Hypothesis,
            claim,
            truth_value,
            certainty: Certainty::Inferred,
            confidence,
            source: Source {
                author,
                model: None,
                evidence,
            },
            location: None,
        }
    }

    /// `true` iff this node originated from a research run (`source.author`
    /// is `"deep-research"`), as opposed to a code-anchored `@ai:` annotation.
    pub fn is_research_claim(&self) -> bool {
        self.source.author == "deep-research"
    }
}

/// A research-domain claim in the small JSON shape the loop runner ingests.
/// The truth-value/confidence *judgment* (mapping a verified finding to a
/// hexavalent value) is the adapter's job — see the `assumption-graph-loop`
/// skill — so this crate stays unopinionated about `/deep-research`'s output.
#[derive(Debug, Clone, Deserialize)]
pub struct ResearchClaim {
    pub claim: String,
    pub truth_value: TruthValue,
    pub confidence: f64,
    #[serde(default = "default_research_source")]
    pub source: String,
    #[serde(default)]
    pub evidence: Option<String>,
}

fn default_research_source() -> String {
    "deep-research".to_string()
}

impl From<ResearchClaim> for AssumptionNode {
    fn from(r: ResearchClaim) -> Self {
        AssumptionNode::research_claim(r.claim, r.truth_value, r.confidence, r.source, r.evidence)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_claim_id_is_stable_and_claim_normalized() {
        let a = AssumptionNode::research_claim("Latency  under 5ms", TruthValue::P, 0.8, "deep-research", None);
        let b = AssumptionNode::research_claim("latency under 5ms", TruthValue::P, 0.8, "deep-research", None);
        assert_eq!(a.id, b.id, "normalized claim ⇒ same id");
        assert!(a.id.starts_with("sha256:"));
        assert!(a.location.is_none());
        assert_eq!(a.kind, AnnotationKind::Hypothesis);
        assert!(a.is_research_claim());
    }

    #[test]
    fn different_source_gives_different_id() {
        let a = AssumptionNode::research_claim("x", TruthValue::P, 0.8, "deep-research", None);
        let b = AssumptionNode::research_claim("x", TruthValue::P, 0.8, "manual-review", None);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn research_claim_from_struct_carries_fields() {
        let r = ResearchClaim {
            claim: "buffer is flushed".to_string(),
            truth_value: TruthValue::F,
            confidence: 0.7,
            source: "deep-research".to_string(),
            evidence: Some("arxiv:2510.11822".to_string()),
        };
        let n: AssumptionNode = r.into();
        assert_eq!(n.truth_value, TruthValue::F);
        assert_eq!(n.source.author, "deep-research");
        assert_eq!(n.source.evidence.as_deref(), Some("arxiv:2510.11822"));
    }

    #[test]
    fn research_claim_json_defaults_source() {
        let r: ResearchClaim =
            serde_json::from_str(r#"{"claim":"x","truth_value":"P","confidence":0.6}"#).unwrap();
        assert_eq!(r.source, "deep-research");
        assert!(r.evidence.is_none());
    }
}
