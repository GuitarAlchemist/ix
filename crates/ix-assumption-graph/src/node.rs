//! The assumption-graph node — a superset of an `@ai:` annotation.

use ix_ai_annotations::{Annotation, AnnotationKind, Certainty, Location, Source, TruthValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Kind of assumption node. A superset of the nine `@ai:` annotation kinds plus
/// `ResearchClaim` — first-class as of Phase 4 (the annotation enum has no such
/// variant, so the node model owns its own kind). Matches the contract's node
/// `kind` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NodeKind {
    Invariant,
    Assumption,
    Hypothesis,
    Contract,
    Smell,
    Decision,
    Hint,
    BusinessValue,
    HotPath,
    /// Ingested from a research run (e.g. `/deep-research`).
    ResearchClaim,
}

impl NodeKind {
    /// Kebab-case wire name (matches the serde representation).
    pub fn as_str(self) -> &'static str {
        match self {
            NodeKind::Invariant => "invariant",
            NodeKind::Assumption => "assumption",
            NodeKind::Hypothesis => "hypothesis",
            NodeKind::Contract => "contract",
            NodeKind::Smell => "smell",
            NodeKind::Decision => "decision",
            NodeKind::Hint => "hint",
            NodeKind::BusinessValue => "business-value",
            NodeKind::HotPath => "hot-path",
            NodeKind::ResearchClaim => "research-claim",
        }
    }
}

impl From<AnnotationKind> for NodeKind {
    fn from(k: AnnotationKind) -> Self {
        match k {
            AnnotationKind::Invariant => NodeKind::Invariant,
            AnnotationKind::Assumption => NodeKind::Assumption,
            AnnotationKind::Hypothesis => NodeKind::Hypothesis,
            AnnotationKind::Contract => NodeKind::Contract,
            AnnotationKind::Smell => NodeKind::Smell,
            AnnotationKind::Decision => NodeKind::Decision,
            AnnotationKind::Hint => NodeKind::Hint,
            AnnotationKind::BusinessValue => NodeKind::BusinessValue,
            AnnotationKind::HotPath => NodeKind::HotPath,
        }
    }
}

/// How a node's truth value was reached. A superset of the eight `@ai:`
/// certainty markers plus `AdversarialPanel` — a multi-judge verdict, first-class
/// as of Phase 4. Matches the contract's node `certainty` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NodeCertainty {
    Test,
    FormalProof,
    ManuallyReviewed,
    Assumed,
    Uncertain,
    Inferred,
    Dismissed,
    DetectedBySentrux,
    /// Verdict reached by a multi-judge adversarial panel — treated as a
    /// fail-closed gate, not a truth oracle (contract §7.2).
    AdversarialPanel,
}

impl From<Certainty> for NodeCertainty {
    fn from(c: Certainty) -> Self {
        match c {
            Certainty::Test => NodeCertainty::Test,
            Certainty::FormalProof => NodeCertainty::FormalProof,
            Certainty::ManuallyReviewed => NodeCertainty::ManuallyReviewed,
            Certainty::Assumed => NodeCertainty::Assumed,
            Certainty::Uncertain => NodeCertainty::Uncertain,
            Certainty::Inferred => NodeCertainty::Inferred,
            Certainty::Dismissed => NodeCertainty::Dismissed,
            Certainty::DetectedBySentrux => NodeCertainty::DetectedBySentrux,
        }
    }
}

/// A node in the temporal assumption graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionNode {
    /// Deterministic SHA-256 id. Dev nodes share the source annotation's id;
    /// research nodes hash `research:<author>:<normalized claim>`.
    pub id: String,
    pub kind: NodeKind,
    pub claim: String,
    pub truth_value: TruthValue,
    pub certainty: NodeCertainty,
    pub confidence: f64,
    pub source: Source,
    /// Present for code-anchored assumptions; `None` for research claims.
    pub location: Option<Location>,
}

impl From<Annotation> for AssumptionNode {
    fn from(a: Annotation) -> Self {
        Self {
            id: a.id,
            kind: a.kind.into(),
            claim: a.claim,
            truth_value: a.truth_value,
            certainty: a.certainty.into(),
            confidence: a.confidence,
            source: a.source,
            location: Some(a.location),
        }
    }
}

impl AssumptionNode {
    /// Build a research-claim node (no code location) — e.g. a verified finding
    /// from a `/deep-research` run. Kind is [`NodeKind::ResearchClaim`] and
    /// certainty is [`NodeCertainty::AdversarialPanel`] (the panel verdict). Id
    /// is `sha256` over `research:<author>:<normalized claim>` per the contract.
    pub fn research_claim(
        claim: impl Into<String>,
        truth_value: TruthValue,
        confidence: f64,
        source_author: impl Into<String>,
        evidence: Option<String>,
    ) -> Self {
        let claim = claim.into();
        let author = source_author.into();
        let key = format!(
            "research:{}:{}",
            author,
            crate::graph::normalize_claim(&claim)
        );
        let id = format!("sha256:{:x}", Sha256::digest(key.as_bytes()));
        Self {
            id,
            kind: NodeKind::ResearchClaim,
            claim,
            truth_value,
            certainty: NodeCertainty::AdversarialPanel,
            confidence,
            source: Source {
                author,
                model: None,
                evidence,
            },
            location: None,
        }
    }

    /// `true` iff this node is a research claim (vs a code-anchored assumption).
    pub fn is_research_claim(&self) -> bool {
        matches!(self.kind, NodeKind::ResearchClaim)
    }

    /// Coarse domain facet: `"research"` or `"dev"`.
    pub fn domain(&self) -> &'static str {
        if self.is_research_claim() {
            "research"
        } else {
            "dev"
        }
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
        let a = AssumptionNode::research_claim(
            "Latency  under 5ms",
            TruthValue::P,
            0.8,
            "deep-research",
            None,
        );
        let b = AssumptionNode::research_claim(
            "latency under 5ms",
            TruthValue::P,
            0.8,
            "deep-research",
            None,
        );
        assert_eq!(a.id, b.id, "normalized claim ⇒ same id");
        assert!(a.id.starts_with("sha256:"));
        assert!(a.location.is_none());
        assert_eq!(a.kind, NodeKind::ResearchClaim);
        assert!(a.is_research_claim());
        assert_eq!(a.domain(), "research");
    }

    #[test]
    fn different_source_gives_different_id() {
        let a = AssumptionNode::research_claim("x", TruthValue::P, 0.8, "deep-research", None);
        let b = AssumptionNode::research_claim("x", TruthValue::P, 0.8, "manual-review", None);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn annotation_kind_and_certainty_map_to_node_enums() {
        assert_eq!(
            NodeKind::from(AnnotationKind::Invariant),
            NodeKind::Invariant
        );
        assert_eq!(NodeKind::from(AnnotationKind::HotPath), NodeKind::HotPath);
        assert_eq!(
            NodeCertainty::from(Certainty::DetectedBySentrux),
            NodeCertainty::DetectedBySentrux
        );
        // New first-class variants serialize to the contract's kebab names.
        assert_eq!(
            serde_json::to_string(&NodeKind::ResearchClaim).unwrap(),
            "\"research-claim\""
        );
        assert_eq!(
            serde_json::to_string(&NodeCertainty::AdversarialPanel).unwrap(),
            "\"adversarial-panel\""
        );
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
        assert_eq!(n.kind, NodeKind::ResearchClaim);
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
