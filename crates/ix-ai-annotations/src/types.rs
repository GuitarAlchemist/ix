//! Canonical types for AI annotations.
//!
//! Mirrors `docs/contracts/ai-annotation.schema.json`.

use serde::{Deserialize, Serialize};

/// Schema version pinned by the contract.
pub const SCHEMA_VERSION: u32 = 1;

/// Hexavalent truth value, aligned with
/// `governance/demerzel/schemas/hexavalent-distribution.schema.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TruthValue {
    /// True — verified with evidence.
    T,
    /// Probable — evidence leans true, not yet verified.
    P,
    /// Unknown — insufficient evidence, triggers investigation.
    U,
    /// Doubtful — evidence leans false, not yet refuted.
    D,
    /// False — refuted with evidence.
    F,
    /// Contradictory — conflicting evidence, triggers escalation.
    C,
}

impl TruthValue {
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'T' => Some(Self::T),
            'P' => Some(Self::P),
            'U' => Some(Self::U),
            'D' => Some(Self::D),
            'F' => Some(Self::F),
            'C' => Some(Self::C),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::T => "T",
            Self::P => "P",
            Self::U => "U",
            Self::D => "D",
            Self::F => "F",
            Self::C => "C",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Certainty {
    Test,
    FormalProof,
    ManuallyReviewed,
    Assumed,
    Uncertain,
    Inferred,
    Dismissed,
    /// Produced by the sentrux structural-rules engine
    /// (see `ix-sentrux-annotations`). Ground-truth verifier output.
    DetectedBySentrux,
}

impl Certainty {
    /// Parse from the kebab-case form used in the marker bracket.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "test" => Some(Self::Test),
            "formal-proof" => Some(Self::FormalProof),
            "manually-reviewed" => Some(Self::ManuallyReviewed),
            "assumed" => Some(Self::Assumed),
            "uncertain" => Some(Self::Uncertain),
            "inferred" => Some(Self::Inferred),
            "dismissed" => Some(Self::Dismissed),
            "detected-by-sentrux" => Some(Self::DetectedBySentrux),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AnnotationKind {
    Invariant,
    Assumption,
    Hypothesis,
    Contract,
    Smell,
    Decision,
    Hint,
}

impl AnnotationKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "invariant" => Some(Self::Invariant),
            "assumption" => Some(Self::Assumption),
            "hypothesis" => Some(Self::Hypothesis),
            "contract" => Some(Self::Contract),
            "smell" => Some(Self::Smell),
            "decision" => Some(Self::Decision),
            "hint" => Some(Self::Hint),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub author: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub path: String,
    pub line_start: u32,
    pub line_end: u32,
}

/// One extracted annotation, in canonical wire form (serializes to
/// `ai-annotation.schema.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub schema_version: u32,
    pub id: String,
    pub kind: AnnotationKind,
    pub claim: String,
    pub truth_value: TruthValue,
    pub certainty: Certainty,
    pub confidence: f64,
    pub source: Source,
    pub location: Location,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    pub stale: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reconciliation: Option<Reconciliation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Reconciliation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_match: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub promoted_to_c_from: Vec<TruthValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weighted_truth_value: Option<TruthValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weighted_confidence: Option<f64>,
}

/// Deterministic id over `path:line_start:kind:claim`.
pub fn annotation_id(path: &str, line_start: u32, kind: AnnotationKind, claim: &str) -> String {
    use sha2::{Digest, Sha256};
    let key = format!("{}:{}:{:?}:{}", path, line_start, kind, claim);
    let digest = Sha256::digest(key.as_bytes());
    format!("sha256:{:x}", digest)
}
