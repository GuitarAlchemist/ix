//! Canonical types for AI annotations.
//!
//! Mirrors `docs/contracts/ai-annotation.schema.json`.

use serde::{Deserialize, Serialize};

/// Schema version pinned by the contract.
///
/// v2 (2026-05-24): additive `business-value` and `hot-path` kinds for the
/// value × complexity quadrant heatmap. No field removals — v1 readers are
/// expected to drop unknown kinds rather than error.
pub const SCHEMA_VERSION: u32 = 2;

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
    /// Operator-declared assertion that this code drives meaningful product
    /// value. Carries a free-text rationale + optional metric. Source author
    /// should be `human` or `product-owner`; rarely auto-detected. Introduced
    /// in schema v2 (2026-05-24) for the value × complexity quadrant heatmap.
    BusinessValue,
    /// Measured-traffic assertion. Source should be `telemetry` or similar
    /// with a metric reference (e.g., Grafana panel URL). Introduced in
    /// schema v2 (2026-05-24).
    HotPath,
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
            "business-value" => Some(Self::BusinessValue),
            "hot-path" => Some(Self::HotPath),
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

#[cfg(test)]
mod kind_serde_tests {
    use super::*;

    #[test]
    fn business_value_round_trip_kebab_case() {
        let kind = AnnotationKind::BusinessValue;
        let json = serde_json::to_string(&kind).expect("ser ok");
        assert_eq!(json, "\"business-value\"");
        let back: AnnotationKind = serde_json::from_str(&json).expect("de ok");
        assert_eq!(back, AnnotationKind::BusinessValue);
    }

    #[test]
    fn hot_path_round_trip_kebab_case() {
        let kind = AnnotationKind::HotPath;
        let json = serde_json::to_string(&kind).expect("ser ok");
        assert_eq!(json, "\"hot-path\"");
        let back: AnnotationKind = serde_json::from_str(&json).expect("de ok");
        assert_eq!(back, AnnotationKind::HotPath);
    }

    #[test]
    fn all_v1_kinds_still_parse_in_v2() {
        // Backward-compat: v1 readers wrote these strings; the v2 enum must
        // still accept every one.
        for s in [
            "invariant",
            "assumption",
            "hypothesis",
            "contract",
            "smell",
            "decision",
            "hint",
        ] {
            let json = format!("\"{}\"", s);
            let kind: AnnotationKind = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("v1 kind {} should parse: {}", s, e));
            // Round-trips back to the same string.
            let back = serde_json::to_string(&kind).unwrap();
            assert_eq!(back, json);
        }
    }

    #[test]
    fn annotation_round_trip_with_business_value() {
        let a = Annotation {
            schema_version: SCHEMA_VERSION,
            id: annotation_id("foo.rs", 12, AnnotationKind::BusinessValue, "core engine"),
            kind: AnnotationKind::BusinessValue,
            claim: "core engine".to_string(),
            truth_value: TruthValue::T,
            certainty: Certainty::ManuallyReviewed,
            confidence: 0.95,
            source: Source {
                author: "product-owner".to_string(),
                model: None,
                evidence: Some("PR#123".to_string()),
            },
            location: Location {
                path: "foo.rs".to_string(),
                line_start: 12,
                line_end: 12,
            },
            created_at: "2026-05-24T00:00:00Z".to_string(),
            updated_at: "2026-05-24T00:00:00Z".to_string(),
            stale: false,
            reconciliation: None,
        };
        let json = serde_json::to_string(&a).expect("ser ok");
        assert!(json.contains("\"kind\":\"business-value\""));
        assert!(json.contains("\"schema_version\":2"));
        let back: Annotation = serde_json::from_str(&json).expect("de ok");
        assert_eq!(back.kind, AnnotationKind::BusinessValue);
        assert_eq!(back.schema_version, 2);
    }

    #[test]
    fn backward_compat_v1_annotation_parses() {
        // An existing v1 annotation (schema_version: 1, no new kinds) must
        // still parse cleanly into the v2 struct.
        let v1_json = r#"{
            "schema_version": 1,
            "id": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "kind": "invariant",
            "claim": "arr is sorted ascending",
            "truth_value": "T",
            "certainty": "test",
            "confidence": 0.95,
            "source": { "author": "claude" },
            "location": { "path": "foo.rs", "line_start": 1, "line_end": 1 },
            "created_at": "2026-05-24T00:00:00Z",
            "updated_at": "2026-05-24T00:00:00Z"
        }"#;
        let a: Annotation = serde_json::from_str(v1_json).expect("v1 parses into v2 struct");
        assert_eq!(a.schema_version, 1);
        assert_eq!(a.kind, AnnotationKind::Invariant);
        assert!(!a.stale);
        assert!(a.reconciliation.is_none());
    }

    #[test]
    fn parse_kind_string_for_new_kinds() {
        assert_eq!(
            AnnotationKind::parse("business-value"),
            Some(AnnotationKind::BusinessValue)
        );
        assert_eq!(
            AnnotationKind::parse("hot-path"),
            Some(AnnotationKind::HotPath)
        );
        assert_eq!(AnnotationKind::parse("nonsense"), None);
    }
}
