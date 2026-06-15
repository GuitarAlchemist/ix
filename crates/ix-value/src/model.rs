//! The business-value contract types + the RICE→stars math.
//!
//! Schema: `docs/contracts/business-value.schema.json` (contract v0.1 draft).
//! Source-of-truth stays in each repo's hand-authored `state/value/manifest.json`;
//! this is the derived shape that lands in `state/value/catalog.jsonl`.

use serde::{Deserialize, Serialize};

/// Contract schema version. Bump on any field change (coordinate cross-repo).
pub const SCHEMA_VERSION: &str = "0.1.0";

/// What a scored row describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A user-facing demo/surface.
    #[default]
    Demo,
    /// A whole-repo rollup (generated, not hand-authored).
    Repo,
    /// An epic/initiative (reserved; not produced in v1).
    Epic,
}

/// One hand-authored value item in a repo's manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct Item {
    pub id: String,
    #[serde(default)]
    pub kind: Kind,
    pub title: String,
    pub reach: u8,
    pub impact: u8,
    pub confidence: u8,
    #[serde(default)]
    pub rationale: Option<String>,
}

/// An optional explicit repo-level RICE declaration (else rolled up from items).
#[derive(Debug, Clone, Deserialize)]
pub struct RepoScore {
    pub reach: u8,
    pub impact: u8,
    pub confidence: u8,
    #[serde(default)]
    pub rationale: Option<String>,
}

/// A repo's hand-authored value manifest (`state/value/manifest.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub schema_version: Option<String>,
    pub repo: String,
    #[serde(default)]
    pub items: Vec<Item>,
    #[serde(default)]
    pub repo_score: Option<RepoScore>,
}

/// One normalized, scored row for the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueRecord {
    pub schema_version: String,
    /// Stable id: items keep their authored id; repo rollups use the repo name.
    pub id: String,
    pub repo: String,
    pub kind: Kind,
    pub title: String,
    pub reach: u8,
    pub impact: u8,
    pub confidence: u8,
    /// `round(geomean(R,I,C))` clamped to `[1,5]`.
    pub stars: u8,
    /// `geomean(R,I,C) / 5` ∈ `(0,1]` — sortable/DuckDB-friendly.
    pub score01: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

/// Geometric mean of the three RICE axes. Geomean (not arithmetic) so the weakest
/// axis dominates — a low Confidence caps the score (`certainty := strength of
/// live binding`, applied to value).
pub fn geomean(reach: u8, impact: u8, confidence: u8) -> f64 {
    ((reach as f64) * (impact as f64) * (confidence as f64)).powf(1.0 / 3.0)
}

/// `round(geomean)` clamped to `[1,5]`.
pub fn stars(reach: u8, impact: u8, confidence: u8) -> u8 {
    geomean(reach, impact, confidence).round().clamp(1.0, 5.0) as u8
}

/// `geomean / 5` ∈ `(0,1]` — the continuous score for sorting/rollup.
pub fn score01(reach: u8, impact: u8, confidence: u8) -> f64 {
    (geomean(reach, impact, confidence) / 5.0).clamp(0.0, 1.0)
}

/// Whether all three axes are in the valid `1..=5` range.
pub fn axes_valid(reach: u8, impact: u8, confidence: u8) -> bool {
    (1..=5).contains(&reach) && (1..=5).contains(&impact) && (1..=5).contains(&confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stars_geomean_rounds_and_clamps() {
        // The plan's worked case.
        assert_eq!(stars(4, 5, 3), 4); // geomean 3.91 → 4
        // Low-confidence cap: arithmetic mean (5+5+1)/3 = 3.67 → 4, but geomean
        // 2.92 → 3. (The plan's "(5,5,1)→2" example was a miscalculation — 2 would
        // require harmonic mean; the locked rubric is geometric.)
        assert_eq!(stars(5, 5, 1), 3);
        assert_eq!(stars(5, 5, 5), 5);
        assert_eq!(stars(1, 1, 1), 1);
    }

    #[test]
    fn score01_is_geomean_over_five() {
        assert!((score01(5, 5, 5) - 1.0).abs() < 1e-9);
        assert!(score01(1, 1, 1) < score01(3, 3, 3));
    }

    #[test]
    fn axes_validation() {
        assert!(axes_valid(1, 5, 3));
        assert!(!axes_valid(0, 5, 3));
        assert!(!axes_valid(5, 6, 3));
    }
}
