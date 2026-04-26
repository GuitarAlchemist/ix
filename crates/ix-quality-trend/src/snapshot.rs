//! Typed, schema-tolerant snapshot structs.
//!
//! Every field is `Option` so that old snapshots missing newer metrics load
//! cleanly. Field names follow the JSON conventions used by the producers:
//! - `ix-embedding-diagnostics` (snake_case)
//! - `Demos/VoicingAnalysisAudit` (.NET PascalCase — hence the `rename_all`)
//! - `ga-chatbot qa --benchmark` (snake_case summary)

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use serde::Deserialize;
use thiserror::Error;

/// Which quality category a snapshot belongs to. Also the subdirectory name
/// under the snapshots root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SnapshotCategory {
    Embeddings,
    VoicingAnalysis,
    ChatbotQa,
}

impl SnapshotCategory {
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::Embeddings => "embeddings",
            Self::VoicingAnalysis => "voicing-analysis",
            Self::ChatbotQa => "chatbot-qa",
        }
    }

    pub fn display(self) -> &'static str {
        match self {
            Self::Embeddings => "Embeddings",
            Self::VoicingAnalysis => "Voicing analysis",
            Self::ChatbotQa => "Chatbot QA",
        }
    }

    pub fn all() -> [Self; 3] {
        [Self::Embeddings, Self::VoicingAnalysis, Self::ChatbotQa]
    }
}

/// A snapshot wrapped with its date (parsed from filename).
#[derive(Debug, Clone)]
pub struct DatedSnapshot<T> {
    pub date: NaiveDate,
    pub path: PathBuf,
    pub data: T,
}

/// Collection of all snapshots loaded for a report run.
#[derive(Debug, Default)]
pub struct SnapshotSet {
    pub embeddings: Vec<DatedSnapshot<EmbeddingsSnapshot>>,
    pub voicing: Vec<DatedSnapshot<VoicingAnalysisSnapshot>>,
    pub chatbot: Vec<DatedSnapshot<ChatbotQaSnapshot>>,
}

// ============================================================================
// Embeddings
// ============================================================================

/// Embedding diagnostics snapshot (produced by `ix-embedding-diagnostics`).
///
/// Extracts per-partition leak classifier accuracy, retrieval consistency,
/// and per-instrument topology Betti numbers.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingsSnapshot {
    pub timestamp: Option<String>,
    pub corpus: Option<EmbeddingsCorpus>,
    pub leak_detection: Option<LeakDetection>,
    pub retrieval_consistency: Option<RetrievalConsistency>,
    pub topology: Option<Topology>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingsCorpus {
    pub count: Option<u64>,
    pub dims: Option<u64>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct LeakDetection {
    pub full_classifier_accuracy: Option<f64>,
    pub baseline_random: Option<f64>,
    pub by_partition: Option<Vec<PartitionAccuracy>>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct PartitionAccuracy {
    pub partition: Option<String>,
    pub accuracy_mean: Option<f64>,
    pub accuracy_std: Option<f64>,
    pub leak_confirmed: Option<bool>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct RetrievalConsistency {
    pub queries_tested: Option<u64>,
    pub top_k: Option<u64>,
    /// Present as `avg_pc_set_match_pct` in current snapshots; the spec calls
    /// for `avg_pc_set_match_pct_top10`. Both field names are accepted.
    pub avg_pc_set_match_pct: Option<f64>,
    pub avg_pc_set_match_pct_top10: Option<f64>,
}

impl RetrievalConsistency {
    /// Preferred pitch-class-set match percentage, regardless of which schema
    /// version produced the snapshot.
    pub fn match_pct(&self) -> Option<f64> {
        self.avg_pc_set_match_pct_top10
            .or(self.avg_pc_set_match_pct)
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct Topology {
    pub guitar: Option<BettiNumbers>,
    pub bass: Option<BettiNumbers>,
    pub ukulele: Option<BettiNumbers>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct BettiNumbers {
    pub sample_size: Option<u64>,
    pub beta_0: Option<i64>,
    pub beta_1: Option<i64>,
}

impl EmbeddingsSnapshot {
    /// Look up a partition accuracy by name (case-insensitive match).
    pub fn partition_accuracy(&self, name: &str) -> Option<f64> {
        let parts = self.leak_detection.as_ref()?.by_partition.as_ref()?;
        for p in parts {
            if let Some(pn) = p.partition.as_deref() {
                if pn.eq_ignore_ascii_case(name) {
                    return p.accuracy_mean;
                }
            }
        }
        None
    }
}

// ============================================================================
// Voicing analysis audit (PascalCase — .NET producer)
// ============================================================================

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
pub struct VoicingAnalysisSnapshot {
    pub timestamp: Option<String>,
    pub corpus: Option<VoicingCorpus>,
    pub chord_recognition: Option<ChordRecognition>,
    pub forte_coverage: Option<ForteCoverage>,
    pub cross_instrument_consistency: Option<CrossInstrumentConsistency>,
    pub cardinality_distribution: Option<BTreeMap<String, u64>>,
    pub two_note_with_chord_name: Option<u64>,
    pub invariant_failures: Option<InvariantFailures>,
    pub performance: Option<VoicingPerformance>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
pub struct VoicingCorpus {
    pub guitar: Option<u64>,
    pub bass: Option<u64>,
    pub ukulele: Option<u64>,
    pub total: Option<u64>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
pub struct ChordRecognition {
    pub null_chord_name: Option<CountPct>,
    pub unknown_chord_name: Option<CountPct>,
    pub distinct_chord_names: Option<u64>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
pub struct CountPct {
    pub count: Option<u64>,
    pub pct: Option<f64>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
pub struct ForteCoverage {
    pub resolved: Option<u64>,
    pub total: Option<u64>,
    pub pct: Option<f64>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
pub struct CrossInstrumentConsistency {
    pub shared_sets: Option<u64>,
    pub consistent: Option<u64>,
    /// Optional pre-computed consistency percentage (0..100). If absent we
    /// derive it from `consistent` / `shared_sets`.
    pub pct: Option<f64>,
}

impl CrossInstrumentConsistency {
    /// Consistency percentage (0..100). Prefers an explicit `pct` field, else
    /// computes consistent / shared_sets * 100.
    pub fn consistency_pct(&self) -> Option<f64> {
        if let Some(p) = self.pct {
            return Some(p);
        }
        match (self.consistent, self.shared_sets) {
            (Some(c), Some(s)) if s > 0 => Some(c as f64 / s as f64 * 100.0),
            _ => None,
        }
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
pub struct InvariantFailures {
    pub midi_notes_mismatch: Option<u64>,
    pub null_pitch_class_set: Option<u64>,
    pub negative_physical_layout: Option<u64>,
    pub interval_spread_invariant: Option<u64>,
}

impl InvariantFailures {
    pub fn total(&self) -> u64 {
        self.midi_notes_mismatch.unwrap_or(0)
            + self.null_pitch_class_set.unwrap_or(0)
            + self.negative_physical_layout.unwrap_or(0)
            + self.interval_spread_invariant.unwrap_or(0)
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
pub struct VoicingPerformance {
    pub runtime_seconds: Option<f64>,
    pub voicings_per_sec: Option<f64>,
}

// ============================================================================
// Chatbot QA
// ============================================================================

/// Summary form of ga-chatbot QA benchmark (one file per date).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct ChatbotQaSnapshot {
    pub timestamp: Option<String>,
    pub total_prompts: Option<u64>,
    pub pass_pct: Option<f64>,
    pub avg_response_ms: Option<u64>,
    /// Per-level breakdown (L1..L5). Keys match the producer's naming.
    pub by_category: Option<BTreeMap<String, ChatbotCategoryStats>>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct ChatbotCategoryStats {
    pub pass_pct: Option<f64>,
    pub total: Option<u64>,
}

// ============================================================================
// Loader
// ============================================================================

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("i/o error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("json parse error in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

/// Load every snapshot under `<root>/<category>/*.json`. Files whose stem does
/// not parse as `YYYY-MM-DD` are silently skipped. JSON parse errors are
/// surfaced (we do not silently eat them — tolerance happens at the *field*
/// level via `Option`, not at the document level).
pub fn load_all(root: &Path) -> Result<SnapshotSet, LoadError> {
    let mut set = SnapshotSet {
        embeddings: load_category::<EmbeddingsSnapshot>(root, SnapshotCategory::Embeddings)?,
        voicing: load_category::<VoicingAnalysisSnapshot>(root, SnapshotCategory::VoicingAnalysis)?,
        chatbot: load_category::<ChatbotQaSnapshot>(root, SnapshotCategory::ChatbotQa)?,
    };

    // Sort each series ascending by date so downstream trend code can assume it.
    set.embeddings.sort_by_key(|s| s.date);
    set.voicing.sort_by_key(|s| s.date);
    set.chatbot.sort_by_key(|s| s.date);

    Ok(set)
}

fn load_category<T>(root: &Path, cat: SnapshotCategory) -> Result<Vec<DatedSnapshot<T>>, LoadError>
where
    T: for<'de> Deserialize<'de>,
{
    let dir = root.join(cat.dir_name());
    if !dir.is_dir() {
        return Ok(Vec::new());
    }

    let read = fs::read_dir(&dir).map_err(|e| LoadError::Io {
        path: dir.clone(),
        source: e,
    })?;

    let mut out = Vec::new();
    for entry in read {
        let entry = entry.map_err(|e| LoadError::Io {
            path: dir.clone(),
            source: e,
        })?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(date) = NaiveDate::parse_from_str(stem, "%Y-%m-%d") else {
            continue;
        };
        let bytes = fs::read(&path).map_err(|e| LoadError::Io {
            path: path.clone(),
            source: e,
        })?;
        let data: T = serde_json::from_slice(&bytes).map_err(|e| LoadError::Json {
            path: path.clone(),
            source: e,
        })?;
        out.push(DatedSnapshot { date, path, data });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_snapshot_parses_real_sample() {
        let json = r#"{
            "timestamp": "2026-04-17T00:00:00Z",
            "leak_detection": {
                "full_classifier_accuracy": 0.74,
                "by_partition": [
                    {"partition": "STRUCTURE", "accuracy_mean": 0.56},
                    {"partition": "MORPHOLOGY", "accuracy_mean": 0.79}
                ]
            },
            "retrieval_consistency": { "avg_pc_set_match_pct": 0.866 },
            "topology": { "guitar": { "beta_0": 685, "beta_1": 48 } }
        }"#;
        let s: EmbeddingsSnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(s.partition_accuracy("STRUCTURE"), Some(0.56));
        assert_eq!(s.partition_accuracy("morphology"), Some(0.79));
        assert_eq!(s.partition_accuracy("CONTEXT"), None);
        assert_eq!(s.retrieval_consistency.unwrap().match_pct(), Some(0.866));
    }

    #[test]
    fn voicing_snapshot_parses_pascal_case() {
        let json = r#"{
            "Corpus": { "Total": 313047 },
            "ChordRecognition": {
                "UnknownChordName": { "Count": 98772, "Pct": 31.552 }
            },
            "CrossInstrumentConsistency": {
                "SharedSets": 793,
                "Consistent": 233
            },
            "ForteCoverage": { "Pct": 100.0 },
            "InvariantFailures": { "IntervalSpreadInvariant": 50 }
        }"#;
        let s: VoicingAnalysisSnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(s.corpus.as_ref().unwrap().total, Some(313047));
        assert_eq!(
            s.chord_recognition
                .as_ref()
                .unwrap()
                .unknown_chord_name
                .as_ref()
                .unwrap()
                .pct,
            Some(31.552)
        );
        let pct = s
            .cross_instrument_consistency
            .as_ref()
            .unwrap()
            .consistency_pct()
            .unwrap();
        assert!((pct - 29.3820932).abs() < 1e-3);
        assert_eq!(s.invariant_failures.unwrap().total(), 50);
    }

    #[test]
    fn missing_fields_are_tolerated() {
        // A snapshot with almost nothing in it still parses.
        let s: EmbeddingsSnapshot = serde_json::from_str("{}").unwrap();
        assert!(s.timestamp.is_none());
        assert!(s.leak_detection.is_none());
    }
}
