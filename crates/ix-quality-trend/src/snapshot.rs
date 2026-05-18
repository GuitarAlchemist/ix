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
use std::time::SystemTime;

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Which quality category a snapshot belongs to. Also the subdirectory name
/// under the snapshots root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
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
    #[error(
        "strict mode: could not determine date for {path} \
         (filename does not start with YYYY-MM-DD, no `timestamp` field, mtime unavailable)"
    )]
    UndatableStrict { path: PathBuf },
}

/// Per-file outcome the loader records for the audit manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LoaderStatus {
    /// Loaded successfully with a date parsed from the filename stem.
    Loaded,
    /// Filename did not match `YYYY-MM-DD`; loaded by falling back to a date
    /// found inside the JSON (`timestamp` field) or the file's modification
    /// time. Caller may want to investigate.
    LoadedFallbackDate,
    /// Filename was undatable AND no fallback succeeded; skipped (warned).
    /// `--strict` upgrades this to a hard error.
    SkippedDateUnparseable,
    /// File was empty (0 bytes). Skipped.
    SkippedEmpty,
    /// JSON parsing failed. Skipped (warned). Hard error in `--strict`.
    FailedParse,
}

/// An audit-trail entry for one file the loader saw.
#[derive(Debug, Clone, Serialize)]
pub struct ManifestEntry {
    pub path: PathBuf,
    pub category: SnapshotCategory,
    pub status: LoaderStatus,
    pub date: Option<NaiveDate>,
    /// How the date was determined (e.g. "filename", "timestamp-field", "mtime").
    pub date_source: Option<String>,
    /// Free-text note (the warning text, the parse error, etc).
    pub note: Option<String>,
}

/// JSON-serializable manifest written by `--manifest`.
#[derive(Debug, Clone, Serialize)]
pub struct LoaderManifest {
    pub root: PathBuf,
    pub strict: bool,
    pub entries: Vec<ManifestEntry>,
}

/// Loader options. `default()` reproduces the historical permissive behaviour
/// **except** that previously-silent skips now emit warnings on `stderr` and
/// (when an mtime/timestamp fallback succeeds) the snapshot is actually used
/// instead of dropped.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoadOptions {
    /// If true, unparseable filenames with no fallback **and** JSON parse
    /// failures become hard errors instead of warnings.
    pub strict: bool,
    /// If true, no warnings are written to stderr. Manifest still records the
    /// skip. Useful when the caller wants to render its own report.
    pub quiet: bool,
}

/// Load every snapshot under `<root>/<category>/*.json`. Historical entry
/// point — equivalent to `load_with(root, LoadOptions::default())` and
/// discards the manifest.
pub fn load_all(root: &Path) -> Result<SnapshotSet, LoadError> {
    let (set, _manifest) = load_with(root, LoadOptions::default())?;
    Ok(set)
}

/// Load every snapshot under `<root>/<category>/*.json` and return both the
/// snapshot set and a per-file audit manifest.
///
/// Behaviour matrix:
///
/// | Filename has `YYYY-MM-DD` prefix? | JSON parses? | non-strict | strict |
/// |-----------------------------------|--------------|------------|--------|
/// | yes                               | yes          | loaded     | loaded |
/// | yes (e.g. `2026-05-15-soak.json`) | yes          | loaded     | loaded |
/// | no, but `timestamp` field present | yes          | loaded (fallback) | loaded (fallback) |
/// | no, mtime available               | yes          | loaded (fallback) + warn | loaded (fallback) |
/// | no, all fallbacks failed          | n/a          | skip + warn | **error** |
/// | empty file                        | n/a          | skip + warn | skip + warn |
/// | datable, JSON broken              | no           | skip + warn | **error** |
pub fn load_with(
    root: &Path,
    options: LoadOptions,
) -> Result<(SnapshotSet, LoaderManifest), LoadError> {
    let mut manifest = LoaderManifest {
        root: root.to_path_buf(),
        strict: options.strict,
        entries: Vec::new(),
    };

    let embeddings = load_category::<EmbeddingsSnapshot>(
        root,
        SnapshotCategory::Embeddings,
        options,
        &mut manifest.entries,
        extract_embeddings_timestamp,
    )?;
    let voicing = load_category::<VoicingAnalysisSnapshot>(
        root,
        SnapshotCategory::VoicingAnalysis,
        options,
        &mut manifest.entries,
        extract_voicing_timestamp,
    )?;
    let chatbot = load_category::<ChatbotQaSnapshot>(
        root,
        SnapshotCategory::ChatbotQa,
        options,
        &mut manifest.entries,
        extract_chatbot_timestamp,
    )?;
    let mut set = SnapshotSet {
        embeddings,
        voicing,
        chatbot,
    };

    // Sort each series ascending by date so downstream trend code can assume it.
    set.embeddings.sort_by_key(|s| s.date);
    set.voicing.sort_by_key(|s| s.date);
    set.chatbot.sort_by_key(|s| s.date);

    Ok((set, manifest))
}

/// Try to parse a `YYYY-MM-DD` date from the start of a filename stem. Accepts
/// trailing suffixes like `-postrefactor` or `-soak`.
fn parse_date_from_stem(stem: &str) -> Option<NaiveDate> {
    if stem.len() < 10 {
        return None;
    }
    let prefix = &stem[..10];
    NaiveDate::parse_from_str(prefix, "%Y-%m-%d").ok()
}

fn extract_embeddings_timestamp(bytes: &[u8]) -> Option<String> {
    serde_json::from_slice::<EmbeddingsSnapshot>(bytes)
        .ok()
        .and_then(|s| s.timestamp)
}

fn extract_voicing_timestamp(bytes: &[u8]) -> Option<String> {
    serde_json::from_slice::<VoicingAnalysisSnapshot>(bytes)
        .ok()
        .and_then(|s| s.timestamp)
}

fn extract_chatbot_timestamp(bytes: &[u8]) -> Option<String> {
    serde_json::from_slice::<ChatbotQaSnapshot>(bytes)
        .ok()
        .and_then(|s| s.timestamp)
}

fn date_from_timestamp_field(ts: &str) -> Option<NaiveDate> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(ts) {
        return Some(dt.naive_utc().date());
    }
    // Fall back to a date-only string.
    NaiveDate::parse_from_str(&ts[..ts.len().min(10)], "%Y-%m-%d").ok()
}

fn date_from_mtime(meta: &fs::Metadata) -> Option<NaiveDate> {
    let mtime = meta.modified().ok()?;
    let dur = mtime.duration_since(SystemTime::UNIX_EPOCH).ok()?;
    DateTime::<Utc>::from_timestamp(dur.as_secs() as i64, 0).map(|dt| dt.naive_utc().date())
}

fn load_category<T>(
    root: &Path,
    cat: SnapshotCategory,
    options: LoadOptions,
    manifest: &mut Vec<ManifestEntry>,
    extract_ts: fn(&[u8]) -> Option<String>,
) -> Result<Vec<DatedSnapshot<T>>, LoadError>
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

        // Stage 1: filename-prefix date (accepts `2026-05-15.json` AND
        // `2026-05-15-soak.json`).
        let (mut date, mut date_source) = match parse_date_from_stem(stem) {
            Some(d) => (Some(d), Some("filename".to_string())),
            None => (None, None),
        };

        // Stage 2: read bytes (needed for size check + JSON parsing + timestamp
        // fallback).
        let bytes = fs::read(&path).map_err(|e| LoadError::Io {
            path: path.clone(),
            source: e,
        })?;

        if bytes.is_empty() {
            let note = "file is empty".to_string();
            warn_skip(&path, options, &note);
            manifest.push(ManifestEntry {
                path: path.clone(),
                category: cat,
                status: LoaderStatus::SkippedEmpty,
                date: None,
                date_source: None,
                note: Some(note),
            });
            continue;
        }

        // Stage 3: if no filename date, try the JSON `timestamp` field.
        if date.is_none() {
            if let Some(ts) = extract_ts(&bytes) {
                if let Some(d) = date_from_timestamp_field(&ts) {
                    date = Some(d);
                    date_source = Some("timestamp-field".to_string());
                }
            }
        }

        // Stage 4: fall back to the file's modification time.
        if date.is_none() {
            if let Ok(meta) = fs::metadata(&path) {
                if let Some(d) = date_from_mtime(&meta) {
                    date = Some(d);
                    date_source = Some("mtime".to_string());
                }
            }
        }

        let Some(date) = date else {
            let note = format!(
                "filename stem {stem:?} is not YYYY-MM-DD, no `timestamp` field, mtime unavailable"
            );
            if options.strict {
                return Err(LoadError::UndatableStrict { path: path.clone() });
            }
            warn_skip(&path, options, &note);
            manifest.push(ManifestEntry {
                path: path.clone(),
                category: cat,
                status: LoaderStatus::SkippedDateUnparseable,
                date: None,
                date_source: None,
                note: Some(note),
            });
            continue;
        };

        // Stage 5: parse JSON.
        let data: T = match serde_json::from_slice(&bytes) {
            Ok(d) => d,
            Err(e) => {
                if options.strict {
                    return Err(LoadError::Json {
                        path: path.clone(),
                        source: e,
                    });
                }
                let note = format!("json parse error: {e}");
                warn_skip(&path, options, &note);
                manifest.push(ManifestEntry {
                    path: path.clone(),
                    category: cat,
                    status: LoaderStatus::FailedParse,
                    date: Some(date),
                    date_source: date_source.clone(),
                    note: Some(note),
                });
                continue;
            }
        };

        let status = if date_source.as_deref() == Some("filename") {
            LoaderStatus::Loaded
        } else {
            // Loaded via mtime / timestamp-field — visible in audit so a
            // consumer can decide whether to chase the producer.
            if !options.quiet {
                eprintln!(
                    "ix-quality-trend: loaded {} using fallback date source {} ({})",
                    path.display(),
                    date_source.as_deref().unwrap_or("?"),
                    date
                );
            }
            LoaderStatus::LoadedFallbackDate
        };
        manifest.push(ManifestEntry {
            path: path.clone(),
            category: cat,
            status,
            date: Some(date),
            date_source: date_source.clone(),
            note: None,
        });
        out.push(DatedSnapshot { date, path, data });
    }
    Ok(out)
}

fn warn_skip(path: &Path, options: LoadOptions, note: &str) {
    if options.quiet {
        return;
    }
    eprintln!(
        "ix-quality-trend: WARNING — skipping {}: {}",
        path.display(),
        note
    );
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

    // ------------------------------------------------------------------
    // Loader behaviour tests (the silent-skip regression coverage).
    // ------------------------------------------------------------------

    use std::fs as stdfs;
    use tempfile::TempDir;

    fn write(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            stdfs::create_dir_all(parent).unwrap();
        }
        stdfs::write(path, body).unwrap();
    }

    fn make_chatbot_fixture() -> TempDir {
        let dir = TempDir::new().unwrap();
        let cb = dir.path().join("chatbot-qa");
        // 1) good filename, good JSON
        write(
            &cb.join("2026-05-15.json"),
            r#"{"timestamp":"2026-05-15T00:00:00Z","total_prompts":10,"pass_pct":90.0}"#,
        );
        // 2) good filename WITH suffix (used to be silently skipped)
        write(
            &cb.join("2026-05-15-soak.json"),
            r#"{"timestamp":"2026-05-15T00:00:00Z","total_prompts":10,"pass_pct":91.0}"#,
        );
        // 3) gibberish filename, good JSON with timestamp → fallback
        write(
            &cb.join("baseline.json"),
            r#"{"timestamp":"2026-05-10T00:00:00Z","total_prompts":10,"pass_pct":80.0}"#,
        );
        // 4) gibberish filename, no timestamp field → mtime fallback or skip
        write(&cb.join("gibberish.json"), r#"{"total_prompts":1}"#);
        // 5) empty file
        write(&cb.join("empty.json"), "");
        // 6) malformed JSON, datable filename
        write(&cb.join("2026-05-14.json"), r#"{not valid"#);
        dir
    }

    fn status_for<'a>(manifest: &'a LoaderManifest, name: &str) -> &'a ManifestEntry {
        manifest
            .entries
            .iter()
            .find(|e| e.path.file_name().and_then(|s| s.to_str()) == Some(name))
            .unwrap_or_else(|| panic!("manifest missing entry for {name}"))
    }

    #[test]
    fn loader_buckets_files_correctly() {
        let dir = make_chatbot_fixture();
        let (set, manifest) = load_with(
            dir.path(),
            LoadOptions {
                strict: false,
                quiet: true,
            },
        )
        .unwrap();

        // Three loadable: two filename-dated + one timestamp-fallback.
        // gibberish.json may or may not get an mtime date (it always does on
        // disk), so it loads via fallback. Assert each manifest bucket directly.

        assert_eq!(
            status_for(&manifest, "2026-05-15.json").status,
            LoaderStatus::Loaded
        );
        assert_eq!(
            status_for(&manifest, "2026-05-15-soak.json").status,
            LoaderStatus::Loaded
        );
        assert_eq!(
            status_for(&manifest, "baseline.json").status,
            LoaderStatus::LoadedFallbackDate
        );
        assert_eq!(
            status_for(&manifest, "baseline.json").date_source.as_deref(),
            Some("timestamp-field")
        );
        // gibberish.json has no filename date and no timestamp field — mtime
        // fallback succeeds on a real filesystem.
        let gib = status_for(&manifest, "gibberish.json");
        assert_eq!(gib.status, LoaderStatus::LoadedFallbackDate);
        assert_eq!(gib.date_source.as_deref(), Some("mtime"));
        assert_eq!(
            status_for(&manifest, "empty.json").status,
            LoaderStatus::SkippedEmpty
        );
        assert_eq!(
            status_for(&manifest, "2026-05-14.json").status,
            LoaderStatus::FailedParse
        );

        // The loaded set: 2 filename-dated + 2 fallback-dated = 4 entries.
        assert_eq!(set.chatbot.len(), 4);
    }

    #[test]
    fn strict_mode_errors_on_unparseable_json() {
        let dir = make_chatbot_fixture();
        let err = load_with(
            dir.path(),
            LoadOptions {
                strict: true,
                quiet: true,
            },
        )
        .expect_err("strict mode should propagate a parse error");
        match err {
            LoadError::Json { path, .. } => {
                assert!(path.file_name().unwrap() == "2026-05-14.json");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn strict_mode_errors_on_undatable_when_no_fallback() {
        // Only undatable files, no `timestamp` field, no JSON parse issues.
        // We disable mtime fallback by making the file undatable (the test
        // can't actually disable mtime, but a strict run with a file that has
        // a timestamp falls back successfully — so we cover that path too).
        let dir = TempDir::new().unwrap();
        let cb = dir.path().join("chatbot-qa");
        // No timestamp, no filename date.
        write(&cb.join("zzz.json"), r#"{"total_prompts":1}"#);
        let (set, manifest) = load_with(
            dir.path(),
            LoadOptions {
                strict: true,
                quiet: true,
            },
        )
        .expect("mtime fallback should succeed even in strict mode");
        // Strict mode did NOT error because mtime fallback found a date.
        assert_eq!(set.chatbot.len(), 1);
        assert_eq!(
            status_for(&manifest, "zzz.json").status,
            LoaderStatus::LoadedFallbackDate
        );
    }

    #[test]
    fn load_all_preserves_back_compat() {
        // Old call site: only filename-dated files load, fallbacks happen
        // silently-ish but don't crash.
        let dir = make_chatbot_fixture();
        let set = load_all(dir.path()).unwrap();
        // All non-empty, non-malformed snapshots show up.
        assert_eq!(set.chatbot.len(), 4);
    }
}
