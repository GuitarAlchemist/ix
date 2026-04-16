//! `ix-voicings` — Phase A of the voicings study pipeline.
//!
//! This crate shells out to GA's `FretboardVoicingsCLI --export` with a
//! `--tuning {guitar|bass|ukulele}` flag (added in the matching ga commit)
//! and turns the JSONL stream into two on-disk artifacts per instrument:
//!
//! - `state/voicings/raw/{instrument}.jsonl` — the raw GA output
//! - `state/voicings/{instrument}-corpus.json` — compacted JSON array of
//!   [`VoicingRow`] entries
//! - `state/voicings/{instrument}-features.json` — z-scored feature matrix
//!   (schema in [`FEATURE_COLUMNS`])
//!
//! Phase B will add `cluster`, `topology`, `transitions`, `progressions`,
//! and `render_book` nodes (see `docs/plans/2026-04-15-002-feat-voicings-study-plan.md`).
//! Only `enumerate` and `featurize` are wired here.
//!
//! # Deviation from the plan
//!
//! The plan's architecture sketch uses `ix_pipeline::PipelineBuilder`. For
//! Phase A the DAG has exactly two nodes (`enumerate` → `featurize`), so
//! the pipeline harness buys nothing and makes testing harder (inputs to
//! the enumerate node depend on process state we don't want to pickle
//! through the pipeline executor). We keep the library functions plain
//! and typed, and Phase B will introduce the `PipelineBuilder` wrapping
//! once there are ≥3 nodes and upstream fan-out actually matters.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

/// Supported instrument presets. Must match the GA CLI's `--tuning` flag
/// values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Instrument {
    Guitar,
    Bass,
    Ukulele,
}

impl Instrument {
    pub const ALL: [Instrument; 3] = [Instrument::Guitar, Instrument::Bass, Instrument::Ukulele];

    pub fn as_str(self) -> &'static str {
        match self {
            Instrument::Guitar => "guitar",
            Instrument::Bass => "bass",
            Instrument::Ukulele => "ukulele",
        }
    }

    pub fn parse(s: &str) -> Result<Self, VoicingsError> {
        match s.to_ascii_lowercase().as_str() {
            "guitar" => Ok(Instrument::Guitar),
            "bass" => Ok(Instrument::Bass),
            "ukulele" | "uke" => Ok(Instrument::Ukulele),
            other => Err(VoicingsError::UnknownInstrument(other.to_string())),
        }
    }
}

/// One voicing row as produced by `FretboardVoicingsCLI --export`.
///
/// Field names match the GA DTO so `serde_json` can parse the JSONL lines
/// directly. The `instrument` and `string_count` fields were added in the
/// matching ga commit; older GA builds will fail to parse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicingRow {
    pub instrument: String,
    #[serde(rename = "stringCount")]
    pub string_count: u32,
    pub diagram: String,
    /// Per-string fret strings: `"x"` for muted, otherwise the fret number.
    /// Order is highest-string-index-first (GA convention).
    pub frets: Vec<String>,
    #[serde(rename = "midiNotes")]
    pub midi_notes: Vec<i32>,
    #[serde(rename = "minFret")]
    pub min_fret: i32,
    #[serde(rename = "maxFret")]
    pub max_fret: i32,
    #[serde(rename = "fretSpan")]
    pub fret_span: i32,
}

/// Feature matrix for a single instrument — the output of [`featurize`].
///
/// `rows` is `(n_voicings, FEATURE_COLUMNS.len())`, already z-scored for
/// numeric columns. Categorical one-hots are left raw.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMatrix {
    pub instrument: String,
    pub columns: Vec<String>,
    pub numeric_cols: Vec<String>,
    pub rows: Vec<Vec<f64>>,
    pub normalization: HashMap<String, Normalization>,
    /// Count of voicings that fed into the matrix (== `rows.len()`).
    pub voicing_count: usize,
}

/// Mean + stddev used to z-score one numeric column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalization {
    pub mean: f64,
    pub stddev: f64,
}

/// Error type for the voicings pipeline.
#[derive(Debug, Error)]
pub enum VoicingsError {
    #[error("unknown instrument '{0}' (expected guitar|bass|ukulele)")]
    UnknownInstrument(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error on line {line}: {source}")]
    JsonLine {
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("GA CLI `{program}` exited with status {status}: {stderr}")]
    CliFailed {
        program: String,
        status: i32,
        stderr: String,
    },

    #[error("GA CLI produced zero voicings for {0}")]
    EmptyCorpus(String),

    #[error("feature matrix shape mismatch on row {row}: expected {expected} columns, got {got}")]
    ShapeMismatch {
        row: usize,
        expected: usize,
        got: usize,
    },
}

/// Fixed coarse quality vocabulary used for the one-hot. The quality is
/// inferred from the sorted interval set modulo 12 — enough to stratify
/// the corpus without waiting on GA's full analysis pass (which doesn't
/// run in `--export` mode).
pub const QUALITY_VOCAB: &[&str] = &[
    "dyad",       // 2 distinct pitch classes
    "maj",        // major triad (0,4,7)
    "min",        // minor triad (0,3,7)
    "dim",        // diminished (0,3,6)
    "aug",        // augmented (0,4,8)
    "sus",        // sus2 / sus4 triad (0,2,7) / (0,5,7)
    "maj7",       // (0,4,7,11)
    "dom7",       // (0,4,7,10)
    "min7",       // (0,3,7,10)
    "min7b5",     // (0,3,6,10)
    "dim7",       // (0,3,6,9)
    "other4",     // 4 distinct pitch classes, none of the above
    "other",      // fallback
];

/// Column schema for the feature matrix.
///
/// The `frets_used_*` slots are padded to 6 regardless of instrument
/// (bass/ukulele have 4 strings — the extra slots are filled with -1).
/// Numeric columns (everything up to the one-hot) are z-scored; one-hot
/// columns are left as 0/1.
pub const FEATURE_COLUMNS: &[&str] = &[
    "fret_span",
    "frets_used_0",
    "frets_used_1",
    "frets_used_2",
    "frets_used_3",
    "frets_used_4",
    "frets_used_5",
    "string_count_played",
    "is_barre",
    "min_fret",
    "max_fret",
    "midi_note_count",
    "lowest_midi",
    "highest_midi",
    // quality one-hot tail — indices match QUALITY_VOCAB
    "quality_dyad",
    "quality_maj",
    "quality_min",
    "quality_dim",
    "quality_aug",
    "quality_sus",
    "quality_maj7",
    "quality_dom7",
    "quality_min7",
    "quality_min7b5",
    "quality_dim7",
    "quality_other4",
    "quality_other",
];

/// Columns that participate in z-scoring. Everything else (the one-hot
/// tail + the boolean `is_barre`) is left raw.
pub const NUMERIC_COLUMNS: &[&str] = &[
    "fret_span",
    "frets_used_0",
    "frets_used_1",
    "frets_used_2",
    "frets_used_3",
    "frets_used_4",
    "frets_used_5",
    "string_count_played",
    "min_fret",
    "max_fret",
    "midi_note_count",
    "lowest_midi",
    "highest_midi",
];

/// Resolve the state root, honoring the `IX_VOICINGS_STATE_DIR` override
/// (used by the smoke test). Default: `./state`.
pub fn state_root() -> PathBuf {
    match std::env::var("IX_VOICINGS_STATE_DIR") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => PathBuf::from("state"),
    }
}

/// Absolute path to the GA CLI executable. Honors `IX_VOICINGS_GA_CLI`
/// (full path to a `FretboardVoicingsCLI.exe`), otherwise defaults to
/// the checked-in debug build next to this repo.
pub fn ga_cli_path() -> PathBuf {
    if let Ok(v) = std::env::var("IX_VOICINGS_GA_CLI") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    // Default: sibling `ga` repo's debug build. The smoke test overrides
    // this via env var when running in a sandbox that doesn't have ga/
    // next door.
    PathBuf::from(
        r"C:\Users\spare\source\repos\ga\Demos\Music Theory\FretboardVoicingsCLI\bin\Debug\net10.0\FretboardVoicingsCLI.exe",
    )
}

/// Artifacts written by [`enumerate`].
#[derive(Debug, Clone)]
pub struct EnumerateArtifacts {
    pub raw_jsonl: PathBuf,
    pub corpus_json: PathBuf,
    pub voicing_count: usize,
}

/// Shell out to the GA CLI and capture its JSONL stream for one
/// instrument. Writes the raw stream to `state/voicings/raw/{i}.jsonl`
/// and a compacted JSON array to `state/voicings/{i}-corpus.json`.
pub fn enumerate(
    instrument: Instrument,
    export_max: Option<usize>,
) -> Result<EnumerateArtifacts, VoicingsError> {
    let state = state_root();
    let raw_dir = state.join("voicings").join("raw");
    let out_dir = state.join("voicings");
    std::fs::create_dir_all(&raw_dir)?;
    std::fs::create_dir_all(&out_dir)?;

    let raw_path = raw_dir.join(format!("{}.jsonl", instrument.as_str()));
    let corpus_path = out_dir.join(format!("{}-corpus.json", instrument.as_str()));

    let cli = ga_cli_path();
    let mut cmd = Command::new(&cli);
    cmd.arg("--export")
        .arg("--tuning")
        .arg(instrument.as_str());
    if let Some(max) = export_max {
        cmd.arg("--export-max").arg(max.to_string());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("failed to spawn GA CLI at {}: {e}", cli.display()),
        )
    })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("GA CLI stdout not captured"))?;

    // Stream stdout line-by-line into both the raw file and the parsed
    // corpus so we can surface parse errors immediately without buffering
    // the whole output in memory.
    let mut raw_file = std::fs::File::create(&raw_path)?;
    let reader = BufReader::new(stdout);
    let mut rows: Vec<VoicingRow> = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        use std::io::Write as _;
        writeln!(raw_file, "{line}")?;
        let row: VoicingRow = serde_json::from_str(&line).map_err(|e| VoicingsError::JsonLine {
            line: lineno + 1,
            source: e,
        })?;
        rows.push(row);
    }

    let status = child.wait()?;
    if !status.success() {
        let mut stderr_buf = String::new();
        if let Some(mut s) = child.stderr.take() {
            use std::io::Read as _;
            let _ = s.read_to_string(&mut stderr_buf);
        }
        return Err(VoicingsError::CliFailed {
            program: cli.display().to_string(),
            status: status.code().unwrap_or(-1),
            stderr: stderr_buf,
        });
    }

    if rows.is_empty() {
        return Err(VoicingsError::EmptyCorpus(instrument.as_str().to_string()));
    }

    // Compacted corpus — a JSON array instead of line-delimited.
    let corpus_json = serde_json::to_vec(&rows)?;
    std::fs::write(&corpus_path, corpus_json)?;

    Ok(EnumerateArtifacts {
        raw_jsonl: raw_path,
        corpus_json: corpus_path,
        voicing_count: rows.len(),
    })
}

/// Artifacts written by [`featurize`].
#[derive(Debug, Clone)]
pub struct FeaturizeArtifacts {
    pub features_path: PathBuf,
    pub voicing_count: usize,
    pub column_count: usize,
}

/// Read a corpus JSON file back into memory.
pub fn load_corpus(path: &Path) -> Result<Vec<VoicingRow>, VoicingsError> {
    let bytes = std::fs::read(path)?;
    let rows: Vec<VoicingRow> = serde_json::from_slice(&bytes)?;
    Ok(rows)
}

/// Build the feature matrix for the already-enumerated corpus of one
/// instrument, z-score the numeric columns, and write
/// `state/voicings/{i}-features.json`.
pub fn featurize(instrument: Instrument) -> Result<FeaturizeArtifacts, VoicingsError> {
    let state = state_root();
    let corpus_path = state
        .join("voicings")
        .join(format!("{}-corpus.json", instrument.as_str()));
    let rows = load_corpus(&corpus_path)?;

    let mut matrix: Vec<Vec<f64>> = rows.iter().map(build_raw_feature_row).collect();

    // Sanity check — every row must match the schema.
    for (i, row) in matrix.iter().enumerate() {
        if row.len() != FEATURE_COLUMNS.len() {
            return Err(VoicingsError::ShapeMismatch {
                row: i,
                expected: FEATURE_COLUMNS.len(),
                got: row.len(),
            });
        }
    }

    let normalization = zscore_numeric_columns(&mut matrix);

    let features = FeatureMatrix {
        instrument: instrument.as_str().to_string(),
        columns: FEATURE_COLUMNS.iter().map(|s| s.to_string()).collect(),
        numeric_cols: NUMERIC_COLUMNS.iter().map(|s| s.to_string()).collect(),
        rows: matrix,
        normalization,
        voicing_count: rows.len(),
    };

    let features_path = state
        .join("voicings")
        .join(format!("{}-features.json", instrument.as_str()));
    std::fs::write(&features_path, serde_json::to_vec_pretty(&features)?)?;

    Ok(FeaturizeArtifacts {
        features_path,
        voicing_count: features.voicing_count,
        column_count: FEATURE_COLUMNS.len(),
    })
}

/// Build the un-normalized feature row for a single voicing. Order matches
/// [`FEATURE_COLUMNS`] exactly.
pub fn build_raw_feature_row(row: &VoicingRow) -> Vec<f64> {
    let played: Vec<i32> = row
        .frets
        .iter()
        .map(|s| s.parse::<i32>().unwrap_or(-1))
        .collect();

    // frets_used padded to 6. -1 means not played (or instrument has fewer
    // strings than the pad width).
    let mut padded: [f64; 6] = [-1.0; 6];
    for (i, f) in played.iter().enumerate().take(6) {
        padded[i] = *f as f64;
    }

    let string_count_played = played.iter().filter(|f| **f >= 0).count() as f64;

    // Barre heuristic — ≥3 adjacent played strings at the same fret equal
    // to `min_fret` (mirrors the plan's informal definition).
    let is_barre = detect_barre(&played, row.min_fret) as u8 as f64;

    let midi_count = row.midi_notes.len() as f64;
    let lowest_midi = row.midi_notes.iter().copied().min().unwrap_or(0) as f64;
    let highest_midi = row.midi_notes.iter().copied().max().unwrap_or(0) as f64;

    let quality = classify_quality(&row.midi_notes);
    let mut onehot = [0.0f64; QUALITY_VOCAB.len()];
    if let Some(idx) = QUALITY_VOCAB.iter().position(|q| *q == quality) {
        onehot[idx] = 1.0;
    }

    let mut out = Vec::with_capacity(FEATURE_COLUMNS.len());
    out.push(row.fret_span as f64);
    out.extend_from_slice(&padded);
    out.push(string_count_played);
    out.push(is_barre);
    out.push(row.min_fret as f64);
    out.push(row.max_fret as f64);
    out.push(midi_count);
    out.push(lowest_midi);
    out.push(highest_midi);
    out.extend_from_slice(&onehot);
    out
}

/// Return true if the voicing has ≥3 adjacent played strings at
/// `min_fret`. `frets` is per-string with -1 for muted.
fn detect_barre(frets: &[i32], min_fret: i32) -> bool {
    if min_fret <= 0 {
        return false;
    }
    let mut run = 0;
    for f in frets {
        if *f == min_fret {
            run += 1;
            if run >= 3 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Classify a voicing into one of [`QUALITY_VOCAB`] based on the set of
/// distinct pitch classes. Rooted at the lowest MIDI note.
fn classify_quality(midi: &[i32]) -> &'static str {
    if midi.is_empty() {
        return "other";
    }
    let lowest = *midi.iter().min().unwrap();
    let mut classes: Vec<i32> = midi
        .iter()
        .map(|n| (*n - lowest).rem_euclid(12))
        .collect();
    classes.sort_unstable();
    classes.dedup();

    match classes.as_slice() {
        [_, _] => "dyad",
        [0, 4, 7] => "maj",
        [0, 3, 7] => "min",
        [0, 3, 6] => "dim",
        [0, 4, 8] => "aug",
        [0, 2, 7] | [0, 5, 7] => "sus",
        [0, 4, 7, 11] => "maj7",
        [0, 4, 7, 10] => "dom7",
        [0, 3, 7, 10] => "min7",
        [0, 3, 6, 10] => "min7b5",
        [0, 3, 6, 9] => "dim7",
        s if s.len() == 4 => "other4",
        _ => "other",
    }
}

/// Z-score the numeric columns in-place and return the mean/stddev
/// lookup table used. Columns with stddev ≈ 0 are centered but not
/// scaled (to avoid dividing by zero).
fn zscore_numeric_columns(matrix: &mut [Vec<f64>]) -> HashMap<String, Normalization> {
    let mut norms = HashMap::new();
    if matrix.is_empty() {
        return norms;
    }

    for col_name in NUMERIC_COLUMNS {
        let col_idx = FEATURE_COLUMNS
            .iter()
            .position(|c| c == col_name)
            .expect("NUMERIC_COLUMNS must be a subset of FEATURE_COLUMNS");
        let n = matrix.len() as f64;
        let mean: f64 = matrix.iter().map(|r| r[col_idx]).sum::<f64>() / n;
        let var: f64 = matrix
            .iter()
            .map(|r| (r[col_idx] - mean).powi(2))
            .sum::<f64>()
            / n;
        let stddev = var.sqrt();
        for r in matrix.iter_mut() {
            let centered = r[col_idx] - mean;
            r[col_idx] = if stddev > f64::EPSILON {
                centered / stddev
            } else {
                centered
            };
        }
        norms.insert(
            (*col_name).to_string(),
            Normalization { mean, stddev },
        );
    }
    norms
}

/// Convenience value used in tests to identify a thin "manifest" payload
/// describing a finished run.
pub fn run_manifest(
    instrument: Instrument,
    enumerate: &EnumerateArtifacts,
    featurize: &FeaturizeArtifacts,
) -> Value {
    json!({
        "phase": "A",
        "instrument": instrument.as_str(),
        "voicing_count": enumerate.voicing_count,
        "feature_rows": featurize.voicing_count,
        "feature_cols": featurize.column_count,
        "artifacts": {
            "raw_jsonl": enumerate.raw_jsonl.display().to_string(),
            "corpus_json": enumerate.corpus_json.display().to_string(),
            "features_json": featurize.features_path.display().to_string(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row() -> VoicingRow {
        VoicingRow {
            instrument: "guitar".into(),
            string_count: 6,
            diagram: "3-1-x-x-x-x".into(),
            frets: vec![
                "3".into(),
                "1".into(),
                "x".into(),
                "x".into(),
                "x".into(),
                "x".into(),
            ],
            midi_notes: vec![67, 60],
            min_fret: 1,
            max_fret: 3,
            fret_span: 2,
        }
    }

    #[test]
    fn instrument_parse_roundtrip() {
        for inst in Instrument::ALL {
            assert_eq!(Instrument::parse(inst.as_str()).unwrap(), inst);
        }
        assert!(Instrument::parse("dulcimer").is_err());
    }

    #[test]
    fn feature_row_has_schema_length() {
        let row = sample_row();
        let fv = build_raw_feature_row(&row);
        assert_eq!(fv.len(), FEATURE_COLUMNS.len());
    }

    #[test]
    fn quality_classification_triads() {
        assert_eq!(classify_quality(&[60, 64, 67]), "maj"); // C major
        assert_eq!(classify_quality(&[60, 63, 67]), "min"); // C minor
        assert_eq!(classify_quality(&[60, 63, 66]), "dim");
        assert_eq!(classify_quality(&[60, 64, 68]), "aug");
        assert_eq!(classify_quality(&[60, 64, 67, 71]), "maj7");
        assert_eq!(classify_quality(&[60, 64, 67, 70]), "dom7");
        assert_eq!(classify_quality(&[60, 67]), "dyad");
    }

    #[test]
    fn barre_detected_on_three_in_a_row() {
        // fret 5 on strings 0,1,2 — barre at fret 5
        assert!(detect_barre(&[5, 5, 5, -1, -1, -1], 5));
        // open strings don't count
        assert!(!detect_barre(&[0, 0, 0, -1, -1, -1], 0));
        // only two in a row
        assert!(!detect_barre(&[5, 5, -1, 5, -1, -1], 5));
    }

    #[test]
    fn zscore_centers_numeric_columns() {
        let mut matrix = vec![
            build_raw_feature_row(&sample_row()),
            build_raw_feature_row(&VoicingRow {
                fret_span: 4,
                min_fret: 3,
                max_fret: 7,
                ..sample_row()
            }),
            build_raw_feature_row(&VoicingRow {
                fret_span: 0,
                min_fret: 0,
                max_fret: 0,
                ..sample_row()
            }),
        ];
        let norms = zscore_numeric_columns(&mut matrix);
        // fret_span column sum after z-scoring is ~0 (centering check).
        let col_idx = FEATURE_COLUMNS
            .iter()
            .position(|c| *c == "fret_span")
            .unwrap();
        let sum: f64 = matrix.iter().map(|r| r[col_idx]).sum();
        assert!(sum.abs() < 1e-9, "z-scored column should sum to ~0, got {sum}");
        assert!(norms.contains_key("fret_span"));
    }
}
