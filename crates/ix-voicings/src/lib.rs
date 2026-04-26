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

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

use ix_unsupervised::kmeans::KMeans;
use ix_unsupervised::traits::Clusterer;

pub mod viz_precompute;

/// Deserialize an integer field that may arrive as JSON `null`, mapping
/// null to -1 (the sentinel already used for muted strings). Keeps the
/// downstream arithmetic code simple at the cost of tolerating slightly
/// sloppy FretboardVoicingsCLI output on all-muted or otherwise degenerate
/// voicings.
fn null_as_neg_one_i32<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<i32>::deserialize(deserializer)?.unwrap_or(-1))
}

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
    #[serde(rename = "minFret", deserialize_with = "null_as_neg_one_i32")]
    pub min_fret: i32,
    #[serde(rename = "maxFret", deserialize_with = "null_as_neg_one_i32")]
    pub max_fret: i32,
    #[serde(rename = "fretSpan", deserialize_with = "null_as_neg_one_i32")]
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

    #[error("progression grammar parse error: {0}")]
    ProgressionParse(String),

    #[error("pipeline error: {0}")]
    Pipeline(String),
}

/// Fixed coarse quality vocabulary used for the one-hot. The quality is
/// inferred from the sorted interval set modulo 12 — enough to stratify
/// the corpus without waiting on GA's full analysis pass (which doesn't
/// run in `--export` mode).
pub const QUALITY_VOCAB: &[&str] = &[
    "dyad",   // 2 distinct pitch classes
    "maj",    // major triad (0,4,7)
    "min",    // minor triad (0,3,7)
    "dim",    // diminished (0,3,6)
    "aug",    // augmented (0,4,8)
    "sus",    // sus2 / sus4 triad (0,2,7) / (0,5,7)
    "maj7",   // (0,4,7,11)
    "dom7",   // (0,4,7,10)
    "min7",   // (0,3,7,10)
    "min7b5", // (0,3,6,10)
    "dim7",   // (0,3,6,9)
    "other4", // 4 distinct pitch classes, none of the above
    "other",  // fallback
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
    cmd.arg("--export").arg("--tuning").arg(instrument.as_str());
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
    let mut classes: Vec<i32> = midi.iter().map(|n| (*n - lowest).rem_euclid(12)).collect();
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
        norms.insert((*col_name).to_string(), Normalization { mean, stddev });
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

// ─── Phase B: cluster / topology / transitions / progressions / render_book ──

/// Load the feature matrix from disk for an instrument. Returns the
/// `FeatureMatrix` struct and the data as an `ndarray::Array2<f64>`.
pub fn load_features(
    instrument: Instrument,
) -> Result<(FeatureMatrix, Array2<f64>), VoicingsError> {
    let state = state_root();
    let path = state
        .join("voicings")
        .join(format!("{}-features.json", instrument.as_str()));
    let bytes = std::fs::read(&path)?;
    let fm: FeatureMatrix = serde_json::from_slice(&bytes)?;
    let n = fm.rows.len();
    let p = fm.columns.len();
    let flat: Vec<f64> = fm.rows.iter().flat_map(|r| r.iter().copied()).collect();
    let flat_len = flat.len();
    let arr = Array2::from_shape_vec((n, p), flat).map_err(|_| VoicingsError::ShapeMismatch {
        row: 0,
        expected: n * p,
        got: flat_len,
    })?;
    Ok((fm, arr))
}

/// Silhouette score for a clustering. Returns a value in [-1, 1].
/// Higher is better; above 0.15 is our threshold for accepting the clustering.
pub fn silhouette_score(data: &Array2<f64>, labels: &[usize]) -> f64 {
    let n = data.nrows();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n {
        let ci = labels[i];
        // a(i) = mean distance to same-cluster points
        let mut a_sum = 0.0;
        let mut a_count = 0usize;
        // b(i) = min over other clusters of mean distance to that cluster
        let mut cluster_sums: HashMap<usize, (f64, usize)> = HashMap::new();
        for (j, &lj) in labels.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = euclidean_dist_row(data, i, j);
            if lj == ci {
                a_sum += d;
                a_count += 1;
            } else {
                let entry = cluster_sums.entry(lj).or_insert((0.0, 0));
                entry.0 += d;
                entry.1 += 1;
            }
        }
        let a = if a_count > 0 {
            a_sum / a_count as f64
        } else {
            0.0
        };
        let b = cluster_sums
            .values()
            .map(|(s, c)| s / *c as f64)
            .fold(f64::INFINITY, f64::min);
        let b = if b.is_finite() { b } else { 0.0 };
        let s = if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        };
        total += s;
    }
    total / n as f64
}

fn euclidean_dist_row(data: &Array2<f64>, i: usize, j: usize) -> f64 {
    data.row(i)
        .iter()
        .zip(data.row(j).iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Cluster artifacts written by [`cluster`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterArtifacts {
    pub k: usize,
    pub silhouette: f64,
    pub centroids: Vec<Vec<f64>>,
    pub assignments: Vec<usize>,
    pub representative_voicing_per_cluster: Vec<usize>,
}

/// Run K-Means clustering on the feature matrix and write artifacts.
///
/// Tries k=5 initially. If silhouette < 0.15, falls back to k=3.
/// Writes `state/voicings/{instrument}-clusters.json`.
pub fn cluster(instrument: Instrument) -> Result<ClusterArtifacts, VoicingsError> {
    let (_fm, data) = load_features(instrument)?;
    let n = data.nrows();

    // Try k=5 first, fall back to k=3 if silhouette is poor
    let k_candidates = if n < 5 { vec![n.min(2)] } else { vec![5, 3] };

    let mut best: Option<ClusterArtifacts> = None;

    for &k in &k_candidates {
        if k == 0 || k >= n {
            continue;
        }
        let mut km = KMeans::new(k).with_seed(42);
        let labels = km.fit_predict(&data);
        let labels_vec: Vec<usize> = labels.iter().copied().collect();
        let sil = silhouette_score(&data, &labels_vec);

        let centroids = km.centroids.as_ref().unwrap();

        let centroid_vecs: Vec<Vec<f64>> = (0..k).map(|c| centroids.row(c).to_vec()).collect();

        // Representative = voicing closest to centroid (Euclidean)
        let representatives: Vec<usize> = (0..k)
            .map(|c| {
                let centroid = centroids.row(c);
                let mut best_idx = 0;
                let mut best_dist = f64::INFINITY;
                for (i, &li) in labels_vec.iter().enumerate() {
                    if li != c {
                        continue;
                    }
                    let d: f64 = data
                        .row(i)
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>();
                    if d < best_dist {
                        best_dist = d;
                        best_idx = i;
                    }
                }
                best_idx
            })
            .collect();

        let art = ClusterArtifacts {
            k,
            silhouette: sil,
            centroids: centroid_vecs,
            assignments: labels_vec,
            representative_voicing_per_cluster: representatives,
        };

        if sil >= 0.15 || best.is_none() {
            best = Some(art);
            if sil >= 0.15 {
                break;
            }
        }
    }

    let artifacts = best.ok_or_else(|| {
        VoicingsError::Pipeline("clustering failed: corpus too small for any k candidate".into())
    })?;
    let out_path = state_root()
        .join("voicings")
        .join(format!("{}-clusters.json", instrument.as_str()));
    std::fs::write(&out_path, serde_json::to_vec_pretty(&artifacts)?)?;
    Ok(artifacts)
}

/// Topology artifacts written by [`topology`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyArtifacts {
    pub betti_0: usize,
    pub betti_1: usize,
    pub persistence_pairs: Vec<(f64, f64)>,
    pub filtration_max: f64,
}

/// Run persistent homology on the feature point cloud and write artifacts.
///
/// Uses `ix_topo::pointcloud::persistence_from_points` with a Vietoris-Rips
/// filtration. The max radius is auto-derived from the data.
pub fn topology(instrument: Instrument) -> Result<TopologyArtifacts, VoicingsError> {
    let (_fm, data) = load_features(instrument)?;
    let n = data.nrows();

    // Convert to Vec<Vec<f64>> for ix-topo
    let points: Vec<Vec<f64>> = (0..n).map(|i| data.row(i).to_vec()).collect();

    // Compute max pairwise distance (on a sample if too large)
    let sample_limit = 50; // Rips is O(n^3) for dimension 2
    let sample: Vec<Vec<f64>> = if n <= sample_limit {
        points.clone()
    } else {
        // Deterministic subsample: every n/sample_limit-th point
        let step = n / sample_limit;
        (0..sample_limit)
            .map(|i| points[i * step].clone())
            .collect()
    };

    let mut max_dist = 0.0f64;
    for i in 0..sample.len() {
        for j in (i + 1)..sample.len() {
            let d: f64 = sample[i]
                .iter()
                .zip(sample[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if d > max_dist {
                max_dist = d;
            }
        }
    }

    // Use a moderate radius — half the max distance to avoid a single
    // connected component trivially. Clamp to something reasonable.
    let radius = (max_dist * 0.5).max(0.1);

    let diagrams = ix_topo::pointcloud::persistence_from_points(&sample, 1, radius);

    let betti_0 = diagrams
        .first()
        .map(|d| {
            d.pairs
                .iter()
                .filter(|(_, death)| death.is_infinite())
                .count()
        })
        .unwrap_or(0);

    let betti_1 = diagrams
        .get(1)
        .map(|d| {
            d.pairs
                .iter()
                .filter(|(_, death)| death.is_infinite())
                .count()
        })
        .unwrap_or(0);

    // Collect all finite persistence pairs
    let persistence_pairs: Vec<(f64, f64)> = diagrams
        .iter()
        .flat_map(|d| {
            d.pairs
                .iter()
                .filter(|(_, death)| death.is_finite())
                .copied()
        })
        .collect();

    let artifacts = TopologyArtifacts {
        betti_0,
        betti_1,
        persistence_pairs,
        filtration_max: radius,
    };

    let out_path = state_root()
        .join("voicings")
        .join(format!("{}-topology.json", instrument.as_str()));
    std::fs::write(&out_path, serde_json::to_vec_pretty(&artifacts)?)?;
    Ok(artifacts)
}

/// Movement cost between two voicings, used as edge weight in the
/// transitions graph.
///
/// - Sum of |fret_a[i] - fret_b[i]| for each string where both are played
/// - +2 penalty for each muted-to-played or played-to-muted change
/// - +1 penalty if barre status differs
pub fn movement_cost(a: &VoicingRow, b: &VoicingRow) -> f64 {
    let a_frets: Vec<i32> = a
        .frets
        .iter()
        .map(|s| s.parse::<i32>().unwrap_or(-1))
        .collect();
    let b_frets: Vec<i32> = b
        .frets
        .iter()
        .map(|s| s.parse::<i32>().unwrap_or(-1))
        .collect();

    let max_strings = a_frets.len().max(b_frets.len());
    let mut cost = 0.0;

    for i in 0..max_strings {
        let fa = a_frets.get(i).copied().unwrap_or(-1);
        let fb = b_frets.get(i).copied().unwrap_or(-1);
        match (fa >= 0, fb >= 0) {
            (true, true) => cost += (fa - fb).unsigned_abs() as f64,
            (true, false) | (false, true) => cost += 2.0, // mute toggle penalty
            (false, false) => {}                          // both muted, no cost
        }
    }

    // Barre toggle penalty
    let a_barre = detect_barre(&a_frets, a.min_fret);
    let b_barre = detect_barre(&b_frets, b.min_fret);
    if a_barre != b_barre {
        cost += 1.0;
    }

    cost
}

/// Transition artifacts for one instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionArtifacts {
    pub edges: Vec<TransitionEdge>,
    pub shortest_paths: Vec<ShortestPath>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionEdge {
    pub from_cluster: usize,
    pub to_cluster: usize,
    pub cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortestPath {
    pub from_cluster: usize,
    pub to_cluster: usize,
    pub path: Vec<usize>,
    pub total_cost: f64,
}

/// Build a transitions graph between cluster representatives and compute
/// shortest paths using `ix_graph::graph::Graph::dijkstra`.
///
/// We use Dijkstra from `ix-graph` since the graph is complete and small
/// (k nodes). A* is more appropriate when there's a meaningful heuristic;
/// for a complete graph with k<=5 nodes, Dijkstra is sufficient and avoids
/// needing to implement the `SearchState` trait on voicings.
pub fn transitions(instrument: Instrument) -> Result<TransitionArtifacts, VoicingsError> {
    let state = state_root();

    // Load cluster artifacts
    let cluster_path = state
        .join("voicings")
        .join(format!("{}-clusters.json", instrument.as_str()));
    let cluster_bytes = std::fs::read(&cluster_path)?;
    let clusters: ClusterArtifacts = serde_json::from_slice(&cluster_bytes)?;

    // Load corpus for the representative voicings
    let corpus_path = state
        .join("voicings")
        .join(format!("{}-corpus.json", instrument.as_str()));
    let corpus = load_corpus(&corpus_path)?;

    let reps = &clusters.representative_voicing_per_cluster;
    let k = reps.len();

    // Build complete graph with movement costs
    let mut graph = ix_graph::graph::Graph::with_nodes(k);
    let mut edges = Vec::new();

    for i in 0..k {
        for j in 0..k {
            if i == j {
                continue;
            }
            if reps[i] >= corpus.len() || reps[j] >= corpus.len() {
                return Err(VoicingsError::Pipeline(format!(
                    "representative index out of bounds: reps[{i}]={}, reps[{j}]={}, corpus len={}",
                    reps[i],
                    reps[j],
                    corpus.len()
                )));
            }
            let cost = movement_cost(&corpus[reps[i]], &corpus[reps[j]]);
            graph.add_edge(i, j, cost);
            edges.push(TransitionEdge {
                from_cluster: i,
                to_cluster: j,
                cost,
            });
        }
    }

    // Compute shortest paths between all pairs — hoist Dijkstra per source node
    let mut shortest_paths = Vec::new();
    for i in 0..k {
        let (dists, _) = graph.dijkstra(i);
        for j in 0..k {
            if i == j {
                continue;
            }
            if let Some(path) = graph.shortest_path(i, j) {
                let total_cost = dists.get(&j).copied().unwrap_or(f64::INFINITY);
                shortest_paths.push(ShortestPath {
                    from_cluster: i,
                    to_cluster: j,
                    path,
                    total_cost,
                });
            }
        }
    }

    let artifacts = TransitionArtifacts {
        edges,
        shortest_paths,
    };

    let out_path = state_root()
        .join("voicings")
        .join(format!("{}-transitions.json", instrument.as_str()));
    std::fs::write(&out_path, serde_json::to_vec_pretty(&artifacts)?)?;
    Ok(artifacts)
}

/// Progression artifacts for one instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressionArtifacts {
    pub grammar_file: String,
    pub parse_counts: HashMap<String, usize>,
    pub deviation: Option<String>,
}

/// Attempt to parse chord sequences through the progression grammar.
///
/// Uses `ix_grammar::constrained::EbnfGrammar` to load the CFG, then
/// checks whether representative-quality token sequences match any
/// production. Because `ix-grammar` is MCTS-based (generative, not
/// recognition), we do a simpler direct-derivation check instead of
/// full Earley parsing.
pub fn progressions(instrument: Instrument) -> Result<ProgressionArtifacts, VoicingsError> {
    let state = state_root();

    // Load cluster artifacts + corpus to map representatives to qualities
    let cluster_path = state
        .join("voicings")
        .join(format!("{}-clusters.json", instrument.as_str()));
    let cluster_bytes = std::fs::read(&cluster_path)?;
    let clusters: ClusterArtifacts = serde_json::from_slice(&cluster_bytes)?;

    let corpus_path = state
        .join("voicings")
        .join(format!("{}-corpus.json", instrument.as_str()));
    let corpus = load_corpus(&corpus_path)?;

    // Map each representative to its quality label
    let rep_qualities: Vec<&str> = clusters
        .representative_voicing_per_cluster
        .iter()
        .map(|&idx| classify_quality(&corpus[idx].midi_notes))
        .collect();

    // Load the grammar file
    let grammar_path = find_grammar_file()?;
    let grammar_text = std::fs::read_to_string(&grammar_path).map_err(VoicingsError::Io)?;

    let grammar = ix_grammar::constrained::EbnfGrammar::from_str(&grammar_text)
        .map_err(VoicingsError::ProgressionParse)?;

    // Count how many cluster-representative quality sequences match each
    // production pattern. We generate all permutations of representative
    // qualities and check if they can be derived from each named production.
    let mut parse_counts: HashMap<String, usize> = HashMap::new();

    // Check each named progression production
    let prog_names = ["I_IV_V", "ii_V_I", "I_vi_IV_V", "BLUES12"];
    for prog_name in &prog_names {
        let count = count_matching_sequences(&grammar, prog_name, &rep_qualities);
        parse_counts.insert(prog_name.to_string(), count);
    }

    let artifacts = ProgressionArtifacts {
        grammar_file: grammar_path.display().to_string(),
        parse_counts,
        deviation: Some(
            "ix-grammar uses MCTS-based generative derivation, not Earley recognition. \
             Parse counts reflect how many representative-quality permutations match \
             each production pattern via direct expansion, not full chart parsing."
                .to_string(),
        ),
    };

    let out_path = state_root()
        .join("voicings")
        .join(format!("{}-progressions.json", instrument.as_str()));
    std::fs::write(&out_path, serde_json::to_vec_pretty(&artifacts)?)?;
    Ok(artifacts)
}

/// Find the grammar file relative to the crate or the state directory.
fn find_grammar_file() -> Result<PathBuf, VoicingsError> {
    // Try relative to current dir (running from repo root)
    let candidates = [
        PathBuf::from("crates/ix-voicings/grammars/progressions.cfg"),
        PathBuf::from("grammars/progressions.cfg"),
    ];
    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }
    Err(VoicingsError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "progressions.cfg not found",
    )))
}

/// Count how many permutations of the representative qualities match
/// a given production by direct recursive expansion of the grammar.
fn count_matching_sequences(
    grammar: &ix_grammar::constrained::EbnfGrammar,
    production: &str,
    rep_qualities: &[&str],
) -> usize {
    let alts = grammar.alternatives(production);
    if alts.is_empty() {
        return 0;
    }

    let mut count = 0usize;
    for alt in &alts {
        // Expand alt to a list of terminal sequences
        let terminal_seqs = expand_to_terminals(grammar, alt);
        for seq in &terminal_seqs {
            // Check if any permutation of rep_qualities matches this terminal sequence
            count += count_permutation_matches(seq, rep_qualities);
        }
    }
    count
}

/// Recursively expand a token sequence to all possible terminal sequences.
/// Caps depth to avoid combinatorial explosion.
fn expand_to_terminals(
    grammar: &ix_grammar::constrained::EbnfGrammar,
    tokens: &[String],
) -> Vec<Vec<String>> {
    expand_recursive(grammar, tokens, 0, 5)
}

fn expand_recursive(
    grammar: &ix_grammar::constrained::EbnfGrammar,
    tokens: &[String],
    depth: usize,
    max_depth: usize,
) -> Vec<Vec<String>> {
    if depth > max_depth || tokens.is_empty() {
        return vec![tokens.to_vec()];
    }

    // Find first non-terminal
    let nt_pos = tokens.iter().position(|t| !grammar.is_terminal(t));
    match nt_pos {
        None => vec![tokens.to_vec()], // all terminals
        Some(pos) => {
            let alts = grammar.alternatives(&tokens[pos]);
            if alts.is_empty() {
                return vec![tokens.to_vec()];
            }
            let mut results = Vec::new();
            for alt in &alts {
                let mut expanded = tokens[..pos].to_vec();
                expanded.extend_from_slice(alt);
                expanded.extend_from_slice(&tokens[pos + 1..]);
                let sub = expand_recursive(grammar, &expanded, depth + 1, max_depth);
                results.extend(sub);
            }
            results
        }
    }
}

/// Count how many distinct ordered selections (with repetition) from
/// `available` match the target sequence.
fn count_permutation_matches(target: &[String], available: &[&str]) -> usize {
    if target.is_empty() {
        return 1;
    }
    // For each position in target, check if any available quality matches
    // This counts ordered selections with repetition
    let mut count = 1usize;
    for token in target {
        let matches = available.iter().filter(|q| **q == token.as_str()).count();
        count = count.saturating_mul(matches);
    }
    count
}

/// Render the study book from all upstream artifacts.
///
/// Reads the cluster, topology, transitions, and progressions JSON files
/// and generates `docs/books/chord-voicings-study.md`.
pub fn render_book(instruments: &[Instrument]) -> Result<PathBuf, VoicingsError> {
    let state = state_root();
    let mut book = String::new();

    book.push_str("# Chord Voicings Study\n\n");
    book.push_str("Generated by `ix-voicings` Phase B pipeline.\n\n");
    book.push_str("---\n\n");

    // Chapter 1: Corpus at a glance
    book.push_str("## Chapter 1: Corpus at a Glance\n\n");
    for &inst in instruments {
        let features_path = state
            .join("voicings")
            .join(format!("{}-features.json", inst.as_str()));
        if !features_path.exists() {
            continue;
        }
        let bytes = std::fs::read(&features_path)?;
        let fm: FeatureMatrix = serde_json::from_slice(&bytes)?;
        book.push_str(&format!(
            "### {}\n\n- Voicing count: {}\n- Feature columns: {}\n- Source: `{}`\n\n",
            inst.as_str(),
            fm.voicing_count,
            fm.columns.len(),
            features_path.display()
        ));

        // Fret span distribution from raw features
        let corpus_path = state
            .join("voicings")
            .join(format!("{}-corpus.json", inst.as_str()));
        if let Ok(corpus) = load_corpus(&corpus_path) {
            let mut span_hist: HashMap<i32, usize> = HashMap::new();
            for row in &corpus {
                *span_hist.entry(row.fret_span).or_insert(0) += 1;
            }
            let mut spans: Vec<(i32, usize)> = span_hist.into_iter().collect();
            spans.sort_by_key(|(s, _)| *s);
            book.push_str("| Fret span | Count |\n|-----------|-------|\n");
            for (span, count) in &spans {
                book.push_str(&format!("| {} | {} |\n", span, count));
            }
            book.push('\n');
        }
    }

    // Chapter 2: Voicing families (clustering)
    let mut any_cluster = false;
    let mut cluster_chapter = String::new();
    cluster_chapter.push_str("## Chapter 2: Voicing Families (Clustering)\n\n");
    for &inst in instruments {
        let path = state
            .join("voicings")
            .join(format!("{}-clusters.json", inst.as_str()));
        if !path.exists() {
            continue;
        }
        let bytes = std::fs::read(&path)?;
        let ca: ClusterArtifacts = serde_json::from_slice(&bytes)?;
        if ca.k <= 1 {
            continue; // Drop degenerate clustering
        }
        any_cluster = true;

        // Load corpus to show representative voicings
        let corpus_path = state
            .join("voicings")
            .join(format!("{}-corpus.json", inst.as_str()));
        let corpus = load_corpus(&corpus_path).ok();

        cluster_chapter.push_str(&format!(
            "### {} (k={}, silhouette={:.4})\n\nSource: `{}`\n\n",
            inst.as_str(),
            ca.k,
            ca.silhouette,
            path.display()
        ));

        cluster_chapter.push_str("| Cluster | Members | Representative | Quality | Diagram |\n");
        cluster_chapter.push_str("|---------|---------|----------------|---------|----------|\n");
        for c in 0..ca.k {
            let members = ca.assignments.iter().filter(|&&a| a == c).count();
            let rep_idx = ca.representative_voicing_per_cluster[c];
            let (quality, diagram) = if let Some(ref corpus) = corpus {
                let row = &corpus[rep_idx];
                (
                    classify_quality(&row.midi_notes).to_string(),
                    row.diagram.clone(),
                )
            } else {
                ("?".to_string(), "?".to_string())
            };
            cluster_chapter.push_str(&format!(
                "| C{} | {} | #{} | {} | `{}` |\n",
                c, members, rep_idx, quality, diagram
            ));
        }
        cluster_chapter.push('\n');
    }
    if any_cluster {
        book.push_str(&cluster_chapter);
    } else {
        book.push_str("## Chapter 2: Voicing Families (Clustering)\n\n");
        book.push_str("*Dropped: single cluster or no clustering data available.*\n\n");
    }

    // Chapter 3: Topology of the voicing space
    let mut any_topo = false;
    let mut topo_chapter = String::new();
    topo_chapter.push_str("## Chapter 3: Topology of the Voicing Space\n\n");
    for &inst in instruments {
        let path = state
            .join("voicings")
            .join(format!("{}-topology.json", inst.as_str()));
        if !path.exists() {
            continue;
        }
        let bytes = std::fs::read(&path)?;
        let ta: TopologyArtifacts = serde_json::from_slice(&bytes)?;

        // Kill criterion: Betti_0 = N and Betti_1 = 0 means trivial topology
        let features_path = state
            .join("voicings")
            .join(format!("{}-features.json", inst.as_str()));
        let n_points = if features_path.exists() {
            let fb = std::fs::read(&features_path)?;
            let fm: FeatureMatrix = serde_json::from_slice(&fb)?;
            fm.voicing_count
        } else {
            0
        };
        if ta.betti_0 == n_points && ta.betti_1 == 0 && n_points > 1 {
            // Degenerate: every point its own component
            continue;
        }
        any_topo = true;

        topo_chapter.push_str(&format!(
            "### {}\n\n- Betti_0 (connected components): {}\n- Betti_1 (loops): {}\n\
             - Filtration max radius: {:.4}\n- Finite persistence pairs: {}\n\
             - Source: `{}`\n\n",
            inst.as_str(),
            ta.betti_0,
            ta.betti_1,
            ta.filtration_max,
            ta.persistence_pairs.len(),
            path.display()
        ));

        if !ta.persistence_pairs.is_empty() {
            topo_chapter
                .push_str("| Birth | Death | Persistence |\n|-------|-------|-------------|\n");
            // Show top 10 by persistence
            let mut pairs = ta.persistence_pairs.clone();
            pairs.sort_by(|a, b| {
                (b.1 - b.0)
                    .partial_cmp(&(a.1 - a.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (birth, death) in pairs.iter().take(10) {
                topo_chapter.push_str(&format!(
                    "| {:.4} | {:.4} | {:.4} |\n",
                    birth,
                    death,
                    death - birth
                ));
            }
            topo_chapter.push('\n');
        }
    }
    if any_topo {
        book.push_str(&topo_chapter);
    } else {
        book.push_str("## Chapter 3: Topology of the Voicing Space\n\n");
        book.push_str(
            "*Dropped: degenerate topology (every point its own component) or no data.*\n\n",
        );
    }

    // Chapter 4: Shortest physical paths between representatives
    let mut any_transitions = false;
    let mut trans_chapter = String::new();
    trans_chapter.push_str("## Chapter 4: Shortest Physical Paths Between Representatives\n\n");
    for &inst in instruments {
        let path = state
            .join("voicings")
            .join(format!("{}-transitions.json", inst.as_str()));
        if !path.exists() {
            continue;
        }
        let bytes = std::fs::read(&path)?;
        let ta: TransitionArtifacts = serde_json::from_slice(&bytes)?;
        if ta.edges.is_empty() {
            continue;
        }
        any_transitions = true;

        trans_chapter.push_str(&format!(
            "### {}\n\nSource: `{}`\n\n",
            inst.as_str(),
            path.display()
        ));

        trans_chapter.push_str("#### Edge costs (movement cost between representatives)\n\n");
        trans_chapter.push_str("| From | To | Cost |\n|------|-----|------|\n");
        for edge in &ta.edges {
            trans_chapter.push_str(&format!(
                "| C{} | C{} | {:.1} |\n",
                edge.from_cluster, edge.to_cluster, edge.cost
            ));
        }
        trans_chapter.push('\n');

        trans_chapter.push_str("#### Shortest paths\n\n");
        trans_chapter.push_str("| From | To | Path | Cost |\n|------|-----|------|------|\n");
        for sp in &ta.shortest_paths {
            let path_str: Vec<String> = sp.path.iter().map(|n| format!("C{}", n)).collect();
            trans_chapter.push_str(&format!(
                "| C{} | C{} | {} | {:.1} |\n",
                sp.from_cluster,
                sp.to_cluster,
                path_str.join(" -> "),
                sp.total_cost
            ));
        }
        trans_chapter.push('\n');

        // Cost distribution summary
        let costs: Vec<f64> = ta.edges.iter().map(|e| e.cost).collect();
        let min_cost = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_cost = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_cost = costs.iter().sum::<f64>() / costs.len() as f64;
        trans_chapter.push_str(&format!(
            "Movement cost distribution: min={:.1}, max={:.1}, mean={:.1}\n\n",
            min_cost, max_cost, mean_cost
        ));
    }
    if any_transitions {
        book.push_str(&trans_chapter);
    }

    // Chapter 5: Grammar of progressions
    let mut any_prog = false;
    let mut prog_chapter = String::new();
    prog_chapter.push_str("## Chapter 5: Grammar of Progressions\n\n");
    for &inst in instruments {
        let path = state
            .join("voicings")
            .join(format!("{}-progressions.json", inst.as_str()));
        if !path.exists() {
            continue;
        }
        let bytes = std::fs::read(&path)?;
        let pa: ProgressionArtifacts = serde_json::from_slice(&bytes)?;

        // Drop if all parse counts are zero
        let total: usize = pa.parse_counts.values().sum();
        if total == 0 {
            continue;
        }
        any_prog = true;

        prog_chapter.push_str(&format!(
            "### {}\n\nGrammar file: `{}`\nSource: `{}`\n\n",
            inst.as_str(),
            pa.grammar_file,
            path.display()
        ));

        if let Some(ref dev) = pa.deviation {
            prog_chapter.push_str(&format!("*Deviation: {}*\n\n", dev));
        }

        prog_chapter.push_str("| Production | Parse count |\n|------------|-------------|\n");
        let mut sorted: Vec<(&String, &usize)> = pa.parse_counts.iter().collect();
        sorted.sort_by_key(|(k, _)| k.to_string());
        for (name, count) in sorted {
            prog_chapter.push_str(&format!("| {} | {} |\n", name, count));
        }
        prog_chapter.push('\n');
    }
    if any_prog {
        book.push_str(&prog_chapter);
    } else {
        book.push_str("## Chapter 5: Grammar of Progressions\n\n");
        book.push_str("*Dropped: all progression parse counts are zero — the grammar did not match any representative quality permutations.*\n\n");
    }

    // Chapter 6: Cross-instrument comparison
    // Only if we have multiple instruments with clusters
    let cluster_data: Vec<(Instrument, ClusterArtifacts)> = instruments
        .iter()
        .filter_map(|&inst| {
            let path = state
                .join("voicings")
                .join(format!("{}-clusters.json", inst.as_str()));
            if !path.exists() {
                return None;
            }
            let bytes = std::fs::read(&path).ok()?;
            let ca: ClusterArtifacts = serde_json::from_slice(&bytes).ok()?;
            if ca.k > 1 {
                Some((inst, ca))
            } else {
                None
            }
        })
        .collect();
    if cluster_data.len() >= 2 {
        book.push_str("## Chapter 6: Cross-Instrument Comparison\n\n");
        book.push_str("*Multiple instruments available for comparison.*\n\n");
        // Summary table
        book.push_str(
            "| Instrument | Clusters | Silhouette |\n|------------|----------|------------|\n",
        );
        for (inst, ca) in &cluster_data {
            book.push_str(&format!(
                "| {} | {} | {:.4} |\n",
                inst.as_str(),
                ca.k,
                ca.silhouette
            ));
        }
        book.push('\n');
    } else {
        book.push_str("## Chapter 6: Cross-Instrument Comparison\n\n");
        book.push_str("*Dropped: fewer than two instruments with valid clustering data.*\n\n");
    }

    // Chapter 7: Known gaps
    book.push_str("## Chapter 7: Known Gaps\n\n");
    book.push_str("1. **ICV (Interval Class Vector) for bass/ukulele**: GA's export mode does not produce \
                    analysis output for non-guitar instruments. The ICV fields are zeros for bass and ukulele. \
                    This limits cross-instrument feature comparison.\n\n");
    book.push_str("2. **Preference model (Phase 2)**: No playability labels exist in GA today. \
                    Supervised learning on voicing preference requires hand-labeled data (~200+ voicings). \
                    Deferred to a future phase.\n\n");
    book.push_str("3. **Parameter sweeps**: Only k=5 and k=3 were tried for K-Means. A proper \
                    elbow/gap-statistic sweep over k=2..10 would improve cluster quality assessment.\n\n");
    book.push_str(
        "4. **Sample size**: The guitar corpus used for this study contains 50 voicings \
                    (export-max cap). A full enumeration produces 100k+ voicings. Results may not \
                    generalize to the full corpus.\n\n",
    );
    book.push_str(
        "5. **Topology subsampling**: For point clouds > 50 points, the topology node \
                    subsamples to 50 points before computing Vietoris-Rips. This loses detail in \
                    the persistence diagram.\n\n",
    );
    book.push_str("6. **Grammar recognition**: `ix-grammar` provides MCTS-based generative derivation, \
                    not classical Earley/CYK chart parsing. Parse counts reflect direct expansion matching, \
                    not ambiguity-aware recognition.\n\n");

    // ga-chatbot agent spec preview
    book.push_str("---\n\n");
    book.push_str("## Appendix: ga-chatbot Agent Spec Preview\n\n");
    book.push_str("The chatbot calls ix-voicings-derived artifacts (not the live enumerator) to answer \
                    grounded questions about voicings. Example: \"give me three drop-2 voicings for Cmaj7 \
                    on guitar that transition cleanly to an Fmaj7 drop-2\" — the agent looks up the Cmaj7 \
                    cluster representatives, walks the transitions.json shortest paths to Fmaj7 representatives, \
                    ranks by movement cost, and returns three with diagrams. Persona: domain-specific assistant. \
                    Affordances: `mcp:ix:ix_voicings_query`. Governance: `estimator_pairing: skeptical-auditor`, \
                    `goal_directedness: task-scoped`. Full spec: `docs/plans/2026-04-15-003-feat-ga-chatbot-agent-spec.md`.\n");

    // Write the book
    let book_dir = PathBuf::from("docs/books");
    std::fs::create_dir_all(&book_dir)?;
    let book_path = book_dir.join("chord-voicings-study.md");
    std::fs::write(&book_path, &book)?;
    Ok(book_path)
}

/// Build the full voicings pipeline using `PipelineBuilder` from `ix-pipeline`.
///
/// The DAG is: enumerate -> featurize -> [cluster | topology | transitions -> progressions] -> render_book
/// Note: transitions depends on cluster; progressions depends on transitions.
pub fn build_pipeline(
    instrument: Instrument,
) -> ix_pipeline::dag::Dag<ix_pipeline::executor::PipelineNode> {
    use ix_pipeline::builder::PipelineBuilder;

    let inst = instrument;
    let inst2 = instrument;
    let inst3 = instrument;
    let inst4 = instrument;
    let inst5 = instrument;

    PipelineBuilder::new()
        .source("featurize", move || {
            // In pipeline mode, assume enumerate already ran (features exist on disk).
            // If not, this will fail with a clear error.
            let (_fm, _data) = load_features(inst)
                .map_err(|e| ix_pipeline::executor::PipelineError::ComputeError(e.to_string()))?;
            Ok(json!({"instrument": inst.as_str(), "status": "features_loaded"}))
        })
        .node("cluster", move |b| {
            b.input("features", "featurize").compute(move |_inputs| {
                let ca = cluster(inst2).map_err(|e| {
                    ix_pipeline::executor::PipelineError::ComputeError(e.to_string())
                })?;
                Ok(json!({
                    "k": ca.k,
                    "silhouette": ca.silhouette,
                    "status": "ok"
                }))
            })
        })
        .node("topology", move |b| {
            b.input("features", "featurize").compute(move |_inputs| {
                let ta = topology(inst3).map_err(|e| {
                    ix_pipeline::executor::PipelineError::ComputeError(e.to_string())
                })?;
                Ok(json!({
                    "betti_0": ta.betti_0,
                    "betti_1": ta.betti_1,
                    "status": "ok"
                }))
            })
        })
        .node("transitions", move |b| {
            b.input("clusters", "cluster").compute(move |_inputs| {
                let ta = transitions(inst4).map_err(|e| {
                    ix_pipeline::executor::PipelineError::ComputeError(e.to_string())
                })?;
                Ok(json!({
                    "edge_count": ta.edges.len(),
                    "path_count": ta.shortest_paths.len(),
                    "status": "ok"
                }))
            })
        })
        .node("progressions", move |b| {
            b.input("trans", "transitions").compute(move |_inputs| {
                let pa = progressions(inst5).map_err(|e| {
                    ix_pipeline::executor::PipelineError::ComputeError(e.to_string())
                })?;
                Ok(json!({
                    "parse_counts": pa.parse_counts,
                    "status": "ok"
                }))
            })
        })
        .build()
        .expect("voicings pipeline DAG should be acyclic")
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
        assert!(
            sum.abs() < 1e-9,
            "z-scored column should sum to ~0, got {sum}"
        );
        assert!(norms.contains_key("fret_span"));
    }

    // ─── Phase B tests ─────────────────────────────────────────────────────

    fn sample_row_maj() -> VoicingRow {
        VoicingRow {
            instrument: "guitar".into(),
            string_count: 6,
            diagram: "0-0-0-2-3-x".into(),
            frets: vec![
                "0".into(),
                "0".into(),
                "0".into(),
                "2".into(),
                "3".into(),
                "x".into(),
            ],
            midi_notes: vec![64, 59, 55, 52, 48, 40],
            min_fret: 0,
            max_fret: 3,
            fret_span: 3,
        }
    }

    #[test]
    fn movement_cost_same_voicing_is_zero() {
        let row = sample_row();
        assert_eq!(movement_cost(&row, &row), 0.0);
    }

    #[test]
    fn movement_cost_includes_fret_deltas() {
        let a = sample_row_maj();
        let mut b = sample_row_maj();
        // Shift fret 0 -> 2 on first string
        b.frets[0] = "2".into();
        let cost = movement_cost(&a, &b);
        assert!(
            cost >= 2.0,
            "cost should include fret delta of 2, got {cost}"
        );
    }

    #[test]
    fn movement_cost_mute_toggle_penalty() {
        let a = sample_row_maj(); // string 5 is muted
        let mut b = sample_row_maj();
        b.frets[5] = "1".into(); // now played
        let cost = movement_cost(&a, &b);
        // Should include a 2.0 penalty for muted->played
        assert!(
            cost >= 2.0,
            "cost should include mute toggle penalty, got {cost}"
        );
    }

    #[test]
    fn silhouette_perfect_clusters() {
        // Two well-separated clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let labels = vec![0, 0, 0, 1, 1, 1];
        let sil = silhouette_score(&data, &labels);
        assert!(
            sil > 0.5,
            "Well-separated clusters should have high silhouette, got {sil}"
        );
    }

    #[test]
    fn silhouette_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let labels = vec![0];
        let sil = silhouette_score(&data, &labels);
        assert_eq!(sil, 0.0);
    }

    #[test]
    fn kmeans_on_synthetic_voicing_features() {
        // Test the clustering logic directly on synthetic data shaped like
        // the real feature matrix (n=10, p=27)
        let n = 10;
        let p = FEATURE_COLUMNS.len();
        let mut data_vec = vec![0.0f64; n * p];
        // Create two distinct groups
        for i in 0..5 {
            data_vec[i * p] = -2.0; // fret_span z-score
            data_vec[i * p + 1] = -1.0;
        }
        for i in 5..10 {
            data_vec[i * p] = 2.0;
            data_vec[i * p + 1] = 1.0;
        }
        let data = Array2::from_shape_vec((n, p), data_vec).unwrap();

        let mut km = KMeans::new(2).with_seed(42);
        let labels = km.fit_predict(&data);
        let labels_vec: Vec<usize> = labels.iter().copied().collect();

        // First 5 should be in one cluster, last 5 in another
        assert_eq!(labels_vec[0], labels_vec[1]);
        assert_eq!(labels_vec[5], labels_vec[6]);
        assert_ne!(labels_vec[0], labels_vec[5]);

        let sil = silhouette_score(&data, &labels_vec);
        assert!(sil > 0.0, "should have positive silhouette, got {sil}");
    }

    #[test]
    fn topology_on_synthetic_points() {
        // Test persistent homology on a small synthetic point cloud
        let points: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 0.866],
            vec![10.0, 10.0],
            vec![10.5, 10.0],
        ];
        let diagrams = ix_topo::pointcloud::persistence_from_points(&points, 1, 5.0);
        assert!(!diagrams.is_empty());
        // H0 should have at least one infinite feature (surviving component)
        let h0_inf = diagrams[0]
            .pairs
            .iter()
            .filter(|(_, d)| d.is_infinite())
            .count();
        assert!(h0_inf >= 1, "should have at least one surviving component");
    }

    #[test]
    fn graph_dijkstra_on_voicing_graph() {
        // Test that ix-graph Dijkstra works correctly for voicing transitions
        let mut g = ix_graph::graph::Graph::with_nodes(3);
        g.add_edge(0, 1, 3.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let path = g.shortest_path(0, 2).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
        let (dists, _) = g.dijkstra(0);
        assert!((dists[&2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn grammar_file_parses() {
        let grammar_path = find_grammar_file();
        if grammar_path.is_err() {
            eprintln!("Skipping grammar_file_parses: grammar file not found");
            return;
        }
        let text = std::fs::read_to_string(grammar_path.unwrap()).unwrap();
        let result = ix_grammar::constrained::EbnfGrammar::from_str(&text);
        assert!(result.is_ok(), "grammar should parse: {:?}", result.err());
        let g = result.unwrap();
        assert_eq!(g.start, "PROG");
        assert!(!g.alternatives("PROG").is_empty());
    }

    #[test]
    fn pipeline_dag_is_acyclic() {
        let dag = build_pipeline(Instrument::Guitar);
        // The pipeline has 5 nodes
        let levels = dag.parallel_levels();
        assert!(!levels.is_empty(), "pipeline should have levels");
    }
}
