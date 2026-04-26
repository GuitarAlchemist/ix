//! Phase A smoke test — exercises the real GA CLI, not a fixture.
//!
//! Shells out to `FretboardVoicingsCLI.exe --export --tuning guitar
//! --export-max 50`, writes artifacts under a temp `state/` root, and
//! asserts:
//!
//! - The raw JSONL + compacted corpus + features JSON all landed.
//! - The corpus has at least one row and at most 50 (respecting the cap).
//! - The feature matrix has shape `(rows, FEATURE_COLUMNS.len())`.
//! - Every feature row has the exact column count the schema promises.
//!
//! The test is skipped (not failed) when the GA CLI executable isn't
//! found on disk — CI without the sibling `ga` repo built must still
//! pass `cargo test -p ix-voicings`. Local dev runs the real thing.

use std::fs;

use ix_voicings::{enumerate, featurize, ga_cli_path, FeatureMatrix, Instrument, FEATURE_COLUMNS};

#[test]
fn smoke_phase_a_guitar_50() {
    let cli = ga_cli_path();
    if !cli.exists() {
        eprintln!(
            "SKIP: GA CLI not found at {} (set IX_VOICINGS_GA_CLI or build \
             ga/Demos/Music Theory/FretboardVoicingsCLI in Debug)",
            cli.display()
        );
        return;
    }

    let mut tmp = std::env::temp_dir();
    tmp.push(format!("ix-voicings-smoke-{}", std::process::id()));
    if tmp.exists() {
        let _ = fs::remove_dir_all(&tmp);
    }
    fs::create_dir_all(&tmp).unwrap();

    // SAFETY: `set_var` is legacy-unsafe on 2021-edition; smoke tests in
    // this crate don't run in parallel with anything else that touches
    // IX_VOICINGS_STATE_DIR.
    std::env::set_var("IX_VOICINGS_STATE_DIR", &tmp);

    let enum_out = enumerate(Instrument::Guitar, Some(50))
        .expect("enumerate guitar should succeed with the GA CLI present");
    assert!(enum_out.raw_jsonl.exists(), "raw JSONL missing");
    assert!(enum_out.corpus_json.exists(), "corpus JSON missing");
    assert!(
        enum_out.voicing_count > 0 && enum_out.voicing_count <= 50,
        "expected 1..=50 rows, got {}",
        enum_out.voicing_count
    );

    // Raw JSONL must have the same number of lines as the corpus array.
    let raw = fs::read_to_string(&enum_out.raw_jsonl).unwrap();
    let raw_lines = raw.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(
        raw_lines, enum_out.voicing_count,
        "raw/corpus row count mismatch"
    );

    let feat_out = featurize(Instrument::Guitar).expect("featurize guitar should succeed");
    assert!(feat_out.features_path.exists(), "features file missing");
    assert_eq!(feat_out.voicing_count, enum_out.voicing_count);
    assert_eq!(feat_out.column_count, FEATURE_COLUMNS.len());

    let raw_features = fs::read(&feat_out.features_path).unwrap();
    let matrix: FeatureMatrix = serde_json::from_slice(&raw_features).unwrap();
    assert_eq!(matrix.columns.len(), FEATURE_COLUMNS.len());
    assert_eq!(matrix.rows.len(), enum_out.voicing_count);
    for (i, r) in matrix.rows.iter().enumerate() {
        assert_eq!(
            r.len(),
            FEATURE_COLUMNS.len(),
            "row {i} has {} cols, expected {}",
            r.len(),
            FEATURE_COLUMNS.len()
        );
    }

    // Cleanup (best-effort).
    let _ = fs::remove_dir_all(&tmp);
}
