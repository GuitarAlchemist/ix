//! Cross-repo conformance test for OPTIC-K v4.
//!
//! Verifies that `ix-optick` can correctly open and query a real OPTK v4 index
//! produced by GA's `OptickIndexWriter`. The test is gated on the
//! `OPTICK_INDEX_PATH` environment variable — if not set (CI, fresh clone),
//! all tests silently skip.
//!
//! To run locally after regenerating the index via the GA CLI:
//!
//! ```bash
//! cd ga && dotnet run --project "Demos/Music Theory/FretboardVoicingsCLI" \
//!   -- --export-embeddings --output state/voicings/optick.index
//! cd ../ix && \
//!   OPTICK_INDEX_PATH=".../ga/state/voicings/optick.index" \
//!   cargo test -p ix-optick --test conformance -- --nocapture
//! ```

use ix_optick::OptickIndex;
use std::path::PathBuf;

fn index_path() -> Option<PathBuf> {
    std::env::var("OPTICK_INDEX_PATH").ok().map(PathBuf::from)
}

#[test]
fn conformance_open_real_index() {
    let Some(path) = index_path() else {
        eprintln!("skipping: OPTICK_INDEX_PATH not set");
        return;
    };
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return;
    }

    let index = OptickIndex::open(&path).expect("open real index");
    eprintln!(
        "opened: count={} dim={} version={}",
        index.count(),
        index.dimension(),
        index.header().version
    );

    assert_eq!(index.header().version, 4);
    assert_eq!(index.dimension(), 112, "v4 dimension must be 112");
    assert!(index.count() > 0, "index must contain at least one voicing");
    assert_eq!(index.header().instruments, 3);
}

#[test]
fn conformance_instrument_counts_sum_to_total() {
    let Some(path) = index_path() else {
        return;
    };
    if !path.exists() {
        return;
    }

    let index = OptickIndex::open(&path).unwrap();
    let total: u64 = index
        .header()
        .instrument_slices
        .iter()
        .map(|s| s.count)
        .sum();
    assert_eq!(
        total,
        index.count(),
        "instrument slice counts must sum to total count"
    );
}

#[test]
fn conformance_search_per_instrument() {
    let Some(path) = index_path() else {
        return;
    };
    if !path.exists() {
        return;
    }

    let index = OptickIndex::open(&path).unwrap();
    let dim = index.dimension() as usize;

    // A query aligned with dim 0 (STRUCTURE partition, first pitch-class chroma slot).
    let mut query = vec![0.0f32; dim];
    query[0] = 1.0;

    for inst in &["guitar", "bass", "ukulele"] {
        let idx = match *inst {
            "guitar" => 0,
            "bass" => 1,
            "ukulele" => 2,
            _ => unreachable!(),
        };
        let slice = &index.header().instrument_slices[idx];
        if slice.count == 0 {
            continue;
        }

        let results = index.search(&query, Some(inst), 3).expect("search");
        assert!(!results.is_empty(), "{inst} search returned no results");
        for r in &results {
            assert_eq!(
                r.metadata.instrument, *inst,
                "instrument filter leak: got '{}' in {inst} search",
                r.metadata.instrument
            );
        }
        eprintln!(
            "{inst}: top-1 score={:.4} diagram={}",
            results[0].score, results[0].metadata.diagram
        );
    }
}

#[test]
fn conformance_metadata_random_access() {
    // v4's metadata offset table allows O(1) fetch for any index.
    let Some(path) = index_path() else {
        return;
    };
    if !path.exists() {
        return;
    }

    let index = OptickIndex::open(&path).unwrap();
    let dim = index.dimension() as usize;

    let mut query = vec![0.0f32; dim];
    query[0] = 1.0;
    let results = index.search(&query, Some("guitar"), 1).expect("search");
    assert_eq!(results.len(), 1);
    assert!(
        !results[0].metadata.diagram.is_empty(),
        "metadata must include diagram"
    );
    assert!(
        !results[0].metadata.midi_notes.is_empty(),
        "metadata must include midi_notes"
    );
}
