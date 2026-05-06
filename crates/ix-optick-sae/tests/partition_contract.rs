//! Producer/consumer schema-contract test for the OPTIC-K Phase 1 partition list.
//!
//! Background: PR #82 (GA) shipped an SAE artifact whose `partitions_used` was
//! missing the ROOT partition. ix #29 then nearly repeated the same omission
//! in the Rust producer. Same bug class, two occurrences in one week. The
//! prevention prose lives at
//! `ga/docs/solutions/integration-issues/optick-sae-phase1-partition-and-python-bin-2026-05-05.md` —
//! this file is the actual mechanical guard.
//!
//! When this test fails, treat it as a one-way-door alarm: the producer's
//! `PHASE1_PARTITIONS` constant has drifted from the canonical baseline. Either
//! the constant is wrong (most common — fix the constant) or the canonical
//! baseline legitimately changed (rare — coordinate the change with GA's
//! `state/quality/optick-sae/<date>/optick-sae-artifact.json` and refresh the
//! `tests/fixtures/canonical-partitions.json` snapshot in the SAME PR).
//!
//! The fixture is a deliberately tiny mirror of GA's canonical artifact, not
//! the whole thing — the test only needs the partition list and the compact
//! training dim to fail loud on drift. Vendoring is preferred over fetching
//! from GA at test time so this test runs offline and on every CI runner.

use ix_optick_sae::PHASE1_PARTITIONS;
use serde::Deserialize;

/// Slimmed mirror of the consumer-side artifact contract.
#[derive(Deserialize)]
struct CanonicalSnapshot {
    partitions_used: Vec<String>,
    compact_training_dim: u32,
}

const CANONICAL_FIXTURE: &str = include_str!("fixtures/canonical-partitions.json");

#[test]
fn phase1_partitions_match_canonical_baseline() {
    let canonical: CanonicalSnapshot = serde_json::from_str(CANONICAL_FIXTURE)
        .expect("canonical-partitions.json failed to parse — the fixture is corrupt");

    let producer: Vec<String> = PHASE1_PARTITIONS.iter().map(|s| s.to_string()).collect();

    assert_eq!(
        producer, canonical.partitions_used,
        "\n\
         PHASE1_PARTITIONS in src/lib.rs has drifted from the canonical baseline.\n\
         \n\
           producer (src/lib.rs):  {:?}\n\
           canonical (fixture):    {:?}\n\
         \n\
         If the producer is wrong, fix the constant.\n\
         If the canonical baseline legitimately changed (rare — re-indexing cost is high),\n\
         coordinate the change with GA's state/quality/optick-sae/<date>/optick-sae-artifact.json\n\
         AND refresh tests/fixtures/canonical-partitions.json in the same PR.\n\
         \n\
         This drift is the bug class PR #82 (GA) and ix #29 hit (ROOT missed twice).\n",
        producer, canonical.partitions_used,
    );
}

#[test]
fn compact_training_dim_matches_canonical_baseline() {
    let canonical: CanonicalSnapshot = serde_json::from_str(CANONICAL_FIXTURE)
        .expect("canonical-partitions.json failed to parse");

    // The OPTIC-K v1.8 partition widths (from
    // ga/Common/GA.Business.ML/Embeddings/EmbeddingSchema.cs SimilarityPartitions):
    //   STRUCTURE   = 24
    //   MORPHOLOGY  = 24
    //   CONTEXT     = 12
    //   SYMBOLIC    = 12
    //   MODAL       = 40
    //   ROOT        = 12
    // Sum = 124.
    //
    // We don't have the widths as constants in this crate yet (they live in the
    // Python trainer + GA C# schema). Hardcoding the canonical sum here and
    // asserting against the fixture catches a drift in either direction —
    // partition list shrinks/grows OR the canonical fixture forgets to update.
    const EXPECTED_COMPACT_DIM: u32 = 124;
    assert_eq!(
        canonical.compact_training_dim, EXPECTED_COMPACT_DIM,
        "Canonical fixture's compact_training_dim ({}) drifted from the OPTIC-K v1.8 \
         similarity-partition sum ({}). If partition widths legitimately changed, update \
         this constant AND ga's EmbeddingSchema.cs in the same PR.",
        canonical.compact_training_dim, EXPECTED_COMPACT_DIM,
    );
}
