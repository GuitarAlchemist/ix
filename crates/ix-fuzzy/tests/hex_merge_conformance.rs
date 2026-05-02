//! Hex-merge conformance test suite — IX-side runner against the
//! canonical Demerzel fixture corpus.
//!
//! Loads JSON fixtures from `Demerzel/fixtures/hex-merge/` (reached
//! here through the existing `governance/demerzel` submodule) and
//! asserts that `ix_fuzzy::observations::merge` produces output
//! matching the embedded expected block. When both this test and
//! the analogous Hari-side test
//! (`hari/crates/hari-lattice/tests/hex_merge_conformance.rs`) pass
//! against the same corpus, byte-equivalence between the two
//! implementations is proven by transitivity.
//!
//! See `Demerzel/fixtures/hex-merge/README.md` for the fixture
//! schema and the canonical-vs-mirror policy.
//! `ix-types::Hexavalent` already uses the single-letter wire format
//! (`"T"`/`"P"`/`"U"`/`"D"`/`"F"`/`"C"`) so the fixtures load
//! directly without an adapter.

use ix_fuzzy::observations::{merge, HexObservation};
use ix_types::Hexavalent;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    #[serde(default)]
    description: String,
    input: Input,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Input {
    observations: Vec<WireObs>,
    #[serde(default)]
    current_round: Option<u32>,
    #[serde(default)]
    staleness_k: Option<u32>,
}

/// Wire-only mirror of [`HexObservation`] so the fixture JSON is
/// directly deserializable. ix-fuzzy's own struct intentionally
/// carries no serde derives (the SessionEvent layer owns wire
/// concerns); this struct exists purely to bridge the test fixture
/// JSON into a `HexObservation` value.
#[derive(Debug, Clone, Deserialize)]
struct WireObs {
    source: String,
    diagnosis_id: String,
    round: u32,
    ordinal: u32,
    claim_key: String,
    variant: Hexavalent,
    weight: f64,
    #[serde(default)]
    evidence: Option<String>,
}

impl From<WireObs> for HexObservation {
    fn from(w: WireObs) -> Self {
        Self {
            source: w.source,
            diagnosis_id: w.diagnosis_id,
            round: w.round,
            ordinal: w.ordinal,
            claim_key: w.claim_key,
            variant: w.variant,
            weight: w.weight,
            evidence: w.evidence,
        }
    }
}

#[derive(Debug, Deserialize)]
struct Expected {
    observations_count: usize,
    contradictions_count: usize,
    #[serde(default)]
    contradictions: Vec<ExpectedContradiction>,
    distribution: ExpectedDistribution,
    escalation_triggered: bool,
}

#[derive(Debug, Deserialize)]
struct ExpectedContradiction {
    source: String,
    claim_key: String,
    variant: Hexavalent,
    weight: f64,
    /// Optional — when present, pins the synthesized observation's
    /// content-derived `diagnosis_id` exactly. This is the strongest
    /// byte-equal claim.
    #[serde(default)]
    diagnosis_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct ExpectedDistribution {
    T: f64,
    P: f64,
    U: f64,
    D: f64,
    F: f64,
    C: f64,
}

const TOL: f64 = 1e-9;

fn fixture_dir() -> PathBuf {
    // Canonical home is `Demerzel/fixtures/hex-merge/`, reached here
    // via the existing `governance/demerzel` submodule. From the
    // crate manifest at `crates/ix-fuzzy/`, that's two parents up
    // to the workspace root, then into the submodule.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set in tests");
    Path::new(&manifest)
        .ancestors()
        .nth(2)
        .expect("workspace root above crate")
        .join("governance")
        .join("demerzel")
        .join("fixtures")
        .join("hex-merge")
}

fn load_fixtures() -> Vec<(PathBuf, Fixture)> {
    let dir = fixture_dir();
    assert!(
        dir.is_dir(),
        "expected fixture directory at {}",
        dir.display()
    );
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read fixture dir") {
        let path = entry.expect("dirent").path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let fixture: Fixture =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        out.push((path, fixture));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(
        !out.is_empty(),
        "no fixtures found in {} — at least one is required",
        dir.display()
    );
    out
}

#[test]
fn every_fixture_matches_expected() {
    for (path, fixture) in load_fixtures() {
        check(&path, &fixture);
    }
}

fn check(path: &Path, fx: &Fixture) {
    let observations: Vec<HexObservation> = fx
        .input
        .observations
        .iter()
        .cloned()
        .map(HexObservation::from)
        .collect();

    let state = merge(&observations, fx.input.current_round, fx.input.staleness_k)
        .expect("merge returned error on conformance fixture");

    let label = if fx.description.is_empty() {
        format!("{} ({})", fx.name, path.display())
    } else {
        format!("{} — {} ({})", fx.name, fx.description, path.display())
    };

    assert_eq!(
        state.observations.len(),
        fx.expected.observations_count,
        "{label}: observations_count mismatch (got {:#?})",
        state.observations
    );
    assert_eq!(
        state.contradictions.len(),
        fx.expected.contradictions_count,
        "{label}: contradictions_count mismatch (got {:#?})",
        state.contradictions
    );

    for expected in &fx.expected.contradictions {
        let actual = state
            .contradictions
            .iter()
            .find(|c| {
                c.source == expected.source
                    && c.claim_key == expected.claim_key
                    && c.variant == expected.variant
            })
            .unwrap_or_else(|| {
                panic!(
                    "{label}: no contradiction matching {{source={:?}, claim_key={:?}, variant={:?}}}; got {:#?}",
                    expected.source, expected.claim_key, expected.variant, state.contradictions
                )
            });
        assert!(
            (actual.weight - expected.weight).abs() < TOL,
            "{label}: weight mismatch on {} (expected {}, got {})",
            expected.claim_key,
            expected.weight,
            actual.weight
        );
        if let Some(want_id) = &expected.diagnosis_id {
            assert_eq!(
                &actual.diagnosis_id, want_id,
                "{label}: diagnosis_id mismatch on {}",
                expected.claim_key
            );
        }
    }

    let dist = &state.distribution;
    let pairs = [
        (Hexavalent::True, fx.expected.distribution.T, "T"),
        (Hexavalent::Probable, fx.expected.distribution.P, "P"),
        (Hexavalent::Unknown, fx.expected.distribution.U, "U"),
        (Hexavalent::Doubtful, fx.expected.distribution.D, "D"),
        (Hexavalent::False, fx.expected.distribution.F, "F"),
        (Hexavalent::Contradictory, fx.expected.distribution.C, "C"),
    ];
    for (variant, expected_mass, sym) in pairs {
        let actual = dist.get(&variant);
        assert!(
            (actual - expected_mass).abs() < TOL,
            "{label}: distribution[{sym}] mismatch (expected {expected_mass}, got {actual})"
        );
    }

    assert_eq!(
        ix_fuzzy::hexavalent::escalation_triggered(dist),
        fx.expected.escalation_triggered,
        "{label}: escalation_triggered mismatch"
    );
}
