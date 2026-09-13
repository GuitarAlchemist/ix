//! Public-behaviour tests for the v0a census tracer.
//!
//! Every test here calls exactly one function, `ix_gaia_census::census`. The
//! crate's modules are private, so no test can reach an implementation detail.
//!
//! Every expected digest below is declared by the approved evidence itself —
//! `gaia-s1-r2-bundle-manifest.md` §3 and the S1 R18 review directory name —
//! and is never recomputed by test logic.

use std::path::PathBuf;

use ix_gaia_census::{census, CensusRequest, Hexavalent};

/// Manifest §3, the declared fixed point over the thirteen listed files.
const DECLARED_AGGREGATE: &str =
    "970bb50c3f40aac8da50f5822fc8e6ad48f7e27643dd67176591a7788f0b311c";

/// The fourteen-file ordinal aggregate that names the S1 R18 review subject
/// (`gaia-s1-r18-review-e8a7d092-...`), manifest included.
const SUBJECT_ORDINAL_AGGREGATE: &str =
    "e8a7d0927b03e9b9379892fb00bdb9ac132c760a39b0218fd7423d5b7122598e";

const MANIFEST: &str = "gaia-s1-r2-bundle-manifest.md";

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gaia-s1-r18")
}

/// Every object key in a serialized artifact, at any depth.
fn object_keys(json: &str) -> Vec<String> {
    fn walk(value: &serde_json::Value, out: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    out.push(key.clone());
                    walk(child, out);
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(|item| walk(item, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(&serde_json::from_str(json).expect("artifact is valid JSON"), &mut out);
    out
}

fn fixture_request() -> CensusRequest {
    CensusRequest {
        evidence_root: fixture_root(),
        manifest_file_name: MANIFEST.to_string(),
    }
}

#[test]
fn copied_evidence_reproduces_its_declared_deterministic_census() {
    let artifact = census(&fixture_request()).expect("the pristine fixture must not refuse");

    assert_eq!(
        artifact.aggregate_digest_measured, DECLARED_AGGREGATE,
        "measured aggregate must reproduce the digest declared in manifest §3"
    );
    assert_eq!(
        artifact.subject.ordinal_aggregate_measured, SUBJECT_ORDINAL_AGGREGATE,
        "the fourteen-file ordinal aggregate must reproduce the S1 R18 subject identity"
    );
}

#[test]
fn the_declared_construction_is_the_only_one_that_reproduces_the_fixed_point() {
    // Manifest §3 declares three wrong constructions alongside the right one.
    // None of them was authored here; reproducing the declared value while
    // differing from all three is what distinguishes construction from
    // coincidence, and it is what makes this gate able to fail.
    const TRAILING_NEWLINE: &str =
        "3c53ad575c81f7c0220a112eacd359810949a0b71e4b6d9e4ff74717883d8d74";
    const CRLF_JOINED: &str = "457e745ba91e21f93330296bfbf9d5715d3b665a99178e66cd0fe3338dfc8bdb";
    const CONCATENATED: &str = "ea3b98f1d47f72b2282fba0b066054a3b914a2f727e78899573f6205b36223e6";

    let artifact = census(&fixture_request()).expect("pristine fixture");

    assert_eq!(artifact.aggregate_digest_measured, DECLARED_AGGREGATE);
    for wrong in [TRAILING_NEWLINE, CRLF_JOINED, CONCATENATED] {
        assert_ne!(
            artifact.aggregate_digest_measured, wrong,
            "the measured aggregate must not match a declared wrong construction"
        );
    }
}

#[test]
fn the_declared_totals_are_reported() {
    let artifact = census(&fixture_request()).expect("pristine fixture");

    // Manifest §2: thirteen files, 639,580 bytes, and no fourteenth entry
    // besides the manifest itself.
    assert_eq!(artifact.totals.listed_files, 13);
    assert_eq!(artifact.totals.listed_bytes, 639_580);
    assert_eq!(artifact.totals.measured_bytes, 639_580);
    assert_eq!(artifact.totals.unlisted_files, 0);
    assert!(artifact.totals.unlisted_names.is_empty());
    assert_eq!(artifact.totals.non_file_entries, 0);
    assert!(artifact.totals.non_file_names.is_empty());
    assert_eq!(artifact.agreement, Hexavalent::True);
}

#[test]
fn the_crlf_file_hashes_as_the_raw_bytes_it_holds() {
    // Manifest §6: one file uses CRLF — 214 CRLF pairs, no bare LF. A text-mode
    // read would normalise those bytes away and produce a different digest.
    let artifact = census(&fixture_request()).expect("pristine fixture");

    let row = artifact
        .rows
        .iter()
        .find(|row| row.name == "gaia-s1-r2-validator-output.txt")
        .expect("the CRLF file is declared");
    assert_eq!(row.state, Hexavalent::True);
    assert_eq!(row.measured_bytes, 23_567);
    assert_eq!(
        row.measured_sha256,
        "6e0f9f7338bf724a43d775d78cbf75f99d67c4c2ad922dbf7f59b3d0f75462a3"
    );
}

#[test]
fn the_artifact_carries_no_score_verdict_or_authority_field() {
    let json = census(&fixture_request())
        .expect("pristine fixture")
        .to_canonical_json();

    for forbidden in [
        "score",
        "confidence",
        "weight",
        "rank",
        "health",
        "trust",
        "verdict",
        "approval",
        "acceptance",
        "fresh",
        "safety",
        "authority",
        "quality",
        "risk",
    ] {
        for key in object_keys(&json) {
            assert!(
                !key.contains(forbidden),
                "the artifact must carry no {forbidden:?} field, found key {key:?}"
            );
        }
    }

    let artifact = census(&fixture_request()).expect("pristine fixture");
    for state in artifact
        .rows
        .iter()
        .map(|row| row.state)
        .chain(std::iter::once(artifact.agreement))
    {
        assert!(
            !matches!(state, Hexavalent::Probable | Hexavalent::Doubtful),
            "a byte comparison identifies no evidential gradient, found {state:?}"
        );
    }
}

#[test]
fn every_reported_state_is_attributable() {
    let artifact = census(&fixture_request()).expect("pristine fixture");

    assert_eq!(artifact.subject.manifest_file_name, MANIFEST);
    assert_eq!(artifact.subject.manifest_sha256.len(), 64);
    for row in &artifact.rows {
        assert!(!row.name.is_empty(), "every state names its file");
        assert_eq!(row.measured_sha256.len(), 64, "every state names its bytes");
        assert_eq!(row.declared_sha256.len(), 64);
    }
}

#[test]
fn a_run_leaves_the_evidence_root_unchanged() {
    let before: Vec<String> = listing(&fixture_root());
    let first = census(&fixture_request()).expect("pristine fixture");
    let after: Vec<String> = listing(&fixture_root());
    let second = census(&fixture_request()).expect("pristine fixture");

    assert_eq!(before, after, "a run writes nothing into the evidence root");
    assert_eq!(before.len(), 14, "fourteen files, no fifteenth entry");
    assert_eq!(
        first.subject.ordinal_aggregate_measured, SUBJECT_ORDINAL_AGGREGATE,
        "the subject identity holds before the run"
    );
    assert_eq!(
        second.subject.ordinal_aggregate_measured, SUBJECT_ORDINAL_AGGREGATE,
        "and after it"
    );
}

/// The sorted `name:len` listing of a directory.
fn listing(root: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .expect("evidence root")
        .map(|entry| {
            let entry = entry.expect("entry");
            format!(
                "{}:{}",
                entry.file_name().to_string_lossy(),
                entry.metadata().expect("metadata").len()
            )
        })
        .collect();
    names.sort();
    names
}

#[test]
fn census_serializes_deterministically_and_carries_no_environment() {
    let first = census(&fixture_request())
        .expect("first run")
        .to_canonical_json();
    let second = census(&fixture_request())
        .expect("second run")
        .to_canonical_json();

    assert!(
        first.contains(DECLARED_AGGREGATE),
        "the serialization must carry the measured aggregate; got {first:?}"
    );
    assert_eq!(first, second, "two runs must serialize byte for byte alike");

    for key in object_keys(&first) {
        for clock_shaped in ["time", "date", "stamp", "elapsed", "generated", "now", "duration"] {
            assert!(
                !key.contains(clock_shaped),
                "the artifact must carry no clock-shaped key, found {key:?}"
            );
        }
    }
    for path_shaped in [":\\", "/home/", "/Users/", "fixtures"] {
        assert!(
            !first.contains(path_shaped),
            "the artifact must carry no path fragment, found {path_shaped:?}"
        );
    }
}
