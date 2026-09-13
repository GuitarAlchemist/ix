//! Slice S1 — the two named digest recipes are executable, disjoint, and
//! reproduce the values §3 declares.
//!
//! Red-first discriminator: `advise` over the 14 approved fixtures must emit a
//! provenance carrying `evidence_aggregate_ix_agg_1 == 217815fc…4f9b` under
//! `IX-AGG-1` and `declared_bundle_aggregate_gaia_agg_1 == e8a7d092…598e` under
//! `GAIA-AGG-1`. These are *different values of different constructions*: a
//! run that builds one recipe and binds it to both field names fails here, and
//! that ambiguity is the defect (B6) this slice closes.
//!
//! Also NC-19 (digest naming): the bare identifier `ordinal_aggregate` appears
//! nowhere in `src/`, and every multi-file record aggregate field name carries
//! its recipe.

mod support;

use ix_gaia_advisory::{advise, AdvisoryRequest, RuleId};

use support::{approved_evidence_root, crate_src_root, digest, MANIFEST_FILE_NAME};

const EVIDENCE_IX_AGG_1: &str = "217815fc65ef08c6d72dbe7211129ee60cc73531f2050c6e59788e1a40324f9b";
const EVIDENCE_GAIA_AGG_1: &str =
    "e8a7d0927b03e9b9379892fb00bdb9ac132c760a39b0218fd7423d5b7122598e";

fn pristine_request() -> AdvisoryRequest {
    AdvisoryRequest {
        evidence_root: approved_evidence_root(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule: RuleId::FullDigest,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    }
}

#[test]
fn both_recipes_reproduce_their_declared_values_and_are_not_the_same_value() {
    // Reproduced independently in the test lane, from §3's text alone.
    let root = approved_evidence_root();
    assert_eq!(
        digest::ix_agg_1(&root),
        EVIDENCE_IX_AGG_1,
        "the Lane-W IX-AGG-1 implementation must reproduce the declared value"
    );
    assert_eq!(
        digest::gaia_agg_1(&root),
        EVIDENCE_GAIA_AGG_1,
        "the Lane-W GAIA-AGG-1 implementation must reproduce the declared value"
    );
    assert_ne!(
        EVIDENCE_IX_AGG_1, EVIDENCE_GAIA_AGG_1,
        "the two recipes are different constructions and must not collapse"
    );

    // And now through the public seam.
    let artifact = advise(&pristine_request()).expect("the approved evidence binds");
    assert_eq!(
        artifact.provenance.evidence_aggregate_ix_agg_1, EVIDENCE_IX_AGG_1,
        "evidence_aggregate_ix_agg_1 must be built with IX-AGG-1"
    );
    assert_eq!(
        artifact.provenance.declared_bundle_aggregate_gaia_agg_1, EVIDENCE_GAIA_AGG_1,
        "declared_bundle_aggregate_gaia_agg_1 must be built with GAIA-AGG-1"
    );
}

#[test]
fn nc_19_no_unnamed_aggregate_identifier_in_crate_source() {
    for (path, text) in crate_source_files() {
        for (index, line) in text.lines().enumerate() {
            for hit in occurrences(line, "ordinal_aggregate") {
                let tail = &line[hit + "ordinal_aggregate".len()..];
                let bare = !tail
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_');
                assert!(
                    !bare,
                    "NC-19: the bare identifier `ordinal_aggregate` appears at {}:{}: {line}",
                    path,
                    index + 1
                );
            }
        }
    }
}

#[test]
fn nc_19_every_emitted_aggregate_key_carries_its_recipe() {
    let artifact = advise(&pristine_request()).expect("the approved evidence binds");
    let json: serde_json::Value =
        serde_json::from_str(&artifact.to_canonical_json()).expect("the artifact is valid JSON");
    let provenance = json
        .get("provenance")
        .and_then(|value| value.as_object())
        .expect("the artifact carries a provenance object");

    let mut seen_aggregate_keys = 0usize;
    for key in provenance.keys() {
        if key.contains("aggregate") {
            seen_aggregate_keys += 1;
            assert!(
                key.ends_with("_ix_agg_1") || key.ends_with("_gaia_agg_1"),
                "NC-19: multi-file record aggregate field {key:?} does not name its recipe"
            );
        }
    }
    assert!(
        seen_aggregate_keys >= 2,
        "NC-19 is vacuous unless the artifact actually carries aggregate fields, saw {seen_aggregate_keys}"
    );
}

fn crate_source_files() -> Vec<(String, String)> {
    let mut out = Vec::new();
    collect(&crate_src_root(), &mut out);
    assert!(!out.is_empty(), "the crate has source files to scan");
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn collect(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
    for entry in std::fs::read_dir(dir).expect("src directory is readable") {
        let entry = entry.expect("directory entry is readable");
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let text = std::fs::read_to_string(&path).expect("source file is UTF-8");
            out.push((path.display().to_string(), text));
        }
    }
}

fn occurrences(haystack: &str, needle: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(found) = haystack[from..].find(needle) {
        out.push(from + found);
        from += found + needle.len();
    }
    out
}
