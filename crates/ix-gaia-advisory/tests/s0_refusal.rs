//! Slice S0 — the crate exists and refuses everything it cannot bind.
//!
//! Red-first discriminator: `advise` over a nonexistent evidence root must
//! return `EvidenceRootUnreadable` and construct no artifact. A stub that
//! returns an artifact regardless of whether the root can be read fails here,
//! which is the whole point: "refuse before emit" is not "emit a degraded
//! artifact".
//!
//! **NC-12** lives here in full: an unreadable root, a declared file the binder
//! cannot resolve, and a declared name that is not a plain file name each
//! produce `Err` and **no artifact on any path**.

mod support;

use std::path::PathBuf;

use ix_gaia_advisory::{advise, AdvisoryRefusal, AdvisoryRequest, RuleId};

use support::{fresh_scratch, stage_pristine, MANIFEST_FILE_NAME};

fn request(root: PathBuf) -> AdvisoryRequest {
    AdvisoryRequest {
        evidence_root: root,
        manifest_file_name: "gaia-s1-r2-bundle-manifest.md".to_string(),
        rule: RuleId::FullDigest,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    }
}

#[test]
fn nonexistent_evidence_root_refuses_and_emits_no_artifact() {
    let root = std::env::temp_dir().join("ix-gaia-m3-s0-root-that-does-not-exist");
    assert!(
        !root.exists(),
        "the test's own premise is broken: {} exists",
        root.display()
    );

    let outcome = advise(&request(root.clone()));

    match outcome {
        Err(AdvisoryRefusal::EvidenceRootUnreadable { detail }) => {
            assert!(
                detail.contains("ix-gaia-m3-s0-root-that-does-not-exist"),
                "the refusal must name the root it could not read, got {detail:?}"
            );
        }
        Err(other) => panic!("expected EvidenceRootUnreadable, got {other:?}"),
        Ok(artifact) => panic!(
            "refuse-before-emit violated: an artifact was emitted over an unreadable root: {artifact:?}"
        ),
    }
}

#[test]
fn nc_12_a_declared_manifest_the_root_does_not_hold_is_a_refusal() {
    let root = fresh_scratch("s0-no-manifest").join("root");
    stage_pristine(&root);
    std::fs::remove_file(root.join(MANIFEST_FILE_NAME)).expect("removable");

    match advise(&request(root)) {
        Err(AdvisoryRefusal::BinderIncomplete { field, .. }) => {
            assert_eq!(field, MANIFEST_FILE_NAME);
        }
        Err(other) => panic!("expected BinderIncomplete, got {other:?}"),
        Ok(artifact) => panic!(
            "the binder cannot resolve the one field the request declares, yet emitted {artifact:?}"
        ),
    }
}

#[test]
fn nc_12_a_declared_name_that_is_not_a_plain_file_name_is_a_refusal() {
    let root = fresh_scratch("s0-unsafe-name").join("root");
    stage_pristine(&root);
    let manifest = root.join(MANIFEST_FILE_NAME);
    let text = std::fs::read_to_string(&manifest).expect("readable");
    // A declared name carrying a path segment would read outside the root. It
    // is refused before any read is attempted, not sanitised and followed.
    let poisoned = text.replace(
        "| `gaia-uncertainty-grammar-v0.1.ebnf` |",
        "| `../gaia-uncertainty-grammar-v0.1.ebnf` |",
    );
    assert_ne!(
        poisoned, text,
        "the test's own premise: the row was rewritten"
    );
    std::fs::write(&manifest, poisoned).expect("writable");

    match advise(&request(root)) {
        Err(AdvisoryRefusal::UnsafeManifestName { name }) => {
            assert!(
                name.contains(".."),
                "the refusal names the offending name: {name}"
            );
        }
        Err(other) => panic!("expected UnsafeManifestName, got {other:?}"),
        Ok(artifact) => panic!("a traversing declared name was accepted: {artifact:?}"),
    }
}
