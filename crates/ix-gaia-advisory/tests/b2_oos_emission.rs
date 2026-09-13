//! Blocker B2 — the out-of-sample emission path, at the public seam.
//!
//! The crate has exactly two public entry points, `advise` and `evaluate`, and
//! this binary drives both. What it asserts is deliberately one-sided: that the
//! repair which makes `OutOfSampleStatus::Measured` *constructible* opens no
//! way to *reach* it from outside the crate.
//!
//! Class `Holdout` remains publicly refused. It must stay refused until an
//! independently approved S12/S13 preregistration supplies a real admitted
//! population and ledger, so the earned construction path prepared in
//! `src/lib.rs` is exercised only by the colocated boundary tests one layer in.
//! That asymmetry is the point of this file, not a gap in it: **the seam is the
//! thing being held shut**, and a test that opened it to prove it opens would
//! be the defect rather than the control.
//!
//! Controls closed here: NC-20b at the seam, NC-25 at the seam, and the
//! fail-closed refusal of both reserved classes.

mod support;

use ix_gaia_advisory::{
    advise, evaluate, AdvisoryRefusal, AdvisoryRequest, EvalRequest, EvidenceClass, RuleId,
    RuleWidth,
};

use support::corpus::{self, Family, Options};
use support::{approved_evidence_root, fresh_scratch, MANIFEST_FILE_NAME};

/// The exact `out_of_sample` object every artifact M3-as-selected can emit
/// carries. Pinned as bytes: the B2 repair changed which *conditions* produce
/// it, and must not have changed the value itself.
const FROZEN_OOS_JSON: &str = r#"{"status":"Unknown","reason":"NoEligibleUnseenPopulation","also_applicable":["SyntheticOrExposedPopulationOnly","NoRealLabelPopulation","NoModelFitted"]}"#;

fn advisory_request() -> AdvisoryRequest {
    AdvisoryRequest {
        evidence_root: approved_evidence_root(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule: RuleId::FullDigest,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    }
}

// ------------------------------------------------------------------- NC-20b

#[test]
fn nc_20b_no_advise_path_reaches_measured() {
    // Every rule, and every swept width on the one rule that takes one.
    let mut requests = Vec::new();
    for rule in RuleId::all() {
        if rule == RuleId::WindowProbe {
            for window_bytes in [0, 1024, 4096] {
                requests.push(AdvisoryRequest {
                    rule,
                    window_bytes,
                    window_reference_root: Some(approved_evidence_root()),
                    ..advisory_request()
                });
            }
        } else {
            requests.push(AdvisoryRequest {
                rule,
                ..advisory_request()
            });
        }
    }

    for request in requests {
        let json = advise(&request)
            .expect("the approved evidence binds")
            .to_canonical_json();
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(
            value["out_of_sample"]["status"], "Unknown",
            "NC-20b: {:?}/w={} reached a non-Unknown status",
            request.rule, request.window_bytes
        );
        // Asserted over the emitted bytes, not over a re-serialization: a
        // round trip through `serde_json::Value` reorders object keys, and the
        // whole point of `to_canonical_json` is that the byte sequence is a
        // function of the measured evidence alone.
        assert!(
            json.contains(&format!(r#""out_of_sample":{FROZEN_OOS_JSON}"#)),
            "NC-25: the emitted clause moved for {:?}/w={}; got {json}",
            request.rule,
            request.window_bytes
        );
    }
}

#[test]
fn nc_20b_a_development_evaluation_reaches_no_measured_anywhere_in_its_report() {
    let out = fresh_scratch("b2-development");
    let corpus = corpus::generate(
        &out.join("corpus"),
        &Options {
            families: vec![Family::Restamp, Family::Pristine],
            inject_duplicate: false,
        },
    );

    let report = evaluate(&EvalRequest {
        corpus_root: corpus.root.clone(),
        expected_corpus_digest_ix_agg_1: corpus.digest.clone(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        cells: vec![
            RuleWidth {
                rule: RuleId::StructuralOnly,
                window_bytes: 0,
            },
            RuleWidth {
                rule: RuleId::WindowProbe,
                window_bytes: 1024,
            },
        ],
        window_reference_root: approved_evidence_root(),
        evidence_class: EvidenceClass::Development,
    })
    .expect("the Class-Development corpus binds")
    .to_canonical_json();

    assert!(
        !report.contains("Measured"),
        "NC-20b: a Class-Development report must carry no Measured status anywhere"
    );
    assert!(
        report.contains(r#""evidence_class":"Development""#),
        "the report is class-stamped, so the absence above is about Development"
    );
}

// -------------------------------------------- the two reserved classes, closed

#[test]
fn holdout_fails_closed_at_the_public_seam_before_any_byte_is_read() {
    // The corpus root does not exist. A run that refused on the *class* returns
    // `ReservedEvidenceClass`; a run that had opened the holdout path first
    // would return `EvidenceRootUnreadable` or `CorpusDigestMismatch` instead.
    // So this asserts the ordering, not merely the outcome: **no path under a
    // Class-Holdout corpus is opened, because the refusal precedes every read.**
    let absent = fresh_scratch("b2-holdout-closed").join("no-such-corpus");
    assert!(
        !absent.exists(),
        "the control needs a root that is not there"
    );

    let refusal = evaluate(&EvalRequest {
        corpus_root: absent,
        expected_corpus_digest_ix_agg_1: "0".repeat(64),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        cells: vec![RuleWidth {
            rule: RuleId::StructuralOnly,
            window_bytes: 0,
        }],
        window_reference_root: approved_evidence_root(),
        evidence_class: EvidenceClass::Holdout,
    })
    .expect_err("Class-Holdout is refused, and a refusal produces no report");

    match refusal {
        AdvisoryRefusal::ReservedEvidenceClass { class, .. } => {
            assert_eq!(class, EvidenceClass::Holdout);
        }
        other => {
            panic!("Class-Holdout must be refused on its class, before any read; got {other:?}")
        }
    }
}

#[test]
fn selection_fails_closed_at_the_public_seam_before_any_byte_is_read() {
    let absent = fresh_scratch("b2-selection-closed").join("no-such-corpus");

    let refusal = evaluate(&EvalRequest {
        corpus_root: absent,
        expected_corpus_digest_ix_agg_1: "0".repeat(64),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        cells: vec![RuleWidth {
            rule: RuleId::StructuralOnly,
            window_bytes: 0,
        }],
        window_reference_root: approved_evidence_root(),
        evidence_class: EvidenceClass::Selection,
    })
    .expect_err("Class-Selection is refused, and a refusal produces no report");

    match refusal {
        AdvisoryRefusal::ReservedEvidenceClass { class, .. } => {
            assert_eq!(class, EvidenceClass::Selection);
        }
        other => panic!("Class-Selection must be refused on its class; got {other:?}"),
    }
}

#[test]
fn the_public_input_surface_carries_no_field_that_could_admit_a_holdout() {
    // The seam is the whole attack surface: `AdvisoryRequest` and `EvalRequest`
    // are the only values a caller supplies. Neither carries a ledger receipt,
    // a corpus digest that reaches the out-of-sample gate, or a report digest.
    // `EvalRequest::expected_corpus_digest_ix_agg_1` is a *refusal* threshold —
    // the run refuses unless the corpus re-measures to it — and never a value
    // the emission path consumes; this pins that it cannot become one silently.
    let out = fresh_scratch("b2-surface");
    let corpus = corpus::generate(&out.join("corpus"), &Options::family(Family::Pristine));

    let mismatch = evaluate(&EvalRequest {
        corpus_root: corpus.root.clone(),
        expected_corpus_digest_ix_agg_1: "f".repeat(64),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        cells: vec![RuleWidth {
            rule: RuleId::StructuralOnly,
            window_bytes: 0,
        }],
        window_reference_root: approved_evidence_root(),
        evidence_class: EvidenceClass::Development,
    })
    .expect_err("a caller-declared digest that does not re-measure is a refusal");

    match mismatch {
        AdvisoryRefusal::CorpusDigestMismatch { expected, measured } => {
            assert_eq!(expected, "f".repeat(64));
            assert_eq!(
                measured, corpus.digest,
                "the run reports what it measured, never what it was told"
            );
        }
        other => panic!("a declared-versus-measured corpus digest gap is a refusal; got {other:?}"),
    }
}
