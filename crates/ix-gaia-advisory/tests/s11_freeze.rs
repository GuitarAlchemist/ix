//! Slice S11 — the freeze. **Lane W stops here.**
//!
//! Every value below is written *before* any Class-`Holdout` byte exists,
//! because a preregistration whose numbers are filled in after the data arrives
//! is not a preregistration. No path under a Class-`Holdout` corpus is opened by
//! this suite, and none exists to open.
//!
//! NC-13 closes the loop: after the whole suite has run, both fixed points must
//! still re-measure to the values §3 declares, under **both** named recipes.
//! The generator writes only into a scratch directory outside the repository; if
//! it had ever written into the approved evidence, this is where that shows.

mod support;

use ix_gaia_advisory::{advise, AdvisoryRequest, RuleId, SUBJECT_AGGREGATE_IX_AGG_1};

use support::corpus::{self, Family, Options};
use support::digest as lane_digest;
use support::{approved_evidence_root, fresh_scratch, scratch_root, MANIFEST_FILE_NAME};

const EVIDENCE_IX_AGG_1: &str = "217815fc65ef08c6d72dbe7211129ee60cc73531f2050c6e59788e1a40324f9b";
const EVIDENCE_GAIA_AGG_1: &str =
    "e8a7d0927b03e9b9379892fb00bdb9ac132c760a39b0218fd7423d5b7122598e";
const SUBJECT_IX_AGG_1: &str = "a3e6895759d007c45d7311c9b05387befdf22212cc64671c6a94cd731dc4c235";
/// `IX-AGG-1` over the reviewed M2 crate tree — the 23 files of the reviewed
/// subject that live inside this repository. The remaining three of the 26 are
/// the workspace manifest, the maturity table and the M2 plan; the first two
/// legitimately carry this crate's single registration line each, so the crate
/// tree is what can be pinned here. The full 26-file fixed point is re-measured
/// out of band against the immutable snapshot.
const CENSUS_TREE_IX_AGG_1: &str =
    "cf4c1b0e32047743f8dfe278689369d7ff8ef660be1a5c42cc701df122fac16b";

const SWEPT_WIDTHS: [u32; 3] = [0, 1024, 4096];

fn window_reference_digest(window_bytes: u32) -> String {
    advise(&AdvisoryRequest {
        evidence_root: approved_evidence_root(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule: RuleId::WindowProbe,
        window_bytes,
        window_reference_root: Some(approved_evidence_root()),
        expected_provenance: None,
    })
    .expect("the approved evidence binds")
    .provenance
    .window_reference_digest
}

fn rule_source_digest() -> String {
    advise(&AdvisoryRequest {
        evidence_root: approved_evidence_root(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule: RuleId::FullDigest,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    })
    .expect("binds")
    .provenance
    .rule_source_digest
}

/// `candidate_fingerprint = sha256(rule_source_digest ‖ NUL ‖ "window_bytes=<w>")`
fn candidate_fingerprint(rule_source_digest: &str, window_bytes: u32) -> String {
    let mut input = Vec::new();
    input.extend_from_slice(rule_source_digest.as_bytes());
    input.push(0);
    input.extend_from_slice(format!("window_bytes={window_bytes}").as_bytes());
    lane_digest::sha256_hex(&input)
}

// ---------------------------------------------------------------- NC-13

#[test]
fn nc_13_the_approved_evidence_is_unmutated_after_the_whole_suite() {
    let root = approved_evidence_root();
    assert_eq!(
        lane_digest::ix_agg_1(&root),
        EVIDENCE_IX_AGG_1,
        "NC-13: IX-AGG-1 over the 14 approved fixtures moved"
    );
    assert_eq!(
        lane_digest::gaia_agg_1(&root),
        EVIDENCE_GAIA_AGG_1,
        "NC-13: GAIA-AGG-1 over the 14 approved fixtures moved"
    );

    let files = lane_digest::ordinal_files(&root);
    assert_eq!(files.len(), 14);
    assert_eq!(files.iter().map(|file| file.bytes).sum::<u64>(), 663_567);

    // The reviewed M2 crate tree, byte for byte.
    let census_tree = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("ix-gaia-census");
    assert_eq!(
        lane_digest::ix_agg_1(&census_tree),
        CENSUS_TREE_IX_AGG_1,
        "NC-13: the reviewed M2 crate is an exact fixed point and must not move"
    );

    assert_eq!(
        SUBJECT_AGGREGATE_IX_AGG_1, SUBJECT_IX_AGG_1,
        "the declared constant must be the reviewed subject's identity"
    );
}

// ---------------------------------------------------------------- Lane W stops

#[test]
fn lane_w_opens_no_holdout_path_because_none_exists() {
    let base = scratch_root("");
    if base.exists() {
        let mut offenders = Vec::new();
        collect_holdout_paths(&base, &mut offenders);
        assert!(
            offenders.is_empty(),
            "Lane W must never open, create or evaluate a Class-Holdout path; found {offenders:?}"
        );
    }
}

fn collect_holdout_paths(dir: &std::path::Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if name == "hold" || name == "holdout" || name == "holdout-ledger.jsonl" {
            out.push(path.display().to_string());
        }
        if path.is_dir() {
            collect_holdout_paths(&path, out);
        }
    }
}

// ---------------------------------------------------------------- the freeze

#[test]
fn the_freeze_block_is_filled_from_measurement() {
    let corpus = corpus::generate(
        &fresh_scratch("s11-freeze").join("corpus"),
        &Options {
            families: vec![
                Family::Crlf,
                Family::Debris,
                Family::Restamp,
                Family::StaleFigure,
                Family::Pristine,
            ],
            inject_duplicate: false,
        },
    );

    // The preregistered predictions, checked one last time before they freeze.
    assert_eq!(corpus.enumerated, 26, "X2");
    assert_eq!(corpus.admitted.len(), 25, "X2");
    assert_eq!(corpus.rejected.len(), 1, "X2");
    assert_eq!(corpus.truth_balance(), (1, 24), "X3");

    let rule_source = rule_source_digest();
    assert_eq!(rule_source.len(), 64);

    let mut block = String::new();
    block.push_str("# --- filled at S11 by Lane W, before any Class-Holdout byte exists ---\n");
    block.push_str(&format!("rule_source_digest        = {rule_source}\n"));
    for width in SWEPT_WIDTHS {
        block.push_str(&format!(
            "candidate_fingerprint w={width:<5} = {}\n",
            candidate_fingerprint(&rule_source, width)
        ));
    }
    for width in SWEPT_WIDTHS {
        block.push_str(&format!(
            "window_reference_digest w={width:<4} = {}\n",
            window_reference_digest(width)
        ));
    }
    block.push_str(&format!("corpus_digest_ix_agg_1    = {}\n", corpus.digest));
    block.push_str(&format!(
        "admitted_cases            = {}     rejected_cases = {}\n",
        corpus.admitted.len(),
        corpus.rejected.len()
    ));
    let (t, f) = corpus.truth_balance();
    block.push_str(&format!("truth_label_balance       = {t} : {f}\n"));
    block.push_str("holdout_corpus_digest_ix_agg_1 = <NOT APPLICABLE — no eligible Class-Holdout population exists>\n");
    block.push_str("ledger_digest_ix_agg_1         = <NOT APPLICABLE — ledger is empty>\n");

    let out = scratch_root("s11-freeze").join("freeze-block.txt");
    std::fs::write(&out, &block).expect("the freeze block is writable");
    println!("--- FREEZE BLOCK ---\n{block}--- END FREEZE BLOCK ---");

    // Every width produces a distinct fingerprint and a distinct table.
    let fingerprints: Vec<String> = SWEPT_WIDTHS
        .iter()
        .map(|width| candidate_fingerprint(&rule_source, *width))
        .collect();
    let mut unique = fingerprints.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        3,
        "each swept width is its own fingerprint; three widths, three fingerprints"
    );

    let tables: Vec<String> = SWEPT_WIDTHS
        .iter()
        .map(|w| window_reference_digest(*w))
        .collect();
    let mut unique_tables = tables.clone();
    unique_tables.sort();
    unique_tables.dedup();
    assert_eq!(unique_tables.len(), 3, "each swept width has its own table");
}

#[test]
fn the_two_deliverable_outputs_are_emitted_and_content_addressed() {
    let out = fresh_scratch("s11-deliverables");

    // 1. The advisory artifact over the 14 approved fixtures.
    let artifact = advise(&AdvisoryRequest {
        evidence_root: approved_evidence_root(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule: RuleId::FullDigest,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    })
    .expect("the approved evidence binds")
    .to_canonical_json();
    assert!(artifact.contains(r#""reason":"NoEligibleUnseenPopulation""#));
    std::fs::write(out.join("advisory-artifact.json"), &artifact).expect("writable");

    // 2. The Class-Development characterization report.
    let corpus = corpus::generate(
        &out.join("corpus"),
        &Options {
            families: vec![
                Family::Crlf,
                Family::Debris,
                Family::Restamp,
                Family::StaleFigure,
                Family::Pristine,
            ],
            inject_duplicate: false,
        },
    );
    let mut cells: Vec<ix_gaia_advisory::RuleWidth> = [
        RuleId::AlwaysAgree,
        RuleId::NameSetOnly,
        RuleId::LengthOnly,
        RuleId::StructuralOnly,
        RuleId::FullDigest,
    ]
    .into_iter()
    .map(|rule| ix_gaia_advisory::RuleWidth {
        rule,
        window_bytes: 0,
    })
    .collect();
    for window_bytes in SWEPT_WIDTHS {
        cells.push(ix_gaia_advisory::RuleWidth {
            rule: RuleId::WindowProbe,
            window_bytes,
        });
    }
    let report = ix_gaia_advisory::evaluate(&ix_gaia_advisory::EvalRequest {
        corpus_root: corpus.root.clone(),
        expected_corpus_digest_ix_agg_1: corpus.digest.clone(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        cells,
        window_reference_root: approved_evidence_root(),
        evidence_class: ix_gaia_advisory::EvidenceClass::Development,
    })
    .expect("the corpus binds")
    .to_canonical_json();
    std::fs::write(out.join("evaluation-report.json"), &report).expect("writable");

    println!(
        "advisory-artifact.json   bytes={} sha256={}",
        artifact.len(),
        lane_digest::sha256_hex(artifact.as_bytes())
    );
    println!(
        "evaluation-report.json   bytes={} sha256={}",
        report.len(),
        lane_digest::sha256_hex(report.as_bytes())
    );
    println!("corpus_digest_ix_agg_1   {}", corpus.digest);
}

#[test]
fn the_frozen_values_are_reproducible_across_runs() {
    let a = corpus::generate(
        &fresh_scratch("s11-repro-a").join("corpus"),
        &Options::family(Family::Restamp),
    );
    let b = corpus::generate(
        &fresh_scratch("s11-repro-b").join("corpus"),
        &Options::family(Family::Restamp),
    );
    assert_eq!(a.digest, b.digest);
    assert_eq!(rule_source_digest(), rule_source_digest());
    for width in SWEPT_WIDTHS {
        assert_eq!(
            window_reference_digest(width),
            window_reference_digest(width)
        );
    }
}
