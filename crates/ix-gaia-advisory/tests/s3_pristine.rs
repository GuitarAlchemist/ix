//! Slice S3 — the positive control: the pristine approved evidence reconciles.
//!
//! Red-first discriminator: `advise` with the exact reference rule over the 14
//! approved fixtures must fold to `reconciles == T`. Until the reference arm is
//! implemented the rule honestly reports `U` for every row, and `U` never
//! satisfies a pass — which is exactly what this slice makes it stop doing.
//!
//! Also: provenance binds every digest §3.3 names, and the cost account is the
//! harness's, not the rule's.

mod support;

use ix_gaia_advisory::{advise, AdvisoryRequest, Hexavalent, RuleId};

use support::{approved_evidence_root, MANIFEST_FILE_NAME};

const EVIDENCE_IX_AGG_1: &str = "217815fc65ef08c6d72dbe7211129ee60cc73531f2050c6e59788e1a40324f9b";
const EVIDENCE_GAIA_AGG_1: &str =
    "e8a7d0927b03e9b9379892fb00bdb9ac132c760a39b0218fd7423d5b7122598e";
const SUBJECT_IX_AGG_1: &str = "a3e6895759d007c45d7311c9b05387befdf22212cc64671c6a94cd731dc4c235";
const MANIFEST_SHA256: &str = "275e5b85ede1a97e9d301c850f1731cdb3d2a24d11f6bb303111dfcb6f010fae";
const EVIDENCE_BYTES: u64 = 663_567;

fn request(rule: RuleId) -> AdvisoryRequest {
    AdvisoryRequest {
        evidence_root: approved_evidence_root(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    }
}

#[test]
fn the_pristine_approved_evidence_reconciles_under_the_exact_reference() {
    let artifact = advise(&request(RuleId::FullDigest)).expect("the approved evidence binds");

    assert_eq!(
        artifact.reconciles,
        Hexavalent::True,
        "the positive control must fold to T; rows were {:?}",
        artifact
            .rows
            .iter()
            .map(|row| (row.name.as_str(), row.state))
            .collect::<Vec<_>>()
    );
    for row in &artifact.rows {
        assert_eq!(
            row.state,
            Hexavalent::True,
            "row {} must reconcile",
            row.name
        );
        assert_eq!(
            row.measured_bytes,
            Some(row.declared_bytes),
            "row {} measures what it declares",
            row.name
        );
        assert_eq!(
            row.head_window_matches,
            Hexavalent::Unknown,
            "the exact reference reads no window, so it binds neither"
        );
        assert_eq!(row.tail_window_matches, Hexavalent::Unknown);
    }

    assert_eq!(artifact.root.declared_files, 13);
    assert_eq!(artifact.root.present_files, 13);
    assert!(artifact.root.unlisted_names.is_empty());
    assert!(artifact.root.missing_names.is_empty());
    assert!(artifact.root.non_file_names.is_empty());
}

#[test]
fn provenance_binds_every_named_digest() {
    let artifact = advise(&request(RuleId::FullDigest)).expect("the approved evidence binds");
    let provenance = &artifact.provenance;

    assert_eq!(provenance.manifest_file_name, MANIFEST_FILE_NAME);
    assert_eq!(provenance.manifest_sha256, MANIFEST_SHA256);
    assert_eq!(provenance.evidence_aggregate_ix_agg_1, EVIDENCE_IX_AGG_1);
    assert_eq!(
        provenance.declared_bundle_aggregate_gaia_agg_1,
        EVIDENCE_GAIA_AGG_1
    );
    assert_eq!(provenance.subject_aggregate_ix_agg_1, SUBJECT_IX_AGG_1);
    for (field, value) in [
        (
            "window_reference_digest",
            &provenance.window_reference_digest,
        ),
        ("rule_source_digest", &provenance.rule_source_digest),
    ] {
        assert_eq!(value.len(), 64, "{field} is a SHA-256");
        assert!(
            value
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
            "{field} is lowercase hex"
        );
    }

    // Absence outside an evaluation run is a null, never an omitted key.
    assert_eq!(provenance.corpus_digest_ix_agg_1, None);
    assert_eq!(provenance.ledger_digest_ix_agg_1, None);
    assert_eq!(provenance.case_id, None);
    assert_eq!(provenance.evidence_class, None);
}

#[test]
fn the_cost_account_is_the_harnesss_and_the_budget_invariant_holds() {
    let reference = advise(&request(RuleId::FullDigest)).expect("the approved evidence binds");
    assert_eq!(reference.cost.reference_bytes_read, EVIDENCE_BYTES);
    assert_eq!(
        reference.cost.bytes_read, EVIDENCE_BYTES,
        "the exact reference reads every content byte"
    );
    assert_eq!(reference.cost.files_opened, 14);
    assert!(
        reference.cost.bytes_read <= reference.cost.budget_bytes,
        "invariant: {} <= {}",
        reference.cost.bytes_read,
        reference.cost.budget_bytes
    );
    assert_eq!(reference.cost.cost_ratio.value, Some(1.0));
    assert_eq!(
        reference.cost.provenance_bytes_read, EVIDENCE_BYTES,
        "the provenance pass is disclosed separately, never folded into the rule's figure"
    );
    assert_eq!(
        reference.cost.wrt_construction_bytes, 0,
        "no Window Reference Table is bound for a rule that reads no window"
    );
}

#[test]
fn a_non_zero_width_on_a_rule_that_reads_no_window_is_a_refusal() {
    let mut request = request(RuleId::FullDigest);
    request.window_bytes = 1024;
    let outcome = advise(&request);
    assert!(
        outcome.is_err(),
        "a width the rule cannot honour is an unresolvable declared field, not a value to ignore"
    );
}

#[test]
fn the_artifact_serializes_canonically_and_byte_identically() {
    let first = advise(&request(RuleId::FullDigest)).expect("binds");
    let second = advise(&request(RuleId::FullDigest)).expect("binds");
    assert_eq!(
        first.to_canonical_json(),
        second.to_canonical_json(),
        "same bytes in, byte-identical artifact out"
    );
    assert!(first
        .to_canonical_json()
        .starts_with(r#"{"schema_version":1,"rule":"FullDigest""#));
}
