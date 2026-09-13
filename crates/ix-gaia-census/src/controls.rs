//! The control-coverage map carried by every artifact.
//!
//! The M2 test obligation is the union of the controls enumerated at
//! specification §11 and at engineering-doctrine §4. The **enumerated lists**
//! are authoritative here — §11 runs to twenty-eight entries while other
//! sentences in the same document state the count as eighteen and as
//! twenty-six. That discrepancy is a specification-level observation, not
//! something this crate repairs, and this map does not resolve it: it covers
//! every enumerated row.
//!
//! Most §11 controls presuppose leases, epochs, coordinators, fences,
//! partitions, buses, spend, or acceptance. A read-only local census has none
//! of them, so those controls have no subject. They are marked `NotApplicable`
//! and each names the subject it is missing. None is silently dropped, and no
//! subsystem was built merely to make an irrelevant control apply.

use crate::{ControlCoverage, ControlStatus};

/// `(control_id, covered, detail)` — the detail names the discriminating test
/// when covered, and the missing subject when not.
const CONTROLS: &[(&str, bool, &str)] = &[
    (
        "spec-11-01",
        false,
        "T1 holds no Lease and has no agent incarnation; there is no old lease to reject.",
    ),
    (
        "spec-11-02",
        false,
        "T1 has no Coordinator and no epoch; there is no stale epoch to reject.",
    ),
    (
        "spec-11-03",
        false,
        "T1 claims no lineage and performs no mutation; two claimants cannot arise.",
    ),
    (
        "spec-11-04",
        false,
        "T1 has no lineage and no partition; LINEAGE_DIVERGENCE has no subject.",
    ),
    (
        "spec-11-05",
        false,
        "T1 issues no command and performs no transfer of authority.",
    ),
    (
        "spec-11-06",
        false,
        "T1 calls no bus, sends no intent, and emits no receipt.",
    ),
    (
        "spec-11-07",
        true,
        "a_single_byte_mutation_marks_exactly_one_row_false and \
         a_mutated_declared_digest_retains_both_values observe evidence drift over raw bytes, \
         with the CRLF case pinned by the_crlf_file_hashes_as_the_raw_bytes_it_holds.",
    ),
    (
        "spec-11-08",
        false,
        "T1 has no fence, no coordinator, and no executor to bypass.",
    ),
    (
        "spec-11-09",
        false,
        "T1 fits no model, launches nothing, and incurs no spend; there is no accounting subject.",
    ),
    (
        "spec-11-10",
        false,
        "T1 addresses no recipient and holds no surface identity to recycle.",
    ),
    (
        "spec-11-11",
        true,
        "the_declared_construction_is_the_only_one_that_reproduces_the_fixed_point and \
         a_run_leaves_the_evidence_root_unchanged bind the reported aggregate to measured bytes \
         and to a directory the run did not touch.",
    ),
    (
        "spec-11-12",
        false,
        "T1 performs no review, approval, or acceptance, so there is nothing to self-approve; \
         the expected digests it asserts are declared by the reviewed evidence, not authored here.",
    ),
    (
        "spec-11-13",
        true,
        "manifest_declaring_the_same_name_twice_refuses and \
         a_second_inventory_section_refuses_rather_than_collapsing_into_a_pass keep a \
         contradictory declaration contradictory wherever it sits: a repeated row is refused with \
         the name retained, a repeated inventory section with both line numbers retained, and \
         neither is ever collapsed into a pass.",
    ),
    (
        "spec-11-14",
        false,
        "T1 consults no consensus, no model panel, and no vote.",
    ),
    (
        "spec-11-15",
        true,
        "absent_declared_file_refuses_and_emits_no_artifact and \
         manifest_declaring_a_name_outside_the_evidence_root_refuses reject a missing or \
         out-of-root evidence path before any read leaves the evidence root.",
    ),
    (
        "spec-11-16",
        false,
        "T1 has no actor register and no ledger; nothing is admitted by attribution.",
    ),
    (
        "spec-11-17",
        false,
        "T1 mutates nothing, so a write scope has nothing to violate; a read outside the \
         evidence root is refused under spec-11-15.",
    ),
    (
        "spec-11-18",
        true,
        "the_declared_construction_is_the_only_one_that_reproduces_the_fixed_point fails against \
         the three wrong constructions the manifest declares, so the gate can fail.",
    ),
    (
        "spec-11-19",
        true,
        "a_zero_byte_declared_file_is_true_not_a_failure and \
         absent_declared_file_refuses_and_emits_no_artifact separate an empty value from an \
         absent one: the first is a value, the second is refused and never defaulted.",
    ),
    (
        "spec-11-20",
        false,
        "T1 introduces no shared seam and no adapter set; it consumes the workspace's existing \
         hexavalent type and generalizes nothing.",
    ),
    (
        "spec-11-21",
        true,
        "the_artifact_carries_no_score_verdict_or_authority_field asserts the artifact carries no \
         verdict, freshness, execution, or acceptance value, and no evidential gradient.",
    ),
    (
        "spec-11-22",
        false,
        "T1 references no Actor and no identity class.",
    ),
    (
        "spec-11-23",
        false,
        "T1 adds no verb, no bus alternative, and no ledger terminal.",
    ),
    (
        "spec-11-24",
        true,
        "malformed_manifest_rows_refuse_with_the_offending_line fails closed on an unrecognised \
         row, preserving the offending line rather than turning it into a pass.",
    ),
    (
        "spec-11-25",
        false,
        "T1 reads no grammar and classifies no production.",
    ),
    (
        "spec-11-26",
        true,
        "every_reported_state_is_attributable binds every hexavalent value to a named file and \
         its measured digest, and names the manifest and the manifest's digest as the subject; \
         no acceptance is recorded at all.",
    ),
    (
        "spec-11-27",
        false,
        "T1 reads no grammar and has no bus class to bound.",
    ),
    (
        "spec-11-28",
        false,
        "T1 reads no grammar and has no ledger class to bound.",
    ),
    (
        "doctrine-4-01",
        false,
        "the run's declared write scope is a run-level object; T1 mutates nothing it reads.",
    ),
    (
        "doctrine-4-02",
        true,
        "copied_evidence_reproduces_its_declared_deterministic_census states success as a \
         byte-exact digest declared by the reviewed evidence, not as a judgement.",
    ),
    (
        "doctrine-4-03",
        true,
        "the_declared_construction_is_the_only_one_that_reproduces_the_fixed_point and \
         a_single_byte_mutation_marks_exactly_one_row_false make the gate fail on known mutations.",
    ),
    (
        "doctrine-4-04",
        false,
        "a recorded red state is a process artifact of the writing run, not a census subject; \
         the red log lives in the writer handoff.",
    ),
    (
        "doctrine-4-05",
        true,
        "every module of this crate is private, so every_reported_state_is_attributable and every \
         other test can reach only census() and the artifact it returns.",
    ),
    (
        "doctrine-4-06",
        false,
        "same missing subject as spec-11-20: no generalized seam exists to justify.",
    ),
    (
        "doctrine-4-07",
        false,
        "no prototype, generated file, or ignored artifact is promoted; the load-bearing expected \
         values are declared by the copied evidence.",
    ),
    (
        "doctrine-4-08",
        false,
        "same missing subject as spec-11-12: T1 performs no review, acceptance, or promotion.",
    ),
    (
        "doctrine-4-09",
        true,
        "copied_evidence_reproduces_its_declared_deterministic_census and \
         every_reported_state_is_attributable measure and report the subject digest; no model and \
         no cost boundary exists to be unknown.",
    ),
    (
        "doctrine-4-10",
        false,
        "staging and scope widening are process controls over a working tree, not census subjects.",
    ),
    (
        "doctrine-4-11",
        true,
        "a_run_leaves_the_evidence_root_unchanged and \
         the_declared_construction_is_the_only_one_that_reproduces_the_fixed_point bind the \
         completion of a run to the fixed point and to a clean evidence directory.",
    ),
    (
        "doctrine-4-12",
        true,
        "the_artifact_carries_no_score_verdict_or_authority_field keeps the artifact incapable of \
         granting an effect.",
    ),
    (
        "doctrine-4-13",
        false,
        "T1 reads only the declared evidence files; no chat log and no vendor memory is an input.",
    ),
    (
        "doctrine-4-14",
        false,
        "T1 is a library function returning a typed advisory artifact; it claims no authority for \
         any procedure.",
    ),
];

/// The control-coverage map. Constant, so it neither varies by run nor by host.
pub(crate) fn coverage() -> Vec<ControlCoverage> {
    CONTROLS
        .iter()
        .map(|(control_id, covered, detail)| ControlCoverage {
            control_id: (*control_id).to_string(),
            status: if *covered {
                ControlStatus::Covered
            } else {
                ControlStatus::NotApplicable
            },
            detail: (*detail).to_string(),
        })
        .collect()
}
