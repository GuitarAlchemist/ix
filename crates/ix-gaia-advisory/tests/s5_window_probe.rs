//! Slice S5 — the Window Reference Table and the windowed rule, swept.
//!
//! Red-first discriminators:
//!
//! * `WindowProbe(0)` is *definitionally* the content-free ablation, so it must
//!   produce the ablation's outcome on every root — NC-9. A divergence is a
//!   harness defect, not a finding;
//! * a single flipped byte in the pristine reference changes the table, and a
//!   run that was told which table to expect must **refuse** rather than
//!   silently use the other one — NC-2;
//! * the table costs one full pristine pass, and that cost is reported rather
//!   than hidden — NC-14.
//!
//! No width is called "the candidate" and no width is selected. The sweep is
//! `{0, 1024, 4096}` and every figure names its width.

mod support;

use ix_gaia_advisory::{
    advise, AdvisoryRefusal, AdvisoryRequest, ExpectedProvenance, Hexavalent, RuleId,
};

use support::{approved_evidence_root, fresh_scratch, stage_pristine, MANIFEST_FILE_NAME};

const SWEPT_WIDTHS: [u32; 3] = [0, 1024, 4096];
const EVIDENCE_BYTES: u64 = 663_567;

fn window_request(root: &std::path::Path, window_bytes: u32) -> AdvisoryRequest {
    AdvisoryRequest {
        evidence_root: root.to_path_buf(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule: RuleId::WindowProbe,
        window_bytes,
        window_reference_root: Some(approved_evidence_root()),
        expected_provenance: None,
    }
}

fn plain_request(root: &std::path::Path, rule: RuleId) -> AdvisoryRequest {
    AdvisoryRequest {
        evidence_root: root.to_path_buf(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    }
}

/// A length-preserving single-byte edit at the first ASCII digit of `name`.
fn stage_with_length_preserving_edit(dir: &str, name: &str) -> std::path::PathBuf {
    let root = fresh_scratch(dir).join("root");
    stage_pristine(&root);
    let target = root.join(name);
    let mut bytes = std::fs::read(&target).expect("readable");
    let site = bytes
        .iter()
        .position(|byte| byte.is_ascii_digit())
        .expect("the fixture carries an ASCII digit");
    bytes[site] = b'0' + ((bytes[site] - b'0' + 1) % 10);
    std::fs::write(&target, &bytes).expect("writable");
    root
}

// ---------------------------------------------------------------- NC-9

#[test]
fn nc_9_width_zero_is_exactly_the_content_free_ablation() {
    let pristine = approved_evidence_root();
    let extra = {
        let root = fresh_scratch("s5-extra").join("root");
        stage_pristine(&root);
        std::fs::write(root.join("gaia-s1-r2-validator.py.stdout"), b"surplus").expect("writable");
        root
    };
    let edited = stage_with_length_preserving_edit("s5-edit", "gaia-uncertainty-grammar-v0.1.ebnf");

    for root in [pristine, extra, edited] {
        let ablation = advise(&plain_request(&root, RuleId::StructuralOnly)).expect("binds");
        let width_zero = advise(&window_request(&root, 0)).expect("binds");

        assert_eq!(
            width_zero.reconciles,
            ablation.reconciles,
            "NC-9: WindowProbe(0) diverged from StructuralOnly on {}",
            root.display()
        );
        assert_eq!(
            width_zero
                .rows
                .iter()
                .map(|row| (row.name.as_str(), row.state))
                .collect::<Vec<_>>(),
            ablation
                .rows
                .iter()
                .map(|row| (row.name.as_str(), row.state))
                .collect::<Vec<_>>(),
            "NC-9: the two must agree row for row"
        );
        for row in &width_zero.rows {
            assert_eq!(
                row.head_window_matches,
                Hexavalent::Unknown,
                "at width zero no window is read, so neither window binds"
            );
            assert_eq!(row.tail_window_matches, Hexavalent::Unknown);
        }
        assert_eq!(width_zero.cost.bytes_read, 0);
        assert_eq!(width_zero.cost.wrt_construction_bytes, 0);
    }
}

// ---------------------------------------------------------------- the sweep

#[test]
fn the_windowed_rule_catches_an_edit_inside_its_window_and_misses_one_outside() {
    // `gaia-uncertainty-grammar-v0.1.ebnf` is 10,322 B; its first ASCII digit
    // sits inside the head window at both non-zero widths.
    let inside =
        stage_with_length_preserving_edit("s5-inside", "gaia-uncertainty-grammar-v0.1.ebnf");
    for width in [1024u32, 4096] {
        let artifact = advise(&window_request(&inside, width)).expect("binds");
        assert_eq!(
            artifact.reconciles,
            Hexavalent::False,
            "an edit inside the head window at w={width} must be caught"
        );
        let row = artifact
            .rows
            .iter()
            .find(|row| row.name == "gaia-uncertainty-grammar-v0.1.ebnf")
            .expect("the row is reported");
        assert_eq!(row.head_window_matches, Hexavalent::False);
        assert_eq!(row.tail_window_matches, Hexavalent::True);
        assert_eq!(row.state, Hexavalent::False);
    }

    // The same shape deep inside a large file is outside every window.
    let deep = {
        let root = fresh_scratch("s5-deep").join("root");
        stage_pristine(&root);
        let target = root.join("gaia-s1-r2-change-ledger.md");
        let mut bytes = std::fs::read(&target).expect("readable");
        // Byte 117,421 — a documented census-claim token, far from both edges.
        bytes[117_421] = b'4';
        std::fs::write(&target, &bytes).expect("writable");
        root
    };
    for width in [1024u32, 4096] {
        let artifact = advise(&window_request(&deep, width)).expect("binds");
        assert_eq!(
            artifact.reconciles,
            Hexavalent::True,
            "an edit outside every window at w={width} must be MISSED — that is the measurement"
        );
    }
}

#[test]
fn head_and_tail_overlap_is_defined_and_disclosed_not_guarded_away() {
    // §9.2's measured consequence: exactly one of the fourteen approved files
    // is short enough for the two windows to cover it completely at w = 4096,
    // and none is at w = 1024.
    let root = approved_evidence_root();
    let mut covered_at_4096 = 0usize;
    let mut covered_at_1024 = 0usize;
    for entry in std::fs::read_dir(&root).expect("readable") {
        let len = entry.expect("entry").metadata().expect("stat").len();
        if len <= 2 * 4096 {
            covered_at_4096 += 1;
        }
        if len <= 2 * 1024 {
            covered_at_1024 += 1;
        }
    }
    assert_eq!(
        covered_at_4096, 1,
        "exactly one file is fully covered at w=4096"
    );
    assert_eq!(covered_at_1024, 0, "no file is fully covered at w=1024");
}

// ---------------------------------------------------------------- NC-14

#[test]
fn nc_14_the_cost_account_is_the_harnesss_and_the_budget_bounds_it() {
    let root = approved_evidence_root();
    for width in SWEPT_WIDTHS {
        let artifact = advise(&window_request(&root, width)).expect("binds");

        // Independently derived in the test lane from the file lengths alone.
        let expected: u64 = std::fs::read_dir(&root)
            .expect("readable")
            .map(|entry| {
                let entry = entry.expect("entry");
                let len = entry.metadata().expect("stat").len();
                let declared = entry.file_name() != *std::ffi::OsStr::new(MANIFEST_FILE_NAME);
                if !declared || width == 0 {
                    0
                } else {
                    len.min(2 * u64::from(width))
                }
            })
            .sum();
        assert_eq!(
            artifact.cost.bytes_read, expected,
            "the harness's byte count must equal the rule's declared reads at w={width}"
        );
        assert!(
            artifact.cost.bytes_read <= artifact.cost.budget_bytes,
            "invariant at w={width}: {} <= {}",
            artifact.cost.bytes_read,
            artifact.cost.budget_bytes
        );

        if width > 0 {
            assert_eq!(
                artifact.cost.wrt_construction_bytes, EVIDENCE_BYTES,
                "the table costs one full pristine pass at w={width}, disclosed rather than hidden"
            );
        } else {
            assert_eq!(artifact.cost.wrt_construction_bytes, 0);
        }
        assert!(
            artifact.cost.bytes_read < artifact.cost.reference_bytes_read || width == 0,
            "a windowed read is cheaper than the exact reference at w={width}"
        );
    }
}

// ---------------------------------------------------------------- NC-2

#[test]
fn nc_2_a_tampered_window_reference_table_cannot_be_silently_used() {
    let root = approved_evidence_root();
    let honest = advise(&window_request(&root, 1024)).expect("binds");
    let honest_digest = honest.provenance.window_reference_digest.clone();
    let honest_manifest = honest.provenance.manifest_sha256.clone();

    // A reference root with one byte flipped produces a different table.
    let tampered_reference = fresh_scratch("s5-nc2-reference").join("root");
    stage_pristine(&tampered_reference);
    let victim = tampered_reference.join("gaia-mission-room-domain-context-v0.1.md");
    let mut bytes = std::fs::read(&victim).expect("readable");
    bytes[0] ^= 0x01;
    std::fs::write(&victim, &bytes).expect("writable");

    let mut request = window_request(&root, 1024);
    request.window_reference_root = Some(tampered_reference);
    request.expected_provenance = Some(ExpectedProvenance {
        window_reference_digest: honest_digest.clone(),
        manifest_sha256: honest_manifest.clone(),
    });

    match advise(&request) {
        Err(AdvisoryRefusal::ProvenanceMismatch {
            field,
            expected,
            measured,
        }) => {
            assert_eq!(field, "window_reference_digest");
            assert_eq!(expected, honest_digest);
            assert_ne!(measured, honest_digest);
        }
        Err(other) => panic!("expected ProvenanceMismatch, got {other:?}"),
        Ok(artifact) => panic!("NC-2 violated: a tampered table was used: {artifact:?}"),
    }
}

#[test]
fn nc_2_a_tampered_manifest_cannot_be_silently_used() {
    let root = approved_evidence_root();
    let honest = advise(&window_request(&root, 1024)).expect("binds");

    let staged = fresh_scratch("s5-nc2-manifest").join("root");
    stage_pristine(&staged);
    let manifest = staged.join(MANIFEST_FILE_NAME);
    let mut bytes = std::fs::read(&manifest).expect("readable");
    let site = bytes
        .iter()
        .position(|byte| *byte == b'#')
        .expect("the manifest carries a heading marker");
    bytes[site] = b'=';
    std::fs::write(&manifest, &bytes).expect("writable");

    let mut request = window_request(&staged, 1024);
    request.expected_provenance = Some(ExpectedProvenance {
        window_reference_digest: honest.provenance.window_reference_digest.clone(),
        manifest_sha256: honest.provenance.manifest_sha256.clone(),
    });

    match advise(&request) {
        Err(AdvisoryRefusal::ProvenanceMismatch { field, .. }) => {
            assert_eq!(field, "manifest_sha256");
        }
        Err(other) => panic!("expected ProvenanceMismatch, got {other:?}"),
        Ok(artifact) => panic!("NC-2 violated: a tampered manifest was used: {artifact:?}"),
    }
}

#[test]
fn a_windowed_rule_without_pristine_reference_bytes_refuses() {
    let mut request = window_request(&approved_evidence_root(), 1024);
    request.window_reference_root = None;
    match advise(&request) {
        Err(AdvisoryRefusal::BinderIncomplete { field, .. }) => {
            assert_eq!(field, "window_reference_root");
        }
        other => panic!("a windowed rule with no reference must refuse, got {other:?}"),
    }
}
