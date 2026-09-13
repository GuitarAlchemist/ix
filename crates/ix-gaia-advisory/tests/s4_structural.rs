//! Slice S4 — the content-free ablation, and its blind spot pinned rather than
//! left incidental.
//!
//! Red-first discriminator: a length-preserving single-byte edit must produce
//! `T` from all three structural rules — **a wrong answer, stated in advance**.
//! It is precisely the signal the windowed rule exists to test, and a suite
//! that did not pin it would let a later "fix" quietly close the gap and make
//! the whole characterization meaningless.
//!
//! A note on the slice's shorthand. §16 S4 says an extra file yields `F` "from
//! all three". Taken literally that would require `LengthOnly` to detect a
//! name-set change, which contradicts its normative definition in §9.1
//! (`F` iff any declared length differs from the measured length). The rules
//! are implemented to §9.1 and the assertions below are the per-rule outcomes
//! §9.1 entails, which discriminate strictly more than the shorthand does.

mod support;

use ix_gaia_advisory::{advise, AdvisoryRequest, Hexavalent, RuleId};

use support::{approved_evidence_root, fresh_scratch, stage_pristine, MANIFEST_FILE_NAME};

const A_DECLARED_FILE: &str = "gaia-mission-room-domain-context-v0.1.md";

fn reconciles(root: &std::path::Path, rule: RuleId) -> Hexavalent {
    advise(&AdvisoryRequest {
        evidence_root: root.to_path_buf(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    })
    .expect("the staged root binds")
    .reconciles
}

const STRUCTURAL: [RuleId; 3] = [
    RuleId::NameSetOnly,
    RuleId::LengthOnly,
    RuleId::StructuralOnly,
];

#[test]
fn on_the_pristine_root_every_structural_rule_agrees() {
    let root = approved_evidence_root();
    for rule in STRUCTURAL {
        assert_eq!(
            reconciles(&root, rule),
            Hexavalent::True,
            "{rule:?} must agree on the pristine root"
        );
    }
    assert_eq!(reconciles(&root, RuleId::AlwaysAgree), Hexavalent::True);
}

#[test]
fn an_extra_file_is_caught_by_the_name_set_and_missed_by_the_lengths() {
    let dir = fresh_scratch("s4-extra-file");
    let root = dir.join("root");
    stage_pristine(&root);
    std::fs::write(root.join("gaia-s1-r2-validator.py.stdout"), b"surplus").expect("writable");

    assert_eq!(reconciles(&root, RuleId::NameSetOnly), Hexavalent::False);
    assert_eq!(reconciles(&root, RuleId::StructuralOnly), Hexavalent::False);
    assert_eq!(
        reconciles(&root, RuleId::LengthOnly),
        Hexavalent::True,
        "LengthOnly compares declared lengths and cannot see a surplus name"
    );
    assert_eq!(
        reconciles(&root, RuleId::AlwaysAgree),
        Hexavalent::True,
        "the floor agrees with everything, which is what makes it a floor"
    );
    assert_eq!(reconciles(&root, RuleId::FullDigest), Hexavalent::False);
}

#[test]
fn a_length_changing_edit_is_caught_by_the_lengths_and_missed_by_the_name_set() {
    let dir = fresh_scratch("s4-length-change");
    let root = dir.join("root");
    stage_pristine(&root);
    let target = root.join(A_DECLARED_FILE);
    let mut bytes = std::fs::read(&target).expect("readable");
    bytes.push(b'\n');
    std::fs::write(&target, &bytes).expect("writable");

    assert_eq!(reconciles(&root, RuleId::LengthOnly), Hexavalent::False);
    assert_eq!(reconciles(&root, RuleId::StructuralOnly), Hexavalent::False);
    assert_eq!(
        reconciles(&root, RuleId::NameSetOnly),
        Hexavalent::True,
        "NameSetOnly compares names and cannot see a byte"
    );
    assert_eq!(reconciles(&root, RuleId::FullDigest), Hexavalent::False);
}

#[test]
fn a_length_preserving_single_byte_edit_is_missed_by_every_structural_rule() {
    let dir = fresh_scratch("s4-length-preserving");
    let root = dir.join("root");
    stage_pristine(&root);
    let target = root.join(A_DECLARED_FILE);
    let mut bytes = std::fs::read(&target).expect("readable");
    let site = bytes
        .iter()
        .position(|byte| byte.is_ascii_digit())
        .expect("the fixture carries an ASCII digit");
    let original = bytes[site];
    bytes[site] = b'0' + ((original - b'0' + 1) % 10);
    let length_before = std::fs::metadata(&target).expect("stat").len();
    std::fs::write(&target, &bytes).expect("writable");
    assert_eq!(
        std::fs::metadata(&target).expect("stat").len(),
        length_before,
        "the test's own premise: the edit preserves length"
    );

    for rule in STRUCTURAL {
        assert_eq!(
            reconciles(&root, rule),
            Hexavalent::True,
            "{rule:?} must MISS a length-preserving edit — this blind spot is the measurement, not a defect"
        );
    }
    assert_eq!(
        reconciles(&root, RuleId::FullDigest),
        Hexavalent::False,
        "the exact reference does catch it, which is what makes the blind spot a gap and not a tie"
    );
}

#[test]
fn no_structural_rule_reads_a_content_byte() {
    let root = approved_evidence_root();
    for rule in [
        RuleId::AlwaysAgree,
        RuleId::NameSetOnly,
        RuleId::LengthOnly,
        RuleId::StructuralOnly,
    ] {
        let artifact = advise(&AdvisoryRequest {
            evidence_root: root.clone(),
            manifest_file_name: MANIFEST_FILE_NAME.to_string(),
            rule,
            window_bytes: 0,
            window_reference_root: None,
            expected_provenance: None,
        })
        .expect("binds");
        assert_eq!(artifact.cost.bytes_read, 0, "{rule:?} reads no content");
        assert_eq!(artifact.cost.budget_bytes, 0, "{rule:?} declares no budget");
        assert_eq!(artifact.cost.files_opened, 0, "{rule:?} opens no file");
        assert_eq!(
            artifact.cost.cost_ratio.value,
            Some(0.0),
            "a bound zero over a non-zero denominator is 0.0, not null"
        );
    }
}
