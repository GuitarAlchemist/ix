//! Slice S10 — contradiction retained, never collapsed.
//!
//! Red-first discriminator: when two claim sites in the same document state
//! different figures for the same quantity, the affected row must report
//! `Hexavalent::Contradictory`, **both figures must survive into the artifact**,
//! and the fold must not collapse to `T`.
//!
//! This control is modelled on a real instance in the approved evidence: the
//! bundle manifest records the shipped output at **214** CRLF pairs while three
//! round-scoped ledger census claims stand at **210**, and the manifest states
//! that rewriting those three dated figures to the current one *"was considered
//! and refused: agreement bought by re-stamping history is not agreement."*
//! Collapsing the disagreement to either value would be exactly that purchase.
//!
//! The mechanism is a fact about the *document*, not about any rule, so it
//! applies uniformly — the trivial floor cannot agree its way out of a
//! self-contradicting declaration either.

mod support;

use ix_gaia_advisory::{advise, AdvisoryRequest, Hexavalent, RuleId};

use support::{approved_evidence_root, fresh_scratch, stage_pristine, MANIFEST_FILE_NAME};

const SUBJECT: &str = "gaia-mission-room-domain-context-v0.1.md";
/// What the manifest's §2 Inventory declares for that file.
const DECLARED_BYTES: u64 = 4_859;

/// Append a second claim site: a census table outside the inventory, stating a
/// byte count for a declared file.
fn stage_with_corroborating_claim(slice: &str, claimed: u64) -> std::path::PathBuf {
    let root = fresh_scratch(slice).join("root");
    stage_pristine(&root);
    let manifest = root.join(MANIFEST_FILE_NAME);
    let mut text = std::fs::read_to_string(&manifest).expect("readable");
    text.push_str(&format!(
        "\n## 9. Census\n\n| Relative path | Bytes |\n|---|---:|\n| `{SUBJECT}` | {claimed} |\n"
    ));
    std::fs::write(&manifest, text).expect("writable");
    root
}

fn artifact(root: &std::path::Path, rule: RuleId) -> ix_gaia_advisory::AdvisoryArtifact {
    advise(&AdvisoryRequest {
        evidence_root: root.to_path_buf(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    })
    .expect("the staged root binds")
}

#[test]
fn nc_8_two_disagreeing_claim_sites_are_retained_as_contradictory() {
    let root = stage_with_corroborating_claim("s10-contradiction", 4_210);
    let artifact = artifact(&root, RuleId::FullDigest);

    let row = artifact
        .rows
        .iter()
        .find(|row| row.name == SUBJECT)
        .expect("the declared row is reported");

    assert_eq!(
        row.state,
        Hexavalent::Contradictory,
        "two claim sites disagree; the row is C, not T and not F"
    );
    assert_eq!(
        row.declared_bytes, DECLARED_BYTES,
        "the declaration of record survives"
    );
    assert_eq!(
        row.corroborated_bytes,
        Some(4_210),
        "the second claim site survives too — both values present, neither chosen"
    );
    assert_eq!(
        row.measured_bytes,
        Some(DECLARED_BYTES),
        "and the measurement is still reported alongside both claims"
    );

    assert_eq!(
        artifact.reconciles,
        Hexavalent::Contradictory,
        "the fold must not collapse a retained contradiction to T"
    );
    assert_ne!(artifact.reconciles, Hexavalent::True);

    let json = artifact.to_canonical_json();
    assert!(json.contains(r#""declared_bytes":4859"#));
    assert!(json.contains(r#""corroborated_bytes":4210"#));
    assert!(json.contains(r#""state":"C""#));
    assert!(json.contains(r#""reconciles":"C""#));
}

#[test]
fn nc_8_not_even_the_trivial_floor_agrees_its_way_out_of_a_contradiction() {
    let root = stage_with_corroborating_claim("s10-floor", 4_210);
    for rule in [
        RuleId::AlwaysAgree,
        RuleId::NameSetOnly,
        RuleId::LengthOnly,
        RuleId::StructuralOnly,
        RuleId::FullDigest,
    ] {
        let artifact = artifact(&root, rule);
        assert_ne!(
            artifact.reconciles,
            Hexavalent::True,
            "{rule:?} collapsed a self-contradicting declaration to T"
        );
        assert_eq!(artifact.reconciles, Hexavalent::Contradictory);
    }
}

#[test]
fn a_second_claim_site_that_agrees_is_not_a_contradiction() {
    // The discriminating negative: the mechanism must key on *disagreement*,
    // not on the mere presence of a second site. Without this, every document
    // that states a figure twice would be reported as contradicting itself.
    let root = stage_with_corroborating_claim("s10-agreeing", DECLARED_BYTES);
    let artifact = artifact(&root, RuleId::FullDigest);

    let row = artifact
        .rows
        .iter()
        .find(|row| row.name == SUBJECT)
        .expect("the declared row is reported");
    assert_eq!(row.corroborated_bytes, Some(DECLARED_BYTES));
    assert_eq!(row.state, Hexavalent::True);
    assert_eq!(artifact.reconciles, Hexavalent::True);
}

#[test]
fn the_approved_evidence_itself_carries_no_such_contradiction() {
    let artifact = artifact(&approved_evidence_root(), RuleId::FullDigest);
    assert_eq!(artifact.reconciles, Hexavalent::True);
    for row in &artifact.rows {
        assert_eq!(
            row.corroborated_bytes, None,
            "no second byte-count claim site exists for {} in the approved manifest",
            row.name
        );
    }
}
