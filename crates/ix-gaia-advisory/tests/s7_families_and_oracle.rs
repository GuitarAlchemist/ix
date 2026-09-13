//! Slice S7 — the remaining three families, the positive control, and the oracle.
//!
//! This slice is where the preregistered predictions become checkable, so it is
//! also where they can be **wrong**. Two falsifiers are evaluated here and a
//! failure of either stops the run rather than being reported around:
//!
//! * **X2** — the corpus admits exactly **25** cases and rejects exactly **1**.
//!   Any other pair means the enumeration is wrong;
//! * **X3** — the truth balance is exactly **1 : 24**. Any other truth-`T` case
//!   would be a *welcome* falsification of the degeneracy claim, and it would
//!   still stop the run.
//!
//! Controls: NC-22 (the oracle cannot see the family), NC-23 (no implementer
//! discretion survives into a staged case), NC-24 (a token is not a site, and a
//! target is not a site).

mod support;

use ix_gaia_advisory::Hexavalent;

use support::corpus::{self, Family, Options, RejectionKind};
use support::{fresh_scratch, ordinal_file_names};

const SPEC: &str = "gaia-consolidated-mission-room-factory-spec-v0.2.md";
const LEDGER: &str = "gaia-s1-r2-change-ledger.md";
const PURE_CRLF_FILE: &str = "gaia-s1-r2-validator-output.txt";

/// §10.5.1 — cited line, edit byte offset, original byte, replacement byte.
const RESTAMP_EDITS: [(u32, usize, u8, u8); 7] = [
    (10, 585, b'1', b'2'),
    (16, 2_257, b'3', b'4'),
    (18, 5_897, b'1', b'2'),
    (474, 54_292, b'1', b'2'),
    (484, 55_604, b'1', b'2'),
    (571, 68_932, b'3', b'4'),
    (574, 73_412, b'4', b'5'),
];

/// §10.5.2 — cited 1-based ledger line, and every `210` token offset in it.
const STALE_SITES: [(u32, &[usize]); 3] = [
    (552, &[117_421]),
    (556, &[119_504]),
    (646, &[145_193, 145_448]),
];

/// The occurrence that is a spec line range, not a census claim. It models no
/// documented drift shape and must appear in no site set.
const UNLICENSED_TOKEN_OFFSET: usize = 194_529;

fn full() -> Options {
    Options {
        families: vec![
            Family::Crlf,
            Family::Debris,
            Family::Restamp,
            Family::StaleFigure,
            Family::Pristine,
        ],
        inject_duplicate: false,
    }
}

// ---------------------------------------------------------------- cardinalities

#[test]
fn nc_24_every_family_enumerates_exactly_its_citations_cardinality() {
    let sites = corpus::enumerate(&full());
    let count = |family: Family| sites.iter().filter(|s| s.family == family).count();

    // manifest L105 / L107-108 — a census over all fourteen files.
    assert_eq!(count(Family::Crlf), 14);
    // manifest L101 — one act, one stream, one fifteenth file.
    assert_eq!(count(Family::Debris), 1);
    // manifest L63 — seven cited lines.
    assert_eq!(count(Family::Restamp), 7);
    // manifest L112 — three cited census claims.
    assert_eq!(count(Family::StaleFigure), 3);
    // the unmodified root.
    assert_eq!(count(Family::Pristine), 1);

    assert_eq!(sites.len(), 26, "14 + 1 + 7 + 3 + 1");
}

#[test]
fn nc_24_the_stale_figure_site_set_is_three_claims_and_excludes_the_line_range() {
    let sites = corpus::enumerate(&full());
    let stale: Vec<&str> = sites
        .iter()
        .filter(|s| s.family == Family::StaleFigure)
        .map(|s| s.site.as_str())
        .collect();
    assert_eq!(stale, vec!["ledger:552", "ledger:556", "ledger:646"]);
    for site in &stale {
        assert!(
            !site.contains(&UNLICENSED_TOKEN_OFFSET.to_string()),
            "NC-24: the 210-238 line range must appear in no site set"
        );
    }

    // And the offsets the generator actually rewrites exclude it too.
    let touched = corpus::stale_figure_token_offsets();
    assert_eq!(touched, vec![117_421, 119_504, 145_193, 145_448]);
    assert!(
        !touched.contains(&UNLICENSED_TOKEN_OFFSET),
        "NC-24: a token match is not a site"
    );
}

// ---------------------------------------------------------------- X2 and X3

#[test]
fn x2_and_x3_the_preregistered_counts_hold() {
    let dest = fresh_scratch("s7-full").join("corpus");
    let generated = corpus::generate(&dest, &full());

    assert_eq!(generated.enumerated, 26, "X2: enumerated triples");
    assert_eq!(
        (generated.admitted.len(), generated.rejected.len()),
        (25, 1),
        "X2: the corpus must admit 25 and reject 1; any other pair means this enumeration is wrong"
    );
    assert_eq!(
        generated.admitted.len() + generated.rejected.len(),
        generated.enumerated,
        "NC-15 over the full enumeration"
    );

    // The single rejection is measured, not assumed.
    let rejection = &generated.rejected[0];
    assert_eq!(rejection.family, Family::Crlf);
    assert_eq!(rejection.target, PURE_CRLF_FILE);
    assert_eq!(rejection.kind, RejectionKind::NotApplicable("NoBareLf"));

    let (t, f) = generated.truth_balance();
    assert_eq!(
        (t, f),
        (1, 24),
        "X3: the truth balance is 1 : 24; another truth-T case would be a welcome falsification"
    );
    let positives: Vec<&str> = generated
        .admitted
        .iter()
        .filter(|case| case.label == Hexavalent::True)
        .map(|case| case.family.id())
        .collect();
    assert_eq!(
        positives,
        vec!["F-PRISTINE"],
        "the only truth-T case is the positive control"
    );
}

#[test]
fn x1_no_admitted_case_contradicts_its_familys_documented_direction() {
    // Each drift family models a shape the approved evidence documents as
    // producing a root that does **not** reconcile; the positive control models
    // the one that does. A case admitted against its family's direction means
    // the citation licence has been violated, and the run stops rather than
    // reporting around it.
    let dest = fresh_scratch("s7-x1").join("corpus");
    let generated = corpus::generate(&dest, &full());

    for case in &generated.admitted {
        let expected = if case.family == Family::Pristine {
            Hexavalent::True
        } else {
            Hexavalent::False
        };
        assert_eq!(
            case.label,
            expected,
            "X1: {} at {} was admitted with a label contradicting its family's documented direction",
            case.family.id(),
            case.site
        );
    }
}

#[test]
fn the_pure_crlf_file_is_recorded_not_mislabelled_and_not_aborted_on() {
    let dest = fresh_scratch("s7-crlf").join("corpus");
    let generated = corpus::generate(&dest, &Options::family(Family::Crlf));
    assert_eq!(generated.enumerated, 14);
    assert_eq!(generated.admitted.len(), 13);
    assert_eq!(generated.rejected.len(), 1);
    assert!(
        dest.join("rejections.jsonl").exists(),
        "a rejection is evidence, and it is part of the corpus tree"
    );
    // Measured premise: the file carries 214 CRLF pairs and no bare LF.
    let bytes =
        std::fs::read(support::approved_evidence_root().join(PURE_CRLF_FILE)).expect("read");
    let cr = bytes.iter().filter(|b| **b == b'\r').count();
    let lf = bytes.iter().filter(|b| **b == b'\n').count();
    let bare = bytes
        .iter()
        .enumerate()
        .filter(|(i, b)| **b == b'\n' && (*i == 0 || bytes[i - 1] != b'\r'))
        .count();
    assert_eq!((cr, lf, bare), (214, 214, 0));
}

// ---------------------------------------------------------------- NC-23

#[test]
fn nc_23_every_restamp_diff_is_exactly_the_tabulated_byte_and_nothing_else() {
    let dest = fresh_scratch("s7-restamp").join("corpus");
    let generated = corpus::generate(&dest, &Options::family(Family::Restamp));
    assert_eq!(generated.admitted.len(), 7);

    let pristine = std::fs::read(support::approved_evidence_root().join(SPEC)).expect("read");
    assert_eq!(pristine.len(), 77_673);

    for (line, offset, original, replacement) in RESTAMP_EDITS {
        let case = generated
            .admitted
            .iter()
            .find(|case| case.site == format!("spec-v0.2:{line}"))
            .unwrap_or_else(|| panic!("case for spec-v0.2:{line}"));
        let staged =
            std::fs::read(dest.join("dev").join(&case.case_id).join(SPEC)).expect("read staged");

        assert_eq!(
            pristine[offset], original,
            "premise: original byte at {offset}"
        );
        assert_eq!(
            staged[offset], replacement,
            "the tabulated replacement byte"
        );

        let differing: Vec<usize> = (0..pristine.len())
            .filter(|i| pristine[*i] != staged[*i])
            .collect();
        assert_eq!(
            differing,
            vec![offset],
            "the diff must be exactly one byte, at the tabulated offset, and nothing else"
        );

        // The binding line-count invariant.
        assert_eq!(staged.len(), 77_673);
        assert_eq!(staged.iter().filter(|b| **b == b'\n').count(), 579);
        assert_eq!(staged.iter().filter(|b| **b == b'\r').count(), 0);
    }
}

#[test]
fn nc_23_every_stale_figure_diff_is_exactly_the_tabulated_tokens() {
    let dest = fresh_scratch("s7-stale").join("corpus");
    let generated = corpus::generate(&dest, &Options::family(Family::StaleFigure));
    assert_eq!(generated.admitted.len(), 3);

    let pristine = std::fs::read(support::approved_evidence_root().join(LEDGER)).expect("read");
    assert_eq!(pristine.len(), 206_581);

    for (line, offsets) in STALE_SITES {
        let case = generated
            .admitted
            .iter()
            .find(|case| case.site == format!("ledger:{line}"))
            .unwrap_or_else(|| panic!("case for ledger:{line}"));
        let staged =
            std::fs::read(dest.join("dev").join(&case.case_id).join(LEDGER)).expect("read staged");

        let differing: Vec<usize> = (0..pristine.len())
            .filter(|i| pristine[*i] != staged[*i])
            .collect();
        // `210` -> `214` changes only the third byte of each token.
        let expected: Vec<usize> = offsets.iter().map(|offset| offset + 2).collect();
        assert_eq!(
            differing, expected,
            "ledger:{line} must rewrite exactly its own tokens"
        );
        for offset in offsets {
            assert_eq!(&pristine[*offset..offset + 3], b"210");
            assert_eq!(&staged[*offset..offset + 3], b"214");
        }

        assert_eq!(staged.len(), 206_581);
        assert_eq!(staged.iter().filter(|b| **b == b'\n').count(), 913);
        assert_eq!(staged.iter().filter(|b| **b == b'\r').count(), 0);
    }
}

// ---------------------------------------------------------------- the positive control

#[test]
fn the_pristine_control_stages_the_unmodified_root_and_is_labelled_t() {
    let dest = fresh_scratch("s7-pristine").join("corpus");
    let generated = corpus::generate(&dest, &Options::family(Family::Pristine));
    assert_eq!(
        generated.admitted.len(),
        1,
        "the control is not a no-op rejection"
    );

    let case = &generated.admitted[0];
    assert_eq!(case.label, Hexavalent::True);
    let staged = dest.join("dev").join(&case.case_id);
    assert_eq!(ordinal_file_names(&staged).len(), 14);
    for (name, bytes) in corpus::pristine_files() {
        assert_eq!(
            std::fs::read(staged.join(&name)).expect("read"),
            bytes,
            "{name} is staged unmodified"
        );
    }
}

// ---------------------------------------------------------------- NC-22

#[test]
fn nc_22_the_oracle_cannot_see_the_family() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("support")
            .join("oracle.rs"),
    )
    .expect("the oracle source is readable");

    let start = text
        .find("pub fn label(")
        .expect("the oracle exposes exactly one labelling entry point");
    let signature = &text[start..start + text[start..].find(')').expect("signature closes") + 1];

    for forbidden in ["family", "site", "rule", "Family", "Site", "RuleId"] {
        assert!(
            !signature.contains(forbidden),
            "NC-22: the oracle signature mentions {forbidden:?}: {signature}"
        );
    }
    assert!(signature.contains("staged_root"));
    assert!(signature.contains("frozen_manifest"));
    assert_eq!(
        text.matches("pub fn ").count(),
        1,
        "the oracle exposes one function, so there is no second, family-aware path"
    );
}

#[test]
fn the_oracle_labels_from_the_staged_bytes_not_from_family_membership() {
    // Same family, two staged roots: one perturbed, one not. The oracle must
    // disagree about them, which a family-keyed labeller could not.
    let pristine = corpus::pristine_files();
    let frozen = pristine
        .get(support::MANIFEST_FILE_NAME)
        .expect("manifest")
        .clone();

    let dir = fresh_scratch("s7-oracle");
    let untouched = dir.join("untouched");
    support::stage_pristine(&untouched);
    assert_eq!(
        support::oracle::label(&untouched, support::MANIFEST_FILE_NAME, &frozen),
        Hexavalent::True
    );

    let perturbed = dir.join("perturbed");
    support::stage_pristine(&perturbed);
    let target = perturbed.join(SPEC);
    let mut bytes = std::fs::read(&target).expect("read");
    bytes[585] = b'2';
    std::fs::write(&target, &bytes).expect("write");
    assert_eq!(
        support::oracle::label(&perturbed, support::MANIFEST_FILE_NAME, &frozen),
        Hexavalent::False
    );

    // And a perturbation of the manifest itself is still measured against the
    // frozen original, so it cannot rewrite its own ground truth.
    let self_rewriting = dir.join("self-rewriting");
    support::stage_pristine(&self_rewriting);
    let manifest = self_rewriting.join(support::MANIFEST_FILE_NAME);
    let mut bytes = std::fs::read(&manifest).expect("read");
    bytes.extend_from_slice(b"\n");
    std::fs::write(&manifest, &bytes).expect("write");
    assert_eq!(
        support::oracle::label(&self_rewriting, support::MANIFEST_FILE_NAME, &frozen),
        Hexavalent::False
    );
}
