//! Slice S6 — the corpus generator: totality first, one family end to end.
//!
//! Red-first discriminators:
//!
//! * **NC-24's `F-DEBRIS` limb** — the family enumerates **exactly one** site
//!   and stages **exactly one** fifteenth file. A generator that loops over the
//!   fourteen targets and emits one debris case per target reads as
//!   thoroughness, is unlicensed, and would inflate every frozen count by
//!   thirteen. A target is not a site;
//! * **NC-4** — a duplicate staged aggregate is *rejected*, naming the case it
//!   collides with, and **generation completes**. It is not an abort. Because a
//!   one-case family cannot collide with itself, the collision is injected;
//! * **NC-15** — every enumerated triple is resolved: `admitted + rejected ==
//!   enumerated`, with no triple left neither one nor the other.

mod support;

use ix_gaia_advisory::Hexavalent;

use support::corpus::{self, Family, Options, RejectionKind};
use support::{fresh_scratch, ordinal_file_names};

#[test]
fn nc_24_the_debris_family_enumerates_exactly_one_site() {
    let sites = corpus::enumerate(&Options::family(Family::Debris));
    assert_eq!(
        sites.len(),
        1,
        "NC-24: manifest L101 identifies one act on one stream; the site set is 1, not 14"
    );
    assert_eq!(sites[0].target, corpus::DEBRIS_TARGET);
    assert_eq!(
        sites[0].target, "gaia-s1-r2-validator.py",
        "the site binds to the validator program, whose stdout the citation names"
    );
    assert_ne!(
        sites[0].target, "gaia-s1-r2-validator-output.txt",
        "the captured-output row is the product of redirecting OUTSIDE the directory, \
         which the same citation endorses; it is not the debris act"
    );
}

#[test]
fn the_debris_case_stages_exactly_one_fifteenth_file_and_is_labelled_f() {
    let dest = fresh_scratch("s6-debris").join("corpus");
    let generated = corpus::generate(&dest, &Options::family(Family::Debris));

    assert_eq!(generated.enumerated, 1);
    assert_eq!(generated.admitted.len(), 1);
    assert!(generated.rejected.is_empty());

    let case = &generated.admitted[0];
    let staged = dest.join("dev").join(&case.case_id);
    let names = ordinal_file_names(&staged);
    assert_eq!(
        names.len(),
        15,
        "the staged root gains exactly one fifteenth file"
    );
    assert!(
        names.iter().any(|name| name == corpus::DEBRIS_STAGED_FILE),
        "the fifteenth file is {}, got {names:?}",
        corpus::DEBRIS_STAGED_FILE
    );
    assert_eq!(
        names
            .iter()
            .filter(|name| name.ends_with(".stdout"))
            .count(),
        1,
        "NC-24: no per-target debris enumeration exists"
    );

    let debris = std::fs::read(staged.join(corpus::DEBRIS_STAGED_FILE)).expect("readable");
    let source = std::fs::read(staged.join(corpus::DEBRIS_TARGET)).expect("readable");
    assert_eq!(debris.len(), corpus::DEBRIS_BYTES);
    assert_eq!(debris, source[..corpus::DEBRIS_BYTES].to_vec());

    assert_eq!(
        case.label,
        Hexavalent::False,
        "the oracle labels from the staged bytes; a surplus name does not reconcile"
    );
}

#[test]
fn nc_4_a_duplicate_staged_aggregate_is_rejected_and_generation_completes() {
    let dest = fresh_scratch("s6-duplicate").join("corpus");
    let generated = corpus::generate(
        &dest,
        &Options {
            families: vec![Family::Debris],
            inject_duplicate: true,
        },
    );

    assert_eq!(generated.enumerated, 2, "two triples were enumerated");
    assert_eq!(
        generated.admitted.len(),
        1,
        "one of the colliding pair survives"
    );
    assert_eq!(generated.rejected.len(), 1);

    let survivor = &generated.admitted[0].case_id;
    match &generated.rejected[0].kind {
        RejectionKind::DuplicateStagedAggregate { collides_with } => {
            assert_eq!(
                collides_with, survivor,
                "the rejection names the case it collides with"
            );
        }
        other => panic!("expected DuplicateStagedAggregate, got {other:?}"),
    }

    // Generation completed: the ledger exists and the surviving root is staged.
    assert!(dest.join("rejections.jsonl").exists());
    assert!(dest.join("dev").join(survivor).is_dir());
    assert_eq!(
        std::fs::read_dir(dest.join("dev"))
            .expect("readable")
            .count(),
        1,
        "the rejected case leaves no staged root behind"
    );
}

#[test]
fn nc_15_every_enumerated_triple_is_resolved() {
    for options in [
        Options::family(Family::Debris),
        Options {
            families: vec![Family::Debris],
            inject_duplicate: true,
        },
    ] {
        let dest = fresh_scratch("s6-totality").join("corpus");
        let generated = corpus::generate(&dest, &options);
        assert_eq!(
            generated.admitted.len() + generated.rejected.len(),
            generated.enumerated,
            "NC-15: a triple that is neither admitted nor rejected has vanished"
        );
    }
}

#[test]
fn the_corpus_digest_is_reproducible_and_path_independent() {
    let first = corpus::generate(
        &fresh_scratch("s6-digest-a").join("corpus"),
        &Options::family(Family::Debris),
    );
    let second = corpus::generate(
        &fresh_scratch("s6-digest-b").join("corpus"),
        &Options::family(Family::Debris),
    );
    assert_ne!(first.root, second.root);
    assert_eq!(
        first.digest, second.digest,
        "IX-AGG-1 embeds paths relative to the corpus root, so two locations lock identically"
    );
    assert_eq!(first.digest.len(), 64);
}

#[test]
fn the_case_id_salt_is_v4_and_a_site_set_change_would_change_every_identity() {
    assert_eq!(corpus::CASE_ID_SALT, "gaia-m3-case-v4");
    let real = corpus::case_id(
        Family::Debris,
        corpus::DEBRIS_TARGET,
        "validator-stdout-redirect",
        "abc",
    );
    let other_site = corpus::case_id(
        Family::Debris,
        corpus::DEBRIS_TARGET,
        "some-other-site",
        "abc",
    );
    assert_ne!(
        real, other_site,
        "a case is content-addressed, so it cannot be silently redefined"
    );
}
