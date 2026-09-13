//! Slice S9 — determinism, and the name-handling defects a lossy decode hides.
//!
//! Red-first discriminators:
//!
//! * **NC-5** — the same bytes produce a **byte-identical** report across cell
//!   orderings, traversal orders and thread counts. Ordering that leaks into
//!   output is the classic way a "deterministic" artifact stops being one;
//! * **NC-6** — two names differing only in case are reported as two distinct
//!   names or refused, **never silently merged**; a name that is not valid
//!   UTF-8 is a refusal, not a lossily-decoded string that collides with a real
//!   one; and locale and timezone move nothing, because no emitted value is
//!   locale-dependent and no wall-clock value is emitted at all.

mod support;

use std::sync::{Mutex, OnceLock};

use ix_gaia_advisory::{
    advise, evaluate, AdvisoryRefusal, AdvisoryRequest, EvalRequest, EvidenceClass, RuleId,
    RuleWidth,
};

use support::corpus::{self, Family, Options};
use support::{approved_evidence_root, fresh_scratch, stage_pristine, MANIFEST_FILE_NAME};

/// Serialises the process-global environment mutations NC-6's locale limb needs.
fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn cells() -> Vec<RuleWidth> {
    vec![
        RuleWidth {
            rule: RuleId::StructuralOnly,
            window_bytes: 0,
        },
        RuleWidth {
            rule: RuleId::WindowProbe,
            window_bytes: 1024,
        },
        RuleWidth {
            rule: RuleId::FullDigest,
            window_bytes: 0,
        },
    ]
}

fn full_corpus(slice: &str) -> corpus::Corpus {
    corpus::generate(
        &fresh_scratch(slice).join("corpus"),
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
    )
}

fn request(corpus: &corpus::Corpus, cells: Vec<RuleWidth>) -> EvalRequest {
    EvalRequest {
        corpus_root: corpus.root.clone(),
        expected_corpus_digest_ix_agg_1: corpus.digest.clone(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        cells,
        window_reference_root: approved_evidence_root(),
        evidence_class: EvidenceClass::Development,
    }
}

// ---------------------------------------------------------------- NC-5

#[test]
fn nc_5_cell_order_traversal_order_and_thread_count_move_no_byte() {
    let corpus = full_corpus("s9-nc5");

    let forward = evaluate(&request(&corpus, cells())).expect("binds");

    let mut reversed = cells();
    reversed.reverse();
    let reverse = evaluate(&request(&corpus, reversed)).expect("binds");

    let mut rotated = cells();
    rotated.rotate_left(1);
    let rotate = evaluate(&request(&corpus, rotated)).expect("binds");

    assert_eq!(
        forward.to_canonical_json(),
        reverse.to_canonical_json(),
        "NC-5: reversing the requested cells changed the report"
    );
    assert_eq!(
        forward.to_canonical_json(),
        rotate.to_canonical_json(),
        "NC-5: rotating the requested cells changed the report"
    );

    // And concurrently, against the same on-disk corpus.
    let expected = forward.to_canonical_json();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let request = request(&corpus, cells());
            std::thread::spawn(move || evaluate(&request).expect("binds").to_canonical_json())
        })
        .collect();
    for handle in handles {
        assert_eq!(
            handle.join().expect("the thread completed"),
            expected,
            "NC-5: a concurrent run produced a different report"
        );
    }
}

#[test]
fn nc_5_the_corpus_locks_identically_from_two_locations() {
    let a = full_corpus("s9-lock-a");
    let b = full_corpus("s9-lock-b");
    assert_ne!(a.root, b.root);
    assert_eq!(a.digest, b.digest);
    assert_eq!(
        a.admitted
            .iter()
            .map(|case| (case.case_id.as_str(), case.label))
            .collect::<Vec<_>>(),
        b.admitted
            .iter()
            .map(|case| (case.case_id.as_str(), case.label))
            .collect::<Vec<_>>(),
        "sealed labels and case identities are a function of the bytes, not of the path"
    );
}

// ---------------------------------------------------------------- NC-6, locale

#[test]
fn nc_6_locale_and_timezone_move_no_byte() {
    let _guard = env_lock().lock().expect("the env lock is not poisoned");
    let root = approved_evidence_root();
    let build = || {
        advise(&AdvisoryRequest {
            evidence_root: root.clone(),
            manifest_file_name: MANIFEST_FILE_NAME.to_string(),
            rule: RuleId::FullDigest,
            window_bytes: 0,
            window_reference_root: None,
            expected_provenance: None,
        })
        .expect("binds")
        .to_canonical_json()
    };

    let mut emitted = Vec::new();
    for (tz, locale) in [("UTC", "C"), ("Pacific/Kiritimati", "tr_TR.ISO-8859-9")] {
        std::env::set_var("TZ", tz);
        std::env::set_var("LC_ALL", locale);
        std::env::set_var("LANG", locale);
        emitted.push(build());
    }
    std::env::remove_var("TZ");
    std::env::remove_var("LC_ALL");
    std::env::remove_var("LANG");

    assert_eq!(
        emitted[0], emitted[1],
        "NC-6: the artifact must be byte-identical across locales and timezones"
    );
    // The `tr_TR` pair is chosen deliberately: it is the locale whose
    // case-folding of `I` breaks naive ASCII assumptions. Nothing here folds
    // case, and this asserts that stays true.
    assert!(
        !emitted[0].contains("\"timestamp\"") && !emitted[0].contains("\"generated_at\""),
        "no wall-clock value appears in any emitted artifact"
    );
}

// ---------------------------------------------------------------- NC-6, names

#[test]
fn nc_6_a_case_colliding_pair_is_reported_as_two_names_never_merged() {
    let dir = fresh_scratch("s9-case-collision");
    let root = dir.join("root");
    stage_pristine(&root);

    let declared = "gaia-uncertainty-grammar-v0.1.ebnf";
    let colliding = "GAIA-UNCERTAINTY-GRAMMAR-V0.1.EBNF";
    std::fs::rename(root.join(declared), root.join(colliding)).expect("renameable");

    let artifact = advise(&AdvisoryRequest {
        evidence_root: root.clone(),
        manifest_file_name: MANIFEST_FILE_NAME.to_string(),
        rule: RuleId::StructuralOnly,
        window_bytes: 0,
        window_reference_root: None,
        expected_provenance: None,
    })
    .expect("binds");

    assert!(
        artifact.root.missing_names.contains(&declared.to_string()),
        "NC-6: the declared name must be reported missing, not silently matched \
         against a differently-cased entry; got {:?}",
        artifact.root.missing_names
    );
    assert!(
        artifact
            .root
            .unlisted_names
            .iter()
            .any(|name| name.eq_ignore_ascii_case(declared) && name != declared),
        "NC-6: the differently-cased entry must be reported as its own name; got {:?}",
        artifact.root.unlisted_names
    );
    assert_eq!(
        artifact.reconciles,
        ix_gaia_advisory::Hexavalent::False,
        "a root whose names do not match the declaration does not reconcile"
    );

    let row = artifact
        .rows
        .iter()
        .find(|row| row.name == declared)
        .expect("the declared row is still reported");
    assert_eq!(
        row.measured_bytes, None,
        "NC-6: an unmatched declared name measures null, never the colliding file's length"
    );
}

#[cfg(windows)]
#[test]
fn nc_6_a_name_that_is_not_valid_utf8_is_a_refusal() {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    let dir = fresh_scratch("s9-non-utf8");
    let root = dir.join("root");
    stage_pristine(&root);

    // An unpaired high surrogate: valid UTF-16, not representable in UTF-8.
    let name = OsString::from_wide(&[0xD800, 0x002E, 0x0074, 0x0078, 0x0074]);
    assert!(
        name.to_str().is_none(),
        "the test's own premise: this name is not valid UTF-8"
    );

    match std::fs::write(root.join(&name), b"surplus") {
        Ok(()) => {
            match advise(&AdvisoryRequest {
                evidence_root: root.clone(),
                manifest_file_name: MANIFEST_FILE_NAME.to_string(),
                rule: RuleId::StructuralOnly,
                window_bytes: 0,
                window_reference_root: None,
                expected_provenance: None,
            }) {
                Err(AdvisoryRefusal::NonUtf8EntryName { .. }) => {}
                Err(other) => panic!("expected NonUtf8EntryName, got {other:?}"),
                Ok(artifact) => panic!(
                    "NC-6 violated: a non-UTF-8 name was lossily decoded rather than refused: {:?}",
                    artifact.root
                ),
            }
        }
        Err(error) => panic!(
            "NC-6's non-UTF-8 limb could not be staged on this filesystem ({error}); \
             the refusal path is therefore unexercised and must not be reported as green"
        ),
    }
}
