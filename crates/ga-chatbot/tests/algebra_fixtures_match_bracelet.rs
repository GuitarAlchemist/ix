//! Verify that every "algebra" stub fixture answer agrees with the algebraic
//! ground truth from `ix-bracelet`.
//!
//! The fixture answers under the `algebra-*` corpus were derived offline from
//! `ix-bracelet`'s ICV / Grothendieck-delta / orbit machinery. If the
//! computation drifts (a bug lands in `ix_bracelet::icv`, the prime-form
//! algorithm regresses, or an editor changes a fixture string with the wrong
//! number), this test catches the divergence before it reaches a live LLM
//! evaluation cycle.
//!
//! Each assertion checks specific numerical claims embedded in the fixture
//! answer string against what `ix-bracelet` currently computes. It does not
//! lock the prose — only the algebraic facts.

use ga_chatbot::load_fixtures;
use ix_bracelet::{bracelet_prime_form, grothendieck_delta, icv, z_related_pairs, PcSet};
use std::path::Path;

fn fixtures() -> std::collections::HashMap<String, ga_chatbot::ChatbotResponse> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root from crate manifest");
    let path = workspace.join("tests/adversarial/fixtures/stub-responses.jsonl");
    load_fixtures(&path)
}

fn fixture_for(prefix: &str) -> ga_chatbot::ChatbotResponse {
    fixtures()
        .get(prefix)
        .cloned()
        .unwrap_or_else(|| panic!("missing fixture for prefix '{prefix}'"))
}

#[test]
fn algebra_001_major_minor_triads_share_set_class() {
    // Answer claims both triads are Forte 3-11 and share ICV [0,0,1,1,1,0].
    // Verify via ix-bracelet.
    let c_maj = PcSet::from_pcs([0u8, 4, 7]);
    let c_min = PcSet::from_pcs([0u8, 3, 7]);
    assert_eq!(bracelet_prime_form(c_maj), bracelet_prime_form(c_min));
    assert_eq!(icv(c_maj).data, [0, 0, 1, 1, 1, 0]);
    assert_eq!(icv(c_maj), icv(c_min));

    let resp = fixture_for("are c major and c minor triads in the same set class");
    assert!(resp.answer.contains("3-11"), "answer should cite Forte 3-11");
    assert!(
        resp.answer.contains("[0, 0, 1, 1, 1, 0]"),
        "answer should cite the ICV [0, 0, 1, 1, 1, 0]"
    );
}

#[test]
fn algebra_002_c_major_scale_icv() {
    let c_major = PcSet::from_pcs([0u8, 2, 4, 5, 7, 9, 11]);
    assert_eq!(icv(c_major).data, [2, 5, 4, 3, 6, 1]);

    let resp = fixture_for("what is the interval-class vector of the c major scale");
    assert!(resp.answer.contains("[2, 5, 4, 3, 6, 1]"));
}

#[test]
fn algebra_003_major_and_natural_minor_share_icv() {
    let c_major = PcSet::from_pcs([0u8, 2, 4, 5, 7, 9, 11]);
    let c_nat_min = PcSet::from_pcs([0u8, 2, 3, 5, 7, 8, 10]);
    assert_eq!(icv(c_major), icv(c_nat_min));
    assert!(grothendieck_delta(c_major, c_nat_min).is_zero());

    let resp = fixture_for(
        "are c major and c natural minor scales harmonically equivalent in interval content",
    );
    assert!(
        resp.answer.contains("[2, 5, 4, 3, 6, 1]"),
        "answer should cite the shared ICV"
    );
    assert!(
        resp.answer.to_lowercase().contains("zero"),
        "answer should mention the zero delta"
    );
}

#[test]
fn algebra_004_g7_db7_tritone_substitution() {
    // G7 = G B D F = {7, 11, 2, 5}; Db7 = Db F Ab Cb = {1, 5, 8, 11}
    let g7 = PcSet::from_pcs([7u8, 11, 2, 5]);
    let db7 = PcSet::from_pcs([1u8, 5, 8, 11]);
    assert_eq!(bracelet_prime_form(g7), bracelet_prime_form(db7));
    assert_eq!(icv(g7), icv(db7));
    assert_eq!(icv(g7).data, [0, 1, 2, 1, 1, 1]);
    assert!(grothendieck_delta(g7, db7).is_zero());

    let resp = fixture_for("is db7 a valid substitution for g7");
    assert!(resp.answer.contains("4-27"), "answer should cite Forte 4-27");
    assert!(
        resp.answer.contains("[0, 1, 2, 1, 1, 1]"),
        "answer should cite the dominant-7th ICV"
    );
}

#[test]
fn algebra_005_z_pair_4z15_4z29_distinct() {
    // 4-Z15 = {0,1,4,6}, 4-Z29 = {0,1,3,7} — same ICV, different orbits.
    let z15 = PcSet::from_pcs([0u8, 1, 4, 6]);
    let z29 = PcSet::from_pcs([0u8, 1, 3, 7]);
    assert_eq!(icv(z15).data, [1, 1, 1, 1, 1, 1]);
    assert_eq!(icv(z15), icv(z29));
    assert_ne!(bracelet_prime_form(z15), bracelet_prime_form(z29));

    // And verify the pair is actually in our enumerated z_related_pairs list.
    let z15_pf = bracelet_prime_form(z15);
    let z29_pf = bracelet_prime_form(z29);
    let listed = z_related_pairs()
        .iter()
        .any(|(a, b)| (*a == z15_pf && *b == z29_pf) || (*a == z29_pf && *b == z15_pf));
    assert!(listed, "4-Z15/4-Z29 should appear in z_related_pairs()");

    let resp = fixture_for("are forte 4-z15 and 4-z29 equivalent set classes");
    assert!(
        resp.answer.contains("[1, 1, 1, 1, 1, 1]"),
        "answer should cite the all-interval ICV"
    );
}

#[test]
fn algebra_006_aug_dim_distance_is_l1_6() {
    let c_aug = PcSet::from_pcs([0u8, 4, 8]);
    let c_dim = PcSet::from_pcs([0u8, 3, 6]);
    assert_eq!(icv(c_aug).data, [0, 0, 0, 3, 0, 0]);
    assert_eq!(icv(c_dim).data, [0, 0, 2, 0, 0, 1]);
    let delta = grothendieck_delta(c_aug, c_dim);
    assert_eq!(delta.data, [0, 0, 2, -3, 0, 1]);
    assert_eq!(delta.l1_norm(), 6);

    let resp = fixture_for(
        "what is the harmonic distance between c augmented and c diminished triads",
    );
    assert!(resp.answer.contains("[0, 0, 0, 3, 0, 0]"));
    assert!(resp.answer.contains("[0, 0, 2, 0, 0, 1]"));
    assert!(
        resp.answer.contains("is 6"),
        "answer should cite L1 distance of 6"
    );
}
