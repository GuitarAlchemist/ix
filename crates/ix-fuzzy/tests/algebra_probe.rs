//! Randomized + adversarial probes of the algebraic claims made by
//! `ix_fuzzy::observations::merge`.
//!
//! The in-module `proof_*` tests each check ONE hand-picked instance.
//! These probes check the same claims over thousands of seeded-random
//! inputs and a few adversarial constructions. Each test asserts the
//! DOCUMENTED claim (module comments: "restores associativity",
//! "load-bearing for CRDT correctness", "reproducible across runs
//! regardless of input order") — a failure here is a counterexample to
//! the documentation, not necessarily to the intended design.
//!
//! Ported from hari-lattice's `tests/algebra_probe.rs` (seeds and trial
//! counts kept identical so results are comparable across repos) as part
//! of the hex-merge purity fix — see GuitarAlchemist/hari#27 and hari
//! commit 924b5a4.
//!
//! Deterministic: fixed-seed xorshift, no external deps.

use ix_fuzzy::observations::{merge, merge_all, HexObservation, MergedState};
use ix_types::Hexavalent;

// ───────────────────────── tiny deterministic RNG ─────────────────────────

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const SOURCES: [&str; 4] = ["tars", "ix", "ga", "demerzel"];
const CLAIMS: [&str; 5] = [
    "k1::valuable",
    "k1::safe",
    "k2::valuable",
    "k2::cheap",
    "k3",
];
const VARIANTS: [Hexavalent; 6] = [
    Hexavalent::True,
    Hexavalent::Probable,
    Hexavalent::Unknown,
    Hexavalent::Doubtful,
    Hexavalent::False,
    Hexavalent::Contradictory,
];

fn rand_obs(rng: &mut Rng) -> HexObservation {
    HexObservation {
        source: SOURCES[rng.below(4) as usize].to_string(),
        diagnosis_id: format!("d{}", rng.below(4)),
        round: rng.below(6) as u32,
        ordinal: rng.below(3) as u32,
        claim_key: CLAIMS[rng.below(5) as usize].to_string(),
        variant: VARIANTS[rng.below(6) as usize],
        // Weights quantized to 1/16 steps in (0,1] so equality is exact.
        weight: (rng.below(16) + 1) as f64 / 16.0,
        evidence: None,
    }
}

/// Random multiset with DISTINCT dedup keys (well-formed input).
fn rand_wellformed(rng: &mut Rng, n: usize) -> Vec<HexObservation> {
    let mut out: Vec<HexObservation> = Vec::new();
    'outer: for _ in 0..n * 3 {
        if out.len() == n {
            break;
        }
        let o = rand_obs(rng);
        for e in &out {
            if e.dedup_key() == o.dedup_key() {
                continue 'outer;
            }
        }
        out.push(o);
    }
    out
}

fn shuffle(rng: &mut Rng, v: &mut [HexObservation]) {
    for i in (1..v.len()).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        v.swap(i, j);
    }
}

fn states_equal(a: &MergedState, b: &MergedState) -> bool {
    if a.observations != b.observations || a.contradictions != b.contradictions {
        return false;
    }
    VARIANTS
        .iter()
        .all(|v| (a.distribution.get(v) - b.distribution.get(v)).abs() < 1e-12)
}

// ───────────────────────── probes ─────────────────────────

/// THEOREM 1 (pinned): output is "reproducible across runs regardless
/// of input order" — TRUE for well-formed input (distinct dedup keys).
/// Checked over 2000 random multisets. See
/// `theorem_key_collision_resolves_order_independent` for the boundary
/// that once diverged and is now fixed.
#[test]
fn probe_permutation_invariance_wellformed() {
    let mut rng = Rng(0xDEADBEEF);
    for trial in 0..2000 {
        let base = rand_wellformed(&mut rng, 8);
        let mut perm = base.clone();
        shuffle(&mut rng, &mut perm);
        let sa = merge_all(&base).unwrap();
        let sb = merge_all(&perm).unwrap();
        assert!(
            states_equal(&sa, &sb),
            "permutation changed merge output (trial {trial})\nbase: {base:#?}"
        );
    }
}

/// THEOREM 4 (pinned; was KNOWN DIVERGENCE #1, fixed): colliding dedup
/// keys with divergent payloads resolve order-independently. A source
/// that asserts two different variants in the same observation slot
/// has contradicted itself, and the resolution is `Contradictory` at
/// the minimum weight — irreconcilable evidence is preserved, not
/// silently tie-broken (the hexavalent preservation ethos). Same-variant
/// collisions keep the variant at minimum (conservative) weight.
///
/// NOTE: this deliberately diverges from Demerzel-canonical /
/// first-write-wins semantics until the fix is propagated.
#[test]
fn theorem_key_collision_resolves_order_independent() {
    let a = HexObservation {
        source: "tars".into(),
        diagnosis_id: "d0".into(),
        round: 0,
        ordinal: 0,
        claim_key: "k1::valuable".into(),
        variant: Hexavalent::True,
        weight: 1.0,
        evidence: None,
    };
    let b = HexObservation {
        variant: Hexavalent::False, // same key, opposite verdict
        weight: 0.5,
        ..a.clone()
    };
    let ab = merge_all(&[a.clone(), b.clone()]).unwrap();
    let ba = merge_all(&[b.clone(), a.clone()]).unwrap();
    assert!(
        states_equal(&ab, &ba),
        "key-collision resolution is order-dependent again"
    );
    // Divergent variants → self-conflict C at min weight.
    assert_eq!(ab.observations[0].variant, Hexavalent::Contradictory);
    assert_eq!(ab.observations[0].weight, 0.5);

    // Same variant, divergent weight → variant kept at min weight.
    let b2 = HexObservation {
        weight: 0.25,
        ..a.clone()
    };
    let same = merge_all(&[a.clone(), b2.clone()]).unwrap();
    let same_rev = merge_all(&[b2, a]).unwrap();
    assert!(states_equal(&same, &same_rev));
    assert_eq!(same.observations[0].variant, Hexavalent::True);
    assert_eq!(same.observations[0].weight, 0.25);
}

/// THEOREM 5 (pinned): permutation invariance and associativity hold
/// UNCONDITIONALLY — no well-formedness precondition — now that
/// key collisions resolve via an ACI fold. Random observations with
/// no key-distinctness filtering at all: intra-set and cross-set
/// collisions, divergent payloads, everything representable.
#[test]
fn theorem_unconditional_invariance_and_associativity() {
    let mut rng = Rng(0x5EED);
    for trial in 0..1500 {
        // Raw random observations — collisions everywhere.
        let a: Vec<_> = (0..4).map(|_| rand_obs(&mut rng)).collect();
        let b: Vec<_> = (0..4).map(|_| rand_obs(&mut rng)).collect();
        let c: Vec<_> = (0..4).map(|_| rand_obs(&mut rng)).collect();

        let flat: Vec<_> = a.iter().chain(b.iter()).chain(c.iter()).cloned().collect();
        let s_flat = merge_all(&flat).unwrap();

        // Permutation of the flat set.
        let mut perm = flat.clone();
        shuffle(&mut rng, &mut perm);
        assert!(
            states_equal(&s_flat, &merge_all(&perm).unwrap()),
            "permutation changed output under collisions (trial {trial})"
        );

        // Carried re-merge, both groupings.
        let ab_state = merge_all(&a.iter().chain(b.iter()).cloned().collect::<Vec<_>>()).unwrap();
        let left: Vec<_> = ab_state
            .observations
            .iter()
            .chain(c.iter())
            .cloned()
            .collect();
        assert!(
            states_equal(&merge_all(&left).unwrap(), &s_flat),
            "left-carried != flat under collisions (trial {trial})"
        );

        let bc_state = merge_all(&b.iter().chain(c.iter()).cloned().collect::<Vec<_>>()).unwrap();
        let right: Vec<_> = a
            .iter()
            .cloned()
            .chain(bc_state.observations.iter().cloned())
            .collect();
        assert!(
            states_equal(&merge_all(&right).unwrap(), &s_flat),
            "right-carried != flat under collisions (trial {trial})"
        );
    }
}

/// THEOREM 2 (pinned): the content-derived synthesis-id design
/// ("the property that restores associativity",
/// `synthesis_diagnosis_id` doc) — TRUE conditional on globally
/// distinct dedup keys. Checked over 1000 random triples: carrying
/// merge(A∪B).observations into a merge with C equals the flat
/// merge(A∪B∪C), in both groupings.
///
/// Before the key-collision fix the UNCONDITIONAL claim was false: the
/// same probe with dedup keys distinct only within each set (collisions
/// possible across sets) failed. That break was entirely downstream of
/// the key-collision defect (divergence #1): an inner merge synthesized
/// `C` from a payload that lost its dedup battle in the flat ordering,
/// and the carried synthesis had no surviving derivation. One defect,
/// two symptoms — see `theorem_unconditional_invariance_and_associativity`
/// for the now-unconditional form.
#[test]
fn probe_associativity_globally_distinct_keys() {
    let mut rng = Rng(0xC0FFEE);
    for trial in 0..1000 {
        let pool = rand_wellformed(&mut rng, 12);
        if pool.len() < 12 {
            continue;
        }
        let a = pool[0..4].to_vec();
        let b = pool[4..8].to_vec();
        let c = pool[8..12].to_vec();

        let flat: Vec<_> = pool.clone();
        let s_flat = merge_all(&flat).unwrap();

        let ab_state = merge_all(&a.iter().chain(b.iter()).cloned().collect::<Vec<_>>()).unwrap();
        let left: Vec<_> = ab_state
            .observations
            .iter()
            .chain(c.iter())
            .cloned()
            .collect();
        let s_left = merge_all(&left).unwrap();

        let bc_state = merge_all(&b.iter().chain(c.iter()).cloned().collect::<Vec<_>>()).unwrap();
        let right: Vec<_> = a
            .iter()
            .cloned()
            .chain(bc_state.observations.iter().cloned())
            .collect();
        let s_right = merge_all(&right).unwrap();

        assert!(
            states_equal(&s_left, &s_flat),
            "left-carried != flat with globally distinct keys (trial {trial})"
        );
        assert!(
            states_equal(&s_right, &s_flat),
            "right-carried != flat with globally distinct keys (trial {trial})"
        );
    }
}

/// Re-merge stability: merge(merge(X).observations) == merge(X).
/// Follows from the same content-derived-id design; checked at scale.
#[test]
fn probe_remerge_idempotence_randomized() {
    let mut rng = Rng(0xB16B00B5);
    for trial in 0..1000 {
        let x = rand_wellformed(&mut rng, 10);
        let s1 = merge_all(&x).unwrap();
        let s2 = merge_all(&s1.observations).unwrap();
        assert!(
            states_equal(&s1, &s2),
            "re-merge of own output changed state (trial {trial})"
        );
    }
}

/// THEOREM 6 (pinned; was KNOWN DIVERGENCE #2, fixed): carried state
/// and evidence-recompute agree under staleness. The fix: a
/// synthesized C is stamped `round = min(parents)`, so it expires
/// exactly when the pair stops coexisting — both semantics reduce to
/// `min(parents) >= cutoff`. Evidence-recompute is authoritative; a
/// derived contradiction is supported only while all of its evidence
/// is inside the window the caller declared valid
/// (contradiction-preservation, not contradiction-immortality).
///
/// Deterministic construction (the original ghost) plus a randomized
/// sweep over rounds and windows.
#[test]
fn theorem_staleness_carried_equals_recompute() {
    let a = HexObservation {
        source: "tars".into(),
        diagnosis_id: "d0".into(),
        round: 1,
        ordinal: 0,
        claim_key: "k::valuable".into(),
        variant: Hexavalent::True,
        weight: 1.0,
        evidence: None,
    };
    let b = HexObservation {
        source: "ix".into(),
        diagnosis_id: "d1".into(),
        round: 5,
        ordinal: 0,
        claim_key: "k::valuable".into(),
        variant: Hexavalent::False,
        weight: 1.0,
        evidence: None,
    };

    // Merge while both are live (no staleness yet): synthesizes C.
    let live = merge(&[a.clone(), b.clone()], None, None).unwrap();
    assert_eq!(live.contradictions.len(), 1, "precondition: C synthesized");

    // The original ghost window: recompute and carried must agree.
    let recompute = merge(&[a, b], Some(5), Some(3)).unwrap();
    let carried = merge(&live.observations, Some(5), Some(3)).unwrap();
    assert_eq!(recompute.contradictions.len(), 0);
    assert_eq!(
        carried.contradictions.len(),
        0,
        "ghost is back: carried C outlived its round-1 parent"
    );
    assert!(states_equal(&recompute, &carried));

    // Randomized sweep: any raw set, any window — carried == recompute.
    let mut rng = Rng(0x9057);
    for trial in 0..1500 {
        let raw: Vec<_> = (0..8).map(|_| rand_obs(&mut rng)).collect();
        let live = merge_all(&raw).unwrap();
        let current = rng.below(8) as u32;
        let k = rng.below(4) as u32;
        let recompute = merge(&raw, Some(current), Some(k)).unwrap();
        let carried = merge(&live.observations, Some(current), Some(k)).unwrap();
        assert!(
            states_equal(&recompute, &carried),
            "carried != recompute at current={current} k={k} (trial {trial})"
        );
    }
}

/// THEOREM 3 (pinned): anti-dilution. Naive normalization arithmetic
/// predicts an escalated contradiction can be washed out by piling on
/// corroborating support (C-mass 1/(3+n) < 0.3 after one P). The
/// naive arithmetic is wrong about this merge: each corroborating P
/// itself conflicts with the standing F ((P,F) → 0.8 in the Belnap
/// table) and synthesizes additional C mass. Measured trajectory
/// RISES monotonically from 1/3 toward the 0.8/1.8 ≈ 0.444 asymptote.
/// An escalation cannot be muted by corroboration while the dissent
/// stands — a genuine robustness property of the substrate, worth
/// pinning: it is what makes `escalation_triggered` resistant to
/// consensus-flooding by agreeing sources.
#[test]
fn theorem_escalation_is_antidilutive() {
    let mk = |src: &str, dx: &str, variant, ordinal| HexObservation {
        source: src.into(),
        diagnosis_id: dx.into(),
        round: 0,
        ordinal,
        claim_key: "k::valuable".into(),
        variant,
        weight: 1.0,
        evidence: None,
    };
    let a = mk("tars", "d0", Hexavalent::True, 0);
    let b = mk("ix", "d1", Hexavalent::False, 0);

    let mut obs = vec![a, b];
    let mut trajectory = Vec::new();
    for n in 0..20 {
        let state = merge_all(&obs).unwrap();
        let c_mass = state.distribution.get(&Hexavalent::Contradictory);
        trajectory.push(c_mass);
        if !ix_fuzzy::hexavalent::escalation_triggered(&state.distribution) {
            panic!("escalation muted after {n} corroborations; trajectory {trajectory:?}");
        }
        // one more corroborating source-distinct P observation
        obs.push(mk(
            &format!("extra{n}"),
            &format!("dx{n}"),
            Hexavalent::Probable,
            n,
        ));
    }
    // Naive normalization arithmetic predicts dilution: C-mass 1/(3+n)
    // drops below the 0.3 escalation threshold after ONE corroborating
    // P. The naive arithmetic is WRONG: each corroborating P itself
    // conflicts with the standing F ((P,F) → 0.8 in the Belnap table),
    // synthesizing additional C mass. Corroboration under standing
    // dissent COMPOUNDS the contradiction instead of washing it out —
    // the escalation flag cannot be muted by piling on support. This
    // is a genuine robustness property of the merge; pin it.
    eprintln!("C-mass trajectory under corroboration: {trajectory:?}");
    assert!(
        trajectory.iter().all(|&m| m > 0.3),
        "anti-dilution property lost: {trajectory:?}"
    );
}
