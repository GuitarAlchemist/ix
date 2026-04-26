//! Grothendieck-group operations on pitch-class sets.
//!
//! The interval-class vector (ICV) sends a `PcSet` into the monoid `ℕ⁶`. Lifting
//! that monoid to its Grothendieck group `ℤ⁶` lets us subtract — answering
//! "what intervals are gained or lost going from set A to set B?" with a signed
//! 6-vector. This is the algebraic backbone of GA's
//! `GA.Domain.Services.Atonal.Grothendieck` module, mirrored here in pure Rust
//! so cross-language equivalence is testable end-to-end.
//!
//! - [`Icv`] — interval-class vector `(ic1, ic2, ic3, ic4, ic5, ic6) ∈ ℕ⁶`.
//! - [`Delta`] — signed delta in `ℤ⁶`; closed under add/sub/neg (the group lift).
//! - [`icv`] — compute ICV from `PcSet`.
//! - [`grothendieck_delta`] — `target - source` in ℤ⁶.
//! - [`find_nearby`] — orbit-aware search for PC-sets within an L1 budget.

use crate::orbit::{all_prime_forms, orbit_unique};
use crate::pc_set::PcSet;
use core::ops::{Add, Neg, Sub};

/// Interval-class vector: count of each interval class `1..=6` in a PC-set.
///
/// For a set of cardinality `n`, the entries sum to `C(n, 2) = n·(n-1)/2`.
/// Two PC-sets in the same D₁₂ orbit (TᵢI-equivalent) share the same ICV; the
/// converse does **not** hold (Z-related sets are the canonical counterexample).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Icv {
    /// `[ic1, ic2, ic3, ic4, ic5, ic6]`. Indices are 0-based; `ic_k = data[k-1]`.
    pub data: [u32; 6],
}

impl Icv {
    pub const fn new(data: [u32; 6]) -> Self {
        Self { data }
    }

    pub const fn zero() -> Self {
        Self { data: [0; 6] }
    }

    /// `target - source` in the Grothendieck group ℤ⁶.
    #[inline]
    pub fn delta(self, target: Icv) -> Delta {
        let mut out = [0i32; 6];
        for (slot, (&t, &s)) in out.iter_mut().zip(target.data.iter().zip(self.data.iter())) {
            *slot = t as i32 - s as i32;
        }
        Delta { data: out }
    }

    /// Sum of entries — equals `C(n, 2)` where `n` is set cardinality.
    pub fn total(self) -> u32 {
        self.data.iter().sum()
    }
}

/// Signed Grothendieck delta on ICVs, valued in ℤ⁶.
///
/// Closed under addition, subtraction, and negation — the group axioms that
/// distinguish this from the underlying ICV monoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Delta {
    pub data: [i32; 6],
}

impl Delta {
    pub const fn new(data: [i32; 6]) -> Self {
        Self { data }
    }

    pub const fn zero() -> Self {
        Self { data: [0; 6] }
    }

    /// L1 (Manhattan) norm — total number of interval-class steps gained or lost.
    /// Used by `find_nearby` as the harmonic distance metric.
    pub fn l1_norm(self) -> u32 {
        self.data.iter().map(|d| d.unsigned_abs()).sum()
    }

    /// L2 (Euclidean) norm.
    pub fn l2_norm(self) -> f64 {
        let s: i64 = self.data.iter().map(|&d| (d as i64) * (d as i64)).sum();
        (s as f64).sqrt()
    }

    pub fn is_zero(self) -> bool {
        self.data.iter().all(|&d| d == 0)
    }
}

impl Add for Delta {
    type Output = Delta;
    fn add(self, rhs: Delta) -> Delta {
        let mut out = [0i32; 6];
        for (slot, (&a, &b)) in out.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *slot = a + b;
        }
        Delta { data: out }
    }
}

impl Sub for Delta {
    type Output = Delta;
    fn sub(self, rhs: Delta) -> Delta {
        let mut out = [0i32; 6];
        for (slot, (&a, &b)) in out.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *slot = a - b;
        }
        Delta { data: out }
    }
}

impl Neg for Delta {
    type Output = Delta;
    fn neg(self) -> Delta {
        let mut out = [0i32; 6];
        for (slot, &v) in out.iter_mut().zip(self.data.iter()) {
            *slot = -v;
        }
        Delta { data: out }
    }
}

/// Compute the interval-class vector of a PC-set.
///
/// For each unordered pair `(a, b)` of distinct PCs in the set, the interval
/// class is `min(|a − b|, 12 − |a − b|) ∈ {1, 2, 3, 4, 5, 6}` (tritone caps the
/// IC at 6). The result counts each IC across all such pairs.
pub fn icv(set: PcSet) -> Icv {
    let pcs: Vec<u8> = set.iter_pcs().collect();
    let mut data = [0u32; 6];
    for i in 0..pcs.len() {
        for j in (i + 1)..pcs.len() {
            let diff = pcs[j].wrapping_sub(pcs[i]) % 12;
            let ic = if diff <= 6 { diff } else { 12 - diff };
            // ic ∈ 1..=6 since pcs are distinct mod 12
            data[(ic - 1) as usize] += 1;
        }
    }
    Icv { data }
}

/// Grothendieck delta `target − source` between two PC-sets, computed via their ICVs.
///
/// Equivalent to `icv(source).delta(icv(target))` but offered as a top-level
/// shortcut to match the GA `IGrothendieckService.ComputeDelta` shape.
pub fn grothendieck_delta(source: PcSet, target: PcSet) -> Delta {
    icv(source).delta(icv(target))
}

/// Find PC-sets whose ICV lies within `max_l1` Grothendieck-distance of `source`.
///
/// **Orbit-aware**: ICV is TᵢI-invariant, so all members of a D₁₂ orbit share an
/// ICV (and hence a delta to the source). We compute the delta once per
/// `all_prime_forms()` orbit rep (224 reps) and then expand each surviving rep
/// to its full orbit, instead of iterating all 4096 raw subsets — typically
/// ~18× fewer ICV computations than a brute-force scan.
///
/// Returns `(set, delta_from_source, l1_cost)` triples. The `source` itself
/// appears with `Delta::zero()` and `cost = 0`. Order is unspecified beyond
/// "grouped by orbit rep, reps in `all_prime_forms` order".
pub fn find_nearby(source: PcSet, max_l1: u32) -> Vec<(PcSet, Delta, u32)> {
    let src_icv = icv(source);
    let mut out = Vec::new();
    for &rep in all_prime_forms() {
        let delta = src_icv.delta(icv(rep));
        let cost = delta.l1_norm();
        if cost <= max_l1 {
            for member in orbit_unique(rep) {
                out.push((member, delta, cost));
            }
        }
    }
    out
}

/// Pairs of D₁₂ orbit representatives that share an interval-class vector but
/// belong to **different** orbits — the classical "Z-relation" of set theory.
///
/// Z-related sets are interesting precisely because the ICV does *not*
/// distinguish them: any embedding that compresses voicings down to ICV-only
/// would collapse Z-pairs into the same point even though they are distinct
/// set classes. Phase-3 OPTIC-K invariant checks use this list to verify
/// STRUCTURE doesn't accidentally degenerate to ICV: voicings drawn from
/// Z-related orbit reps MUST have STRUCTURE cosine strictly less than 1.0.
///
/// Returns each pair as `(rep_a, rep_b)` with `rep_a.raw() < rep_b.raw()` for
/// stable iteration. Counts: in 12-TET, Z-related set classes appear at
/// cardinalities 4 (one pair), 5 (three pairs), 6 (the "all-Z hexachords",
/// 15 pairs), 7 (three pairs, complements of 5), and 8 (one pair, complement
/// of 4). Total: 23 unordered pairs.
pub fn z_related_pairs() -> Vec<(PcSet, PcSet)> {
    use std::collections::BTreeMap;
    let primes = all_prime_forms();
    // Key by (cardinality, ICV) so the empty set's all-zero ICV doesn't
    // collide with a singleton's all-zero ICV — Z-relation requires equal
    // cardinality by definition.
    let mut by_card_icv: BTreeMap<(u32, [u32; 6]), Vec<PcSet>> = BTreeMap::new();
    for &p in primes {
        by_card_icv
            .entry((p.cardinality(), icv(p).data))
            .or_default()
            .push(p);
    }
    let mut pairs = Vec::new();
    for sets in by_card_icv.values() {
        if sets.len() < 2 {
            continue;
        }
        // Stable ordering: sets are already in `all_prime_forms()` order
        // (sorted by raw mask), so emit (i, j) with i < j.
        for i in 0..sets.len() {
            for j in (i + 1)..sets.len() {
                pairs.push((sets[i], sets[j]));
            }
        }
    }
    pairs
}

/// Search-state node for the harmonic-path A* — pairs the current PC-set with
/// the (constant) target so the state implements the goal predicate locally.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PathNode {
    current: PcSet,
    target: PcSet,
}

impl ix_search::astar::SearchState for PathNode {
    type Action = PcSet;

    fn successors(&self) -> Vec<(PcSet, Self, f64)> {
        let card = self.current.cardinality();
        find_nearby(self.current, FIND_NEARBY_RADIUS_PER_STEP)
            .into_iter()
            .filter(|(s, _, _)| s.cardinality() == card && *s != self.current)
            .map(|(s, _, _)| {
                (
                    s,
                    PathNode {
                        current: s,
                        target: self.target,
                    },
                    1.0,
                )
            })
            .collect()
    }

    fn is_goal(&self) -> bool {
        self.current == self.target
    }
}

/// Per-step ICV-L1 expansion radius for `find_shortest_path`. Mirrors GA's
/// `FindShortestPath` (radius=2): "connects closely related diatonic
/// collections (e.g., C major → G major) that typically differ by one
/// accidental yet may exceed radius=1 under the ICV L1 metric".
const FIND_NEARBY_RADIUS_PER_STEP: u32 = 2;

/// A* shortest harmonic path between two PC-sets of equal cardinality.
///
/// Drop-in replacement for GA's BFS `FindShortestPath`. Uses
/// `ix_search::astar::astar` with the admissible heuristic
/// `L1(δ(current, target)) / 2.0` — admissible because each step expands by
/// `find_nearby(_, 2)`, so a single step can drop the L1-to-target by at most
/// `2 * 2 = 4`... wait, actually each step can change ICV L1 by at most
/// `2 * FIND_NEARBY_RADIUS_PER_STEP = 4` in the worst case (the step itself
/// has L1 ≤ 2, and triangle inequality gives `|L1(after)−L1(before)| ≤ 2`).
/// So `L1/2.0` is the right admissible scaling.
///
/// Returns the full path including `source` and `target`. Returns an empty
/// `Vec` if no path exists or if the optimal path exceeds `max_steps` edges
/// (matching GA's `path.Count >= maxSteps + 1` cutoff).
pub fn find_shortest_path(source: PcSet, target: PcSet, max_steps: usize) -> Vec<PcSet> {
    if source.cardinality() != target.cardinality() {
        return Vec::new();
    }
    if source == target {
        return vec![source];
    }

    let initial = PathNode {
        current: source,
        target,
    };
    let heuristic = move |node: &PathNode| -> f64 {
        // Each step has L1 ≤ 2 (find_nearby radius), so L1/2 lower-bounds steps.
        grothendieck_delta(node.current, node.target).l1_norm() as f64 / 2.0
    };

    let result = match ix_search::astar::astar(initial, heuristic) {
        Some(r) => r,
        None => return Vec::new(),
    };

    // GA's BFS cutoff: `path.Count >= maxSteps + 1` halts expansion.
    // Equivalent post-filter: drop paths with more than max_steps edges.
    if result.path.len() > max_steps + 1 {
        return Vec::new();
    }
    result.path.into_iter().map(|n| n.current).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pcset(pcs: &[u8]) -> PcSet {
        PcSet::from_pcs(pcs.iter().copied())
    }

    #[test]
    fn icv_empty_set_is_zero() {
        assert_eq!(icv(PcSet::empty()), Icv::zero());
    }

    #[test]
    fn icv_singleton_is_zero() {
        // No pairs in a 1-element set
        assert_eq!(icv(pcset(&[5])), Icv::zero());
    }

    #[test]
    fn icv_c_major_triad() {
        // {C, E, G} = {0, 4, 7}: intervals 4 (M3=ic4), 3 (m3=ic3), 7→ic5
        // Pairs: (0,4)=4→ic4, (0,7)=7→ic5, (4,7)=3→ic3
        // Expected: ic3=1, ic4=1, ic5=1, others 0
        assert_eq!(icv(pcset(&[0, 4, 7])).data, [0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn icv_c_major_scale_matches_ga_reference() {
        // C major scale {0,2,4,5,7,9,11}: canonical ICV is <2, 5, 4, 3, 6, 1>
        // (per GA's GrothendieckService README and standard music-theory references)
        assert_eq!(icv(pcset(&[0, 2, 4, 5, 7, 9, 11])).data, [2, 5, 4, 3, 6, 1]);
    }

    #[test]
    fn icv_total_equals_n_choose_2() {
        // For any set of cardinality n, ICV entries sum to C(n, 2) = n*(n-1)/2.
        // saturating_sub handles n=0 cleanly (avoids u32 underflow).
        for mask in 0u16..=0x0FFF {
            let s = PcSet::new(mask);
            let n = s.cardinality();
            let expected = n * n.saturating_sub(1) / 2;
            assert_eq!(
                icv(s).total(),
                expected,
                "ICV total mismatch for mask {mask:03X} (n={n})"
            );
        }
    }

    #[test]
    fn icv_is_t_i_invariant() {
        // ICV is preserved under all 24 D₁₂ operations — verify on a few sets
        use crate::orbit::orbit_unique;
        for mask in [0x091, 0x0A4, 0x0FFF, 0x000, 0x0AB] {
            let x = PcSet::new(mask);
            let target = icv(x);
            for y in orbit_unique(x) {
                assert_eq!(icv(y), target, "ICV not TᵢI-invariant for {mask:03X}");
            }
        }
    }

    #[test]
    fn grothendieck_delta_major_triad_to_augmented_triad() {
        // C major triad {0,4,7} ICV: ic3=1, ic4=1, ic5=1
        // C augmented triad {0,4,8} ICV: ic4=3 (three M3s)
        // Delta: <0, 0, -1, +2, -1, 0>
        let c_maj = pcset(&[0, 4, 7]);
        let c_aug = pcset(&[0, 4, 8]);
        let d = grothendieck_delta(c_maj, c_aug);
        assert_eq!(d.data, [0, 0, -1, 2, -1, 0]);
    }

    #[test]
    fn grothendieck_delta_major_to_natural_minor_scale_is_zero() {
        // C major scale {0,2,4,5,7,9,11} and C natural-minor scale {0,2,3,5,7,8,10}
        // are MODES of the same diatonic collection (Forte 7-35), so they share an
        // ICV — and the Grothendieck delta is zero. Worth pinning down because GA's
        // own README cites a non-zero delta for this pair, which contradicts the
        // mode-of-the-same-collection fact.
        let c_maj = pcset(&[0, 2, 4, 5, 7, 9, 11]);
        let c_nat_min = pcset(&[0, 2, 3, 5, 7, 8, 10]);
        assert_eq!(icv(c_maj), icv(c_nat_min));
        assert!(grothendieck_delta(c_maj, c_nat_min).is_zero());
    }

    #[test]
    fn delta_zero_iff_same_icv() {
        let a = pcset(&[0, 4, 7]);
        let b = pcset(&[2, 6, 9]); // same ICV (transposed major triad)
        assert!(grothendieck_delta(a, b).is_zero());

        let c = pcset(&[0, 4, 8]); // augmented triad — different ICV
        assert!(!grothendieck_delta(a, c).is_zero());
    }

    #[test]
    fn delta_l1_l2_norms() {
        let d = Delta::new([0, -1, 1, 1, -1, 0]);
        assert_eq!(d.l1_norm(), 4);
        assert!((d.l2_norm() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn delta_negation_inverts_direction() {
        let a = pcset(&[0, 2, 4, 5, 7, 9, 11]);
        let b = pcset(&[0, 2, 3, 5, 7, 8, 10]);
        let ab = grothendieck_delta(a, b);
        let ba = grothendieck_delta(b, a);
        assert_eq!(ba, -ab);
    }

    #[test]
    fn group_axiom_delta_is_additive_through_intermediate() {
        // Grothendieck-group axiom: δ(A,B) + δ(B,C) = δ(A,C)
        // This is the key invariant the GA Grothendieck module relies on.
        let a = pcset(&[0, 4, 7]);
        let b = pcset(&[0, 3, 7]);
        let c = pcset(&[0, 3, 6]);
        let ab = grothendieck_delta(a, b);
        let bc = grothendieck_delta(b, c);
        let ac = grothendieck_delta(a, c);
        assert_eq!(ab + bc, ac);
    }

    #[test]
    fn group_axiom_delta_round_trip_to_zero() {
        // δ(A,B) + δ(B,A) = 0 — corollary of negation + additivity.
        let a = pcset(&[0, 2, 4, 5, 7, 9, 11]);
        let b = pcset(&[1, 3, 5, 6, 8, 10, 0]);
        let ab = grothendieck_delta(a, b);
        let ba = grothendieck_delta(b, a);
        assert!((ab + ba).is_zero());
    }

    #[test]
    fn find_nearby_includes_source_at_zero() {
        let src = pcset(&[0, 4, 7]);
        let near = find_nearby(src, 0);
        // At cost 0, we get the entire orbit of src (all sets with the same ICV
        // *via the major-triad orbit*; Z-relations would not appear here since
        // they are different orbit reps).
        assert!(near.iter().any(|(s, d, c)| {
            *s == src && d.is_zero() && *c == 0
        }));
    }

    #[test]
    fn find_nearby_grows_monotone_with_budget() {
        let src = pcset(&[0, 4, 7]);
        let n0 = find_nearby(src, 0).len();
        let n2 = find_nearby(src, 2).len();
        let n6 = find_nearby(src, 6).len();
        assert!(n0 <= n2);
        assert!(n2 <= n6);
    }

    #[test]
    fn find_nearby_chromatic_budget_includes_everything() {
        // A large enough budget reaches every PC-set (4096 of them).
        // The maximum L1 between two ICVs is bounded by max ICV total = C(12,2) = 66
        // plus the same on the other side, so 2·66 = 132 is a hard ceiling.
        let src = pcset(&[0, 4, 7]);
        let near = find_nearby(src, 132);
        assert_eq!(near.len(), 4096);
    }

    #[test]
    fn shortest_path_source_equals_target_is_singleton() {
        let s = pcset(&[0, 4, 7]);
        assert_eq!(find_shortest_path(s, s, 5), vec![s]);
    }

    #[test]
    fn shortest_path_different_cardinalities_is_empty() {
        // Triad → seventh chord — find_shortest_path only spans equal-cardinality
        // sets (matches GA's `Where(s => s.Cardinality == current.Cardinality)`).
        let triad = pcset(&[0, 4, 7]);
        let seventh = pcset(&[0, 4, 7, 11]);
        assert!(find_shortest_path(triad, seventh, 5).is_empty());
    }

    #[test]
    fn shortest_path_major_to_minor_triad_one_step() {
        // C major triad {0,4,7} → C minor triad {0,3,7} share orbit (both Forte 3-11)
        // so they reach each other inside one find_nearby(_, 2) expansion.
        let c_maj = pcset(&[0, 4, 7]);
        let c_min = pcset(&[0, 3, 7]);
        let path = find_shortest_path(c_maj, c_min, 5);
        assert!(!path.is_empty(), "expected path C maj → C min");
        assert_eq!(path.first(), Some(&c_maj));
        assert_eq!(path.last(), Some(&c_min));
        assert!(
            path.len() <= 2,
            "expected ≤1 hop (path={:?}, len={})",
            path,
            path.len()
        );
    }

    #[test]
    fn shortest_path_returns_empty_when_max_steps_too_small() {
        // Major triad → augmented triad needs ≥1 step, but max_steps=0 forbids any
        // hop beyond the source itself.
        let c_maj = pcset(&[0, 4, 7]);
        let c_aug = pcset(&[0, 4, 8]);
        assert!(find_shortest_path(c_maj, c_aug, 0).is_empty());
    }

    #[test]
    fn z_related_pairs_have_identical_icvs() {
        // Definitional check: every pair returned must share an ICV.
        let pairs = z_related_pairs();
        assert!(!pairs.is_empty(), "12-TET has known Z-pairs at cardinalities 4-8");
        for (a, b) in &pairs {
            assert_eq!(
                icv(*a),
                icv(*b),
                "Z-pair {a} ↔ {b} disagrees on ICV (would not be Z-related)"
            );
            assert_ne!(a, b, "Z-pair must consist of distinct orbit reps");
        }
    }

    #[test]
    fn z_related_pairs_belong_to_different_orbits() {
        // Z-pairs are by definition different orbits — same ICV, different
        // bracelet prime form. (Same orbit ⇒ same prime form ⇒ same listed rep,
        // so this collapses to "the pair members differ" but worth pinning.)
        use crate::prime_form::bracelet_prime_form;
        for (a, b) in z_related_pairs() {
            assert_ne!(
                bracelet_prime_form(a),
                bracelet_prime_form(b),
                "Z-pair {a} ↔ {b} share a bracelet prime form — they are the same orbit"
            );
        }
    }

    #[test]
    fn z_related_pairs_count_matches_12tet_literature() {
        // Standard music-theory result: 12-TET has exactly 23 unordered Z-pairs
        // distributed across cardinalities 4 (1), 5 (3), 6 (15 — the all-Z
        // hexachords), 7 (3, complements of card-5), 8 (1, complement of card-4).
        // If this count drifts, either the all_prime_forms enumeration changed
        // (224 → other) or a Z-pair was lost/gained.
        let pairs = z_related_pairs();
        assert_eq!(
            pairs.len(),
            23,
            "Expected 23 Z-pairs in 12-TET; got {}",
            pairs.len()
        );
    }

    #[test]
    fn z_related_pairs_include_canonical_4_z15_4_z29() {
        // The smallest Z-pair: 4-Z15 = {0,1,4,6} and 4-Z29 = {0,1,3,7}.
        // Both have ICV [1,1,1,1,1,1] (all-interval tetrachords).
        let pairs = z_related_pairs();
        let z15 = PcSet::from_pcs([0u8, 1, 4, 6]);
        let z29 = PcSet::from_pcs([0u8, 1, 3, 7]);
        let z15_pf = crate::prime_form::bracelet_prime_form(z15);
        let z29_pf = crate::prime_form::bracelet_prime_form(z29);

        let found = pairs
            .iter()
            .any(|(a, b)| (*a == z15_pf && *b == z29_pf) || (*a == z29_pf && *b == z15_pf));
        assert!(
            found,
            "expected canonical 4-Z15/4-Z29 pair in z_related_pairs()"
        );
        assert_eq!(icv(z15).data, [1, 1, 1, 1, 1, 1]);
        assert_eq!(icv(z29).data, [1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn shortest_path_endpoints_match_source_and_target() {
        let c_maj_scale = pcset(&[0, 2, 4, 5, 7, 9, 11]);
        let g_maj_scale = pcset(&[0, 2, 4, 6, 7, 9, 11]); // C major + F♯ instead of F
        let path = find_shortest_path(c_maj_scale, g_maj_scale, 8);
        assert!(!path.is_empty(), "expected a path between adjacent diatonic scales");
        assert_eq!(*path.first().unwrap(), c_maj_scale);
        assert_eq!(*path.last().unwrap(), g_maj_scale);
    }
}
