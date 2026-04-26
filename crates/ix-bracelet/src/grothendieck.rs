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
}
