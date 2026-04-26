//! Canonical orbit representatives.
//!
//! [`necklace_prime_form`] gives the smallest mask under the cyclic ⟨r⟩ action
//! (transposition only). [`bracelet_prime_form`] gives the smallest under the
//! full dihedral D₁₂ action (transposition + inversion) — matching Forte's
//! set-class convention.

use crate::action::Action;
use crate::dihedral::DihedralElement;
use crate::pc_set::PcSet;

/// Lex order on sorted PC lists, encoded directly on 12-bit masks.
///
/// Raw-u16 comparison gives the wrong answer for sets like `{0, 11}` vs `{1, 2}`
/// — the set whose smallest differing PC is lower must win, which corresponds to
/// the first set-bit of `a ^ b` belonging to `a`.
#[inline]
fn lex_less(a: u16, b: u16) -> bool {
    if a == b {
        return false;
    }
    let diff = a ^ b;
    let first_diff_bit = diff.trailing_zeros();
    (a >> first_diff_bit) & 1 != 0
}

/// Minimum (under [`lex_less`]) over the 12 transpositions of `x`.
pub fn necklace_prime_form(x: PcSet) -> PcSet {
    let mut best = x;
    for n in 1u8..12 {
        let cand = DihedralElement::rotation(n).apply(x);
        if lex_less(cand.raw(), best.raw()) {
            best = cand;
        }
    }
    best
}

/// Minimum (under [`lex_less`]) over all 24 dihedral images of `x`.
pub fn bracelet_prime_form(x: PcSet) -> PcSet {
    let mut best = x;
    for reflected in [false, true] {
        for rotation in 0u8..12 {
            let g = DihedralElement {
                rotation,
                reflected,
            };
            let cand = g.apply(x);
            if lex_less(cand.raw(), best.raw()) {
                best = cand;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sets() -> Vec<PcSet> {
        vec![
            PcSet::empty(),
            PcSet::chromatic(),
            PcSet::from_pcs([0, 4, 7]),
            PcSet::from_pcs([0, 3, 7]),
            PcSet::from_pcs([0, 2, 4, 5, 7, 9, 11]),
            PcSet::from_pcs([0, 1, 4, 6]),
            PcSet::from_pcs([0, 2, 4, 6, 8, 10]),
            PcSet::from_pcs([0]),
            PcSet::from_pcs([0, 6]),
            PcSet::from_pcs([1, 5, 8]),
        ]
    }

    fn all_elements() -> Vec<DihedralElement> {
        let mut v = Vec::with_capacity(24);
        for reflected in [false, true] {
            for rotation in 0u8..12 {
                v.push(DihedralElement {
                    rotation,
                    reflected,
                });
            }
        }
        v
    }

    #[test]
    fn lex_less_basic() {
        // {0, 11} vs {1, 2} — the u16-comparison trap.
        let a = PcSet::from_pcs([0, 11]).raw();
        let b = PcSet::from_pcs([1, 2]).raw();
        assert!(lex_less(a, b));
        assert!(!lex_less(b, a));
        assert!(!lex_less(a, a));
    }

    #[test]
    fn lex_less_empty_vs_nonempty() {
        // Empty has no set bits; any non-empty differs and wins at its lowest PC.
        let empty = PcSet::empty().raw();
        let single = PcSet::from_pcs([0]).raw();
        assert!(!lex_less(empty, single));
        assert!(lex_less(single, empty));
    }

    #[test]
    fn lex_less_matches_sorted_list_order() {
        // Sanity check the encoding on a handful of pairs.
        fn sorted(s: PcSet) -> Vec<u8> {
            s.iter_pcs().collect()
        }
        let pairs: &[(&[u8], &[u8])] = &[
            (&[0, 3, 7], &[0, 4, 7]),
            (&[0, 1, 2], &[0, 1, 3]),
            (&[0, 11], &[1]),
            (&[0, 5, 10], &[0, 5, 11]),
        ];
        for (ai, bi) in pairs {
            let a = PcSet::from_pcs(ai.iter().copied());
            let b = PcSet::from_pcs(bi.iter().copied());
            let expected = sorted(a) < sorted(b);
            assert_eq!(lex_less(a.raw(), b.raw()), expected, "{ai:?} vs {bi:?}");
        }
    }

    #[test]
    fn necklace_prime_form_idempotent() {
        for x in sample_sets() {
            let pf = necklace_prime_form(x);
            assert_eq!(necklace_prime_form(pf), pf);
        }
    }

    #[test]
    fn necklace_prime_form_transposition_invariant() {
        for x in sample_sets() {
            let base = necklace_prime_form(x);
            for n in 0u8..12 {
                let y = DihedralElement::rotation(n).apply(x);
                assert_eq!(necklace_prime_form(y), base, "x={x:?} n={n}");
            }
        }
    }

    #[test]
    fn bracelet_prime_form_idempotent() {
        for x in sample_sets() {
            let pf = bracelet_prime_form(x);
            assert_eq!(bracelet_prime_form(pf), pf);
        }
    }

    #[test]
    fn bracelet_prime_form_dihedral_invariant() {
        for x in sample_sets() {
            let base = bracelet_prime_form(x);
            for g in all_elements() {
                let y = g.apply(x);
                assert_eq!(bracelet_prime_form(y), base, "x={x:?} g={g:?}");
            }
        }
    }

    #[test]
    fn bracelet_refines_necklace() {
        // The D₁₂ orbit contains the cyclic orbit, so its minimum is ≤ the cyclic minimum.
        for x in sample_sets() {
            let b = bracelet_prime_form(x).raw();
            let n = necklace_prime_form(x).raw();
            assert!(
                b == n || lex_less(b, n),
                "x={x:?} bracelet={b:012b} necklace={n:012b}"
            );
        }
    }

    #[test]
    fn major_triad_prime_form_is_minor_triad_mask() {
        // Major and minor triads share a D₁₂ orbit (Forte 3-11); canonical rep is {0,3,7}.
        let major = PcSet::from_pcs([0, 4, 7]);
        let minor = PcSet::from_pcs([0, 3, 7]);
        assert_eq!(major.raw(), 0x91);
        assert_eq!(minor.raw(), 0x89);
        assert_eq!(bracelet_prime_form(major), minor);
        assert_eq!(bracelet_prime_form(minor), minor);
    }

    #[test]
    fn empty_and_chromatic_are_fixed_points() {
        assert_eq!(bracelet_prime_form(PcSet::empty()), PcSet::empty());
        assert_eq!(bracelet_prime_form(PcSet::chromatic()), PcSet::chromatic());
        assert_eq!(necklace_prime_form(PcSet::empty()), PcSet::empty());
        assert_eq!(necklace_prime_form(PcSet::chromatic()), PcSet::chromatic());
    }
}
