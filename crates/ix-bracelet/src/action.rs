//! Group action trait and the D₁₂ action on [`PcSet`].

use crate::dihedral::DihedralElement;
use crate::pc_set::PcSet;

const MASK12: u16 = 0x0FFF;

/// A left action of `Self` on values of type `T`.
///
/// Law (verified in tests, not by this trait): `(g·h).apply(x) == g.apply(h.apply(x))`.
pub trait Action<T> {
    fn apply(&self, x: T) -> T;
}

/// Rotate the 12-bit mask left by `n` within its 12-bit window.
#[inline]
fn rotate_mask(mask: u16, n: u8) -> u16 {
    let n = n % 12;
    if n == 0 {
        mask & MASK12
    } else {
        ((mask << n) | (mask >> (12 - n))) & MASK12
    }
}

/// Inversion about PC 0: bit `k` → bit `(12 - k) mod 12`. Bit 0 is a fixed point.
#[inline]
fn reflect_mask(mask: u16) -> u16 {
    let mut out = 0u16;
    for k in 0u16..12 {
        if (mask >> k) & 1 == 1 {
            let target = (12 - k) % 12;
            out |= 1u16 << target;
        }
    }
    out & MASK12
}

impl Action<PcSet> for DihedralElement {
    fn apply(&self, x: PcSet) -> PcSet {
        // Normal form r^rot · s^refl: reflect first, then rotate.
        // Matches compose() so that (g·h).apply(x) == g.apply(h.apply(x)).
        let mut m = x.raw();
        if self.reflected {
            m = reflect_mask(m);
        }
        m = rotate_mask(m, self.rotation);
        PcSet::new(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dihedral::Group;

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

    fn sample_sets() -> Vec<PcSet> {
        vec![
            PcSet::empty(),
            PcSet::chromatic(),
            PcSet::from_pcs([0, 4, 7]),              // C major triad
            PcSet::from_pcs([0, 3, 7]),              // C minor triad
            PcSet::from_pcs([0, 2, 4, 5, 7, 9, 11]), // C major scale
            PcSet::from_pcs([0, 1, 4, 6]),
            PcSet::from_pcs([0, 2, 4, 6, 8, 10]), // whole tone
            PcSet::from_pcs([0]),
            PcSet::from_pcs([0, 6]), // tritone
        ]
    }

    #[test]
    fn identity_fixes_everything() {
        let id = DihedralElement::identity();
        for x in sample_sets() {
            assert_eq!(id.apply(x), x);
        }
    }

    #[test]
    fn t12_is_identity_on_action() {
        // rotation(12) normalizes to identity, and the action must leave the mask untouched.
        let g = DihedralElement::rotation(12);
        assert_eq!(g, DihedralElement::identity());
        for x in sample_sets() {
            assert_eq!(g.apply(x), x);
        }
    }

    #[test]
    fn twelve_rotations_return_to_start() {
        let r = DihedralElement::rotation(1);
        for x in sample_sets() {
            let mut y = x;
            for _ in 0..12 {
                y = r.apply(y);
            }
            assert_eq!(y, x);
        }
    }

    #[test]
    fn inversion_is_involution_on_sets() {
        let i = DihedralElement::inversion();
        for x in sample_sets() {
            assert_eq!(i.apply(i.apply(x)), x);
        }
    }

    #[test]
    fn inversion_fixes_zero_bit() {
        // PC 0 is the axis of reflection.
        let i = DihedralElement::inversion();
        let s = PcSet::from_pcs([0, 3, 5]);
        let t = i.apply(s);
        assert!(t.contains(0));
        assert!(t.contains(9));
        assert!(t.contains(7));
        assert_eq!(t.cardinality(), 3);
    }

    #[test]
    fn action_composition_matches_group_composition() {
        let elems = all_elements();
        let xs = sample_sets();
        // Triple sample to keep it bounded: 24 × 24 × |xs| = ~5k checks.
        for g in &elems {
            for h in &elems {
                let gh = g.compose(h);
                for x in &xs {
                    assert_eq!(
                        gh.apply(*x),
                        g.apply(h.apply(*x)),
                        "action/compose mismatch: g={g:?} h={h:?} x={x:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn rotation_moves_pc_by_n() {
        // A single bit at position k, rotated by n, lands at (k+n) mod 12.
        for k in 0u8..12 {
            let x = PcSet::from_pcs([k]);
            for n in 0u8..12 {
                let y = DihedralElement::rotation(n).apply(x);
                let expected = PcSet::from_pcs([(k + n) % 12]);
                assert_eq!(y, expected, "k={k} n={n}");
            }
        }
    }
}
