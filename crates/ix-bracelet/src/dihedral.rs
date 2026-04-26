//! Dihedral group D₁₂ = ⟨r, s | r¹² = s² = e, srs = r⁻¹⟩.
//!
//! Canonical normal form: `g = r^rotation · s^(reflected as 0/1)`.

/// Generic finite-group interface, used here for D₁₂ and its subgroups.
pub trait Group: Sized {
    fn identity() -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn inverse(&self) -> Self;
    /// Smallest `n ≥ 1` with `self^n == identity()`. For D₁₂, divides 24.
    fn order(&self) -> u8;
}

/// An element of D₁₂. 1 byte in size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DihedralElement {
    /// Rotation exponent, always in `0..12`.
    pub rotation: u8,
    /// Whether a reflection factor is present in the normal form.
    pub reflected: bool,
}

impl DihedralElement {
    #[inline]
    pub const fn identity() -> Self {
        Self {
            rotation: 0,
            reflected: false,
        }
    }

    /// Pure rotation `Tₙ`. `n` is reduced mod 12.
    #[inline]
    pub const fn rotation(n: u8) -> Self {
        Self {
            rotation: n % 12,
            reflected: false,
        }
    }

    /// Inversion about pitch-class 0 (the operator `I = T₀I`).
    #[inline]
    pub const fn inversion() -> Self {
        Self {
            rotation: 0,
            reflected: true,
        }
    }

    /// Builder: `Tₙ` if `!inverted`, `TₙI` if `inverted`.
    #[inline]
    pub const fn from_tn_tni(n: u8, inverted: bool) -> Self {
        Self {
            rotation: n % 12,
            reflected: inverted,
        }
    }
}

impl Group for DihedralElement {
    #[inline]
    fn identity() -> Self {
        DihedralElement::identity()
    }

    fn compose(&self, other: &Self) -> Self {
        // (r^i s^j)(r^k s^l) = r^(i + (-1)^j · k) s^(j+l).
        // The sign flip encodes sr = r^(-1)s: rotation "conjugates" through reflection.
        let i = self.rotation as i16;
        let k = other.rotation as i16;
        let sign: i16 = if self.reflected { -1 } else { 1 };
        let new_rot = (i + sign * k).rem_euclid(12) as u8;
        Self {
            rotation: new_rot,
            reflected: self.reflected ^ other.reflected,
        }
    }

    fn inverse(&self) -> Self {
        if self.reflected {
            // Every reflection is an involution: (r^i s)^2 = e.
            *self
        } else {
            Self {
                rotation: (12 - self.rotation) % 12,
                reflected: false,
            }
        }
    }

    fn order(&self) -> u8 {
        let id = Self::identity();
        let mut acc = *self;
        for n in 1u8..=24 {
            if acc == id {
                return n;
            }
            acc = acc.compose(self);
        }
        // Unreachable: every D₁₂ element has order dividing 24.
        24
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn constructors_normalize() {
        assert_eq!(DihedralElement::rotation(12), DihedralElement::identity());
        assert_eq!(DihedralElement::rotation(13), DihedralElement::rotation(1));
        assert_eq!(DihedralElement::rotation(24), DihedralElement::identity());
        assert_eq!(
            DihedralElement::from_tn_tni(15, true),
            DihedralElement {
                rotation: 3,
                reflected: true
            }
        );
    }

    #[test]
    fn group_law_exhaustive_closure_and_inverse() {
        let elems = all_elements();
        let id = DihedralElement::identity();
        for g in &elems {
            assert_eq!(g.compose(&g.inverse()), id, "g·g⁻¹ ≠ e for {g:?}");
            assert_eq!(g.inverse().compose(g), id, "g⁻¹·g ≠ e for {g:?}");
            assert_eq!(id.compose(g), *g, "e·g ≠ g for {g:?}");
            assert_eq!(g.compose(&id), *g, "g·e ≠ g for {g:?}");
            for h in &elems {
                let gh = g.compose(h);
                // Closure: result is a valid normal-form element.
                assert!(gh.rotation < 12, "rotation out of range: {gh:?}");
            }
        }
        // All 576 products land in the 24-element set.
        let mut products = std::collections::HashSet::new();
        for g in &elems {
            for h in &elems {
                products.insert(g.compose(h));
            }
        }
        assert_eq!(products.len(), 24);
    }

    #[test]
    fn order_of_rotations() {
        assert_eq!(DihedralElement::identity().order(), 1);
        assert_eq!(DihedralElement::rotation(1).order(), 12);
        assert_eq!(DihedralElement::rotation(6).order(), 2);
        assert_eq!(DihedralElement::rotation(4).order(), 3);
        assert_eq!(DihedralElement::rotation(3).order(), 4);
        assert_eq!(DihedralElement::rotation(2).order(), 6);
    }

    #[test]
    fn order_of_reflections() {
        // Every reflection is an involution.
        assert_eq!(DihedralElement::inversion().order(), 2);
        for n in 0u8..12 {
            assert_eq!(
                DihedralElement::from_tn_tni(n, true).order(),
                2,
                "TnI n={n}"
            );
        }
    }

    #[test]
    fn cyclic_subgroup_of_rotations() {
        // The 12 Tn form ⟨r⟩ ≅ Z/12.
        let r = DihedralElement::rotation(1);
        let id = DihedralElement::identity();
        let mut acc = id;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..12 {
            seen.insert(acc);
            acc = acc.compose(&r);
        }
        assert_eq!(acc, id, "r^12 ≠ e");
        assert_eq!(seen.len(), 12);
        for n in 0u8..12 {
            assert!(seen.contains(&DihedralElement::rotation(n)));
        }
    }

    #[test]
    fn inversion_is_involution() {
        let i = DihedralElement::inversion();
        assert_eq!(i.compose(&i), DihedralElement::identity());
    }

    #[test]
    fn composition_non_commutative() {
        // Sanity: T3·I ≠ I·T3, since (r^3·s)(r^0·s) = r^3, but (r^0·s)(r^3·s) = r^(-3) = r^9.
        let t3 = DihedralElement::rotation(3);
        let inv = DihedralElement::inversion();
        let left = t3.compose(&inv);
        let right = inv.compose(&t3);
        assert_ne!(left, right);
        assert_eq!(
            left,
            DihedralElement {
                rotation: 3,
                reflected: true
            }
        );
        assert_eq!(
            right,
            DihedralElement {
                rotation: 9,
                reflected: true
            }
        );
    }
}
