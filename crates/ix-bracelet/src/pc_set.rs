//! 12-bit pitch-class set: bit `k` set ⟺ pitch-class `k ∈ Z/12` present.

use core::fmt;

const MASK12: u16 = 0x0FFF;

/// Set of pitch classes in `Z/12`, stored as the low 12 bits of a `u16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcSet(u16);

impl PcSet {
    /// Construct from a raw mask. Upper 4 bits are discarded.
    #[inline]
    pub const fn new(mask: u16) -> Self {
        Self(mask & MASK12)
    }

    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub const fn chromatic() -> Self {
        Self(MASK12)
    }

    /// Build from an iterator of pitch classes; values are reduced mod 12.
    pub fn from_pcs<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = u8>,
    {
        let mut mask = 0u16;
        for pc in iter {
            mask |= 1u16 << (pc % 12);
        }
        Self(mask & MASK12)
    }

    #[inline]
    pub const fn insert(self, pc: u8) -> Self {
        Self((self.0 | (1u16 << (pc % 12))) & MASK12)
    }

    #[inline]
    pub const fn remove(self, pc: u8) -> Self {
        Self(self.0 & !(1u16 << (pc % 12)) & MASK12)
    }

    #[inline]
    pub const fn contains(self, pc: u8) -> bool {
        (self.0 >> (pc % 12)) & 1 == 1
    }

    #[inline]
    pub const fn cardinality(self) -> u32 {
        self.0.count_ones()
    }

    /// Raw 12-bit mask.
    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Pitch classes in ascending order, `0..12`.
    pub fn iter_pcs(self) -> impl Iterator<Item = u8> {
        let mask = self.0;
        (0u8..12).filter(move |k| (mask >> k) & 1 == 1)
    }

    /// The `Z/12` complement: every pitch class **not** in `self`. A hexachord and its
    /// complement are the two halves of the aggregate (the basis of combinatoriality).
    #[inline]
    pub const fn complement(self) -> Self {
        Self((!self.0) & MASK12)
    }

    /// Multiplicative transform `Mₘ`: maps each pitch class `x ↦ (m · x) mod 12`. `M5` is
    /// the circle-of-fourths transform, `M7` the circle-of-fifths, `M11 = I₀` (inversion).
    /// For `m` coprime to 12 (1, 5, 7, 11) this is a bijection; otherwise the image collapses.
    pub fn multiply(self, m: u8) -> Self {
        Self::from_pcs(self.iter_pcs().map(|x| (x * (m % 12)) % 12))
    }
}

impl fmt::Display for PcSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        let mut first = true;
        for pc in self.iter_pcs() {
            if !first {
                f.write_str(", ")?;
            }
            write!(f, "{pc}")?;
            first = false;
        }
        f.write_str("}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_masks_upper_bits() {
        // Upper 4 bits must be discarded so PcSet stays a proper 12-bit carrier.
        assert_eq!(PcSet::new(0x1000).raw(), 0);
        assert_eq!(PcSet::new(0xF000).raw(), 0);
        assert_eq!(PcSet::new(0xF007).raw(), 0x007);
    }

    #[test]
    fn empty_and_chromatic() {
        assert_eq!(PcSet::empty().cardinality(), 0);
        assert_eq!(PcSet::chromatic().cardinality(), 12);
        assert_eq!(PcSet::chromatic().raw(), 0x0FFF);
    }

    #[test]
    fn from_pcs_reduces_mod_12() {
        let s = PcSet::from_pcs([0, 4, 7, 12, 16, 19]);
        assert_eq!(s, PcSet::from_pcs([0, 4, 7]));
    }

    #[test]
    fn insert_remove_contains() {
        let s = PcSet::empty().insert(3).insert(11);
        assert!(s.contains(3));
        assert!(s.contains(11));
        assert!(!s.contains(0));
        let s = s.remove(3);
        assert!(!s.contains(3));
        assert!(s.contains(11));
    }

    #[test]
    fn iter_pcs_sorted() {
        let pcs: Vec<u8> = PcSet::chromatic().iter_pcs().collect();
        assert_eq!(pcs, (0u8..12).collect::<Vec<_>>());

        let pcs: Vec<u8> = PcSet::from_pcs([7, 0, 4]).iter_pcs().collect();
        assert_eq!(pcs, vec![0, 4, 7]);
    }

    #[test]
    fn complement_is_z12_negation() {
        let s = PcSet::from_pcs([0, 4, 7]);
        let c = s.complement();
        assert_eq!(c, PcSet::from_pcs([1, 2, 3, 5, 6, 8, 9, 10, 11]));
        // Involution; complement of complement is the original.
        assert_eq!(c.complement(), s);
        // A set and its complement partition the aggregate.
        assert_eq!(s.cardinality() + c.cardinality(), 12);
        assert_eq!(PcSet::empty().complement(), PcSet::chromatic());
    }

    #[test]
    fn multiply_m5_m7_m11() {
        // M7 (circle of fifths) on {0,1,2} -> {0,7,2}.
        assert_eq!(
            PcSet::from_pcs([0, 1, 2]).multiply(7),
            PcSet::from_pcs([0, 7, 2])
        );
        // M5 (circle of fourths) on {0,1,2} -> {0,5,10}.
        assert_eq!(
            PcSet::from_pcs([0, 1, 2]).multiply(5),
            PcSet::from_pcs([0, 5, 10])
        );
        // M11 == inversion about 0: {0,1,2} -> {0,11,10}.
        assert_eq!(
            PcSet::from_pcs([0, 1, 2]).multiply(11),
            PcSet::from_pcs([0, 11, 10])
        );
    }

    #[test]
    fn display_forte_style() {
        assert_eq!(PcSet::empty().to_string(), "{}");
        assert_eq!(PcSet::from_pcs([0, 4, 7]).to_string(), "{0, 4, 7}");
        assert_eq!(
            PcSet::chromatic().to_string(),
            "{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}"
        );
    }
}
