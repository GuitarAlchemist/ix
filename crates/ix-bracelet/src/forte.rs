//! Allen Forte set-class names for the 224 D₁₂ orbits on subsets of `Z/12`.
//!
//! Forte's 1973 *The Structure of Atonal Music* catalogues each set class by
//! a pair `(cardinality, ordinal)` rendered `"{card}-{ord}"`, e.g. `"3-11"` for
//! the major/minor-triad class. Classes that share an interval-class vector but
//! are not T₁₂/T₁₂I-equivalent (**Z-relations**) carry a `Z` prefix on the
//! ordinal — `"4-Z15"` vs `"4-Z29"`. Under our D₁₂ action these Z-partners have
//! *distinct* bracelet orbits, so they receive distinct [`ForteNumber`] values.
//!
//! Asymmetric classes whose inversion differs from their transposition
//! representative split into Forte "A" (prime) and "B" (inverted) rows in the
//! full T₁₂ catalogue. D₁₂ merges those two rows into a single orbit, so we emit
//! the unlettered primary number — e.g. the major/minor triad both map to
//! `3-11`, not `3-11A`/`3-11B`.
//!
//! The mapping is built by taking each of the 224 representatives from
//! [`crate::all_prime_forms`] and pairing it with its Forte number looked up
//! from the standard published list. All bracelet prime forms in our enumeration
//! coincide exactly with the canonical Forte orbits (no divergence observed at
//! construction time; see the `every_forte_mask_is_a_bracelet_prime_form` test).
//!
//! # Example
//! ```
//! use ix_bracelet::{forte_number, ForteNumber, PcSet};
//!
//! let major_triad = PcSet::from_pcs([0, 4, 7]);
//! let minor_triad = PcSet::from_pcs([0, 3, 7]);
//! assert_eq!(forte_number(major_triad), Some(ForteNumber::new(3, 11)));
//! assert_eq!(forte_number(major_triad), forte_number(minor_triad));
//! assert_eq!(forte_number(major_triad).unwrap().to_string(), "3-11");
//! ```
//!
//! The Z flag is carried on the number:
//! ```
//! use ix_bracelet::{forte_number, PcSet};
//! let z15 = PcSet::from_pcs([0, 1, 4, 6]);
//! let z29 = PcSet::from_pcs([0, 1, 3, 7]);
//! let n15 = forte_number(z15).unwrap();
//! let n29 = forte_number(z29).unwrap();
//! assert_eq!(n15.to_string(), "4-Z15");
//! assert_eq!(n29.to_string(), "4-Z29");
//! assert_ne!(n15, n29);
//! ```

use crate::pc_set::PcSet;
use crate::prime_form::bracelet_prime_form;

use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

/// Allen Forte's set-class label: cardinality (number of pitch classes) paired
/// with an ordinal within that cardinality.
///
/// The `z` flag marks a **Z-relation** — two set classes that share an
/// interval-class vector but are not T₁₂/T₁₂I-equivalent. Under D₁₂ these remain
/// distinct orbits; the flag is preserved so `Display` emits `"{card}-Z{ord}"`.
///
/// Forte's A/B sub-labels (distinguishing a prime form from its T-inverse under
/// the T₁₂-only convention) collapse under D₁₂ and are not carried here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForteNumber {
    /// Cardinality of the set class (0–12).
    pub cardinality: u8,
    /// Ordinal within the cardinality (1-based).
    pub ordinal: u16,
    /// Z-relation flag.
    pub z: bool,
}

impl ForteNumber {
    /// Construct a non-Z Forte number.
    pub const fn new(cardinality: u8, ordinal: u16) -> Self {
        Self {
            cardinality,
            ordinal,
            z: false,
        }
    }

    /// Construct a Z-related Forte number (displayed with a `Z` prefix on the ordinal).
    pub const fn new_z(cardinality: u8, ordinal: u16) -> Self {
        Self {
            cardinality,
            ordinal,
            z: true,
        }
    }
}

impl fmt::Display for ForteNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.z {
            write!(f, "{}-Z{}", self.cardinality, self.ordinal)
        } else {
            write!(f, "{}-{}", self.cardinality, self.ordinal)
        }
    }
}

/// Static `(prime_form_mask, ForteNumber)` pairs — exactly the 224 D₁₂ orbits.
/// The prime form is the raw 12-bit mask produced by [`bracelet_prime_form`].
const FORTE_TABLE: &[(u16, ForteNumber)] = &[
    // Cardinality 0 – empty set
    (0x000, ForteNumber::new(0, 1)),
    // Cardinality 1 – singleton
    (0x001, ForteNumber::new(1, 1)),
    // Cardinality 2 – dyads (intervals 1..6)
    (0x003, ForteNumber::new(2, 1)),
    (0x005, ForteNumber::new(2, 2)),
    (0x009, ForteNumber::new(2, 3)),
    (0x011, ForteNumber::new(2, 4)),
    (0x021, ForteNumber::new(2, 5)),
    (0x041, ForteNumber::new(2, 6)),
    // Cardinality 3 – trichords
    (0x007, ForteNumber::new(3, 1)),
    (0x00B, ForteNumber::new(3, 2)),
    (0x013, ForteNumber::new(3, 3)),
    (0x023, ForteNumber::new(3, 4)),
    (0x043, ForteNumber::new(3, 5)),
    (0x015, ForteNumber::new(3, 6)),
    (0x025, ForteNumber::new(3, 7)),
    (0x045, ForteNumber::new(3, 8)),
    (0x085, ForteNumber::new(3, 9)),
    (0x049, ForteNumber::new(3, 10)),
    (0x089, ForteNumber::new(3, 11)), // major/minor triad orbit
    (0x111, ForteNumber::new(3, 12)), // augmented triad
    // Cardinality 4 – tetrachords (incl. Z-pair 4-Z15 / 4-Z29)
    (0x00F, ForteNumber::new(4, 1)),
    (0x017, ForteNumber::new(4, 2)),
    (0x01B, ForteNumber::new(4, 3)),
    (0x027, ForteNumber::new(4, 4)),
    (0x047, ForteNumber::new(4, 5)),
    (0x087, ForteNumber::new(4, 6)),
    (0x033, ForteNumber::new(4, 7)),
    (0x063, ForteNumber::new(4, 8)),
    (0x0C3, ForteNumber::new(4, 9)),
    (0x40B, ForteNumber::new(4, 10)),
    (0x02B, ForteNumber::new(4, 11)),
    (0x20B, ForteNumber::new(4, 12)),
    (0x04B, ForteNumber::new(4, 13)),
    (0x10B, ForteNumber::new(4, 14)),
    (0x053, ForteNumber::new_z(4, 15)),
    (0x0A3, ForteNumber::new(4, 16)),
    (0x213, ForteNumber::new(4, 17)),
    (0x093, ForteNumber::new(4, 18)),
    (0x113, ForteNumber::new(4, 19)),
    (0x123, ForteNumber::new(4, 20)),
    (0x055, ForteNumber::new(4, 21)),
    (0x095, ForteNumber::new(4, 22)),
    (0x0A5, ForteNumber::new(4, 23)),
    (0x115, ForteNumber::new(4, 24)),
    (0x145, ForteNumber::new(4, 25)),
    (0x225, ForteNumber::new(4, 26)),
    (0x125, ForteNumber::new(4, 27)),
    (0x249, ForteNumber::new(4, 28)), // diminished seventh
    (0x08B, ForteNumber::new_z(4, 29)),
    // Cardinality 5 – pentachords
    (0x01F, ForteNumber::new(5, 1)),
    (0x02F, ForteNumber::new(5, 2)),
    (0x037, ForteNumber::new(5, 3)),
    (0x04F, ForteNumber::new(5, 4)),
    (0x08F, ForteNumber::new(5, 5)),
    (0x067, ForteNumber::new(5, 6)),
    (0x0C7, ForteNumber::new(5, 7)),
    (0x417, ForteNumber::new(5, 8)),
    (0x057, ForteNumber::new(5, 9)),
    (0x05B, ForteNumber::new(5, 10)),
    (0x217, ForteNumber::new(5, 11)),
    (0x06B, ForteNumber::new_z(5, 12)),
    (0x117, ForteNumber::new(5, 13)),
    (0x0A7, ForteNumber::new(5, 14)),
    (0x147, ForteNumber::new(5, 15)),
    (0x09B, ForteNumber::new(5, 16)),
    (0x11B, ForteNumber::new_z(5, 17)),
    (0x30B, ForteNumber::new_z(5, 18)),
    (0x0CB, ForteNumber::new(5, 19)),
    (0x18B, ForteNumber::new(5, 20)),
    (0x133, ForteNumber::new(5, 21)),
    (0x193, ForteNumber::new(5, 22)),
    (0x42B, ForteNumber::new(5, 23)),
    (0x0AB, ForteNumber::new(5, 24)),
    (0x44B, ForteNumber::new(5, 25)),
    (0x22B, ForteNumber::new(5, 26)),
    (0x12B, ForteNumber::new(5, 27)),
    (0x28B, ForteNumber::new(5, 28)),
    (0x14B, ForteNumber::new(5, 29)),
    (0x153, ForteNumber::new(5, 30)),
    (0x24B, ForteNumber::new(5, 31)),
    (0x253, ForteNumber::new(5, 32)),
    (0x155, ForteNumber::new(5, 33)),
    (0x255, ForteNumber::new(5, 34)),
    (0x295, ForteNumber::new(5, 35)), // pentatonic
    (0x097, ForteNumber::new_z(5, 36)),
    (0x227, ForteNumber::new_z(5, 37)),
    (0x127, ForteNumber::new_z(5, 38)),
    // Cardinality 6 – hexachords (15 Z-pairs in the published catalogue)
    (0x03F, ForteNumber::new(6, 1)),
    (0x05F, ForteNumber::new(6, 2)),
    (0x06F, ForteNumber::new_z(6, 3)),
    (0x077, ForteNumber::new_z(6, 4)),
    (0x0CF, ForteNumber::new(6, 5)),
    (0x0E7, ForteNumber::new_z(6, 6)),
    (0x1C7, ForteNumber::new(6, 7)),
    (0x42F, ForteNumber::new(6, 8)),
    (0x0AF, ForteNumber::new(6, 9)),
    (0x437, ForteNumber::new_z(6, 10)),
    (0x0B7, ForteNumber::new_z(6, 11)),
    (0x0D7, ForteNumber::new_z(6, 12)),
    (0x0DB, ForteNumber::new_z(6, 13)),
    (0x237, ForteNumber::new(6, 14)),
    (0x137, ForteNumber::new(6, 15)),
    (0x317, ForteNumber::new(6, 16)),
    (0x197, ForteNumber::new_z(6, 17)),
    (0x1A7, ForteNumber::new(6, 18)),
    (0x19B, ForteNumber::new_z(6, 19)),
    (0x333, ForteNumber::new(6, 20)),
    (0x457, ForteNumber::new(6, 21)),
    (0x157, ForteNumber::new(6, 22)),
    (0x45B, ForteNumber::new_z(6, 23)),
    (0x15B, ForteNumber::new_z(6, 24)),
    (0x16B, ForteNumber::new_z(6, 25)),
    (0x1AB, ForteNumber::new_z(6, 26)),
    (0x25B, ForteNumber::new(6, 27)),
    (0x26B, ForteNumber::new_z(6, 28)),
    (0x34B, ForteNumber::new_z(6, 29)),
    (0x2CB, ForteNumber::new(6, 30)),
    (0x32B, ForteNumber::new(6, 31)),
    (0x52B, ForteNumber::new(6, 32)),
    (0x4AB, ForteNumber::new(6, 33)),
    (0x2AB, ForteNumber::new(6, 34)),
    (0x555, ForteNumber::new(6, 35)), // whole-tone hexachord
    (0x09F, ForteNumber::new_z(6, 36)),
    (0x11F, ForteNumber::new_z(6, 37)),
    (0x18F, ForteNumber::new_z(6, 38)),
    (0x22F, ForteNumber::new_z(6, 39)),
    (0x12F, ForteNumber::new_z(6, 40)),
    (0x14F, ForteNumber::new_z(6, 41)),
    (0x24F, ForteNumber::new_z(6, 42)),
    (0x167, ForteNumber::new_z(6, 43)),
    (0x267, ForteNumber::new_z(6, 44)),
    (0x497, ForteNumber::new_z(6, 45)),
    (0x257, ForteNumber::new_z(6, 46)),
    (0x297, ForteNumber::new_z(6, 47)),
    (0x2A7, ForteNumber::new_z(6, 48)),
    (0x29B, ForteNumber::new_z(6, 49)),
    (0x4CB, ForteNumber::new_z(6, 50)),
    // Cardinality 7 – heptachords
    (0x07F, ForteNumber::new(7, 1)),
    (0x0BF, ForteNumber::new(7, 2)),
    (0x13F, ForteNumber::new(7, 3)),
    (0x0DF, ForteNumber::new(7, 4)),
    (0x0EF, ForteNumber::new(7, 5)),
    (0x19F, ForteNumber::new(7, 6)),
    (0x1CF, ForteNumber::new(7, 7)),
    (0x45F, ForteNumber::new(7, 8)),
    (0x15F, ForteNumber::new(7, 9)),
    (0x25F, ForteNumber::new(7, 10)),
    (0x46F, ForteNumber::new(7, 11)),
    (0x29F, ForteNumber::new_z(7, 12)),
    (0x177, ForteNumber::new(7, 13)),
    (0x1AF, ForteNumber::new(7, 14)),
    (0x1D7, ForteNumber::new(7, 15)),
    (0x26F, ForteNumber::new(7, 16)),
    (0x277, ForteNumber::new_z(7, 17)),
    (0x32F, ForteNumber::new_z(7, 18)),
    (0x2CF, ForteNumber::new(7, 19)),
    (0x397, ForteNumber::new(7, 20)),
    (0x337, ForteNumber::new(7, 21)),
    (0x367, ForteNumber::new(7, 22)),
    (0x4AF, ForteNumber::new(7, 23)),
    (0x2AF, ForteNumber::new(7, 24)),
    (0x4B7, ForteNumber::new(7, 25)),
    (0x537, ForteNumber::new(7, 26)),
    (0x2B7, ForteNumber::new(7, 27)),
    (0x4D7, ForteNumber::new(7, 28)),
    (0x2D7, ForteNumber::new(7, 29)),
    (0x357, ForteNumber::new(7, 30)),
    (0x2DB, ForteNumber::new(7, 31)),
    (0x35B, ForteNumber::new(7, 32)),
    (0x557, ForteNumber::new(7, 33)),
    (0x55B, ForteNumber::new(7, 34)),
    (0x56B, ForteNumber::new(7, 35)), // diatonic heptachord (major scale)
    (0x16F, ForteNumber::new_z(7, 36)),
    (0x637, ForteNumber::new_z(7, 37)),
    (0x1B7, ForteNumber::new_z(7, 38)),
    // Cardinality 8 – octachords (complements of tetrachords)
    (0x0FF, ForteNumber::new(8, 1)),
    (0x17F, ForteNumber::new(8, 2)),
    (0x27F, ForteNumber::new(8, 3)),
    (0x1BF, ForteNumber::new(8, 4)),
    (0x1DF, ForteNumber::new(8, 5)),
    (0x1EF, ForteNumber::new(8, 6)),
    (0x33F, ForteNumber::new(8, 7)),
    (0x39F, ForteNumber::new(8, 8)),
    (0x3CF, ForteNumber::new(8, 9)),
    (0x4BF, ForteNumber::new(8, 10)),
    (0x2BF, ForteNumber::new(8, 11)),
    (0x4DF, ForteNumber::new(8, 12)),
    (0x2DF, ForteNumber::new(8, 13)),
    (0x4EF, ForteNumber::new(8, 14)),
    (0x35F, ForteNumber::new_z(8, 15)),
    (0x3AF, ForteNumber::new(8, 16)),
    (0x66F, ForteNumber::new(8, 17)),
    (0x36F, ForteNumber::new(8, 18)),
    (0x377, ForteNumber::new(8, 19)),
    (0x3B7, ForteNumber::new(8, 20)),
    (0x55F, ForteNumber::new(8, 21)),
    (0x56F, ForteNumber::new(8, 22)),
    (0x5AF, ForteNumber::new(8, 23)),
    (0x577, ForteNumber::new(8, 24)),
    (0x5D7, ForteNumber::new(8, 25)),
    (0x6B7, ForteNumber::new(8, 26)),
    (0x5B7, ForteNumber::new(8, 27)),
    (0x6DB, ForteNumber::new(8, 28)),
    (0x2EF, ForteNumber::new_z(8, 29)),
    // Cardinality 9 – nonachords
    (0x1FF, ForteNumber::new(9, 1)),
    (0x2FF, ForteNumber::new(9, 2)),
    (0x37F, ForteNumber::new(9, 3)),
    (0x3BF, ForteNumber::new(9, 4)),
    (0x3DF, ForteNumber::new(9, 5)),
    (0x57F, ForteNumber::new(9, 6)),
    (0x5BF, ForteNumber::new(9, 7)),
    (0x5DF, ForteNumber::new(9, 8)),
    (0x5EF, ForteNumber::new(9, 9)),
    (0x6DF, ForteNumber::new(9, 10)),
    (0x6EF, ForteNumber::new(9, 11)),
    (0x777, ForteNumber::new(9, 12)),
    // Cardinality 10 – decachords
    (0x3FF, ForteNumber::new(10, 1)),
    (0x5FF, ForteNumber::new(10, 2)),
    (0x6FF, ForteNumber::new(10, 3)),
    (0x77F, ForteNumber::new(10, 4)),
    (0x7BF, ForteNumber::new(10, 5)),
    (0x7DF, ForteNumber::new(10, 6)),
    // Cardinality 11 – undecachord
    (0x7FF, ForteNumber::new(11, 1)),
    // Cardinality 12 – chromatic aggregate
    (0xFFF, ForteNumber::new(12, 1)),
];

/// `(PcSet, ForteNumber)` pairs for all 224 D₁₂ orbits — pairs come in the same
/// order as [`FORTE_TABLE`], which groups by cardinality then Forte ordinal.
pub fn all_forte_numbers() -> &'static [(PcSet, ForteNumber)] {
    static CACHE: OnceLock<Vec<(PcSet, ForteNumber)>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            FORTE_TABLE
                .iter()
                .map(|&(mask, fn_)| (PcSet::new(mask), fn_))
                .collect()
        })
        .as_slice()
}

/// Forte set-class label for `x`. Computes [`bracelet_prime_form`] then looks
/// it up. Returns `Some` for every input — [`None`] would indicate a missing
/// entry in [`FORTE_TABLE`] and is checked by the test suite.
pub fn forte_number(x: PcSet) -> Option<ForteNumber> {
    static LOOKUP: OnceLock<HashMap<u16, ForteNumber>> = OnceLock::new();
    let map = LOOKUP.get_or_init(|| FORTE_TABLE.iter().copied().collect());
    map.get(&bracelet_prime_form(x).raw()).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::all_prime_forms;
    use std::collections::HashSet;

    #[test]
    fn table_has_224_entries() {
        assert_eq!(FORTE_TABLE.len(), 224);
        assert_eq!(all_forte_numbers().len(), 224);
    }

    #[test]
    fn table_has_no_duplicate_masks() {
        let mut seen = HashSet::new();
        for (mask, _) in FORTE_TABLE {
            assert!(seen.insert(*mask), "duplicate mask 0x{mask:03X}");
        }
    }

    #[test]
    fn table_has_no_duplicate_forte_numbers() {
        let mut seen = HashSet::new();
        for (_, fn_) in FORTE_TABLE {
            assert!(seen.insert(*fn_), "duplicate Forte number {fn_}");
        }
    }

    #[test]
    fn every_prime_form_has_a_forte_number() {
        for &pf in all_prime_forms() {
            let fn_ = forte_number(pf);
            assert!(
                fn_.is_some(),
                "prime form 0x{:03X} missing Forte number",
                pf.raw()
            );
            assert_eq!(
                fn_.unwrap().cardinality as u32,
                pf.cardinality(),
                "cardinality mismatch for 0x{:03X}",
                pf.raw()
            );
        }
    }

    #[test]
    fn every_forte_mask_is_a_bracelet_prime_form() {
        // Guard against Wikipedia-vs-D12 convention drift: every mask we list
        // must be a fixed point of bracelet_prime_form.
        for (mask, _) in FORTE_TABLE {
            let pf = bracelet_prime_form(PcSet::new(*mask));
            assert_eq!(pf.raw(), *mask, "0x{mask:03X} is not a bracelet prime form");
        }
    }

    #[test]
    fn forte_number_is_d12_invariant() {
        use crate::action::Action;
        use crate::dihedral::DihedralElement;

        let probes = [0x089u16, 0x091, 0x111, 0x249, 0x555, 0x56B, 0x08B, 0x053];
        for mask in probes {
            let x = PcSet::new(mask);
            let base = forte_number(x).unwrap();
            for reflected in [false, true] {
                for rotation in 0u8..12 {
                    let g = DihedralElement {
                        rotation,
                        reflected,
                    };
                    let y = g.apply(x);
                    assert_eq!(
                        forte_number(y).unwrap(),
                        base,
                        "mask 0x{:03X} under g={:?} got {:?}",
                        mask,
                        g,
                        forte_number(y)
                    );
                }
            }
        }
    }

    #[test]
    fn counts_per_cardinality_match_forte_catalogue() {
        let mut counts = [0u32; 13];
        for (_, fn_) in FORTE_TABLE {
            counts[fn_.cardinality as usize] += 1;
        }
        // 0-1, 1-1, 2-{1..6}, 3-{1..12}, 4-{1..29}, 5-{1..38}, 6-{1..50},
        // 7-{1..38}, 8-{1..29}, 9-{1..12}, 10-{1..6}, 11-1, 12-1.
        assert_eq!(counts, [1, 1, 6, 12, 29, 38, 50, 38, 29, 12, 6, 1, 1]);
    }

    #[test]
    fn display_formats_match_forte_convention() {
        assert_eq!(ForteNumber::new(3, 11).to_string(), "3-11");
        assert_eq!(ForteNumber::new_z(4, 15).to_string(), "4-Z15");
        assert_eq!(ForteNumber::new(6, 35).to_string(), "6-35");
        assert_eq!(ForteNumber::new_z(6, 50).to_string(), "6-Z50");
    }

    // --- Known fixtures from the task specification ---------------------

    #[test]
    fn empty_set_is_0_dash_1() {
        let fn_ = forte_number(PcSet::empty()).unwrap();
        assert_eq!(fn_, ForteNumber::new(0, 1));
        assert_eq!(fn_.to_string(), "0-1");
    }

    #[test]
    fn chromatic_aggregate_is_12_dash_1() {
        let fn_ = forte_number(PcSet::chromatic()).unwrap();
        assert_eq!(fn_, ForteNumber::new(12, 1));
        assert_eq!(fn_.to_string(), "12-1");
    }

    #[test]
    fn major_triad_is_3_dash_11() {
        let fn_ = forte_number(PcSet::from_pcs([0, 4, 7])).unwrap();
        assert_eq!(fn_, ForteNumber::new(3, 11));
        assert_eq!(fn_.to_string(), "3-11");
    }

    #[test]
    fn minor_triad_is_3_dash_11() {
        // D12 merges Forte's 3-11A (minor) and 3-11B (major) into one orbit.
        let fn_ = forte_number(PcSet::from_pcs([0, 3, 7])).unwrap();
        assert_eq!(fn_, ForteNumber::new(3, 11));
    }

    #[test]
    fn major_and_minor_triad_share_forte_number() {
        let major = forte_number(PcSet::from_pcs([0, 4, 7]));
        let minor = forte_number(PcSet::from_pcs([0, 3, 7]));
        assert_eq!(major, minor);
    }

    #[test]
    fn augmented_triad_is_3_dash_12() {
        let fn_ = forte_number(PcSet::from_pcs([0, 4, 8])).unwrap();
        assert_eq!(fn_, ForteNumber::new(3, 12));
    }

    #[test]
    fn diminished_seventh_is_4_dash_28() {
        let fn_ = forte_number(PcSet::from_pcs([0, 3, 6, 9])).unwrap();
        assert_eq!(fn_, ForteNumber::new(4, 28));
        assert_eq!(fn_.to_string(), "4-28");
    }

    #[test]
    fn whole_tone_hexachord_is_6_dash_35() {
        let fn_ = forte_number(PcSet::from_pcs([0, 2, 4, 6, 8, 10])).unwrap();
        assert_eq!(fn_, ForteNumber::new(6, 35));
    }

    #[test]
    fn diatonic_heptachord_is_7_dash_35() {
        // {0,1,3,5,6,8,10} is the bracelet prime form of the diatonic class
        // (raw mask 0x56B = 1+2+8+32+64+256+1024).
        let pf = PcSet::from_pcs([0, 1, 3, 5, 6, 8, 10]);
        assert_eq!(pf.raw(), 0x56B);
        assert_eq!(forte_number(pf).unwrap(), ForteNumber::new(7, 35));
        // The standard spelling of C major reaches the same class via D12.
        let cmaj = PcSet::from_pcs([0, 2, 4, 5, 7, 9, 11]);
        assert_eq!(forte_number(cmaj), Some(ForteNumber::new(7, 35)));
    }

    #[test]
    fn z_related_pairs_are_distinct() {
        // Every Z-pair must produce two different Forte numbers because D12
        // does not merge them — they are the whole point of the Z relation.
        // Each (mask_a, mask_b) pair is a canonical same-cardinality Z-pair.
        let pairs: &[(u16, u16, &str, &str)] = &[
            (0x053, 0x08B, "4-Z15", "4-Z29"),
            (0x06B, 0x097, "5-Z12", "5-Z36"),
            (0x11B, 0x227, "5-Z17", "5-Z37"),
            (0x30B, 0x127, "5-Z18", "5-Z38"),
            (0x06F, 0x09F, "6-Z3", "6-Z36"),
            (0x077, 0x11F, "6-Z4", "6-Z37"),
            (0x0E7, 0x18F, "6-Z6", "6-Z38"),
            (0x437, 0x22F, "6-Z10", "6-Z39"),
            (0x0B7, 0x12F, "6-Z11", "6-Z40"),
            (0x0D7, 0x14F, "6-Z12", "6-Z41"),
            (0x0DB, 0x24F, "6-Z13", "6-Z42"),
            (0x197, 0x167, "6-Z17", "6-Z43"),
            (0x19B, 0x267, "6-Z19", "6-Z44"),
            (0x45B, 0x497, "6-Z23", "6-Z45"),
            (0x15B, 0x257, "6-Z24", "6-Z46"),
            (0x16B, 0x297, "6-Z25", "6-Z47"),
            (0x1AB, 0x2A7, "6-Z26", "6-Z48"),
            (0x26B, 0x29B, "6-Z28", "6-Z49"),
            (0x34B, 0x4CB, "6-Z29", "6-Z50"),
            (0x29F, 0x16F, "7-Z12", "7-Z36"),
            (0x277, 0x637, "7-Z17", "7-Z37"),
            (0x32F, 0x1B7, "7-Z18", "7-Z38"),
            (0x35F, 0x2EF, "8-Z15", "8-Z29"),
        ];
        for (a, b, label_a, label_b) in pairs.iter().copied() {
            let na = forte_number(PcSet::new(a)).unwrap();
            let nb = forte_number(PcSet::new(b)).unwrap();
            assert_eq!(
                na.to_string(),
                label_a,
                "0x{a:03X} should be {label_a}, got {na}"
            );
            assert_eq!(
                nb.to_string(),
                label_b,
                "0x{b:03X} should be {label_b}, got {nb}"
            );
            assert_ne!(na, nb, "Z-partners {label_a} and {label_b} collided");
            assert!(na.z && nb.z, "Z flag missing on {label_a}/{label_b}");
            assert_eq!(
                na.cardinality, nb.cardinality,
                "Z-pair cardinality mismatch"
            );
        }
    }

    #[test]
    fn all_forte_numbers_iterates_every_pair() {
        let table = all_forte_numbers();
        assert_eq!(table.len(), 224);
        for &(pf, fn_) in table {
            assert_eq!(forte_number(pf), Some(fn_));
        }
    }
}
