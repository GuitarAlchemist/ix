//! Minimal invariant checkers + exemplar corpus for first-run firings.
//!
//! Phase 1: purely algebraic invariants on pitch-class sets, no external deps
//! (no `optick.index`, no voicing data, no C# bridge).
//!
//! The `PcSet(u16)` type here is a stand-in — `ix-bracelet` will provide the
//! canonical version. When that lands, callers of this module should be
//! migrated and this type removed.

use crate::coverage::{Exemplar, Firings};
use std::collections::{BTreeMap, BTreeSet};

/// 12-bit pitch-class mask. Bit `k` set ⟺ pitch-class `k` present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcSet(pub u16);

impl PcSet {
    pub fn new(mask: u16) -> Self {
        Self(mask & 0xFFF)
    }

    pub fn from_pcs<I: IntoIterator<Item = u8>>(pcs: I) -> Self {
        let mut m = 0u16;
        for pc in pcs {
            m |= 1 << (pc % 12);
        }
        Self(m & 0xFFF)
    }

    pub fn cardinality(self) -> u32 {
        self.0.count_ones()
    }

    pub fn contains(self, pc: u8) -> bool {
        (self.0 >> (pc % 12)) & 1 == 1
    }

    /// Rotation (transposition) by `n` semitones on the 12-bit circle.
    pub fn transpose(self, n: u8) -> Self {
        let n = (n % 12) as u16;
        if n == 0 {
            return self;
        }
        let m = self.0 as u32;
        let rotated = ((m << n) | (m >> (12 - n))) & 0xFFF;
        Self(rotated as u16)
    }

    /// Inversion about pitch-class 0 (reflection on the 12-cycle).
    /// PC `k` → PC `(12 - k) % 12`. Bit 0 fixed; bits 1..12 swap with their complements.
    pub fn invert(self) -> Self {
        let mut out = 0u16;
        for pc in 0..12 {
            if (self.0 >> pc) & 1 == 1 {
                let mapped = (12 - pc) % 12;
                out |= 1 << mapped;
            }
        }
        Self(out & 0xFFF)
    }

    /// Interval-class vector: 6 entries, `icv[k]` = number of pairs at interval class `k+1`
    /// (ic 1..6). Index 0 unused; returned as `[u32; 7]` for 1-indexed natural access.
    pub fn icv(self) -> [u32; 7] {
        let mut v = [0u32; 7];
        for a in 0..12u8 {
            if !self.contains(a) {
                continue;
            }
            for b in (a + 1)..12 {
                if !self.contains(b) {
                    continue;
                }
                let diff = b - a;
                let ic = if diff <= 6 { diff } else { 12 - diff };
                if ic >= 1 {
                    v[ic as usize] += 1;
                }
            }
        }
        v
    }

    /// Smallest non-zero `n` in 1..12 such that `transpose(n) == self`, or `None`
    /// if the set has no non-trivial rotational symmetry.
    pub fn rotational_symmetry(self) -> Option<u8> {
        (1..12u8).find(|&n| self.transpose(n) == self)
    }
}

// ---------------------------------------------------------------------------
// Exemplar corpus
// ---------------------------------------------------------------------------

struct Spec {
    id: &'static str,
    description: &'static str,
    kind: &'static str,
    pcs: PcSet,
}

fn corpus() -> Vec<Spec> {
    let mut out = Vec::new();

    out.push(Spec {
        id: "empty",
        description: "empty set",
        kind: "EdgeCase",
        pcs: PcSet::new(0),
    });
    out.push(Spec {
        id: "singleton-C",
        description: "{C}",
        kind: "EdgeCase",
        pcs: PcSet::from_pcs([0]),
    });
    out.push(Spec {
        id: "chromatic",
        description: "chromatic aggregate",
        kind: "EdgeCase",
        pcs: PcSet::new(0xFFF),
    });

    // Twelve major triads (root moves through all pitch classes) — asymmetric.
    for root in 0..12u8 {
        let id = Box::leak(format!("major-triad-{root}").into_boxed_str());
        let desc = Box::leak(format!("major triad rooted at pc {root}").into_boxed_str());
        out.push(Spec {
            id,
            description: desc,
            kind: "Triad",
            pcs: PcSet::from_pcs([root, root + 4, root + 7]),
        });
    }

    // Twelve minor triads — asymmetric.
    for root in 0..12u8 {
        let id = Box::leak(format!("minor-triad-{root}").into_boxed_str());
        let desc = Box::leak(format!("minor triad rooted at pc {root}").into_boxed_str());
        out.push(Spec {
            id,
            description: desc,
            kind: "Triad",
            pcs: PcSet::from_pcs([root, root + 3, root + 7]),
        });
    }

    // Symmetric sets — these exercise the rotational-symmetry discriminator.
    out.push(Spec {
        id: "tritone-pair",
        description: "{C, F#} — T6-symmetric",
        kind: "Symmetric",
        pcs: PcSet::from_pcs([0, 6]),
    });
    out.push(Spec {
        id: "augmented-triad",
        description: "{C, E, G#} — T4-symmetric",
        kind: "Symmetric",
        pcs: PcSet::from_pcs([0, 4, 8]),
    });
    out.push(Spec {
        id: "diminished-seventh",
        description: "{C, Eb, F#, A} — T3-symmetric",
        kind: "Symmetric",
        pcs: PcSet::from_pcs([0, 3, 6, 9]),
    });
    out.push(Spec {
        id: "whole-tone-hexachord",
        description: "{C, D, E, F#, G#, A#} — T2-symmetric",
        kind: "Symmetric",
        pcs: PcSet::from_pcs([0, 2, 4, 6, 8, 10]),
    });

    // Diatonic scales and dominant chords — non-symmetric extensions to exercise ICV.
    out.push(Spec {
        id: "major-scale-C",
        description: "C major scale",
        kind: "Scale",
        pcs: PcSet::from_pcs([0, 2, 4, 5, 7, 9, 11]),
    });
    out.push(Spec {
        id: "minor-scale-C",
        description: "C natural minor scale",
        kind: "Scale",
        pcs: PcSet::from_pcs([0, 2, 3, 5, 7, 8, 10]),
    });
    out.push(Spec {
        id: "pentatonic-C",
        description: "C pentatonic",
        kind: "Scale",
        pcs: PcSet::from_pcs([0, 2, 4, 7, 9]),
    });
    out.push(Spec {
        id: "dom7-C",
        description: "C7 dominant seventh",
        kind: "Seventh",
        pcs: PcSet::from_pcs([0, 4, 7, 10]),
    });
    out.push(Spec {
        id: "maj7-C",
        description: "Cmaj7",
        kind: "Seventh",
        pcs: PcSet::from_pcs([0, 4, 7, 11]),
    });
    out.push(Spec {
        id: "m7-C",
        description: "Cm7",
        kind: "Seventh",
        pcs: PcSet::from_pcs([0, 3, 7, 10]),
    });

    out
}

// ---------------------------------------------------------------------------
// Invariant checkers
// ---------------------------------------------------------------------------

/// An invariant check that either holds (true) or doesn't (false) on an exemplar.
struct Check {
    id: u32,
    holds: fn(PcSet) -> bool,
}

/// Catalog invariant #16: T₁₂ on any PitchClassSet returns the same set.
fn check_16_t12_identity(x: PcSet) -> bool {
    x.transpose(12) == x
}

/// Catalog invariant #17: Double inversion is the identity on PitchClassSet.
fn check_17_double_inversion(x: PcSet) -> bool {
    x.invert().invert() == x
}

/// Catalog invariant #18: literal reading — `T_card(x) == x`. Fires only on sets
/// whose cardinality divides their rotational-symmetry period (rare — only the
/// symmetric sets plus empty/chromatic). This makes #18 a discriminator relative
/// to the universal invariants, which is what the rank analysis wants.
fn check_18_rotate_by_cardinality(x: PcSet) -> bool {
    let n = x.cardinality();
    if n == 0 {
        return true;
    }
    x.transpose(n as u8) == x
}

/// Catalog invariant #20: IntervalClassVector is palindrome-invariant under inversion.
/// Practically: ICV(I(x)) == ICV(x). This is a theorem of set theory, so it fires
/// on every exemplar where ICV is well-defined (which is all of them).
fn check_20_icv_inversion_invariant(x: PcSet) -> bool {
    x.icv() == x.invert().icv()
}

/// Catalog invariant #24: ICV has exactly 6 entries summing to n(n-1)/2 where
/// n = |x|. Indices 1..6 carry the counts; index 0 is unused/0.
fn check_24_icv_shape(x: PcSet) -> bool {
    let n = x.cardinality();
    let expected_sum = n * n.saturating_sub(1) / 2;
    let v = x.icv();
    let sum: u32 = v[1..=6].iter().sum();
    v[0] == 0 && sum == expected_sum
}

/// Catalog invariant #19: `PrimeFormId` is self-representative — `PrimeForm(x) == x`
/// iff `x` is already in prime form. Delegates to `ix-bracelet` for the canonical
/// representative. Fires on exemplars whose raw mask equals their D₁₂-orbit minimum.
fn check_19_prime_form_self_representative(x: PcSet) -> bool {
    let bx = ix_bracelet::PcSet::new(x.0);
    ix_bracelet::bracelet_prime_form(bx).raw() == x.0
}

/// Catalog invariant #14: Neo-Riemannian P, L, R are involutions on consonant
/// triads. Fires iff `x` is a consonant triad AND P(P(x)) == x AND L(L(x)) == x
/// AND R(R(x)) == x. Non-triads silently skip (not a violation, just out of scope).
fn check_14_plr_involutions(x: PcSet) -> bool {
    let bx = ix_bracelet::PcSet::new(x.0);
    let Some(px) = ix_bracelet::p(bx) else { return false };
    let Some(lx) = ix_bracelet::l(bx) else { return false };
    let Some(rx) = ix_bracelet::r(bx) else { return false };
    ix_bracelet::p(px) == Some(bx)
        && ix_bracelet::l(lx) == Some(bx)
        && ix_bracelet::r(rx) == Some(bx)
}

/// Catalog invariant #15: PLR composites match the named Slide and Nebenverwandt
/// operators — `S == L∘P∘R` and `N == R∘L∘P`. Fires on consonant triads, where all
/// five operators are defined. True by construction in `ix-bracelet`, so this
/// instrument wires the catalog row to "tested" without adding discriminating
/// power over #14 (a known strict duplicate).
fn check_15_plr_composites(x: PcSet) -> bool {
    let bx = ix_bracelet::PcSet::new(x.0);
    let (Some(rx), Some(px)) = (ix_bracelet::r(bx), ix_bracelet::p(bx)) else {
        return false;
    };
    let (Some(prx), Some(lpx)) = (ix_bracelet::p(rx), ix_bracelet::l(px)) else {
        return false;
    };
    let (Some(lprx), Some(rlpx)) = (ix_bracelet::l(prx), ix_bracelet::r(lpx)) else {
        return false;
    };
    ix_bracelet::s(bx) == Some(lprx) && ix_bracelet::n(bx) == Some(rlpx)
}

/// Catalog invariant #21: Triad templates have exactly 3 distinct pitch classes
/// after octave reduction. At the PC-set level this reduces to `|x| == 3`.
fn check_21_triad_cardinality(x: PcSet) -> bool {
    x.cardinality() == 3
}

/// Catalog invariant #22: Seventh-chord templates have exactly 4 distinct pitch
/// classes after octave reduction. At the PC-set level this reduces to `|x| == 4`.
fn check_22_seventh_cardinality(x: PcSet) -> bool {
    x.cardinality() == 4
}

fn checks() -> Vec<Check> {
    vec![
        Check { id: 14, holds: check_14_plr_involutions },
        Check { id: 15, holds: check_15_plr_composites },
        Check { id: 16, holds: check_16_t12_identity },
        Check { id: 17, holds: check_17_double_inversion },
        Check { id: 18, holds: check_18_rotate_by_cardinality },
        Check { id: 19, holds: check_19_prime_form_self_representative },
        Check { id: 20, holds: check_20_icv_inversion_invariant },
        Check { id: 21, holds: check_21_triad_cardinality },
        Check { id: 22, holds: check_22_seventh_cardinality },
        Check { id: 24, holds: check_24_icv_shape },
    ]
}

// ---------------------------------------------------------------------------
// Firings producer
// ---------------------------------------------------------------------------

/// Run the checker library against the built-in exemplar corpus and return the
/// `Firings` structure that `ix-invariant-coverage` consumes.
pub fn produce_firings() -> Firings {
    let corpus = corpus();
    let checks = checks();

    let exemplars: Vec<Exemplar> = corpus
        .iter()
        .map(|s| Exemplar {
            id: s.id.to_string(),
            description: s.description.to_string(),
            kind: s.kind.to_string(),
        })
        .collect();

    let mut fired: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    for check in &checks {
        let mut hits = BTreeSet::new();
        for spec in &corpus {
            if (check.holds)(spec.pcs) {
                hits.insert(spec.id.to_string());
            }
        }
        fired.insert(check.id, hits);
    }

    Firings { exemplars, fired }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_by_zero_is_identity() {
        let x = PcSet::from_pcs([0, 4, 7]);
        assert_eq!(x.transpose(0), x);
        assert_eq!(x.transpose(12), x);
    }

    #[test]
    fn transpose_rotates_bits_on_circle() {
        let x = PcSet::from_pcs([0]);
        assert_eq!(x.transpose(1), PcSet::from_pcs([1]));
        assert_eq!(x.transpose(11), PcSet::from_pcs([11]));
        let chromatic_like = PcSet::from_pcs([11]);
        assert_eq!(chromatic_like.transpose(1), PcSet::from_pcs([0]));
    }

    #[test]
    fn inversion_maps_each_pc_to_its_complement() {
        let x = PcSet::from_pcs([0]);
        assert_eq!(x.invert(), PcSet::from_pcs([0]));
        let y = PcSet::from_pcs([3]);
        assert_eq!(y.invert(), PcSet::from_pcs([9]));
        let triad = PcSet::from_pcs([0, 4, 7]);
        assert_eq!(triad.invert(), PcSet::from_pcs([0, 8, 5]));
    }

    #[test]
    fn double_inversion_is_identity_on_all_triads() {
        for root in 0..12u8 {
            let maj = PcSet::from_pcs([root, root + 4, root + 7]);
            assert_eq!(maj.invert().invert(), maj);
        }
    }

    #[test]
    fn icv_of_major_triad_has_expected_shape() {
        let maj = PcSet::from_pcs([0, 4, 7]);
        let v = maj.icv();
        // major triad intervals: 4 (M3), 3 (m3) → IC 4, 3; 7 → IC 5. Each pair once.
        assert_eq!(v[0], 0);
        assert_eq!(v[3], 1);
        assert_eq!(v[4], 1);
        assert_eq!(v[5], 1);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 0);
        assert_eq!(v[6], 0);
        let sum: u32 = v[1..=6].iter().sum();
        assert_eq!(sum, 3); // C(3,2) = 3
    }

    #[test]
    fn icv_is_invariant_under_inversion_for_sample_sets() {
        for pcs in [
            PcSet::from_pcs([0, 4, 7]),
            PcSet::from_pcs([0, 3, 7, 10]),
            PcSet::from_pcs([0, 2, 4, 5, 7, 9, 11]),
            PcSet::from_pcs([0, 3, 6, 9]),
        ] {
            assert_eq!(pcs.icv(), pcs.invert().icv());
        }
    }

    #[test]
    fn rotational_symmetry_detects_known_cases() {
        assert_eq!(PcSet::from_pcs([0, 3, 6, 9]).rotational_symmetry(), Some(3));
        assert_eq!(PcSet::from_pcs([0, 4, 8]).rotational_symmetry(), Some(4));
        assert_eq!(PcSet::from_pcs([0, 6]).rotational_symmetry(), Some(6));
        assert_eq!(PcSet::from_pcs([0, 2, 4, 6, 8, 10]).rotational_symmetry(), Some(2));
        assert_eq!(PcSet::from_pcs([0, 4, 7]).rotational_symmetry(), None);
    }

    #[test]
    fn universal_checks_fire_on_every_exemplar() {
        let corp = corpus();
        for spec in &corp {
            assert!(check_16_t12_identity(spec.pcs), "invariant 16 failed on {}", spec.id);
            assert!(check_17_double_inversion(spec.pcs), "invariant 17 failed on {}", spec.id);
            assert!(check_20_icv_inversion_invariant(spec.pcs), "invariant 20 failed on {}", spec.id);
            assert!(check_24_icv_shape(spec.pcs), "invariant 24 failed on {}", spec.id);
        }
    }

    #[test]
    fn rotate_by_cardinality_fires_only_when_symmetry_period_equals_cardinality() {
        // Literal reading of #18: T_card(x) == x.
        // Fires on empty (0-card no-op), chromatic (T_12 = id), and the whole-tone
        // hexachord (T_6 maps it to itself).
        assert!(check_18_rotate_by_cardinality(PcSet::new(0)));
        assert!(check_18_rotate_by_cardinality(PcSet::new(0xFFF)));
        assert!(check_18_rotate_by_cardinality(PcSet::from_pcs([0, 2, 4, 6, 8, 10])));
        // Does NOT fire on sets whose symmetry period ≠ cardinality:
        //   augmented triad {0,4,8}: T4-symmetric, card 3 — T_3 gives {3,7,11}, not self.
        //   diminished 7th {0,3,6,9}: T3-symmetric, card 4 — T_4 gives {4,7,10,1}.
        //   major triad: no rotational symmetry.
        assert!(!check_18_rotate_by_cardinality(PcSet::from_pcs([0, 4, 8])));
        assert!(!check_18_rotate_by_cardinality(PcSet::from_pcs([0, 3, 6, 9])));
        assert!(!check_18_rotate_by_cardinality(PcSet::from_pcs([0, 4, 7])));
    }

    #[test]
    fn produce_firings_yields_expected_exemplars_and_coverage_pattern() {
        let firings = produce_firings();
        // The corpus is deterministic; assert structural properties rather than exact counts.
        assert!(!firings.exemplars.is_empty());
        assert!(firings.fired.contains_key(&16));
        assert!(firings.fired.contains_key(&18));
        assert!(firings.fired.contains_key(&19));
        // Universal #16 should fire on every exemplar.
        assert_eq!(firings.fired[&16].len(), firings.exemplars.len());
        // Discriminator #18 fires on a strict subset.
        assert!(firings.fired[&18].len() < firings.exemplars.len());
        assert!(!firings.fired[&18].is_empty());
        // Prime-form discriminator #19 fires on a distinct subset — it must include
        // the minor triad {0,3,7} (canonical Forte 3-11 rep) but NOT the major
        // triad {0,4,7} (which maps to {0,3,7} under I).
        assert!(firings.fired[&19].contains("minor-triad-0"));
        assert!(!firings.fired[&19].contains("major-triad-0"));
    }

    #[test]
    fn prime_form_self_rep_distinguishes_major_from_minor_triad() {
        let maj = PcSet::from_pcs([0, 4, 7]);
        let min = PcSet::from_pcs([0, 3, 7]);
        assert!(!check_19_prime_form_self_representative(maj));
        assert!(check_19_prime_form_self_representative(min));
    }

    #[test]
    fn plr_involutions_fire_on_consonant_triads_only() {
        // All 24 consonant triads → fires.
        for root in 0..12u8 {
            let maj = PcSet::from_pcs([root, root + 4, root + 7]);
            let min = PcSet::from_pcs([root, root + 3, root + 7]);
            assert!(check_14_plr_involutions(maj), "maj at root {root}");
            assert!(check_14_plr_involutions(min), "min at root {root}");
        }
        // Non-triads → skips (returns false because P/L/R are None).
        assert!(!check_14_plr_involutions(PcSet::from_pcs([0, 4, 8]))); // augmented
        assert!(!check_14_plr_involutions(PcSet::from_pcs([0, 3, 6, 9]))); // dim7
        assert!(!check_14_plr_involutions(PcSet::new(0))); // empty
    }

    #[test]
    fn plr_composites_fire_on_consonant_triads_only() {
        for root in 0..12u8 {
            let maj = PcSet::from_pcs([root, root + 4, root + 7]);
            let min = PcSet::from_pcs([root, root + 3, root + 7]);
            assert!(check_15_plr_composites(maj), "maj at root {root}");
            assert!(check_15_plr_composites(min), "min at root {root}");
        }
        assert!(!check_15_plr_composites(PcSet::from_pcs([0, 4, 8]))); // augmented
        assert!(!check_15_plr_composites(PcSet::from_pcs([0, 3, 6, 9]))); // dim7
        assert!(!check_15_plr_composites(PcSet::new(0))); // empty
    }

    #[test]
    fn cardinality_checks_discriminate_triads_and_sevenths() {
        assert!(check_21_triad_cardinality(PcSet::from_pcs([0, 4, 7])));
        assert!(check_21_triad_cardinality(PcSet::from_pcs([0, 4, 8]))); // augmented: 3 PCs
        assert!(!check_21_triad_cardinality(PcSet::from_pcs([0, 6]))); // dyad
        assert!(!check_21_triad_cardinality(PcSet::from_pcs([0, 4, 7, 10]))); // dom7

        assert!(check_22_seventh_cardinality(PcSet::from_pcs([0, 4, 7, 10])));
        assert!(check_22_seventh_cardinality(PcSet::from_pcs([0, 3, 6, 9]))); // dim7
        assert!(!check_22_seventh_cardinality(PcSet::from_pcs([0, 4, 7])));
        assert!(!check_22_seventh_cardinality(PcSet::new(0)));
    }
}
