//! Discrete Fourier transform of pitch-class sets — the Amiot / Quinn / Lewin
//! "Fourier space" of `Z/12`.
//!
//! `F_k(S) = Σ_{p∈S} e^{-2πikp/12}`, `k = 0..6` (k = 7..11 are conjugates of 5..1,
//! so 0..6 is the complete independent spectrum). By the pitch-class instance of
//! the Wiener–Khinchin theorem (Amiot 2016), the interval-class vector determines
//! **exactly** the magnitudes `|F_k|` — so homometric / Z-related sets share all
//! magnitudes and are separated *only* by phase. `icv` / `grothendieck_delta`
//! (which key on ICV) provably cannot tell the 23 Z-related pairs apart; the
//! phase information here can.
//!
//! Design notes (see docs/plans/2026-07-21-feat-dft-phase-invariants.md):
//! - Coefficients come from an exact 12-root table, not per-element `cos/sin`,
//!   so a true-zero coefficient stays at ~1e-15 instead of acquiring a
//!   random-looking phase.
//! - [`phase_aligned_similarity`] maximises a magnitude-weighted cross-correlation
//!   over the whole dihedral group D₁₂ in complex arithmetic — it never extracts a
//!   phase, so nil coefficients contribute ~0 weight by construction. With
//!   Plancherel weights it equals the normalised maximal common-tone overlap
//!   (Lewin 1959), which the tests use as an independent integer oracle.

use crate::pc_set::PcSet;

/// Complex number as `(re, im)`. `num-complex` is intentionally NOT a dependency
/// of this leaf crate (whose deps are only `thiserror` + `ix-search`).
pub type Cf = (f64, f64);

/// Zero-magnitude threshold `τ`. A nonzero coefficient satisfies
/// `|F_k| ≥ 1/12³ ≈ 5.8e-4` (each `F_k ∈ Z[ζ₁₂]`); exact-root rounding noise is
/// `≤ ~3e-15`. `1e-6` sits inside that ~11-order gap, so it can never
/// misclassify a genuine zero.
pub const ZERO_TOL: f64 = 1e-6;

const SQRT3_2: f64 = 0.866_025_403_784_438_6;

/// The twelve roots of unity under the forward kernel: `ROOTS[j] = e^{-2πij/12}`,
/// `j = 0..11`. Exact literals (coords in `{0, ±½, ±1, ±√3/2}`).
const ROOTS: [Cf; 12] = [
    (1.0, 0.0),
    (SQRT3_2, -0.5),
    (0.5, -SQRT3_2),
    (0.0, -1.0),
    (-0.5, -SQRT3_2),
    (-SQRT3_2, -0.5),
    (-1.0, 0.0),
    (-SQRT3_2, 0.5),
    (-0.5, SQRT3_2),
    (0.0, 1.0),
    (0.5, SQRT3_2),
    (SQRT3_2, 0.5),
];

/// Plancherel weights: `k = 1..5` each appear twice in the full 12-point spectrum
/// (conjugate symmetry), `k = 6` once; `k = 0` (= cardinality) is excluded so it
/// does not dilute shape. With these weights [`phase_aligned_similarity`] has a
/// closed form as normalised common-tone overlap.
const W: [f64; 7] = [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0];

#[inline]
fn cmul(a: Cf, b: Cf) -> Cf {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn conj(a: Cf) -> Cf {
    (a.0, -a.1)
}

/// `Re(a · conj(b)) = a.re·b.re + a.im·b.im` — the magnitude-weighted cosine kernel.
#[inline]
fn re_dot(a: Cf, b: Cf) -> f64 {
    a.0 * b.0 + a.1 * b.1
}

/// Fourier coefficients `F_0..F_6` of a pitch-class set. `F_0` is the cardinality
/// (real); `F_6` is always real (`= #even − #odd`).
pub fn dft(set: PcSet) -> [Cf; 7] {
    let mut f = [(0.0, 0.0); 7];
    for p in set.iter_pcs() {
        for (k, fk) in f.iter_mut().enumerate() {
            let (re, im) = ROOTS[((k as u32 * p as u32) % 12) as usize];
            fk.0 += re;
            fk.1 += im;
        }
    }
    f
}

/// Magnitudes `|F_0|..|F_6|`. Determined by the ICV (Amiot's theorem), hence
/// equal for any two Z-related sets.
pub fn dft_magnitudes(set: PcSet) -> [f64; 7] {
    dft(set).map(|(re, im)| (re * re + im * im).sqrt())
}

/// Phases `arg(F_0)..arg(F_6)` in radians, or `None` where `|F_k| < ZERO_TOL`.
/// The phase of a nil coefficient is undefined, and nil coefficients are common
/// (e.g. `F_1 = 0` for `{0,6}`, diminished sevenths, and whole-tone sets), so
/// consumers must treat `None` as "no phase here", not as `0.0`.
pub fn dft_phases(set: PcSet) -> [Option<f64>; 7] {
    dft(set).map(|(re, im)| {
        if (re * re + im * im).sqrt() < ZERO_TOL {
            None
        } else {
            Some(im.atan2(re))
        }
    })
}

/// Weighted spectral norm `√(Σ_{k=1..6} w_k |F_k|²)`; invariant under transposition
/// and inversion (both preserve every `|F_k|`).
fn wnorm(f: &[Cf; 7]) -> f64 {
    let mut s = 0.0;
    for k in 1..7 {
        s += W[k] * (f[k].0 * f[k].0 + f[k].1 * f[k].1);
    }
    s.sqrt()
}

/// Transposition/inversion-invariant Fourier similarity, in `[-1, 1]`.
///
/// Maximises the magnitude-weighted cross-correlation over all 24 elements of the
/// dihedral group D₁₂ (12 transpositions `T_n` × optional inversion `I`) in pure
/// complex arithmetic — no `atan2`, no branch on `|F_k| < τ`, because near-zero
/// coefficients carry near-zero weight automatically. `sim = 1` **iff** the sets
/// are D₁₂-equivalent (same set class); Z-related pairs are homometric but
/// inequivalent, so they score `sim < 1` for any correct implementation. Returns
/// `None` for the empty set or the full aggregate (zero spectral norm — no shape
/// to compare).
///
/// This is the honest refinement of [`crate::grothendieck::icv`]: ICV is
/// D₁₂-invariant and conflates Z-related pairs; this separates them.
pub fn phase_aligned_similarity(a: PcSet, b: PcSet) -> Option<f64> {
    let fa = dft(a);
    let fb = dft(b);
    let na = wnorm(&fa);
    let nb = wnorm(&fb);
    if na < ZERO_TOL || nb < ZERO_TOL {
        return None;
    }
    let mut best = f64::NEG_INFINITY;
    for inv in [false, true] {
        for n in 0..12u32 {
            // F_k(T_n·[I]·B) = ρ_{kn} · (conj F_k(B) if inverted else F_k(B)),
            // with ρ_j = ROOTS[j]. Compare against F_k(A) via Re(·conj·).
            let mut num = 0.0;
            for k in 1..7usize {
                let base = if inv { conj(fb[k]) } else { fb[k] };
                let fgb = cmul(ROOTS[((k as u32 * n) % 12) as usize], base);
                num += W[k] * re_dot(fa[k], fgb);
            }
            if num > best {
                best = num;
            }
        }
    }
    Some(best / (na * nb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grothendieck::{icv, z_related_pairs};
    use crate::orbit::all_prime_forms;

    // Independent, Fourier-free ICV oracle (combinatorial interval-class count).
    fn icv_of(set: PcSet) -> [u32; 6] {
        let pcs: Vec<u8> = set.iter_pcs().collect();
        let mut v = [0u32; 6];
        for i in 0..pcs.len() {
            for j in (i + 1)..pcs.len() {
                let d = (pcs[j] + 12 - pcs[i]) % 12;
                let ic = if d <= 6 { d } else { 12 - d };
                v[(ic - 1) as usize] += 1;
            }
        }
        v
    }

    fn transpose(set: PcSet, n: u8) -> PcSet {
        PcSet::from_pcs(set.iter_pcs().map(|p| (p + n) % 12))
    }

    fn invert(set: PcSet) -> PcSet {
        set.multiply(11) // M₁₁ = I₀ : p ↦ -p (mod 12)
    }

    fn cabs(z: Cf) -> f64 {
        (z.0 * z.0 + z.1 * z.1).sqrt()
    }

    // All 4096 subsets of Z/12 (raw masks 0..4096).
    fn all_subsets() -> impl Iterator<Item = PcSet> {
        (0u16..4096).map(PcSet::new)
    }

    /// F1 — the exact magnitude/ICV identity, over ALL 4096 subsets, against the
    /// independent combinatorial ICV (not the DFT). `|F_k|² = n + 2Σ_d icv_d·cos(2πkd/12)`.
    #[test]
    fn f1_magnitude_equals_icv_reconstruction_all_subsets() {
        let mut checked = 0u32;
        for s in all_subsets() {
            let mag = dft_magnitudes(s);
            let v = icv_of(s);
            let n = s.cardinality() as f64;
            for k in 0..7usize {
                let mut recon = n;
                for d in 1..=6usize {
                    recon += 2.0 * v[d - 1] as f64 * ROOTS[(k * d) % 12].0;
                }
                assert!(
                    (mag[k] * mag[k] - recon).abs() < 1e-9,
                    "F1 mismatch s={s} k={k}: |F_k|²={} recon={recon}",
                    mag[k] * mag[k]
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 4096, "must cover every subset");
    }

    /// P1 — `F_0` is the cardinality (real).
    #[test]
    fn p1_f0_is_cardinality() {
        for s in all_subsets() {
            let f = dft(s);
            assert!((f[0].0 - s.cardinality() as f64).abs() < 1e-12);
            assert!(f[0].1.abs() < 1e-12);
        }
    }

    /// P2 — magnitudes are transposition-invariant and `F_k(T_n S) = ρ_{kn}·F_k(S)`.
    #[test]
    fn p2_transposition_covariance() {
        for &s in all_prime_forms() {
            let f = dft(s);
            for n in 0..12u8 {
                let ft = dft(transpose(s, n));
                for k in 0..7usize {
                    assert!((cabs(ft[k]) - cabs(f[k])).abs() < 1e-12);
                    let expect = cmul(ROOTS[((k as u32 * n as u32) % 12) as usize], f[k]);
                    assert!((ft[k].0 - expect.0).abs() < 1e-12);
                    assert!((ft[k].1 - expect.1).abs() < 1e-12);
                }
            }
        }
    }

    /// P3 — inversion conjugates: `F_k(I₀ S) = conj(F_k(S))`.
    #[test]
    fn p3_inversion_conjugates() {
        for &s in all_prime_forms() {
            let f = dft(s);
            let fi = dft(invert(s));
            for k in 0..7usize {
                assert!((fi[k].0 - f[k].0).abs() < 1e-12);
                assert!((fi[k].1 + f[k].1).abs() < 1e-12);
            }
        }
    }

    /// P5 — `F_6` is real for every subset.
    #[test]
    fn p5_f6_is_real() {
        for s in all_subsets() {
            assert!(dft(s)[6].1.abs() < 1e-12);
        }
    }

    /// P6 — `sim` is a proper similarity: reflexive at 1, symmetric, and 1 on the
    /// whole D₁₂ orbit of a set (invariance is the point).
    #[test]
    fn p6_sim_is_a_proper_similarity() {
        for &s in all_prime_forms() {
            if s.cardinality() == 0 || s.cardinality() == 12 {
                continue;
            }
            let self_sim = phase_aligned_similarity(s, s).unwrap();
            assert!((self_sim - 1.0).abs() < 1e-9, "sim(S,S)={self_sim}");
            for n in 0..12u8 {
                for g in [transpose(s, n), invert(transpose(s, n))] {
                    let sim = phase_aligned_similarity(s, g).unwrap();
                    assert!((sim - 1.0).abs() < 1e-9, "sim(S, g·S)={sim}");
                    // symmetric
                    let rev = phase_aligned_similarity(g, s).unwrap();
                    assert!((sim - rev).abs() < 1e-12);
                }
            }
        }
    }

    // Independent common-tone oracle for P7 (no DFT code path).
    fn max_common_tones(a: PcSet, b: PcSet) -> u32 {
        let mut best = 0;
        for inv in [false, true] {
            let bb = if inv { invert(b) } else { b };
            for n in 0..12u8 {
                let ct = (a.raw() & transpose(bb, n).raw()).count_ones();
                best = best.max(ct);
            }
        }
        best
    }

    /// P7 — the closed form: `sim == (12·maxCT − |A||B|)/√((12|A|−|A|²)(12|B|−|B|²))`.
    #[test]
    fn p7_sim_equals_common_tone_closed_form() {
        let forms = all_prime_forms();
        for (i, &a) in forms.iter().enumerate() {
            if a.cardinality() == 0 || a.cardinality() == 12 {
                continue;
            }
            // Pair each form with a spread of others (every 7th) to keep it O(224·32).
            for &b in forms.iter().skip(i).step_by(7) {
                if b.cardinality() == 0 || b.cardinality() == 12 {
                    continue;
                }
                let sim = phase_aligned_similarity(a, b).unwrap();
                let (na, nb) = (a.cardinality() as f64, b.cardinality() as f64);
                let ct = max_common_tones(a, b) as f64;
                let closed =
                    (12.0 * ct - na * nb) / ((12.0 * na - na * na) * (12.0 * nb - nb * nb)).sqrt();
                assert!((sim - closed).abs() < 1e-9, "P7 sim={sim} closed={closed}");
            }
        }
    }

    /// P8 — degenerate inputs return `None`; a nil coefficient yields a `None` phase.
    #[test]
    fn p8_degenerate_inputs() {
        assert_eq!(
            phase_aligned_similarity(PcSet::empty(), PcSet::from_pcs([0, 4, 7])),
            None
        );
        assert_eq!(
            phase_aligned_similarity(PcSet::chromatic(), PcSet::from_pcs([0, 4, 7])),
            None
        );
        // [0,1,2,5,6,7] has F_4 = F_6 = 0 exactly (by symmetry) → None phase there.
        let ph = dft_phases(PcSet::from_pcs([0, 1, 2, 5, 6, 7]));
        assert!(ph[4].is_none(), "F_4 should be a nil coefficient");
        assert!(ph[6].is_none(), "F_6 should be a nil coefficient");
    }

    /// F3 (the ship gate) — the metric separates every one of the 23 Z-related
    /// pairs that ICV / `grothendieck_delta` cannot, with a wide margin.
    #[test]
    fn f3_separates_all_z_related_pairs() {
        let pairs = z_related_pairs();
        assert_eq!(
            pairs.len(),
            23,
            "the 23 12-TET Z-pairs (guards the generator)"
        );
        let mut separated = 0;
        let mut worst = f64::NEG_INFINITY;
        for (a, b) in pairs {
            // Precondition: ICV genuinely cannot tell them apart.
            assert_eq!(icv(a).total(), icv(b).total());
            assert_eq!(icv_of(a), icv_of(b), "Z-pair must share ICV");
            let sim = phase_aligned_similarity(a, b).unwrap();
            // Margin, not a bare `< 1.0`: the empirical worst case is exactly 2/3,
            // and a regression that degraded separation to 0.999 must also fail.
            assert!(sim <= 0.67, "Z-pair {a}/{b} not separated: sim={sim}");
            worst = worst.max(sim);
            separated += 1;
        }
        assert_eq!(separated, 23, "all 23 pairs separated");
        assert!(worst <= 0.67, "worst-case separation margin: sim={worst}");
    }
}
