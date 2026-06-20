//! Twelve-tone serialism: **ordered** rows over `Z/12` and their transformations.
//!
//! The rest of this crate works with *unordered* pitch-class sets ([`PcSet`]) — the
//! atonal set-theory world (orbits, prime forms, Forte numbers). Serialism is the
//! complementary *ordered* world: a [`ToneRow`] is a permutation of all 12 pitch
//! classes, and the four classical transformations act on that ordering:
//!
//! - **Prime (P)** — the row, transposed: `Tₙ(row)[k] = (row[k] + n) mod 12`.
//! - **Inversion (I)** — intervals mirrored: `I₀(row)[k] = (12 − row[k]) mod 12`
//!   (the same `Iₙ(x) = (n − x) mod 12` convention [`crate::dihedral`] uses), then transposed.
//! - **Retrograde (R)** — the row reversed.
//! - **Retrograde-Inversion (RI)** — the inversion, reversed.
//!
//! [`ToneRow::matrix`] builds the classic 12×12 row matrix; [`ToneRow::all_forms`]
//! enumerates the 48 forms (4 transformations × 12 transpositions). Hexachordal
//! combinatoriality — the property Schoenberg exploited so a row and one of its
//! transforms jointly complete the aggregate without overlap — is decided by
//! [`ToneRow::combinatorial_prime_levels`] / [`ToneRow::combinatorial_inversion_levels`].

use core::fmt;

use crate::pc_set::PcSet;

/// Why a tone-row / multiplicative operation was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerialError {
    /// The input was not a permutation of all 12 pitch classes (duplicate, gap, or out of range).
    NotARow,
    /// The input slice did not have exactly 12 elements.
    WrongLength(usize),
    /// A multiplicative factor not coprime to 12 (only 1, 5, 7, 11 are bijections on `Z/12`).
    NotCoprime(u8),
}

impl fmt::Display for SerialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerialError::NotARow => {
                f.write_str("not a tone row: must be a permutation of the 12 pitch classes")
            }
            SerialError::WrongLength(n) => write!(f, "a tone row needs 12 pitch classes, got {n}"),
            SerialError::NotCoprime(m) => {
                write!(
                    f,
                    "multiplicative factor {m} is not coprime to 12 (use 1, 5, 7, or 11)"
                )
            }
        }
    }
}

impl std::error::Error for SerialError {}

/// The four classical row transformations, each carrying a transposition level `0..12`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowForm {
    /// Prime — the row transposed by `n`.
    Prime(u8),
    /// Inversion — the row inverted then transposed by `n`.
    Inversion(u8),
    /// Retrograde — `Prime(n)` reversed.
    Retrograde(u8),
    /// Retrograde-inversion — `Inversion(n)` reversed.
    RetrogradeInversion(u8),
}

impl fmt::Display for RowForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RowForm::Prime(n) => write!(f, "P{n}"),
            RowForm::Inversion(n) => write!(f, "I{n}"),
            RowForm::Retrograde(n) => write!(f, "R{n}"),
            RowForm::RetrogradeInversion(n) => write!(f, "RI{n}"),
        }
    }
}

/// An ordered twelve-tone row: a permutation of the 12 pitch classes of `Z/12`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToneRow([u8; 12]);

impl ToneRow {
    /// Build from 12 pitch classes, validating that they form a permutation of `0..12`
    /// (each pitch class exactly once). Values are taken as-is (must already be `< 12`).
    pub fn new(pcs: [u8; 12]) -> Result<Self, SerialError> {
        let mut seen = 0u16;
        for &p in &pcs {
            if p >= 12 {
                return Err(SerialError::NotARow);
            }
            let bit = 1u16 << p;
            if seen & bit != 0 {
                return Err(SerialError::NotARow);
            }
            seen |= bit;
        }
        // A 12-element array with no duplicates and all values < 12 is necessarily a permutation.
        Ok(Self(pcs))
    }

    /// Build from a slice; errors if it isn't exactly 12 elements or isn't a permutation.
    pub fn from_slice(pcs: &[u8]) -> Result<Self, SerialError> {
        let arr: [u8; 12] = pcs
            .try_into()
            .map_err(|_| SerialError::WrongLength(pcs.len()))?;
        Self::new(arr)
    }

    /// The 12 pitch classes in row order.
    #[inline]
    pub fn pcs(&self) -> &[u8; 12] {
        &self.0
    }

    /// Transpose: `Tₙ(row)[k] = (row[k] + n) mod 12`. (A bijection, so the result is a row.)
    pub fn transpose(&self, n: u8) -> ToneRow {
        let n = n % 12;
        let mut out = [0u8; 12];
        for (o, &p) in out.iter_mut().zip(self.0.iter()) {
            *o = (p + n) % 12;
        }
        ToneRow(out)
    }

    /// Invert about pitch class 0: `I₀(row)[k] = (12 − row[k]) mod 12`
    /// (the `Iₙ(x) = (n − x) mod 12` convention with `n = 0`). A bijection ⇒ still a row.
    pub fn invert(&self) -> ToneRow {
        let mut out = [0u8; 12];
        for (o, &p) in out.iter_mut().zip(self.0.iter()) {
            *o = (12 - p) % 12;
        }
        ToneRow(out)
    }

    /// Reverse the row order (the Retrograde of the prime).
    pub fn retrograde(&self) -> ToneRow {
        let mut out = self.0;
        out.reverse();
        ToneRow(out)
    }

    /// Realise one of the 48 [`RowForm`]s as a concrete row.
    ///
    /// Forms are defined relative to this row as `P₀`:
    /// `Pₙ = Tₙ(row)`, `Iₙ = Tₙ(I₀(row))`, `Rₙ = retrograde(Pₙ)`, `RIₙ = retrograde(Iₙ)`.
    pub fn form(&self, form: RowForm) -> ToneRow {
        match form {
            RowForm::Prime(n) => self.transpose(n),
            RowForm::Inversion(n) => self.invert().transpose(n),
            RowForm::Retrograde(n) => self.transpose(n).retrograde(),
            RowForm::RetrogradeInversion(n) => self.invert().transpose(n).retrograde(),
        }
    }

    /// All 48 forms (P/I/R/RI × 12 transpositions), each paired with its label.
    /// For symmetric rows some forms coincide as *rows* though their labels differ.
    pub fn all_forms(&self) -> Vec<(RowForm, ToneRow)> {
        let mut out = Vec::with_capacity(48);
        for n in 0..12u8 {
            for f in [
                RowForm::Prime(n),
                RowForm::Inversion(n),
                RowForm::Retrograde(n),
                RowForm::RetrogradeInversion(n),
            ] {
                out.push((f, self.form(f)));
            }
        }
        out
    }

    /// The classic 12×12 row matrix. `M[i][j] = (row[j] + row[0] − row[i]) mod 12`.
    ///
    /// Reading conventions: row `i` left→right is a Prime form, right→left a Retrograde;
    /// column `j` top→bottom is an Inversion, bottom→top a Retrograde-Inversion. The first
    /// row is the row itself, the first column is its inversion about the first pitch, and
    /// the main diagonal is constant (= `row[0]`).
    pub fn matrix(&self) -> [[u8; 12]; 12] {
        let r0 = self.0[0];
        let mut m = [[0u8; 12]; 12];
        for (i, mi) in m.iter_mut().enumerate() {
            for (j, cell) in mi.iter_mut().enumerate() {
                // + 24 keeps the intermediate non-negative before the mod.
                *cell = ((self.0[j] as u16 + r0 as u16 + 24 - self.0[i] as u16) % 12) as u8;
            }
        }
        m
    }

    /// Multiplicative transform `Mₘ(row)[k] = (m · row[k]) mod 12`. Only `m ∈ {1,5,7,11}`
    /// (coprime to 12) are bijections on `Z/12`; others are rejected. `M5` is the
    /// "circle-of-fourths" transform, `M7` the circle-of-fifths; `M11 = I₀`.
    pub fn multiply(&self, m: u8) -> Result<ToneRow, SerialError> {
        if !matches!(m % 12, 1 | 5 | 7 | 11) {
            return Err(SerialError::NotCoprime(m));
        }
        let mut out = [0u8; 12];
        for (o, &p) in out.iter_mut().zip(self.0.iter()) {
            *o = ((m as u16 * p as u16) % 12) as u8;
        }
        Ok(ToneRow(out))
    }

    /// The row's two hexachords as unordered sets: the first six and last six pitch classes.
    /// They always partition the aggregate (the row is a permutation), so the second is the
    /// [`PcSet::complement`] of the first.
    pub fn hexachords(&self) -> (PcSet, PcSet) {
        let h1 = PcSet::from_pcs(self.0[..6].iter().copied());
        let h2 = PcSet::from_pcs(self.0[6..].iter().copied());
        (h1, h2)
    }

    /// Transposition levels `n ∈ 1..12` at which the row is **prime-combinatorial**: the
    /// first hexachord `H₁` maps onto the second `H₂` under `Tₙ` (so `H₁ ∪ Tₙ(H₁)` completes
    /// the aggregate). `n = 6` is the classic semi-combinatorial case.
    pub fn combinatorial_prime_levels(&self) -> Vec<u8> {
        let (h1, h2) = self.hexachords();
        (1..12u8).filter(|&n| transpose_set(h1, n) == h2).collect()
    }

    /// Transposition levels `n ∈ 0..12` at which the row is **inversion-combinatorial**:
    /// the first hexachord `H₁` maps onto the second `H₂` under `Iₙ` (`x ↦ (n − x) mod 12`).
    pub fn combinatorial_inversion_levels(&self) -> Vec<u8> {
        let (h1, h2) = self.hexachords();
        (0..12u8).filter(|&n| invert_set(h1, n) == h2).collect()
    }
}

/// `Tₙ` on a pitch-class set: `{(x + n) mod 12}`.
fn transpose_set(s: PcSet, n: u8) -> PcSet {
    PcSet::from_pcs(s.iter_pcs().map(|x| (x + n) % 12))
}

/// `Iₙ` on a pitch-class set: `{(n − x) mod 12}`.
fn invert_set(s: PcSet, n: u8) -> PcSet {
    PcSet::from_pcs(s.iter_pcs().map(|x| (n + 12 - x) % 12))
}

impl fmt::Display for ToneRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, p) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{p}")?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chromatic row 0,1,…,11 — a clean reference whose transforms are exactly computable.
    fn chromatic() -> ToneRow {
        ToneRow::new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unwrap()
    }

    #[test]
    fn new_validates_permutation() {
        assert!(ToneRow::new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).is_ok());
        // Duplicate (two 0s, no 11).
        assert_eq!(
            ToneRow::new([0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]),
            Err(SerialError::NotARow)
        );
        // Out of range.
        assert_eq!(
            ToneRow::new([12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            Err(SerialError::NotARow)
        );
    }

    #[test]
    fn from_slice_checks_length() {
        assert_eq!(
            ToneRow::from_slice(&[0, 1, 2]),
            Err(SerialError::WrongLength(3))
        );
        assert!(ToneRow::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).is_ok());
    }

    #[test]
    fn transpose_wraps_mod_12() {
        let t = chromatic().transpose(3);
        assert_eq!(t.pcs(), &[3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2]);
    }

    #[test]
    fn invert_about_zero() {
        // I0 of 0,1,2,…,11 = 0,11,10,…,1.
        let i = chromatic().invert();
        assert_eq!(i.pcs(), &[0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn retrograde_reverses() {
        let r = chromatic().retrograde();
        assert_eq!(r.pcs(), &[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn ri_is_retrograde_of_inversion() {
        let row = ToneRow::new([0, 11, 7, 4, 2, 9, 3, 8, 10, 1, 5, 6]).unwrap();
        for n in 0..12u8 {
            let ri = row.form(RowForm::RetrogradeInversion(n));
            let expected = row.invert().transpose(n).retrograde();
            assert_eq!(ri, expected, "RI{n} must equal retrograde of I{n}");
        }
    }

    #[test]
    fn every_form_is_a_valid_row() {
        // All 48 transforms of a row are themselves rows (permutations).
        let row = ToneRow::new([0, 11, 7, 4, 2, 9, 3, 8, 10, 1, 5, 6]).unwrap();
        let forms = row.all_forms();
        assert_eq!(forms.len(), 48);
        for (label, r) in forms {
            assert!(
                ToneRow::new(*r.pcs()).is_ok(),
                "{label} is not a valid permutation: {r}"
            );
        }
    }

    #[test]
    fn matrix_structural_invariants() {
        let row = ToneRow::new([0, 11, 7, 4, 2, 9, 3, 8, 10, 1, 5, 6]).unwrap();
        let m = row.matrix();
        // First row is the row itself.
        assert_eq!(m[0], *row.pcs());
        // Main diagonal is constant = row[0].
        for (i, mi) in m.iter().enumerate() {
            assert_eq!(mi[i], row.pcs()[0], "diagonal cell ({i},{i})");
        }
        // First column read top→bottom is an inversion sharing the first pitch.
        let first_col: Vec<u8> = m.iter().map(|r| r[0]).collect();
        assert_eq!(first_col[0], row.pcs()[0]);
        // Every row and every column is a permutation of 0..12.
        for mi in &m {
            assert!(ToneRow::new(*mi).is_ok(), "matrix row not a permutation");
        }
        let cols: Vec<[u8; 12]> = (0..12).map(|j| std::array::from_fn(|i| m[i][j])).collect();
        for (j, col) in cols.iter().enumerate() {
            assert!(
                ToneRow::new(*col).is_ok(),
                "matrix column {j} not a permutation"
            );
        }
    }

    #[test]
    fn matrix_chromatic_is_difference_table() {
        // For the chromatic row (row[0]=0), M[i][j] = (j - i) mod 12.
        let m = chromatic().matrix();
        for (i, mi) in m.iter().enumerate() {
            for (j, &cell) in mi.iter().enumerate() {
                assert_eq!(cell, ((j as i32 - i as i32).rem_euclid(12)) as u8);
            }
        }
    }

    #[test]
    fn multiply_m7_is_bijection_m11_is_inversion() {
        // M7 of the chromatic row = 0,7,2,9,4,11,6,1,8,3,10,5 (×7 mod 12).
        let m7 = chromatic().multiply(7).unwrap();
        assert_eq!(m7.pcs(), &[0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]);
        // M11 == I0.
        assert_eq!(chromatic().multiply(11).unwrap(), chromatic().invert());
        // Non-coprime factor is rejected.
        assert_eq!(chromatic().multiply(2), Err(SerialError::NotCoprime(2)));
        assert_eq!(chromatic().multiply(6), Err(SerialError::NotCoprime(6)));
    }

    #[test]
    fn hexachords_partition_the_aggregate() {
        let row = ToneRow::new([0, 11, 7, 4, 2, 9, 3, 8, 10, 1, 5, 6]).unwrap();
        let (h1, h2) = row.hexachords();
        assert_eq!(h1.cardinality(), 6);
        assert_eq!(h2.cardinality(), 6);
        // The two hexachords are complements.
        assert_eq!(h2, h1.complement());
    }

    #[test]
    fn chromatic_row_is_prime_combinatorial_at_six() {
        // H1 = {0..5}, H2 = {6..11} = T6(H1).
        let levels = chromatic().combinatorial_prime_levels();
        assert!(
            levels.contains(&6),
            "chromatic row is T6-combinatorial, got {levels:?}"
        );
    }

    #[test]
    fn inversion_combinatoriality_detects_a_known_case() {
        // A row whose first hexachord {0,2,4,6,8,10} (whole-tone) maps to {1,3,5,7,9,11}
        // under I1: (1 - x) mod 12 sends 0→1,2→11,4→9,… = {1,11,9,7,5,3} = H2. So I1 holds.
        let row = ToneRow::new([0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]).unwrap();
        let levels = row.combinatorial_inversion_levels();
        assert!(
            levels.contains(&1),
            "expected I1-combinatorial, got {levels:?}"
        );
    }
}
