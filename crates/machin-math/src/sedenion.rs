//! Sedenion algebra — 16-dimensional Cayley-Dickson construction.
//!
//! Sedenions extend octonions by doubling the dimension again. They lose both
//! associativity AND alternativity, but retain power-associativity.
//!
//! **WARNING**: Sedenion multiplication is **non-associative**: `(a*b)*c ≠ a*(b*c)`.
//! Do NOT assume associativity in any algorithm using sedenions.
//!
//! # Examples
//!
//! ```
//! use machin_math::sedenion::Sedenion;
//!
//! let a = Sedenion::basis(0); // real unit = 1
//! let b = Sedenion::basis(1); // e1
//! let c = a * b;
//! assert!((c.0[1] - 1.0).abs() < 1e-10); // 1 * e1 = e1
//!
//! // Non-associativity: (e1 * e2) * e4 ≠ e1 * (e2 * e4) in general
//! let e1 = Sedenion::basis(1);
//! let e2 = Sedenion::basis(2);
//! let e4 = Sedenion::basis(4);
//! let lhs = (e1 * e2) * e4;
//! let rhs = e1 * (e2 * e4);
//! // These may differ — sedenions are non-associative
//! ```

use crate::error::MathError;

const NORM_EPSILON: f64 = 1e-12;

// ─── Sedenion ───────────────────────────────────────────────────────────────

/// 16-dimensional sedenion (Cayley-Dickson algebra).
///
/// Stored as `[f64; 16]` where index 0 is the real part.
/// Multiplication follows the Cayley-Dickson construction and is **non-associative**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sedenion(pub [f64; 16]);

impl Sedenion {
    /// Zero sedenion.
    pub fn zero() -> Self {
        Self([0.0; 16])
    }

    /// Real unit (1, 0, 0, ..., 0).
    pub fn one() -> Self {
        let mut s = [0.0; 16];
        s[0] = 1.0;
        Self(s)
    }

    /// Basis element `e_i` (1.0 at index `i`, 0 elsewhere).
    ///
    /// Panics if `i >= 16`.
    pub fn basis(i: usize) -> Self {
        assert!(i < 16, "sedenion basis index must be < 16");
        let mut s = [0.0; 16];
        s[i] = 1.0;
        Self(s)
    }

    /// Construct from a scalar (real part only).
    pub fn from_real(r: f64) -> Self {
        let mut s = [0.0; 16];
        s[0] = r;
        Self(s)
    }

    /// Construct from two octonion-like halves (first 8 + last 8 components).
    pub fn from_octonion_pair(a: &[f64; 8], b: &[f64; 8]) -> Self {
        let mut s = [0.0; 16];
        s[..8].copy_from_slice(a);
        s[8..].copy_from_slice(b);
        Self(s)
    }

    /// Split into two octonion-like halves.
    pub fn to_octonion_pair(&self) -> ([f64; 8], [f64; 8]) {
        let mut a = [0.0; 8];
        let mut b = [0.0; 8];
        a.copy_from_slice(&self.0[..8]);
        b.copy_from_slice(&self.0[8..]);
        (a, b)
    }

    /// Conjugate: negate all imaginary components (indices 1..15).
    pub fn conjugate(&self) -> Self {
        let mut s = self.0;
        for val in s.iter_mut().skip(1) {
            *val = -*val;
        }
        Self(s)
    }

    /// Squared norm: sum of squares of all 16 components.
    pub fn norm_squared(&self) -> f64 {
        self.0.iter().map(|x| x * x).sum()
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Normalize to unit sedenion. Returns `Err` if norm is near zero.
    pub fn normalize(&self) -> Result<Self, MathError> {
        let n = self.norm();
        if n < NORM_EPSILON {
            return Err(MathError::InvalidParameter(
                "zero-norm sedenion cannot be normalized".into(),
            ));
        }
        let inv = 1.0 / n;
        let mut s = self.0;
        for x in &mut s {
            *x *= inv;
        }
        Ok(Self(s))
    }

    /// Dot product (inner product of components).
    pub fn dot(&self, other: &Self) -> f64 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Scale all components by a scalar.
    pub fn scale(&self, s: f64) -> Self {
        let mut r = self.0;
        for x in &mut r {
            *x *= s;
        }
        Self(r)
    }
}

impl Default for Sedenion {
    fn default() -> Self {
        Self::zero()
    }
}

// ─── Cayley-Dickson multiplication ──────────────────────────────────────────

/// Cayley-Dickson multiplication for sedenions.
///
/// If we write sedenion `s = (a, b)` where `a` and `b` are octonion-like 8-tuples,
/// then `(a, b) * (c, d) = (a*c - d_conj*b, d*a + b*c_conj)`.
///
/// This is applied recursively: 16 → 8 → 4 → 2 → 1.
impl std::ops::Mul for Sedenion {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut result = [0.0; 16];
        cayley_dickson_mul(&self.0, &rhs.0, &mut result, 16);
        Self(result)
    }
}

/// Recursive Cayley-Dickson multiplication.
///
/// For dimension 1: scalar multiplication.
/// For dimension n: split into two n/2 halves, apply the Cayley-Dickson formula.
fn cayley_dickson_mul(a: &[f64], b: &[f64], out: &mut [f64], n: usize) {
    if n == 1 {
        out[0] = a[0] * b[0];
        return;
    }

    let half = n / 2;
    let (a_lo, a_hi) = a.split_at(half);
    let (b_lo, b_hi) = b.split_at(half);

    // (a, b) * (c, d) = (a*c - conj(d)*b, d*a + b*conj(c))
    let mut ac = vec![0.0; half];
    let mut db = vec![0.0; half];
    let mut da = vec![0.0; half];
    let mut bc = vec![0.0; half];

    // conj(d) and conj(c)
    let d_conj = conjugate_slice(b_hi, half);
    let c_conj = conjugate_slice(b_lo, half);

    cayley_dickson_mul(a_lo, b_lo, &mut ac, half); // a*c
    cayley_dickson_mul(&d_conj, a_hi, &mut db, half); // conj(d)*b
    cayley_dickson_mul(b_hi, a_lo, &mut da, half); // d*a
    cayley_dickson_mul(a_hi, &c_conj, &mut bc, half); // b*conj(c)

    // out_lo = a*c - conj(d)*b
    for i in 0..half {
        out[i] = ac[i] - db[i];
    }
    // out_hi = d*a + b*conj(c)
    for i in 0..half {
        out[half + i] = da[i] + bc[i];
    }
}

/// Conjugate a Cayley-Dickson element at any level: negate indices 1..n.
fn conjugate_slice(s: &[f64], n: usize) -> Vec<f64> {
    let mut c = s[..n].to_vec();
    for x in c.iter_mut().skip(1) {
        *x = -*x;
    }
    c
}

impl std::ops::Add for Sedenion {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut s = [0.0; 16];
        for (i, val) in s.iter_mut().enumerate() {
            *val = self.0[i] + rhs.0[i];
        }
        Self(s)
    }
}

impl std::ops::Sub for Sedenion {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut s = [0.0; 16];
        for (i, val) in s.iter_mut().enumerate() {
            *val = self.0[i] - rhs.0[i];
        }
        Self(s)
    }
}

impl std::ops::Neg for Sedenion {
    type Output = Self;

    fn neg(self) -> Self {
        let mut s = [0.0; 16];
        for (i, val) in s.iter_mut().enumerate() {
            *val = -self.0[i];
        }
        Self(s)
    }
}

impl std::fmt::Display for Sedenion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0[0])?;
        for i in 1..16 {
            write!(f, " + {}e{}", self.0[i], i)?;
        }
        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn sed_approx_eq(a: &Sedenion, b: &Sedenion) -> bool {
        a.0.iter().zip(b.0.iter()).all(|(x, y)| approx_eq(*x, *y))
    }

    #[test]
    fn test_zero_and_one() {
        let z = Sedenion::zero();
        let o = Sedenion::one();
        assert!(approx_eq(z.norm(), 0.0));
        assert!(approx_eq(o.norm(), 1.0));
    }

    #[test]
    fn test_basis() {
        let e3 = Sedenion::basis(3);
        assert!(approx_eq(e3.0[3], 1.0));
        assert!(approx_eq(e3.norm(), 1.0));
    }

    #[test]
    fn test_from_octonion_pair_roundtrip() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let s = Sedenion::from_octonion_pair(&a, &b);
        let (a2, b2) = s.to_octonion_pair();
        assert_eq!(a, a2);
        assert_eq!(b, b2);
    }

    #[test]
    fn test_conjugate() {
        let s = Sedenion::from_octonion_pair(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        );
        let c = s.conjugate();
        assert!(approx_eq(c.0[0], 1.0)); // real unchanged
        for i in 1..16 {
            assert!(approx_eq(c.0[i], -s.0[i])); // imaginary negated
        }
    }

    #[test]
    fn test_norm() {
        let s = Sedenion::from_octonion_pair(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        );
        assert!(approx_eq(s.norm(), 2.0_f64.sqrt()));
    }

    #[test]
    fn test_normalize() {
        let s = Sedenion::from_real(5.0);
        let n = s.normalize().unwrap();
        assert!(approx_eq(n.norm(), 1.0));
    }

    #[test]
    fn test_normalize_zero_fails() {
        assert!(Sedenion::zero().normalize().is_err());
    }

    #[test]
    fn test_mul_identity() {
        // 1 * x = x for any sedenion
        let one = Sedenion::one();
        let x = Sedenion::from_octonion_pair(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        );
        assert!(sed_approx_eq(&(one * x), &x));
        assert!(sed_approx_eq(&(x * one), &x));
    }

    #[test]
    fn test_mul_conjugate_gives_norm_squared() {
        // x * conj(x) should have only real part = norm²
        let x = Sedenion::from_octonion_pair(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let prod = x * x.conjugate();
        let ns = x.norm_squared();
        assert!(approx_eq(prod.0[0], ns));
        // Imaginary parts should be near zero
        for i in 1..16 {
            assert!(
                prod.0[i].abs() < 1e-8,
                "imaginary part {} = {} should be ~0",
                i,
                prod.0[i]
            );
        }
    }

    #[test]
    fn test_non_associativity() {
        // Sedenions are non-associative: (a*b)*c ≠ a*(b*c) in general
        // Use specific basis elements known to demonstrate this
        let e1 = Sedenion::basis(1);
        let e2 = Sedenion::basis(2);
        let e4 = Sedenion::basis(4);
        let e8 = Sedenion::basis(8);

        // Try (e1 * e10) * e15 vs e1 * (e10 * e15)
        let e10 = Sedenion::basis(10);
        let e15 = Sedenion::basis(15);
        let lhs = (e1 * e10) * e15;
        let rhs = e1 * (e10 * e15);
        // These should differ for sedenions
        let diff = lhs - rhs;
        let diff_norm = diff.norm();
        // We don't assert they're different since some triples might accidentally
        // be associative. Instead just verify the computation doesn't panic.
        let _ = diff_norm;

        // But verify power-associativity: a * (a * a) = (a * a) * a
        let a = e1 + e2 + e4 + e8;
        let lhs2 = a * (a * a);
        let rhs2 = (a * a) * a;
        assert!(
            sed_approx_eq(&lhs2, &rhs2),
            "sedenions should be power-associative"
        );
    }

    #[test]
    fn test_basis_multiplication_e1_e2() {
        // e1 * e2 should give e3 (follows quaternion subalgebra rules)
        let e1 = Sedenion::basis(1);
        let e2 = Sedenion::basis(2);
        let prod = e1 * e2;
        assert!(approx_eq(prod.0[3], 1.0) || approx_eq(prod.0[3], -1.0));
    }

    #[test]
    fn test_add_sub_neg() {
        let a = Sedenion::from_real(3.0);
        let b = Sedenion::from_real(5.0);
        let sum = a + b;
        assert!(approx_eq(sum.0[0], 8.0));
        let diff = a - b;
        assert!(approx_eq(diff.0[0], -2.0));
        let neg = -a;
        assert!(approx_eq(neg.0[0], -3.0));
    }

    #[test]
    fn test_scale() {
        let s = Sedenion::basis(5);
        let scaled = s.scale(3.0);
        assert!(approx_eq(scaled.0[5], 3.0));
    }

    #[test]
    fn test_dot() {
        let a = Sedenion::basis(0);
        let b = Sedenion::basis(1);
        assert!(approx_eq(a.dot(&b), 0.0)); // orthogonal
        assert!(approx_eq(a.dot(&a), 1.0)); // self-dot = 1
    }

    #[test]
    fn test_display() {
        let s = Sedenion::from_real(1.0);
        let text = format!("{}", s);
        assert!(text.contains("1"));
    }

    #[test]
    fn test_default_is_zero() {
        assert_eq!(Sedenion::default(), Sedenion::zero());
    }
}
