use crate::octonion::Octonion;
use std::ops::{Add, Mul, Neg, Sub};

/// A sedenion: 16-dimensional hypercomplex number built via Cayley-Dickson
/// construction from octonion pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sedenion {
    pub components: [f64; 16],
}

impl Sedenion {
    /// Create a sedenion from 16 components.
    pub fn new(components: [f64; 16]) -> Self {
        Self { components }
    }

    /// The zero sedenion.
    pub fn zero() -> Self {
        Self {
            components: [0.0; 16],
        }
    }

    /// The multiplicative identity (1, 0, 0, ..., 0).
    pub fn one() -> Self {
        let mut c = [0.0; 16];
        c[0] = 1.0;
        Self { components: c }
    }

    /// The i-th basis element e_i (0-indexed). e_0 = 1.
    pub fn basis(i: usize) -> Self {
        assert!(i < 16, "Sedenion basis index must be 0..15");
        let mut c = [0.0; 16];
        c[i] = 1.0;
        Self { components: c }
    }

    /// Extract the first octonion half.
    fn first_oct(&self) -> Octonion {
        let mut c = [0.0; 8];
        c.copy_from_slice(&self.components[0..8]);
        Octonion::new(c)
    }

    /// Extract the second octonion half.
    fn second_oct(&self) -> Octonion {
        let mut c = [0.0; 8];
        c.copy_from_slice(&self.components[8..16]);
        Octonion::new(c)
    }

    /// Build a sedenion from two octonion halves.
    fn from_oct_pair(first: &Octonion, second: &Octonion) -> Sedenion {
        let mut c = [0.0; 16];
        c[0..8].copy_from_slice(&first.components);
        c[8..16].copy_from_slice(&second.components);
        Sedenion { components: c }
    }

    /// Component-wise addition.
    pub fn add(&self, other: &Sedenion) -> Sedenion {
        let mut c = [0.0; 16];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.components[i] + other.components[i];
        }
        Sedenion { components: c }
    }

    /// Component-wise subtraction.
    pub fn sub(&self, other: &Sedenion) -> Sedenion {
        let mut c = [0.0; 16];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.components[i] - other.components[i];
        }
        Sedenion { components: c }
    }

    /// Cayley-Dickson multiplication using octonion pairs.
    /// (a, b) * (c, d) = (a*c - conj(d)*b, d*a + b*conj(c))
    /// where a, b, c, d are octonions.
    pub fn mul(&self, other: &Sedenion) -> Sedenion {
        let a = self.first_oct();
        let b = self.second_oct();
        let c = other.first_oct();
        let d = other.second_oct();

        let conj_d = d.conjugate();
        let conj_c = c.conjugate();

        // first half: a*c - conj(d)*b
        let ac = Octonion::mul(&a, &c);
        let conj_d_b = Octonion::mul(&conj_d, &b);
        let first = Octonion::sub(&ac, &conj_d_b);

        // second half: d*a + b*conj(c)
        let da = Octonion::mul(&d, &a);
        let b_conj_c = Octonion::mul(&b, &conj_c);
        let second = Octonion::add(&da, &b_conj_c);

        Self::from_oct_pair(&first, &second)
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Sedenion {
        let mut c = [0.0; 16];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.components[i] * s;
        }
        Sedenion { components: c }
    }

    /// Sedenion conjugate: negate all imaginary parts (indices 1..15).
    pub fn conjugate(&self) -> Sedenion {
        let mut c = self.components;
        for ci in c.iter_mut().skip(1) {
            *ci = -*ci;
        }
        Sedenion { components: c }
    }

    /// Squared norm: sum of squares of all components.
    pub fn norm_squared(&self) -> f64 {
        self.components.iter().map(|x| x * x).sum()
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Multiplicative inverse: conj(x) / norm²(x).
    pub fn inverse(&self) -> Sedenion {
        let ns = self.norm_squared();
        assert!(ns > 1e-15, "Cannot invert zero sedenion");
        self.conjugate().scale(1.0 / ns)
    }

    /// Sedenion exponential via scalar+vector decomposition.
    ///
    /// For `s = a + v` where `a` is the scalar part (`e_0` component) and
    /// `v` is the 15-component "vector" part, `exp(s) = exp(a) * (cos|v| +
    /// (v/|v|) * sin|v|)`. This is the direct generalization of the
    /// quaternion exponential, extended to 16 dimensions.
    ///
    /// Ported from TARS v1's `HyperComplexGeometricDSL::SedenionOps::exp`.
    pub fn exp(&self) -> Sedenion {
        let scalar = self.components[0];
        let vec_norm_sq: f64 = self.components[1..].iter().map(|x| x * x).sum();
        let vec_norm = vec_norm_sq.sqrt();
        let exp_scalar = scalar.exp();

        let mut out = [0.0_f64; 16];
        if vec_norm < 1e-12 {
            // Pure-scalar sedenion: exp reduces to real exponential.
            out[0] = exp_scalar;
        } else {
            let cos_vn = vec_norm.cos();
            let sin_vn = vec_norm.sin();
            let factor = exp_scalar * sin_vn / vec_norm;
            out[0] = exp_scalar * cos_vn;
            for (i, slot) in out.iter_mut().enumerate().skip(1) {
                *slot = factor * self.components[i];
            }
        }
        Sedenion::new(out)
    }

    /// Sedenion logarithm via scalar+vector decomposition.
    ///
    /// For `s = a + v` with norm `r = |s|` and vector-part norm `|v|`:
    ///
    /// ```text
    ///   log(s) = log(r) + (v / |v|) * atan2(|v|, a)
    /// ```
    ///
    /// This is the inverse of `exp` for sedenions with small vector-part
    /// norm. For zero-vector sedenions with a positive real part
    /// (`s = a, a > 0`) the result reduces to the real log. For zero-vector
    /// sedenions with a *negative* real part (`s = -|a|`), the principal
    /// log needs an angle of `pi` along an imaginary direction — we use
    /// `e_1` (the first imaginary basis element) by convention, matching
    /// the quaternion/complex-number definition. Returns NaN/Inf for
    /// `s = 0` since `log(0)` is undefined.
    ///
    /// Ported from TARS v1's `HyperComplexGeometricDSL::SedenionOps::log`,
    /// with the negative-real-axis case fixed after the 2026-04-09 review.
    pub fn log(&self) -> Sedenion {
        let scalar = self.components[0];
        let vec_norm_sq: f64 = self.components[1..].iter().map(|x| x * x).sum();
        let vec_norm = vec_norm_sq.sqrt();
        let norm = self.norm();

        let mut out = [0.0_f64; 16];
        out[0] = norm.ln();

        if vec_norm >= 1e-12 {
            // General case: log has a well-defined vector direction.
            let angle = vec_norm.atan2(scalar);
            let factor = angle / vec_norm;
            for (i, slot) in out.iter_mut().enumerate().skip(1) {
                *slot = factor * self.components[i];
            }
        } else if scalar < 0.0 {
            // Negative-real-axis case: principal log needs angle pi along
            // some unit imaginary direction. Choose e_1 by convention so
            // that `exp(log(-1)) == -1` as expected, matching the complex
            // principal value Log(-1) = i*pi.
            out[1] = std::f64::consts::PI;
        }
        // Positive-real case with vec_norm ≈ 0: out already has log(|s|)
        // in the scalar slot and zeros elsewhere, which is correct.

        Sedenion::new(out)
    }
}

impl Add for Sedenion {
    type Output = Sedenion;
    fn add(self, rhs: Sedenion) -> Sedenion {
        Sedenion::add(&self, &rhs)
    }
}

impl Sub for Sedenion {
    type Output = Sedenion;
    fn sub(self, rhs: Sedenion) -> Sedenion {
        Sedenion::sub(&self, &rhs)
    }
}

impl Mul for Sedenion {
    type Output = Sedenion;
    fn mul(self, rhs: Sedenion) -> Sedenion {
        Sedenion::mul(&self, &rhs)
    }
}

impl Neg for Sedenion {
    type Output = Sedenion;
    fn neg(self) -> Sedenion {
        self.scale(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: &Sedenion, b: &Sedenion) -> bool {
        a.components
            .iter()
            .zip(b.components.iter())
            .all(|(x, y)| (x - y).abs() < EPS)
    }

    #[test]
    fn test_unit_times_unit_is_unit() {
        let one = Sedenion::one();
        assert!(approx_eq(&Sedenion::mul(&one, &one), &one));
    }

    #[test]
    fn test_unit_multiply() {
        let one = Sedenion::one();
        let s = Sedenion::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        assert!(approx_eq(&Sedenion::mul(&one, &s), &s));
        assert!(approx_eq(&Sedenion::mul(&s, &one), &s));
    }

    #[test]
    fn test_basis_squared_is_minus_one() {
        let one = Sedenion::one();
        let neg_one = -one;
        for i in 1..16 {
            let ei = Sedenion::basis(i);
            let sq = Sedenion::mul(&ei, &ei);
            assert!(
                approx_eq(&sq, &neg_one),
                "e_{}^2 should be -1, got {:?}",
                i,
                sq.components
            );
        }
    }

    #[test]
    fn test_conjugate_times_self_is_norm_squared() {
        let s = Sedenion::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let conj = s.conjugate();
        let product = Sedenion::mul(&conj, &s);
        let ns = s.norm_squared();
        assert!(
            (product.components[0] - ns).abs() < EPS,
            "real part should be norm², got {} vs {}",
            product.components[0],
            ns
        );
        for i in 1..16 {
            assert!(
                product.components[i].abs() < EPS,
                "imaginary part {} should be 0, got {}",
                i,
                product.components[i]
            );
        }
    }

    #[test]
    fn test_power_associativity() {
        // Sedenions are power-associative: x*(x*x) = (x*x)*x
        let x = Sedenion::new([
            1.0, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        ]);
        let xx = Sedenion::mul(&x, &x);
        let lhs = Sedenion::mul(&x, &xx);
        let rhs = Sedenion::mul(&xx, &x);
        assert!(
            approx_eq(&lhs, &rhs),
            "Power associativity failed:\nlhs={:?}\nrhs={:?}",
            lhs.components,
            rhs.components
        );
    }

    #[test]
    fn test_scalar_multiplication() {
        let s = Sedenion::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let scaled = s.scale(2.0);
        for i in 0..16 {
            assert!((scaled.components[i] - 2.0 * s.components[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_zero() {
        let z = Sedenion::zero();
        assert!((z.norm() - 0.0).abs() < EPS);
        let one = Sedenion::one();
        assert!(approx_eq(&(one + z), &one));
    }

    #[test]
    fn test_inverse() {
        let s = Sedenion::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        ]);
        let inv = s.inverse();
        let product = Sedenion::mul(&s, &inv);
        assert!(
            approx_eq(&product, &Sedenion::one()),
            "s * s^-1 should be 1, got {:?}",
            product.components
        );
    }

    #[test]
    fn test_add_sub() {
        let a = Sedenion::new([1.0; 16]);
        let b = Sedenion::new([2.0; 16]);
        let sum = a + b;
        for i in 0..16 {
            assert!((sum.components[i] - 3.0).abs() < EPS);
        }
        let diff = b - a;
        for i in 0..16 {
            assert!((diff.components[i] - 1.0).abs() < EPS);
        }
    }

    #[test]
    fn test_neg() {
        let a = Sedenion::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let neg_a = -a;
        for i in 0..16 {
            assert!((neg_a.components[i] + a.components[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_norm() {
        let one = Sedenion::one();
        assert!((one.norm() - 1.0).abs() < EPS);

        let e5 = Sedenion::basis(5);
        assert!((e5.norm() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_exp_of_zero_is_one() {
        let z = Sedenion::zero();
        let e = z.exp();
        assert!((e.components[0] - 1.0).abs() < EPS);
        for i in 1..16 {
            assert!(e.components[i].abs() < EPS);
        }
    }

    #[test]
    fn test_exp_pure_scalar() {
        let mut comps = [0.0; 16];
        comps[0] = 2.0;
        let s = Sedenion::new(comps);
        let e = s.exp();
        assert!((e.components[0] - 2.0_f64.exp()).abs() < EPS);
        for i in 1..16 {
            assert!(e.components[i].abs() < EPS);
        }
    }

    #[test]
    fn test_exp_pure_vector_norm_is_one() {
        // Pure imaginary unit (e_1): exp(e_1) should have norm 1
        let e1 = Sedenion::basis(1);
        let e = e1.exp();
        assert!((e.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_of_one_is_zero() {
        let one = Sedenion::one();
        let l = one.log();
        for i in 0..16 {
            assert!(l.components[i].abs() < EPS);
        }
    }

    #[test]
    fn test_log_of_negative_one_has_pi() {
        // exp(log(-1)) should equal -1, matching the complex principal
        // value Log(-1) = i*pi. The fix puts the pi angle on e_1.
        let mut comps = [0.0; 16];
        comps[0] = -1.0;
        let neg_one = Sedenion::new(comps);
        let l = neg_one.log();
        // Real part of log: ln|-1| = 0
        assert!(l.components[0].abs() < EPS);
        // Imaginary part: pi on e_1
        assert!((l.components[1] - std::f64::consts::PI).abs() < 1e-10);
        for i in 2..16 {
            assert!(l.components[i].abs() < EPS);
        }
        // Round trip: exp(log(-1)) == -1
        let back = l.exp();
        assert!(
            (back.components[0] - (-1.0)).abs() < 1e-10,
            "exp(log(-1)) scalar = {}, expected -1",
            back.components[0]
        );
        for i in 1..16 {
            assert!(back.components[i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_log_exp_roundtrip_small_vector() {
        // For small vector parts, log(exp(s)) = s
        let mut comps = [0.0; 16];
        comps[0] = 0.3;
        comps[1] = 0.1;
        comps[2] = 0.05;
        let s = Sedenion::new(comps);
        let roundtrip = s.exp().log();
        for i in 0..16 {
            assert!(
                (roundtrip.components[i] - s.components[i]).abs() < 1e-9,
                "mismatch at component {}: got {}, want {}",
                i,
                roundtrip.components[i],
                s.components[i]
            );
        }
    }
}
