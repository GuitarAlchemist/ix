use std::ops::{Add, Mul, Neg, Sub};

/// An octonion: 8-dimensional hypercomplex number built via Cayley-Dickson
/// construction from quaternion pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Octonion {
    pub components: [f64; 8],
}

/// Quaternion multiplication helper: (a0 + a1*i + a2*j + a3*k) * (b0 + b1*i + b2*j + b3*k)
fn quat_mul(a: &[f64], b: &[f64]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Quaternion conjugate: negate imaginary parts
fn quat_conj(a: &[f64]) -> [f64; 4] {
    [a[0], -a[1], -a[2], -a[3]]
}

/// Quaternion addition
fn quat_add(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Quaternion subtraction
fn quat_sub(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

impl Octonion {
    /// Create an octonion from 8 components.
    pub fn new(components: [f64; 8]) -> Self {
        Self { components }
    }

    /// The zero octonion.
    pub fn zero() -> Self {
        Self {
            components: [0.0; 8],
        }
    }

    /// The multiplicative identity (1, 0, 0, ..., 0).
    pub fn one() -> Self {
        let mut c = [0.0; 8];
        c[0] = 1.0;
        Self { components: c }
    }

    /// The i-th basis element e_i (0-indexed). e_0 = 1.
    pub fn basis(i: usize) -> Self {
        assert!(i < 8, "Octonion basis index must be 0..7");
        let mut c = [0.0; 8];
        c[i] = 1.0;
        Self { components: c }
    }

    /// Component-wise addition.
    pub fn add(&self, other: &Octonion) -> Octonion {
        let mut c = [0.0; 8];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.components[i] + other.components[i];
        }
        Octonion { components: c }
    }

    /// Component-wise subtraction.
    pub fn sub(&self, other: &Octonion) -> Octonion {
        let mut c = [0.0; 8];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.components[i] - other.components[i];
        }
        Octonion { components: c }
    }

    /// Cayley-Dickson multiplication using quaternion pairs.
    /// (a, b) * (c, d) = (a*c - conj(d)*b, d*a + b*conj(c))
    /// where a, b, c, d are quaternions.
    pub fn mul(&self, other: &Octonion) -> Octonion {
        let a = &self.components[0..4];
        let b = &self.components[4..8];
        let c = &other.components[0..4];
        let d = &other.components[4..8];

        let conj_d = quat_conj(d);
        let conj_c = quat_conj(c);

        // first half: a*c - conj(d)*b
        let ac = quat_mul(a, c);
        let conj_d_b = quat_mul(&conj_d, b);
        let first = quat_sub(&ac, &conj_d_b);

        // second half: d*a + b*conj(c)
        let da = quat_mul(d, a);
        let b_conj_c = quat_mul(b, &conj_c);
        let second = quat_add(&da, &b_conj_c);

        let mut result = [0.0; 8];
        result[0..4].copy_from_slice(&first);
        result[4..8].copy_from_slice(&second);
        Octonion { components: result }
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Octonion {
        let mut c = [0.0; 8];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.components[i] * s;
        }
        Octonion { components: c }
    }

    /// Octonion conjugate: negate all imaginary parts (indices 1..7).
    pub fn conjugate(&self) -> Octonion {
        let mut c = self.components;
        for ci in c.iter_mut().skip(1) {
            *ci = -*ci;
        }
        Octonion { components: c }
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
    pub fn inverse(&self) -> Octonion {
        let ns = self.norm_squared();
        assert!(ns > 1e-15, "Cannot invert zero octonion");
        self.conjugate().scale(1.0 / ns)
    }
}

impl Add for Octonion {
    type Output = Octonion;
    fn add(self, rhs: Octonion) -> Octonion {
        Octonion::add(&self, &rhs)
    }
}

impl Sub for Octonion {
    type Output = Octonion;
    fn sub(self, rhs: Octonion) -> Octonion {
        Octonion::sub(&self, &rhs)
    }
}

impl Mul for Octonion {
    type Output = Octonion;
    fn mul(self, rhs: Octonion) -> Octonion {
        Octonion::mul(&self, &rhs)
    }
}

impl Neg for Octonion {
    type Output = Octonion;
    fn neg(self) -> Octonion {
        self.scale(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: &Octonion, b: &Octonion) -> bool {
        a.components
            .iter()
            .zip(b.components.iter())
            .all(|(x, y)| (x - y).abs() < EPS)
    }

    #[test]
    fn test_unit_multiply() {
        let one = Octonion::one();
        let o = Octonion::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(approx_eq(&Octonion::mul(&one, &o), &o));
        assert!(approx_eq(&Octonion::mul(&o, &one), &o));
    }

    #[test]
    fn test_basis_squared_is_minus_one() {
        let one = Octonion::one();
        let neg_one = -one;
        for i in 1..8 {
            let ei = Octonion::basis(i);
            let sq = Octonion::mul(&ei, &ei);
            assert!(
                approx_eq(&sq, &neg_one),
                "e_{i}^2 should be -1, got {:?}",
                sq.components
            );
        }
    }

    #[test]
    fn test_conjugate_times_self_is_norm_squared() {
        let o = Octonion::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let conj = o.conjugate();
        let product = Octonion::mul(&conj, &o);
        let ns = o.norm_squared();
        assert!((product.components[0] - ns).abs() < EPS);
        for i in 1..8 {
            assert!(
                product.components[i].abs() < EPS,
                "imaginary part {} should be 0, got {}",
                i,
                product.components[i]
            );
        }
    }

    #[test]
    fn test_non_associativity() {
        let e1 = Octonion::basis(1);
        let e2 = Octonion::basis(2);
        let e4 = Octonion::basis(4);

        let e1e2 = Octonion::mul(&e1, &e2);
        let lhs = Octonion::mul(&e1e2, &e4);
        let e2e4 = Octonion::mul(&e2, &e4);
        let rhs = Octonion::mul(&e1, &e2e4);
        assert!(
            !approx_eq(&lhs, &rhs),
            "Octonions should NOT be associative!\nlhs={:?}\nrhs={:?}",
            lhs.components,
            rhs.components
        );
    }

    #[test]
    fn test_alternative_law() {
        let x = Octonion::new([1.0, 2.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0]);
        let y = Octonion::new([0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0]);

        let xy = Octonion::mul(&x, &y);
        let lhs = Octonion::mul(&x, &xy);
        let xx = Octonion::mul(&x, &x);
        let rhs = Octonion::mul(&xx, &y);
        assert!(
            approx_eq(&lhs, &rhs),
            "Left alternative law failed:\nlhs={:?}\nrhs={:?}",
            lhs.components,
            rhs.components
        );

        let yx = Octonion::mul(&y, &x);
        let lhs2 = Octonion::mul(&yx, &x);
        let rhs2 = Octonion::mul(&y, &xx);
        assert!(
            approx_eq(&lhs2, &rhs2),
            "Right alternative law failed:\nlhs={:?}\nrhs={:?}",
            lhs2.components,
            rhs2.components
        );
    }

    #[test]
    fn test_inverse() {
        let o = Octonion::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let inv = o.inverse();
        let product = Octonion::mul(&o, &inv);
        assert!(
            approx_eq(&product, &Octonion::one()),
            "o * o^-1 should be 1, got {:?}",
            product.components
        );
    }

    #[test]
    fn test_norm() {
        let o = Octonion::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((o.norm() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_zero() {
        let z = Octonion::zero();
        assert!((z.norm() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_add_sub() {
        let a = Octonion::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Octonion::new([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let sum = a + b;
        for i in 0..8 {
            assert!((sum.components[i] - 9.0).abs() < EPS);
        }
        let diff = a - b;
        assert!((diff.components[0] - (-7.0)).abs() < EPS);
    }

    #[test]
    fn test_neg() {
        let a = Octonion::new([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
        let neg_a = -a;
        for i in 0..8 {
            assert!((neg_a.components[i] + a.components[i]).abs() < EPS);
        }
    }
}
