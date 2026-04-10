/// Generic Cayley-Dickson doubling construction.
///
/// Provides multiplication, conjugation, and norm for any 2^n-dimensional
/// hypercomplex algebra: complex (2) -> quaternion (4) -> octonion (8) ->
/// sedenion (16) -> 32-ion, etc.
///
/// Cayley-Dickson multiplication for two slices of length 2^n.
///
/// Recursion:
/// - dim=1: scalar multiplication
/// - dim=2^n: (a,b)*(c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
///   where a,b,c,d are half-size elements.
pub fn double_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    assert_eq!(n, b.len(), "Cayley-Dickson multiply: mismatched dimensions");
    assert!(n.is_power_of_two(), "Cayley-Dickson: dimension must be a power of 2");

    if n == 1 {
        return vec![a[0] * b[0]];
    }

    let half = n / 2;
    let (a1, a2) = a.split_at(half);
    let (b1, b2) = b.split_at(half);

    let conj_b2 = double_conjugate(b2);
    let conj_b1 = double_conjugate(b1);

    // first half: a1*b1 - conj(b2)*a2
    let a1_b1 = double_multiply(a1, b1);
    let conj_b2_a2 = double_multiply(&conj_b2, a2);
    let first: Vec<f64> = a1_b1.iter().zip(conj_b2_a2.iter())
        .map(|(x, y)| x - y).collect();

    // second half: b2*a1 + a2*conj(b1)
    let b2_a1 = double_multiply(b2, a1);
    let a2_conj_b1 = double_multiply(a2, &conj_b1);
    let second: Vec<f64> = b2_a1.iter().zip(a2_conj_b1.iter())
        .map(|(x, y)| x + y).collect();

    let mut result = first;
    result.extend(second);
    result
}

/// Cayley-Dickson conjugate: negate all imaginary parts.
///
/// For dim=1, returns the value unchanged (real scalars).
/// For dim>1, conjugate = (conj(a), -b) where (a, b) are the halves.
pub fn double_conjugate(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    if n == 1 {
        return vec![a[0]];
    }
    let mut result = vec![a[0]];
    for item in a.iter().take(n).skip(1) {
        result.push(-item);
    }
    result
}

/// Cayley-Dickson norm: sqrt(sum of squares).
pub fn double_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cayley-Dickson algebra element with compile-time dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct CayleyDickson<const N: usize> {
    pub components: [f64; N],
}

impl<const N: usize> CayleyDickson<N> {
    /// Create a new element.
    pub fn new(components: [f64; N]) -> Self {
        Self { components }
    }

    /// The zero element.
    pub fn zero() -> Self {
        Self { components: [0.0; N] }
    }

    /// The multiplicative identity.
    pub fn one() -> Self {
        let mut c = [0.0; N];
        c[0] = 1.0;
        Self { components: c }
    }

    /// Multiply two elements using the generic Cayley-Dickson formula.
    pub fn mul(&self, other: &Self) -> Self {
        let result = double_multiply(&self.components, &other.components);
        let mut c = [0.0; N];
        c.copy_from_slice(&result);
        Self { components: c }
    }

    /// Conjugate.
    pub fn conjugate(&self) -> Self {
        let result = double_conjugate(&self.components);
        let mut c = [0.0; N];
        c.copy_from_slice(&result);
        Self { components: c }
    }

    /// Norm.
    pub fn norm(&self) -> f64 {
        double_norm(&self.components)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_complex_multiplication() {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let a = [3.0, 4.0]; // 3 + 4i
        let b = [1.0, 2.0]; // 1 + 2i
        let result = double_multiply(&a, &b);
        // (3+4i)(1+2i) = 3+6i+4i+8i² = 3+10i-8 = -5+10i
        assert!((result[0] - (-5.0)).abs() < EPS, "real part: {}", result[0]);
        assert!((result[1] - 10.0).abs() < EPS, "imag part: {}", result[1]);
    }

    #[test]
    fn test_quaternion_multiplication() {
        // Test i*j = k for quaternions [w, i, j, k]
        let i = [0.0, 1.0, 0.0, 0.0];
        let j = [0.0, 0.0, 1.0, 0.0];
        let k = [0.0, 0.0, 0.0, 1.0];
        let result = double_multiply(&i, &j);
        assert!((result[0] - k[0]).abs() < EPS);
        assert!((result[1] - k[1]).abs() < EPS);
        assert!((result[2] - k[2]).abs() < EPS);
        assert!((result[3] - k[3]).abs() < EPS, "i*j should be k, got {:?}", result);

        // j*i = -k
        let result2 = double_multiply(&j, &i);
        assert!((result2[3] - (-1.0)).abs() < EPS, "j*i should be -k, got {:?}", result2);
    }

    #[test]
    fn test_quaternion_basis_squared() {
        // i² = j² = k² = -1 for quaternions
        let neg1 = [-1.0, 0.0, 0.0, 0.0];
        for idx in 1..4 {
            let mut e = [0.0; 4];
            e[idx] = 1.0;
            let sq = double_multiply(&e, &e);
            for c in 0..4 {
                assert!((sq[c] - neg1[c]).abs() < EPS,
                    "e_{}^2 component {}: got {}, expected {}", idx, c, sq[c], neg1[c]);
            }
        }
    }

    #[test]
    fn test_sedenion_matches_struct() {
        // Verify that double_multiply for 16D matches Sedenion::mul
        use crate::sedenion::Sedenion;

        let a_arr = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                     0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8];
        let b_arr = [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,
                     8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0];

        let generic = double_multiply(&a_arr, &b_arr);
        let sa = Sedenion::new(a_arr);
        let sb = Sedenion::new(b_arr);
        let specific = Sedenion::mul(&sa, &sb);

        for (i, &g) in generic.iter().enumerate().take(16) {
            assert!((g - specific.components[i]).abs() < EPS,
                "mismatch at {}: generic={}, specific={}", i, g, specific.components[i]);
        }
    }

    #[test]
    fn test_conjugate() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let conj = double_conjugate(&a);
        assert!((conj[0] - 1.0).abs() < EPS);
        assert!((conj[1] - (-2.0)).abs() < EPS);
        assert!((conj[2] - (-3.0)).abs() < EPS);
        assert!((conj[3] - (-4.0)).abs() < EPS);
    }

    #[test]
    fn test_norm() {
        let a = [3.0, 4.0];
        assert!((double_norm(&a) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_cayley_dickson_struct() {
        let a = CayleyDickson::<4>::new([1.0, 0.0, 0.0, 0.0]);
        let b = CayleyDickson::<4>::new([0.0, 1.0, 0.0, 0.0]);
        let product = a.mul(&b);
        // 1 * i = i
        assert!((product.components[0] - 0.0).abs() < EPS);
        assert!((product.components[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_32ion_basis_squared() {
        // Even 32-ions should have e_i^2 = -1
        for idx in 1..32 {
            let mut e = [0.0f64; 32];
            e[idx] = 1.0;
            let sq = double_multiply(&e, &e);
            assert!((sq[0] - (-1.0)).abs() < EPS,
                "32-ion e_{}^2 real part: {}", idx, sq[0]);
            for (c, &v) in sq.iter().enumerate().take(32).skip(1) {
                assert!(v.abs() < EPS,
                    "32-ion e_{}^2 imag part {}: {}", idx, c, v);
            }
        }
    }

    #[test]
    fn test_identity_multiply() {
        let one = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = double_multiply(&one, &x);
        for i in 0..8 {
            assert!((result[i] - x[i]).abs() < EPS);
        }
    }
}
