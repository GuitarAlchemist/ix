use crate::modular::{mod_inverse, mod_pow};

/// A point on an elliptic curve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Point {
    /// The point at infinity (identity element).
    Identity,
    /// An affine point (x, y).
    Affine(u64, u64),
}

/// An elliptic curve y² = x³ + ax + b over the finite field F_p.
#[derive(Debug, Clone)]
pub struct EllipticCurve {
    pub a: i64,
    pub b: i64,
    pub p: u64,
}

impl EllipticCurve {
    /// Create a new elliptic curve y² = x³ + ax + b (mod p).
    pub fn new(a: i64, b: i64, p: u64) -> Self {
        Self { a, b, p }
    }

    /// Check whether a point lies on the curve.
    pub fn is_on_curve(&self, point: &Point) -> bool {
        match point {
            Point::Identity => true,
            Point::Affine(x, y) => {
                let x = *x;
                let y = *y;
                let p = self.p;
                // y² mod p
                let lhs = mod_pow(y, 2, p);
                // x³ + a*x + b mod p
                let x3 = mod_pow(x, 3, p);
                let ax = self.mod_mul_signed(self.a, x as i64, p);
                let b_mod = ((self.b % p as i64) + p as i64) as u64 % p;
                let rhs = (x3 + ax + b_mod) % p;
                lhs == rhs
            }
        }
    }

    /// Helper to compute (signed_a * signed_b) mod p, handling negative values.
    fn mod_mul_signed(&self, a: i64, b: i64, p: u64) -> u64 {
        let a_mod = ((a % p as i64) + p as i64) as u128 % p as u128;
        let b_mod = ((b % p as i64) + p as i64) as u128 % p as u128;
        (a_mod * b_mod % p as u128) as u64
    }

    /// Point addition on the curve.
    pub fn add(&self, p1: &Point, p2: &Point) -> Point {
        match (p1, p2) {
            (Point::Identity, _) => p2.clone(),
            (_, Point::Identity) => p1.clone(),
            (Point::Affine(x1, y1), Point::Affine(x2, y2)) => {
                let p = self.p;
                let x1 = *x1;
                let y1 = *y1;
                let x2 = *x2;
                let y2 = *y2;

                if x1 == x2 {
                    if y1 != y2 || y1 == 0 {
                        // P + (-P) = O, or tangent is vertical
                        return Point::Identity;
                    }
                    // Point doubling: λ = (3x² + a) / (2y)
                    let x1_sq = (x1 as u128 * x1 as u128) % (p as u128);
                    let three_x1_sq = (3u128 * x1_sq) % (p as u128);
                    let a_mod = ((self.a % p as i64 + p as i64) as u128) % (p as u128);
                    let numerator = (three_x1_sq + a_mod) % (p as u128);
                    let denominator = (2u128 * y1 as u128) % (p as u128);
                    let inv = match mod_inverse(denominator as u64, p) {
                        Some(v) => v,
                        None => return Point::Identity,
                    };
                    let lambda = (numerator * inv as u128 % p as u128) as u64;
                    self.compute_new_point(lambda, x1, y1, x2)
                } else {
                    // Different points: λ = (y2 - y1) / (x2 - x1)
                    let numerator =
                        ((y2 as i128 - y1 as i128) % p as i128 + p as i128) as u128 % p as u128;
                    let denominator =
                        ((x2 as i128 - x1 as i128) % p as i128 + p as i128) as u128 % p as u128;
                    let inv = match mod_inverse(denominator as u64, p) {
                        Some(v) => v,
                        None => return Point::Identity,
                    };
                    let lambda = (numerator * inv as u128 % p as u128) as u64;
                    self.compute_new_point(lambda, x1, y1, x2)
                }
            }
        }
    }

    fn compute_new_point(&self, lambda: u64, x1: u64, y1: u64, x2: u64) -> Point {
        let p = self.p;
        let l2 = (lambda as u128 * lambda as u128) % p as u128;
        let x3 = ((l2 + p as u128 * 2 - x1 as u128 - x2 as u128) % p as u128) as u64;
        let y3 = ((lambda as u128 * ((x1 as u128 + p as u128 - x3 as u128) % p as u128)
            + p as u128
            - y1 as u128)
            % p as u128) as u64;
        Point::Affine(x3, y3)
    }

    /// Scalar multiplication using double-and-add.
    pub fn scalar_mul(&self, k: u64, p: &Point) -> Point {
        if k == 0 {
            return Point::Identity;
        }
        let mut result = Point::Identity;
        let mut current = p.clone();
        let mut k = k;
        while k > 0 {
            if k & 1 == 1 {
                result = self.add(&result, &current);
            }
            current = self.add(&current, &current);
            k >>= 1;
        }
        result
    }

    /// Count the number of points on the curve by brute force (small p only).
    ///
    /// Includes the point at infinity.
    pub fn curve_order_naive(&self) -> u64 {
        let p = self.p;
        let mut count = 1u64; // point at infinity
        for x in 0..p {
            let rhs = (mod_pow(x, 3, p)
                + ((self.a as i128 * x as i128 % p as i128 + p as i128) % p as i128) as u64
                + ((self.b % p as i64 + p as i64) as u64 % p))
                % p;
            for y in 0..p {
                if mod_pow(y, 2, p) == rhs {
                    count += 1;
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_curve() -> EllipticCurve {
        // y² = x³ + 2x + 3 (mod 97)
        EllipticCurve::new(2, 3, 97)
    }

    #[test]
    fn test_identity_on_curve() {
        let curve = test_curve();
        assert!(curve.is_on_curve(&Point::Identity));
    }

    #[test]
    fn test_point_on_curve() {
        let curve = test_curve();
        // Find a valid point by brute force
        let point = find_point(&curve).expect("should find a point on curve");
        assert!(curve.is_on_curve(&point));
    }

    #[test]
    fn test_point_not_on_curve() {
        let curve = test_curve();
        // Very unlikely (0, 1) is on y²=x³+2x+3 mod 97
        // rhs = 0 + 0 + 3 = 3; y²=1 mod 97 → 1≠3
        assert!(!curve.is_on_curve(&Point::Affine(0, 1)));
    }

    #[test]
    fn test_add_identity() {
        let curve = test_curve();
        let p = find_point(&curve).unwrap();
        assert_eq!(curve.add(&p, &Point::Identity), p);
        assert_eq!(curve.add(&Point::Identity, &p), p);
    }

    #[test]
    fn test_add_inverse() {
        let curve = test_curve();
        if let Point::Affine(x, y) = find_point(&curve).unwrap() {
            let neg = Point::Affine(x, (curve.p - y) % curve.p);
            let sum = curve.add(&Point::Affine(x, y), &neg);
            assert_eq!(sum, Point::Identity);
        }
    }

    #[test]
    fn test_scalar_mul_zero() {
        let curve = test_curve();
        let p = find_point(&curve).unwrap();
        assert_eq!(curve.scalar_mul(0, &p), Point::Identity);
    }

    #[test]
    fn test_scalar_mul_one() {
        let curve = test_curve();
        let p = find_point(&curve).unwrap();
        assert_eq!(curve.scalar_mul(1, &p), p);
    }

    #[test]
    fn test_scalar_mul_order() {
        let curve = test_curve();
        let order = curve.curve_order_naive();
        // By Lagrange's theorem, n*P = O for n = curve order (for any point P)
        // This holds when the group is cyclic or the point's order divides the group order.
        let p = find_point(&curve).unwrap();
        let result = curve.scalar_mul(order, &p);
        assert_eq!(result, Point::Identity);
    }

    #[test]
    fn test_curve_order_naive() {
        // Small curve: y² = x³ + x + 1 (mod 5)
        let curve = EllipticCurve::new(1, 1, 5);
        let order = curve.curve_order_naive();
        // Manually: check each x in 0..5
        // x=0: rhs=1, y²=1 → y=1,4 (2 points)
        // x=1: rhs=3, y²=3 → need y²≡3 mod 5: 0,1,4,4,1 → no
        // x=2: rhs=11%5=1, y²=1 → y=1,4 (2 points)
        // x=3: rhs=31%5=1, y²=1 → y=1,4 (2 points)
        // x=4: rhs=69%5=4, y²=4 → y=2,3 (2 points)
        // Total affine = 8, plus identity = 9
        assert_eq!(order, 9);
    }

    #[test]
    fn test_addition_associativity() {
        let curve = test_curve();
        let pts: Vec<Point> = find_n_points(&curve, 3);
        if pts.len() == 3 {
            let ab_c = curve.add(&curve.add(&pts[0], &pts[1]), &pts[2]);
            let a_bc = curve.add(&pts[0], &curve.add(&pts[1], &pts[2]));
            assert_eq!(ab_c, a_bc);
        }
    }

    /// Helper: find the first affine point on the curve.
    fn find_point(curve: &EllipticCurve) -> Option<Point> {
        for x in 0..curve.p {
            let rhs = (mod_pow(x, 3, curve.p)
                + ((curve.a as i128 * x as i128 % curve.p as i128 + curve.p as i128)
                    % curve.p as i128) as u64
                + ((curve.b % curve.p as i64 + curve.p as i64) as u64 % curve.p))
                % curve.p;
            for y in 0..curve.p {
                if mod_pow(y, 2, curve.p) == rhs {
                    return Some(Point::Affine(x, y));
                }
            }
        }
        None
    }

    /// Helper: find n distinct affine points on the curve.
    fn find_n_points(curve: &EllipticCurve, n: usize) -> Vec<Point> {
        let mut pts = Vec::new();
        for x in 0..curve.p {
            let rhs = (mod_pow(x, 3, curve.p)
                + ((curve.a as i128 * x as i128 % curve.p as i128 + curve.p as i128)
                    % curve.p as i128) as u64
                + ((curve.b % curve.p as i64 + curve.p as i64) as u64 % curve.p))
                % curve.p;
            for y in 0..curve.p {
                if mod_pow(y, 2, curve.p) == rhs {
                    pts.push(Point::Affine(x, y));
                    if pts.len() >= n {
                        return pts;
                    }
                }
            }
        }
        pts
    }
}
