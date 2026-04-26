use crate::modular::{extended_gcd, gcd};

/// Chinese Remainder Theorem: given a list of (remainder, modulus) pairs,
/// find the smallest non-negative x satisfying all congruences simultaneously.
///
/// Returns `None` if the moduli are not pairwise coprime.
///
/// # Example
/// ```
/// use ix_number_theory::crt::chinese_remainder_theorem;
/// // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → x = 23
/// let x = chinese_remainder_theorem(&[(2, 3), (3, 5), (2, 7)]);
/// assert_eq!(x, Some(23));
/// ```
pub fn chinese_remainder_theorem(residues: &[(u64, u64)]) -> Option<u64> {
    if residues.is_empty() {
        return Some(0);
    }

    // Check pairwise coprimality
    for i in 0..residues.len() {
        for j in (i + 1)..residues.len() {
            if gcd(residues[i].1, residues[j].1) != 1 {
                return None;
            }
        }
    }

    let mut combined_remainder = residues[0].0 as i128;
    let mut combined_modulus = residues[0].1 as i128;

    for &(r, m) in &residues[1..] {
        let r = r as i128;
        let m = m as i128;
        let (_, x, _) = extended_gcd(combined_modulus as i64, m as i64);
        let x = x as i128;
        // combined_remainder + combined_modulus * x * (r - combined_remainder) ≡ r (mod m)
        let diff = ((r - combined_remainder) % m + m) % m;
        let x_mod = ((x % m) + m) % m;
        let step = (x_mod * (diff % m)) % m;
        combined_remainder += combined_modulus * step;
        combined_modulus *= m;
        combined_remainder =
            ((combined_remainder % combined_modulus) + combined_modulus) % combined_modulus;
    }

    Some(combined_remainder as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crt_basic() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → x = 23
        let result = chinese_remainder_theorem(&[(2, 3), (3, 5), (2, 7)]);
        assert_eq!(result, Some(23));
    }

    #[test]
    fn test_crt_two_congruences() {
        // x ≡ 1 (mod 2), x ≡ 2 (mod 3) → x = 5
        let result = chinese_remainder_theorem(&[(1, 2), (2, 3)]);
        assert_eq!(result, Some(5));
    }

    #[test]
    fn test_crt_non_coprime() {
        // moduli 4 and 6 share factor 2
        let result = chinese_remainder_theorem(&[(1, 4), (2, 6)]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_crt_single() {
        let result = chinese_remainder_theorem(&[(3, 7)]);
        assert_eq!(result, Some(3));
    }

    #[test]
    fn test_crt_empty() {
        let result = chinese_remainder_theorem(&[]);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_crt_verify() {
        let residues = [(2, 3), (3, 5), (2, 7)];
        let x = chinese_remainder_theorem(&residues).unwrap();
        for &(r, m) in &residues {
            assert_eq!(x % m, r, "x={x} should be ≡ {r} (mod {m})");
        }
    }
}
