//! Number-theoretic utilities: prime generation, testing, and patterns.
//!
//! Provides sieve-based generation, deterministic primality testing (exact for all `u64`),
//! prime triplet enumeration, and prime factorization.
//!
//! # Examples
//!
//! ```
//! use machin_math::primes;
//!
//! // Generate primes up to 30
//! let ps = primes::sieve_of_eratosthenes(30).unwrap();
//! assert_eq!(ps, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
//!
//! // Deterministic primality testing (exact for all u64)
//! assert!(primes::is_prime(1_000_000_007));
//! assert!(!primes::is_prime(1_000_000_006));
//!
//! // Factorize a number
//! let factors = primes::prime_factors(360);
//! assert_eq!(factors, vec![(2, 3), (3, 2), (5, 1)]); // 2³ × 3² × 5
//!
//! // Find the 100th prime
//! assert_eq!(primes::nth_prime(100).unwrap(), 541);
//! ```

use crate::error::MathError;

/// Maximum allowed sieve limit (100M — ~12.5 MB bit-packed).
const MAX_SIEVE_LIMIT: u64 = 100_000_000;

/// Maximum allowed `n` for `nth_prime`.
const MAX_NTH_PRIME: usize = 10_000_000;

// ─── Sieve ──────────────────────────────────────────────────────────────────

/// Sieve of Eratosthenes using bit-packing for memory efficiency.
///
/// Returns all primes up to `limit` (inclusive). Uses a `Vec<u64>` where each
/// bit represents an odd number, yielding 8× memory reduction over `Vec<bool>`.
///
/// Returns `Err` if `limit` exceeds 100,000,000.
pub fn sieve_of_eratosthenes(limit: u64) -> Result<Vec<u64>, MathError> {
    if limit > MAX_SIEVE_LIMIT {
        return Err(MathError::InvalidParameter(format!(
            "sieve limit {} exceeds maximum {}",
            limit, MAX_SIEVE_LIMIT
        )));
    }
    if limit < 2 {
        return Ok(vec![]);
    }

    // Bit-packed sieve for odd numbers only.
    // Index i represents the number 2*i + 1.
    let size = (limit as usize) / 2 + 1;
    let num_words = size.div_ceil(64);
    let mut bits = vec![!0u64; num_words]; // all set = all composite initially treated as prime

    // Mark 1 as not prime (index 0 represents 1)
    bits[0] &= !1u64;

    let sqrt_limit = (limit as f64).sqrt() as usize;
    for i in 1..=(sqrt_limit / 2) {
        if bits[i / 64] & (1u64 << (i % 64)) != 0 {
            let prime = 2 * i + 1;
            // Mark multiples starting at prime²
            let mut j = (prime * prime - 1) / 2;
            while j < size {
                bits[j / 64] &= !(1u64 << (j % 64));
                j += prime;
            }
        }
    }

    let mut primes = Vec::with_capacity((limit as f64 / (limit as f64).ln()) as usize + 10);
    primes.push(2);
    for i in 1..size {
        if 2 * i + 1 > limit as usize {
            break;
        }
        if bits[i / 64] & (1u64 << (i % 64)) != 0 {
            primes.push((2 * i + 1) as u64);
        }
    }

    Ok(primes)
}

// ─── Primality testing ──────────────────────────────────────────────────────

/// Deterministic primality test, exact for all `u64`.
///
/// - `n < 1_000_000`: trial division with sqrt optimization.
/// - `n >= 1_000_000`: deterministic Miller-Rabin with 12 witnesses.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    if n < 1_000_000 {
        return trial_division(n);
    }
    miller_rabin(n)
}

fn trial_division(n: u64) -> bool {
    let mut i = 5u64;
    while i * i <= n {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
        i += 6;
    }
    true
}

/// Deterministic Miller-Rabin with witnesses exact for all n < 2^64.
fn miller_rabin(n: u64) -> bool {
    const WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        r += 1;
    }

    'witness: for &a in &WITNESSES {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Modular exponentiation using u128 to prevent overflow.
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = mod_mul(result, base, modulus);
        }
        exp /= 2;
        base = mod_mul(base, base, modulus);
    }
    result
}

/// Modular multiplication using u128 intermediates.
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

// ─── nth_prime ──────────────────────────────────────────────────────────────

/// Returns the n-th prime (1-indexed: `nth_prime(1) = 2`).
///
/// Uses the prime-counting estimate to size an internal sieve.
/// Returns `Err` if `n` is 0 or exceeds 10,000,000.
pub fn nth_prime(n: usize) -> Result<u64, MathError> {
    if n == 0 {
        return Err(MathError::InvalidParameter(
            "nth_prime is 1-indexed; n must be >= 1".into(),
        ));
    }
    if n > MAX_NTH_PRIME {
        return Err(MathError::InvalidParameter(format!(
            "nth_prime({}) exceeds maximum {}",
            n, MAX_NTH_PRIME
        )));
    }
    if n <= 6 {
        return Ok([2, 3, 5, 7, 11, 13][n - 1]);
    }

    // Upper bound estimate: p_n < n * (ln(n) + ln(ln(n))) * 1.3
    let nf = n as f64;
    let estimate = (nf * (nf.ln() + nf.ln().ln()) * 1.3) as u64;
    let limit = estimate.min(MAX_SIEVE_LIMIT);

    let primes = sieve_of_eratosthenes(limit)?;
    if primes.len() < n {
        return Err(MathError::InvalidParameter(format!(
            "sieve too small for nth_prime({}); try a smaller n",
            n
        )));
    }
    Ok(primes[n - 1])
}

// ─── Prime triplets ─────────────────────────────────────────────────────────

/// Find all prime triplets `(p, p+2, p+6)` where `p <= limit`.
///
/// Returns `Err` if `limit` exceeds the sieve maximum.
pub fn prime_triplets(limit: u64) -> Result<Vec<(u64, u64, u64)>, MathError> {
    let primes = sieve_of_eratosthenes(limit)?;
    let prime_set: std::collections::HashSet<u64> = primes.iter().copied().collect();

    let mut triplets = Vec::new();
    for &p in &primes {
        if p + 6 > limit {
            break;
        }
        if prime_set.contains(&(p + 2)) && prime_set.contains(&(p + 6)) {
            triplets.push((p, p + 2, p + 6));
        }
    }
    Ok(triplets)
}

// ─── Prime factorization ────────────────────────────────────────────────────

/// Factorize `n` into (prime, exponent) pairs via trial division.
///
/// Returns an empty vec for `n < 2`.
pub fn prime_factors(n: u64) -> Vec<(u64, u32)> {
    if n < 2 {
        return vec![];
    }
    let mut factors = Vec::new();
    let mut remaining = n;

    // Check factor 2
    let mut count = 0u32;
    while remaining.is_multiple_of(2) {
        count += 1;
        remaining /= 2;
    }
    if count > 0 {
        factors.push((2, count));
    }

    // Check odd factors from 3
    let mut i = 3u64;
    while i * i <= remaining {
        let mut count = 0u32;
        while remaining.is_multiple_of(i) {
            count += 1;
            remaining /= i;
        }
        if count > 0 {
            factors.push((i, count));
        }
        i += 2;
    }

    if remaining > 1 {
        factors.push((remaining, 1));
    }

    factors
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sieve_small() {
        let primes = sieve_of_eratosthenes(30).unwrap();
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_sieve_zero_and_one() {
        assert_eq!(sieve_of_eratosthenes(0).unwrap(), vec![]);
        assert_eq!(sieve_of_eratosthenes(1).unwrap(), vec![]);
    }

    #[test]
    fn test_sieve_two() {
        assert_eq!(sieve_of_eratosthenes(2).unwrap(), vec![2]);
    }

    #[test]
    fn test_sieve_exceeds_limit() {
        assert!(sieve_of_eratosthenes(MAX_SIEVE_LIMIT + 1).is_err());
    }

    #[test]
    fn test_sieve_count_to_1000() {
        let primes = sieve_of_eratosthenes(1000).unwrap();
        assert_eq!(primes.len(), 168); // known count of primes <= 1000
    }

    #[test]
    fn test_is_prime_small() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(!is_prime(9));
        assert!(is_prime(97));
    }

    #[test]
    fn test_is_prime_large() {
        // Known large primes
        assert!(is_prime(999_999_937)); // largest prime below 10^9
        assert!(is_prime(1_000_000_007)); // common hash prime
        assert!(!is_prime(1_000_000_006));
    }

    #[test]
    fn test_is_prime_consistency_with_sieve() {
        let primes = sieve_of_eratosthenes(10_000).unwrap();
        let prime_set: std::collections::HashSet<u64> = primes.iter().copied().collect();
        for n in 0..10_000u64 {
            assert_eq!(
                is_prime(n),
                prime_set.contains(&n),
                "disagreement at n={}",
                n
            );
        }
    }

    #[test]
    fn test_nth_prime() {
        assert_eq!(nth_prime(1).unwrap(), 2);
        assert_eq!(nth_prime(2).unwrap(), 3);
        assert_eq!(nth_prime(5).unwrap(), 11);
        assert_eq!(nth_prime(100).unwrap(), 541);
        assert_eq!(nth_prime(1000).unwrap(), 7919);
    }

    #[test]
    fn test_nth_prime_zero_fails() {
        assert!(nth_prime(0).is_err());
    }

    #[test]
    fn test_prime_triplets_small() {
        let trips = prime_triplets(30).unwrap();
        // (5,7,11), (7,9,13)? No — 9 isn't prime. (5,7,11), (11,13,17)
        assert!(trips.contains(&(5, 7, 11)));
        assert!(trips.contains(&(11, 13, 17)));
    }

    #[test]
    fn test_prime_factors_basic() {
        assert_eq!(prime_factors(1), vec![]);
        assert_eq!(prime_factors(2), vec![(2, 1)]);
        assert_eq!(prime_factors(12), vec![(2, 2), (3, 1)]);
        assert_eq!(prime_factors(100), vec![(2, 2), (5, 2)]);
    }

    #[test]
    fn test_prime_factors_prime_input() {
        assert_eq!(prime_factors(97), vec![(97, 1)]);
    }

    #[test]
    fn test_prime_factors_large() {
        // 2^10 = 1024
        assert_eq!(prime_factors(1024), vec![(2, 10)]);
    }

    #[test]
    fn test_prime_factors_roundtrip() {
        // Verify factors multiply back to original
        let n = 2 * 2 * 3 * 5 * 7 * 11 * 13;
        let factors = prime_factors(n);
        let reconstructed: u64 = factors.iter().map(|(p, e)| p.pow(*e)).product();
        assert_eq!(reconstructed, n);
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24); // 1024 % 1000
        assert_eq!(mod_pow(3, 0, 7), 1);
    }
}
