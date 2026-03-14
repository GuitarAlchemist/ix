use crate::modular::mod_pow;

/// Primality test via trial division. Handles all values of `n`.
pub fn is_prime_trial(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5u64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Miller-Rabin primality test, deterministic for n < 3.3×10^24.
///
/// Uses fixed witnesses [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].
/// The `_rounds` parameter is accepted for API compatibility but ignored
/// since the deterministic witness set is always used.
pub fn is_prime_miller_rabin(n: u64, _rounds: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    let witnesses: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    'witness: for &a in witnesses {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mod_pow(x, 2, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Fermat primality test. Probabilistic — can be fooled by Carmichael numbers.
///
/// Tests with witnesses 2..2+rounds (skipping any >= n).
pub fn is_prime_fermat(n: u64, rounds: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    for i in 0..rounds {
        let a = 2 + i as u64;
        if a >= n {
            break;
        }
        if mod_pow(a, n - 1, n) != 1 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_small() {
        assert!(!is_prime_trial(0));
        assert!(!is_prime_trial(1));
        assert!(is_prime_trial(2));
        assert!(is_prime_trial(3));
        assert!(!is_prime_trial(4));
        assert!(is_prime_trial(5));
        assert!(is_prime_trial(97));
        assert!(!is_prime_trial(100));
    }

    #[test]
    fn test_miller_rabin_small() {
        assert!(is_prime_miller_rabin(2, 10));
        assert!(is_prime_miller_rabin(3, 10));
        assert!(!is_prime_miller_rabin(4, 10));
        assert!(is_prime_miller_rabin(7, 10));
        assert!(is_prime_miller_rabin(104729, 10));
    }

    #[test]
    fn test_miller_rabin_carmichael() {
        // Carmichael numbers fool Fermat but not Miller-Rabin
        assert!(!is_prime_miller_rabin(561, 10));
        assert!(!is_prime_miller_rabin(1105, 10));
        assert!(!is_prime_miller_rabin(1729, 10));
    }

    #[test]
    fn test_fermat_small() {
        assert!(is_prime_fermat(2, 5));
        assert!(is_prime_fermat(3, 5));
        assert!(!is_prime_fermat(4, 5));
        assert!(is_prime_fermat(97, 10));
    }

    #[test]
    fn test_fermat_carmichael_fools() {
        // 561 is Carmichael: Fermat with base 2 says "prime" (since 2^560 ≡ 1 mod 561)
        // But it is NOT prime. This demonstrates the weakness.
        // Note: Fermat may or may not be fooled depending on chosen bases.
        // 561 = 3 * 11 * 17. Bases 2,.. that are coprime to 561 will pass.
        // a=3 divides 561, so mod_pow(3, 560, 561) != 1, Fermat catches it with enough rounds.
        // With rounds=1 (only base 2): Fermat is fooled.
        assert!(is_prime_fermat(561, 1)); // fooled with just base 2
    }

    #[test]
    fn test_large_prime() {
        // A known large prime
        assert!(is_prime_miller_rabin(1_000_000_007, 10));
        assert!(is_prime_trial(1_000_000_007));
    }

    #[test]
    fn test_agreement_small_range() {
        for n in 0..1000 {
            assert_eq!(
                is_prime_trial(n),
                is_prime_miller_rabin(n, 10),
                "disagreement at n={n}"
            );
        }
    }
}
