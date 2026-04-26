/// Euler's totient function φ(n): count of integers in [1, n] coprime to n.
pub fn euler_totient(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut result = n;
    let mut m = n;
    let mut p = 2u64;
    while p * p <= m {
        if m % p == 0 {
            while m % p == 0 {
                m /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if m > 1 {
        result -= result / m;
    }
    result
}

/// Compute φ(i) for all i in 0..=limit using a sieve.
pub fn totient_sieve(limit: usize) -> Vec<u64> {
    let mut phi: Vec<u64> = (0..=limit as u64).collect();
    for i in 2..=limit {
        if phi[i] == i as u64 {
            // i is prime
            for j in (i..=limit).step_by(i) {
                phi[j] -= phi[j] / i as u64;
            }
        }
    }
    phi
}

/// Möbius function μ(n).
///
/// - μ(1) = 1
/// - μ(n) = 0 if n has a squared prime factor
/// - μ(n) = (-1)^k if n is a product of k distinct primes
pub fn mobius(n: u64) -> i8 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut m = n;
    let mut count = 0i8;
    let mut p = 2u64;
    while p * p <= m {
        if m % p == 0 {
            m /= p;
            if m % p == 0 {
                return 0; // squared factor
            }
            count += 1;
        }
        p += 1;
    }
    if m > 1 {
        count += 1;
    }
    if count % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Number of positive divisors of n.
pub fn divisor_count(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut count = 1u64;
    let mut m = n;
    let mut p = 2u64;
    while p * p <= m {
        if m % p == 0 {
            let mut exp = 0u64;
            while m % p == 0 {
                m /= p;
                exp += 1;
            }
            count *= exp + 1;
        }
        p += 1;
    }
    if m > 1 {
        count *= 2; // remaining prime factor with exponent 1
    }
    count
}

/// Sum of positive divisors σ(n).
pub fn divisor_sum(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut result = 1u64;
    let mut m = n;
    let mut p = 2u64;
    while p * p <= m {
        if m % p == 0 {
            let mut sum = 1u64;
            let mut power = 1u64;
            while m % p == 0 {
                m /= p;
                power *= p;
                sum += power;
            }
            result *= sum;
        }
        p += 1;
    }
    if m > 1 {
        result *= 1 + m;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_totient_basic() {
        assert_eq!(euler_totient(1), 1);
        assert_eq!(euler_totient(2), 1); // p → p-1
        assert_eq!(euler_totient(7), 6); // prime
        assert_eq!(euler_totient(12), 4);
        assert_eq!(euler_totient(0), 0);
    }

    #[test]
    fn test_totient_prime() {
        // φ(p) = p-1 for prime p
        for p in [2, 3, 5, 7, 11, 13, 97] {
            assert_eq!(euler_totient(p), p - 1, "φ({p}) should be {}", p - 1);
        }
    }

    #[test]
    fn test_totient_sieve() {
        let phi = totient_sieve(20);
        for i in 0..=20u64 {
            assert_eq!(
                phi[i as usize],
                euler_totient(i),
                "sieve disagrees at i={i}"
            );
        }
    }

    #[test]
    fn test_mobius() {
        assert_eq!(mobius(1), 1);
        assert_eq!(mobius(2), -1); // single prime
        assert_eq!(mobius(6), 1); // 6 = 2*3, two distinct primes
        assert_eq!(mobius(4), 0); // 4 = 2², squared factor
        assert_eq!(mobius(30), -1); // 30 = 2*3*5, three distinct primes
        assert_eq!(mobius(12), 0); // 12 = 2²*3
    }

    #[test]
    fn test_divisor_count() {
        assert_eq!(divisor_count(1), 1);
        assert_eq!(divisor_count(6), 4); // 1,2,3,6
        assert_eq!(divisor_count(12), 6); // 1,2,3,4,6,12
        assert_eq!(divisor_count(7), 2); // prime
    }

    #[test]
    fn test_divisor_sum() {
        assert_eq!(divisor_sum(1), 1);
        assert_eq!(divisor_sum(6), 12); // 1+2+3+6 = 12 (perfect number)
        assert_eq!(divisor_sum(28), 56); // perfect number: σ(28) = 2*28
        assert_eq!(divisor_sum(7), 8); // 1+7
    }

    #[test]
    fn test_divisor_sum_0() {
        assert_eq!(divisor_sum(0), 0);
        assert_eq!(divisor_count(0), 0);
    }
}
