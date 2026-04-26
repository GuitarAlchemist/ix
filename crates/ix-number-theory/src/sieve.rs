/// Returns all primes up to `limit` using the Sieve of Eratosthenes.
pub fn sieve_of_eratosthenes(limit: usize) -> Vec<usize> {
    if limit < 2 {
        return Vec::new();
    }
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut i = 2;
    while i * i <= limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    is_prime
        .iter()
        .enumerate()
        .filter_map(|(idx, &p)| if p { Some(idx) } else { None })
        .collect()
}

/// Returns all primes up to `limit` using the Sieve of Atkin.
pub fn sieve_of_atkin(limit: usize) -> Vec<usize> {
    if limit < 2 {
        return Vec::new();
    }
    let mut sieve = vec![false; limit + 1];

    let mut x = 1usize;
    while x * x <= limit {
        let mut y = 1usize;
        while y * y <= limit {
            // 4x² + y²
            let n = 4 * x * x + y * y;
            if n <= limit && (n % 12 == 1 || n % 12 == 5) {
                sieve[n] = !sieve[n];
            }
            // 3x² + y²
            let n = 3 * x * x + y * y;
            if n <= limit && n % 12 == 7 {
                sieve[n] = !sieve[n];
            }
            // 3x² - y² (only when x > y)
            if x > y {
                let n = 3 * x * x - y * y;
                if n <= limit && n % 12 == 11 {
                    sieve[n] = !sieve[n];
                }
            }
            y += 1;
        }
        x += 1;
    }

    // Eliminate composites by sieving
    let mut r = 5;
    while r * r <= limit {
        if sieve[r] {
            let mut i = r * r;
            while i <= limit {
                sieve[i] = false;
                i += r * r;
            }
        }
        r += 1;
    }

    let mut result = Vec::new();
    if limit >= 2 {
        result.push(2);
    }
    if limit >= 3 {
        result.push(3);
    }
    for (i, &is_prime) in sieve.iter().enumerate().take(limit + 1).skip(5) {
        if is_prime {
            result.push(i);
        }
    }
    result
}

/// Returns all primes in the range [lo, hi] using a segmented sieve.
pub fn segmented_sieve(lo: usize, hi: usize) -> Vec<usize> {
    if hi < 2 || lo > hi {
        return Vec::new();
    }
    let lo = if lo < 2 { 2 } else { lo };

    // Get small primes up to sqrt(hi)
    let isqrt = (hi as f64).sqrt() as usize + 1;
    let small_primes = sieve_of_eratosthenes(isqrt);

    let size = hi - lo + 1;
    let mut is_prime = vec![true; size];

    for &p in &small_primes {
        // Find the first multiple of p >= lo
        let start = if p * p >= lo {
            p * p - lo
        } else {
            let rem = lo % p;
            if rem == 0 {
                0
            } else {
                p - rem
            }
        };
        let mut j = start;
        while j < size {
            // Don't mark p itself as composite
            if j + lo != p {
                is_prime[j] = false;
            }
            j += p;
        }
    }

    is_prime
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| if p { Some(i + lo) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRIMES_UP_TO_100: &[usize] = &[
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97,
    ];

    #[test]
    fn test_eratosthenes_100() {
        assert_eq!(sieve_of_eratosthenes(100), PRIMES_UP_TO_100);
    }

    #[test]
    fn test_atkin_100() {
        assert_eq!(sieve_of_atkin(100), PRIMES_UP_TO_100);
    }

    #[test]
    fn test_sieves_agree() {
        for limit in [10, 50, 200, 1000] {
            assert_eq!(
                sieve_of_eratosthenes(limit),
                sieve_of_atkin(limit),
                "sieves disagree at limit={limit}"
            );
        }
    }

    #[test]
    fn test_segmented_matches_full() {
        let full = sieve_of_eratosthenes(1000);
        let seg = segmented_sieve(2, 1000);
        assert_eq!(full, seg);
    }

    #[test]
    fn test_segmented_subrange() {
        let seg = segmented_sieve(50, 100);
        let expected: Vec<usize> = PRIMES_UP_TO_100
            .iter()
            .copied()
            .filter(|&p| p >= 50)
            .collect();
        assert_eq!(seg, expected);
    }

    #[test]
    fn test_sieve_2() {
        assert_eq!(sieve_of_eratosthenes(2), vec![2]);
        assert_eq!(sieve_of_atkin(2), vec![2]);
    }

    #[test]
    fn test_sieve_0() {
        assert_eq!(sieve_of_eratosthenes(0), Vec::<usize>::new());
        assert_eq!(sieve_of_atkin(0), Vec::<usize>::new());
    }

    #[test]
    fn test_sieve_1() {
        assert_eq!(sieve_of_eratosthenes(1), Vec::<usize>::new());
        assert_eq!(sieve_of_atkin(1), Vec::<usize>::new());
    }

    #[test]
    fn test_segmented_empty_range() {
        assert_eq!(segmented_sieve(100, 50), Vec::<usize>::new());
        assert_eq!(segmented_sieve(0, 1), Vec::<usize>::new());
    }
}
