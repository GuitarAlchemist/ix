---
name: ix-number-theory
description: Number theory — prime sieving, primality tests, modular arithmetic
disable-model-invocation: true
---

# Number Theory

Prime number operations and modular arithmetic utilities.

## When to Use
When the user needs prime numbers, primality testing, modular exponentiation, GCD/LCM, or modular inverses.

## Capabilities
- **Sieve of Eratosthenes** — Generate all primes up to a limit
- **Primality testing** — Trial division, Miller-Rabin (deterministic), Fermat
- **Modular exponentiation** — Fast `base^exp mod m` via squaring
- **GCD / LCM** — Greatest common divisor and least common multiple
- **Modular inverse** — Compute `a^{-1} mod m` when it exists
- **Prime gaps** — Analyze gaps between consecutive primes

## Programmatic Usage
```rust
use ix_number_theory::sieve::sieve_of_eratosthenes;
use ix_number_theory::primality::{is_prime_trial, is_prime_miller_rabin};
use ix_number_theory::modular::{mod_pow, gcd, lcm, mod_inverse};
```

## MCP Tool
Tool name: `ix_number_theory`
Operations: `sieve`, `is_prime`, `mod_pow`, `gcd`, `lcm`, `mod_inverse`, `prime_gaps`
