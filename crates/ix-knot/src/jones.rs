//! The Jones polynomial of a braid's closure: the Kauffman bracket, evaluated in
//! the Temperley–Lieb algebra.
//!
//! Each crossing expands as σₖ = A·1 + A⁻¹·eₖ and σₖ⁻¹ = A⁻¹·1 + A·eₖ, where eₖ
//! joins positions k and k + 1 with a cap above and a cup below. The word's
//! product is then a combination of planar diagrams, each a non-crossing
//! matching of the n top points and n bottom points, picking up the loop factor
//! d = -A² - A⁻² whenever a composition closes a circle. There are Catalan(n) of
//! them, so the cost grows with the crossings, not 2^crossings.
//!
//! The closure joins top point j to bottom point j. A diagram that closes into c
//! circles contributes d^(c-1), so the unknot's bracket is 1, and
//! V(t) = (-A³)^(-writhe) · <closure> with t = A⁻⁴. With these conventions the
//! closure of σ₁³ is the right-handed trefoil, V = t + t³ - t⁴.

use crate::braid::Braid;
use crate::poly::Laurent;
use std::collections::BTreeMap;
use std::fmt;

/// A Jones polynomial, as (power of t^(1/2), coefficient) in ascending order
/// with no zero coefficients. The powers are counted in halves because a link
/// with an even number of components has half-integer powers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jones {
    terms: Vec<(i32, i128)>,
}

impl Jones {
    /// (power of t^(1/2), coefficient), ascending.
    pub fn terms(&self) -> &[(i32, i128)] {
        &self.terms
    }

    /// V(1/t), the Jones polynomial of the mirror image.
    pub fn mirror(&self) -> Self {
        let mut terms: Vec<_> = self.terms.iter().map(|(h, c)| (-h, *c)).collect();
        terms.reverse();
        Self { terms }
    }

    /// V(t) = V(1/t). An amphichiral link has it; having it does not prove a
    /// link amphichiral.
    pub fn is_symmetric(&self) -> bool {
        *self == self.mirror()
    }

    /// V(1), which is (-2)^(components - 1) for every link.
    pub fn at_one(&self) -> i128 {
        self.terms.iter().map(|(_, c)| c).sum()
    }
}

/// Written in t, e.g. `t + t^3 - t^4`, `-t^(1/2) - t^(5/2)`, `t^-2 - t^-1 + 1`.
impl fmt::Display for Jones {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, &(half, coeff)) in self.terms.iter().enumerate() {
            let sign = if coeff < 0 { "-" } else { "+" };
            match i {
                0 if coeff < 0 => f.write_str("-")?,
                0 => {}
                _ => write!(f, " {sign} ")?,
            }
            let power = match (half % 2 == 0, half / 2) {
                (true, 0) => None,
                (true, 1) => Some("t".to_string()),
                (true, p) => Some(format!("t^{p}")),
                (false, _) => Some(format!("t^({half}/2)")),
            };
            match (coeff.unsigned_abs(), power) {
                (c, None) => write!(f, "{c}")?,
                (1, Some(p)) => f.write_str(&p)?,
                (c, Some(p)) => write!(f, "{c}{p}")?,
            }
        }
        Ok(())
    }
}

/// The Jones polynomial of the braid's closure.
// @ai:invariant jones() of a braid closure agrees with the 2^crossings Kauffman state sum, is unchanged by conjugation and Markov stabilization, and gives V(1) = (-2)^(components-1) [T:test conf:0.9 src:jones::tests::the_algebra_agrees_with_the_state_sum_and_the_markov_moves]
pub fn jones(braid: &Braid) -> Jones {
    normalize(&bracket(braid), braid.writhe())
}

/// A planar diagram: `m[p]` is the point matched to point `p`. Points `0..n` are
/// the top, `n..2n` the bottom, left to right. `n <= 8`, so they fit a `u8`.
type Diagram = Vec<u8>;

/// The bracket of the closure, normalized so the unknot's is 1.
fn bracket(braid: &Braid) -> Laurent {
    let n = braid.strands();
    let identity: Diagram = (0..2 * n).map(|p| ((p + n) % (2 * n)) as u8).collect();
    let mut states = BTreeMap::from([(identity, Laurent::one())]);
    for &g in braid.word() {
        let k = g.unsigned_abs() as usize - 1;
        let a = g.signum(); // A's exponent on the identity term; the cup-cap term gets -a
        let mut next: BTreeMap<Diagram, Laurent> = BTreeMap::new();
        for (diagram, coeff) in &states {
            next.entry(diagram.clone())
                .or_default()
                .add_scaled(coeff, 1, a);
            let (capped, closed_loop) = with_cup_cap(diagram, n, k);
            let coeff = if closed_loop {
                coeff.times_loop()
            } else {
                coeff.clone()
            };
            next.entry(capped).or_default().add_scaled(&coeff, 1, -a);
        }
        next.retain(|_, c| *c != Laurent::default());
        states = next;
    }
    let mut total = Laurent::default();
    for (diagram, coeff) in &states {
        let mut term = coeff.clone();
        for _ in 1..closure_loops(diagram, n) {
            term = term.times_loop();
        }
        total.add_scaled(&term, 1, 0);
    }
    total
}

/// `diagram` with eₖ stacked below it: its bottom points k and k+1 are capped
/// together (closing a circle if they were matched to each other), and the new
/// bottom points k and k+1 are a cup.
fn with_cup_cap(diagram: &[u8], n: usize, k: usize) -> (Diagram, bool) {
    let (left, right) = (n + k, n + k + 1);
    let (x, y) = (diagram[left] as usize, diagram[right] as usize);
    let mut out = diagram.to_vec();
    let closed_loop = x == right;
    if !closed_loop {
        out[x] = y as u8;
        out[y] = x as u8;
    }
    out[left] = right as u8;
    out[right] = left as u8;
    (out, closed_loop)
}

/// The circles the closure of `diagram` makes: each step crosses a matching arc
/// then the closing arc from a bottom point to the top point above it, or back.
fn closure_loops(diagram: &[u8], n: usize) -> usize {
    let mut seen = vec![false; 2 * n];
    let mut loops = 0;
    for start in 0..2 * n {
        if seen[start] {
            continue;
        }
        loops += 1;
        let mut p = start;
        loop {
            seen[p] = true;
            let q = diagram[p] as usize;
            seen[q] = true;
            p = (q + n) % (2 * n);
            if p == start {
                break;
            }
        }
    }
    loops
}

/// V(t) = (-A³)^(-writhe) · bracket, then A^e = t^(-e/4) = (t^(1/2))^(-e/2).
/// Every exponent of the bracket has the parity of the crossing count, as does
/// 3·writhe, so `e - 3·writhe` is even.
fn normalize(bracket: &Laurent, writhe: i64) -> Jones {
    let sign = if writhe % 2 == 0 { 1 } else { -1 };
    let shift = 3 * writhe as i32;
    let mut terms: Vec<(i32, i128)> = bracket
        .terms()
        .map(|(e, c)| (-(e - shift) / 2, sign * c))
        .collect();
    terms.sort_unstable();
    Jones { terms }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(text: &str) -> Jones {
        jones(&Braid::parse(None, text).unwrap())
    }

    fn on(strands: usize, text: &str) -> Jones {
        jones(&Braid::parse(Some(strands), text).unwrap())
    }

    /// Product of two Jones polynomials, for connected sums.
    fn times(a: &Jones, b: &Jones) -> Jones {
        let mut acc: BTreeMap<i32, i128> = BTreeMap::new();
        for (ha, ca) in a.terms() {
            for (hb, cb) in b.terms() {
                *acc.entry(ha + hb).or_insert(0) += ca * cb;
            }
        }
        Jones {
            terms: acc.into_iter().filter(|(_, c)| *c != 0).collect(),
        }
    }

    #[test]
    fn known_knots_and_links() {
        assert_eq!(on(1, "").to_string(), "1");
        assert_eq!(v("s1").to_string(), "1");
        assert_eq!(v("s1^-1 s2 s3^-1").to_string(), "1");
        // Three unlinked circles: d² = (-t^(1/2) - t^(-1/2))².
        assert_eq!(on(3, "").to_string(), "t^-1 + 2 + t");
        assert_eq!(v("s1^2").to_string(), "-t^(1/2) - t^(5/2)");
        assert_eq!(v("s1^3").to_string(), "t + t^3 - t^4");
        assert_eq!(v("s1^-3").to_string(), "-t^-4 + t^-3 + t^-1");
        assert_eq!(
            v("s1 s2^-1 s1 s2^-1").to_string(),
            "t^-2 - t^-1 + 1 - t + t^2"
        );
        // Borromean rings, L6a4 in the Thistlethwaite table.
        assert_eq!(
            v("s1 s2^-1 s1 s2^-1 s1 s2^-1").to_string(),
            "-t^-3 + 3t^-2 - 2t^-1 + 4 - 2t + 3t^2 - t^3"
        );
    }

    #[test]
    fn the_reef_knot_is_symmetric_and_the_granny_is_not() {
        let trefoil = v("s1^3");
        let reef = v("s1^3 s2^-3");
        let granny = v("s1^3 s2^3");
        assert!(reef.is_symmetric());
        assert!(!granny.is_symmetric());
        assert!(!trefoil.is_symmetric());
        assert!(v("s1 s2^-1 s1 s2^-1").is_symmetric());
        // A connected sum's polynomial is the product of the summands'.
        assert_eq!(granny, times(&trefoil, &trefoil));
        assert_eq!(reef, times(&trefoil, &trefoil.mirror()));
        assert_eq!(v("s1^-3"), trefoil.mirror());
    }

    /// The bracket as its definition reads: every one of the 2^crossings
    /// smoothings, its circles counted by union-find over the closed diagram.
    /// Shares only the crossing convention with the Temperley–Lieb version.
    fn state_sum(braid: &Braid) -> Laurent {
        let (n, c) = (braid.strands(), braid.crossings());
        let node = |level: usize, position: usize| (level % c.max(1)) * n + position;
        let mut total = Laurent::default();
        for mask in 0u64..1 << c {
            let mut parent: Vec<usize> = (0..n * c.max(1)).collect();
            fn find(parent: &mut [usize], mut x: usize) -> usize {
                while parent[x] != x {
                    parent[x] = parent[parent[x]];
                    x = parent[x];
                }
                x
            }
            let mut join = |a: usize, b: usize| {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent[ra] = rb;
            };
            let mut exponent = 0;
            for (j, &g) in braid.word().iter().enumerate() {
                let k = g.unsigned_abs() as usize - 1;
                let cupped = mask >> j & 1 == 1;
                exponent += if cupped { -g.signum() } else { g.signum() };
                for p in (0..n).filter(|p| *p != k && *p != k + 1) {
                    join(node(j, p), node(j + 1, p));
                }
                if cupped {
                    join(node(j, k), node(j, k + 1));
                    join(node(j + 1, k), node(j + 1, k + 1));
                } else {
                    join(node(j, k), node(j + 1, k));
                    join(node(j, k + 1), node(j + 1, k + 1));
                }
            }
            let circles = if c == 0 {
                n
            } else {
                (0..n * c).filter(|x| find(&mut parent, *x) == *x).count()
            };
            let mut term = Laurent::one();
            for _ in 1..circles {
                term = term.times_loop();
            }
            total.add_scaled(&term, 1, exponent);
        }
        total
    }

    /// Words from a fixed linear congruential sequence: no RNG dependency, the
    /// same words on every run.
    fn words(count: usize) -> Vec<Braid> {
        let mut x: u64 = 0x2026_1004;
        let mut next = |m: u64| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) % m
        };
        (0..count)
            .map(|_| {
                let strands = 2 + next(3) as usize; // 2..=4
                let len = next(11) as usize; // 0..=10
                let word = (0..len)
                    .map(|_| {
                        let k = 1 + next(strands as u64 - 1) as i32;
                        if next(2) == 0 {
                            k
                        } else {
                            -k
                        }
                    })
                    .collect();
                Braid::new(strands, word).unwrap()
            })
            .collect()
    }

    #[test]
    fn the_algebra_agrees_with_the_state_sum_and_the_markov_moves() {
        for b in words(150) {
            let n = b.strands();
            let jb = jones(&b);
            assert_eq!(jb, normalize(&state_sum(&b), b.writhe()), "state sum: {b}");
            let components = b.components() as u32;
            assert_eq!(jb.at_one(), (-2i128).pow(components - 1), "V(1): {b}");
            assert_eq!(jones(&b.mirror()), jb.mirror(), "mirror: {b}");
            // Stabilization: one more strand, crossed once, either way.
            for s in [n as i32, -(n as i32)] {
                let mut word = b.word().to_vec();
                word.push(s);
                assert_eq!(
                    jones(&Braid::new(n + 1, word).unwrap()),
                    jb,
                    "stabilize {s}: {b}"
                );
            }
            // Conjugation, by a generator and by a rotation of the word.
            let mut word = vec![1];
            word.extend_from_slice(b.word());
            word.push(-1);
            assert_eq!(jones(&Braid::new(n, word).unwrap()), jb, "conjugate: {b}");
            if let Some((first, rest)) = b.word().split_first() {
                let mut rotated = rest.to_vec();
                rotated.push(*first);
                assert_eq!(jones(&Braid::new(n, rotated).unwrap()), jb, "rotate: {b}");
            }
        }
    }

    /// At the crossing cap on the widest braid, the coefficients stay in range:
    /// a debug build panics on overflow.
    #[test]
    fn words_at_the_caps_evaluate() {
        let torus = on(2, "s1^64");
        assert_eq!(torus.at_one(), -2);
        let wide: Vec<i32> = (1..8).cycle().take(63).collect();
        let b = Braid::new(8, wide).unwrap();
        assert_eq!(jones(&b).at_one(), (-2i128).pow(b.components() as u32 - 1));
    }
}
