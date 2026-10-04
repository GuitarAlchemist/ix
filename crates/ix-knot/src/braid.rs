//! Braid words: parsing, the strand permutation, components and writhe.

use std::fmt;

/// The most strands a braid may have. The bracket's state space is the
/// Temperley–Lieb basis, Catalan(n) diagrams: 1430 at n = 8.
pub const MAX_STRANDS: usize = 8;
/// The most crossings a braid word may have, after any repetition.
///
/// The bound keeps the bracket's coefficients inside `i128`. Each crossing at
/// most triples the sum of the absolute values of all the coefficients (the
/// identity term keeps it, the cup-cap term may double it with a loop factor),
/// and the closure multiplies it by at most 2^(strands-1). So no coefficient
/// exceeds 3^64 · 2^7 < 2^110.
pub const MAX_CROSSINGS: usize = 64;

/// Why a braid was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BraidError {
    #[error("a braid needs 1 to {MAX_STRANDS} strands, got {0}")]
    Strands(usize),
    #[error("generator {generator} needs 1 <= |k| < strands ({strands})")]
    Generator { generator: i32, strands: usize },
    #[error("a braid word may have at most {MAX_CROSSINGS} crossings, got {0}")]
    Crossings(usize),
    #[error("cannot read {0:?} as a generator: write s1, s2^-1, s1^3 or a signed integer")]
    Token(String),
}

/// A braid word on `strands` strands.
///
/// Generator `k > 0` is σₖ, where the strand in position k crosses in front of
/// the strand in position k + 1; `-k` is σₖ⁻¹, the same pair crossing the other
/// way. Positions count from 1, left to right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Braid {
    strands: usize,
    word: Vec<i32>,
}

impl Braid {
    /// A braid from its generators, each a nonzero `±k` with `k < strands`.
    pub fn new(strands: usize, word: Vec<i32>) -> Result<Self, BraidError> {
        if strands == 0 || strands > MAX_STRANDS {
            return Err(BraidError::Strands(strands));
        }
        if word.len() > MAX_CROSSINGS {
            return Err(BraidError::Crossings(word.len()));
        }
        if let Some(&generator) = word
            .iter()
            .find(|g| **g == 0 || g.unsigned_abs() as usize >= strands)
        {
            return Err(BraidError::Generator { generator, strands });
        }
        Ok(Self { strands, word })
    }

    /// Read a word such as `"s1 s2^-1"`, `"s1^3 s2^-3"` or `"1 -2"`. Tokens are
    /// separated by spaces or commas; `σ` may stand for `s`. Without `strands`,
    /// the braid has one more strand than its largest generator.
    pub fn parse(strands: Option<usize>, text: &str) -> Result<Self, BraidError> {
        let mut word = Vec::new();
        for token in text
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
        {
            let (k, power) = parse_token(token)?;
            let len = word.len().saturating_add(power.unsigned_abs() as usize);
            if len > MAX_CROSSINGS {
                return Err(BraidError::Crossings(len));
            }
            let generator = if power > 0 { k } else { -k };
            // `repeat_n` would need Rust 1.82; the workspace MSRV is 1.80.
            word.extend(std::iter::repeat(generator).take(power.unsigned_abs() as usize));
        }
        let strands = strands.unwrap_or_else(|| {
            word.iter()
                .map(|g| g.unsigned_abs() as usize + 1)
                .max()
                .unwrap_or(1)
        });
        Self::new(strands, word)
    }

    pub fn strands(&self) -> usize {
        self.strands
    }

    /// The generators, `±k` for σₖ^±1.
    pub fn word(&self) -> &[i32] {
        &self.word
    }

    pub fn crossings(&self) -> usize {
        self.word.len()
    }

    /// The sum of the crossing signs: +1 for each σₖ, -1 for each σₖ⁻¹.
    pub fn writhe(&self) -> i64 {
        self.word.iter().map(|g| i64::from(g.signum())).sum()
    }

    /// The word written `times` times in a row.
    pub fn repeat(&self, times: usize) -> Result<Self, BraidError> {
        let len = self.word.len().saturating_mul(times);
        if len > MAX_CROSSINGS {
            return Err(BraidError::Crossings(len));
        }
        Self::new(self.strands, self.word.repeat(times))
    }

    /// The mirror image: every crossing flipped.
    pub fn mirror(&self) -> Self {
        Self {
            strands: self.strands,
            word: self.word.iter().map(|g| -g).collect(),
        }
    }

    /// `perm[i]` is the position, 0-based, at which the strand that starts in
    /// position `i` ends.
    pub fn permutation(&self) -> Vec<usize> {
        let mut at: Vec<usize> = (0..self.strands).collect(); // at[position] = strand
        for g in &self.word {
            let k = g.unsigned_abs() as usize;
            at.swap(k - 1, k);
        }
        let mut perm = vec![0; self.strands];
        for (position, strand) in at.into_iter().enumerate() {
            perm[strand] = position;
        }
        perm
    }

    /// The number of components of the closure: one per cycle of the
    /// permutation, since the closure joins each end position to the same start
    /// position.
    pub fn components(&self) -> usize {
        let perm = self.permutation();
        let mut seen = vec![false; perm.len()];
        let mut cycles = 0;
        for start in 0..perm.len() {
            if seen[start] {
                continue;
            }
            cycles += 1;
            let mut i = start;
            while !seen[i] {
                seen[i] = true;
                i = perm[i];
            }
        }
        cycles
    }
}

fn parse_token(token: &str) -> Result<(i32, i32), BraidError> {
    let bad = || BraidError::Token(token.to_string());
    if let Ok(g) = token.parse::<i32>() {
        // `checked_abs`: i32::MIN has no absolute value in i32.
        return match g.checked_abs() {
            Some(k) if k != 0 => Ok((k, g.signum())),
            _ => Err(bad()),
        };
    }
    let rest = token
        .strip_prefix('s')
        .or_else(|| token.strip_prefix('σ'))
        .ok_or_else(bad)?;
    let (index, power) = match rest.split_once('^') {
        Some((index, power)) => (index, power.parse::<i32>().map_err(|_| bad())?),
        None => (rest, 1),
    };
    let k = index.parse::<i32>().map_err(|_| bad())?;
    if k <= 0 || power == 0 {
        return Err(bad());
    }
    Ok((k, power))
}

/// Runs of one generator are written as powers: `s1^3 s2^-1`.
impl fmt::Display for Braid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for run in self.word.chunk_by(|a, b| a == b) {
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            let power = run.len() as i64 * i64::from(run[0].signum());
            match power {
                1 => write!(f, "s{}", run[0].abs())?,
                _ => write!(f, "s{}^{power}", run[0].abs())?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_spelling_to_the_same_word() {
        let expect = [1, -2, 1, 1, 1];
        for text in ["s1 s2^-1 s1^3", "1,-2, 1 1 1", "σ1 σ2^-1 σ1^3"] {
            assert_eq!(
                Braid::parse(None, text).unwrap().word(),
                &expect[..],
                "{text}"
            );
        }
        let b = Braid::parse(None, "s1 s2^-1 s1^3").unwrap();
        assert_eq!(b.strands(), 3);
        assert_eq!(b.to_string(), "s1 s2^-1 s1^3");
        assert_eq!(Braid::parse(Some(5), "s1").unwrap().strands(), 5);
        assert_eq!(Braid::parse(None, "").unwrap().strands(), 1);
    }

    #[test]
    fn refuses_what_is_not_a_braid() {
        for text in ["s0", "x1", "s1^0", "0", "s-1", "s1^x", "-2147483648"] {
            assert!(
                matches!(Braid::parse(None, text), Err(BraidError::Token(_))),
                "{text}"
            );
        }
        assert_eq!(
            Braid::parse(Some(2), "s2"),
            Err(BraidError::Generator {
                generator: 2,
                strands: 2
            })
        );
        assert_eq!(Braid::parse(None, "s9"), Err(BraidError::Strands(10)));
        assert!(matches!(
            Braid::parse(None, "s1^300"),
            Err(BraidError::Crossings(300))
        ));
        assert!(matches!(
            Braid::parse(None, "s1").unwrap().repeat(MAX_CROSSINGS + 1),
            Err(BraidError::Crossings(_))
        ));
    }

    #[test]
    fn components_count_the_cycles_of_the_permutation() {
        let c = |text: &str| Braid::parse(None, text).unwrap().components();
        assert_eq!(c("s1"), 1); // unknot
        assert_eq!(c("s1^2"), 2); // Hopf link
        assert_eq!(c("s1^3"), 1); // trefoil
        assert_eq!(c("s1 s2^-1 s1 s2^-1"), 1); // figure-eight
        assert_eq!(c("s1 s2^-1 s1 s2^-1 s1 s2^-1"), 3); // Borromean rings
        assert_eq!(Braid::parse(Some(4), "").unwrap().components(), 4);
    }

    #[test]
    fn writhe_and_mirror() {
        let b = Braid::parse(None, "s1^3 s2^-1").unwrap();
        assert_eq!(b.writhe(), 2);
        assert_eq!(b.mirror().writhe(), -2);
        assert_eq!(b.mirror().to_string(), "s1^-3 s2");
    }
}
