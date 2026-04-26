//! [`FuzzyBuilder`] and the four evidence-combiner strategies from
//! the Demerzel `fuzzy { }` computation expression spec.
//!
//! Strategies (per `fuzzy-membership.md`):
//!
//! | Strategy       | Combiner                               | Use case                       |
//! |----------------|----------------------------------------|--------------------------------|
//! | Multiplicative | `a * b`                                | default, independent evidence  |
//! | Zadeh          | `min(a, b)`                            | classical, conservative        |
//! | Bayesian       | `a*b / (a*b + (1-a)*(1-b))`            | posterior evidence update      |
//! | Custom(fn)     | user-defined                           | domain-specific                |
//!
//! The builder starts from a seed distribution, folds evidence
//! contributions into it using the chosen combiner, then produces a
//! final [`crate::FuzzyDistribution`] on demand.

use crate::distribution::FuzzyDistribution;
use crate::error::FuzzyError;

/// Evidence combination strategy.
///
/// `Custom` boxes a `fn(f64, f64) -> f64` so the builder can be
/// stored in a field without generic plumbing. The closure form
/// trades a little indirection for a much simpler call site — the
/// builder is used once per evidence accumulation and the cost is
/// negligible.
pub enum Combiner {
    /// `a * b` — default, treats evidence as independent.
    Multiplicative,
    /// `min(a, b)` — Zadeh's classical fuzzy AND.
    Zadeh,
    /// `a*b / (a*b + (1-a)*(1-b))` — Bayesian posterior update.
    Bayesian,
    /// User-supplied combiner.
    Custom(fn(f64, f64) -> f64),
}

impl Combiner {
    /// Apply this combiner to two membership values. Clamps the
    /// inputs into `[0.0, 1.0]` so stray values from custom
    /// combiners don't poison the downstream renormalization.
    pub fn combine(&self, a: f64, b: f64) -> f64 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        let out = match self {
            Combiner::Multiplicative => a * b,
            Combiner::Zadeh => a.min(b),
            Combiner::Bayesian => {
                let num = a * b;
                let den = num + (1.0 - a) * (1.0 - b);
                if den == 0.0 {
                    0.0
                } else {
                    num / den
                }
            }
            Combiner::Custom(f) => f(a, b),
        };
        out.clamp(0.0, 1.0)
    }
}

/// Fluent builder for accumulating evidence into a
/// [`FuzzyDistribution`].
///
/// Example — Zadeh combination of two pieces of evidence:
///
/// ```
/// use ix_fuzzy::{Combiner, FuzzyBuilder, FuzzyDistribution};
///
/// let seed = FuzzyDistribution::uniform(vec!["T", "F", "U"]).unwrap();
/// let e1   = FuzzyDistribution::new(vec![("T", 0.7), ("F", 0.2), ("U", 0.1)]).unwrap();
/// let e2   = FuzzyDistribution::new(vec![("T", 0.6), ("F", 0.3), ("U", 0.1)]).unwrap();
/// let out  = FuzzyBuilder::new(seed, Combiner::Zadeh)
///     .accumulate(&e1)
///     .accumulate(&e2)
///     .finalize()
///     .unwrap();
/// // "T" ends up with the highest membership after two agreeing pieces of
/// // evidence:
/// assert_eq!(*out.argmax(), "T");
/// ```
pub struct FuzzyBuilder<T: Ord + Clone> {
    state: FuzzyDistribution<T>,
    combiner: Combiner,
    /// Set to `Some(err)` when a fold step would produce an
    /// unrecoverable state (e.g., every combined mass collapsed to
    /// zero). The error is propagated at `finalize` time so the
    /// fluent style is preserved.
    deferred_error: Option<FuzzyError>,
}

impl<T: Ord + Clone> FuzzyBuilder<T> {
    /// Start a new builder from a seed distribution.
    pub fn new(seed: FuzzyDistribution<T>, combiner: Combiner) -> Self {
        Self {
            state: seed,
            combiner,
            deferred_error: None,
        }
    }

    /// Fold one evidence distribution into the current state using
    /// the configured combiner, then renormalize. If a previous
    /// step set `deferred_error`, this call is a no-op.
    pub fn accumulate(mut self, evidence: &FuzzyDistribution<T>) -> Self {
        if self.deferred_error.is_some() {
            return self;
        }
        let keys: std::collections::BTreeSet<T> = self
            .state
            .memberships()
            .keys()
            .chain(evidence.memberships().keys())
            .cloned()
            .collect();
        let mut combined: Vec<(T, f64)> = Vec::with_capacity(keys.len());
        for k in keys {
            let a = self.state.get(&k);
            let b = evidence.get(&k);
            combined.push((k, self.combiner.combine(a, b)));
        }
        let sum: f64 = combined.iter().map(|(_, v)| *v).sum();
        if sum <= 0.0 {
            self.deferred_error = Some(FuzzyError::BadSum { sum });
            return self;
        }
        let normalized: Vec<(T, f64)> = combined.into_iter().map(|(k, v)| (k, v / sum)).collect();
        match FuzzyDistribution::new(normalized) {
            Ok(next) => self.state = next,
            Err(e) => self.deferred_error = Some(e),
        }
        self
    }

    /// Consume the builder and return the accumulated
    /// distribution, or propagate any deferred error.
    pub fn finalize(self) -> Result<FuzzyDistribution<T>, FuzzyError> {
        match self.deferred_error {
            Some(e) => Err(e),
            None => Ok(self.state),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combiner_multiplicative() {
        assert!((Combiner::Multiplicative.combine(0.5, 0.4) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn combiner_zadeh_is_min() {
        assert!((Combiner::Zadeh.combine(0.5, 0.4) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn combiner_bayesian_formula() {
        // 0.5 * 0.4 / (0.5*0.4 + 0.5*0.6) = 0.2 / (0.2 + 0.3) = 0.4
        assert!((Combiner::Bayesian.combine(0.5, 0.4) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn combiner_custom_respects_clamp() {
        let c = Combiner::Custom(|a, b| a + b);
        // Custom combiner may overshoot; builder clamps on output.
        assert_eq!(c.combine(0.7, 0.8), 1.0);
    }

    #[test]
    fn builder_chains_multiplicative_accumulation() {
        let seed = FuzzyDistribution::uniform(vec!["a", "b"]).unwrap();
        let ev1 = FuzzyDistribution::new(vec![("a", 0.8), ("b", 0.2)]).unwrap();
        let ev2 = FuzzyDistribution::new(vec![("a", 0.9), ("b", 0.1)]).unwrap();
        let out = FuzzyBuilder::new(seed, Combiner::Multiplicative)
            .accumulate(&ev1)
            .accumulate(&ev2)
            .finalize()
            .unwrap();
        assert_eq!(*out.argmax(), "a");
        assert!(out.get(&"a") > 0.9, "{}", out.get(&"a"));
    }

    #[test]
    fn builder_propagates_deferred_error() {
        let seed = FuzzyDistribution::pure("a", vec!["a", "b"]).unwrap();
        let ev = FuzzyDistribution::pure("b", vec!["a", "b"]).unwrap();
        // Multiplicative: 1.0 * 0.0 = 0; 0.0 * 1.0 = 0; sum = 0.
        let err = FuzzyBuilder::new(seed, Combiner::Multiplicative)
            .accumulate(&ev)
            .finalize()
            .unwrap_err();
        assert!(matches!(err, FuzzyError::BadSum { .. }));
    }

    #[test]
    fn builder_is_no_op_after_error() {
        let seed = FuzzyDistribution::pure("a", vec!["a", "b"]).unwrap();
        let bad = FuzzyDistribution::pure("b", vec!["a", "b"]).unwrap();
        let ok = FuzzyDistribution::uniform(vec!["a", "b"]).unwrap();
        let err = FuzzyBuilder::new(seed, Combiner::Multiplicative)
            .accumulate(&bad)
            .accumulate(&ok)
            .finalize()
            .unwrap_err();
        assert!(matches!(err, FuzzyError::BadSum { .. }));
    }
}
