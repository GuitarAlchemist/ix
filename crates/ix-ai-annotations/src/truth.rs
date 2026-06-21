//! The hexavalent **truth-value algebra** (CONTEXT.md: T / P / U / D / F / C).
//!
//! This is the one place that encodes what the six truth values *mean* in
//! relation to each other: their evidential [`Polarity`], when two of them
//! [`conflicts`], and how a set of them resolves under confidence-weighted
//! voting ([`weighted`]). It lives beside [`TruthValue`](crate::types::TruthValue)
//! so the semantics travel with the type.
//!
//! Both consumers consult this module rather than re-encoding the order:
//! `ix-ai-annotations`'s reconciler (weighted voting) and `ix-assumption-graph`'s
//! contradiction derivation (polarity conflict). Each keeps its own *grouping*
//! and *output*; only the algebra is shared.

use crate::types::TruthValue;

/// A truth value's evidential direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// Leans true: `T` (True) or `P` (Probable).
    Positive,
    /// Leans false: `D` (Doubtful) or `F` (False).
    Negative,
    /// No evidential direction: `U` (Unknown).
    Neutral,
    /// Already contradictory: `C` — flagged at the source, not re-derived pairwise.
    Contradictory,
}

/// Map a hexavalent truth value to its evidential polarity.
// @ai:invariant every TruthValue maps to exactly one Polarity (total match) [T:formal-proof conf:0.95 src:exhaustive match, compiler-checked]
pub fn polarity(tv: TruthValue) -> Polarity {
    match tv {
        TruthValue::T | TruthValue::P => Polarity::Positive,
        TruthValue::D | TruthValue::F => Polarity::Negative,
        TruthValue::U => Polarity::Neutral,
        TruthValue::C => Polarity::Contradictory,
    }
}

/// Two truth values conflict iff one leans true and the other leans false.
// @ai:invariant conflicts() is symmetric: conflicts(a,b) == conflicts(b,a) [T:test conf:0.95 src:ix-ai-annotations::truth::tests::conflicts_is_symmetric]
pub fn conflicts(a: TruthValue, b: TruthValue) -> bool {
    matches!(
        (polarity(a), polarity(b)),
        (Polarity::Positive, Polarity::Negative) | (Polarity::Negative, Polarity::Positive)
    )
}

/// Confidence-weighted hexavalent voting over `(truth_value, confidence)` pairs.
/// Returns `(argmax_truth_value, avg_confidence)`: the value with the greatest
/// summed confidence, and the mean confidence across all votes.
///
/// Ties break by an **escalation-favoring** order `C > U > F > D > T > P`, so an
/// unresolved or contradictory reading wins over a confident-but-tied positive.
// @ai:invariant weighted() tie-break order is C > U > F > D > T > P (escalation-favoring) [T:test conf:0.9 src:ix-ai-annotations::truth::tests::weighted_tie_breaks_toward_escalation]
pub fn weighted(votes: &[(TruthValue, f64)]) -> (TruthValue, f64) {
    use std::collections::HashMap;
    // No evidence → Unknown. (Unreachable from the reconciler, which only votes
    // over groups of ≥2; guarded so the escalation tie-break can't pick C here.)
    if votes.is_empty() {
        return (TruthValue::U, 0.0);
    }
    let mut buckets: HashMap<TruthValue, f64> = HashMap::new();
    let mut total = 0.0;
    for (tv, conf) in votes {
        *buckets.entry(*tv).or_insert(0.0) += *conf;
        total += *conf;
    }
    let avg = total / votes.len() as f64;
    let order = [
        TruthValue::C,
        TruthValue::U,
        TruthValue::F,
        TruthValue::D,
        TruthValue::T,
        TruthValue::P,
    ];
    let max_weight = buckets.values().copied().fold(0.0_f64, f64::max);
    let winner = order
        .into_iter()
        .find(|tv| buckets.get(tv).copied().unwrap_or(0.0) == max_weight)
        .unwrap_or(TruthValue::U);
    (winner, avg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflicts_is_symmetric() {
        for &a in &[
            TruthValue::T,
            TruthValue::P,
            TruthValue::U,
            TruthValue::D,
            TruthValue::F,
            TruthValue::C,
        ] {
            for &b in &[
                TruthValue::T,
                TruthValue::P,
                TruthValue::U,
                TruthValue::D,
                TruthValue::F,
                TruthValue::C,
            ] {
                assert_eq!(conflicts(a, b), conflicts(b, a), "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn conflicts_is_polarity_opposition() {
        assert!(conflicts(TruthValue::T, TruthValue::F));
        assert!(conflicts(TruthValue::P, TruthValue::D));
        // Same polarity → no conflict.
        assert!(!conflicts(TruthValue::T, TruthValue::P));
        assert!(!conflicts(TruthValue::F, TruthValue::D));
        // Neutral / already-contradictory never pairwise-conflict.
        assert!(!conflicts(TruthValue::U, TruthValue::F));
        assert!(!conflicts(TruthValue::C, TruthValue::T));
    }

    #[test]
    fn weighted_picks_highest_summed_confidence() {
        let (tv, avg) = weighted(&[
            (TruthValue::T, 0.9),
            (TruthValue::T, 0.8),
            (TruthValue::F, 0.5),
        ]);
        assert_eq!(tv, TruthValue::T); // 1.7 > 0.5
        assert!((avg - (2.2 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn weighted_tie_breaks_toward_escalation() {
        // T and F tie on weight → escalation order prefers F over T.
        let (tv, _) = weighted(&[(TruthValue::T, 0.5), (TruthValue::F, 0.5)]);
        assert_eq!(tv, TruthValue::F);
        // C beats everything on a tie.
        let (tv, _) = weighted(&[(TruthValue::C, 0.5), (TruthValue::U, 0.5)]);
        assert_eq!(tv, TruthValue::C);
    }

    #[test]
    fn weighted_empty_is_unknown() {
        assert_eq!(weighted(&[]), (TruthValue::U, 0.0));
    }
}
