//! The hexavalent **truth-value algebra** (CONTEXT.md: T / P / U / D / F / C),
//! as seen by `ix-ai-annotations`.
//!
//! The algebra itself — what the six values *mean* in relation to each other —
//! now lives on the canonical [`ix_types::Hexavalent`]. This module is a thin
//! **adapter**: it keeps the historical `ix_ai_annotations::truth::{polarity,
//! conflicts, weighted}` surface (consumed by the reconciler's weighted voting
//! and `ix-assumption-graph`'s contradiction derivation) and forwards each call
//! to the one canonical implementation, translating the wire-contract
//! [`TruthValue`](crate::types::TruthValue) at the boundary. There is no second
//! truth table here to drift from the first.

use crate::types::TruthValue;

/// A truth value's evidential direction. Re-exported from the canonical
/// [`ix_types::Polarity`] so there is a single definition.
pub use ix_types::Polarity;

/// Map a hexavalent truth value to its evidential [`Polarity`] (delegates to
/// [`ix_types::Hexavalent::polarity`]).
pub fn polarity(tv: TruthValue) -> Polarity {
    ix_types::Hexavalent::from(tv).polarity()
}

/// Two truth values conflict iff one leans true and the other leans false
/// (delegates to [`ix_types::Hexavalent::conflicts`]).
// @ai:invariant conflicts() is symmetric: conflicts(a,b) == conflicts(b,a) [T:test conf:0.95 src:ix_types::hexavalent_tests::polarity_and_conflicts]
pub fn conflicts(a: TruthValue, b: TruthValue) -> bool {
    ix_types::Hexavalent::from(a).conflicts(ix_types::Hexavalent::from(b))
}

/// Confidence-weighted hexavalent voting over `(truth_value, confidence)` pairs
/// (delegates to [`ix_types::weighted`], translating the wire enum at the
/// boundary). Returns `(argmax_truth_value, avg_confidence)`; ties break by the
/// escalation-favoring order `C > U > F > D > T > P`.
pub fn weighted(votes: &[(TruthValue, f64)]) -> (TruthValue, f64) {
    let hex: Vec<(ix_types::Hexavalent, f64)> =
        votes.iter().map(|&(tv, c)| (tv.into(), c)).collect();
    let (winner, avg) = ix_types::weighted(&hex);
    (winner.into(), avg)
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
