//! Subjective-logic opinion (Jøsang) — the per-claim certainty carrier — and
//! the bridge between `ix-ai-annotations`' [`TruthValue`] and the canonical
//! [`ix_types::Hexavalent`].

use ix_ai_annotations::TruthValue;
use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};

/// A binomial subjective-logic opinion: belief / disbelief / uncertainty / base rate.
///
/// Invariant: `b + d + u == 1` (within tolerance). Projected probability is
/// `E = b + a·u` ([`Opinion::projected`]). See the design doc §5 and contract §4.
///
/// `C` (Contradictory) lies OUTSIDE this simplex — it needs both high belief
/// and high disbelief, which `b + d + u = 1` cannot express — so
/// [`Opinion::from_truth`] returns `None` for `C`. Contradiction is represented
/// by *fusion* of conflicting opinions (see [`crate::fusion`]), not by a single
/// point opinion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Opinion {
    /// Belief mass.
    pub b: f64,
    /// Disbelief mass.
    pub d: f64,
    /// Uncertainty mass.
    pub u: f64,
    /// Base rate (prior).
    pub a: f64,
}

/// Default base rate (prior) when none is known — maximally non-committal.
pub const DEFAULT_BASE_RATE: f64 = 0.5;

impl Opinion {
    /// Projected probability `E = b + a·u`. The scalar used for ranking and
    /// promotion thresholds.
    pub fn projected(&self) -> f64 {
        self.b + self.a * self.u
    }

    /// Build the certainty carrier from a hexavalent truth value + confidence.
    ///
    /// Direction (belief vs disbelief) comes from the truth value's polarity;
    /// magnitude is `confidence × verification_weight`, where verified values
    /// (T/F) commit fully and leaning values (P/D) commit partially — so `P`
    /// carries more residual uncertainty than `T` at equal confidence (the
    /// verified-vs-probable distinction the `certainty` field also records). `U`
    /// is total uncertainty. `C` is not representable and returns `None`.
    ///
    // @ai:invariant b+d+u == 1 for every non-C truth value [T:test conf:0.95 src:opinion.rs::tests::masses_sum_to_one]
    pub fn from_truth(tv: TruthValue, confidence: f64) -> Option<Self> {
        let c = confidence.clamp(0.0, 1.0);
        // (directed magnitude, sign): +1 belief side, -1 disbelief side, 0 vacuous.
        let (directed, sign) = match tv {
            TruthValue::T => (c, 1.0),
            TruthValue::P => (c * 0.6, 1.0),
            TruthValue::D => (c * 0.6, -1.0),
            TruthValue::F => (c, -1.0),
            TruthValue::U => (0.0, 0.0),
            TruthValue::C => return None,
        };
        let (b, d) = match sign {
            s if s > 0.0 => (directed, 0.0),
            s if s < 0.0 => (0.0, directed),
            _ => (0.0, 0.0),
        };
        Some(Opinion {
            b,
            d,
            u: 1.0 - b - d,
            a: DEFAULT_BASE_RATE,
        })
    }
}

/// Bridge: `ix-ai-annotations` [`TruthValue`] → canonical [`ix_types::Hexavalent`].
pub fn to_hexavalent(tv: TruthValue) -> Hexavalent {
    match tv {
        TruthValue::T => Hexavalent::True,
        TruthValue::P => Hexavalent::Probable,
        TruthValue::U => Hexavalent::Unknown,
        TruthValue::D => Hexavalent::Doubtful,
        TruthValue::F => Hexavalent::False,
        TruthValue::C => Hexavalent::Contradictory,
    }
}

/// Bridge: canonical [`ix_types::Hexavalent`] → `ix-ai-annotations` [`TruthValue`].
pub fn from_hexavalent(h: Hexavalent) -> TruthValue {
    match h {
        Hexavalent::True => TruthValue::T,
        Hexavalent::Probable => TruthValue::P,
        Hexavalent::Unknown => TruthValue::U,
        Hexavalent::Doubtful => TruthValue::D,
        Hexavalent::False => TruthValue::F,
        Hexavalent::Contradictory => TruthValue::C,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(x: f64, y: f64) -> bool {
        (x - y).abs() < 1e-9
    }

    #[test]
    fn true_is_high_belief_low_uncertainty() {
        let o = Opinion::from_truth(TruthValue::T, 0.9).unwrap();
        assert!(approx(o.b, 0.9) && approx(o.d, 0.0) && approx(o.u, 0.1));
        assert!(approx(o.projected(), 0.95));
    }

    #[test]
    fn false_is_high_disbelief_low_projection() {
        let o = Opinion::from_truth(TruthValue::F, 0.9).unwrap();
        assert!(approx(o.b, 0.0) && approx(o.d, 0.9) && approx(o.u, 0.1));
        assert!(approx(o.projected(), 0.05));
    }

    #[test]
    fn probable_carries_more_uncertainty_than_true() {
        let t = Opinion::from_truth(TruthValue::T, 0.9).unwrap();
        let p = Opinion::from_truth(TruthValue::P, 0.9).unwrap();
        assert!(p.b < t.b, "P should commit less belief than T");
        assert!(p.u > t.u, "P should retain more uncertainty than T");
        assert!(p.projected() > 0.5, "P still leans true");
    }

    #[test]
    fn unknown_is_total_uncertainty_at_base_rate() {
        let o = Opinion::from_truth(TruthValue::U, 0.9).unwrap();
        assert!(approx(o.b, 0.0) && approx(o.d, 0.0) && approx(o.u, 1.0));
        assert!(approx(o.projected(), DEFAULT_BASE_RATE));
    }

    #[test]
    fn contradictory_is_outside_the_simplex() {
        assert!(Opinion::from_truth(TruthValue::C, 0.9).is_none());
    }

    #[test]
    fn masses_sum_to_one() {
        use TruthValue::*;
        for &tv in &[T, P, U, D, F] {
            for &conf in &[0.0, 0.3, 0.5, 0.7, 1.0] {
                let o = Opinion::from_truth(tv, conf).unwrap();
                assert!(approx(o.b + o.d + o.u, 1.0), "sum!=1 for {tv:?}@{conf}");
                assert!(o.b >= 0.0 && o.d >= 0.0 && o.u >= 0.0);
            }
        }
    }

    #[test]
    fn hexavalent_bridge_round_trips() {
        use TruthValue::*;
        for &tv in &[T, P, U, D, F, C] {
            assert_eq!(from_hexavalent(to_hexavalent(tv)), tv);
        }
    }
}
