//! The rate algebra of §8 — every numerator, denominator, null rule and unit.
//!
//! **The six counts are the primary reported output.** Every ratio below is
//! derived from them and is secondary, so a reader who distrusts a ratio can
//! recompute all of them from the counts printed alongside. There is no
//! seventh count and no `L = U`: a case whose truth label cannot be
//! established never enters the corpus at all — it is a typed rejection at
//! generation, recorded rather than silently labelled.

use serde::Serialize;

/// A count-valued ratio, with its null rule stated once and applied everywhere.
///
/// `value` is `Some(numerator / denominator)` **iff** `denominator != 0`, and
/// `None` **iff** `denominator == 0`. There is no other case. A null is never
/// substituted by `0.0`, by `1.0`, or by an omitted key; and when the
/// denominator is non-zero, `0.0` and `1.0` are legal values emitted as such.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Ratio {
    /// unit: count for the five derived rates; **bytes** for `cost_ratio`.
    pub numerator: u64,
    /// unit: count for the five derived rates; **bytes** for `cost_ratio`.
    pub denominator: u64,
    /// dimensionless; `null` iff `denominator == 0`. Never averaged, never
    /// summed, never subtracted — re-aggregate from numerator and denominator.
    pub value: Option<f64>,
}

impl Ratio {
    /// The only constructor. The null rule lives here and nowhere else.
    pub fn new(numerator: u64, denominator: u64) -> Self {
        Self {
            numerator,
            denominator,
            value: if denominator == 0 {
                None
            } else {
                Some(numerator as f64 / denominator as f64)
            },
        }
    }
}

/// The six counts of one cell — `(rule_id, window_bytes, family, evidence_class)`.
///
/// Truth label `L ∈ {T, F}` from the oracle; rule prediction `P ∈ {T, F, U}`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CountVector {
    /// truth `T`, predicted `T` — unit: count
    pub n_tt: u64,
    /// truth `T`, predicted `F` — unit: count
    pub n_tf: u64,
    /// truth `T`, predicted `U` — unit: count
    pub n_tu: u64,
    /// truth `F`, predicted `T` — unit: count
    pub n_ft: u64,
    /// truth `F`, predicted `F` — unit: count
    pub n_ff: u64,
    /// truth `F`, predicted `U` — unit: count
    pub n_fu: u64,
}

impl CountVector {
    /// `N` — every case in the cell. unit: count
    pub fn total(&self) -> u64 {
        self.n_tt + self.n_tf + self.n_tu + self.n_ft + self.n_ff + self.n_fu
    }

    /// Among truth-`F` cases the rule **bound**, the share it called `T` — the
    /// dangerous direction.
    pub fn false_agreement(&self) -> Ratio {
        Ratio::new(self.n_ft, self.n_ft + self.n_ff)
    }

    /// The complement of `false_agreement`, reported explicitly rather than
    /// derived by subtraction.
    pub fn detection(&self) -> Ratio {
        Ratio::new(self.n_ff, self.n_ft + self.n_ff)
    }

    /// Among truth-`T` cases the rule bound, the share it called `F` — the
    /// guardrail direction.
    pub fn false_drift(&self) -> Ratio {
        Ratio::new(self.n_tf, self.n_tf + self.n_tt)
    }

    /// Share of all cases left unbound.
    pub fn unknown_rate(&self) -> Ratio {
        Ratio::new(self.n_tu + self.n_fu, self.total())
    }

    /// Share of all cases bound — **its own numerator**, not `1 − unknown_rate`.
    pub fn coverage(&self) -> Ratio {
        Ratio::new(self.n_tt + self.n_tf + self.n_ft + self.n_ff, self.total())
    }

    /// Sum two cells' counts. This is how ratios re-aggregate: by summing the
    /// **counts** and building a new `Ratio`, never by combining ratios.
    pub fn plus(&self, other: &Self) -> Self {
        Self {
            n_tt: self.n_tt + other.n_tt,
            n_tf: self.n_tf + other.n_tf,
            n_tu: self.n_tu + other.n_tu,
            n_ft: self.n_ft + other.n_ft,
            n_ff: self.n_ff + other.n_ff,
            n_fu: self.n_fu + other.n_fu,
        }
    }
}
