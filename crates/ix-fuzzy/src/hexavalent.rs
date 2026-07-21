//! Hexavalent specialization of [`crate::FuzzyDistribution`].
//!
//! Honors the Demerzel spec for hexavalent-specific behavior:
//!
//! - Tiebreak order for `argmax`: **C > U > D > P > T > F**
//!   (conservative — contradictions and unknowns surface first,
//!   doubt before hope)
//! - Escalation trigger: `C / (T+P+D+F+C) > 0.3` (Contradictory
//!   against *informative* mass — Unknown excluded) forces human
//!   review; all-Unknown input never escalates
//! - Sharpen threshold: argmax mass `> 0.8` collapses to a discrete
//!   `Hexavalent` value
//! - Hexavalent NOT: `T↔F`, `P↔D`, `U`/`C` preserved (distinct from
//!   the generic `1.0 - v` implementation in `ops.rs`)

use ix_types::Hexavalent;

use crate::distribution::FuzzyDistribution;
use crate::error::FuzzyError;

/// Type alias for a fuzzy distribution over [`Hexavalent`] variants.
pub type HexavalentDistribution = FuzzyDistribution<Hexavalent>;

/// Escalation threshold on the Contradictory *share of informative
/// mass* — above this the distribution must be escalated to a human
/// per `fuzzy-membership.md`. See [`escalation_triggered`] for the
/// informative-mass denominator.
pub const ESCALATION_THRESHOLD: f64 = 0.3;

/// Sharpen threshold on the argmax mass — above this the
/// distribution may collapse to a discrete truth value.
pub const SHARPEN_THRESHOLD: f64 = 0.8;

/// `true` iff the Contradictory share of *informative* mass exceeds
/// [`ESCALATION_THRESHOLD`]: `C / (T + P + D + F + C) > 0.3`. Unknown
/// (abstention) is excluded from the denominator, so piling on
/// non-evidence cannot dilute an evidence-based alarm; an all-Unknown
/// distribution has zero informative mass and never escalates.
/// Callers should check this immediately after each
/// AND/OR/accumulate step.
///
/// The informative-mass denominator is the Demerzel spec v1.2 /
/// hari GuitarAlchemist/hari#28 abstention-muting fix; the prior
/// semantics compared raw `C > 0.3`.
pub fn escalation_triggered(dist: &HexavalentDistribution) -> bool {
    let c = dist.get(&Hexavalent::Contradictory);
    let informative = dist.get(&Hexavalent::True)
        + dist.get(&Hexavalent::Probable)
        + dist.get(&Hexavalent::Doubtful)
        + dist.get(&Hexavalent::False)
        + c;
    if informative <= 0.0 {
        return false;
    }
    c / informative > ESCALATION_THRESHOLD
}

/// Hexavalent-specific NOT: swap `T↔F`, swap `P↔D`, leave `U` and
/// `C` unchanged. This is the operation from `fuzzy-membership.md`
/// that `crate::FuzzyDistribution::not_generic` is NOT suitable for
/// — generic `1.0 - v` doesn't respect the per-variant semantics.
///
/// The swap preserves the `sum = 1.0` invariant without renormalization.
pub fn hexavalent_not(dist: &HexavalentDistribution) -> Result<HexavalentDistribution, FuzzyError> {
    let t = dist.get(&Hexavalent::True);
    let p = dist.get(&Hexavalent::Probable);
    let u = dist.get(&Hexavalent::Unknown);
    let d = dist.get(&Hexavalent::Doubtful);
    let f = dist.get(&Hexavalent::False);
    let c = dist.get(&Hexavalent::Contradictory);
    FuzzyDistribution::new(vec![
        (Hexavalent::True, f),
        (Hexavalent::Probable, d),
        (Hexavalent::Unknown, u),
        (Hexavalent::Doubtful, p),
        (Hexavalent::False, t),
        (Hexavalent::Contradictory, c),
    ])
}

/// Hexavalent-aware argmax that honors the spec's tiebreak order
/// `C > U > D > P > T > F`. Use this in preference to the generic
/// [`FuzzyDistribution::argmax`] when working on hexavalent
/// distributions so ties resolve to the conservative variant.
pub fn hexavalent_argmax(dist: &HexavalentDistribution) -> Hexavalent {
    // Enumerate variants in the *tiebreak priority order* so that
    // `max_by` with a stable comparator naturally keeps the winner.
    const PRIORITY: [Hexavalent; 6] = [
        Hexavalent::Contradictory,
        Hexavalent::Unknown,
        Hexavalent::Doubtful,
        Hexavalent::Probable,
        Hexavalent::True,
        Hexavalent::False,
    ];
    let mut best = PRIORITY[0];
    let mut best_mass = dist.get(&best);
    for &v in &PRIORITY[1..] {
        let m = dist.get(&v);
        // Strictly greater — ties keep the earlier (higher-priority)
        // variant because the PRIORITY array is walked in order.
        if m > best_mass {
            best = v;
            best_mass = m;
        }
    }
    best
}

/// Try to collapse the distribution to a discrete [`Hexavalent`]
/// value. Returns `Some(variant)` iff the (tiebreak-aware) argmax
/// mass exceeds [`SHARPEN_THRESHOLD`].
pub fn try_sharpen(dist: &HexavalentDistribution) -> Option<Hexavalent> {
    let winner = hexavalent_argmax(dist);
    if dist.get(&winner) > SHARPEN_THRESHOLD {
        Some(winner)
    } else {
        None
    }
}

/// Build a [`HexavalentDistribution`] from six (T, P, U, D, F, C)
/// masses. Validates invariants. The ordering matches the Demerzel
/// spec and the `hexavalent-state.schema.json` field order.
pub fn hexavalent_from_tpudfc(
    t: f64,
    p: f64,
    u: f64,
    d: f64,
    f: f64,
    c: f64,
) -> Result<HexavalentDistribution, FuzzyError> {
    FuzzyDistribution::new(vec![
        (Hexavalent::True, t),
        (Hexavalent::Probable, p),
        (Hexavalent::Unknown, u),
        (Hexavalent::Doubtful, d),
        (Hexavalent::False, f),
        (Hexavalent::Contradictory, c),
    ])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiebreak_prefers_c_over_everything() {
        // All six variants tied at 1/6 → argmax should be C.
        let d = HexavalentDistribution::uniform(vec![
            Hexavalent::True,
            Hexavalent::Probable,
            Hexavalent::Unknown,
            Hexavalent::Doubtful,
            Hexavalent::False,
            Hexavalent::Contradictory,
        ])
        .unwrap();
        assert_eq!(hexavalent_argmax(&d), Hexavalent::Contradictory);
    }

    #[test]
    fn tiebreak_prefers_u_when_no_c() {
        // 4-variant tie, no C → U wins.
        let d = hexavalent_from_tpudfc(0.25, 0.0, 0.25, 0.25, 0.25, 0.0).unwrap();
        assert_eq!(hexavalent_argmax(&d), Hexavalent::Unknown);
    }

    #[test]
    fn strict_max_beats_priority_order() {
        // T=0.9 should win over a C=0.1 even though C has higher priority.
        let d = hexavalent_from_tpudfc(0.9, 0.0, 0.0, 0.0, 0.0, 0.1).unwrap();
        assert_eq!(hexavalent_argmax(&d), Hexavalent::True);
    }

    #[test]
    fn escalation_triggers_above_threshold() {
        let d = hexavalent_from_tpudfc(0.4, 0.0, 0.0, 0.0, 0.2, 0.4).unwrap();
        assert!(escalation_triggered(&d));
    }

    #[test]
    fn escalation_does_not_trigger_at_exactly_threshold() {
        // No Unknown mass → informative denominator is 1.0, so
        // C/informative = C = 0.3, which is not strictly greater.
        let d = hexavalent_from_tpudfc(0.4, 0.0, 0.0, 0.0, 0.3, 0.3).unwrap();
        assert!(!escalation_triggered(&d));
    }

    #[test]
    fn escalation_survives_unknown_abstention() {
        // T, F, C at equal mass with an arbitrary amount of Unknown
        // padding. Escalation weighs C against informative mass only
        // (C / (T+P+D+F+C) = 1/3 > 0.3), so it fires for ANY amount
        // of Unknown — abstention cannot mute an evidence-based
        // alarm. Under the prior raw `C > 0.3` rule the growing
        // Unknown mass drove raw C below 0.3 and (wrongly) silenced
        // the alarm; this assertion is the FLIP. Demerzel spec v1.2 /
        // hari GuitarAlchemist/hari#28.
        for u in [0.0, 0.3, 0.6, 0.9, 0.99] {
            let each = (1.0 - u) / 3.0;
            let d = hexavalent_from_tpudfc(each, 0.0, u, 0.0, each, each).unwrap();
            assert!(
                escalation_triggered(&d),
                "abstention muted escalation at U={u} (raw C={each})"
            );
        }
    }

    #[test]
    fn all_unknown_does_not_escalate() {
        // Pure abstention: zero informative mass, so no evidence and
        // no alarm. Demerzel spec v1.2 / hari GuitarAlchemist/hari#28.
        let d = hexavalent_from_tpudfc(0.0, 0.0, 1.0, 0.0, 0.0, 0.0).unwrap();
        assert!(!escalation_triggered(&d));
    }

    #[test]
    fn sharpen_collapses_above_threshold() {
        let d = hexavalent_from_tpudfc(0.9, 0.0, 0.05, 0.0, 0.05, 0.0).unwrap();
        assert_eq!(try_sharpen(&d), Some(Hexavalent::True));
    }

    #[test]
    fn sharpen_refuses_below_threshold() {
        let d = hexavalent_from_tpudfc(0.6, 0.0, 0.2, 0.0, 0.2, 0.0).unwrap();
        assert!(try_sharpen(&d).is_none());
    }

    #[test]
    fn hex_not_swaps_t_and_f() {
        let d = hexavalent_from_tpudfc(0.7, 0.0, 0.1, 0.0, 0.1, 0.1).unwrap();
        let n = hexavalent_not(&d).unwrap();
        assert!((n.get(&Hexavalent::True) - 0.1).abs() < 1e-9);
        assert!((n.get(&Hexavalent::False) - 0.7).abs() < 1e-9);
        assert!((n.get(&Hexavalent::Unknown) - 0.1).abs() < 1e-9);
        assert!((n.get(&Hexavalent::Contradictory) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn hex_not_swaps_p_and_d_preserves_u_and_c() {
        let d = hexavalent_from_tpudfc(0.0, 0.3, 0.1, 0.2, 0.0, 0.4).unwrap();
        let n = hexavalent_not(&d).unwrap();
        assert!((n.get(&Hexavalent::Probable) - 0.2).abs() < 1e-9);
        assert!((n.get(&Hexavalent::Doubtful) - 0.3).abs() < 1e-9);
        assert!((n.get(&Hexavalent::Unknown) - 0.1).abs() < 1e-9);
        assert!((n.get(&Hexavalent::Contradictory) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn hex_not_on_pure_u_is_identity() {
        let d = hexavalent_from_tpudfc(0.0, 0.0, 1.0, 0.0, 0.0, 0.0).unwrap();
        let n = hexavalent_not(&d).unwrap();
        assert_eq!(
            serde_json::to_string(&d).unwrap(),
            serde_json::to_string(&n).unwrap()
        );
    }

    #[test]
    fn serde_round_trip_preserves_content() {
        let d = hexavalent_from_tpudfc(0.2, 0.1, 0.3, 0.1, 0.2, 0.1).unwrap();
        let json = serde_json::to_string(&d).unwrap();
        let back: HexavalentDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
    }
}
