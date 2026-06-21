//! The hexavalent state machine (T/P/U/D/F/C) as a PURE function over the *already
//! decided* signals. This is Candidate 1's core: the anti-Goodhart conjunction —
//! **never an average** — extracted from the I/O of `evaluate` so the full truth table
//! (especially the **C** reward-hack case: metric↑ ∧ guardrail broke) is unit-testable
//! without DuckDB, git, or fixtures.
//!
//! Fail-closed ordering (each step only reached if the prior didn't escalate):
//! 1. untrusted iteration provenance → **U** (forged / dirty / unmatched key),
//! 2. a required-but-dormant lens → **U**,
//! 3. the hard conjunction over (metric_up, guardrail_held):
//!    - either missing → **U**,
//!    - metric↑ ∧ guardrail broke → **C** (reward-hack: reject + alarm),
//!    - otherwise no improvement → **F** (reject),
//!    - metric↑ ∧ guardrail held → the soft lenses may downgrade the accept:
//!      oscillating → **D**, drifting → **P**, else clean → **T**.

/// The decided inputs to the state machine — every field is already resolved to its
/// tri-state (`Option<bool>`), so the function is pure (no I/O, no fixtures).
pub(crate) struct Decided<'a> {
    pub metric_up: Option<bool>,
    pub guardrail_held: Option<bool>,
    pub converging: Option<bool>,
    pub drifting: Option<bool>,
    /// `Some(reason)` if the iteration's correlation key can't be trusted.
    pub provenance_fail: Option<&'static str>,
    /// A required lens (`require_loops`/`require_ood`) is dormant (no signal).
    pub dormant_required: bool,
    /// The guardrail report status string, for the "inconclusive" reason line.
    pub guardrail_report_status: &'a str,
}

/// Fuse the decided signals into `(status, decision, reason)`. The single source of
/// truth for the hexavalent verdict — `evaluate` only marshals inputs into this.
// @ai:invariant fuse returns C (reject) when metric improved while the guardrail regressed — the reward-hack signature is never averaged away [T:test conf:0.95 src:ix_duck::maintain::verdict_fusion::tests::reward_hack_is_contradiction]
pub(crate) fn fuse(d: &Decided) -> (&'static str, &'static str, String) {
    if let Some(why) = d.provenance_fail {
        return (
            "U",
            "escalate",
            format!("iteration provenance untrusted: {why}"),
        );
    }
    if d.dormant_required {
        return (
            "U",
            "escalate",
            "a required lens is dormant (no signal)".to_string(),
        );
    }
    match (d.metric_up, d.guardrail_held) {
        (None, _) => (
            "U",
            "escalate",
            "metric evidence missing — cannot decide".to_string(),
        ),
        (_, None) => (
            "U",
            "escalate",
            format!("guardrail inconclusive ({})", d.guardrail_report_status),
        ),
        (Some(true), Some(false)) => (
            "C",
            "reject",
            "REWARD-HACK: metric improved while a held capability regressed".to_string(),
        ),
        (Some(false), Some(false)) => (
            "F",
            "reject",
            "no improvement and guardrail regressed".to_string(),
        ),
        (Some(false), Some(true)) => ("F", "reject", "no metric improvement".to_string()),
        // Hard conjunction holds — the soft lenses can downgrade an accept.
        (Some(true), Some(true)) => {
            if d.converging == Some(false) {
                (
                    "D",
                    "escalate",
                    "metric improved but the loop is oscillating (not converging)".to_string(),
                )
            } else if d.drifting == Some(true) {
                (
                    "P",
                    "accept",
                    "metric improved and guardrail held, but queries are drifting \
                     out-of-distribution"
                        .to_string(),
                )
            } else {
                (
                    "T",
                    "accept",
                    "metric improved and guardrail held".to_string(),
                )
            }
        }
    }
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;

    /// A clean accept baseline; tweak fields per-case.
    fn base() -> Decided<'static> {
        Decided {
            metric_up: Some(true),
            guardrail_held: Some(true),
            converging: None,
            drifting: None,
            provenance_fail: None,
            dormant_required: false,
            guardrail_report_status: "pass",
        }
    }

    fn status(d: &Decided) -> &'static str {
        fuse(d).0
    }

    #[test]
    fn clean_accept_is_t() {
        assert_eq!(status(&base()), "T");
        assert_eq!(fuse(&base()).1, "accept");
    }

    #[test]
    fn reward_hack_is_contradiction() {
        // The case that matters most: metric↑ while the guardrail broke → C, never averaged.
        let d = Decided { guardrail_held: Some(false), ..base() };
        let (s, decision, reason) = fuse(&d);
        assert_eq!(s, "C");
        assert_eq!(decision, "reject");
        assert!(reason.contains("REWARD-HACK"));
    }

    #[test]
    fn no_improvement_rejects_to_f() {
        // metric flat, guardrail held → F.
        assert_eq!(status(&Decided { metric_up: Some(false), ..base() }), "F");
        // metric flat AND guardrail broke → still F (no reward-hack: no improvement to alarm).
        assert_eq!(
            status(&Decided { metric_up: Some(false), guardrail_held: Some(false), ..base() }),
            "F"
        );
    }

    #[test]
    fn missing_hard_signal_escalates_to_u() {
        assert_eq!(status(&Decided { metric_up: None, ..base() }), "U");
        assert_eq!(status(&Decided { guardrail_held: None, ..base() }), "U");
    }

    #[test]
    fn oscillating_disputes_an_accept_to_d() {
        assert_eq!(status(&Decided { converging: Some(false), ..base() }), "D");
        assert_eq!(fuse(&Decided { converging: Some(false), ..base() }).1, "escalate");
    }

    #[test]
    fn drift_flags_an_accept_to_p() {
        assert_eq!(status(&Decided { drifting: Some(true), ..base() }), "P");
        assert_eq!(fuse(&Decided { drifting: Some(true), ..base() }).1, "accept");
        // Converging + in-distribution explicitly true → still clean T.
        assert_eq!(
            status(&Decided { converging: Some(true), drifting: Some(false), ..base() }),
            "T"
        );
    }

    #[test]
    fn oscillation_dominates_drift() {
        // Both soft lenses negative — oscillation (D, escalate) outranks drift (P, accept).
        assert_eq!(
            status(&Decided { converging: Some(false), drifting: Some(true), ..base() }),
            "D"
        );
    }

    #[test]
    fn provenance_failure_escalates_before_any_lens() {
        // Even a perfect accept is capped at U by untrusted provenance.
        let d = Decided { provenance_fail: Some("forged"), ..base() };
        let (s, decision, reason) = fuse(&d);
        assert_eq!(s, "U");
        assert_eq!(decision, "escalate");
        assert!(reason.contains("forged"));
    }

    #[test]
    fn dormant_required_lens_escalates_after_provenance() {
        assert_eq!(status(&Decided { dormant_required: true, ..base() }), "U");
        // Provenance failure still wins the tie (it is checked first).
        let d = Decided { dormant_required: true, provenance_fail: Some("x"), ..base() };
        assert!(fuse(&d).2.contains("provenance"));
    }
}
