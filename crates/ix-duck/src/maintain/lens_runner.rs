//! Lens runner — the I/O side of the gate. Threads the DuckDB `conn` through the four
//! lenses (metric / guardrail / convergence / drift) and the iteration-provenance check,
//! resolving each to its tri-state ([`Option<bool>`]) and assembling the per-signal
//! report. Behaviour is IDENTICAL to the original inline orchestration: the loop ledger
//! is built once (shared by the key-match and the convergence lens), a dormant lens is
//! `None` (never a positive signal), and the provenance signal is only pushed when scoped.
//!
//! [`run`] returns a [`LensResults`] of the *decided* signals; `evaluate` then feeds them
//! to the pure [`super::verdict_fusion::fuse`] and [`super::evidence::build_evidence`].

use std::path::Path;

use duckdb::Connection;

use super::provenance::{key_matched, provenance_failure, verify_commit};
use super::{MaintainConfig, MaintainError, MaintainInputs, Signal};
use crate::chatbot::{self, GateReport};

/// The decided outcome of running every lens over one iteration — the hand-off from the
/// I/O side ([`run`]) to the pure verdict fusion + evidence build.
pub(crate) struct LensResults {
    /// Externally-derived yield delta (`None` = no metric evidence).
    pub delta: Option<f64>,
    /// `Some(true)` = metric improved past the epsilon; `None` = no evidence.
    pub metric_up: Option<bool>,
    /// The guardrail gate report (carries status + baseline ref for evidence).
    pub report: GateReport,
    /// `Some(true)` held, `Some(false)` regressed, `None` inconclusive (caps at U).
    pub guardrail_held: Option<bool>,
    /// `Some(true)` converging, `Some(false)` oscillating, `None` not consulted/dormant.
    pub converging: Option<bool>,
    /// `Some(true)` drifting OOD, `Some(false)` in-distribution, `None` not consulted.
    pub drifting: Option<bool>,
    /// `Some(reason)` if the iteration's correlation key can't be trusted.
    pub provenance_fail: Option<&'static str>,
    /// The per-lens signal report, in emission order.
    pub signals: Vec<Signal>,
}

/// Yield delta from an externally-derived `hits.jsonl`: mean `coverage_max` over
/// `compiled` rows in the later half minus the earlier half, split at the median
/// `ts_ms`. `None` if the file is absent/empty or a half has no compiled rows
/// (split by ts, NOT the cumulative mean — the blended mean hides pre/post).
fn metric_delta(conn: &Connection, hits_path: &Path) -> Result<Option<f64>, MaintainError> {
    if !hits_path.exists() {
        return Ok(None);
    }
    let p = hits_path
        .to_string_lossy()
        .replace('\\', "/")
        .replace('\'', "''");
    let (before, after): (Option<f64>, Option<f64>) = conn.query_row(
        &format!(
            "WITH rows AS (SELECT ts_ms, outcome, coverage_max FROM read_json_auto('{p}')),
                  m AS (SELECT median(ts_ms) AS mid FROM rows)
             SELECT avg(coverage_max) FILTER (WHERE outcome='compiled' AND ts_ms <  (SELECT mid FROM m)),
                    avg(coverage_max) FILTER (WHERE outcome='compiled' AND ts_ms >= (SELECT mid FROM m))
             FROM rows"
        ),
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    Ok(match (before, after) {
        (Some(b), Some(a)) => Some(a - b),
        _ => None,
    })
}

/// Run every lens over one iteration, returning the decided signals + per-signal report.
/// Pure-by-handoff: no verdict decision here — that lives in [`super::verdict_fusion`].
pub(crate) fn run(
    conn: &Connection,
    cfg: &MaintainConfig,
    inputs: &MaintainInputs,
) -> Result<LensResults, MaintainError> {
    // --- metric lens (externally derived) ---
    let delta = metric_delta(conn, inputs.hits_path)?;
    let metric_up = delta.map(|d| d >= cfg.min_metric_delta);

    // --- guardrail lens ---
    let report = chatbot::check_regressions(conn, inputs.corpus_dir)?;
    // pass → held; regression → broke; degraded/skipped → no signal (caps at U).
    let guardrail_held = match report.status.as_str() {
        "pass" => Some(true),
        "regression" => Some(false),
        _ => None,
    };

    let mut signals = vec![
        Signal {
            lens: "metric".into(),
            ok: metric_up,
            detail: match delta {
                Some(d) => format!("yield delta {d:+.4}"),
                None => "no externally-derived metric evidence".into(),
            },
        },
        Signal {
            lens: "guardrail".into(),
            ok: guardrail_held,
            detail: format!(
                "chatbot gate: {} ({} regression(s))",
                report.status,
                report.regressions.len()
            ),
        },
    ];

    // --- iteration provenance (Phase 3a): verify the correlation key against git ---
    let trust = inputs
        .iteration
        .as_ref()
        .map(|s| verify_commit(s.repo_dir, s.commit_sha));

    // Build the loop ledger once — both the key match and the convergence lens use it.
    let loops_built = match inputs.loops_dir {
        Some(dir) => {
            crate::loops::build_loop_iterations(conn, dir)?;
            true
        }
        None => false,
    };

    // Correlation: a scope must match a REAL recorded row for (loop_id, commit_sha).
    // Verifying the commit exists in git is NOT enough — a clean but unrelated commit
    // for a loop with earlier improving rows would otherwise be scored as that loop and
    // wrongly return T (Codex P1).
    let key = key_matched(conn, inputs.iteration.as_ref(), loops_built)?;

    // --- convergence lens (loops) + drift lens (ood) ---
    // Phase 3a: convergence is scoped to the iteration's `loop_id` when a scope is
    // given (Phase 1 aggregated the whole ledger). Drift scoping awaits a query→loop
    // tag in Contract B — until then drift stays corpus-level advisory (documented).
    //
    // A supplied-but-empty/dormant lens is *unknown* (None), NOT a positive signal:
    // reading "no data" as converging / in-distribution would be a green light from
    // no evidence — the opposite of fail-closed (and, when required, must escalate).
    let converging = if loops_built {
        let summaries = crate::loops::loop_summary(conn)?; // excludes seed/test rows
        match &inputs.iteration {
            Some(scope) => {
                if summaries.iter().any(|s| s.loop_id == scope.loop_id) {
                    // This loop is converging iff it is not in the oscillating set.
                    let osc = crate::loops::oscillating_loops(conn)?;
                    Some(!osc.iter().any(|(id, _, _)| id == scope.loop_id))
                } else {
                    None // no real rows for this loop yet
                }
            }
            None if summaries.is_empty() => None,
            None => Some(crate::loops::oscillating_loops(conn)?.is_empty()),
        }
    } else {
        None
    };
    let drifting = match inputs.query_embeddings_dir {
        Some(dir) => {
            // Need ≥2 queries to score nearest-neighbour density; fewer → unknown.
            if crate::ood::build_query_embeddings(conn, dir)? < 2 {
                None
            } else {
                Some(!crate::ood::flag_ood(conn, cfg.ood_k, cfg.ood_threshold)?.is_empty())
            }
        }
        None => None,
    };
    signals.push(Signal {
        lens: "convergence".into(),
        ok: converging,
        detail: match converging {
            Some(true) => "loops converging (no oscillation)".into(),
            Some(false) => "loops OSCILLATING — thrash, not convergence".into(),
            None if cfg.require_loops => "required but loop ledger dormant".into(),
            None => "no loop data (advisory)".into(),
        },
    });
    signals.push(Signal {
        lens: "drift".into(),
        ok: drifting.map(|d| !d),
        detail: match drifting {
            Some(true) => "queries DRIFTING out-of-distribution".into(),
            Some(false) => "queries in-distribution".into(),
            None if cfg.require_ood => "required but embedding sink dormant".into(),
            None => "no query embeddings (advisory)".into(),
        },
    });

    // Provenance is trusted only if the commit is real + clean of tracked WIP AND a
    // recorded loop row exists for (loop_id, commit_sha) — see `provenance_failure`.
    let provenance_fail = provenance_failure(inputs.iteration.as_ref(), trust, key);
    if inputs.iteration.is_some() {
        signals.push(Signal {
            lens: "provenance".into(),
            ok: Some(provenance_fail.is_none()),
            detail: provenance_fail
                .unwrap_or("iteration commit verified, no tracked WIP, loop row matched")
                .to_string(),
        });
    }

    Ok(LensResults {
        delta,
        metric_up,
        report,
        guardrail_held,
        converging,
        drifting,
        provenance_fail,
        signals,
    })
}
