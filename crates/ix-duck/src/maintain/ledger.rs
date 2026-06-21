//! The verdict ledger (Candidate 4) — the append-only **write side** and the
//! convergence-trend **read side** of the maintain gate, plus the GA-envelope snapshot
//! materialization. [`VerdictLedger`] owns a ledger path and offers
//! [`append`](VerdictLedger::append) (one JSON line per verdict),
//! [`summarize`](VerdictLedger::summarize) (aggregate into a [`TrendSummary`]), and
//! [`materialize`](VerdictLedger::materialize) (write the current-verdict scorecard
//! snapshot atomically).
//!
//! The free functions [`append_to_ledger`] / [`maintain_trend`] / [`build_snapshot`] /
//! [`write_snapshot_atomic`] remain as the stable public surface (re-exported from
//! `maintain`); the struct is the deeper, fluent API over the same primitives.

use std::path::Path;

use duckdb::Connection;
use serde::Serialize;

use super::{MaintainError, MaintainVerdict};

/// A convergence summary over the verdict ledger — the **read side** of the maintain-gate
/// "trend table" (the same `state/thinking-machine/maintain-gate.jsonl` the gate appends to,
/// schema `maintain-gate.contract.md`). `ix-duck` reads it via `read_json_auto`, so a
/// Demerzel / IXQL convergence loop can ask "are we converging?" in one query rather than
/// re-deriving it from raw verdicts.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrendSummary {
    pub total: i64,
    pub accepts: i64,
    pub rejects: i64,
    pub escalates: i64,
    /// `status == "C"`: metric improved while a held capability regressed (reward-hack).
    pub reward_hacks: i64,
    /// `(status, count)` descending — the hexavalent verdict distribution.
    pub by_status: Vec<(String, i64)>,
    /// The most recent verdict's status (by `run_at`), if any.
    pub latest_status: Option<String>,
}

/// One lens verdict, flattened for the scorecard snapshot — the tri-state `ok` of a
/// [`Signal`](super::Signal) rendered as a coarse `"ok" | "bad" | "unknown"` string so a
/// dashboard reads it without re-encoding `Option<bool>`.
#[derive(Debug, Clone, Serialize)]
pub struct SnapshotSignal {
    pub lens: String,
    pub status: String,
    pub detail: String,
}

/// A **current-verdict** snapshot in GA's canonical dashboard-envelope shape
/// (`ga/docs/contracts/quality-snapshot.schema.json`): the GA-required envelope fields
/// (`domain`/`emitted_at`/`metric_name`/`metric_value`/`oracle_status`/`summary`) PLUS the
/// maintain-specific fused verdict (`advisory`/`status`/`decision`/`signals`/`maintain_trend`),
/// carried as additive pass-through fields the envelope permits. Holds the latest hexavalent
/// verdict plus a [`maintain_trend`] rollup, so a dashboard reads ONE file instead of
/// re-deriving the verdict from the append-only ledger. `emitted_at` is the freshness stamp (a
/// stale snapshot must never read green); `oracle_status` is the coarse traffic-light, `status`
/// the richer hexavalent verdict.
///
/// Phase A writes this to the IX tree (**formats-not-coupling** — no GA-tree write here);
/// Phase B federates it into `ga/state/quality/maintain-gate`. **Advisory until Phase 3b**.
#[derive(Debug, Clone, Serialize)]
pub struct VerdictSnapshot {
    pub schema_version: String,
    /// Producer slug — the GA envelope `domain` (lower-kebab). Matches the registry domain + dir.
    pub domain: String,
    /// Freshness — RFC3339 (mirrors `run_at`). The GA envelope's `emitted_at`; consumers compare
    /// it so stale never reads green.
    pub emitted_at: String,
    /// Headline metric identifier (GA envelope `metric_name`).
    pub metric_name: String,
    /// Headline metric value (GA envelope) — the externally-derived yield delta (`0.0` when no
    /// metric evidence; the `signals`/`oracle_status` carry the "no evidence" nuance).
    pub metric_value: f64,
    /// Traffic-light state: `ok | warn | error`, mapped from the hexavalent verdict (GA envelope).
    pub oracle_status: String,
    /// One-line human summary (GA envelope).
    pub summary: String,
    /// Non-binding marker — `true` until IX Phase-3b makes the verdict gating (maintain-specific).
    pub advisory: bool,
    /// The raw hexavalent verdict (T/P/U/D/F/C) — richer than `oracle_status` (maintain-specific).
    pub status: String,
    /// `accept | reject | escalate` (maintain-specific).
    pub decision: String,
    /// Per-signal lens verdicts — locked sub-signal keys (metric / guardrail / convergence /
    /// drift / provenance). The GA-readable cross-signal contract surface.
    pub signals: Vec<SnapshotSignal>,
    /// Convergence rollup over the append-only verdict ledger (reuses [`maintain_trend`]).
    pub maintain_trend: TrendSummary,
}

/// Append the verdict as one JSON line to a tamper-evident, append-only ledger.
pub fn append_to_ledger(verdict: &MaintainVerdict, path: &Path) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let line = serde_json::to_string(verdict).map_err(std::io::Error::other)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")
}

/// Aggregate the append-only verdict ledger into a [`TrendSummary`] (the convergence-trend
/// read side of opportunity #3 in `docs/adr/0001-…`). An absent ledger → an empty summary
/// (degrade, never error) — the convergence loop reads "no data yet".
// @ai:invariant maintain_trend aggregates the verdict ledger by decision/status and reads an absent ledger as an empty summary rather than erroring [T:test conf:0.9 src:ix_duck::maintain::tests::maintain_trend_aggregates_the_ledger]
pub fn maintain_trend(conn: &Connection, ledger_path: &Path) -> Result<TrendSummary, MaintainError> {
    if !ledger_path.exists() {
        return Ok(TrendSummary::default());
    }
    let p = ledger_path
        .to_string_lossy()
        .replace('\\', "/")
        .replace('\'', "''");
    let src = format!("read_json_auto('{p}', union_by_name=true)");
    let (total, accepts, rejects, escalates, reward_hacks): (i64, i64, i64, i64, i64) = conn
        .query_row(
            &format!(
                "SELECT count(*),
                        count(*) FILTER (WHERE decision = 'accept'),
                        count(*) FILTER (WHERE decision = 'reject'),
                        count(*) FILTER (WHERE decision = 'escalate'),
                        count(*) FILTER (WHERE status = 'C')
                 FROM {src}"
            ),
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )?;
    let by_status: Vec<(String, i64)> = {
        let mut stmt = conn.prepare(&format!(
            "SELECT status, count(*) FROM {src} GROUP BY status ORDER BY count(*) DESC, status"
        ))?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<duckdb::Result<_>>()?;
        rows
    };
    let latest_status: Option<String> = conn
        .query_row(
            &format!("SELECT status FROM {src} ORDER BY run_at DESC LIMIT 1"),
            [],
            |r| r.get(0),
        )
        .ok();
    Ok(TrendSummary {
        total,
        accepts,
        rejects,
        escalates,
        reward_hacks,
        by_status,
        latest_status,
    })
}

/// Map the hexavalent verdict to the scorecard's coarse oracle status: accept→`ok`,
/// accept-with-flags / escalate→`warn`, reject→`error`.
fn oracle_status(hex: &str) -> &'static str {
    match hex {
        "T" => "ok",
        "P" | "U" | "D" => "warn",
        "F" | "C" => "error",
        _ => "warn",
    }
}

/// Build a GA-envelope-shaped [`VerdictSnapshot`] from a fused verdict + a ledger trend rollup.
/// Pure (no I/O): the caller supplies the trend (via [`maintain_trend`]) and writes the
/// result with [`write_snapshot_atomic`].
// @ai:invariant build_snapshot emits GA's canonical dashboard envelope (domain + emitted_at + metric_name + metric_value + oracle_status + summary) plus the maintain verdict (advisory + hexavalent status + per-signal + maintain_trend) [T:test conf:0.9 src:ix_duck::maintain::tests::snapshot_is_scorecard_shaped]
pub fn build_snapshot(verdict: &MaintainVerdict, trend: &TrendSummary) -> VerdictSnapshot {
    let signals = verdict
        .signals
        .iter()
        .map(|s| SnapshotSignal {
            lens: s.lens.clone(),
            status: match s.ok {
                Some(true) => "ok",
                Some(false) => "bad",
                None => "unknown",
            }
            .to_string(),
            detail: s.detail.clone(),
        })
        .collect();
    VerdictSnapshot {
        schema_version: "maintain-verdict-snapshot.v0.1".into(),
        domain: "maintain-gate".into(),
        emitted_at: verdict.run_at.clone(),
        metric_name: "maintain_yield_delta".into(),
        metric_value: verdict.metric_delta.unwrap_or(0.0),
        oracle_status: oracle_status(&verdict.status).into(),
        summary: verdict.reason.clone(),
        advisory: true,
        status: verdict.status.clone(),
        decision: verdict.decision.clone(),
        signals,
        maintain_trend: trend.clone(),
    }
}

/// Write the snapshot **atomically** (temp file in the same dir + rename) so a concurrent
/// reader never sees a half-written scorecard. `rename` is atomic on the same filesystem and
/// replaces an existing target on both Unix and Windows.
pub fn write_snapshot_atomic(snapshot: &VerdictSnapshot, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(snapshot).map_err(std::io::Error::other)?;
    // Temp lives beside the target so the rename stays on one filesystem (atomic).
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes())?;
    std::fs::rename(&tmp, path)
}

/// An append-only verdict ledger at a fixed path — the deeper, fluent API over the same
/// primitives (`append` / `summarize` / `materialize`) the free functions expose.
pub struct VerdictLedger<'a> {
    path: &'a Path,
}

impl<'a> VerdictLedger<'a> {
    /// Open (lazily) the ledger at `path`. No I/O until [`append`](Self::append).
    pub fn new(path: &'a Path) -> Self {
        Self { path }
    }

    /// Append one verdict as a JSON line (creating parents if needed).
    pub fn append(&self, verdict: &MaintainVerdict) -> std::io::Result<()> {
        append_to_ledger(verdict, self.path)
    }

    /// Aggregate the ledger into a [`TrendSummary`] (absent ledger → empty summary).
    pub fn summarize(&self, conn: &Connection) -> Result<TrendSummary, MaintainError> {
        maintain_trend(conn, self.path)
    }

    /// Build the current-verdict scorecard for `verdict` (folding in this ledger's trend)
    /// and write it atomically to `snapshot_path`.
    pub fn materialize(
        &self,
        conn: &Connection,
        verdict: &MaintainVerdict,
        snapshot_path: &Path,
    ) -> Result<(), MaintainError> {
        let trend = self.summarize(conn)?;
        let snap = build_snapshot(verdict, &trend);
        write_snapshot_atomic(&snap, snapshot_path)?;
        Ok(())
    }
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;

    /// A minimal verdict for ledger round-trips — only the fields the trend reads matter.
    fn verdict(run_at: &str, status: &str, decision: &str) -> MaintainVerdict {
        MaintainVerdict {
            schema_version: "maintain-gate.v0.1".into(),
            run_at: run_at.into(),
            status: status.into(),
            decision: decision.into(),
            metric_delta: None,
            metric_up: None,
            guardrail_held: None,
            converging: None,
            drifting: None,
            signals: vec![],
            evidence: vec![],
            reason: "x".into(),
        }
    }

    #[test]
    fn append_then_summarize_round_trips() {
        let conn = crate::open_bench().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("maintain-gate.jsonl");
        let ledger = VerdictLedger::new(&path);

        // Absent ledger summarizes to an empty trend (degrade, never error).
        assert_eq!(ledger.summarize(&conn).unwrap().total, 0);

        ledger.append(&verdict("2026-06-18T00:00:00Z", "T", "accept")).unwrap();
        ledger.append(&verdict("2026-06-18T00:01:00Z", "C", "reject")).unwrap();
        ledger.append(&verdict("2026-06-18T00:02:00Z", "U", "escalate")).unwrap();

        // Append is additive: 3 writes = 3 lines.
        let body = std::fs::read_to_string(&path).unwrap();
        assert_eq!(body.lines().count(), 3, "append-only");

        // Summarize reads back exactly what was appended.
        let t = ledger.summarize(&conn).unwrap();
        assert_eq!(t.total, 3);
        assert_eq!(t.accepts, 1);
        assert_eq!(t.rejects, 1);
        assert_eq!(t.escalates, 1);
        assert_eq!(t.reward_hacks, 1, "the one status=C");
        assert_eq!(t.latest_status.as_deref(), Some("U"), "latest by run_at");

        // Materialize folds the same trend into a GA-envelope snapshot.
        let snap_path = dir.path().join("snap").join("last.json");
        ledger
            .materialize(&conn, &verdict("2026-06-18T00:03:00Z", "T", "accept"), &snap_path)
            .unwrap();
        let parsed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&snap_path).unwrap()).unwrap();
        assert_eq!(parsed["domain"], "maintain-gate");
        assert_eq!(parsed["status"], "T");
        assert_eq!(parsed["maintain_trend"]["total"], 3, "trend rides along");
    }
}
