//! Routing-quality trend lens over GA's `routing-eval-YYYY-MM-DD.json`.
//!
//! GA computes per-intent precision/recall/F1 each run, but only the **top-line**
//! accuracy is materialised daily (`build-views.sql` → `routing_eval`). The
//! actionable maintain signal — *which intents are weak, and which are degrading
//! run-over-run* — lives unqueried across the dated files. This reads them all
//! into `routing_evals` (one row per run × intent) + `routing_overall`, so the
//! trend is one SQL query. Read-only analyst lens; source of truth stays in GA.
//!
//! Absent/empty directory degrades to empty tables (never an error). The dotted
//! intent keys (`skill.scaleinfo`) are extracted via JSON-Pointer paths
//! (`/key/field`) — JSONPath's `$.` would mis-split the dot into nesting.

use std::path::{Path, PathBuf};

use duckdb::Connection;

use crate::source::{self, Col, Files};

/// Errors from the routing lens — the shared artifact-source error
/// ([`crate::source::SourceError`]); aliased so the lens's public API keeps its name.
pub type RoutingError = source::SourceError;

fn is_routing_eval(name: &str) -> bool {
    name.starts_with("routing-eval-") && name.ends_with(".json")
}

/// `routing_overall` column spec — the flat top-line metrics. The InScope/OOS/margin
/// metrics were added 2026-05-30, so older eval files lack them; reading each via
/// [`Col::extract`] (json_extract, not struct-field access) yields `NULL` on
/// whole-corpus absence rather than a bind error or a fabricated zero.
const OVERALL_SPEC: &[Col] = &[
    Col::direct("generated_at", "VARCHAR", "generatedAt::VARCHAR"),
    Col::direct("day", "VARCHAR", "(generatedAt::VARCHAR)[1:10]"),
    Col::extract("accuracy", "DOUBLE", "overall", "$.Accuracy"),
    Col::extract("in_scope_accuracy", "DOUBLE", "overall", "$.InScopeAccuracy"),
    Col::extract("oos_decline_rate", "DOUBLE", "overall", "$.OosDeclineRate"),
    Col::extract("mean_in_scope_margin", "DOUBLE", "overall", "$.MeanInScopeMargin"),
];

/// Build `routing_evals` (run × intent) and `routing_overall` (run). Returns the
/// per-intent row count; 0 when the directory is absent/empty.
// @ai:invariant build_routing_evals creates routing_evals with one row per (run, intent) parsed from each routing-eval-*.json perIntent map [T:test conf:0.9 src:ix_duck::routing::tests::evals_row_per_run_intent]
pub fn build_routing_evals(conn: &Connection, quality_dir: &Path) -> Result<usize, RoutingError> {
    let files = source::select_files(Files { dir: quality_dir, matches: is_routing_eval })?;
    // routing_overall is a flat projection → the deep artifact-source path owns the safe
    // read + empty-fallback (it can't drift back into struct-access / coalesce-0).
    source::materialize_files(conn, "routing_overall", &files, OVERALL_SPEC)?;
    // routing_evals is a map-explode (one row per perIntent key) — not a flat spec — so it
    // keeps a custom SELECT, but still reuses the shared file list + read flags.
    build_routing_evals_table(conn, &files)?;
    let n: i64 = conn.query_row("SELECT count(*) FROM routing_evals", [], |r| r.get(0))?;
    Ok(n as usize)
}

/// The `perIntent` map-explode (custom shape — out of scope for the flat [`Col`] spec —
/// but shares [`source::READ_FLAGS`] + [`source::sql_list`]). Dotted intent keys use
/// JSON-Pointer (`/key/field`); JSONPath's `$.` would mis-split the dot into nesting.
fn build_routing_evals_table(conn: &Connection, files: &[PathBuf]) -> Result<(), RoutingError> {
    let cols = "generated_at VARCHAR, day VARCHAR, intent VARCHAR, support BIGINT, \
                precision DOUBLE, recall DOUBLE, f1 DOUBLE, status VARCHAR";
    if files.is_empty() {
        conn.execute_batch(&format!("CREATE OR REPLACE TABLE routing_evals ({cols});"))?;
        return Ok(());
    }
    let list = source::sql_list(files);
    let flags = source::READ_FLAGS;
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE routing_evals AS
         WITH raw AS (
             SELECT generatedAt::VARCHAR AS generated_at, to_json(perIntent) AS pi
             FROM read_json_auto({list}, {flags})
         ),
         keyed AS (SELECT generated_at, pi, unnest(json_keys(pi)) AS intent FROM raw)
         SELECT generated_at, generated_at[1:10] AS day, intent,
                json_extract(pi, '/' || intent || '/Support')::BIGINT   AS support,
                json_extract(pi, '/' || intent || '/Precision')::DOUBLE AS precision,
                json_extract(pi, '/' || intent || '/Recall')::DOUBLE    AS recall,
                json_extract(pi, '/' || intent || '/F1')::DOUBLE        AS f1,
                json_extract_string(pi, '/' || intent || '/Status')     AS status
         FROM keyed;"
    ))?;
    Ok(())
}

/// Weakest intents in the **latest** run: `(intent, f1, support)` with `f1 < threshold`.
pub fn weakest_intents(
    conn: &Connection,
    threshold: f64,
) -> duckdb::Result<Vec<(String, f64, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT intent, f1, support FROM routing_evals
         WHERE generated_at = (SELECT max(generated_at) FROM routing_evals) AND f1 < ?
         ORDER BY f1",
    )?;
    stmt.query_map([threshold], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect()
}

/// Per-intent F1 regressions between the two most recent runs:
/// `(intent, prev_f1, latest_f1, delta)` where `delta < 0`, worst first. Empty if <2 runs.
pub fn intent_regressions(conn: &Connection) -> duckdb::Result<Vec<(String, f64, f64, f64)>> {
    let mut stmt = conn.prepare(
        "WITH runs AS (
             SELECT DISTINCT generated_at FROM routing_evals ORDER BY generated_at DESC LIMIT 2
         ),
         latest AS (SELECT max(generated_at) g FROM runs),
         prev   AS (SELECT min(generated_at) g FROM runs)
         SELECT l.intent, p.f1, l.f1, l.f1 - p.f1 AS delta
         FROM routing_evals l
         JOIN routing_evals p USING (intent)
         WHERE l.generated_at = (SELECT g FROM latest)
           AND p.generated_at = (SELECT g FROM prev)
           AND (SELECT count(*) FROM runs) = 2
           AND l.f1 < p.f1
         ORDER BY delta",
    )?;
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect()
}

/// One row of [`overall_trend`]: `(day, accuracy, oos_decline_rate, mean_in_scope_margin)`.
/// Each metric is `Option` because an eval file predating a metric carries `None` for it.
pub type TrendRow = (String, Option<f64>, Option<f64>, Option<f64>);

/// Overall accuracy / OOS-decline / margin trend, oldest→newest. Each metric is `None`
/// when the eval file predates that field — the `InScope`/OOS metrics were added on
/// 2026-05-30, so older `overall` objects carry only `Accuracy`. A *missing* metric must
/// read as **absent**, not `0.0`: coalescing to zero renders "not measured" identically to
/// a real zero and fabricates a trend (e.g. OOS-decline `0.000 → 0.875` looks like an
/// improvement that never happened — it was simply unrecorded before). The build reads
/// these via `json_extract` (not struct-field access), so a corpus where the field is
/// absent from *every* file yields `NULL` rather than a bind-time error.
// @ai:invariant overall_trend surfaces a metric absent from an older eval file as None, never 0.0, so a missing metric cannot be read as a genuine zero [T:test conf:0.9 src:ix_duck::routing::tests::missing_metric_reads_as_none_not_zero]
pub fn overall_trend(conn: &Connection) -> duckdb::Result<Vec<TrendRow>> {
    let mut stmt = conn.prepare(
        "SELECT day, accuracy, oos_decline_rate, mean_in_scope_margin
         FROM routing_overall ORDER BY generated_at",
    )?;
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect()
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/routing")
    }

    #[test]
    fn evals_row_per_run_intent() {
        let conn = crate::open_bench().unwrap();
        let n = build_routing_evals(&conn, &fixtures()).unwrap();
        // 2 runs × 3 intents.
        assert_eq!(n, 6, "one routing_evals row per (run, intent)");
    }

    #[test]
    fn weakest_intents_from_latest_run() {
        let conn = crate::open_bench().unwrap();
        build_routing_evals(&conn, &fixtures()).unwrap();
        let weak = weakest_intents(&conn, 0.8).unwrap();
        // Latest run (06-02): only scaleinfo is below 0.8 (F1 0.7).
        assert_eq!(weak.len(), 1);
        assert_eq!(weak[0].0, "skill.scaleinfo");
        assert!((weak[0].1 - 0.7).abs() < 1e-9);
    }

    #[test]
    fn detects_intent_regression() {
        let conn = crate::open_bench().unwrap();
        build_routing_evals(&conn, &fixtures()).unwrap();
        let regs = intent_regressions(&conn).unwrap();
        // scaleinfo dropped 0.9 → 0.7 between the two runs.
        assert_eq!(regs.len(), 1, "only scaleinfo regressed");
        let (intent, prev, latest, delta) = &regs[0];
        assert_eq!(intent, "skill.scaleinfo");
        assert!((prev - 0.9).abs() < 1e-9 && (latest - 0.7).abs() < 1e-9);
        assert!(*delta < 0.0);
    }

    #[test]
    fn overall_trend_two_runs_oos_drop() {
        let conn = crate::open_bench().unwrap();
        build_routing_evals(&conn, &fixtures()).unwrap();
        let trend = overall_trend(&conn).unwrap();
        assert_eq!(trend.len(), 2);
        // OOS decline rate fell 1.0 → 0.5 (a real maintain signal); both runs record it.
        assert!(
            (trend[0].2.unwrap() - 1.0).abs() < 1e-9 && (trend[1].2.unwrap() - 0.5).abs() < 1e-9
        );
    }

    /// An eval file predating the 2026-05-30 schema has no OOS/margin fields. A corpus of
    /// only such files would, with struct-field access, fail at bind time
    /// ("Could not find key"); via `json_extract` it builds, and the absent metrics read as
    /// `None` (not measured), NOT `0.0` — else a missing metric fakes a trend (the real-data
    /// bug: oos_decline showed 0.000→0.875, an "improvement" that was just the field being
    /// added). Guards both the crash and the false-zero.
    #[test]
    fn missing_metric_reads_as_none_not_zero() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("routing-eval-2026-05-11.json"),
            r#"{"generatedAt":"2026-05-11T00:00:00Z","schemaVersion":"1","totalPrompts":2,
                "overall":{"Accuracy":0.875,"Correct":7,"Total":8,"UnmatchedFallthrough":0},
                "perIntent":{"skill.scaleinfo":{"Support":2,"Precision":0.9,"Recall":0.9,"F1":0.9,"Status":"ok"}},
                "prompts":[]}"#,
        )
        .unwrap();
        let conn = crate::open_bench().unwrap();
        build_routing_evals(&conn, dir.path()).unwrap();
        let trend = overall_trend(&conn).unwrap();
        assert_eq!(trend.len(), 1);
        let (_, acc, oos, margin) = &trend[0];
        assert_eq!(*acc, Some(0.875), "accuracy is present");
        assert_eq!(*oos, None, "absent OOS-decline must be None, not 0.0");
        assert_eq!(*margin, None, "absent margin must be None, not 0.0");
    }

    #[test]
    fn absent_dir_degrades_to_empty() {
        let conn = crate::open_bench().unwrap();
        let n = build_routing_evals(&conn, Path::new("/no/such/dir")).unwrap();
        assert_eq!(n, 0);
        assert!(weakest_intents(&conn, 1.0).unwrap().is_empty());
        assert!(intent_regressions(&conn).unwrap().is_empty());
    }
}
