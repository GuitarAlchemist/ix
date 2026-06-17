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

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use duckdb::Connection;

/// Errors from the routing lens: directory I/O vs DuckDB.
#[derive(Debug)]
pub enum RoutingError {
    Io(std::io::Error),
    Duck(duckdb::Error),
}
impl std::fmt::Display for RoutingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoutingError::Io(e) => write!(f, "routing-eval I/O error: {e}"),
            RoutingError::Duck(e) => write!(f, "duckdb error: {e}"),
        }
    }
}
impl std::error::Error for RoutingError {}
impl From<std::io::Error> for RoutingError {
    fn from(e: std::io::Error) -> Self {
        RoutingError::Io(e)
    }
}
impl From<duckdb::Error> for RoutingError {
    fn from(e: duckdb::Error) -> Self {
        RoutingError::Duck(e)
    }
}

/// `routing-eval-*.json` files in `quality_dir` (sorted). Absent dir → empty (skip);
/// any other read error on a present dir → surfaced.
fn eval_files(quality_dir: &Path) -> Result<Vec<PathBuf>, RoutingError> {
    let rd = match std::fs::read_dir(quality_dir) {
        Ok(r) => r,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut out = Vec::new();
    for entry in rd {
        let p = entry?.path();
        if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
            if n.starts_with("routing-eval-") && n.ends_with(".json") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// DuckDB list literal of POSIX-slashed, quote-escaped paths.
fn sql_list(paths: &[PathBuf]) -> String {
    let items: Vec<String> = paths
        .iter()
        .map(|p| format!("'{}'", p.to_string_lossy().replace('\\', "/").replace('\'', "''")))
        .collect();
    format!("[{}]", items.join(", "))
}

/// Build `routing_evals` (run × intent) and `routing_overall` (run). Returns the
/// per-intent row count; 0 when the directory is absent/empty.
// @ai:invariant build_routing_evals creates routing_evals with one row per (run, intent) parsed from each routing-eval-*.json perIntent map [T:test conf:0.9 src:ix_duck::routing::tests::evals_row_per_run_intent]
pub fn build_routing_evals(conn: &Connection, quality_dir: &Path) -> Result<usize, RoutingError> {
    let intent_cols = "generated_at VARCHAR, day VARCHAR, intent VARCHAR, support BIGINT, \
                       precision DOUBLE, recall DOUBLE, f1 DOUBLE, status VARCHAR";
    let overall_cols = "generated_at VARCHAR, day VARCHAR, accuracy DOUBLE, \
                        in_scope_accuracy DOUBLE, oos_decline_rate DOUBLE, mean_in_scope_margin DOUBLE";
    let files = eval_files(quality_dir)?;
    if files.is_empty() {
        conn.execute_batch(&format!(
            "CREATE OR REPLACE TABLE routing_evals ({intent_cols});
             CREATE OR REPLACE TABLE routing_overall ({overall_cols});"
        ))?;
        return Ok(0);
    }
    let list = sql_list(&files);
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE routing_evals AS
         WITH raw AS (
             SELECT generatedAt::VARCHAR AS generated_at, to_json(perIntent) AS pi
             FROM read_json_auto({list}, filename=true, union_by_name=true, sample_size=-1)
         ),
         keyed AS (SELECT generated_at, pi, unnest(json_keys(pi)) AS intent FROM raw)
         SELECT generated_at, generated_at[1:10] AS day, intent,
                json_extract(pi, '/' || intent || '/Support')::BIGINT   AS support,
                json_extract(pi, '/' || intent || '/Precision')::DOUBLE AS precision,
                json_extract(pi, '/' || intent || '/Recall')::DOUBLE    AS recall,
                json_extract(pi, '/' || intent || '/F1')::DOUBLE        AS f1,
                json_extract_string(pi, '/' || intent || '/Status')     AS status
         FROM keyed;
         CREATE OR REPLACE TABLE routing_overall AS
         SELECT generatedAt::VARCHAR AS generated_at, (generatedAt::VARCHAR)[1:10] AS day,
                TRY_CAST(overall.Accuracy AS DOUBLE)           AS accuracy,
                TRY_CAST(overall.InScopeAccuracy AS DOUBLE)    AS in_scope_accuracy,
                TRY_CAST(overall.OosDeclineRate AS DOUBLE)     AS oos_decline_rate,
                TRY_CAST(overall.MeanInScopeMargin AS DOUBLE)  AS mean_in_scope_margin
         FROM read_json_auto({list}, filename=true, union_by_name=true, sample_size=-1);"
    ))?;
    let n: i64 = conn.query_row("SELECT count(*) FROM routing_evals", [], |r| r.get(0))?;
    Ok(n as usize)
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

/// Overall accuracy / OOS-decline / margin trend, oldest→newest:
/// `(day, accuracy, oos_decline_rate, mean_in_scope_margin)`.
pub fn overall_trend(conn: &Connection) -> duckdb::Result<Vec<(String, f64, f64, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT day,
                coalesce(accuracy, 0), coalesce(oos_decline_rate, 0),
                coalesce(mean_in_scope_margin, 0)
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
        // OOS decline rate fell 1.0 → 0.5 (a real maintain signal).
        assert!((trend[0].2 - 1.0).abs() < 1e-9 && (trend[1].2 - 0.5).abs() < 1e-9);
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
