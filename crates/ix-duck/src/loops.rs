//! AFK / self-improvement **loop-iteration** lens over GA's
//! `state/quality/loops/<loop_id>.iterations.jsonl`.
//!
//! GA's loop-convergence ledger (Phase 1) + loop-decide controller (Phase 2) append
//! one row per optimize/AFK iteration. The *maintain* signal that lives unqueried
//! across those rows is **failure clustering**: which worst-items recur, which loops
//! oscillate instead of converging, and which artifacts churn. This reads the JSONL
//! into `loop_iterations` so each is one SQL query. Read-only; GA owns the files.
//!
//! Contract A (firm): the row schema is pinned by GA's fixture
//! `state/quality/_fixtures/loop-iterations.sample.jsonl`. The empty-state
//! `__seed__.iterations.jsonl` is loaded but carries `domain = "__seed__"`; the
//! analyses below exclude seed/test domains so a fresh checkout reads as "no signal"
//! rather than a phantom. Absent directory → empty tables (never an error).

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use duckdb::Connection;

/// Errors from the loop lens: directory I/O vs DuckDB.
#[derive(Debug)]
pub enum LoopError {
    Io(std::io::Error),
    Duck(duckdb::Error),
}
impl std::fmt::Display for LoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopError::Io(e) => write!(f, "loop-iterations I/O error: {e}"),
            LoopError::Duck(e) => write!(f, "duckdb error: {e}"),
        }
    }
}
impl std::error::Error for LoopError {}
impl From<std::io::Error> for LoopError {
    fn from(e: std::io::Error) -> Self {
        LoopError::Io(e)
    }
}
impl From<duckdb::Error> for LoopError {
    fn from(e: duckdb::Error) -> Self {
        LoopError::Duck(e)
    }
}

/// `*.iterations.jsonl` files in `loops_dir` (sorted). Absent dir → empty (skip);
/// any other read error on a present dir → surfaced.
fn ledger_files(loops_dir: &Path) -> Result<Vec<PathBuf>, LoopError> {
    let rd = match std::fs::read_dir(loops_dir) {
        Ok(r) => r,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut out = Vec::new();
    for entry in rd {
        let p = entry?.path();
        if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
            if n.ends_with(".iterations.jsonl") {
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
        .map(|p| {
            format!(
                "'{}'",
                p.to_string_lossy().replace('\\', "/").replace('\'', "''")
            )
        })
        .collect();
    format!("[{}]", items.join(", "))
}

const COLS: &str = "loop_id VARCHAR, domain VARCHAR, iteration BIGINT, ts VARCHAR, \
                    oracle_status VARCHAR, metric_name VARCHAR, metric_before DOUBLE, \
                    metric_after DOUBLE, metric_delta DOUBLE, verdict VARCHAR, \
                    worst_item VARCHAR, artifact_edited VARCHAR, commit_sha VARCHAR, \
                    roundtrip_passed BOOLEAN";

/// Build `loop_iterations` (one row per loop × iteration). Returns the row count
/// (including any seed/test rows); 0 when the directory is absent/empty.
// @ai:invariant build_loop_iterations creates loop_iterations with one row per JSONL line across every *.iterations.jsonl in loops_dir [T:test conf:0.9 src:ix_duck::loops::tests::rows_one_per_iteration]
pub fn build_loop_iterations(conn: &Connection, loops_dir: &Path) -> Result<usize, LoopError> {
    let files = ledger_files(loops_dir)?;
    if files.is_empty() {
        conn.execute_batch(&format!(
            "CREATE OR REPLACE TABLE loop_iterations ({COLS});"
        ))?;
        return Ok(0);
    }
    let list = sql_list(&files);
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE loop_iterations AS
         SELECT loop_id::VARCHAR              AS loop_id,
                domain::VARCHAR               AS domain,
                TRY_CAST(iteration AS BIGINT) AS iteration,
                ts::VARCHAR                   AS ts,
                oracle_status::VARCHAR        AS oracle_status,
                metric_name::VARCHAR          AS metric_name,
                TRY_CAST(metric_before AS DOUBLE) AS metric_before,
                TRY_CAST(metric_after  AS DOUBLE) AS metric_after,
                TRY_CAST(metric_delta  AS DOUBLE) AS metric_delta,
                verdict::VARCHAR              AS verdict,
                worst_item::VARCHAR           AS worst_item,
                artifact_edited::VARCHAR      AS artifact_edited,
                commit_sha::VARCHAR           AS commit_sha,
                TRY_CAST(roundtrip_passed AS BOOLEAN) AS roundtrip_passed
         FROM read_json_auto({list}, union_by_name=true, sample_size=-1);"
    ))?;
    let n: i64 = conn.query_row("SELECT count(*) FROM loop_iterations", [], |r| r.get(0))?;
    Ok(n as usize)
}

/// Real (non-seed, non-test) rows only — the predicate every analysis shares so a
/// fresh checkout (seed-only) reads as "no signal" instead of a phantom cluster.
const REAL: &str = "domain NOT IN ('__seed__', '__test__') \
                    AND loop_id NOT IN ('__seed__')";

/// Failure-signature cluster: worst-items recurring across iterations.
/// `(worst_item, occurrences, distinct_loops)` with `occurrences >= min_occurrences`,
/// most-recurrent first. The core "what keeps failing" maintain signal.
pub fn recurring_worst_items(
    conn: &Connection,
    min_occurrences: i64,
) -> duckdb::Result<Vec<(String, i64, i64)>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT worst_item, count(*) AS n, count(DISTINCT loop_id) AS loops
         FROM loop_iterations
         WHERE {REAL} AND worst_item IS NOT NULL AND worst_item NOT IN ('none', '')
         GROUP BY worst_item HAVING count(*) >= ?
         ORDER BY n DESC, worst_item"
    ))?;
    stmt.query_map([min_occurrences], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect()
}

/// Non-convergence signal: loops carrying both improving and regressing verdicts.
/// `(loop_id, n_improved, n_regressed)`, most-regressed first. A loop that flips
/// direction is thrashing, not converging.
pub fn oscillating_loops(conn: &Connection) -> duckdb::Result<Vec<(String, i64, i64)>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT loop_id,
                count(*) FILTER (WHERE verdict ILIKE '%improv%')  AS n_improved,
                count(*) FILTER (WHERE verdict ILIKE '%regress%') AS n_regressed
         FROM loop_iterations WHERE {REAL}
         GROUP BY loop_id
         HAVING n_improved > 0 AND n_regressed > 0
         ORDER BY n_regressed DESC, loop_id"
    ))?;
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect()
}

/// Artifact thrash: files edited across multiple iterations.
/// `(artifact_edited, edits, distinct_loops)` with `edits >= min_edits`, hottest first.
pub fn artifact_churn(
    conn: &Connection,
    min_edits: i64,
) -> duckdb::Result<Vec<(String, i64, i64)>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT artifact_edited, count(*) AS edits, count(DISTINCT loop_id) AS loops
         FROM loop_iterations
         WHERE {REAL} AND artifact_edited IS NOT NULL AND artifact_edited NOT IN ('none', '')
         GROUP BY artifact_edited HAVING count(*) >= ?
         ORDER BY edits DESC, artifact_edited"
    ))?;
    stmt.query_map([min_edits], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect()
}

/// One row of [`loop_summary`] — a loop's convergence at a glance.
#[derive(Debug, Clone)]
pub struct LoopSummary {
    pub loop_id: String,
    pub domain: String,
    pub iterations: i64,
    /// Sum of per-iteration metric deltas (net movement over the run).
    pub net_delta: f64,
    pub final_verdict: String,
}

/// Per-loop convergence summary, most iterations first.
pub fn loop_summary(conn: &Connection) -> duckdb::Result<Vec<LoopSummary>> {
    // Aggregate once per loop (no join → no row multiplication), then pull the final
    // verdict via a correlated lookup on the max iteration.
    let mut stmt = conn.prepare(&format!(
        "WITH agg AS (
             SELECT loop_id, any_value(domain) AS domain, count(*) AS iterations,
                    coalesce(sum(metric_delta), 0) AS net_delta, max(iteration) AS last_iter
             FROM loop_iterations WHERE {REAL}
             GROUP BY loop_id
         )
         SELECT a.loop_id, a.domain, a.iterations, a.net_delta,
                (SELECT i.verdict FROM loop_iterations i
                 WHERE i.loop_id = a.loop_id AND i.iteration = a.last_iter
                 LIMIT 1) AS final_verdict
         FROM agg a
         ORDER BY a.iterations DESC, a.loop_id"
    ))?;
    stmt.query_map([], |r| {
        Ok(LoopSummary {
            loop_id: r.get(0)?,
            domain: r.get(1)?,
            iterations: r.get(2)?,
            net_delta: r.get(3)?,
            final_verdict: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
        })
    })?
    .collect()
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/loops")
    }

    #[test]
    fn rows_one_per_iteration() {
        let conn = crate::open_bench().unwrap();
        let n = build_loop_iterations(&conn, &fixtures()).unwrap();
        // 3 improving + 3 oscillating + 1 seed row = 7 lines.
        assert_eq!(n, 7);
    }

    #[test]
    fn recurring_worst_items_cluster() {
        let conn = crate::open_bench().unwrap();
        build_loop_iterations(&conn, &fixtures()).unwrap();
        let rec = recurring_worst_items(&conn, 2).unwrap();
        // "p-flaky" recurs across the oscillating loop; seed's "none" excluded.
        assert!(rec.iter().any(|(w, n, _)| w == "p-flaky" && *n >= 2));
        assert!(!rec.iter().any(|(w, _, _)| w == "none"));
    }

    #[test]
    fn oscillating_loop_flagged() {
        let conn = crate::open_bench().unwrap();
        build_loop_iterations(&conn, &fixtures()).unwrap();
        let osc = oscillating_loops(&conn).unwrap();
        assert_eq!(osc.len(), 1, "only the oscillating loop flips direction");
        assert_eq!(osc[0].0, "chatbot-oscillating");
        assert!(osc[0].1 > 0 && osc[0].2 > 0);
    }

    #[test]
    fn summary_excludes_seed() {
        let conn = crate::open_bench().unwrap();
        build_loop_iterations(&conn, &fixtures()).unwrap();
        let sum = loop_summary(&conn).unwrap();
        // Two real loops; the __seed__ row is excluded.
        assert_eq!(sum.len(), 2);
        assert!(!sum.iter().any(|s| s.loop_id == "__seed__"));
        let improving = sum
            .iter()
            .find(|s| s.loop_id == "chatbot-improving")
            .unwrap();
        // Exact values — guards against row-multiplication (a cartesian join would
        // report iterations=9 and net_delta=0.18 for this 3-row loop).
        assert_eq!(
            improving.iterations, 3,
            "one row per iteration, not multiplied"
        );
        assert!(
            (improving.net_delta - 0.06).abs() < 1e-9,
            "net delta = 0.02+0.03+0.01, got {}",
            improving.net_delta
        );
    }

    #[test]
    fn absent_dir_degrades_to_empty() {
        let conn = crate::open_bench().unwrap();
        let n = build_loop_iterations(&conn, Path::new("/no/such/dir")).unwrap();
        assert_eq!(n, 0);
        assert!(recurring_worst_items(&conn, 1).unwrap().is_empty());
        assert!(oscillating_loops(&conn).unwrap().is_empty());
        assert!(loop_summary(&conn).unwrap().is_empty());
    }

    /// Seed-only checkout (the fresh-clone state) must read as "no signal".
    #[test]
    fn seed_only_reads_as_no_signal() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("__seed__.iterations.jsonl"),
            "{\"loop_id\":\"__seed__\",\"domain\":\"__seed__\",\"iteration\":0,\"ts\":\"1970-01-01T00:00:00Z\",\"oracle_status\":\"ok\",\"metric_name\":\"none\",\"metric_before\":0.0,\"metric_after\":0.0,\"metric_delta\":0.0,\"verdict\":\"improved\",\"worst_item\":\"none\",\"artifact_edited\":\"none\",\"commit_sha\":\"none\",\"roundtrip_passed\":false}\n",
        )
        .unwrap();
        let conn = crate::open_bench().unwrap();
        let n = build_loop_iterations(&conn, dir.path()).unwrap();
        assert_eq!(n, 1, "the seed row is loaded");
        assert!(
            loop_summary(&conn).unwrap().is_empty(),
            "but excluded from signal"
        );
        assert!(recurring_worst_items(&conn, 1).unwrap().is_empty());
    }
}
