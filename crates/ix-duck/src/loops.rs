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

use std::path::Path;

use duckdb::Connection;

use crate::source::{self, ArtifactLens, Col};

/// Errors from the loop lens — the shared artifact-source error
/// ([`crate::source::SourceError`]); aliased so the lens's public API keeps its name.
pub type LoopError = source::SourceError;

/// `loop_iterations` column spec. Every field is top-level in GA's JSONL row, so each is a
/// flat [`Col::direct`] projection (`TRY_CAST` for numerics/booleans → NULL, never an error,
/// on a bad/absent value). The deep artifact-source path owns the safe read + empty-fallback.
const LOOP_SPEC: &[Col] = &[
    Col::direct("loop_id", "VARCHAR", "loop_id::VARCHAR"),
    Col::direct("domain", "VARCHAR", "domain::VARCHAR"),
    Col::direct("iteration", "BIGINT", "TRY_CAST(iteration AS BIGINT)"),
    Col::direct("ts", "VARCHAR", "ts::VARCHAR"),
    Col::direct("oracle_status", "VARCHAR", "oracle_status::VARCHAR"),
    Col::direct("metric_name", "VARCHAR", "metric_name::VARCHAR"),
    Col::direct("metric_before", "DOUBLE", "TRY_CAST(metric_before AS DOUBLE)"),
    Col::direct("metric_after", "DOUBLE", "TRY_CAST(metric_after AS DOUBLE)"),
    Col::direct("metric_delta", "DOUBLE", "TRY_CAST(metric_delta AS DOUBLE)"),
    Col::direct("verdict", "VARCHAR", "verdict::VARCHAR"),
    Col::direct("worst_item", "VARCHAR", "worst_item::VARCHAR"),
    Col::direct("artifact_edited", "VARCHAR", "artifact_edited::VARCHAR"),
    Col::direct("commit_sha", "VARCHAR", "commit_sha::VARCHAR"),
    Col::direct("roundtrip_passed", "BOOLEAN", "TRY_CAST(roundtrip_passed AS BOOLEAN)"),
];

fn is_iterations_jsonl(name: &str) -> bool {
    name.ends_with(".iterations.jsonl")
}

/// The loop lens is a flat artifact lens: a dir of `*.iterations.jsonl` → the
/// `loop_iterations` table via [`LOOP_SPEC`].
const LENS: ArtifactLens = ArtifactLens {
    table: "loop_iterations",
    matches: is_iterations_jsonl,
    spec: LOOP_SPEC,
};

/// Build `loop_iterations` (one row per loop × iteration). Returns the row count
/// (including any seed/test rows); 0 when the directory is absent/empty.
// @ai:invariant build_loop_iterations creates loop_iterations with one row per JSONL line across every *.iterations.jsonl in loops_dir [T:test conf:0.9 src:ix_duck::loops::tests::rows_one_per_iteration]
pub fn build_loop_iterations(conn: &Connection, loops_dir: &Path) -> Result<usize, LoopError> {
    LENS.materialize(conn, loops_dir)
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
    /// Sum of per-iteration metric deltas (net movement over the run). `None` when no
    /// iteration recorded a delta (all unmeasured, e.g. oracle couldn't run) — distinct
    /// from a *measured* net of `0.0`. (Absence is not zero — the defect class swept out
    /// of `routing`/`chatbot`.)
    pub net_delta: Option<f64>,
    pub final_verdict: String,
}

/// Per-loop convergence summary, most iterations first.
pub fn loop_summary(conn: &Connection) -> duckdb::Result<Vec<LoopSummary>> {
    // Aggregate once per loop (no join → no row multiplication), then pull the final
    // verdict via a correlated lookup on the max iteration.
    let mut stmt = conn.prepare(&format!(
        "WITH agg AS (
             SELECT loop_id, any_value(domain) AS domain, count(*) AS iterations,
                    sum(metric_delta) AS net_delta, max(iteration) AS last_iter
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
    use std::path::PathBuf;

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
            (improving.net_delta.unwrap() - 0.06).abs() < 1e-9,
            "net delta = 0.02+0.03+0.01, got {:?}",
            improving.net_delta
        );
    }

    #[test]
    fn net_delta_none_when_all_deltas_unmeasured() {
        // A loop whose iterations all recorded a null metric_delta (oracle couldn't run)
        // has net_delta = None — "unmeasured", not a measured net of 0.0.
        let dir = tempfile::tempdir().unwrap();
        let row = |it: i32| {
            format!(
                "{{\"loop_id\":\"l-x\",\"domain\":\"chatbot\",\"iteration\":{it},\"ts\":\"2026-06-18T00:0{it}:00Z\",\"oracle_status\":\"couldnt_run\",\"metric_name\":\"p\",\"metric_before\":null,\"metric_after\":null,\"metric_delta\":null,\"verdict\":\"couldnt_run\",\"worst_item\":\"x\",\"artifact_edited\":\"a\",\"commit_sha\":\"c\",\"roundtrip_passed\":false}}\n"
            )
        };
        std::fs::write(
            dir.path().join("x.iterations.jsonl"),
            format!("{}{}", row(1), row(2)),
        )
        .unwrap();
        let conn = crate::open_bench().unwrap();
        build_loop_iterations(&conn, dir.path()).unwrap();
        let sum = loop_summary(&conn).unwrap();
        assert_eq!(sum.len(), 1);
        assert_eq!(sum[0].iterations, 2);
        assert_eq!(sum[0].net_delta, None, "all-null deltas → None, not 0.0");
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
