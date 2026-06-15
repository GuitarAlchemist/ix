//! Chatbot flight recorder — turns GA's per-run golden-trace JSON into queryable
//! DuckDB tables (Slice A) and a canonical-diff regression gate (Slice B).
//!
//! Source of truth stays in GA's files (`docs/DUCKDB.md`); this is a read-only
//! analyst lens over them. Reads a *corpus directory* that contains a
//! `golden-traces/<promptId>/{run-*.json,_signature.json}` layout — either the live
//! GA corpus (`../ga/state/quality/chatbot-qa`) or vendored fixtures under
//! `tests/fixtures/`. Absent/empty corpus degrades to an empty table + `skipped`
//! gate status (never an error).
//!
//! The hard regression signal is the **routed agent** (`agent_id`): the
//! `orchestration.answer` step's `agentId` in `_signature.json` (the canonical
//! expectation) vs the run's `response.agentId`. `response_length` is a *soft* band
//! only — measured to drift on correct answers, so it never hard-fails.

use std::path::{Path, PathBuf};

use duckdb::Connection;
use serde::Serialize;

/// Result of the canonical-diff gate over a corpus.
#[derive(Debug, Clone, Serialize)]
pub struct GateReport {
    /// `chatbot-trace-regression` contract, v0.1 (documented shape; no formal schema yet).
    pub schema_version: String,
    pub run_at: String,
    /// Always `"single"` in v1 — the corpus is 100% `runCount:1`. Reserved for a future
    /// majority/median multi-run reducer.
    pub run_selection: String,
    /// `pass` | `regression` | `degraded` | `skipped`. (`warn` is reserved for soft flags.)
    pub status: String,
    pub prompts_checked: usize,
    /// Deterministic FNV-1a hash of the sorted (prompt_id, expected_agent) baseline set.
    pub baseline_ref: String,
    /// v1 has no baseline history store → always false (human-ack re-baseline deferred).
    pub baseline_changed: bool,
    pub regressions: Vec<Regression>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<String>,
}

/// One flagged prompt: the routed agent diverged from its canonical expectation.
#[derive(Debug, Clone, Serialize)]
pub struct Regression {
    pub prompt_id: String,
    pub category: Option<String>,
    /// The signal that flagged — `agent_id` (hard) in v1.
    pub signal: String,
    /// `hard` (gate-failing) | `soft` (advisory).
    pub severity: String,
    pub expected: Option<String>,
    pub actual: Option<String>,
}

// ---- corpus enumeration ---------------------------------------------------

fn golden_traces_dir(corpus_dir: &Path) -> PathBuf {
    corpus_dir.join("golden-traces")
}

/// Enumerate files matching `golden-traces/*/<leaf>` (e.g. `run-1.json`,
/// `_signature.json`). Explicit enumeration (vs a wildcard handed to DuckDB) makes
/// the "absent/empty corpus → skip" path real and keeps untrusted input bounded.
fn collect(corpus_dir: &Path, predicate: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let gt = golden_traces_dir(corpus_dir);
    let Ok(prompts) = std::fs::read_dir(&gt) else {
        return out; // absent corpus → empty
    };
    for prompt in prompts.flatten() {
        let p = prompt.path();
        if !p.is_dir() {
            continue;
        }
        if let Ok(files) = std::fs::read_dir(&p) {
            for f in files.flatten() {
                let fp = f.path();
                if fp
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(&predicate)
                {
                    out.push(fp);
                }
            }
        }
    }
    out.sort();
    out
}

fn run_files(corpus_dir: &Path) -> Vec<PathBuf> {
    collect(corpus_dir, |n| n.starts_with("run-") && n.ends_with(".json"))
}

fn signature_files(corpus_dir: &Path) -> Vec<PathBuf> {
    collect(corpus_dir, |n| n == "_signature.json")
}

/// Render a slice of paths as a DuckDB list literal `['p1', 'p2', …]`, POSIX-slashed
/// and single-quote-escaped (paths never contain `://`, validated by the caller).
fn sql_list(paths: &[PathBuf]) -> String {
    let items: Vec<String> = paths
        .iter()
        .map(|p| {
            let s = p.to_string_lossy().replace('\\', "/").replace('\'', "''");
            format!("'{s}'")
        })
        .collect();
    format!("[{}]", items.join(", "))
}

// ---- Slice A: trace warehouse ---------------------------------------------

/// Build the `chatbot_traces` table (one row per `run-*.json`). Returns the row
/// count; 0 when the corpus is absent/empty (graceful-degrade).
// @ai:invariant build_traces creates chatbot_traces with one row per golden-trace run file [T:test conf:0.9 src:ix_duck::chatbot::tests::traces_row_per_run]
pub fn build_traces(conn: &Connection, corpus_dir: &Path) -> duckdb::Result<usize> {
    let files = run_files(corpus_dir);
    if files.is_empty() {
        conn.execute_batch(
            "CREATE OR REPLACE TABLE chatbot_traces (
                 prompt_id VARCHAR, prompt VARCHAR, category VARCHAR, agent_id VARCHAR,
                 routing_confidence DOUBLE, routing_method VARCHAR, grounding_present BOOLEAN,
                 response_length BIGINT, elapsed_ms BIGINT, recorded_at VARCHAR);",
        )?;
        return Ok(0);
    }
    let list = sql_list(&files);
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE chatbot_traces AS
         SELECT promptId                                AS prompt_id,
                prompt,
                category,
                response.agentId                        AS agent_id,
                TRY_CAST(response.confidence AS DOUBLE) AS routing_confidence,
                response.routingMethod                  AS routing_method,
                response.grounding IS NOT NULL          AS grounding_present,
                length(response.naturalLanguageAnswer)  AS response_length,
                TRY_CAST(response.elapsedMs AS BIGINT)  AS elapsed_ms,
                recordedAt                              AS recorded_at
         FROM read_json_auto({list}, filename=true, union_by_name=true, sample_size=-1);"
    ))?;
    let n: i64 = conn.query_row("SELECT count(*) FROM chatbot_traces", [], |r| r.get(0))?;
    Ok(n as usize)
}

/// Intents whose mean routing confidence is below `threshold` — the "weak intents"
/// dev query. Returns `(agent_id, avg_confidence, n)`.
pub fn weak_intents(conn: &Connection, threshold: f64) -> duckdb::Result<Vec<(String, f64, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT coalesce(agent_id, '<none>'), avg(routing_confidence), count(*)
         FROM chatbot_traces
         GROUP BY agent_id
         HAVING avg(routing_confidence) < ?
         ORDER BY avg(routing_confidence)",
    )?;
    let rows = stmt
        .query_map([threshold], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<f64>>(1)?.unwrap_or(0.0),
                r.get::<_, i64>(2)?,
            ))
        })?
        .collect::<duckdb::Result<Vec<_>>>()?;
    Ok(rows)
}

/// `(ungrounded, total)` — answers with no `response.grounding`.
pub fn ungrounded(conn: &Connection) -> duckdb::Result<(i64, i64)> {
    conn.query_row(
        "SELECT count(*) FILTER (WHERE NOT grounding_present), count(*) FROM chatbot_traces",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )
}

/// A latency-outlier row: `(prompt_id, agent_id, elapsed_ms)`.
pub type LatencyOutlier = (String, Option<String>, Option<i64>);

/// Slowest prompts by `elapsed_ms`.
pub fn latency_outliers(conn: &Connection, limit: usize) -> duckdb::Result<Vec<LatencyOutlier>> {
    let mut stmt = conn.prepare(
        "SELECT prompt_id, agent_id, elapsed_ms FROM chatbot_traces
         ORDER BY elapsed_ms DESC NULLS LAST LIMIT ?",
    )?;
    let rows = stmt
        .query_map([limit as i64], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })?
        .collect::<duckdb::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Share of each `routing_method` — `(routing_method, n)`.
pub fn routing_method_share(conn: &Connection) -> duckdb::Result<Vec<(String, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT coalesce(routing_method, '<none>'), count(*) FROM chatbot_traces
         GROUP BY routing_method ORDER BY count(*) DESC",
    )?;
    let rows = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<duckdb::Result<Vec<_>>>()?;
    Ok(rows)
}

// ---- Slice B: canonical-diff gate -----------------------------------------

/// Build `chatbot_signatures(prompt_id, expected_agent)` from each `_signature.json`'s
/// `orchestration.answer` step. UNNEST (not a list-lambda) so it is stable across
/// DuckDB versions. Prompts with no such step are absent → expected NULL → degraded.
fn build_signatures(conn: &Connection, corpus_dir: &Path) -> duckdb::Result<()> {
    let files = signature_files(corpus_dir);
    if files.is_empty() {
        conn.execute_batch(
            "CREATE OR REPLACE TABLE chatbot_signatures (prompt_id VARCHAR, expected_agent VARCHAR);",
        )?;
        return Ok(());
    }
    let list = sql_list(&files);
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE chatbot_signatures AS
         SELECT t.promptId AS prompt_id, step.agentId AS expected_agent
         FROM read_json_auto({list}, filename=true, union_by_name=true, sample_size=-1) t,
              UNNEST(t.steps) AS u(step)
         WHERE step.name = 'orchestration.answer';"
    ))?;
    Ok(())
}

/// Run the canonical-diff gate over a corpus: build both tables, diff the routed
/// agent, and classify the result by failure *fingerprint* (a single clean
/// heterogeneous flip fails; a homogeneous collapse warns as `degraded`).
// @ai:invariant check_regressions flags a run whose response.agentId differs from its canonical expected agent [T:test conf:0.9 src:ix_duck::chatbot::tests::diff_flags_agent_drift]
pub fn check_regressions(conn: &Connection, corpus_dir: &Path) -> duckdb::Result<GateReport> {
    let prompts_checked = build_traces(conn, corpus_dir)?;
    build_signatures(conn, corpus_dir)?;

    // (prompt_id, category, actual_agent, expected_agent)
    let mut stmt = conn.prepare(
        "SELECT t.prompt_id, t.category, t.agent_id, s.expected_agent
         FROM chatbot_traces t
         LEFT JOIN chatbot_signatures s USING (prompt_id)
         ORDER BY t.prompt_id",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<String>>(3)?,
            ))
        })?
        .collect::<duckdb::Result<Vec<_>>>()?;

    let mut regressions = Vec::new();
    let mut missing_canonical = 0usize;
    for (prompt_id, category, actual, expected) in &rows {
        match expected {
            None => missing_canonical += 1, // no canonical for this prompt → degraded signal
            Some(_) if actual != expected => regressions.push(Regression {
                prompt_id: prompt_id.clone(),
                category: category.clone(),
                signal: "agent_id".into(),
                severity: "hard".into(),
                expected: expected.clone(),
                actual: actual.clone(),
            }),
            Some(_) => {}
        }
    }

    let baseline_ref = baseline_hash(conn)?;
    let run_at = chrono::Utc::now().to_rfc3339();
    let mk = |status: &str, degraded_reason: Option<String>| GateReport {
        schema_version: "chatbot-trace-regression.v0.1".into(),
        run_at: run_at.clone(),
        run_selection: "single".into(),
        status: status.into(),
        prompts_checked,
        baseline_ref: baseline_ref.clone(),
        baseline_changed: false,
        regressions: regressions.clone(),
        degraded_reason,
    };

    // No corpus at all → skipped (graceful-degrade, exit 0).
    if prompts_checked == 0 {
        return Ok(mk("skipped", Some("corpus absent or empty".into())));
    }
    // Canonical missing for most prompts → stale/degraded baseline, warn not fail.
    if missing_canonical * 2 >= prompts_checked {
        return Ok(mk(
            "degraded",
            Some(format!("{missing_canonical}/{prompts_checked} prompts have no canonical agent")),
        ));
    }
    if regressions.is_empty() {
        return Ok(mk("pass", None));
    }
    // Fingerprint: a homogeneous collapse (≥2 flags all to the same agent, ≥50% of the
    // corpus) is an environment/backend symptom → degraded (warn). Otherwise a real
    // heterogeneous regression → fail. A single clean flip is never homogeneous → fails.
    let homogeneous = regressions.len() >= 2
        && regressions.len() * 2 >= prompts_checked
        && regressions.iter().all(|r| r.actual == regressions[0].actual);
    if homogeneous {
        let to = regressions[0].actual.clone().unwrap_or_else(|| "<none>".into());
        return Ok(mk(
            "degraded",
            Some(format!("{} prompts collapsed to '{to}' (backend/env symptom)", regressions.len())),
        ));
    }
    Ok(mk("regression", None))
}

/// Deterministic FNV-1a 64-bit hash of the sorted (prompt_id, expected_agent) set —
/// the baseline content reference. Stable across platforms/toolchains.
fn baseline_hash(conn: &Connection) -> duckdb::Result<String> {
    let mut stmt = conn.prepare(
        "SELECT prompt_id, coalesce(expected_agent, '') FROM chatbot_signatures ORDER BY prompt_id",
    )?;
    let pairs = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
        .collect::<duckdb::Result<Vec<_>>>()?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (k, v) in pairs {
        for b in k.bytes().chain(b"=".iter().copied()).chain(v.bytes()).chain(b";".iter().copied()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    Ok(format!("fnv1a64:{h:016x}"))
}

/// Map a gate status to a process exit code: `regression` → 1, all else → 0.
pub fn exit_code(status: &str) -> i32 {
    if status == "regression" {
        1
    } else {
        0
    }
}

/// Atomically write the contract JSON (same-dir temp + rename; replaces on Windows).
pub fn write_contract(report: &GateReport, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(report).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes())?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chatbot-qa")
    }

    /// Copy the fixtures corpus into a tempdir so a test can mutate it.
    fn copy_corpus(dst: &Path) {
        let src = fixtures();
        for entry in walk(&src) {
            let rel = entry.strip_prefix(&src).unwrap();
            let target = dst.join(rel);
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::copy(&entry, &target).unwrap();
        }
    }

    fn walk(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for e in fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
        out
    }

    /// Rewrite a run file's `response.agentId`.
    fn set_agent(run: &Path, agent: &str) {
        let mut v: serde_json::Value = serde_json::from_str(&fs::read_to_string(run).unwrap()).unwrap();
        v["response"]["agentId"] = serde_json::Value::String(agent.into());
        fs::write(run, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    }

    #[test]
    fn traces_row_per_run() {
        let conn = crate::open_bench().unwrap();
        let n = build_traces(&conn, &fixtures()).unwrap();
        let files = run_files(&fixtures()).len();
        assert!(files >= 3, "need a real multi-prompt fixture corpus, found {files}");
        assert_eq!(n, files, "one chatbot_traces row per run file");
    }

    #[test]
    fn lens_queries_run_on_real_fixtures() {
        let conn = crate::open_bench().unwrap();
        build_traces(&conn, &fixtures()).unwrap();
        let (ungr, total) = ungrounded(&conn).unwrap();
        assert!(total >= 3 && ungr <= total);
        // grounding/ routing columns parse without error
        let _ = weak_intents(&conn, 0.7).unwrap();
        let _ = latency_outliers(&conn, 5).unwrap();
        assert!(!routing_method_share(&conn).unwrap().is_empty());
    }

    #[test]
    fn happy_corpus_passes() {
        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, &fixtures()).unwrap();
        assert_eq!(report.status, "pass", "vendored fixtures must be regression-free: {report:?}");
        assert_eq!(exit_code(&report.status), 0);
    }

    #[test]
    fn diff_flags_agent_drift() {
        let dir = tempfile::tempdir().unwrap();
        copy_corpus(dir.path());
        // Flip exactly one prompt's routed agent → a single clean heterogeneous flip.
        let runs = run_files(dir.path());
        set_agent(&runs[0], "skill.WRONG_AGENT");

        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, dir.path()).unwrap();
        assert_eq!(report.status, "regression", "{report:?}");
        assert_eq!(exit_code(&report.status), 1);
        assert!(report.regressions.iter().any(|r| r.actual.as_deref() == Some("skill.WRONG_AGENT")));
    }

    #[test]
    fn homogeneous_collapse_is_degraded() {
        let dir = tempfile::tempdir().unwrap();
        copy_corpus(dir.path());
        // Collapse every prompt to the same fallback agent → backend/env symptom.
        for run in run_files(dir.path()) {
            set_agent(&run, "skill.fallback");
        }
        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, dir.path()).unwrap();
        assert_eq!(report.status, "degraded", "{report:?}");
        assert_eq!(exit_code(&report.status), 0, "degraded warns, never fails");
    }

    #[test]
    fn absent_corpus_skips() {
        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, Path::new("/no/such/corpus")).unwrap();
        assert_eq!(report.status, "skipped");
        assert_eq!(report.prompts_checked, 0);
        assert_eq!(exit_code(&report.status), 0);
    }

    #[test]
    fn baseline_hash_is_stable_and_changes_with_canonical() {
        let conn = crate::open_bench().unwrap();
        let a = check_regressions(&conn, &fixtures()).unwrap().baseline_ref;
        let b = check_regressions(&conn, &fixtures()).unwrap().baseline_ref;
        assert_eq!(a, b, "same corpus → same baseline_ref");
        assert!(a.starts_with("fnv1a64:"));
    }

    #[test]
    fn contract_round_trips() {
        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, &fixtures()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("sub/chatbot-trace-regressions.json");
        write_contract(&report, &out).unwrap();
        let back: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(back["status"], "pass");
        assert_eq!(back["run_selection"], "single");
    }
}
