//! `ix_maintain_gate` MCP tool — the governance-verdict callable seam.
//!
//! Wraps [`ix_duck::maintain::evaluate`] (the hexavalent T/P/U/D/F/C RSI oracle) as an
//! MCP tool so an agent — or an IXQL pipeline via `mcp_tool_output → when verdict.status
//! == "T"` — can read the verdict in-process. **Feature-gated** (`maintain-gate`): it pulls
//! the bundled-DuckDB `duck` build, which the default agent never compiles. See
//! `docs/adr/0001-ixql-duckdb-integration-via-mcp-seam.md`.
//!
//! Read-only: it returns the verdict and does NOT append to the ledger (an ad-hoc MCP call
//! must not pollute the tamper-evident `maintain-gate.jsonl`). The verdict is **advisory**
//! until ledger write-isolation (Phase-3b) — it is not yet a binding governance gate.

use std::path::Path;

use ix_duck::maintain::{self, IterationScope, MaintainConfig, MaintainInputs};
use serde_json::Value;

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

/// MCP handler: `ix_maintain_gate(hits_path, corpus_dir, [loops_dir, query_embeddings_dir,
/// loop_id, commit_sha, repo_dir, run_at])` → the serialized `MaintainVerdict`.
pub fn ix_maintain_gate(args: Value) -> Result<Value, String> {
    let hits = str_arg(&args, "hits_path").ok_or("hits_path (string) is required")?;
    let corpus = str_arg(&args, "corpus_dir").ok_or("corpus_dir (string) is required")?;
    // Mirror the CLI guard: local paths only (no URLs / remote schemes).
    for p in [hits, corpus] {
        if p.contains("://") {
            return Err(format!("refusing non-local path: {p}"));
        }
    }

    let loops_dir = str_arg(&args, "loops_dir");
    let emb_dir = str_arg(&args, "query_embeddings_dir");
    let loop_id = str_arg(&args, "loop_id");
    let commit_sha = str_arg(&args, "commit_sha");
    let repo_dir = str_arg(&args, "repo_dir");
    // Caller may pin run_at (RFC3339); default to now so the verdict carries a real stamp.
    let run_at_owned = str_arg(&args, "run_at")
        .map(|s| s.to_string())
        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

    // Iteration scope is all-or-nothing: a loop_id without a verifiable commit can't be scoped.
    let iteration = match (loop_id, commit_sha, repo_dir) {
        (Some(l), Some(c), Some(r)) => Some(IterationScope {
            loop_id: l,
            commit_sha: c,
            repo_dir: Path::new(r),
        }),
        _ => None,
    };

    let conn = ix_duck::open_bench().map_err(|e| format!("open_bench: {e}"))?;
    let inputs = MaintainInputs {
        hits_path: Path::new(hits),
        corpus_dir: Path::new(corpus),
        loops_dir: loops_dir.map(Path::new),
        query_embeddings_dir: emb_dir.map(Path::new),
        iteration,
        run_at: &run_at_owned,
    };
    let verdict = maintain::evaluate(&conn, &MaintainConfig::default(), &inputs)
        .map_err(|e| format!("maintain-gate: {e}"))?;
    serde_json::to_value(&verdict).map_err(|e| format!("serialize verdict: {e}"))
}
