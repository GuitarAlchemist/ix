//! `ix_maintain_gate` — the RSI evaluation oracle CLI (Phase 0). Thin shell over
//! `ix_duck::maintain`: fuse the externally-derived yield metric ∧ the chatbot
//! guardrail into one hexavalent verdict, append it to a tamper-evident ledger, and
//! set the exit code (accept=0, reject=1, escalate=2).
//!
//!   cargo run -p ix-duck --features duck --example ix_maintain_gate -- <hits.jsonl> <corpus-dir>
//!   cargo run -p ix-duck --features duck --example ix_maintain_gate -- --json …   # machine-readable
//!   cargo run -p ix-duck --features duck --example ix_maintain_gate              # defaults below
//!
//! `--json` (first arg) emits the `MaintainVerdict` as one JSON object to **stdout**
//! and nothing else — the machine-readable *seam* an MCP wrapper or the IXQL executor
//! (`mcp_tool_output → when verdict.status == "T"`) consumes. See
//! `docs/adr/0001-ixql-duckdb-integration-via-mcp-seam.md`. Human-readable is the default.
//!
//! Defaults: metric ← `state/thinking-machine/hits.jsonl` (IX-local, harness-written);
//! guardrail ← `../ga/state/quality/chatbot-qa`. Verdict appended to
//! `state/thinking-machine/maintain-gate.jsonl`.

use std::path::PathBuf;

use ix_duck::maintain::{self, IterationScope, MaintainConfig, MaintainInputs};

fn default_hits() -> PathBuf {
    PathBuf::from("state/thinking-machine/hits.jsonl")
}

fn ga_quality() -> PathBuf {
    if let Ok(root) = std::env::var("GA_ROOT") {
        return PathBuf::from(root).join("state/quality");
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let ix_root = cwd.canonicalize().unwrap_or(cwd);
    ix_root
        .parent()
        .map(|p| p.join("ga/state/quality"))
        .unwrap_or_else(|| PathBuf::from("../ga/state/quality"))
}

fn default_corpus() -> PathBuf {
    ga_quality().join("chatbot-qa")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    // `--json` (first token) → emit ONLY the verdict JSON to stdout (the callable seam).
    let json = argv.first().map(|s| s == "--json").unwrap_or(false);
    if json {
        argv.remove(0);
    }
    let mut args = argv.into_iter();
    let hits = args.next().map(PathBuf::from).unwrap_or_else(default_hits);
    let corpus = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(default_corpus);
    // Optional Phase-3a iteration correlation: <loop_id> <commit_sha>.
    let loop_id = args.next();
    let commit_sha = args.next();
    for p in [&hits, &corpus] {
        if p.to_string_lossy().contains("://") {
            eprintln!("refusing non-local path: {}", p.display());
            std::process::exit(2);
        }
    }

    let conn = ix_duck::open_bench()?;
    let run_at = chrono::Utc::now().to_rfc3339();
    // Phase 1: also consult the convergence + drift lenses (advisory — absent dirs
    // degrade to "no data", never block).
    let loops_dir = ga_quality().join("loops");
    let emb_dir = ga_quality().join("query-embeddings");
    let ga_root = ga_quality()
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or_default();
    let iteration = match (&loop_id, &commit_sha) {
        (Some(l), Some(c)) => Some(IterationScope {
            loop_id: l,
            commit_sha: c,
            repo_dir: &ga_root,
        }),
        _ => None,
    };
    let inputs = MaintainInputs {
        hits_path: &hits,
        corpus_dir: &corpus,
        loops_dir: Some(&loops_dir),
        query_embeddings_dir: Some(&emb_dir),
        iteration,
        run_at: &run_at,
    };
    let verdict = maintain::evaluate(&conn, &MaintainConfig::default(), &inputs)?;

    if json {
        // The seam: one verdict object on stdout, nothing else (pipe-friendly).
        println!("{}", serde_json::to_string(&verdict)?);
    } else {
        println!(
            "maintain-gate verdict: {} ({})",
            verdict.status, verdict.decision
        );
        println!("  reason: {}", verdict.reason);
        println!("  metric: {}", hits.display());
        println!("  guardrail: {}", corpus.display());
        println!("\n  signals:");
        for s in &verdict.signals {
            let mark = match s.ok {
                Some(true) => "ok ",
                Some(false) => "BAD",
                None => " ? ",
            };
            println!("    [{mark}] {:<10} {}", s.lens, s.detail);
        }
        println!("\n  evidence:");
        for e in &verdict.evidence {
            println!("    {:<18} {}  ({})", e.kind, e.hash, e.source);
        }
    }

    let ledger = PathBuf::from("state/thinking-machine/maintain-gate.jsonl");
    match maintain::append_to_ledger(&verdict, &ledger) {
        // In --json mode stdout stays pure verdict JSON; the ledger note goes to stderr.
        Ok(()) if json => eprintln!("appended to {}", ledger.display()),
        Ok(()) => println!("\n  → appended to {}", ledger.display()),
        Err(e) => eprintln!(
            "warning: could not append to ledger {}: {e}",
            ledger.display()
        ),
    }

    std::process::exit(maintain::exit_code(&verdict.status));
}
