//! `ix_maintain_gate` — the RSI evaluation oracle CLI (Phase 0). Thin shell over
//! `ix_duck::maintain`: fuse the externally-derived yield metric ∧ the chatbot
//! guardrail into one hexavalent verdict, append it to a tamper-evident ledger, and
//! set the exit code (accept=0, reject=1, escalate=2).
//!
//!   cargo run -p ix-duck --features duck --example ix_maintain_gate -- <hits.jsonl> <corpus-dir>
//!   cargo run -p ix-duck --features duck --example ix_maintain_gate           # defaults below
//!
//! Defaults: metric ← `state/thinking-machine/hits.jsonl` (IX-local, harness-written);
//! guardrail ← `../ga/state/quality/chatbot-qa`. Verdict appended to
//! `state/thinking-machine/maintain-gate.jsonl`.

use std::path::PathBuf;

use ix_duck::maintain::{self, MaintainConfig, MaintainInputs};

fn default_hits() -> PathBuf {
    PathBuf::from("state/thinking-machine/hits.jsonl")
}

fn default_corpus() -> PathBuf {
    if let Ok(root) = std::env::var("GA_ROOT") {
        return PathBuf::from(root).join("state/quality/chatbot-qa");
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let ix_root = cwd.canonicalize().unwrap_or(cwd);
    ix_root
        .parent()
        .map(|p| p.join("ga/state/quality/chatbot-qa"))
        .unwrap_or_else(|| PathBuf::from("../ga/state/quality/chatbot-qa"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let hits = args.next().map(PathBuf::from).unwrap_or_else(default_hits);
    let corpus = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(default_corpus);
    for p in [&hits, &corpus] {
        if p.to_string_lossy().contains("://") {
            eprintln!("refusing non-local path: {}", p.display());
            std::process::exit(2);
        }
    }

    let conn = ix_duck::open_bench()?;
    let run_at = chrono::Utc::now().to_rfc3339();
    let inputs = MaintainInputs {
        hits_path: &hits,
        corpus_dir: &corpus,
        run_at: &run_at,
    };
    let verdict = maintain::evaluate(&conn, &MaintainConfig::default(), &inputs)?;

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

    let ledger = PathBuf::from("state/thinking-machine/maintain-gate.jsonl");
    if let Err(e) = maintain::append_to_ledger(&verdict, &ledger) {
        eprintln!(
            "warning: could not append to ledger {}: {e}",
            ledger.display()
        );
    } else {
        println!("\n  → appended to {}", ledger.display());
    }

    std::process::exit(maintain::exit_code(&verdict.status));
}
