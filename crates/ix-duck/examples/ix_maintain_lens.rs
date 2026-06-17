//! `ix_maintain_lens` — the AFK self-improvement maintain signal over GA's loop
//! ledger + chatbot query embeddings. Thin shell over `ix_duck::{loops, ood}`:
//! which worst-items recur, which loops oscillate, which artifacts churn, and which
//! queries are out-of-domain. Both inputs degrade gracefully when absent.
//!
//!   cargo run -p ix-duck --features duck --example ix_maintain_lens             # live ../ga
//!   cargo run -p ix-duck --features duck --example ix_maintain_lens -- <ga-root>
//!
//! Reads `<ga-root>/state/quality/loops/*.iterations.jsonl` (Contract A, live once a
//! real loop run writes rows) and `<ga-root>/state/quality/query-embeddings/*.jsonl`
//! (Contract B, proposed — dormant until GA persists per-query vectors).

use std::path::PathBuf;

use ix_duck::{loops, ood};

fn default_ga_root() -> PathBuf {
    if let Ok(root) = std::env::var("GA_ROOT") {
        return PathBuf::from(root);
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let ix_root = cwd.canonicalize().unwrap_or(cwd);
    ix_root.parent().map(|p| p.join("ga")).unwrap_or_else(|| PathBuf::from("../ga"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ga_root = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(default_ga_root);
    if ga_root.to_string_lossy().contains("://") {
        eprintln!("refusing non-local path: {}", ga_root.display());
        std::process::exit(2);
    }

    let conn = ix_duck::open_bench()?;
    let loops_dir = ga_root.join("state/quality/loops");
    let emb_dir = ga_root.join("state/quality/query-embeddings");
    println!("ga root: {}\n", ga_root.display());

    // --- Loop-iteration lens (Contract A) ---
    let n = loops::build_loop_iterations(&conn, &loops_dir)?;
    println!("== loops ==  {n} iteration row(s) from {}", loops_dir.display());
    if n == 0 {
        println!("  (no *.iterations.jsonl — directory absent or seed-only)\n");
    } else {
        println!("\n  per-loop convergence:");
        for s in loops::loop_summary(&conn)? {
            println!(
                "    {:<22} {:<10} iters={:<3} net={:+.3}  {}",
                s.loop_id, s.domain, s.iterations, s.net_delta, s.final_verdict
            );
        }
        println!("\n  recurring worst-items (≥2):");
        let rec = loops::recurring_worst_items(&conn, 2)?;
        if rec.is_empty() {
            println!("    (none)");
        }
        for (item, occ, lps) in rec {
            println!("    {occ:>3}×  loops={lps:<3}  {item}");
        }
        println!("\n  oscillating loops (improve↔regress):");
        let osc = loops::oscillating_loops(&conn)?;
        if osc.is_empty() {
            println!("    (none)");
        }
        for (id, up, down) in osc {
            println!("    {id:<22} +{up} / -{down}");
        }
        println!();
    }

    // --- OOD query lens (Contract B, proposed) ---
    let m = ood::build_query_embeddings(&conn, &emb_dir)?;
    println!("== ood ==  {m} query embedding(s) from {}", emb_dir.display());
    if m == 0 {
        println!("  (no query-embeddings/*.jsonl — Contract B not yet emitting)");
        return Ok(());
    }
    println!("\n  out-of-domain queries (mean top-3 cosine < 0.5):");
    let flagged = ood::flag_ood(&conn, 3, 0.5)?;
    if flagged.is_empty() {
        println!("    (none)");
    }
    for (qid, intent, score) in flagged {
        println!("    {score:.3}  {intent:<14}  {qid}");
    }
    Ok(())
}
