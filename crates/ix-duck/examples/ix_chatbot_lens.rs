//! `ix_chatbot_lens` — the chatbot flight recorder CLI (Slice A + B). Thin shell over
//! `ix_duck::chatbot`: arg-parse, print, write the contract, set the exit code. All SQL
//! and gate logic live in the library (so `lens` and `check` share one builder).
//!
//! Reads a *corpus dir* containing `golden-traces/<id>/{run-*.json,_signature.json}` —
//! defaults to the sibling `../ga/state/quality/chatbot-qa`; pass one explicitly for
//! vendored fixtures. Routed through `ix_duck` so the build-script link directives
//! (rstrtmgr on Windows) reach this example binary.
//!
//!   cargo run -p ix-duck --features duck --example ix_chatbot_lens                 # lens, live ga
//!   cargo run -p ix-duck --features duck --example ix_chatbot_lens -- lens <corpus>
//!   cargo run -p ix-duck --features duck --example ix_chatbot_lens -- check <corpus>

use std::path::PathBuf;

use ix_duck::chatbot;

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
    let cmd = args.next().unwrap_or_else(|| "lens".to_string());
    let corpus = args.next().map(PathBuf::from).unwrap_or_else(default_corpus);

    // Path safety: this lens only reads local directories.
    if corpus.to_string_lossy().contains("://") {
        eprintln!("refusing non-local corpus path: {}", corpus.display());
        std::process::exit(2);
    }

    let conn = ix_duck::open_bench()?;
    println!("corpus: {}\n", corpus.display());

    match cmd.as_str() {
        "lens" => {
            let n = chatbot::build_traces(&conn, &corpus)?;
            if n == 0 {
                println!("(no traces — corpus absent or empty; nothing to analyze)");
                return Ok(());
            }
            println!("{n} traces\n");

            let (ungr, total) = chatbot::ungrounded(&conn)?;
            println!("ungrounded answers: {ungr}/{total}");

            println!("\nrouting-method share:");
            for (m, c) in chatbot::routing_method_share(&conn)? {
                println!("  {c:>4}  {m}");
            }

            println!("\nweak intents (mean confidence < 0.7):");
            let weak = chatbot::weak_intents(&conn, 0.7)?;
            if weak.is_empty() {
                println!("  (none)");
            }
            for (a, conf, c) in weak {
                println!("  {conf:.3}  n={c:<3}  {a}");
            }

            println!("\nlatency outliers (top 5 by elapsed_ms):");
            for (p, a, ms) in chatbot::latency_outliers(&conn, 5)? {
                let ms = ms.map(|v| v.to_string()).unwrap_or_else(|| "?".into());
                let a = a.unwrap_or_else(|| "<none>".into());
                println!("  {ms:>6}ms  {a:<24}  {p}");
            }

            let g = chatbot::grounding_report(&conn, &corpus)?;
            let (grounded, gtotal, valid, invalid, unvalidated) = chatbot::grounding_summary(&g);
            println!("\ngrounding quality:");
            println!(
                "  coverage: {grounded}/{gtotal} have facts  |  IX-validated: {valid} valid, \
                 {invalid} INVALID, {unvalidated} unvalidated"
            );
            for a in g.iter().filter(|a| a.validation == "invalid") {
                println!("  HALLUCINATED FACT  {}  ({}): {}", a.prompt_id, a.query_type, a.detail);
            }
        }
        "check" => {
            let report = chatbot::check_regressions(&conn, &corpus)?;
            let out = PathBuf::from("state/quality/analytics/chatbot-trace-regressions.json");
            if let Err(e) = chatbot::write_contract(&report, &out) {
                eprintln!("warning: could not write contract to {}: {e}", out.display());
            } else {
                println!("contract → {}", out.display());
            }
            println!(
                "status: {}  ({} prompts, {} regression(s)){}",
                report.status,
                report.prompts_checked,
                report.regressions.len(),
                report.degraded_reason.as_deref().map(|r| format!("  — {r}")).unwrap_or_default(),
            );
            for r in &report.regressions {
                println!(
                    "  REGRESSION {}: expected {:?} got {:?}",
                    r.prompt_id, r.expected, r.actual
                );
            }
            std::process::exit(chatbot::exit_code(&report.status));
        }
        other => {
            eprintln!("unknown command '{other}' (expected: lens | check)");
            std::process::exit(2);
        }
    }
    Ok(())
}
