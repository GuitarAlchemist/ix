//! `ix_routing_lens` — routing-quality trend over GA's `routing-eval-*.json`.
//! Thin shell over `ix_duck::routing`: which intents are weak, which regressed
//! run-over-run, and the overall accuracy / OOS-decline / margin trend.
//!
//!   cargo run -p ix-duck --features duck --example ix_routing_lens            # live ../ga
//!   cargo run -p ix-duck --features duck --example ix_routing_lens -- <dir>   # explicit quality dir

use std::path::PathBuf;

use ix_duck::routing;

fn default_quality_dir() -> PathBuf {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(default_quality_dir);
    if dir.to_string_lossy().contains("://") {
        eprintln!("refusing non-local path: {}", dir.display());
        std::process::exit(2);
    }

    let conn = ix_duck::open_bench()?;
    println!("quality dir: {}\n", dir.display());

    let n = routing::build_routing_evals(&conn, &dir)?;
    if n == 0 {
        println!("(no routing-eval-*.json — directory absent or empty)");
        return Ok(());
    }
    println!("{n} (run × intent) rows\n");

    println!("overall trend (oldest→newest; — = not recorded in that run):");
    println!("  {:<12} {:>9} {:>12} {:>8}", "day", "accuracy", "oos_decline", "margin");
    let fmt = |v: Option<f64>| v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "—".to_string());
    for (day, acc, oos, margin) in routing::overall_trend(&conn)? {
        println!(
            "  {day:<12} {:>9} {:>12} {:>8}",
            fmt(acc),
            fmt(oos),
            fmt(margin)
        );
    }

    println!("\nweakest intents (latest run, F1 < 0.9):");
    let weak = routing::weakest_intents(&conn, 0.9)?;
    if weak.is_empty() {
        println!("  (none)");
    }
    for (intent, f1, support) in weak {
        println!("  {f1:.3}  n={support:<3}  {intent}");
    }

    println!("\nintent regressions (latest vs previous run):");
    let regs = routing::intent_regressions(&conn)?;
    if regs.is_empty() {
        println!("  (none)");
    }
    for (intent, prev, latest, delta) in regs {
        println!("  {prev:.3} → {latest:.3}  ({delta:+.3})  {intent}");
    }
    Ok(())
}
