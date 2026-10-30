//! Distil PR lifecycle history into Seldon Markov transitions (Demerzel#596).
//!
//! ```text
//! bash scripts/fetch-pr-lifecycle.sh > prs.jsonl
//! cargo run -p memristive-markov --example work_history -- prs.jsonl
//! ```
//!
//! Optional trailing args: `stale_days train_fraction max_order min_observations`
//! (defaults `14 0.8 3 5`). Prints one JSON report on stdout.

use memristive_markov::work_history::{parse_jsonl, report};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: work_history <prs.jsonl> [stale_days train_fraction max_order min_obs]");
        std::process::exit(2);
    };
    let arg = |i: usize, default: &str| args.get(i).cloned().unwrap_or_else(|| default.into());
    let stale_days: i64 = arg(1, "14").parse().expect("stale_days");
    let train_fraction: f64 = arg(2, "0.8").parse().expect("train_fraction");
    let max_order: usize = arg(3, "3").parse().expect("max_order");
    let min_obs: usize = arg(4, "5").parse().expect("min_observations");

    let text = std::fs::read_to_string(path).expect("read history");
    let prs = parse_jsonl(&text).expect("parse history");
    let out = report(&prs, stale_days, train_fraction, max_order, min_obs).expect("report");
    println!("{}", serde_json::to_string_pretty(&out).expect("serialize"));
}
