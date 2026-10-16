//! `ix-loop`: run a loop, list what loops proposed, accept a proposal.
//!
//! ```text
//! ix-loop run <script.lua> [--calls N] [--fuel N] [--seconds N]
//! ix-loop proposals
//! ix-loop accept <id>
//! ```
//!
//! The store is `IX_LOOP_STORE`, else `~/.cache/ix-loop`. Claude Code runs in that folder, and
//! each call, child run, proposal and judgement is a line of its `ledger.jsonl`.

use ix_loop::{ClaudeCode, Grant, Host, Ledger, Store};
use serde_json::json;
use std::process::ExitCode;
use std::rc::Rc;

const USAGE: &str =
    "usage: ix-loop run <script.lua> [--calls N] [--fuel N] [--seconds N] | proposals | accept <id>";

fn grant(flags: &[String]) -> Result<Grant, String> {
    let mut g = Grant {
        calls: 20,
        fuel: 50_000_000,
        seconds: 900.0,
    };
    let mut it = flags.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .ok_or_else(|| format!("{flag} needs a number"))?;
        match flag.as_str() {
            "--calls" => g.calls = value as i64,
            "--fuel" => g.fuel = value as i64,
            "--seconds" => g.seconds = value,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(g)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let store = Store::at(Store::default_root());
    match args.first().map(String::as_str) {
        Some("run") if args.len() >= 2 => {
            let grant = match grant(&args[2..]) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("{e}\n{USAGE}");
                    return ExitCode::from(2);
                }
            };
            let src = match std::fs::read_to_string(&args[1]) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read {}: {e}", args[1]);
                    return ExitCode::from(2);
                }
            };
            let node = Rc::new(ClaudeCode::new(store.root()));
            let ledger = Ledger::at(store.root().join("ledger.jsonl"));
            let host = Rc::new(Host::new(node, store).with_ledger(ledger));
            let out = host.run(&src, &args[1], &grant);
            let failed = out.value.is_err();
            let summary = json!({
                "result": match out.value { Ok(v) => v, Err(e) => json!({ "error": e }) },
                "calls_used": out.calls_used,
                "fuel_used": out.fuel_used,
                "seconds": out.seconds,
                "notional_usd_at_list_price": out.notional_usd,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&summary).unwrap_or_default()
            );
            if failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Some("proposals") => {
            for doc in store.proposals() {
                println!(
                    "{}  {} ({})  {}",
                    doc["id"].as_str().unwrap_or("?"),
                    doc["name"].as_str().unwrap_or("?"),
                    doc["kind"].as_str().unwrap_or("?"),
                    doc["note"].as_str().unwrap_or("")
                );
            }
            ExitCode::SUCCESS
        }
        Some("accept") if args.len() == 2 => match store.accept(&args[1]) {
            Ok(file) => {
                println!("accepted {} as {}", args[1], file.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
