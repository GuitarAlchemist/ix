//! `ix-loop`: run a loop, list what loops proposed, accept a proposal.
//!
//! ```text
//! ix-loop run <script.lua> [--calls N] [--fuel N] [--seconds N] [--voice] [--github [--gh-owner O]...] [--wasm]
//! ix-loop proposals
//! ix-loop accept <id>
//! ```
//!
//! The store is `IX_LOOP_STORE`, else `~/.cache/ix-loop`. Claude Code runs in that folder, and
//! each call, child run, proposal and judgement is a line of its `ledger.jsonl`.
//!
//! `--voice` lets loops speak through the local voice service (`IX_VOICE_URL`, else
//! 127.0.0.1:59126). `--github` lets them read GitHub through `gh`, for the owners named by
//! `--gh-owner` (GuitarAlchemist when none is named). `--wasm` lets them build and run stages in
//! other languages (Go through `IX_GO`, else `go` on `PATH`), cached under `<store>/wasm`.

use ix_loop::{ClaudeCode, GhCli, Grant, Host, Ledger, Stages, Store, VoiceService};
use serde_json::json;
use std::process::ExitCode;
use std::rc::Rc;

const USAGE: &str =
    "usage: ix-loop run <script.lua> [--calls N] [--fuel N] [--seconds N] [--voice] \
                     [--github [--gh-owner O]...] [--wasm] | proposals | accept <id>";

/// What `run` was asked for.
struct RunFlags {
    grant: Grant,
    voice: bool,
    github: bool,
    owners: Vec<String>,
    wasm: bool,
}

fn run_flags(flags: &[String]) -> Result<RunFlags, String> {
    let mut f = RunFlags {
        grant: Grant {
            calls: 20,
            fuel: 50_000_000,
            seconds: 900.0,
        },
        voice: false,
        github: false,
        owners: Vec::new(),
        wasm: false,
    };
    let mut it = flags.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--voice" => f.voice = true,
            "--github" => f.github = true,
            "--wasm" => f.wasm = true,
            "--gh-owner" => f
                .owners
                .push(it.next().ok_or("--gh-owner needs an owner")?.to_string()),
            "--calls" | "--fuel" | "--seconds" => {
                let value = it
                    .next()
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or_else(|| format!("{flag} needs a number"))?;
                match flag.as_str() {
                    "--calls" => f.grant.calls = value as i64,
                    "--fuel" => f.grant.fuel = value as i64,
                    _ => f.grant.seconds = value,
                }
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if !f.owners.is_empty() && !f.github {
        return Err("--gh-owner needs --github".into());
    }
    if f.github && f.owners.is_empty() {
        f.owners.push("GuitarAlchemist".into());
    }
    Ok(f)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let store = Store::at(Store::default_root());
    match args.first().map(String::as_str) {
        Some("run") if args.len() >= 2 => {
            let flags = match run_flags(&args[2..]) {
                Ok(f) => f,
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
            let stages = store.root().join("wasm");
            let mut host = Host::new(node, store).with_ledger(ledger);
            if flags.voice {
                match ix_voice::Client::from_env() {
                    Ok(c) => host = host.with_voice(Box::new(VoiceService(c))),
                    Err(e) => {
                        eprintln!("--voice: {e}");
                        return ExitCode::from(2);
                    }
                }
            }
            if flags.github {
                let gh = GhCli {
                    bin: std::env::var_os("GH_BIN").map_or_else(|| "gh".into(), Into::into),
                };
                host = host.with_github(Box::new(gh), flags.owners);
            }
            if flags.wasm {
                let go = std::env::var_os("IX_GO").unwrap_or_else(|| "go".into());
                host = host.with_wasm(Stages::new(stages, go));
            }
            let host = Rc::new(host);
            let out = host.run(&src, &args[1], &flags.grant);
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
