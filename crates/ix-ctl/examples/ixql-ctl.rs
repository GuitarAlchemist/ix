//! Runs an IXQL program against an ixctl plane and prints its bindings as JSON.
//!
//! ```text
//! cargo run -p ix-ctl --example ixql-ctl -- crates/ix-ctl/examples/pods.ixql [--state FILE]
//! ```
//!
//! `--state` is the file `ixctl up` writes on ready. It defaults to `local.json` in the plane's
//! default state folder: `%LOCALAPPDATA%\ixctl` on Windows, `$XDG_STATE_HOME/ixctl` or
//! `~/.local/state/ixctl` elsewhere. `ix.io.read` and `ix.io.write` are confined to the
//! current folder.

use ix_ixql::{Executor, FsHost};
use serde_json::Value;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

const USAGE: &str = "usage: ixql-ctl <file.ixql> [--state FILE]";

fn main() -> ExitCode {
    let mut program = None;
    let mut state = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--state" => match args.next() {
                Some(path) => state = Some(PathBuf::from(path)),
                None => return usage("--state needs a file"),
            },
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            flag if flag.starts_with('-') => return usage(&format!("unknown flag {flag}")),
            path => program = Some(PathBuf::from(path)),
        }
    }
    let Some(program) = program else {
        return usage("no program");
    };
    let Some(state) = state.or_else(default_state) else {
        return usage("no --state, and no default state folder on this system");
    };

    let source = match std::fs::read_to_string(&program) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: {e}", program.display());
            return ExitCode::from(66);
        }
    };
    let ctl = match ix_ctl::Ctl::from_state(&state) {
        Ok(ctl) => ctl,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(69);
        }
    };
    let root = std::env::current_dir().expect("a current folder");
    let mut executor = Executor::new(Arc::new(FsHost::new(root)));
    ix_ctl::register(executor.capabilities(), ctl).expect("the ix.ctl names are free");

    match executor.run_source(&source) {
        Ok(outcome) => {
            let bindings: serde_json::Map<String, Value> = outcome.env.into_iter().collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&bindings).expect("JSON values serialize")
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{}: {e}", program.display());
            ExitCode::FAILURE
        }
    }
}

fn usage(why: &str) -> ExitCode {
    eprintln!("{why}\n{USAGE}");
    ExitCode::from(64)
}

/// `<default state folder>/local.json`, the folder `ixctl up` uses without `--state`.
fn default_state() -> Option<PathBuf> {
    let var = |name| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let dir = if cfg!(windows) {
        var("LOCALAPPDATA")?.join("ixctl")
    } else {
        var("XDG_STATE_HOME")
            .or_else(|| var("HOME").map(|h| h.join(".local").join("state")))?
            .join("ixctl")
    };
    Some(dir.join("local.json"))
}
