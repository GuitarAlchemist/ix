//! `ix-blast-radius` CLI.
//!
//! Reads changed file paths (one per line) from stdin or `--paths-file`,
//! emits the `blast_radius` JSON to stdout. Designed to drop into the
//! `qa-architect-cycle.ixql` Phase 1 step:
//!
//! ```ixql
//! changed_files <- ix.io.cmd("git diff --name-only {{base_sha}} {{sha}}")
//! blast_radius  <- ix.io.cmd("ix-blast-radius", stdin: changed_files)
//! ```

use clap::Parser;
use ix_blast_radius::analyze;
use std::io::{self, Read};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "ix-blast-radius",
    about = "Path-based blast-radius analyzer for the qa-architect-cycle pipeline"
)]
struct Cli {
    /// Read paths from this file instead of stdin (one path per line).
    #[arg(long)]
    paths_file: Option<PathBuf>,
    /// Pretty-print the JSON output.
    #[arg(long)]
    pretty: bool,
}

fn main() {
    let cli = Cli::parse();

    let raw = match cli.paths_file {
        Some(path) => std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!(
                "[ix-blast-radius] cannot read {}: {e}",
                path.display()
            );
            std::process::exit(2);
        }),
        None => {
            let mut buf = String::new();
            io::stdin()
                .lock()
                .read_to_string(&mut buf)
                .unwrap_or_else(|e| {
                    eprintln!("[ix-blast-radius] read stdin failed: {e}");
                    std::process::exit(2);
                });
            buf
        }
    };

    let paths: Vec<String> = raw
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let result = analyze(&paths);
    let out = if cli.pretty {
        serde_json::to_string_pretty(&result)
    } else {
        serde_json::to_string(&result)
    }
    .unwrap_or_else(|e| {
        eprintln!("[ix-blast-radius] serialize failed: {e}");
        std::process::exit(2);
    });
    println!("{out}");
}

