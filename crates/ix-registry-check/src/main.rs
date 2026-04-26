//! `ix-registry-check` — Buf-style breaking-change detector for
//! `governance/demerzel/schemas/capability-registry.json`.
//!
//! # Usage
//!
//! ```bash
//! ix-registry-check --old <path> --new <path> [--allowlist <path>]
//! ```
//!
//! Exit codes:
//! - `0` — no breaking changes (or all allowed by allowlist)
//! - `1` — one or more breaking changes found
//! - `2` — usage or I/O error

use ix_registry_check::{compare, load, Finding, Severity};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::ExitCode;

fn print_usage(program: &str) {
    eprintln!("usage: {program} --old <path> --new <path> [--allowlist <path>] [--json]");
    eprintln!();
    eprintln!("Compares two versions of capability-registry.json and");
    eprintln!("fails (exit 1) on any breaking change not in the allowlist.");
    eprintln!();
    eprintln!("Allowlist format: one path per line, e.g.");
    eprintln!("    repos.ix.tools.sequence.ix_markov");
    eprintln!("    repos.tars.tools.grammar");
    eprintln!("Lines beginning with # are comments.");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "ix-registry-check".into());

    let mut old: Option<PathBuf> = None;
    let mut new: Option<PathBuf> = None;
    let mut allowlist: Option<PathBuf> = None;
    let mut json_out = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--old" => {
                i += 1;
                old = args.get(i).map(PathBuf::from);
            }
            "--new" => {
                i += 1;
                new = args.get(i).map(PathBuf::from);
            }
            "--allowlist" => {
                i += 1;
                allowlist = args.get(i).map(PathBuf::from);
            }
            "--json" => json_out = true,
            "-h" | "--help" => {
                print_usage(&program);
                return ExitCode::from(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_usage(&program);
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let (old_path, new_path) = match (old, new) {
        (Some(o), Some(n)) => (o, n),
        _ => {
            print_usage(&program);
            return ExitCode::from(2);
        }
    };

    let old_reg = match load(&old_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to load old registry: {e}");
            return ExitCode::from(2);
        }
    };
    let new_reg = match load(&new_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to load new registry: {e}");
            return ExitCode::from(2);
        }
    };

    let allowed: BTreeSet<String> = match allowlist.as_ref() {
        Some(p) => match std::fs::read_to_string(p) {
            Ok(s) => s
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(String::from)
                .collect(),
            Err(e) => {
                eprintln!("failed to read allowlist {}: {e}", p.display());
                return ExitCode::from(2);
            }
        },
        None => BTreeSet::new(),
    };

    let findings = compare(&old_reg, &new_reg);

    // Partition findings: allowed breaking vs unallowed breaking vs non-breaking.
    let (allowed_breaking, unallowed_breaking, non_breaking): (
        Vec<&Finding>,
        Vec<&Finding>,
        Vec<&Finding>,
    ) = findings.iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut al, mut un, mut nb), f| {
            match f.severity {
                Severity::Breaking => {
                    if allowed.contains(&f.path) {
                        al.push(f);
                    } else {
                        un.push(f);
                    }
                }
                _ => nb.push(f),
            }
            (al, un, nb)
        },
    );

    if json_out {
        let report = serde_json::json!({
            "summary": {
                "total": findings.len(),
                "breaking": unallowed_breaking.len() + allowed_breaking.len(),
                "breaking_allowlisted": allowed_breaking.len(),
                "breaking_unallowlisted": unallowed_breaking.len(),
                "compatible": non_breaking.iter().filter(|f| matches!(f.severity, Severity::Compatible)).count(),
                "informational": non_breaking.iter().filter(|f| matches!(f.severity, Severity::Informational)).count(),
            },
            "findings": findings,
            "allowlisted": allowed_breaking,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        if !unallowed_breaking.is_empty() {
            eprintln!("BREAKING CHANGES ({} unallowed):", unallowed_breaking.len());
            for f in &unallowed_breaking {
                eprintln!("  [BREAK] {} — {}", f.path, f.message);
            }
            eprintln!();
        }
        if !allowed_breaking.is_empty() {
            eprintln!(
                "Breaking changes ALLOWED by allowlist ({}):",
                allowed_breaking.len()
            );
            for f in &allowed_breaking {
                eprintln!("  [ALLOW] {} — {}", f.path, f.message);
            }
            eprintln!();
        }
        if !non_breaking.is_empty() {
            println!("Non-breaking changes ({}):", non_breaking.len());
            for f in &non_breaking {
                let tag = match f.severity {
                    Severity::Compatible => "ADD",
                    Severity::Informational => "INFO",
                    Severity::Breaking => unreachable!(),
                };
                println!("  [{tag}] {} — {}", f.path, f.message);
            }
        }
        if findings.is_empty() {
            println!("No differences.");
        }
    }

    if unallowed_breaking.is_empty() {
        ExitCode::from(0)
    } else {
        ExitCode::from(1)
    }
}
