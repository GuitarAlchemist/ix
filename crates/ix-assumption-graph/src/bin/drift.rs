//! Drift gate for `@ai:` annotations: snapshot the workspace's production
//! claims, and on a later run report whether the code under each claim changed
//! or a cited test vanished. See [`ix_assumption_graph::drift`].
//!
//! ```text
//! ix-assumption-graph-drift --snapshot PATH [--workspace DIR]   # capture baseline
//! ix-assumption-graph-drift --check    PATH [--workspace DIR]   # compare; exit 2 on drift
//! ```
//!
//! `--check` exits non-zero when a claim needs human re-confirmation (the
//! annotated code drifted, or a `src:test_*` binding is broken), so it can gate
//! CI. Moves, additions, and verdict revisions are reported but do not fail.

use std::path::{Path, PathBuf};
use std::process::exit;

use ix_assumption_graph::drift::{self, DriftItem, Snapshot};

fn main() {
    let mut workspace = ".".to_string();
    let mut snapshot_out: Option<String> = None;
    let mut check_in: Option<String> = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--workspace" => workspace = take(&args, &mut i),
            "--snapshot" => snapshot_out = Some(take(&args, &mut i)),
            "--check" => check_in = Some(take(&args, &mut i)),
            "-h" | "--help" => {
                println!("ix-assumption-graph-drift --snapshot PATH | --check PATH [--workspace DIR]");
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                exit(1);
            }
        }
        i += 1;
    }

    let ws = PathBuf::from(&workspace);

    if let Some(out) = snapshot_out {
        let snap = snapshot_or_die(&ws);
        write_snapshot(&snap, Path::new(&out));
        println!("captured {} claims -> {out}", snap.claims.len());
        return;
    }

    let Some(baseline_path) = check_in else {
        eprintln!("error: pass --snapshot PATH or --check PATH");
        exit(1);
    };

    let baseline = read_snapshot(Path::new(&baseline_path));
    let current = snapshot_or_die(&ws);
    let mut report = drift::diff(&baseline, &current);
    drift::verify_bindings(&ws, &current, &mut report);

    println!("assumption-graph drift check ({} claims)", current.claims.len());
    println!(
        "  unchanged {}  moved {}  added {}  removed {}  verdict-revised {}",
        report.unchanged,
        report.moved.len(),
        report.added.len(),
        report.removed.len(),
        report.verdict_changed.len()
    );
    print_items("↻ verdict revised", &report.verdict_changed);
    print_items("→ moved", &report.moved);
    print_items("⚠ SPAN DRIFT (re-confirm)", &report.span_drifted);
    print_items("⚠ BROKEN TEST BINDING", &report.broken_bindings);

    if report.is_clean() {
        println!("  ✓ no claim needs re-confirmation");
    } else {
        println!(
            "  ✗ {} span-drift + {} broken-binding finding(s) — re-confirm and re-snapshot",
            report.span_drifted.len(),
            report.broken_bindings.len()
        );
        exit(2);
    }
}

fn print_items(label: &str, items: &[DriftItem]) {
    for it in items {
        println!("  {label}: {}:{}  {}  — {}", it.path, it.line, it.detail, it.claim);
    }
}

fn snapshot_or_die(ws: &Path) -> Snapshot {
    drift::snapshot(ws).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        exit(1);
    })
}

fn write_snapshot(snap: &Snapshot, path: &Path) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let json = serde_json::to_string_pretty(snap).expect("snapshot serializes");
    if let Err(e) = std::fs::write(path, json) {
        eprintln!("error writing {}: {e}", path.display());
        exit(1);
    }
}

fn read_snapshot(path: &Path) -> Snapshot {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error reading baseline {}: {e}", path.display());
        exit(1);
    });
    serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("error parsing baseline {}: {e}", path.display());
        exit(1);
    })
}

fn take(args: &[String], i: &mut usize) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("missing value for flag");
        exit(1);
    })
}
