//! Run `ix_code_topology` over every crate in this workspace and print the
//! table that `docs/research/2026-09-08-code-topology-over-ix.md` reports.
//!
//! ```text
//! cargo run -p ix-agent --example code_topology_corpus
//! cargo run -p ix-agent --example code_topology_corpus -- --csv
//! ```
//!
//! Deterministic: the walk sorts by path, the module resolution is
//! order-independent, and no timing or randomness enters the output. Re-running
//! it on the same tree reproduces the same numbers, which is what makes it
//! quotable in a research doc.

use std::path::{Path, PathBuf};

use serde_json::json;

/// One crate's row.
struct Row {
    krate: String,
    units: usize,
    edges: usize,
    circuit_rank: i64,
    betti_0: usize,
    betti_1: usize,
    max_persistence: f64,
    total_persistence: f64,
    resolved: u64,
    external: u64,
    ambiguous: u64,
}

fn crates_dir() -> PathBuf {
    // crates/ix-agent -> crates
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ix-agent lives under crates/")
        .to_path_buf()
}

fn main() {
    let csv = std::env::args().any(|a| a == "--csv");
    let root = crates_dir();

    let mut names: Vec<String> = std::fs::read_dir(&root)
        .expect("crates/ is readable")
        .flatten()
        .filter(|e| e.path().join("src").is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();

    let mut rows = Vec::new();
    for krate in names {
        let src = root.join(&krate).join("src");
        let params = json!({ "path": src.display().to_string(), "granularity": "module" });
        let value = match ix_agent::handlers::code_topology(params) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("{krate}: {error}");
                continue;
            }
        };
        let topology = &value["topology"];
        let resolution = &value["resolution"];
        rows.push(Row {
            krate,
            units: value["n_units"].as_u64().unwrap_or(0) as usize,
            edges: value["undirected_edges"].as_u64().unwrap_or(0) as usize,
            circuit_rank: value["circuit_rank"].as_i64().unwrap_or(-1),
            betti_0: topology["betti_0"].as_u64().unwrap_or(0) as usize,
            betti_1: topology["betti_1"].as_u64().unwrap_or(0) as usize,
            max_persistence: topology["max_persistence"].as_f64().unwrap_or(0.0),
            total_persistence: topology["total_persistence"].as_f64().unwrap_or(0.0),
            resolved: resolution["resolved_calls"].as_u64().unwrap_or(0),
            external: resolution["external_calls"].as_u64().unwrap_or(0),
            ambiguous: resolution["ambiguous_calls"].as_u64().unwrap_or(0),
        });
    }

    if csv {
        println!(
            "crate,units,undirected_edges,betti_0,betti_1,circuit_rank,max_persistence,\
             total_persistence,resolved_calls,external_calls,ambiguous_calls"
        );
        for r in &rows {
            println!(
                "{},{},{},{},{},{},{:.4},{:.4},{},{},{}",
                r.krate,
                r.units,
                r.edges,
                r.betti_0,
                r.betti_1,
                r.circuit_rank,
                r.max_persistence,
                r.total_persistence,
                r.resolved,
                r.external,
                r.ambiguous
            );
        }
        return;
    }

    println!(
        "{:<28} {:>5} {:>6} {:>4} {:>4} {:>8} {:>9} {:>8} {:>8} {:>6}",
        "crate", "units", "undEdg", "b0", "b1", "maxPers", "totPers", "resolved", "external", "amb"
    );
    for r in &rows {
        println!(
            "{:<28} {:>5} {:>6} {:>4} {:>4} {:>8.3} {:>9.3} {:>8} {:>8} {:>6}",
            r.krate,
            r.units,
            r.edges,
            r.betti_0,
            r.betti_1,
            r.max_persistence,
            r.total_persistence,
            r.resolved,
            r.external,
            r.ambiguous
        );
    }

    let multi: Vec<&Row> = rows.iter().filter(|r| r.units > 1).collect();
    let tangled = multi.iter().filter(|r| r.betti_1 > 0).count();
    println!(
        "\n{} crates, {} with more than one source file, {} of those carrying at \
         least one undirected cycle.",
        rows.len(),
        multi.len(),
        tangled
    );

    // The identity the research note rests on, checked on every row rather than
    // argued for: with the filtration capped at dimension 1, betti_1 IS the
    // circuit rank. Any crate that breaks it means the claim needs rewriting.
    let broken: Vec<&Row> = rows
        .iter()
        .filter(|r| r.betti_1 as i64 != r.circuit_rank)
        .collect();
    if broken.is_empty() {
        println!(
            "betti_1 == circuit rank (E - V + betti_0) on all {} crates.",
            rows.len()
        );
    } else {
        println!("betti_1 != circuit rank on {} crate(s):", broken.len());
        for r in broken {
            println!(
                "  {} betti_1={} rank={}",
                r.krate, r.betti_1, r.circuit_rank
            );
        }
    }
}
