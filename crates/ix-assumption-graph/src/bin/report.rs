//! Build the assumption graph for a workspace and print a summary.
//!
//! Usage: `ix-assumption-graph-report [workspace_dir] [--prime-radiant-json PATH]`
//! (workspace defaults to "."). With `--prime-radiant-json`, the
//! Prime-Radiant-format node+edge graph is written to PATH (the JSON-on-disk
//! source the ga Prime Radiant viz consumes).

use std::path::PathBuf;

use ix_assumption_graph::{AssumptionGraph, ResearchClaim};

fn main() {
    let mut dir = ".".to_string();
    let mut prime_radiant_json: Option<String> = None;
    let mut html_path: Option<String> = None;
    let mut research_path: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--prime-radiant-json" => match args.next() {
                Some(p) => prime_radiant_json = Some(p),
                None => {
                    eprintln!("error: --prime-radiant-json requires a path");
                    std::process::exit(1);
                }
            },
            "--html" => match args.next() {
                Some(p) => html_path = Some(p),
                None => {
                    eprintln!("error: --html requires a path");
                    std::process::exit(1);
                }
            },
            "--research" => match args.next() {
                Some(p) => research_path = Some(p),
                None => {
                    eprintln!("error: --research requires a path");
                    std::process::exit(1);
                }
            },
            other => dir = other.to_string(),
        }
    }

    let research: Vec<ResearchClaim> = match &research_path {
        Some(p) => match std::fs::read_to_string(p).map(|s| serde_json::from_str(&s)) {
            Ok(Ok(claims)) => claims,
            Ok(Err(e)) => {
                eprintln!("error parsing {p}: {e}");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("error reading {p}: {e}");
                std::process::exit(1);
            }
        },
        None => Vec::new(),
    };

    let build = AssumptionGraph::from_workspace_with_research(&PathBuf::from(&dir), research);
    let graph = match build {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    if let Some(path) = &prime_radiant_json {
        let json = serde_json::to_string_pretty(&graph.prime_radiant_graph())
            .expect("prime_radiant_graph serializes");
        if let Some(parent) = PathBuf::from(path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        if let Err(e) = std::fs::write(path, json) {
            eprintln!("error writing {path}: {e}");
            std::process::exit(1);
        }
        println!("wrote Prime Radiant graph → {path}");
    }

    if let Some(path) = &html_path {
        let json = serde_json::to_string(&graph.prime_radiant_graph())
            .expect("prime_radiant_graph serializes");
        let html = ix_assumption_graph::html::render(&json);
        if let Some(parent) = PathBuf::from(path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        if let Err(e) = std::fs::write(path, html) {
            eprintln!("error writing {path}: {e}");
            std::process::exit(1);
        }
        println!("wrote 2D viewer → {path}  (open in a browser)");
    }

    println!("assumption graph for {dir}");
    println!("  nodes:                  {}", graph.node_count());
    println!(
        "  structural conflicts:   {}  (Phase 1: any opposing polarity)",
        graph.contradictions().len()
    );
    for c in graph.contradictions() {
        println!(
            "    ⚠ \"{}\": {} vs {}",
            c.claim,
            c.a_truth.as_str(),
            c.b_truth.as_str()
        );
    }

    // Phase 4: faceted navigation view (by namespace / kind / domain).
    match graph.view() {
        Ok(view) => {
            println!("  fused claims:           {}", view.claim_count);
            println!("  escalated to C:         {}", view.escalated_count);

            println!("  by namespace:");
            for (ns, claims) in &view.by_namespace {
                let esc = claims.iter().filter(|c| c.escalated).count();
                let flag = if esc > 0 {
                    format!("  ⚠ {esc} escalated")
                } else {
                    String::new()
                };
                println!("    {:<24} {} claims{}", ns, claims.len(), flag);
            }

            print!("  by domain:   ");
            for (d, n) in &view.by_domain {
                print!("{d}={n}  ");
            }
            println!();

            for c in &view.escalations {
                println!(
                    "    ⚠ ESCALATE [{}] \"{}\": verdict {} ({} sources, {} contradictions)",
                    c.namespace, c.claim, c.verdict, c.source_count, c.contradiction_count
                );
            }
        }
        Err(e) => eprintln!("view error: {e}"),
    }
}
