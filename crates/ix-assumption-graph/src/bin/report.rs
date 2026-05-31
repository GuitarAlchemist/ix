//! Build the assumption graph for a workspace and print a summary.
//!
//! Usage: `ix-assumption-graph-report [workspace_dir]` (defaults to ".").

use std::path::PathBuf;

use ix_assumption_graph::AssumptionGraph;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let graph = match AssumptionGraph::from_workspace(&PathBuf::from(&dir)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    println!("assumption graph for {dir}");
    println!("  nodes:          {}", graph.node_count());
    println!("  contradictions: {}", graph.contradictions().len());

    for c in graph.contradictions() {
        println!(
            "  ⚠ \"{}\": {} vs {}",
            c.claim,
            c.a_truth.as_str(),
            c.b_truth.as_str()
        );
    }
}
