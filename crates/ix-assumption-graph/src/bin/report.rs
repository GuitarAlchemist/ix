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
    println!("  nodes:                  {}", graph.node_count());
    println!(
        "  structural conflicts:   {}  (Phase 1: any opposing polarity)",
        graph.contradictions().len()
    );
    for c in graph.contradictions() {
        println!("    ⚠ \"{}\": {} vs {}", c.claim, c.a_truth.as_str(), c.b_truth.as_str());
    }

    // Phase 2: independence-aware fused verdicts.
    match graph.fuse() {
        Ok(fused) => {
            let escalated: Vec<_> = fused.iter().filter(|f| f.escalated).collect();
            println!("  fused claims:           {}", fused.len());
            println!(
                "  escalated to C:         {}  (Phase 2: independent sources disagree)",
                escalated.len()
            );
            for f in escalated {
                println!(
                    "    ⚠ ESCALATE \"{}\": verdict {} ({} sources, {} contradictions)",
                    f.claim, f.verdict, f.source_count, f.contradiction_count
                );
            }
        }
        Err(e) => eprintln!("fuse error: {e}"),
    }
}
