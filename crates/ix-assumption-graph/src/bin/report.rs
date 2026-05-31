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
