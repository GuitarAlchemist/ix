//! Autonomous-loop runner — one turn of the longitudinal belief loop.
//!
//! Ingests the workspace's `@ai:` annotations plus (optionally) research claims
//! from a JSON file, loads the persistent belief log, calls
//! [`ix_assumption_graph::AssumptionGraph::revise`], and writes the log back.
//! Scheduling this on an interval (and producing the research-claims file from a
//! `/deep-research` run) is the `assumption-graph-loop` skill's job.
//!
//! ```text
//! ix-assumption-graph-loop [--workspace DIR] [--research FILE.json]
//!     [--log FILE.jsonl] [--trigger LABEL]
//! ```
//! Defaults: workspace `.`, log `state/assumptions/belief-events.jsonl`,
//! trigger `loop`.

use std::error::Error;
use std::fs;
use std::path::Path;

use chrono::Utc;
use ix_assumption_graph::{AssumptionGraph, BeliefLog, ResearchClaim};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut workspace = ".".to_string();
    let mut research_path: Option<String> = None;
    let mut log_path = "state/assumptions/belief-events.jsonl".to_string();
    let mut trigger = "loop".to_string();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--workspace" => workspace = next(&args, &mut i)?,
            "--research" => research_path = Some(next(&args, &mut i)?),
            "--log" => log_path = next(&args, &mut i)?,
            "--trigger" => trigger = next(&args, &mut i)?,
            "-h" | "--help" => {
                println!("ix-assumption-graph-loop [--workspace DIR] [--research FILE.json] [--log FILE.jsonl] [--trigger LABEL]");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    let research: Vec<ResearchClaim> = match &research_path {
        Some(p) => serde_json::from_str(&fs::read_to_string(p)?)?,
        None => Vec::new(),
    };

    let graph = AssumptionGraph::from_workspace_with_research(Path::new(&workspace), research)?;

    let mut log = match fs::read_to_string(&log_path) {
        Ok(s) => BeliefLog::from_jsonl(&s)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => BeliefLog::new(),
        Err(e) => return Err(e.into()),
    };

    let appended = graph.revise(&mut log, Utc::now(), &trigger)?;

    if let Some(parent) = Path::new(&log_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&log_path, log.to_jsonl())?;

    println!("assumption-graph loop ({trigger})");
    println!("  nodes:          {}", graph.node_count());
    println!(
        "  belief events:  {} (+{} this run)",
        log.len(),
        appended.len()
    );
    for e in &appended {
        let from = e
            .from_truth_value
            .map(|h| h.to_string())
            .unwrap_or_else(|| "-".to_string());
        let short = &e.node_id[..e.node_id.len().min(19)];
        println!("  ↻ {short}… : {from} -> {}  [{}]", e.to_truth_value, e.trigger);
    }
    Ok(())
}

fn next(args: &[String], i: &mut usize) -> Result<String, Box<dyn Error>> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| Box::<dyn Error>::from("missing value for flag"))
}
