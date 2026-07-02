//! `ix_pipeline_mesh_hub` — correlation + centrality computed **inside** an executable
//! pipeline, via the `mesh_correlate` fan-in skill.
//!
//! Companion to `ix_pipeline_mesh` (the pairwise tier, Option C of the plan): there the
//! hub was read back in the harness, because a spec can't threshold edges at build time.
//! `mesh_correlate` does the |Pearson| ≥ τ thresholding + betweenness centrality at run
//! time inside one executor node (Option A), so the hub is computed *by the pipeline*.
//!
//!   mesh    = mesh_correlate(streams, τ)          → {correlation, centrality, hub, …}
//!   summary = stats({from: mesh.centrality})      → spread of the centrality scores
//!
//! Validation: a planted hub-and-spoke — one hub correlated with K orthogonal spokes
//! (so spokes are mutually uncorrelated); the executed mesh must return the hub.
//!
//! (Why streams are passed to `mesh_correlate` directly rather than through N per-stream
//! stages: the obvious cond, `autocorrelation`, distorts correlation structure — every
//! ACF shares the lag-0 = 1 spike, so unrelated streams correlate. A correlation-preserving
//! identity cond skill would restore the N-per-stream-pipeline shape; see the plan.)
//!
//! Run: `cargo run -p ix-agent --example ix_pipeline_mesh_hub`

use std::collections::{BTreeMap, HashMap};

use ix_agent as _; // force-link the skill registry
use ix_pipeline::executor::{execute, NoCache};
use ix_pipeline::lower::lower;
use ix_pipeline::spec::{PipelineSpec, StageSpec};
use serde_json::{json, Value};

const N: usize = 40;
const SPOKES: usize = 3; // indices 1..=SPOKES; HUB = 0; the rest are distractors
const HUB: usize = 0;
const THRESHOLD: f64 = 0.4;

fn main() {
    let streams = planted_hub_and_spoke();

    // Stage 1: mesh_correlate (the fan-in). Stage 2: stats over its centrality vector —
    // a downstream consumer, proving the mesh output flows on through the pipeline.
    let mut stages: BTreeMap<String, StageSpec> = BTreeMap::new();
    stages.insert(
        "mesh".into(),
        stage("mesh_correlate", json!({ "series": streams, "threshold": THRESHOLD })),
    );
    stages.insert(
        "summary".into(),
        stage("stats", json!({ "data": { "from": "mesh.centrality" } })),
    );
    let spec = PipelineSpec { version: "1".into(), params: BTreeMap::new(), stages, x_editor: Value::Null };

    let dag = lower(&spec).expect("lower");
    println!(
        "executable mesh (in-pipeline centrality) — {} streams → {} nodes, {} levels",
        N,
        dag.node_count(),
        dag.parallel_levels().len()
    );

    let result = execute(&dag, &HashMap::new(), &NoCache).expect("execute");
    println!("✓ executed {} nodes through ix_pipeline::executor\n", result.node_results.len());

    let mesh = result.output("mesh").expect("mesh output");
    let hub = mesh.get("hub").and_then(Value::as_u64).unwrap_or(0) as usize;
    let centrality: Vec<f64> = mesh
        .get("centrality")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
        .unwrap_or_default();
    let mut ranked: Vec<(usize, f64)> = centrality.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("mesh_correlate betweenness (top 5), computed in-pipeline:");
    for (i, c) in ranked.iter().take(5) {
        let tag = if *i == HUB { "  ← planted hub" } else if (1..=SPOKES).contains(i) { "  (spoke)" } else { "" };
        println!("   stream {i:>3}: {c:.2}{tag}");
    }
    // The downstream stage saw the same vector.
    if let Some(s) = result.output("summary") {
        println!("\ndownstream `stats` over the centrality vector: mean={}, max={}", s.get("mean").unwrap_or(&Value::Null), s.get("max").unwrap_or(&Value::Null));
    }
    println!(
        "\nvalidation — mesh node returned hub = {hub}; planted hub = {HUB}  →  {}",
        if hub == HUB { "✓ RECOVERED (centrality computed inside the executed pipeline)" } else { "✗ not recovered" }
    );
}

/// HUB = mean of SPOKES orthogonal tones (correlates ~1/√K with each); the rest are
/// independent pure-noise distractors (uncorrelated → isolated, betweenness 0).
fn planted_hub_and_spoke() -> Vec<Vec<f64>> {
    let len = 256;
    let tone = |freq: f64, seed: u64| -> Vec<f64> {
        (0..len)
            .map(|t| (2.0 * std::f64::consts::PI * freq * t as f64 / len as f64).sin() + noise(t, seed))
            .collect()
    };
    let spokes: Vec<Vec<f64>> = (0..SPOKES).map(|k| tone(2.0 + k as f64, 100 + k as u64)).collect();
    let mut streams: Vec<Vec<f64>> = Vec::with_capacity(N);
    streams.push((0..len).map(|t| spokes.iter().map(|s| s[t]).sum::<f64>() / SPOKES as f64).collect());
    streams.extend(spokes);
    for i in (SPOKES + 1)..N {
        streams.push((0..len).map(|t| noise(t, 9000 + i as u64)).collect());
    }
    streams
}

fn noise(t: usize, seed: u64) -> f64 {
    let mut x = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64 - 0.5) * 0.6
}

fn stage(skill: &str, args: Value) -> StageSpec {
    StageSpec { skill: skill.into(), args, deps: vec![], cache: None }
}
