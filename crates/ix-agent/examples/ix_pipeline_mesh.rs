//! `ix_pipeline_mesh` — an **executable** pipeline mesh on `ix-pipeline` (phase 2 of
//! docs/plans/2026-06-23-executable-pipeline-mesh.md). Unlike `ix_voicing_mesh` (which
//! *reified* a DAG for display), this builds a real `PipelineSpec`, `lower()`s it to a
//! `Dag<PipelineNode>`, and `execute()`s it through the ix-pipeline executor — every node
//! is a real `ix-agent` skill invocation.
//!
//! Shape (Option C): N per-stream pipelines + a pairwise mesh tier.
//!   cond_i   = autocorrelation(series_i)          [N nodes, level 0]
//!   dist_ij  = distance(cond_i, cond_j, cosine)   [C(N,2) nodes, level 1]
//! N = 16 → **136 real skill nodes executed** (16 + 120). Each `dist_ij` is fed by two
//! `cond` stages via `{from: "cond_i.autocorrelation"}` refs — no glue code.
//!
//! The hub is read back from the executed pairwise outputs (stream with max mean
//! distance). It is *not* a `graph`-pagerank node: pagerank on a complete graph is
//! uniform regardless of weight, and the spec can't threshold edges at build time (the
//! weights are run-time values) — a real constraint surfaced while building. A
//! thresholded/weighted centrality would need a dedicated fan-in skill (Option A in the
//! plan); kept example-local here per Option C.
//!
//! Validation: the streams are 15 mutually-similar tones + 1 planted **outlier**; the
//! executed mesh must recover that outlier as the hub (a ground-truth correctness check).
//!
//! Run: `cargo run -p ix-agent --example ix_pipeline_mesh`

use std::collections::{BTreeMap, HashMap};

// Force-link ix-agent's rlib so its `#[ix_skill]` linkme entries register.
use ix_agent as _;
use ix_pipeline::executor::{execute, NoCache};
use ix_pipeline::lower::lower;
use ix_pipeline::spec::{PipelineSpec, StageSpec};
use serde_json::{json, Value};

/// Streams per mesh — N×(N-1)/2 pairwise stages dominate the node count, so N ≈ 16 lands
/// a ~137-node executed DAG (16 cond + 120 dist + 1 graph).
const N: usize = 16;
/// The planted outlier's stream index. Its signal is unlike the rest, so on the
/// |distance|-weighted graph it should accumulate the most pagerank.
const OUTLIER: usize = N - 1;

fn main() {
    let streams = planted_streams();

    let spec = build_mesh_spec(&streams);
    println!("executable pipeline mesh — {} stages (N={N} streams)", spec.stages.len());

    let dag = lower(&spec).expect("lower");
    let levels = dag.parallel_levels();
    println!(
        "lowered → {} nodes, {} edges, {} levels (widest {} — the pairwise tier)",
        dag.node_count(),
        dag.edge_count(),
        levels.len(),
        levels.iter().map(Vec::len).max().unwrap_or(0)
    );

    let result = execute(&dag, &HashMap::new(), &NoCache).expect("execute");
    println!("✓ executed {} real skill nodes through ix_pipeline::executor (not reified)\n", result.node_results.len());

    // Read the N×N distance matrix from the executed pairwise nodes, then rank streams
    // by mean distance to the rest (an outlier-centrality from the executed mesh — the
    // `graph` skill's pagerank is uniform on a complete graph, so the aggregation is
    // computed here from the pipeline's real outputs).
    let mut mean_dist = [0.0f64; N];
    for i in 0..N {
        for j in (i + 1)..N {
            let d = result
                .output(&format!("dist_{i}_{j}"))
                .and_then(|v| v.get("distance"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            mean_dist[i] += d / (N - 1) as f64;
            mean_dist[j] += d / (N - 1) as f64;
        }
    }
    let mut ranked: Vec<(usize, f64)> = (0..N).map(|i| (i, mean_dist[i])).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("mean cosine distance to the rest, over the executed mesh (top 5):");
    for (i, d) in ranked.iter().take(5) {
        let tag = if *i == OUTLIER { "  ← planted outlier" } else { "" };
        println!("   stream {i:>2}: {d:.4}{tag}");
    }

    // Validation: does the executed mesh recover the planted outlier as the hub?
    let hub = ranked[0].0;
    let pass = hub == OUTLIER;
    println!(
        "\nvalidation — hub = stream {hub}; planted outlier = {OUTLIER}  →  {}",
        if pass { "✓ RECOVERED (the executed mesh found the planted structure)" } else { "✗ not recovered" }
    );
}

/// 15 mutually-similar streams (a shared base tone + per-stream deterministic noise)
/// plus one planted outlier at a very different frequency.
fn planted_streams() -> Vec<Vec<f64>> {
    let len = 24;
    (0..N)
        .map(|i| {
            let freq = if i == OUTLIER { 9.0 } else { 2.0 };
            (0..len)
                .map(|t| {
                    let phase = 2.0 * std::f64::consts::PI * freq * t as f64 / len as f64;
                    phase.sin() + noise(t, i as u64)
                })
                .collect()
        })
        .collect()
}

/// Deterministic ±0.05 pseudo-noise (no RNG → reproducible), per (t, stream).
fn noise(t: usize, seed: u64) -> f64 {
    let mut x = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64 - 0.5) * 0.1
}

/// Generate the executable mesh spec for `streams`: N autocorrelation stages, C(N,2)
/// pairwise distance stages, one graph-pagerank fan-in stage.
fn build_mesh_spec(streams: &[Vec<f64>]) -> PipelineSpec {
    let n = streams.len();
    let mut stages: BTreeMap<String, StageSpec> = BTreeMap::new();

    // Per-stream condition pipelines.
    for (i, s) in streams.iter().enumerate() {
        stages.insert(
            format!("cond_{i}"),
            stage("autocorrelation", json!({ "signal": s })),
        );
    }

    // Pairwise mesh tier: C(N,2) distance stages, each fed by two condition stages via
    // `{from:}` refs. This is the executable mesh — the bulk of the 100+ nodes.
    for i in 0..n {
        for j in (i + 1)..n {
            stages.insert(
                format!("dist_{i}_{j}"),
                stage(
                    "distance",
                    json!({
                        "a": { "from": format!("cond_{i}.autocorrelation") },
                        "b": { "from": format!("cond_{j}.autocorrelation") },
                        "metric": "cosine"
                    }),
                ),
            );
        }
    }

    PipelineSpec {
        version: "1".into(),
        params: BTreeMap::new(),
        stages,
        x_editor: Value::Null,
    }
}

fn stage(skill: &str, args: Value) -> StageSpec {
    StageSpec { skill: skill.into(), args, deps: vec![], cache: None }
}
