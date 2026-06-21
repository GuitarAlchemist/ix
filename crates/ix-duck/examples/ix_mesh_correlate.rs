//! `ix_mesh_correlate` — a runnable **mesh of pipelines** over N real-world-style
//! streams, composed entirely on the DuckDB analyst bench with IX UDFs as stages:
//!
//! ```text
//!   N streams ─▶ ix_wavelet_denoise   (per-stream smoothing pipeline)
//!             ─▶ ix_pearson           (pairwise correlation matrix — the mesh)
//!             ─▶ threshold ▶ edge list ▶ ix_connected_components  (incident clusters)
//!             ─▶ ix_centrality        (lead / hub indicator)
//! ```
//!
//! This is the "external data correlation" substrate: each stream is a JSON series
//! (sensor, market tick, log-rate, metric…); the mesh correlates all N×N pairs and
//! distills *which streams move together* (clusters) and *which one drives them*
//! (centrality). DuckDB SQL is the pipeline-declaration + composition language;
//! the IX UDFs are the stages. No production state, no source-of-truth — pure
//! analyst-bench reads, per `docs/DUCKDB.md`.
//!
//! Run: `cargo run -p ix-duck --example ix_mesh_correlate --features duck`
//!
//! The synthetic streams have a KNOWN structure so the result is verifiable:
//! P, Q, R are orthogonal (distinct-frequency) signals; H = mean(P,Q,R) is a
//! latent hub that correlates with each spoke but the spokes do not correlate with
//! each other; D1, D2 are independent distractors. Expected mesh outcome:
//! **cluster {H,P,Q,R}**, D1 and D2 isolated, and **H is the most-central (lead)**.

use std::collections::BTreeMap;

use ix_duck::open_bench;

const N: usize = 64; // samples per stream (power of two → clean wavelet levels)
const TAU: f64 = 0.4; // |r| edge threshold (orthogonal pairs ≈ 0, hub↔spoke ≈ 0.58)

/// Deterministic ±0.05 pseudo-noise, so the demo is reproducible without an RNG.
fn noise(t: usize, seed: u64) -> f64 {
    let mut x = (t as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64 - 0.5) * 0.1
}

/// A pure tone of integer frequency `f` over the window, plus a little noise.
fn tone(f: f64, seed: u64) -> Vec<f64> {
    (0..N)
        .map(|t| (2.0 * std::f64::consts::PI * f * t as f64 / N as f64).sin() + noise(t, seed))
        .collect()
}

fn arr(v: &[f64]) -> String {
    let body: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", body.join(","))
}

// The N×N correlation-matrix fill genuinely indexes corr[i][j]/corr[j][i]/smoothed[j].
#[allow(clippy::needless_range_loop)]
fn main() -> ix_duck::Result<()> {
    let conn = open_bench()?;
    let names = ["H", "P", "Q", "R", "D1", "D2"];

    // ── 1. Build the streams (the "real-world data scenarios") ──────────────────
    let p = tone(1.0, 11);
    let q = tone(2.0, 22);
    let r = tone(3.0, 33);
    let h: Vec<f64> = (0..N).map(|t| (p[t] + q[t] + r[t]) / 3.0 + noise(t, 99)).collect();
    let d1 = tone(5.0, 44);
    let d2 = tone(6.0, 55);
    let raw = [h, p, q, r, d1, d2];

    println!("ix mesh — {} streams × {} samples, |r| edge threshold τ = {TAU}\n", names.len(), N);

    // ── 2. Smoothing pipeline: ix_wavelet_denoise per stream ────────────────────
    // Each stream flows through the same wavelet-denoise "pipeline" declared once.
    let smoothed: Vec<Vec<f64>> = raw
        .iter()
        .map(|s| -> ix_duck::Result<Vec<f64>> {
            let sql = format!(
                "SELECT value FROM ix_wavelet_denoise('{}', 2, 0.05) ORDER BY i",
                arr(s)
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map([], |row| row.get::<_, f64>(0))?;
            rows.collect::<Result<Vec<f64>, _>>()
        })
        .collect::<Result<_, _>>()?;
    println!("stage 1 ✓ ix_wavelet_denoise smoothed all {} streams", names.len());

    // ── 3. The mesh: pairwise ix_pearson correlation matrix ─────────────────────
    let mut corr = vec![vec![0.0f64; names.len()]; names.len()];
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..names.len() {
        for j in 0..names.len() {
            if i == j {
                corr[i][j] = 1.0;
                continue;
            }
            if j < i {
                corr[i][j] = corr[j][i];
                continue;
            }
            let rij: f64 = conn.query_row(
                &format!("SELECT ix_pearson({}::DOUBLE[], {}::DOUBLE[])", arr(&smoothed[i]), arr(&smoothed[j])),
                [],
                |row| row.get(0),
            )?;
            corr[i][j] = rij;
            if rij.abs() >= TAU {
                edges.push((i, j));
            }
        }
    }

    print!("\nstage 2 ✓ ix_pearson correlation matrix\n     ");
    for n in &names {
        print!("{n:>6}");
    }
    println!();
    for (i, n) in names.iter().enumerate() {
        print!("{n:>4} ");
        for j in 0..names.len() {
            print!("{:>6.2}", corr[i][j]);
        }
        println!();
    }

    // ── 4. Correlation graph → ix_connected_components (incident clusters) ──────
    // Self-loops guarantee every stream is a node even if it has no strong edge.
    let mut edge_json: Vec<String> = (0..names.len()).map(|i| format!("[{i},{i}]")).collect();
    edge_json.extend(edges.iter().map(|(i, j)| format!("[{i},{j}]")));
    let edge_list = format!("[{}]", edge_json.join(","));

    let mut clusters: BTreeMap<i64, Vec<&str>> = BTreeMap::new();
    {
        let mut stmt = conn.prepare(&format!(
            "SELECT node, component FROM ix_connected_components('{edge_list}') ORDER BY node"
        ))?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
        for row in rows {
            let (node, comp) = row?;
            clusters.entry(comp).or_default().push(names[node as usize]);
        }
    }
    println!("\nstage 3 ✓ ix_connected_components → {} incident clusters:", clusters.len());
    for (c, members) in &clusters {
        println!("   cluster {c}: {{ {} }}", members.join(", "));
    }

    // ── 5. ix_centrality → the lead / hub indicator ─────────────────────────────
    // Betweenness is the hub measure: the latent driver sits on every spoke-to-spoke
    // shortest path. (Eigenvector centrality oscillates on a bipartite star, so it's
    // the wrong lens for hub-and-spoke; it suits dense, mutually-correlated clusters.)
    let mut lead = ("", -1.0f64);
    {
        let mut stmt = conn.prepare(&format!(
            "SELECT node, score FROM ix_centrality('{edge_list}', 'betweenness') ORDER BY score DESC, node"
        ))?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)))?;
        println!("\nstage 4 ✓ ix_centrality (betweenness) — lead indicator ranking:");
        for row in rows {
            let (node, score) = row?;
            let name = names[node as usize];
            println!("   {name:>3}: {score:.3}");
            if score > lead.1 {
                lead = (name, score);
            }
        }
    }

    println!("\n▶ mesh verdict: streams {{H,P,Q,R}} form the correlated cluster; \
              D1/D2 are independent; lead (hub) stream = {}", lead.0);
    Ok(())
}
