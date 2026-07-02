//! `ix_mesh_correlate` — a runnable demo of the **pipeline mesh** correlation
//! substrate (ADR-0004): N real-world-style streams composed on the DuckDB analyst
//! bench with IX UDFs as stages, via the reusable [`ix_duck::mesh`] driver:
//!
//! ```text
//!   N streams ─▶ ix_wavelet_denoise (condition) ─▶ ix_pearson (N×N correlation)
//!             ─▶ threshold ─▶ ix_connected_components (incident clusters)
//!             ─▶ ix_centrality betweenness (lead / hub indicator)
//! ```
//!
//! Run: `cargo run -p ix-duck --example ix_mesh_correlate --features duck`
//!
//! The synthetic streams have a KNOWN hub-and-spoke structure so the result is
//! verifiable: P, Q, R are orthogonal (distinct-frequency) signals; H = mean(P,Q,R)
//! is a latent hub correlated with each spoke (but the spokes are mutually
//! uncorrelated); D1, D2 are independent distractors. Expected mesh outcome:
//! **cluster {H,P,Q,R}**, D1/D2 isolated, **H the lead** (betweenness).

use ix_duck::mesh::{correlate, MeshConfig};
use ix_duck::open_bench;

const N: usize = 64; // samples per stream (power of two → clean wavelet levels)

/// Deterministic ±0.05 pseudo-noise, so the demo is reproducible without an RNG.
fn noise(t: usize, seed: u64) -> f64 {
    let mut x = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
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

fn main() -> ix_duck::Result<()> {
    let conn = open_bench()?;

    // Build the streams (the "real-world data scenarios").
    let p = tone(1.0, 11);
    let q = tone(2.0, 22);
    let r = tone(3.0, 33);
    let h: Vec<f64> = (0..N).map(|t| (p[t] + q[t] + r[t]) / 3.0 + noise(t, 99)).collect();
    let streams = vec![
        ("H".to_string(), h),
        ("P".to_string(), p),
        ("Q".to_string(), q),
        ("R".to_string(), r),
        ("D1".to_string(), tone(5.0, 44)),
        ("D2".to_string(), tone(6.0, 55)),
    ];

    let cfg = MeshConfig::default();
    println!(
        "ix pipeline mesh — {} streams × {N} samples, |r| threshold τ = {}, lead lens = {:?}\n",
        streams.len(),
        cfg.threshold,
        cfg.centrality
    );

    let mesh = correlate(&conn, &streams, &cfg)?;

    // Correlation matrix.
    print!("ix_pearson correlation matrix\n     ");
    for n in &mesh.names {
        print!("{n:>6}");
    }
    println!();
    for (i, name) in mesh.names.iter().enumerate() {
        print!("{name:>4} ");
        for j in 0..mesh.names.len() {
            print!("{:>6.2}", mesh.correlation[i][j]);
        }
        println!();
    }

    // Clusters.
    println!("\nix_connected_components → {} incident clusters:", mesh.clusters.len());
    for (c, members) in mesh.clusters.iter().enumerate() {
        let labels: Vec<&str> = members.iter().map(|&i| mesh.names[i].as_str()).collect();
        println!("   cluster {c}: {{ {} }}", labels.join(", "));
    }

    // Lead indicator.
    println!("\nix_centrality ({:?}) — lead indicator ranking:", cfg.centrality);
    for &(i, score) in &mesh.centrality {
        println!("   {:>3}: {score:.3}", mesh.names[i]);
    }

    println!(
        "\n▶ mesh verdict: {{H,P,Q,R}} correlated; D1/D2 independent; lead (hub) = {}",
        mesh.lead_name().unwrap_or("?")
    );
    Ok(())
}
