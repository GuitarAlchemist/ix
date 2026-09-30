//! Measure ACO on TSPLIB berlin52 (optimum 7542) over several seeds.
//!
//! ```text
//! cargo run -p ix-optimize --release --example aco_berlin52
//! ```

#[path = "../tests/tsplib/mod.rs"]
mod tsplib;

use std::time::Instant;

use ix_optimize::aco::{tour_length, AntColony};
use tsplib::{euc_2d_distances, tour, BERLIN52_OPTIMUM, BERLIN52_OPT_TOUR, BERLIN52_TSP};

fn main() {
    let d = euc_2d_distances(BERLIN52_TSP);
    // Sanity check of the fixture before measuring against it.
    assert_eq!(tour_length(&d, &tour(BERLIN52_OPT_TOUR)), BERLIN52_OPTIMUM);
    let seeds: Vec<u64> = (1..=10).collect();

    let configs: Vec<(&str, AntColony)> = vec![("ant-system (defaults)", AntColony::new())];

    println!(
        "berlin52, optimum {BERLIN52_OPTIMUM}, {} seeds",
        seeds.len()
    );
    println!(
        "{:<28} {:>9} {:>9} {:>9} {:>6} {:>10}",
        "config", "mean gap", "best gap", "worst gap", "opt", "ms/run"
    );
    for (name, config) in configs {
        let mut gaps = Vec::new();
        let start = Instant::now();
        for &seed in &seeds {
            let result = config.clone().with_seed(seed).solve_tsp(&d);
            gaps.push(result.length / BERLIN52_OPTIMUM - 1.0);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / seeds.len() as f64;
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let best = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let hits = gaps.iter().filter(|&&g| g.abs() < 1e-12).count();
        println!(
            "{:<28} {:>8.2}% {:>8.2}% {:>8.2}% {:>3}/{:<2} {:>10.1}",
            name,
            mean * 100.0,
            best * 100.0,
            worst * 100.0,
            hits,
            seeds.len(),
            ms
        );
    }
}
