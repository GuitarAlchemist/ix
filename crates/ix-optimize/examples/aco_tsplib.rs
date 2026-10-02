//! Measure ACO on TSPLIB instances with a known optimum, over several seeds.
//!
//! ```text
//! cargo run -p ix-optimize --release --example aco_tsplib
//! ```

#[path = "../tests/tsplib/mod.rs"]
mod tsplib;

use std::time::Instant;

use ix_optimize::aco::{tour_length, AntColony};
use tsplib::{euc_2d_distances, tour, BERLIN52, KROA100};

fn main() {
    let seeds: Vec<u64> = (1..=10).collect();

    // One ant per city unless the preset says otherwise (max_min_2opt: 25).
    let mut configs: Vec<(String, AntColony)> = Vec::new();
    for it in [200, 1000] {
        let budget = |colony: AntColony| colony.with_max_iterations(it);
        configs.push((format!("ant-system, {it} it"), budget(AntColony::new())));
        configs.push((format!("max-min, {it} it"), budget(AntColony::max_min())));
    }
    for it in [10, 25, 50, 100] {
        let budget = |colony: AntColony| colony.with_max_iterations(it);
        let ls = |colony: AntColony| budget(colony).with_local_search(true);
        configs.push((format!("ant-system + 2-opt, {it} it"), ls(AntColony::new())));
        // Ablation: MAX-MIN's no-local-search settings (rho 0.02) with 2-opt.
        configs.push((
            format!("max-min rho .02 + 2-opt, {it} it"),
            ls(AntColony::max_min()),
        ));
        configs.push((
            format!("max_min_2opt, {it} it"),
            budget(AntColony::max_min_2opt()),
        ));
    }

    for instance in [BERLIN52, KROA100] {
        let d = euc_2d_distances(instance.tsp);
        // Sanity check of the fixture before measuring against it.
        assert_eq!(tour_length(&d, &tour(instance.opt_tour)), instance.optimum);

        println!();
        println!(
            "{}: {} cities, optimum {}, {} seeds",
            instance.name,
            d.nrows(),
            instance.optimum,
            seeds.len()
        );
        println!(
            "{:<32} {:>9} {:>9} {:>9} {:>6} {:>10}",
            "config", "mean gap", "best gap", "worst gap", "opt", "ms/run"
        );
        for (name, config) in &configs {
            let mut gaps = Vec::new();
            let start = Instant::now();
            for &seed in &seeds {
                let result = config.clone().with_seed(seed).solve_tsp(&d);
                gaps.push(result.length / instance.optimum - 1.0);
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / seeds.len() as f64;
            let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
            let best = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
            let worst = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let hits = gaps.iter().filter(|&&g| g.abs() < 1e-12).count();
            println!(
                "{:<32} {:>8.2}% {:>8.2}% {:>8.2}% {:>3}/{:<2} {:>10.1}",
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
}
