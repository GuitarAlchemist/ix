//! ACO on TSPLIB instances with a proven optimum (ix#362).

mod tsplib;

use ix_optimize::aco::{tour_length, AntColony};
use tsplib::{euc_2d_distances, tour, Instance, BERLIN52, KROA100};

/// Binds each fixture: the published optimal tour must measure exactly the
/// published optimum under the TSPLIB rounding, or the coordinates or the
/// distance rule are wrong.
#[test]
fn test_fixture_optimal_tours_measure_their_optimum() {
    for instance in [BERLIN52, KROA100] {
        let d = euc_2d_distances(instance.tsp);
        let optimal = tour(instance.opt_tour);
        assert_eq!(optimal.len(), d.nrows(), "{}", instance.name);
        assert_eq!(
            tour_length(&d, &optimal),
            instance.optimum,
            "{}",
            instance.name
        );
    }
}

fn gap(colony: AntColony, instance: &Instance) -> f64 {
    let result = colony
        .with_seed(42)
        .solve_tsp(&euc_2d_distances(instance.tsp));
    assert!(result.length >= instance.optimum, "{}", instance.name);
    result.length / instance.optimum - 1.0
}

// Guardrails, not targets: each bound sits well above what seeds 1-10 measured
// in examples/aco_tsplib.rs, so a regression trips it but RNG noise does not.

#[test]
fn test_ant_system_gap_on_berlin52() {
    // Measured: mean 1.68%, worst 3.24%.
    let g = gap(AntColony::new(), &BERLIN52);
    assert!(g <= 0.05, "gap {:.2}%", g * 100.0);
}

#[test]
fn test_max_min_2opt_gap_on_kroa100() {
    // Measured at 50 iterations: the optimum on all ten seeds.
    let g = gap(AntColony::max_min_2opt().with_max_iterations(50), &KROA100);
    assert!(g <= 0.005, "gap {:.2}%", g * 100.0);
}
