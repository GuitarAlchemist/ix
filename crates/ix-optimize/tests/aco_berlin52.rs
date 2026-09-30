//! Ant System on TSPLIB berlin52, whose proven optimum is 7542 (ix#362).

mod tsplib;

use ix_optimize::aco::{tour_length, AntColony};
use tsplib::{euc_2d_distances, tour, BERLIN52_OPTIMUM, BERLIN52_OPT_TOUR, BERLIN52_TSP};

/// Binds the fixture: the published optimal tour must measure exactly 7542
/// under the TSPLIB rounding, or the coordinates or the distance rule are wrong.
#[test]
fn test_fixture_optimal_tour_measures_7542() {
    let d = euc_2d_distances(BERLIN52_TSP);
    let optimal = tour(BERLIN52_OPT_TOUR);
    assert_eq!(d.nrows(), 52);
    assert_eq!(optimal.len(), 52);
    assert_eq!(tour_length(&d, &optimal), BERLIN52_OPTIMUM);
}

#[test]
fn test_ant_system_gap_on_berlin52() {
    let d = euc_2d_distances(BERLIN52_TSP);
    let result = AntColony::new().with_seed(42).solve_tsp(&d);
    let gap = result.length / BERLIN52_OPTIMUM - 1.0;
    assert!(result.length >= BERLIN52_OPTIMUM);
    assert!(
        gap <= 0.10,
        "gap {:.2}% (length {})",
        gap * 100.0,
        result.length
    );
}
