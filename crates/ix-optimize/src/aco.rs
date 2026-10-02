//! Ant Colony Optimization (ACO) for the symmetric travelling salesman problem.
//!
//! Each ant builds a closed tour city by city. From city `i` it moves to an
//! unvisited city `j` with probability proportional to
//! `tau[i][j]^alpha * eta[i][j]^beta`, where `tau` is the pheromone and
//! `eta = 1 / d` the heuristic. After each iteration all pheromone evaporates
//! by `rho`, then ants deposit `1 / L` on the edges of a tour of length `L`:
//!
//! * [`Variant::AntSystem`] (Dorigo, Maniezzo & Colorni, 1996): every ant deposits.
//! * [`Variant::MaxMin`] (Stützle & Hoos, 2000): only the iteration's best ant
//!   deposits, and every pheromone value is kept within `[tau_min, tau_max]`,
//!   so the search neither stagnates on one tour nor forgets good edges.
//!
//! With [`AntColony::with_local_search`], every ant's tour is improved by 2-opt
//! before it is scored and deposits pheromone.
//!
//! CPU, `f64`, seeded: the same seed and inputs give the same tour. This is the
//! oracle a future GPU kernel would be checked against (ix#362).
//!
//! # Example
//!
//! The example of `docs/optimization/ant-colony.md`: a van leaves the depot
//! (shop 0) and visits five shops laid out on a 2 x 1 km grid.
//!
//! ```
//! use ix_optimize::aco::AntColony;
//! use ndarray::Array2;
//!
//! let shops: [(f64, f64); 6] = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (0.0, 1.0)];
//! let km = Array2::from_shape_fn((shops.len(), shops.len()), |(i, j)| {
//!     let (dx, dy) = (shops[i].0 - shops[j].0, shops[i].1 - shops[j].1);
//!     (dx * dx + dy * dy).sqrt()
//! });
//!
//! let route = AntColony::max_min_2opt().with_seed(42).solve_tsp(&km);
//!
//! assert_eq!(route.tour[0], 0); // tours start at the depot
//! assert_eq!(route.length, 6.0); // around the grid's edge: the shortest loop
//! ```

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Floor for a distance or a tour length used as a divisor, so a zero distance
/// (two cities at the same point) yields a large finite heuristic, not infinity.
const MIN_DISTANCE: f64 = 1e-12;

/// MAX-MIN Ant System: probability that a converged colony still builds the
/// best tour, which sets the ratio `tau_max / tau_min` (Stützle & Hoos 2000).
const MAX_MIN_P_BEST: f64 = 0.05;

/// A 2-opt move must shorten the tour by more than this share of the two edges
/// it removes, so floating-point noise cannot make the search cycle.
const TWO_OPT_TOLERANCE: f64 = 1e-12;

/// Which pheromone update rule the colony uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Every ant deposits on its own tour.
    AntSystem,
    /// Only the iteration's best ant deposits; pheromone stays within bounds.
    MaxMin,
}

/// Ant colony configuration.
#[derive(Debug, Clone)]
pub struct AntColony {
    pub variant: Variant,
    /// Ants per iteration; `None` sends one ant per city.
    pub num_ants: Option<usize>,
    pub max_iterations: usize,
    /// Pheromone exponent.
    pub alpha: f64,
    /// Heuristic (`1 / distance`) exponent.
    pub beta: f64,
    /// Share of the pheromone that evaporates each iteration, in `(0, 1]`.
    pub evaporation: f64,
    /// Improve every ant's tour with 2-opt before it deposits.
    pub local_search: bool,
    pub seed: u64,
}

impl Default for AntColony {
    /// The settings Dorigo & Stützle (2004, table 3.3) recommend for Ant System.
    fn default() -> Self {
        Self {
            variant: Variant::AntSystem,
            num_ants: None,
            max_iterations: 200,
            alpha: 1.0,
            beta: 2.0,
            evaporation: 0.5,
            local_search: false,
            seed: 42,
        }
    }
}

/// Result of an ACO run.
#[derive(Debug, Clone)]
pub struct TourResult {
    /// The best tour found: every city exactly once, starting at city 0.
    pub tour: Vec<usize>,
    /// Its closed length, including the edge back to the first city.
    pub length: f64,
    /// The best length found so far, after each iteration.
    pub history: Vec<f64>,
}

impl AntColony {
    pub fn new() -> Self {
        Self::default()
    }

    /// MAX-MIN Ant System without local search, with the evaporation rate
    /// Dorigo & Stützle (2004, table 3.3) recommend for it, 0.02.
    ///
    /// At that rate it converges slowly, so the budget is 1000 iterations. On
    /// berlin52 (seeds 1-10) 200 iterations left a mean gap of 9.62% and 1000
    /// reached the optimum every time (`examples/aco_tsplib.rs`).
    pub fn max_min() -> Self {
        Self {
            variant: Variant::MaxMin,
            evaporation: 0.02,
            max_iterations: 1000,
            ..Self::default()
        }
    }

    /// MAX-MIN Ant System with 2-opt local search and the settings Dorigo &
    /// Stützle (2004, table 3.7) recommend with local search: evaporation 0.2
    /// and 25 ants. The best of the presets measured in `examples/aco_tsplib.rs`:
    /// on kroA100 (seeds 1-10) it reached the optimum every time from 50
    /// iterations.
    pub fn max_min_2opt() -> Self {
        Self {
            variant: Variant::MaxMin,
            evaporation: 0.2,
            num_ants: Some(25),
            local_search: true,
            ..Self::default()
        }
    }

    pub fn with_ants(mut self, n: usize) -> Self {
        self.num_ants = Some(n);
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_evaporation(mut self, rho: f64) -> Self {
        self.evaporation = rho;
        self
    }

    pub fn with_local_search(mut self, on: bool) -> Self {
        self.local_search = on;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Find a short closed tour through every city of `distances`.
    ///
    /// # Panics
    ///
    /// If `distances` is not square, not symmetric, or holds a negative or
    /// non-finite value, if `alpha` or `beta` is not finite, or if the
    /// evaporation rate is outside `(0, 1]`.
    pub fn solve_tsp(&self, distances: &Array2<f64>) -> TourResult {
        validate(distances);
        assert!(
            self.evaporation > 0.0 && self.evaporation <= 1.0,
            "evaporation must be in (0, 1], got {}",
            self.evaporation
        );
        // A NaN exponent makes every transition weight NaN, and the walk would
        // silently fall back to the nearest city at every step.
        assert!(
            self.alpha.is_finite(),
            "alpha must be finite, got {}",
            self.alpha
        );
        assert!(
            self.beta.is_finite(),
            "beta must be finite, got {}",
            self.beta
        );

        let n = distances.nrows();
        if n < 2 {
            return TourResult {
                tour: (0..n).collect(),
                length: 0.0,
                history: Vec::new(),
            };
        }

        // @ai:invariant solve_tsp is reproducible: the same seed, parameters and distance matrix give the same tour and the same best-length history on every run, because all randomness comes from this one seeded StdRng [T:test conf:0.9 src:aco::tests::test_same_seed_same_tour]
        let mut rng = StdRng::seed_from_u64(self.seed);
        let ants = self.num_ants.unwrap_or(n).max(1);

        // eta^beta does not change during the run.
        let heuristic = Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                0.0
            } else {
                (1.0 / distances[[i, j]].max(MIN_DISTANCE)).powf(self.beta)
            }
        });

        // Initial pheromone (Dorigo & Stützle 2004, §3.4.1): m / C_nn for Ant
        // System, and the tau_max estimate 1 / (rho * C_nn) for MAX-MIN.
        let nearest = nearest_neighbour_tour(distances, 0);
        let nearest_length = tour_length(distances, &nearest).max(MIN_DISTANCE);
        let tau0 = match self.variant {
            Variant::AntSystem => ants as f64 / nearest_length,
            Variant::MaxMin => 1.0 / (self.evaporation * nearest_length),
        };
        let mut pheromone = Array2::from_elem((n, n), tau0);

        let mut best_tour = Vec::new();
        let mut best_length = f64::INFINITY;
        let mut history = Vec::with_capacity(self.max_iterations);

        for _ in 0..self.max_iterations {
            let choice = Array2::from_shape_fn((n, n), |(i, j)| {
                pheromone[[i, j]].powf(self.alpha) * heuristic[[i, j]]
            });

            let tours: Vec<(Vec<usize>, f64)> = (0..ants)
                .map(|_| {
                    let mut tour = construct_tour(&choice, distances, &mut rng);
                    if self.local_search {
                        two_opt(distances, &mut tour);
                    }
                    let length = tour_length(distances, &tour);
                    (tour, length)
                })
                .collect();

            for (tour, length) in &tours {
                if *length < best_length {
                    best_length = *length;
                    best_tour = tour.clone();
                }
            }

            pheromone.mapv_inplace(|t| t * (1.0 - self.evaporation));
            match self.variant {
                Variant::AntSystem => {
                    for (tour, length) in &tours {
                        deposit(&mut pheromone, tour, *length);
                    }
                }
                Variant::MaxMin => {
                    let (tour, length) = tours
                        .iter()
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .expect("at least one ant");
                    deposit(&mut pheromone, tour, *length);
                    let (lo, hi) = max_min_bounds(self.evaporation, best_length, n);
                    pheromone.mapv_inplace(|t| t.clamp(lo, hi));
                }
            }

            history.push(best_length);
        }

        if best_tour.is_empty() {
            // max_iterations == 0: no ant ran, so fall back to the greedy tour.
            best_length = tour_length(distances, &nearest);
            best_tour = nearest;
        }
        let start = best_tour.iter().position(|&c| c == 0).unwrap_or(0);
        best_tour.rotate_left(start);

        TourResult {
            tour: best_tour,
            length: best_length,
            history,
        }
    }
}

/// Apply improving 2-opt moves to `tour` until none is left: replace edges
/// `(a, b)` and `(c, d)` by `(a, c)` and `(b, d)`, reversing the path between.
fn two_opt(distances: &Array2<f64>, tour: &mut [usize]) {
    let n = tour.len();
    if n < 4 {
        return;
    }
    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..n - 2 {
            // j = n - 1 with i = 0 would pick two edges that share city tour[0].
            let last = if i == 0 { n - 2 } else { n - 1 };
            for j in i + 2..=last {
                let (a, b) = (tour[i], tour[i + 1]);
                let (c, d) = (tour[j], tour[(j + 1) % n]);
                let removed = distances[[a, b]] + distances[[c, d]];
                let added = distances[[a, c]] + distances[[b, d]];
                if added < removed - TWO_OPT_TOLERANCE * removed {
                    tour[i + 1..=j].reverse();
                    improved = true;
                }
            }
        }
    }
}

/// Add `1 / length` on both directions of every edge of `tour`.
fn deposit(pheromone: &mut Array2<f64>, tour: &[usize], length: f64) {
    let amount = 1.0 / length.max(MIN_DISTANCE);
    for (a, b) in edges(tour) {
        pheromone[[a, b]] += amount;
        pheromone[[b, a]] += amount;
    }
}

/// `(tau_min, tau_max)` for MAX-MIN Ant System (Stützle & Hoos 2000), with
/// `n / 2` as the average number of choices an ant has at each step. On very
/// small instances the formula can put `tau_min` above `tau_max`; it is then
/// capped at `tau_max`.
fn max_min_bounds(rho: f64, best_length: f64, n: usize) -> (f64, f64) {
    let tau_max = 1.0 / (rho * best_length.max(MIN_DISTANCE));
    let p_dec = MAX_MIN_P_BEST.powf(1.0 / n as f64);
    let average_choices = n as f64 / 2.0;
    let tau_min = tau_max * (1.0 - p_dec) / ((average_choices - 1.0) * p_dec);
    (tau_min.min(tau_max), tau_max)
}

/// Closed length of `tour`, including the edge from its last city back to its first.
pub fn tour_length(distances: &Array2<f64>, tour: &[usize]) -> f64 {
    edges(tour).map(|(a, b)| distances[[a, b]]).sum()
}

/// The `n` edges of a closed tour of `n >= 2` cities; none for a shorter one.
fn edges(tour: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    let n = if tour.len() < 2 { 0 } else { tour.len() };
    (0..n).map(move |i| (tour[i], tour[(i + 1) % tour.len()]))
}

/// One ant's tour: a random start, then roulette-wheel choices over `choice`.
fn construct_tour(choice: &Array2<f64>, distances: &Array2<f64>, rng: &mut StdRng) -> Vec<usize> {
    let n = choice.nrows();
    let mut visited = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    let mut current = rng.random_range(0..n);
    visited[current] = true;
    tour.push(current);

    for _ in 1..n {
        let total: f64 = (0..n)
            .filter(|&j| !visited[j])
            .map(|j| choice[[current, j]])
            .sum();
        let next = if total > 0.0 && total.is_finite() {
            let mut r = rng.random::<f64>() * total;
            let mut pick = usize::MAX;
            for j in (0..n).filter(|&j| !visited[j]) {
                pick = j;
                let w = choice[[current, j]];
                if r < w {
                    break;
                }
                r -= w;
            }
            pick
        } else {
            // Every weight underflowed (or overflowed): take the nearest city.
            nearest_unvisited(distances, current, &visited)
        };
        visited[next] = true;
        tour.push(next);
        current = next;
    }
    tour
}

fn nearest_unvisited(distances: &Array2<f64>, from: usize, visited: &[bool]) -> usize {
    (0..distances.nrows())
        .filter(|&j| !visited[j])
        .min_by(|&a, &b| distances[[from, a]].total_cmp(&distances[[from, b]]))
        .expect("an unvisited city remains")
}

/// Greedy tour: always move to the nearest unvisited city.
fn nearest_neighbour_tour(distances: &Array2<f64>, start: usize) -> Vec<usize> {
    let n = distances.nrows();
    let mut visited = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    let mut current = start;
    visited[current] = true;
    tour.push(current);
    for _ in 1..n {
        current = nearest_unvisited(distances, current, &visited);
        visited[current] = true;
        tour.push(current);
    }
    tour
}

fn validate(distances: &Array2<f64>) {
    let n = distances.nrows();
    assert_eq!(
        n,
        distances.ncols(),
        "distance matrix must be square, got {n}x{}",
        distances.ncols()
    );
    for i in 0..n {
        for j in 0..n {
            let d = distances[[i, j]];
            assert!(
                d.is_finite() && d >= 0.0,
                "distance [{i}, {j}] must be finite and non-negative, got {d}"
            );
            assert!(
                d == distances[[j, i]],
                "distance matrix must be symmetric: [{i}, {j}] = {d}, [{j}, {i}] = {}",
                distances[[j, i]]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn euclidean(points: &[(f64, f64)]) -> Array2<f64> {
        let n = points.len();
        Array2::from_shape_fn((n, n), |(i, j)| {
            let (dx, dy) = (points[i].0 - points[j].0, points[i].1 - points[j].1);
            (dx * dx + dy * dy).sqrt()
        })
    }

    /// Seeded pseudo-random cities in a 100 x 100 square.
    fn random_cities(n: usize, seed: u64) -> Vec<(f64, f64)> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)))
            .collect()
    }

    /// Exact optimum by enumerating every tour that starts at city 0.
    fn brute_force_optimum(distances: &Array2<f64>) -> f64 {
        fn go(d: &Array2<f64>, tour: &mut Vec<usize>, used: &mut [bool], best: &mut f64) {
            let n = used.len();
            if tour.len() == n {
                *best = best.min(tour_length(d, tour));
                return;
            }
            for c in 1..n {
                if !used[c] {
                    used[c] = true;
                    tour.push(c);
                    go(d, tour, used, best);
                    tour.pop();
                    used[c] = false;
                }
            }
        }
        let n = distances.nrows();
        let mut used = vec![false; n];
        used[0] = true;
        let mut best = f64::INFINITY;
        go(distances, &mut vec![0], &mut used, &mut best);
        best
    }

    #[test]
    fn test_tour_length_closes_the_loop() {
        let d = euclidean(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        assert_eq!(tour_length(&d, &[0, 1, 2, 3]), 4.0);
        assert_eq!(tour_length(&d, &[0]), 0.0);
        assert_eq!(tour_length(&d, &[]), 0.0);
    }

    // @ai:invariant solve_tsp returns every city exactly once, starting at city 0 [T:test conf:0.9 src:aco::tests::test_tour_is_a_permutation_starting_at_zero]
    #[test]
    fn test_tour_is_a_permutation_starting_at_zero() {
        let d = euclidean(&random_cities(20, 7));
        let result = AntColony::new().with_max_iterations(20).solve_tsp(&d);
        let mut sorted = result.tour.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        assert_eq!(result.tour[0], 0);
        assert!((tour_length(&d, &result.tour) - result.length).abs() < 1e-9);
        assert_eq!(result.history.len(), 20);
        assert!(result.history.windows(2).all(|w| w[1] <= w[0]));
    }

    #[test]
    fn test_finds_the_exact_optimum_on_small_instances() {
        for seed in [1, 2, 3] {
            let d = euclidean(&random_cities(8, seed));
            let optimum = brute_force_optimum(&d);
            let result = AntColony::new().with_max_iterations(50).solve_tsp(&d);
            assert!(
                (result.length - optimum).abs() < 1e-9,
                "seed {seed}: ACO {} vs optimum {optimum}",
                result.length
            );
        }
    }

    #[test]
    fn test_max_min_finds_the_exact_optimum_on_small_instances() {
        for seed in [1, 2, 3] {
            let d = euclidean(&random_cities(8, seed));
            let optimum = brute_force_optimum(&d);
            let result = AntColony::max_min().with_max_iterations(50).solve_tsp(&d);
            assert!(
                (result.length - optimum).abs() < 1e-9,
                "seed {seed}: MAX-MIN {} vs optimum {optimum}",
                result.length
            );
        }
    }

    #[test]
    fn test_max_min_bounds_are_ordered() {
        for n in 3..60 {
            let (lo, hi) = max_min_bounds(0.02, 100.0, n);
            assert!(lo > 0.0 && lo <= hi, "n = {n}: ({lo}, {hi})");
        }
    }

    #[test]
    fn test_two_opt_untangles_a_crossing() {
        // Square visited as 0-2-1-3: its two diagonals cross.
        let d = euclidean(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let mut tour = vec![0, 2, 1, 3];
        two_opt(&d, &mut tour);
        assert_eq!(tour_length(&d, &tour), 4.0);
    }

    // @ai:invariant two_opt never lengthens a tour and keeps it a permutation [T:test conf:0.9 src:aco::tests::test_two_opt_never_lengthens_a_tour]
    #[test]
    fn test_two_opt_never_lengthens_a_tour() {
        let d = euclidean(&random_cities(30, 9));
        let mut rng = StdRng::seed_from_u64(4);
        for _ in 0..20 {
            let mut tour: Vec<usize> = (0..30).collect();
            for i in (1..30).rev() {
                tour.swap(i, rng.random_range(0..=i));
            }
            let before = tour_length(&d, &tour);
            two_opt(&d, &mut tour);
            assert!(tour_length(&d, &tour) <= before);
            let mut sorted = tour.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..30).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_same_seed_same_tour() {
        let d = euclidean(&random_cities(15, 11));
        let run = || {
            AntColony::new()
                .with_max_iterations(30)
                .with_seed(5)
                .solve_tsp(&d)
        };
        let (a, b) = (run(), run());
        assert_eq!(a.tour, b.tour);
        assert_eq!(a.history, b.history);
    }

    #[test]
    fn test_degenerate_sizes() {
        assert!(AntColony::new()
            .solve_tsp(&Array2::zeros((0, 0)))
            .tour
            .is_empty());
        assert_eq!(
            AntColony::new().solve_tsp(&Array2::zeros((1, 1))).tour,
            vec![0]
        );
        let two = AntColony::new().solve_tsp(&euclidean(&[(0.0, 0.0), (3.0, 4.0)]));
        assert_eq!((two.tour, two.length), (vec![0, 1], 10.0));
        // Duplicate cities (zero distances) still give a valid tour.
        let same = AntColony::new()
            .with_max_iterations(5)
            .solve_tsp(&Array2::zeros((4, 4)));
        assert_eq!(same.length, 0.0);
        assert_eq!(same.tour.len(), 4);
    }

    #[test]
    fn test_zero_iterations_returns_the_greedy_tour() {
        let d = euclidean(&random_cities(10, 3));
        let result = AntColony::new().with_max_iterations(0).solve_tsp(&d);
        assert_eq!(
            result.length,
            tour_length(&d, &nearest_neighbour_tour(&d, 0))
        );
        assert!(result.history.is_empty());
    }

    #[test]
    #[should_panic(expected = "symmetric")]
    fn test_rejects_an_asymmetric_matrix() {
        let mut d = Array2::zeros((3, 3));
        d[[0, 1]] = 1.0;
        AntColony::new().solve_tsp(&d);
    }

    #[test]
    #[should_panic(expected = "square")]
    fn test_rejects_a_non_square_matrix() {
        AntColony::new().solve_tsp(&Array2::zeros((2, 3)));
    }

    #[test]
    #[should_panic(expected = "alpha must be finite")]
    fn test_rejects_a_nan_alpha() {
        let d = euclidean(&random_cities(5, 1));
        AntColony::new().with_alpha(f64::NAN).solve_tsp(&d);
    }

    #[test]
    #[should_panic(expected = "beta must be finite")]
    fn test_rejects_an_infinite_beta() {
        let d = euclidean(&random_cities(5, 1));
        AntColony::new().with_beta(f64::INFINITY).solve_tsp(&d);
    }
}
