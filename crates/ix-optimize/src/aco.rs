//! Ant Colony Optimization (ACO) for the symmetric travelling salesman problem.
//!
//! Ant System (Dorigo, Maniezzo & Colorni, 1996). Each ant builds a closed tour
//! city by city. From city `i` it moves to an unvisited city `j` with
//! probability proportional to `tau[i][j]^alpha * eta[i][j]^beta`, where `tau`
//! is the pheromone and `eta = 1 / d` the heuristic. After each iteration all
//! pheromone evaporates by `rho`, then every ant deposits `1 / L` on the edges
//! of its tour of length `L`.
//!
//! CPU, `f64`, seeded: the same seed and inputs give the same tour. This is the
//! oracle a future GPU kernel would be checked against (ix#362).

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Floor for a distance or a tour length used as a divisor, so a zero distance
/// (two cities at the same point) yields a large finite heuristic, not infinity.
const MIN_DISTANCE: f64 = 1e-12;

/// Ant System configuration.
#[derive(Debug, Clone)]
pub struct AntColony {
    /// Ants per iteration; `None` sends one ant per city.
    pub num_ants: Option<usize>,
    pub max_iterations: usize,
    /// Pheromone exponent.
    pub alpha: f64,
    /// Heuristic (`1 / distance`) exponent.
    pub beta: f64,
    /// Share of the pheromone that evaporates each iteration, in `(0, 1]`.
    pub evaporation: f64,
    pub seed: u64,
}

impl Default for AntColony {
    /// The settings Dorigo & Stützle (2004, table 3.3) recommend for Ant System.
    fn default() -> Self {
        Self {
            num_ants: None,
            max_iterations: 200,
            alpha: 1.0,
            beta: 2.0,
            evaporation: 0.5,
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

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Find a short closed tour through every city of `distances`.
    ///
    /// # Panics
    ///
    /// If `distances` is not square, not symmetric, or holds a negative or
    /// non-finite value, or if the evaporation rate is outside `(0, 1]`.
    pub fn solve_tsp(&self, distances: &Array2<f64>) -> TourResult {
        validate(distances);
        assert!(
            self.evaporation > 0.0 && self.evaporation <= 1.0,
            "evaporation must be in (0, 1], got {}",
            self.evaporation
        );

        let n = distances.nrows();
        if n < 2 {
            return TourResult {
                tour: (0..n).collect(),
                length: 0.0,
                history: Vec::new(),
            };
        }

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

        // tau0 = m / C_nn (Dorigo & Stützle 2004, §3.4.1).
        let nearest = nearest_neighbour_tour(distances, 0);
        let tau0 = ants as f64 / tour_length(distances, &nearest).max(MIN_DISTANCE);
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
                    let tour = construct_tour(&choice, distances, &mut rng);
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
            for (tour, length) in &tours {
                let deposit = 1.0 / length.max(MIN_DISTANCE);
                for (a, b) in edges(tour) {
                    pheromone[[a, b]] += deposit;
                    pheromone[[b, a]] += deposit;
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
}
