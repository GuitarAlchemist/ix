//! Local search: hill climbing variants, beam search.

use rand::prelude::*;

/// A state for local search.
pub trait LocalSearchState: Clone {
    /// Generate neighbor states.
    fn neighbors(&self) -> Vec<Self>;

    /// Evaluate this state (higher is better).
    fn evaluate(&self) -> f64;
}

/// Simple hill climbing — always move to the best neighbor.
pub fn hill_climbing<S: LocalSearchState>(initial: S, max_steps: usize) -> (S, f64) {
    let mut current = initial;
    let mut current_val = current.evaluate();

    for _ in 0..max_steps {
        let neighbors = current.neighbors();
        if neighbors.is_empty() {
            break;
        }

        let best = neighbors
            .into_iter()
            .max_by(|a, b| a.evaluate().partial_cmp(&b.evaluate()).unwrap());

        if let Some(best) = best {
            let best_val = best.evaluate();
            if best_val > current_val {
                current = best;
                current_val = best_val;
            } else {
                break; // Local optimum
            }
        }
    }

    (current, current_val)
}

/// Stochastic hill climbing — randomly choose among improving neighbors.
pub fn stochastic_hill_climbing<S: LocalSearchState>(
    initial: S,
    max_steps: usize,
    seed: u64,
) -> (S, f64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut current = initial;
    let mut current_val = current.evaluate();

    for _ in 0..max_steps {
        let neighbors = current.neighbors();
        let improving: Vec<S> = neighbors
            .into_iter()
            .filter(|n| n.evaluate() > current_val)
            .collect();

        if improving.is_empty() {
            break;
        }

        let idx = rng.random_range(0..improving.len());
        current = improving.into_iter().nth(idx).unwrap();
        current_val = current.evaluate();
    }

    (current, current_val)
}

/// Random restart hill climbing — run hill climbing multiple times.
pub fn random_restart_hill_climbing<S, F>(
    generate_initial: F,
    max_steps: usize,
    restarts: usize,
) -> (S, f64)
where
    S: LocalSearchState,
    F: Fn() -> S,
{
    let mut best_state = generate_initial();
    let mut best_val = f64::NEG_INFINITY;

    for _ in 0..restarts {
        let initial = generate_initial();
        let (state, val) = hill_climbing(initial, max_steps);
        if val > best_val {
            best_val = val;
            best_state = state;
        }
    }

    (best_state, best_val)
}

/// Beam search — maintain k best states at each level.
pub fn beam_search<S: LocalSearchState>(
    initial: Vec<S>,
    beam_width: usize,
    max_steps: usize,
) -> (S, f64) {
    let mut beam = initial;

    for _ in 0..max_steps {
        let mut candidates: Vec<S> = beam.iter().flat_map(|s| s.neighbors()).collect();

        if candidates.is_empty() {
            break;
        }

        candidates.sort_by(|a, b| b.evaluate().partial_cmp(&a.evaluate()).unwrap());
        candidates.truncate(beam_width);
        beam = candidates;
    }

    beam.into_iter()
        .max_by(|a, b| a.evaluate().partial_cmp(&b.evaluate()).unwrap())
        .map(|s| {
            let v = s.evaluate();
            (s, v)
        })
        .unwrap_or_else(|| panic!("Empty beam"))
}

/// Tabu search — hill climbing with memory to avoid revisiting states.
pub fn tabu_search<S: LocalSearchState + std::hash::Hash + Eq>(
    initial: S,
    max_steps: usize,
    tabu_size: usize,
) -> (S, f64) {
    let mut current = initial;
    let mut current_val = current.evaluate();
    let mut best = current.clone();
    let mut best_val = current_val;
    let _tabu_list: std::collections::VecDeque<S> = std::collections::VecDeque::new();
    let _tabu_set: std::collections::HashSet<S> = std::collections::HashSet::new();

    // Use a simpler approach: track tabu as a ring buffer
    let mut tabu_vec: Vec<S> = Vec::new();

    for _ in 0..max_steps {
        let neighbors = current.neighbors();

        // Find best non-tabu neighbor (or best overall if all tabu)
        let best_neighbor = neighbors
            .into_iter()
            .filter(|n| !tabu_vec.contains(n))
            .max_by(|a, b| a.evaluate().partial_cmp(&b.evaluate()).unwrap());

        if let Some(next) = best_neighbor {
            let next_val = next.evaluate();

            tabu_vec.push(current.clone());
            if tabu_vec.len() > tabu_size {
                tabu_vec.remove(0);
            }

            current = next;
            current_val = next_val;

            if current_val > best_val {
                best_val = current_val;
                best = current.clone();
            }
        } else {
            break;
        }
    }

    (best, best_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Hash, Eq, PartialEq)]
    struct NumState(i32);

    impl LocalSearchState for NumState {
        fn neighbors(&self) -> Vec<Self> {
            vec![NumState(self.0 + 1), NumState(self.0 - 1)]
        }

        fn evaluate(&self) -> f64 {
            // Optimum at 10: -(x-10)^2
            -((self.0 - 10) as f64).powi(2)
        }
    }

    #[test]
    fn test_hill_climbing_finds_optimum() {
        let (state, _) = hill_climbing(NumState(0), 100);
        assert_eq!(state.0, 10);
    }

    #[test]
    fn test_beam_search() {
        let initial = vec![NumState(0), NumState(5), NumState(15)];
        let (state, _) = beam_search(initial, 3, 100);
        // Beam search may land within ±1 of optimum depending on neighbor generation
        assert!(
            (state.0 - 10).abs() <= 1,
            "Expected near 10, got {}",
            state.0
        );
    }
}
