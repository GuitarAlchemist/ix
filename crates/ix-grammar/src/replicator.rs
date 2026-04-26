//! Grammar species replicator dynamics.
//!
//! Ports TARS `ReplicatorDynamics.fs`: each grammar rule competes as a
//! "species" with proportion and fitness.  Implements:
//!   `dx_i/dt = x_i * (f_i - f_avg)`
//! and detects evolutionarily stable strategies (ESS) by proportion threshold.

use serde::{Deserialize, Serialize};

/// A grammar rule modelled as an evolutionary species.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrammarSpecies {
    /// Rule identifier.
    pub id: String,
    /// Current proportion in the population (sums to 1 across all species).
    pub proportion: f64,
    /// Fitness score (e.g. derivation success rate, reward, or manual assignment).
    pub fitness: f64,
    /// Whether this species was detected as an ESS at last detection pass.
    pub is_stable: bool,
}

impl GrammarSpecies {
    pub fn new(id: impl Into<String>, proportion: f64, fitness: f64) -> Self {
        GrammarSpecies {
            id: id.into(),
            proportion,
            fitness,
            is_stable: false,
        }
    }
}

/// Full result from a replicator simulation.
pub struct SimulationResult {
    /// Final species proportions after all steps.
    pub final_species: Vec<GrammarSpecies>,
    /// Proportion snapshots at every step (length = steps + 1).
    pub trajectory: Vec<Vec<GrammarSpecies>>,
    /// Species identified as ESS at the end of the simulation.
    pub ess: Vec<GrammarSpecies>,
}

/// One replicator dynamics step: `dx_i/dt = x_i * (f_i - f_avg)`.
///
/// Species with proportion driven below 0 are clamped to 0.
/// Proportions are re-normalised onto the simplex after each step.
///
/// ```
/// use ix_grammar::replicator::{GrammarSpecies, replicator_step};
/// let species = vec![
///     GrammarSpecies::new("a", 0.5, 1.0),
///     GrammarSpecies::new("b", 0.5, 0.0),
/// ];
/// let next = replicator_step(&species, 0.1);
/// assert!(next[0].proportion > 0.5);
/// assert!(next[1].proportion < 0.5);
/// ```
pub fn replicator_step(species: &[GrammarSpecies], dt: f64) -> Vec<GrammarSpecies> {
    let avg_fitness: f64 = species.iter().map(|s| s.proportion * s.fitness).sum();

    let mut new_species: Vec<GrammarSpecies> = species
        .iter()
        .map(|s| {
            let new_prop = (s.proportion + dt * s.proportion * (s.fitness - avg_fitness)).max(0.0);
            GrammarSpecies {
                proportion: new_prop,
                ..s.clone()
            }
        })
        .collect();

    // Re-normalise onto simplex
    let total: f64 = new_species.iter().map(|s| s.proportion).sum();
    if total > 1e-15 {
        for s in &mut new_species {
            s.proportion /= total;
        }
    }
    new_species
}

/// Identify evolutionarily stable species: those whose proportion ≥ `threshold`.
///
/// Returns clones with `is_stable = true`.
///
/// ```
/// use ix_grammar::replicator::{GrammarSpecies, detect_ess};
/// let species = vec![
///     GrammarSpecies::new("dominant", 0.9, 1.0),
///     GrammarSpecies::new("minor", 0.1, 0.2),
/// ];
/// let ess = detect_ess(&species, 0.5);
/// assert_eq!(ess.len(), 1);
/// assert_eq!(ess[0].id, "dominant");
/// ```
pub fn detect_ess(species: &[GrammarSpecies], threshold: f64) -> Vec<GrammarSpecies> {
    species
        .iter()
        .filter(|s| s.proportion >= threshold)
        .map(|s| GrammarSpecies {
            is_stable: true,
            ..s.clone()
        })
        .collect()
}

/// Run full replicator dynamics simulation.
///
/// Species with proportion below `prune_threshold` are zeroed and
/// re-normalised at each step.
///
/// ```
/// use ix_grammar::replicator::{GrammarSpecies, simulate};
/// let species = vec![
///     GrammarSpecies::new("fit", 0.5, 1.0),
///     GrammarSpecies::new("unfit", 0.5, 0.0),
/// ];
/// let result = simulate(&species, 200, 0.05, 0.001);
/// assert!(result.final_species[0].proportion > 0.95);
/// ```
pub fn simulate(
    initial: &[GrammarSpecies],
    steps: usize,
    dt: f64,
    prune_threshold: f64,
) -> SimulationResult {
    let mut current = initial.to_vec();
    let mut trajectory = Vec::with_capacity(steps + 1);
    trajectory.push(current.clone());

    for _ in 0..steps {
        current = replicator_step(&current, dt);

        // Prune extinct species
        for s in &mut current {
            if s.proportion < prune_threshold {
                s.proportion = 0.0;
            }
        }
        // Re-normalise after pruning
        let total: f64 = current.iter().map(|s| s.proportion).sum();
        if total > 1e-15 {
            for s in &mut current {
                s.proportion /= total;
            }
        }

        trajectory.push(current.clone());
    }

    let ess_threshold = prune_threshold.max(0.5);
    let ess = detect_ess(&current, ess_threshold);

    SimulationResult {
        final_species: current,
        trajectory,
        ess,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replicator_step_fitter_grows() {
        let species = vec![
            GrammarSpecies::new("fit", 0.5, 2.0),
            GrammarSpecies::new("unfit", 0.5, 0.0),
        ];
        let next = replicator_step(&species, 0.1);
        assert!(next[0].proportion > 0.5, "Fitter species should grow");
        assert!(next[1].proportion < 0.5, "Unfit species should shrink");
    }

    #[test]
    fn test_replicator_step_proportions_sum_to_one() {
        let species = vec![
            GrammarSpecies::new("a", 0.3, 1.0),
            GrammarSpecies::new("b", 0.4, 0.5),
            GrammarSpecies::new("c", 0.3, 0.2),
        ];
        let next = replicator_step(&species, 0.05);
        let total: f64 = next.iter().map(|s| s.proportion).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_ess_threshold() {
        let species = vec![
            GrammarSpecies::new("dominant", 0.8, 1.0),
            GrammarSpecies::new("minor", 0.2, 0.5),
        ];
        let ess = detect_ess(&species, 0.5);
        assert_eq!(ess.len(), 1);
        assert_eq!(ess[0].id, "dominant");
        assert!(ess[0].is_stable);
    }

    #[test]
    fn test_simulate_dominant_species_wins() {
        let species = vec![
            GrammarSpecies::new("fit", 0.5, 1.0),
            GrammarSpecies::new("unfit", 0.5, 0.0),
        ];
        let result = simulate(&species, 500, 0.05, 1e-6);
        let fit_prop = result
            .final_species
            .iter()
            .find(|s| s.id == "fit")
            .unwrap()
            .proportion;
        assert!(fit_prop > 0.99, "Fit species should dominate: {}", fit_prop);
    }

    #[test]
    fn test_simulate_trajectory_length() {
        let species = vec![GrammarSpecies::new("a", 1.0, 1.0)];
        let result = simulate(&species, 10, 0.1, 1e-6);
        assert_eq!(result.trajectory.len(), 11); // initial + 10 steps
    }

    #[test]
    fn test_simulate_ess_detection() {
        let species = vec![
            GrammarSpecies::new("dominant", 0.95, 1.0),
            GrammarSpecies::new("rare", 0.05, 0.1),
        ];
        let result = simulate(&species, 100, 0.01, 1e-6);
        assert!(!result.ess.is_empty(), "Should detect at least one ESS");
        assert!(result.ess.iter().any(|s| s.id == "dominant"));
    }
}
