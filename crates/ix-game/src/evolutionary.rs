//! Evolutionary game theory — replicator dynamics, ESS, and population games.

use ndarray::{Array1, Array2};

/// Replicator dynamics: dx_i/dt = x_i * (f_i(x) - f_avg(x))
///
/// `payoff_matrix`: symmetric payoff matrix for a population game.
/// `initial`: initial population proportions (must sum to 1).
/// `dt`: time step.
/// `steps`: number of steps.
///
/// Returns trajectory of population proportions.
pub fn replicator_dynamics(
    payoff_matrix: &Array2<f64>,
    initial: &Array1<f64>,
    dt: f64,
    steps: usize,
) -> Vec<Array1<f64>> {
    let n = initial.len();
    let mut x = initial.clone();
    let mut trajectory = Vec::with_capacity(steps + 1);
    trajectory.push(x.clone());

    for _ in 0..steps {
        // Fitness of each strategy: f_i = sum_j A[i,j] * x_j
        let fitness = payoff_matrix.dot(&x);

        // Average fitness: f_avg = sum_i x_i * f_i
        let avg_fitness = x.dot(&fitness);

        // Replicator update
        let mut new_x = Array1::zeros(n);
        for i in 0..n {
            new_x[i] = x[i] + dt * x[i] * (fitness[i] - avg_fitness);
        }

        // Project back onto simplex (handle numerical errors)
        for v in new_x.iter_mut() {
            *v = v.max(0.0);
        }
        let sum = new_x.sum();
        if sum > 1e-15 {
            new_x /= sum;
        }

        x = new_x;
        trajectory.push(x.clone());
    }

    trajectory
}

/// Check if a strategy is an Evolutionarily Stable Strategy (ESS).
///
/// Strategy `i` is ESS if for all j ≠ i:
///   A[i,i] > A[j,i]  (strict NE condition), OR
///   A[i,i] = A[j,i] AND A[i,j] > A[j,j]  (stability condition)
pub fn is_ess(payoff_matrix: &Array2<f64>, strategy: usize) -> bool {
    let n = payoff_matrix.nrows();

    for j in 0..n {
        if j == strategy {
            continue;
        }

        let a_ii = payoff_matrix[[strategy, strategy]];
        let a_ji = payoff_matrix[[j, strategy]];

        if a_ii < a_ji {
            return false;
        }

        if (a_ii - a_ji).abs() < 1e-10 {
            // Tie — check stability condition
            let a_ij = payoff_matrix[[strategy, j]];
            let a_jj = payoff_matrix[[j, j]];
            if a_ij <= a_jj {
                return false;
            }
        }
    }

    true
}

/// Find all ESS strategies in a symmetric game.
pub fn find_ess(payoff_matrix: &Array2<f64>) -> Vec<usize> {
    (0..payoff_matrix.nrows())
        .filter(|&i| is_ess(payoff_matrix, i))
        .collect()
}

/// Hawk-Dove game payoff matrix.
///
/// V = resource value, C = cost of fighting.
/// Hawk vs Hawk: (V-C)/2
/// Hawk vs Dove: V
/// Dove vs Hawk: 0
/// Dove vs Dove: V/2
pub fn hawk_dove_matrix(value: f64, cost: f64) -> Array2<f64> {
    ndarray::array![[(value - cost) / 2.0, value], [0.0, value / 2.0]]
}

/// Rock-Paper-Scissors payoff matrix (with configurable payoffs).
pub fn rps_matrix(win: f64, lose: f64, draw: f64) -> Array2<f64> {
    ndarray::array![[draw, lose, win], [win, draw, lose], [lose, win, draw]]
}

/// Multi-population replicator dynamics (two asymmetric populations).
///
/// `payoff_a`: payoff matrix for population A.
/// `payoff_b`: payoff matrix for population B (B's payoff when A plays i, B plays j).
pub fn two_population_replicator(
    payoff_a: &Array2<f64>,
    payoff_b: &Array2<f64>,
    initial_a: &Array1<f64>,
    initial_b: &Array1<f64>,
    dt: f64,
    steps: usize,
) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let m = initial_a.len();
    let n = initial_b.len();
    let mut xa = initial_a.clone();
    let mut xb = initial_b.clone();

    let mut traj_a = Vec::with_capacity(steps + 1);
    let mut traj_b = Vec::with_capacity(steps + 1);
    traj_a.push(xa.clone());
    traj_b.push(xb.clone());

    for _ in 0..steps {
        // Fitness for A strategies: expected payoff against B's distribution
        let fa = payoff_a.dot(&xb);
        let avg_fa = xa.dot(&fa);

        // Fitness for B strategies
        let fb = payoff_b.t().dot(&xa);
        let avg_fb = xb.dot(&fb);

        // Update
        let mut new_xa = Array1::zeros(m);
        let mut new_xb = Array1::zeros(n);

        for i in 0..m {
            new_xa[i] = xa[i] + dt * xa[i] * (fa[i] - avg_fa);
        }
        for j in 0..n {
            new_xb[j] = xb[j] + dt * xb[j] * (fb[j] - avg_fb);
        }

        // Project onto simplex
        for v in new_xa.iter_mut() {
            *v = v.max(0.0);
        }
        for v in new_xb.iter_mut() {
            *v = v.max(0.0);
        }
        let sa = new_xa.sum();
        let sb = new_xb.sum();
        if sa > 1e-15 {
            new_xa /= sa;
        }
        if sb > 1e-15 {
            new_xb /= sb;
        }

        xa = new_xa;
        xb = new_xb;
        traj_a.push(xa.clone());
        traj_b.push(xb.clone());
    }

    (traj_a, traj_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_hawk_dove_ess() {
        // When V < C, mixed strategy is ESS (neither pure Hawk nor pure Dove)
        let m = hawk_dove_matrix(2.0, 4.0);
        let ess = find_ess(&m);
        assert!(ess.is_empty(), "Neither pure strategy is ESS when V < C");

        // When V > C, Hawk is ESS
        let m = hawk_dove_matrix(6.0, 4.0);
        let ess = find_ess(&m);
        assert!(ess.contains(&0), "Hawk should be ESS when V > C");
    }

    #[test]
    fn test_replicator_prisoner_dilemma() {
        // PD payoff matrix: Defect dominates
        let payoff = array![[3.0, 0.0], [5.0, 1.0]];
        let initial = Array1::from_vec(vec![0.5, 0.5]);

        let traj = replicator_dynamics(&payoff, &initial, 0.01, 5000);
        let final_state = traj.last().unwrap();

        // Defectors should take over
        assert!(
            final_state[1] > 0.95,
            "Defectors should dominate: {:?}",
            final_state
        );
    }

    #[test]
    fn test_rps_stays_mixed() {
        // In RPS, population should cycle around (1/3, 1/3, 1/3)
        let payoff = rps_matrix(1.0, -1.0, 0.0);
        let initial = Array1::from_vec(vec![0.4, 0.3, 0.3]);

        let traj = replicator_dynamics(&payoff, &initial, 0.01, 10_000);
        let final_state = traj.last().unwrap();

        // All strategies should still be present
        for &x in final_state.iter() {
            assert!(
                x > 0.1,
                "RPS should maintain all strategies: {:?}",
                final_state
            );
        }
    }
}
