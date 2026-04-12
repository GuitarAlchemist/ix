//! Mean field games — large population game dynamics.
//!
//! When individual players are negligible, the population can be modeled
//! as a distribution evolving according to mean field equations.

use ndarray::Array1;

/// Discrete mean field game state.
///
/// Players choose between `num_actions` actions.
/// Population distribution over actions evolves based on payoffs.
pub struct MeanFieldGame {
    pub num_actions: usize,
    /// Payoff function: takes (action, population_distribution) -> payoff
    #[allow(clippy::type_complexity)]
    payoff_fn: Box<dyn Fn(usize, &Array1<f64>) -> f64>,
}

impl MeanFieldGame {
    pub fn new(num_actions: usize, payoff_fn: impl Fn(usize, &Array1<f64>) -> f64 + 'static) -> Self {
        Self { num_actions, payoff_fn: Box::new(payoff_fn) }
    }

    /// Compute payoffs for all actions given current distribution.
    pub fn payoffs(&self, distribution: &Array1<f64>) -> Array1<f64> {
        Array1::from_vec(
            (0..self.num_actions)
                .map(|a| (self.payoff_fn)(a, distribution))
                .collect()
        )
    }

    /// Simulate mean field dynamics using logit choice (softmax).
    ///
    /// At each step, players choose actions proportional to exp(beta * payoff).
    /// `beta`: rationality parameter (higher = more rational).
    pub fn logit_dynamics(
        &self,
        initial: &Array1<f64>,
        beta: f64,
        steps: usize,
        learning_rate: f64,
    ) -> Vec<Array1<f64>> {
        let mut dist = initial.clone();
        let mut trajectory = vec![dist.clone()];

        for _ in 0..steps {
            let payoffs = self.payoffs(&dist);

            // Softmax response
            let max_p = payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_payoffs: Array1<f64> = payoffs.mapv(|p| (beta * (p - max_p)).exp());
            let sum_exp = exp_payoffs.sum();
            let target = &exp_payoffs / sum_exp;

            // Smooth update
            dist = (1.0 - learning_rate) * &dist + learning_rate * &target;

            // Ensure valid distribution
            for v in dist.iter_mut() {
                *v = v.max(0.0);
            }
            let s = dist.sum();
            if s > 1e-15 {
                dist /= s;
            }

            trajectory.push(dist.clone());
        }

        trajectory
    }

    /// Find mean field equilibrium via fixed point iteration.
    ///
    /// A mean field equilibrium is a distribution m* such that the best
    /// response to m* is m* itself.
    pub fn find_equilibrium(
        &self,
        beta: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Array1<f64> {
        let n = self.num_actions;
        let mut dist = Array1::from_vec(vec![1.0 / n as f64; n]);

        let lr = 0.1; // Damped update to avoid oscillation
        for _ in 0..max_iterations {
            let payoffs = self.payoffs(&dist);
            let max_p = payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_payoffs: Array1<f64> = payoffs.mapv(|p| (beta * (p - max_p)).exp());
            let sum_exp = exp_payoffs.sum();
            let target = &exp_payoffs / sum_exp;

            let new_dist = (1.0 - lr) * &dist + lr * &target;
            let diff: f64 = (&new_dist - &dist).mapv(|x| x.abs()).sum();
            dist = new_dist;

            if diff < tolerance {
                break;
            }
        }

        dist
    }
}

/// Congestion game: payoff decreases with more players on same action.
///
/// payoff(action a, distribution m) = base_payoff[a] - congestion_cost * m[a]
pub fn congestion_payoff(
    base_payoffs: &[f64],
    congestion_cost: f64,
) -> impl Fn(usize, &Array1<f64>) -> f64 + '_ {
    move |action: usize, dist: &Array1<f64>| {
        base_payoffs[action] - congestion_cost * dist[action]
    }
}

/// Quantal Response Equilibrium (QRE) — bounded rationality.
///
/// Players make errors proportional to exp(lambda * payoff).
/// As lambda -> inf, converges to Nash; at lambda = 0, uniform random.
pub fn qre_logit(
    payoff_matrix: &ndarray::Array2<f64>,
    lambda: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Array1<f64> {
    let n = payoff_matrix.nrows();
    let mut x = Array1::from_vec(vec![1.0 / n as f64; n]);

    for _ in 0..max_iterations {
        let payoffs = payoff_matrix.dot(&x);
        let max_p = payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_p: Array1<f64> = payoffs.mapv(|p| (lambda * (p - max_p)).exp());
        let sum_exp = exp_p.sum();
        let new_x = &exp_p / sum_exp;

        let diff: f64 = (&new_x - &x).mapv(|v| v.abs()).sum();
        x = new_x;

        if diff < tolerance {
            break;
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_congestion_equilibrium() {
        // 3 routes, base payoffs [10, 8, 6], congestion cost 5
        // Equilibrium: players spread out to equalize payoffs
        let base = [10.0, 8.0, 6.0];
        let payoff_fn = move |a: usize, dist: &Array1<f64>| -> f64 {
            base[a] - 5.0 * dist[a]
        };

        let game = MeanFieldGame::new(3, payoff_fn);
        let eq = game.find_equilibrium(10.0, 1e-6, 10_000);

        // Higher base payoff routes should have more players
        assert!(eq[0] > eq[1], "Route 0 should be more popular: {:?}", eq);
        assert!(eq[1] > eq[2], "Route 1 should be more popular than 2: {:?}", eq);
    }

    #[test]
    fn test_qre_convergence() {
        // Prisoner's dilemma: at high lambda, should approach (Defect, Defect)
        let payoff = array![[3.0, 0.0], [5.0, 1.0]];

        let qre_low = qre_logit(&payoff, 0.1, 1e-8, 1000);
        let qre_high = qre_logit(&payoff, 10.0, 1e-8, 1000);

        // At high lambda, should be mostly Defect (index 1)
        assert!(qre_high[1] > qre_low[1], "Higher lambda should favor dominant strategy");
        assert!(qre_high[1] > 0.8, "Should strongly favor Defect at high lambda");
    }

    #[test]
    fn test_logit_dynamics() {
        let payoff_fn = |a: usize, dist: &Array1<f64>| -> f64 {
            let base = [10.0, 8.0];
            base[a] - 5.0 * dist[a]
        };

        let game = MeanFieldGame::new(2, payoff_fn);
        let initial = Array1::from_vec(vec![0.9, 0.1]);
        let traj = game.logit_dynamics(&initial, 5.0, 100, 0.1);

        // Should converge toward equilibrium
        let last = traj.last().unwrap();
        assert!(last[0] > 0.3 && last[0] < 0.9, "Should approach equilibrium: {:?}", last);
    }
}
