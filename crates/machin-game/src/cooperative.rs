//! Cooperative game theory — Shapley value, the Core, nucleolus.

use std::collections::HashMap;

/// A cooperative game in characteristic function form.
///
/// `v(S)` gives the value that coalition S can guarantee.
/// Players are numbered 0..n-1.
#[derive(Debug, Clone)]
pub struct CooperativeGame {
    pub num_players: usize,
    /// Characteristic function: maps coalition (bitmask) to value.
    values: HashMap<u64, f64>,
}

impl CooperativeGame {
    pub fn new(num_players: usize) -> Self {
        assert!(num_players <= 63, "Max 63 players (bitmask limit)");
        let mut values = HashMap::new();
        values.insert(0, 0.0); // Empty coalition has value 0
        Self { num_players, values }
    }

    /// Set the value of a coalition given as a bitmask.
    pub fn set_value(&mut self, coalition: u64, value: f64) {
        self.values.insert(coalition, value);
    }

    /// Set value using a slice of player indices.
    pub fn set_value_for(&mut self, players: &[usize], value: f64) {
        let mask = players.iter().fold(0u64, |acc, &p| acc | (1 << p));
        self.set_value(mask, value);
    }

    /// Get the value of a coalition.
    pub fn value(&self, coalition: u64) -> f64 {
        *self.values.get(&coalition).unwrap_or(&0.0)
    }

    /// Grand coalition mask (all players).
    pub fn grand_coalition(&self) -> u64 {
        (1u64 << self.num_players) - 1
    }

    /// Compute the Shapley value for each player.
    ///
    /// phi_i = sum over S not containing i:
    ///   |S|!(n-|S|-1)! / n! * [v(S ∪ {i}) - v(S)]
    #[allow(clippy::needless_range_loop)]
    pub fn shapley_value(&self) -> Vec<f64> {
        let n = self.num_players;
        let n_fact = factorial(n);
        let mut phi = vec![0.0; n];

        for i in 0..n {
            let mask_i = 1u64 << i;

            // Iterate over all coalitions not containing i
            for s in 0..(1u64 << n) {
                if s & mask_i != 0 {
                    continue; // Skip coalitions containing i
                }

                let s_size = s.count_ones() as usize;
                let weight = factorial(s_size) * factorial(n - s_size - 1);
                let marginal = self.value(s | mask_i) - self.value(s);
                phi[i] += weight as f64 * marginal / n_fact as f64;
            }
        }

        phi
    }

    /// Check if an allocation is in the Core.
    ///
    /// An allocation x is in the Core if:
    /// 1. sum(x) = v(N) (efficiency)
    /// 2. For all coalitions S: sum_{i in S} x_i >= v(S) (coalition rationality)
    pub fn is_in_core(&self, allocation: &[f64]) -> bool {
        let n = self.num_players;

        // Efficiency check
        let total: f64 = allocation.iter().sum();
        let grand_value = self.value(self.grand_coalition());
        if (total - grand_value).abs() > 1e-8 {
            return false;
        }

        // Coalition rationality check
        for s in 1..(1u64 << n) {
            let coalition_alloc: f64 = (0..n)
                .filter(|&i| s & (1u64 << i) != 0)
                .map(|i| allocation[i])
                .sum();
            if coalition_alloc < self.value(s) - 1e-8 {
                return false;
            }
        }

        true
    }

    /// Check if the game is superadditive.
    ///
    /// v(S ∪ T) >= v(S) + v(T) for all disjoint S, T.
    pub fn is_superadditive(&self) -> bool {
        let n = self.num_players;
        for s in 0..(1u64 << n) {
            // Iterate over subsets of the complement
            let complement = self.grand_coalition() & !s;
            let mut t = complement;
            while t > 0 {
                if s & t == 0 {
                    // Disjoint
                    if self.value(s | t) < self.value(s) + self.value(t) - 1e-10 {
                        return false;
                    }
                }
                t = (t - 1) & complement;
            }
        }
        true
    }

    /// Compute the Banzhaf power index.
    ///
    /// Measures voting power: how often a player is a swing voter.
    #[allow(clippy::needless_range_loop)]
    pub fn banzhaf_index(&self) -> Vec<f64> {
        let n = self.num_players;
        let mut power = vec![0.0; n];

        for i in 0..n {
            let mask_i = 1u64 << i;
            let mut swings = 0u64;
            let mut total = 0u64;

            for s in 0..(1u64 << n) {
                if s & mask_i != 0 {
                    continue;
                }
                total += 1;
                let with_i = self.value(s | mask_i);
                let without_i = self.value(s);
                if with_i > without_i + 1e-10 {
                    swings += 1;
                }
            }

            power[i] = swings as f64 / total as f64;
        }

        // Normalize
        let sum: f64 = power.iter().sum();
        if sum > 1e-15 {
            for p in power.iter_mut() {
                *p /= sum;
            }
        }

        power
    }
}

fn factorial(n: usize) -> u64 {
    (1..=n as u64).product()
}

/// Weighted voting game: a coalition wins if sum of weights >= quota.
pub fn weighted_voting_game(weights: &[f64], quota: f64) -> CooperativeGame {
    let n = weights.len();
    let mut game = CooperativeGame::new(n);

    for s in 0..(1u64 << n) {
        let total_weight: f64 = (0..n)
            .filter(|&i| s & (1u64 << i) != 0)
            .map(|i| weights[i])
            .sum();

        game.set_value(s, if total_weight >= quota { 1.0 } else { 0.0 });
    }

    game
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapley_glove_game() {
        // Glove game: 2 left-hand glove holders (0, 1), 1 right-hand holder (2)
        // A pair is worth 1, singles worth 0
        let mut game = CooperativeGame::new(3);
        // Only coalitions with both L and R have value 1
        game.set_value_for(&[0, 2], 1.0);
        game.set_value_for(&[1, 2], 1.0);
        game.set_value_for(&[0, 1, 2], 1.0);

        let shapley = game.shapley_value();
        // Player 2 (right) should get more than 0 or 1 (left)
        assert!(shapley[2] > shapley[0], "Right glove holder should have more power");
        // Shapley values sum to v(N) = 1
        let total: f64 = shapley.iter().sum();
        assert!((total - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_shapley_symmetric() {
        // Symmetric game: all players contribute equally
        let mut game = CooperativeGame::new(3);
        game.set_value_for(&[0], 1.0);
        game.set_value_for(&[1], 1.0);
        game.set_value_for(&[2], 1.0);
        game.set_value_for(&[0, 1], 2.0);
        game.set_value_for(&[0, 2], 2.0);
        game.set_value_for(&[1, 2], 2.0);
        game.set_value_for(&[0, 1, 2], 3.0);

        let shapley = game.shapley_value();
        // All should be equal = 1.0
        for &s in &shapley {
            assert!((s - 1.0).abs() < 1e-8, "Symmetric game: all Shapley values should be 1.0");
        }
    }

    #[test]
    fn test_core_membership() {
        let mut game = CooperativeGame::new(3);
        game.set_value_for(&[0], 0.0);
        game.set_value_for(&[1], 0.0);
        game.set_value_for(&[2], 0.0);
        game.set_value_for(&[0, 1], 7.0);
        game.set_value_for(&[0, 2], 5.0);
        game.set_value_for(&[1, 2], 3.0);
        game.set_value_for(&[0, 1, 2], 10.0);

        // This allocation should be in the core
        let alloc = vec![5.0, 3.0, 2.0];
        assert!(game.is_in_core(&alloc), "Should be in core");

        // This allocation violates {0,1} >= 7
        let bad_alloc = vec![2.0, 4.0, 4.0];
        assert!(!game.is_in_core(&bad_alloc), "Should NOT be in core");
    }

    #[test]
    fn test_weighted_voting() {
        // UN Security Council simplified: 3 permanent (veto), 2 non-permanent
        // Quota = 3 permanent + 1 non-permanent
        let game = weighted_voting_game(&[3.0, 3.0, 3.0, 1.0, 1.0], 10.0);
        let banzhaf = game.banzhaf_index();

        // Permanent members should have more power
        assert!(banzhaf[0] > banzhaf[3], "Permanent members should have more power");
    }
}
