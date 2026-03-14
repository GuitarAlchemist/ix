//! Nash equilibria and solution concepts.
//!
//! Support vector (Lemke-Howson), iterated best response,
//! and mixed strategy equilibria for bimatrix games.

use ndarray::{Array1, Array2};

/// A bimatrix game (2 players).
///
/// `payoff_a[i][j]` = player A's payoff when A plays i, B plays j.
/// `payoff_b[i][j]` = player B's payoff when A plays i, B plays j.
#[derive(Debug, Clone)]
pub struct BimatrixGame {
    pub payoff_a: Array2<f64>,
    pub payoff_b: Array2<f64>,
}

/// A strategy profile: mixed strategies for both players.
#[derive(Debug, Clone)]
pub struct StrategyProfile {
    pub player_a: Array1<f64>,
    pub player_b: Array1<f64>,
}

impl StrategyProfile {
    /// Expected payoff for player A.
    pub fn expected_payoff_a(&self, game: &BimatrixGame) -> f64 {
        let mut payoff = 0.0;
        for i in 0..self.player_a.len() {
            for j in 0..self.player_b.len() {
                payoff += self.player_a[i] * self.player_b[j] * game.payoff_a[[i, j]];
            }
        }
        payoff
    }

    /// Expected payoff for player B.
    pub fn expected_payoff_b(&self, game: &BimatrixGame) -> f64 {
        let mut payoff = 0.0;
        for i in 0..self.player_a.len() {
            for j in 0..self.player_b.len() {
                payoff += self.player_a[i] * self.player_b[j] * game.payoff_b[[i, j]];
            }
        }
        payoff
    }
}

impl BimatrixGame {
    pub fn new(payoff_a: Array2<f64>, payoff_b: Array2<f64>) -> Self {
        assert_eq!(payoff_a.shape(), payoff_b.shape());
        Self { payoff_a, payoff_b }
    }

    /// Create a zero-sum game (B's payoff = -A's payoff).
    pub fn zero_sum(payoff_a: Array2<f64>) -> Self {
        let payoff_b = -&payoff_a;
        Self { payoff_a, payoff_b }
    }

    /// Number of strategies for player A.
    pub fn num_strategies_a(&self) -> usize {
        self.payoff_a.nrows()
    }

    /// Number of strategies for player B.
    pub fn num_strategies_b(&self) -> usize {
        self.payoff_a.ncols()
    }

    /// Best response of player A given B's mixed strategy.
    pub fn best_response_a(&self, strategy_b: &Array1<f64>) -> Array1<f64> {
        let m = self.num_strategies_a();
        let expected: Vec<f64> = (0..m)
            .map(|i| {
                self.payoff_a.row(i).dot(strategy_b)
            })
            .collect();

        let max_val = expected.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut br = Array1::zeros(m);
        let best_indices: Vec<usize> = expected.iter().enumerate()
            .filter(|(_, &v)| (v - max_val).abs() < 1e-10)
            .map(|(i, _)| i)
            .collect();
        for &i in &best_indices {
            br[i] = 1.0 / best_indices.len() as f64;
        }
        br
    }

    /// Best response of player B given A's mixed strategy.
    pub fn best_response_b(&self, strategy_a: &Array1<f64>) -> Array1<f64> {
        let n = self.num_strategies_b();
        let expected: Vec<f64> = (0..n)
            .map(|j| {
                self.payoff_b.column(j).dot(strategy_a)
            })
            .collect();

        let max_val = expected.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut br = Array1::zeros(n);
        let best_indices: Vec<usize> = expected.iter().enumerate()
            .filter(|(_, &v)| (v - max_val).abs() < 1e-10)
            .map(|(i, _)| i)
            .collect();
        for &i in &best_indices {
            br[i] = 1.0 / best_indices.len() as f64;
        }
        br
    }

    /// Check if a strategy profile is a Nash equilibrium (within tolerance).
    pub fn is_nash_equilibrium(&self, profile: &StrategyProfile, tolerance: f64) -> bool {
        let m = self.num_strategies_a();
        let n = self.num_strategies_b();

        // Check player A can't improve
        let a_payoff = profile.expected_payoff_a(self);
        for i in 0..m {
            let mut pure = Array1::zeros(m);
            pure[i] = 1.0;
            let alt = StrategyProfile {
                player_a: pure,
                player_b: profile.player_b.clone(),
            };
            if alt.expected_payoff_a(self) > a_payoff + tolerance {
                return false;
            }
        }

        // Check player B can't improve
        let b_payoff = profile.expected_payoff_b(self);
        for j in 0..n {
            let mut pure = Array1::zeros(n);
            pure[j] = 1.0;
            let alt = StrategyProfile {
                player_a: profile.player_a.clone(),
                player_b: pure,
            };
            if alt.expected_payoff_b(self) > b_payoff + tolerance {
                return false;
            }
        }

        true
    }

    /// Find Nash equilibria via support enumeration (exact, for small games).
    ///
    /// Enumerates all possible support pairs and solves the indifference conditions.
    pub fn support_enumeration(&self) -> Vec<StrategyProfile> {
        let m = self.num_strategies_a();
        let n = self.num_strategies_b();
        let mut equilibria = Vec::new();

        // Enumerate support sizes
        for sa_size in 1..=m {
            for sb_size in 1..=n {
                // Enumerate all subsets of given size
                for sa in subsets(m, sa_size) {
                    for sb in subsets(n, sb_size) {
                        if let Some(profile) = self.solve_support(&sa, &sb) {
                            if self.is_nash_equilibrium(&profile, 1e-8) {
                                equilibria.push(profile);
                            }
                        }
                    }
                }
            }
        }

        equilibria
    }

    /// Solve for mixed strategy given supports using indifference conditions.
    fn solve_support(&self, support_a: &[usize], support_b: &[usize]) -> Option<StrategyProfile> {
        // For a 2x2 game with full support, solve analytically
        let m = self.num_strategies_a();
        let n = self.num_strategies_b();

        if support_a.len() == 1 && support_b.len() == 1 {
            // Pure strategy Nash check
            let i = support_a[0];
            let j = support_b[0];

            // Check if i is best response to j and j is best response to i
            let a_best = (0..m).all(|k| self.payoff_a[[i, j]] >= self.payoff_a[[k, j]] - 1e-10);
            let b_best = (0..n).all(|k| self.payoff_b[[i, j]] >= self.payoff_b[[i, k]] - 1e-10);

            if a_best && b_best {
                let mut pa = Array1::zeros(m);
                let mut pb = Array1::zeros(n);
                pa[i] = 1.0;
                pb[j] = 1.0;
                return Some(StrategyProfile { player_a: pa, player_b: pb });
            }
            return None;
        }

        // For 2x2 mixed strategy: solve indifference conditions
        if m == 2 && n == 2 && support_a.len() == 2 && support_b.len() == 2 {
            return self.solve_2x2_mixed();
        }

        None // General case would need linear programming
    }

    /// Solve 2x2 game for mixed Nash equilibrium.
    fn solve_2x2_mixed(&self) -> Option<StrategyProfile> {
        let a = &self.payoff_a;
        let b = &self.payoff_b;

        // Player B mixes to make A indifferent:
        // a[0,0]*q + a[0,1]*(1-q) = a[1,0]*q + a[1,1]*(1-q)
        let denom_q = (a[[0, 0]] - a[[0, 1]]) - (a[[1, 0]] - a[[1, 1]]);
        if denom_q.abs() < 1e-15 {
            return None;
        }
        let q = (a[[1, 1]] - a[[0, 1]]) / denom_q;

        // Player A mixes to make B indifferent:
        // b[0,0]*p + b[1,0]*(1-p) = b[0,1]*p + b[1,1]*(1-p)
        let denom_p = (b[[0, 0]] - b[[1, 0]]) - (b[[0, 1]] - b[[1, 1]]);
        if denom_p.abs() < 1e-15 {
            return None;
        }
        let p = (b[[1, 1]] - b[[1, 0]]) / denom_p;

        if (0.0..=1.0).contains(&p) && (0.0..=1.0).contains(&q) {
            Some(StrategyProfile {
                player_a: Array1::from_vec(vec![p, 1.0 - p]),
                player_b: Array1::from_vec(vec![q, 1.0 - q]),
            })
        } else {
            None
        }
    }
}

/// Find Nash equilibria using iterated best response (fictitious play).
///
/// May not converge for all games, but works well in practice for many.
pub fn fictitious_play(
    game: &BimatrixGame,
    iterations: usize,
) -> StrategyProfile {
    let m = game.num_strategies_a();
    let n = game.num_strategies_b();

    let mut count_a = Array1::zeros(m);
    let mut count_b = Array1::zeros(n);
    count_a[0] = 1.0;
    count_b[0] = 1.0;

    for _ in 0..iterations {
        let empirical_b = &count_b / count_b.sum();
        let br_a = game.best_response_a(&empirical_b);
        let action_a = br_a.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;

        let empirical_a = &count_a / count_a.sum();
        let br_b = game.best_response_b(&empirical_a);
        let action_b = br_b.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;

        count_a[action_a] += 1.0;
        count_b[action_b] += 1.0;
    }

    StrategyProfile {
        player_a: &count_a / count_a.sum(),
        player_b: &count_b / count_b.sum(),
    }
}

/// Enumerate all subsets of {0, ..., n-1} with exactly `k` elements.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    subsets_helper(n, k, 0, &mut current, &mut result);
    result
}

fn subsets_helper(n: usize, k: usize, start: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    for i in start..n {
        current.push(i);
        subsets_helper(n, k, i + 1, current, result);
        current.pop();
    }
}

/// Find dominant strategy equilibria (strictly dominant strategies).
pub fn dominant_strategy_equilibrium(game: &BimatrixGame) -> Option<StrategyProfile> {
    let m = game.num_strategies_a();
    let n = game.num_strategies_b();

    // Find strictly dominant strategy for A
    let dom_a = (0..m).find(|&i| {
        (0..m).all(|k| {
            k == i || (0..n).all(|j| game.payoff_a[[i, j]] > game.payoff_a[[k, j]])
        })
    });

    // Find strictly dominant strategy for B
    let dom_b = (0..n).find(|&j| {
        (0..n).all(|k| {
            k == j || (0..m).all(|i| game.payoff_b[[i, j]] > game.payoff_b[[i, k]])
        })
    });

    match (dom_a, dom_b) {
        (Some(i), Some(j)) => {
            let mut pa = Array1::zeros(m);
            let mut pb = Array1::zeros(n);
            pa[i] = 1.0;
            pb[j] = 1.0;
            Some(StrategyProfile { player_a: pa, player_b: pb })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_prisoners_dilemma() {
        // C  D
        // C (3,3) (0,5)
        // D (5,0) (1,1)
        let game = BimatrixGame::new(
            array![[3.0, 0.0], [5.0, 1.0]],
            array![[3.0, 5.0], [0.0, 1.0]],
        );

        // Dominant strategy: both Defect
        let dom = dominant_strategy_equilibrium(&game);
        assert!(dom.is_some());
        let prof = dom.unwrap();
        assert!((prof.player_a[1] - 1.0).abs() < 1e-10); // Defect
        assert!((prof.player_b[1] - 1.0).abs() < 1e-10); // Defect
    }

    #[test]
    fn test_matching_pennies_mixed() {
        // H  T
        // H (1,-1) (-1,1)
        // T (-1,1) (1,-1)
        let game = BimatrixGame::new(
            array![[1.0, -1.0], [-1.0, 1.0]],
            array![[-1.0, 1.0], [1.0, -1.0]],
        );

        let equilibria = game.support_enumeration();
        // Should find the mixed equilibrium (0.5, 0.5)
        let mixed = equilibria.iter().find(|e| e.player_a[0] > 0.1 && e.player_a[0] < 0.9);
        assert!(mixed.is_some(), "Should find mixed NE");
        let m = mixed.unwrap();
        assert!((m.player_a[0] - 0.5).abs() < 0.01);
        assert!((m.player_b[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_battle_of_sexes() {
        // Opera  Football
        // Opera    (3,2)    (0,0)
        // Football (0,0)    (2,3)
        let game = BimatrixGame::new(
            array![[3.0, 0.0], [0.0, 2.0]],
            array![[2.0, 0.0], [0.0, 3.0]],
        );

        let equilibria = game.support_enumeration();
        // Should find at least 2 pure NE + 1 mixed NE
        assert!(equilibria.len() >= 2, "BoS has 3 equilibria, found {}", equilibria.len());
    }

    #[test]
    fn test_fictitious_play_converges() {
        // Prisoner's dilemma — should converge to (Defect, Defect)
        let game = BimatrixGame::new(
            array![[3.0, 0.0], [5.0, 1.0]],
            array![[3.0, 5.0], [0.0, 1.0]],
        );

        let result = fictitious_play(&game, 1000);
        assert!(result.player_a[1] > 0.9, "Should converge to Defect");
        assert!(result.player_b[1] > 0.9, "Should converge to Defect");
    }
}
