//! Mechanism design — VCG mechanism, incentive compatibility.
//!
//! Designing rules so that rational agents reveal truthful information.

/// A player's valuation: maps allocation to value.
#[derive(Debug, Clone)]
pub struct Player {
    /// Valuation function: value for each possible allocation outcome.
    /// `valuations[k]` = this player's value if outcome k is selected.
    pub valuations: Vec<f64>,
}

/// VCG (Vickrey-Clarke-Groves) mechanism.
///
/// Selects the socially optimal outcome and charges each player
/// the externality they impose on others.
pub struct VcgMechanism {
    pub players: Vec<Player>,
    pub num_outcomes: usize,
}

/// Result of running the VCG mechanism.
#[derive(Debug, Clone)]
pub struct VcgResult {
    /// The chosen outcome index.
    pub outcome: usize,
    /// Social welfare of the chosen outcome.
    pub social_welfare: f64,
    /// Payment for each player (positive = player pays).
    pub payments: Vec<f64>,
    /// Utility for each player (valuation - payment).
    pub utilities: Vec<f64>,
}

impl VcgMechanism {
    pub fn new(num_outcomes: usize) -> Self {
        Self {
            players: Vec::new(),
            num_outcomes,
        }
    }

    /// Add a player with their valuations.
    pub fn add_player(&mut self, valuations: Vec<f64>) {
        assert_eq!(valuations.len(), self.num_outcomes);
        self.players.push(Player { valuations });
    }

    /// Run the VCG mechanism.
    #[allow(clippy::needless_range_loop)]
    pub fn run(&self) -> VcgResult {
        let n = self.players.len();
        let k = self.num_outcomes;

        // Find socially optimal outcome: argmax sum of valuations
        let social_values: Vec<f64> = (0..k)
            .map(|outcome| {
                self.players.iter().map(|p| p.valuations[outcome]).sum()
            })
            .collect();

        let (outcome, &social_welfare) = social_values.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // Compute VCG payments for each player
        let mut payments = vec![0.0; n];

        for i in 0..n {
            // Optimal outcome without player i
            let best_without_i: f64 = (0..k)
                .map(|o| {
                    self.players.iter().enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, p)| p.valuations[o])
                        .sum::<f64>()
                })
                .fold(f64::NEG_INFINITY, f64::max);

            // Others' welfare at chosen outcome
            let others_at_chosen: f64 = self.players.iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, p)| p.valuations[outcome])
                .sum();

            // VCG payment: externality imposed on others
            payments[i] = best_without_i - others_at_chosen;
        }

        let utilities: Vec<f64> = (0..n)
            .map(|i| self.players[i].valuations[outcome] - payments[i])
            .collect();

        VcgResult {
            outcome,
            social_welfare,
            payments,
            utilities,
        }
    }
}

/// Myerson's optimal auction (single item, independent private values).
///
/// Computes the expected revenue-maximizing reserve price.
/// Assumes values are drawn from Uniform[0, v_max].
pub fn myerson_optimal_reserve(v_max: f64, num_bidders: usize) -> f64 {
    // For uniform distribution on [0, v_max]:
    // Virtual value: psi(v) = v - (1-F(v))/f(v) = v - (v_max - v) = 2v - v_max
    // Reserve price: psi(v) = 0 => v = v_max / 2
    let _ = num_bidders; // Reserve is independent of n for this distribution
    v_max / 2.0
}

/// Check if a mechanism is individually rational.
///
/// Each player's utility must be non-negative (participation constraint).
pub fn is_individually_rational(result: &VcgResult) -> bool {
    result.utilities.iter().all(|&u| u >= -1e-10)
}

/// Check budget balance: sum of payments >= 0 (no deficit).
pub fn is_weakly_budget_balanced(result: &VcgResult) -> bool {
    result.payments.iter().sum::<f64>() >= -1e-10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vcg_single_item() {
        // Single item auction: 3 bidders with values 10, 7, 5
        // VCG should allocate to bidder 0 (highest value)
        // and charge the second-highest bid = 7
        let mut vcg = VcgMechanism::new(4); // outcomes: nobody wins, 0 wins, 1 wins, 2 wins

        // Bidder 0: values 10 if they win (outcome 1), 0 otherwise
        vcg.add_player(vec![0.0, 10.0, 0.0, 0.0]);
        // Bidder 1: values 7 if they win (outcome 2), 0 otherwise
        vcg.add_player(vec![0.0, 0.0, 7.0, 0.0]);
        // Bidder 2: values 5 if they win (outcome 3), 0 otherwise
        vcg.add_player(vec![0.0, 0.0, 0.0, 5.0]);

        let result = vcg.run();
        assert_eq!(result.outcome, 1, "Should allocate to highest bidder");
        assert!((result.payments[0] - 7.0).abs() < 1e-8, "Winner pays second-highest bid");
        assert!(is_individually_rational(&result));
    }

    #[test]
    fn test_vcg_public_good() {
        // Public good: build a bridge costing 10
        // Player 0 values it at 8, Player 1 at 6
        // Outcomes: 0 = don't build, 1 = build
        let mut vcg = VcgMechanism::new(2);

        vcg.add_player(vec![0.0, 8.0]);  // Player 0
        vcg.add_player(vec![0.0, 6.0]);  // Player 1

        let result = vcg.run();
        assert_eq!(result.outcome, 1, "Should build (total value 14 > 0)");
        assert!(is_individually_rational(&result));
    }

    #[test]
    fn test_myerson_reserve() {
        let reserve = myerson_optimal_reserve(100.0, 2);
        assert!((reserve - 50.0).abs() < 1e-8);
    }
}
