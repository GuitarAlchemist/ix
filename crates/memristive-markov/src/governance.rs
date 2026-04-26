//! Governance state modeling using memristive Markov chains.
//!
//! Maps Demerzel governance states (Healthy, Watch, Warning, Freeze) to
//! a 4-state memristive Markov model. Transition probabilities adapt based
//! on governance history — repeated violations increase resistance to recovery,
//! sustained health decreases resistance to staying healthy.

use crate::engine::MemristiveEngine;
use crate::sampler::SamplingStrategy;
use crate::serde_state::EngineConfig;
use crate::vlmm::FallbackStrategy;
use serde::{Deserialize, Serialize};

/// The four governance states from anti-lolli-inflation-policy.yaml
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GovernanceState {
    /// R >= 0.9, ratio < 1.5 — all systems nominal
    Healthy = 0,
    /// R >= 0.7, ratio 1.5-2.5 — monitor closely
    Watch = 1,
    /// R >= 0.5, ratio 2.5-3.0 — one more cycle triggers freeze
    Warning = 2,
    /// R < 0.5 or ratio > 3.0 for 3 cycles — creation frozen
    Freeze = 3,
}

impl GovernanceState {
    /// Classify current governance state from metrics
    pub fn classify(resilience_score: f64, lolli_ratio: f64) -> Self {
        if lolli_ratio > 3.0 || resilience_score < 0.5 {
            GovernanceState::Freeze
        } else if lolli_ratio > 2.5 || resilience_score < 0.7 {
            GovernanceState::Warning
        } else if lolli_ratio > 1.5 || resilience_score < 0.9 {
            GovernanceState::Watch
        } else {
            GovernanceState::Healthy
        }
    }

    pub fn as_index(self) -> usize {
        self as usize
    }

    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(GovernanceState::Healthy),
            1 => Some(GovernanceState::Watch),
            2 => Some(GovernanceState::Warning),
            3 => Some(GovernanceState::Freeze),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            GovernanceState::Healthy => "healthy",
            GovernanceState::Watch => "watch",
            GovernanceState::Warning => "warning",
            GovernanceState::Freeze => "freeze",
        }
    }
}

/// Governance-specific wrapper around MemristiveEngine
pub struct GovernanceMarkov {
    engine: MemristiveEngine,
    history: Vec<GovernanceState>,
    consecutive_freeze: usize,
}

impl GovernanceMarkov {
    /// Create a new governance Markov model with governance-tuned parameters
    pub fn new() -> Self {
        let config = EngineConfig {
            max_order: 2,        // Consider last 2 states
            session_alpha: 0.15, // Moderate learning rate
            session_beta: 0.05,  // 5% decay per cycle
            long_term_alpha: 0.02,
            long_term_beta: 0.005,
            g_min: 0.01,
            min_observations: 2, // Low threshold for governance (few cycles)
            fallback: FallbackStrategy::MarginalDistribution,
            consolidation_gamma: 0.1,
            min_session_observations: 3,
            default_sampling: SamplingStrategy::Greedy,
        };
        let engine = MemristiveEngine::new(config);

        Self {
            engine,
            history: Vec::new(),
            consecutive_freeze: 0,
        }
    }

    /// Feed a new governance cycle observation
    pub fn observe(&mut self, state: GovernanceState) {
        self.engine.observe(state.as_index());
        self.history.push(state);

        if state == GovernanceState::Freeze {
            self.consecutive_freeze += 1;
        } else {
            self.consecutive_freeze = 0;
        }
    }

    /// Predict probability distribution over next states
    pub fn predict(&mut self) -> [f64; 4] {
        let predictions = self.engine.predict();
        let mut result = [0.0f64; 4];

        for (state_idx, prob) in predictions {
            if state_idx < 4 {
                result[state_idx] = prob;
            }
        }

        // Normalize
        let sum: f64 = result.iter().sum();
        if sum > 0.0 {
            for p in &mut result {
                *p /= sum;
            }
        } else {
            result = [0.25; 4];
        }
        result
    }

    /// Predict most likely next state
    pub fn predict_next(&mut self) -> GovernanceState {
        let probs = self.predict();
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        GovernanceState::from_index(max_idx).unwrap_or(GovernanceState::Healthy)
    }

    /// Probability of reaching freeze within N cycles
    pub fn freeze_probability(&mut self, horizon: usize) -> f64 {
        let probs = self.predict();
        let p_freeze = probs[GovernanceState::Freeze.as_index()];
        if horizon <= 1 {
            return p_freeze;
        }
        1.0 - (1.0 - p_freeze).powi(horizon as i32)
    }

    /// Get recommended intervention based on current prediction
    pub fn recommend(&mut self) -> &'static str {
        let probs = self.predict();
        let p_freeze = probs[GovernanceState::Freeze.as_index()];
        let p_warning = probs[GovernanceState::Warning.as_index()];

        if p_freeze > 0.3 {
            "URGENT: Execute existing artifacts, deprecate unused, freeze creation"
        } else if p_warning > 0.4 {
            "CAUTION: Review LOLLI/ERGOL ratio, run LOLLI lint, address dead bindings"
        } else if p_freeze + p_warning > 0.3 {
            "MONITOR: Trending toward degradation, schedule proactive recon"
        } else {
            "NOMINAL: Continue current approach"
        }
    }

    pub fn current_state(&self) -> Option<GovernanceState> {
        self.history.last().copied()
    }

    pub fn history(&self) -> &[GovernanceState] {
        &self.history
    }

    pub fn consecutive_freeze_count(&self) -> usize {
        self.consecutive_freeze
    }
}

impl Default for GovernanceMarkov {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_healthy() {
        assert_eq!(
            GovernanceState::classify(0.95, 1.0),
            GovernanceState::Healthy
        );
    }

    #[test]
    fn classify_watch() {
        assert_eq!(GovernanceState::classify(0.75, 2.0), GovernanceState::Watch);
    }

    #[test]
    fn classify_warning() {
        assert_eq!(
            GovernanceState::classify(0.6, 2.8),
            GovernanceState::Warning
        );
    }

    #[test]
    fn classify_freeze_by_ratio() {
        assert_eq!(GovernanceState::classify(0.8, 3.5), GovernanceState::Freeze);
    }

    #[test]
    fn classify_freeze_by_resilience() {
        assert_eq!(GovernanceState::classify(0.4, 1.0), GovernanceState::Freeze);
    }

    #[test]
    fn state_roundtrip() {
        for state in [
            GovernanceState::Healthy,
            GovernanceState::Watch,
            GovernanceState::Warning,
            GovernanceState::Freeze,
        ] {
            assert_eq!(GovernanceState::from_index(state.as_index()), Some(state));
        }
    }

    #[test]
    fn new_model_predicts() {
        let mut model = GovernanceMarkov::new();
        model.observe(GovernanceState::Healthy);
        let probs = model.predict();
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Probabilities must sum to ~1.0, got {}",
            sum
        );
    }

    #[test]
    fn observe_updates_history() {
        let mut model = GovernanceMarkov::new();
        model.observe(GovernanceState::Healthy);
        model.observe(GovernanceState::Watch);
        assert_eq!(model.history().len(), 2);
        assert_eq!(model.current_state(), Some(GovernanceState::Watch));
    }

    #[test]
    fn consecutive_freeze_tracking() {
        let mut model = GovernanceMarkov::new();
        model.observe(GovernanceState::Healthy);
        assert_eq!(model.consecutive_freeze_count(), 0);
        model.observe(GovernanceState::Freeze);
        assert_eq!(model.consecutive_freeze_count(), 1);
        model.observe(GovernanceState::Freeze);
        assert_eq!(model.consecutive_freeze_count(), 2);
        model.observe(GovernanceState::Watch);
        assert_eq!(model.consecutive_freeze_count(), 0);
    }

    #[test]
    fn labels_correct() {
        assert_eq!(GovernanceState::Healthy.label(), "healthy");
        assert_eq!(GovernanceState::Freeze.label(), "freeze");
    }

    #[test]
    fn real_governance_scenario() {
        let mut model = GovernanceMarkov::new();

        // chaos-001: R=0.0 → Freeze
        let cycle_001 = GovernanceState::classify(0.0, 3.0);
        assert_eq!(cycle_001, GovernanceState::Freeze);
        model.observe(cycle_001);

        // chaos-002: R=0.64 → Warning (R < 0.7 triggers Warning)
        let cycle_002 = GovernanceState::classify(0.64, 1.4);
        assert_eq!(cycle_002, GovernanceState::Warning);
        model.observe(cycle_002);

        let probs = model.predict();
        assert!(probs.iter().all(|p| *p >= 0.0 && *p <= 1.0));
    }
}
