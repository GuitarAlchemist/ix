use crate::conductance::ConductanceMatrix;
use crate::consolidator::MemoryConsolidator;
use crate::sampler::SamplingStrategy;
use crate::tensor::MarkovTensor;
use crate::vlmm::{FallbackStrategy, VariableOrderSelector};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub max_order: usize,
    pub session_alpha: f64,
    pub session_beta: f64,
    pub long_term_alpha: f64,
    pub long_term_beta: f64,
    pub g_min: f64,
    pub min_observations: usize,
    pub fallback: FallbackStrategy,
    pub consolidation_gamma: f64,
    pub min_session_observations: usize,
    pub default_sampling: SamplingStrategy,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_order: 4,
            session_alpha: 0.1,
            session_beta: 0.01,
            long_term_alpha: 0.01,
            long_term_beta: 0.001,
            g_min: 0.01,
            min_observations: 5,
            fallback: FallbackStrategy::MarginalDistribution,
            consolidation_gamma: 0.1,
            min_session_observations: 20,
            default_sampling: SamplingStrategy::Temperature { t: 1.0 },
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EngineState {
    pub config: EngineConfig,
    pub tensor: MarkovTensor,
    pub session_conductance: ConductanceMatrix,
    pub long_term_conductance: Option<ConductanceMatrix>,
    pub vlmm: VariableOrderSelector,
    pub consolidator: MemoryConsolidator,
    pub context_buffer: Vec<usize>,
    pub total_observations: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_serializes_roundtrip() {
        let config = EngineConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let back: EngineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_order, config.max_order);
    }
}
