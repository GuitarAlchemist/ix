use crate::conductance::ConductanceMatrix;
use crate::consolidator::MemoryConsolidator;
use crate::error::Result;
use crate::serde_state::{EngineConfig, EngineState};
use crate::tensor::MarkovTensor;
use crate::vlmm::VariableOrderSelector;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Serialize, Deserialize)]
pub struct EngineDiagnostics {
    pub total_observations: u64,
    pub unique_states: usize,
    pub tensor_sparsity: f64,
    pub order_histogram: Vec<u64>,
    pub session_observations: u64,
}

pub struct MemristiveEngine {
    tensor: MarkovTensor,
    session_conductance: ConductanceMatrix,
    long_term_conductance: Option<ConductanceMatrix>,
    vlmm: VariableOrderSelector,
    consolidator: MemoryConsolidator,
    context_buffer: VecDeque<usize>,
    config: EngineConfig,
    total_observations: u64,
    session_observations: u64,
}

impl MemristiveEngine {
    pub fn new(config: EngineConfig) -> Self {
        let tensor = MarkovTensor::new(config.max_order);
        let session_conductance =
            ConductanceMatrix::new(config.session_alpha, config.session_beta, config.g_min);
        let long_term_conductance = Some(ConductanceMatrix::new(
            config.long_term_alpha,
            config.long_term_beta,
            config.g_min,
        ));
        let vlmm =
            VariableOrderSelector::new(config.max_order, config.min_observations, config.fallback);
        let consolidator =
            MemoryConsolidator::new(config.consolidation_gamma, config.min_session_observations);
        Self {
            tensor,
            session_conductance,
            long_term_conductance,
            vlmm,
            consolidator,
            context_buffer: VecDeque::with_capacity(config.max_order),
            config,
            total_observations: 0,
            session_observations: 0,
        }
    }

    pub fn from_state(json: &str) -> Result<Self> {
        let state: EngineState = serde_json::from_str(json)?;
        Ok(Self {
            tensor: state.tensor,
            session_conductance: state.session_conductance,
            long_term_conductance: state.long_term_conductance,
            vlmm: state.vlmm,
            consolidator: state.consolidator,
            context_buffer: VecDeque::from(state.context_buffer),
            config: state.config,
            total_observations: state.total_observations,
            session_observations: 0,
        })
    }

    pub fn observe(&mut self, state: usize) {
        if !self.context_buffer.is_empty() {
            let context: Vec<usize> = self.context_buffer.iter().copied().collect();
            self.tensor.observe(&context, state);
            self.session_conductance.strengthen(&context, state);
        }
        self.context_buffer.push_back(state);
        if self.context_buffer.len() > self.config.max_order {
            self.context_buffer.pop_front();
        }
        self.total_observations += 1;
        self.session_observations += 1;
    }

    pub fn observe_sequence(&mut self, states: &[usize]) {
        for &s in states {
            self.observe(s);
        }
    }

    pub fn predict(&mut self) -> Vec<(usize, f64)> {
        let context: Vec<usize> = self.context_buffer.iter().copied().collect();
        let base = self.vlmm.predict(&self.tensor, &context);
        let modulated = self.session_conductance.modulate(&context, &base);
        if let Some(lt) = &self.long_term_conductance {
            let lt_modulated = lt.modulate(&context, &base);
            let blended: Vec<(usize, f64)> = modulated
                .iter()
                .map(|&(s, sp)| {
                    let lp = lt_modulated
                        .iter()
                        .find(|(ls, _)| *ls == s)
                        .map(|(_, p)| *p)
                        .unwrap_or(0.0);
                    (s, 0.7 * sp + 0.3 * lp)
                })
                .collect();
            let total: f64 = blended.iter().map(|(_, p)| p).sum();
            if total > 0.0 {
                return blended.into_iter().map(|(s, p)| (s, p / total)).collect();
            }
        }
        modulated
    }

    pub fn sample(&mut self, rng: &mut impl Rng) -> Option<usize> {
        let dist = self.predict();
        self.config.default_sampling.sample(&dist, rng)
    }

    pub fn consolidate(&mut self) {
        if let Some(lt) = &mut self.long_term_conductance {
            self.consolidator.consolidate(
                &self.session_conductance,
                lt,
                self.session_observations as usize,
            );
        }
        self.session_conductance = ConductanceMatrix::new(
            self.config.session_alpha,
            self.config.session_beta,
            self.config.g_min,
        );
        self.session_observations = 0;
    }

    pub fn reset_session(&mut self) {
        self.session_conductance = ConductanceMatrix::new(
            self.config.session_alpha,
            self.config.session_beta,
            self.config.g_min,
        );
        self.context_buffer.clear();
        self.session_observations = 0;
    }

    pub fn export_state(&self) -> String {
        let state = EngineState {
            config: self.config.clone(),
            tensor: self.tensor.clone(),
            session_conductance: self.session_conductance.clone(),
            long_term_conductance: self.long_term_conductance.clone(),
            vlmm: self.vlmm.clone(),
            consolidator: self.consolidator.clone(),
            context_buffer: self.context_buffer.iter().copied().collect(),
            total_observations: self.total_observations,
        };
        serde_json::to_string_pretty(&state).expect("engine state serialization should not fail")
    }

    pub fn diagnostics(&self) -> EngineDiagnostics {
        EngineDiagnostics {
            total_observations: self.total_observations,
            unique_states: self.tensor.state_count(),
            tensor_sparsity: self.tensor.sparsity(),
            order_histogram: self.vlmm.order_histogram().to_vec(),
            session_observations: self.session_observations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_engine_observe_and_predict() {
        let mut engine = MemristiveEngine::new(EngineConfig::default());
        for _ in 0..20 {
            engine.observe(0);
            engine.observe(1);
            engine.observe(2);
        }
        let dist = engine.predict();
        assert!(!dist.is_empty());
        let p0 = dist
            .iter()
            .find(|(s, _)| *s == 0)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        assert!(p0 > 0.5, "After 2, should predict 0: {}", p0);
    }

    #[test]
    fn test_engine_sample_returns_valid_state() {
        let mut engine = MemristiveEngine::new(EngineConfig::default());
        engine.observe_sequence(&[0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert!(engine.sample(&mut test_rng()).is_some());
    }

    #[test]
    fn test_engine_state_roundtrip() {
        let mut engine = MemristiveEngine::new(EngineConfig::default());
        engine.observe_sequence(&[0, 1, 2, 3, 4]);
        let json = engine.export_state();
        let restored = MemristiveEngine::from_state(&json).unwrap();
        assert_eq!(restored.diagnostics().total_observations, 5);
    }

    #[test]
    fn test_consolidation_resets_session() {
        let mut engine = MemristiveEngine::new(EngineConfig {
            min_session_observations: 5,
            ..EngineConfig::default()
        });
        engine.observe_sequence(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(engine.diagnostics().session_observations, 10);
        engine.consolidate();
        assert_eq!(engine.diagnostics().session_observations, 0);
    }

    #[test]
    fn test_reset_session_discards() {
        let mut engine = MemristiveEngine::new(EngineConfig::default());
        engine.observe_sequence(&[0, 1, 2, 3, 4]);
        engine.reset_session();
        assert_eq!(engine.diagnostics().session_observations, 0);
    }

    #[test]
    fn test_conductance_biases_prediction() {
        let config = EngineConfig {
            session_alpha: 0.5,
            min_observations: 1,
            max_order: 1, // use order 1 for simpler context
            ..EngineConfig::default()
        };
        let mut engine = MemristiveEngine::new(config);
        // Equal base: after 0, both 1 and 2 occur once
        engine.observe_sequence(&[0, 1, 0, 2]);
        // Heavily reinforce 0→1
        for _ in 0..30 {
            engine.observe_sequence(&[0, 1]);
        }
        // Context is now [1], predict after 0:
        engine.observe(0);
        let dist = engine.predict();
        let p1 = dist
            .iter()
            .find(|(s, _)| *s == 1)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        let p2 = dist
            .iter()
            .find(|(s, _)| *s == 2)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        assert!(
            p1 > p2,
            "Conductance should bias toward 1: p1={}, p2={}",
            p1,
            p2
        );
    }
}
