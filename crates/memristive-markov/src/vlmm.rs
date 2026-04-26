use crate::tensor::MarkovTensor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FallbackStrategy {
    Uniform,
    MarginalDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableOrderSelector {
    min_observations: usize,
    max_order: usize,
    fallback: FallbackStrategy,
    order_counts: Vec<u64>,
}

impl VariableOrderSelector {
    pub fn new(max_order: usize, min_observations: usize, fallback: FallbackStrategy) -> Self {
        Self {
            min_observations,
            max_order,
            fallback,
            order_counts: vec![0; max_order + 1],
        }
    }

    pub fn predict(&mut self, tensor: &MarkovTensor, context: &[usize]) -> Vec<(usize, f64)> {
        let max_k = context.len().min(self.max_order);
        for k in (1..=max_k).rev() {
            let ctx = &context[context.len() - k..];
            if tensor.context_count(ctx) >= self.min_observations as f64 {
                self.order_counts[k] += 1;
                return tensor.predict(ctx);
            }
        }
        self.order_counts[0] += 1;
        match self.fallback {
            FallbackStrategy::MarginalDistribution => tensor.predict(&[]),
            FallbackStrategy::Uniform => {
                let n = tensor.state_count();
                if n == 0 {
                    return vec![];
                }
                let p = 1.0 / n as f64;
                (0..n).map(|s| (s, p)).collect()
            }
        }
    }

    pub fn effective_order(&self, tensor: &MarkovTensor, context: &[usize]) -> usize {
        let max_k = context.len().min(self.max_order);
        for k in (1..=max_k).rev() {
            let ctx = &context[context.len() - k..];
            if tensor.context_count(ctx) >= self.min_observations as f64 {
                return k;
            }
        }
        0
    }

    pub fn order_histogram(&self) -> &[u64] {
        &self.order_counts
    }
    pub fn reset_counts(&mut self) {
        self.order_counts.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vlmm_uses_highest_order_with_data() {
        let mut t = MarkovTensor::new(3);
        for _ in 0..10 {
            t.observe(&[0, 1], 2);
        }
        let mut vlmm = VariableOrderSelector::new(3, 5, FallbackStrategy::Uniform);
        let dist = vlmm.predict(&t, &[0, 1]);
        assert!(!dist.is_empty());
        assert_eq!(vlmm.effective_order(&t, &[0, 1]), 2);
    }

    #[test]
    fn test_vlmm_falls_back_on_sparse_data() {
        let mut t = MarkovTensor::new(3);
        t.observe(&[0, 1], 2);
        t.observe(&[0, 1], 3);
        for _ in 0..8 {
            t.observe(&[1], 2);
        }
        let mut vlmm = VariableOrderSelector::new(3, 5, FallbackStrategy::MarginalDistribution);
        let _dist = vlmm.predict(&t, &[0, 1]);
        assert_eq!(vlmm.effective_order(&t, &[0, 1]), 1);
    }

    #[test]
    fn test_vlmm_uniform_fallback() {
        let mut t = MarkovTensor::new(2);
        t.observe(&[0], 1);
        let mut vlmm = VariableOrderSelector::new(2, 5, FallbackStrategy::Uniform);
        let dist = vlmm.predict(&t, &[99]);
        assert!(dist.len() >= 2);
    }

    #[test]
    fn test_order_histogram_tracks_usage() {
        let mut t = MarkovTensor::new(2);
        for _ in 0..10 {
            t.observe(&[0, 1], 2);
        }
        let mut vlmm = VariableOrderSelector::new(2, 5, FallbackStrategy::Uniform);
        vlmm.predict(&t, &[0, 1]);
        vlmm.predict(&t, &[0, 1]);
        assert_eq!(vlmm.order_histogram()[2], 2);
    }
}
