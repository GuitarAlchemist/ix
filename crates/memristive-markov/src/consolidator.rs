use crate::conductance::ConductanceMatrix;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConsolidator {
    gamma: f64,
    min_session_observations: usize,
    decay_on_consolidate: bool,
}

impl MemoryConsolidator {
    pub fn new(gamma: f64, min_session_observations: usize) -> Self {
        Self {
            gamma,
            min_session_observations,
            decay_on_consolidate: true,
        }
    }

    /// Transfer session conductance into long-term.
    /// Returns true if consolidation was performed.
    pub fn consolidate(
        &self,
        session: &ConductanceMatrix,
        long_term: &mut ConductanceMatrix,
        session_observations: usize,
    ) -> bool {
        if session_observations < self.min_session_observations {
            return false;
        }
        let gamma = self.gamma;
        session.for_each(|context, next, session_g| {
            let lt_g = long_term.get(context, next);
            let blended = (1.0 - gamma) * lt_g + gamma * session_g;
            long_term.set(context, next, blended);
        });
        if self.decay_on_consolidate {
            long_term.decay_all();
        }
        true
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidation_blends() {
        let mut session = ConductanceMatrix::new(0.5, 0.01, 0.01);
        for _ in 0..10 {
            session.strengthen(&[0], 1);
        }
        let session_g = session.get(&[0], 1);
        let mut long_term = ConductanceMatrix::new(0.01, 0.001, 0.01);
        let consolidator = MemoryConsolidator::new(0.1, 5);
        assert!(consolidator.consolidate(&session, &mut long_term, 10));
        let lt_g = long_term.get(&[0], 1);
        assert!(lt_g > 0.01 && lt_g < session_g, "Blended: {}", lt_g);
    }

    #[test]
    fn test_consolidation_skips_below_threshold() {
        let session = ConductanceMatrix::new(0.5, 0.01, 0.01);
        let mut long_term = ConductanceMatrix::new(0.01, 0.001, 0.01);
        let consolidator = MemoryConsolidator::new(0.1, 20);
        assert!(!consolidator.consolidate(&session, &mut long_term, 5));
    }
}
