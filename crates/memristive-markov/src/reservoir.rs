use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Uniform;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirNetwork {
    size: usize,
    weights: Array2<f64>,
    input_weights: Array2<f64>,
    state: Array1<f64>,
    spectral_radius: f64,
    sparsity: f64,
    leak_rate: f64,
    readout: Option<Array2<f64>>,
}

impl ReservoirNetwork {
    pub fn new(
        size: usize,
        input_dim: usize,
        spectral_radius: f64,
        sparsity: f64,
        leak_rate: f64,
        rng: &mut impl Rng,
    ) -> Self {
        let dist = Uniform::new(-1.0, 1.0).unwrap();
        let mut weights = Array2::zeros((size, size));
        for w in weights.iter_mut() {
            if rng.random::<f64>() > sparsity {
                *w = rng.sample(dist);
            }
        }
        let max_val = weights
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        if max_val > 0.0 {
            weights.mapv_inplace(|x| x * spectral_radius / max_val);
        }
        let input_weights = Array2::from_shape_fn((input_dim, size), |_| rng.sample(dist) * 0.1);
        Self {
            size,
            weights,
            input_weights,
            state: Array1::zeros(size),
            spectral_radius,
            sparsity,
            leak_rate,
            readout: None,
        }
    }

    pub fn step(&mut self, input: &Array1<f64>) -> &Array1<f64> {
        let pre_activation = input.dot(&self.input_weights) + self.state.dot(&self.weights);
        self.state =
            (1.0 - self.leak_rate) * &self.state + self.leak_rate * pre_activation.mapv(f64::tanh);
        &self.state
    }

    pub fn state(&self) -> &Array1<f64> {
        &self.state
    }
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
    pub fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_reservoir_creation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let r = ReservoirNetwork::new(64, 4, 0.9, 0.9, 0.3, &mut rng);
        assert_eq!(r.size(), 64);
        assert_eq!(r.state().len(), 64);
    }

    #[test]
    fn test_reservoir_step_changes_state() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut r = ReservoirNetwork::new(32, 4, 0.9, 0.9, 0.3, &mut rng);
        let input = Array1::from_vec(vec![1.0, 0.0, 0.5, -0.5]);
        let initial_norm: f64 = r.state().iter().map(|x| x * x).sum();
        r.step(&input);
        let after_norm: f64 = r.state().iter().map(|x| x * x).sum();
        assert!(after_norm > initial_norm);
    }

    #[test]
    fn test_reservoir_state_bounded() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut r = ReservoirNetwork::new(32, 4, 0.9, 0.9, 0.3, &mut rng);
        let input = Array1::from_vec(vec![100.0, -100.0, 50.0, -50.0]);
        for _ in 0..100 {
            r.step(&input);
        }
        for &v in r.state().iter() {
            assert!(v.abs() <= 1.0, "Unbounded: {}", v);
        }
    }

    #[test]
    fn test_reservoir_reset() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut r = ReservoirNetwork::new(32, 4, 0.9, 0.9, 0.3, &mut rng);
        r.step(&Array1::ones(4));
        r.reset();
        assert!(r.state().iter().all(|&v| v == 0.0));
    }
}
