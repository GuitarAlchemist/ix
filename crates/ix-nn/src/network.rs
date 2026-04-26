//! Sequential neural network.
//!
//! Composes layers into a trainable network with forward and backward passes.

use crate::layer::Layer;
use ndarray::Array2;

/// Sequential network — a stack of layers executed in order.
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer to the network.
    pub fn push(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Forward pass through all layers.
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Backward pass through all layers in reverse.
    /// Returns the gradient with respect to the input.
    pub fn backward(&mut self, grad_output: &Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, learning_rate);
        }
        grad
    }

    /// Train the network for one epoch on a batch.
    /// Returns the loss value.
    pub fn train_step(
        &mut self,
        input: &Array2<f64>,
        target: &Array2<f64>,
        learning_rate: f64,
        loss_fn: fn(&Array2<f64>, &Array2<f64>) -> f64,
        loss_grad_fn: fn(&Array2<f64>, &Array2<f64>) -> Array2<f64>,
    ) -> f64 {
        let output = self.forward(input);
        let loss = loss_fn(&output, target);
        let grad = loss_grad_fn(&output, target);
        self.backward(&grad, learning_rate);
        loss
    }

    /// Train for multiple epochs, returning loss history.
    pub fn fit(
        &mut self,
        input: &Array2<f64>,
        target: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
        loss_fn: fn(&Array2<f64>, &Array2<f64>) -> f64,
        loss_grad_fn: fn(&Array2<f64>, &Array2<f64>) -> Array2<f64>,
    ) -> Vec<f64> {
        (0..epochs)
            .map(|_| self.train_step(input, target, learning_rate, loss_fn, loss_grad_fn))
            .collect()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Dense;
    use crate::loss::{mse_gradient, mse_loss};
    use ndarray::array;

    #[test]
    fn test_sequential_forward_shape() {
        let mut net = Sequential::new()
            .push(Box::new(Dense::new(3, 4)))
            .push(Box::new(Dense::new(4, 2)));

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let output = net.forward(&input);
        assert_eq!(output.dim(), (2, 2));
    }

    #[test]
    fn test_sequential_train_loss_decreases() {
        let mut net = Sequential::new()
            .push(Box::new(Dense::new(2, 4)))
            .push(Box::new(Dense::new(4, 1)));

        // Simple regression: y ≈ x1 + x2
        let input = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let target = array![[3.0], [7.0], [11.0], [15.0]];

        let losses = net.fit(&input, &target, 200, 0.01, mse_loss, mse_gradient);

        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "loss should decrease: first={}, last={}",
            losses[0],
            losses.last().unwrap()
        );
    }

    #[test]
    fn test_sequential_len() {
        let net = Sequential::new()
            .push(Box::new(Dense::new(3, 4)))
            .push(Box::new(Dense::new(4, 2)));
        assert_eq!(net.len(), 2);
        assert!(!net.is_empty());
    }
}
