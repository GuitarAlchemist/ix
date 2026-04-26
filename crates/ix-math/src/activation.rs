//! Activation functions and their derivatives.

use ndarray::Array1;

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

pub fn sigmoid_array(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(sigmoid)
}

/// ReLU: max(0, x)
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn relu_array(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(relu)
}

/// Leaky ReLU: max(alpha * x, x)
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

pub fn leaky_relu_derivative(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        alpha
    }
}

/// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
pub fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

pub fn tanh_derivative(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

pub fn tanh_array(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.tanh())
}

/// Softmax: exp(x_i) / sum(exp(x_j)) with numerical stability.
pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = x.mapv(|v| (v - max).exp());
    let sum = exps.sum();
    exps / sum
}

/// Linear (identity) activation.
pub fn linear(x: f64) -> f64 {
    x
}

pub fn linear_derivative(_x: f64) -> f64 {
    1.0
}

/// Swish: x * sigmoid(x)
pub fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

pub fn swish_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s + x * s * (1.0 - s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = array![1.0, 2.0, 3.0];
        let s = softmax(&x);
        assert!((s.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_numerically_stable() {
        let x = array![1000.0, 1001.0, 1002.0];
        let s = softmax(&x);
        assert!((s.sum() - 1.0).abs() < 1e-10);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_tanh_bounds() {
        assert!((tanh_act(0.0)).abs() < 1e-10);
        assert!(tanh_act(100.0) > 0.999);
        assert!(tanh_act(-100.0) < -0.999);
    }
}
