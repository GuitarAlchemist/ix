//! Model privacy — defenses against extraction and inversion attacks.

use ndarray::Array1;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Confidence-based membership inference metric.
///
/// Returns a score indicating how likely `point` was in the training set,
/// based on the model's confidence on the true label. Higher confidence
/// suggests membership.
pub fn membership_inference_score(
    model_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
    point: &Array1<f64>,
    true_label: usize,
) -> f64 {
    let output = model_fn(point);
    if true_label < output.len() {
        output[true_label]
    } else {
        0.0
    }
}

/// Gaussian mechanism differential privacy noise for gradients.
///
/// Adds calibrated Gaussian noise to satisfy (ε, δ)-differential privacy.
/// σ = sensitivity · √(2 ln(1.25/δ)) / ε
pub fn differential_privacy_noise(
    gradient: &Array1<f64>,
    epsilon: f64,
    delta: f64,
    sensitivity: f64,
    seed: u64,
) -> Array1<f64> {
    let sigma = sensitivity * (2.0 * (1.25 / delta).ln()).sqrt() / epsilon;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, sigma).unwrap();
    let noise: Array1<f64> =
        Array1::from_iter((0..gradient.len()).map(|_| normal.sample(&mut rng)));
    gradient + &noise
}

/// Temperature scaling to reduce information leakage.
///
/// Divides logits by `temperature` before applying softmax, producing softer
/// (less informative) probability distributions when temperature > 1.
pub fn model_confidence_masking(logits: &Array1<f64>, temperature: f64) -> Array1<f64> {
    let temp = temperature.max(1e-12);
    let scaled = logits / temp;
    // numerically stable softmax
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = scaled.mapv(|v| (v - max_val).exp());
    let sum = exps.sum();
    exps / sum
}

/// Prediction purification — zero out all but top-k predictions.
///
/// Keeps only the `top_k` highest values, setting the rest to zero.
/// Reduces the amount of information an attacker can extract from outputs.
pub fn prediction_purification(output: &Array1<f64>, top_k: usize) -> Array1<f64> {
    let n = output.len();
    if top_k >= n {
        return output.clone();
    }
    // find the top-k threshold
    let mut sorted_vals: Vec<f64> = output.iter().cloned().collect();
    sorted_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted_vals[top_k.min(n.saturating_sub(1))];

    // zero out values below threshold, keeping exactly top_k
    let mut result = Array1::<f64>::zeros(n);
    let mut kept = 0;
    // first pass: keep values strictly above threshold
    for i in 0..n {
        if output[i] > threshold {
            result[i] = output[i];
            kept += 1;
        }
    }
    // second pass: fill remaining slots with values equal to threshold
    for i in 0..n {
        if kept >= top_k {
            break;
        }
        if output[i] == threshold && result[i] == 0.0 {
            result[i] = output[i];
            kept += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_membership_inference_high_confidence() {
        let score = membership_inference_score(|_x| array![0.1, 0.9, 0.0], &array![1.0, 2.0], 1);
        assert!((score - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_membership_inference_out_of_range() {
        let score = membership_inference_score(
            |_x| array![0.5, 0.5],
            &array![1.0],
            5, // out of range
        );
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dp_noise_changes_gradient() {
        let grad = array![1.0, 2.0, 3.0];
        let noisy = differential_privacy_noise(&grad, 1.0, 1e-5, 1.0, 42);
        assert_eq!(noisy.len(), 3);
        // noise should make it differ from original
        let diff = (&noisy - &grad).mapv(|d| d.abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_dp_noise_deterministic_with_seed() {
        let grad = array![1.0, 2.0];
        let a = differential_privacy_noise(&grad, 1.0, 1e-5, 1.0, 123);
        let b = differential_privacy_noise(&grad, 1.0, 1e-5, 1.0, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn test_confidence_masking_temperature_1() {
        let logits = array![2.0, 1.0, 0.0];
        let probs = model_confidence_masking(&logits, 1.0);
        assert!((probs.sum() - 1.0).abs() < 1e-10);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_confidence_masking_high_temperature() {
        let logits = array![10.0, 0.0];
        let sharp = model_confidence_masking(&logits, 1.0);
        let smooth = model_confidence_masking(&logits, 100.0);
        // High temperature should produce more uniform distribution
        let sharp_diff = (sharp[0] - sharp[1]).abs();
        let smooth_diff = (smooth[0] - smooth[1]).abs();
        assert!(smooth_diff < sharp_diff);
    }

    #[test]
    fn test_confidence_masking_sums_to_one() {
        let logits = array![1.0, 2.0, 3.0, 4.0];
        let probs = model_confidence_masking(&logits, 2.5);
        assert!((probs.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_purification_top1() {
        let output = array![0.1, 0.7, 0.2];
        let purified = prediction_purification(&output, 1);
        assert!((purified[0] - 0.0).abs() < 1e-10);
        assert!((purified[1] - 0.7).abs() < 1e-10);
        assert!((purified[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_purification_top_k_ge_n() {
        let output = array![0.3, 0.7];
        let purified = prediction_purification(&output, 5);
        assert_eq!(purified, output);
    }

    #[test]
    fn test_prediction_purification_top2() {
        let output = array![0.1, 0.5, 0.3, 0.1];
        let purified = prediction_purification(&output, 2);
        let nonzero: Vec<_> = purified.iter().filter(|&&v| v > 0.0).collect();
        assert_eq!(nonzero.len(), 2);
        assert!((purified[1] - 0.5).abs() < 1e-10);
        assert!((purified[2] - 0.3).abs() < 1e-10);
    }
}
