//! Defensive techniques against adversarial examples.

use ndarray::Array1;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Generate adversarial examples via FGSM for training augmentation.
///
/// Returns a new set of perturbed inputs to mix into training data.
pub fn adversarial_training_augment(
    inputs: &[Array1<f64>],
    gradients: &[Array1<f64>],
    epsilon: f64,
) -> Vec<Array1<f64>> {
    inputs
        .iter()
        .zip(gradients.iter())
        .map(|(x, g)| x + &(g.mapv(|v| v.signum()) * epsilon))
        .collect()
}

/// Input gradient regularization penalty.
///
/// Returns ‖gradient‖₂² as a penalty term to encourage smooth model outputs.
pub fn input_gradient_regularization(gradient: &Array1<f64>) -> f64 {
    gradient.mapv(|g| g * g).sum()
}

/// Statistical adversarial detection via input randomization.
///
/// Adds random noise to the input `n_samples` times and checks if the model
/// output variance exceeds `threshold`. High variance suggests the input sits
/// near a decision boundary — a hallmark of adversarial examples.
pub fn detect_adversarial(
    input: &Array1<f64>,
    model_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
    noise_std: f64,
    n_samples: usize,
    threshold: f64,
    seed: u64,
) -> bool {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise_std).unwrap();

    let base_output = model_fn(input);
    let dim = base_output.len();

    // accumulate variance across noisy samples
    let mut variance_sum = 0.0;
    for _ in 0..n_samples {
        let noise: Array1<f64> =
            Array1::from_iter((0..input.len()).map(|_| normal.sample(&mut rng)));
        let noisy_input = input + &noise;
        let output = model_fn(&noisy_input);
        let diff = &output - &base_output;
        variance_sum += diff.mapv(|d| d * d).sum();
    }
    let avg_variance = variance_sum / (n_samples as f64 * dim as f64);
    avg_variance > threshold
}

/// Feature squeezing defense.
///
/// Reduces input precision to `bit_depth` bits, eliminating small adversarial
/// perturbations that fall below the quantization resolution.
pub fn feature_squeezing(input: &Array1<f64>, bit_depth: u32) -> Array1<f64> {
    let levels = (1u64 << bit_depth) as f64 - 1.0;
    input.mapv(|v| {
        let clamped = v.clamp(0.0, 1.0);
        (clamped * levels).round() / levels
    })
}

/// Spatial smoothing defense via moving-average filter.
///
/// Applies a 1-D averaging kernel of `kernel_size` after clamping values to
/// `[min_val, max_val]`. Smoothing removes high-frequency adversarial noise.
pub fn clip_and_smooth(
    input: &Array1<f64>,
    min_val: f64,
    max_val: f64,
    kernel_size: usize,
) -> Array1<f64> {
    let clamped = input.mapv(|v| v.clamp(min_val, max_val));
    let n = clamped.len();
    if kernel_size <= 1 || n == 0 {
        return clamped;
    }
    let half = kernel_size / 2;
    Array1::from_iter((0..n).map(|i| {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let count = (end - start) as f64;
        clamped.slice(ndarray::s![start..end]).sum() / count
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adversarial_training_augment() {
        let inputs = vec![array![0.5, 0.5], array![0.3, 0.7]];
        let gradients = vec![array![1.0, -1.0], array![-0.5, 0.2]];
        let augmented = adversarial_training_augment(&inputs, &gradients, 0.1);
        assert_eq!(augmented.len(), 2);
        assert!((augmented[0][0] - 0.6).abs() < 1e-10);
        assert!((augmented[0][1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_adversarial_training_augment_empty() {
        let result = adversarial_training_augment(&[], &[], 0.1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_gradient_regularization() {
        let grad = array![3.0, 4.0];
        let penalty = input_gradient_regularization(&grad);
        assert!((penalty - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_regularization_zero() {
        let grad = array![0.0, 0.0, 0.0];
        assert!((input_gradient_regularization(&grad)).abs() < 1e-10);
    }

    #[test]
    fn test_detect_adversarial_stable_model() {
        // constant model -> no variance -> not adversarial
        let input = array![0.5, 0.5];
        let detected = detect_adversarial(&input, |_x| array![1.0, 0.0], 0.1, 50, 0.01, 42);
        assert!(!detected);
    }

    #[test]
    fn test_detect_adversarial_sensitive_model() {
        // model that amplifies input -> high variance on noisy inputs
        let input = array![0.5, 0.5];
        let detected = detect_adversarial(&input, |x| x * 100.0, 1.0, 100, 0.001, 42);
        assert!(detected);
    }

    #[test]
    fn test_feature_squeezing_1bit() {
        let input = array![0.3, 0.7, 0.0, 1.0];
        let squeezed = feature_squeezing(&input, 1);
        // 1 bit -> levels=1 -> values snap to 0.0 or 1.0
        assert!((squeezed[0] - 0.0).abs() < 1e-10);
        assert!((squeezed[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_squeezing_high_bit() {
        let input = array![0.5];
        let squeezed = feature_squeezing(&input, 8);
        assert!((squeezed[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_clip_and_smooth_identity() {
        let input = array![0.2, 0.5, 0.8];
        let result = clip_and_smooth(&input, 0.0, 1.0, 1);
        assert!((result[0] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_clip_and_smooth_averaging() {
        let input = array![0.0, 1.0, 0.0];
        let result = clip_and_smooth(&input, 0.0, 1.0, 3);
        // middle element: average of [0, 1, 0] = 0.333..
        assert!((result[1] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_and_smooth_clamps() {
        let input = array![-1.0, 2.0];
        let result = clip_and_smooth(&input, 0.0, 1.0, 1);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }
}
