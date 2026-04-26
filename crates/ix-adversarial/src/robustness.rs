//! Robustness evaluation metrics for adversarial ML.

use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Summary of model robustness against adversarial attack.
#[derive(Debug, Clone)]
pub struct RobustnessReport {
    /// Accuracy on clean (unperturbed) inputs.
    pub clean_accuracy: f64,
    /// Accuracy on adversarially perturbed inputs.
    pub adversarial_accuracy: f64,
    /// Mean L2 norm of adversarial perturbations.
    pub mean_perturbation_norm: f64,
    /// Minimum L2 norm of adversarial perturbations.
    pub min_perturbation_norm: f64,
}

/// Empirical robustness evaluation.
///
/// Measures how well `model_fn` performs on clean vs adversarially perturbed
/// inputs. `attack_fn` generates an adversarial version of each input.
pub fn empirical_robustness(
    inputs: &[Array1<f64>],
    labels: &[usize],
    model_fn: impl Fn(&Array1<f64>) -> usize,
    attack_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
) -> RobustnessReport {
    let n = inputs.len();
    if n == 0 {
        return RobustnessReport {
            clean_accuracy: 0.0,
            adversarial_accuracy: 0.0,
            mean_perturbation_norm: 0.0,
            min_perturbation_norm: 0.0,
        };
    }

    let mut clean_correct = 0usize;
    let mut adv_correct = 0usize;
    let mut total_norm = 0.0;
    let mut min_norm = f64::MAX;

    for (x, &y) in inputs.iter().zip(labels.iter()) {
        // clean prediction
        if model_fn(x) == y {
            clean_correct += 1;
        }
        // adversarial prediction
        let adv = attack_fn(x);
        if model_fn(&adv) == y {
            adv_correct += 1;
        }
        let perturbation = &adv - x;
        let norm = perturbation.mapv(|v| v * v).sum().sqrt();
        total_norm += norm;
        if norm < min_norm {
            min_norm = norm;
        }
    }

    RobustnessReport {
        clean_accuracy: clean_correct as f64 / n as f64,
        adversarial_accuracy: adv_correct as f64 / n as f64,
        mean_perturbation_norm: total_norm / n as f64,
        min_perturbation_norm: if min_norm == f64::MAX { 0.0 } else { min_norm },
    }
}

/// Estimate the local Lipschitz constant of a model around a point.
///
/// Samples `n_samples` random neighbours within `radius` and computes the
/// maximum ratio ‖f(x') - f(x)‖ / ‖x' - x‖.
pub fn lipschitz_estimate(
    model_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
    point: &Array1<f64>,
    n_samples: usize,
    radius: f64,
    seed: u64,
) -> f64 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let f_x = model_fn(point);
    let mut max_ratio = 0.0f64;

    for _ in 0..n_samples {
        // sample random direction and scale to within radius
        let direction: Array1<f64> =
            Array1::from_iter((0..point.len()).map(|_| normal.sample(&mut rng)));
        let dir_norm = direction.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let unit_dir = &direction / dir_norm;

        // uniform distance within radius
        let scale = radius * (rng.random::<f64>()).powf(1.0 / point.len() as f64);
        let delta = &unit_dir * scale;
        let x_prime = point + &delta;

        let f_x_prime = model_fn(&x_prime);
        let output_diff = (&f_x_prime - &f_x).mapv(|v| v * v).sum().sqrt();
        let input_diff = delta.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let ratio = output_diff / input_diff;
        if ratio > max_ratio {
            max_ratio = ratio;
        }
    }
    max_ratio
}

/// Certified L2 radius via randomized smoothing (Cohen et al., 2019).
///
/// Given the logits (pre-softmax outputs) from a smoothed classifier and the
/// noise level `sigma`, returns the certified radius within which the
/// prediction is guaranteed not to change.
pub fn certified_radius(logits: &Array1<f64>, sigma: f64) -> f64 {
    if logits.len() < 2 {
        return 0.0;
    }
    // find top two classes
    let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
    sorted_indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    let p_a = logits[sorted_indices[0]];
    let p_b = logits[sorted_indices[1]];

    // Certified radius = σ/2 · (Φ⁻¹(pA) - Φ⁻¹(pB))
    // For softmax probabilities, approximate Φ⁻¹ using probit approximation
    // Simple case: if logits represent probabilities, use log-ratio
    // r = σ · (pA - pB) / 2  (simplified bound)
    if p_a <= p_b {
        return 0.0;
    }

    // Use the inverse normal CDF approximation for better accuracy
    let p_a_clamped = p_a.clamp(1e-10, 1.0 - 1e-10);
    let p_b_clamped = p_b.clamp(1e-10, 1.0 - 1e-10);

    let phi_inv_a = probit(p_a_clamped);
    let phi_inv_b = probit(p_b_clamped);

    let radius = sigma / 2.0 * (phi_inv_a - phi_inv_b);
    radius.max(0.0)
}

/// Rational approximation of the probit function (inverse normal CDF).
fn probit(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm (simplified)
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    // rational approximation constants
    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;

    let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 {
        -val
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_empirical_robustness_perfect_model() {
        let inputs = vec![array![0.0, 0.0], array![1.0, 1.0]];
        let labels = vec![0, 1];
        // model always correct, attack does nothing
        let report = empirical_robustness(
            &inputs,
            &labels,
            |x| if x[0] < 0.5 { 0 } else { 1 },
            |x| x.clone(),
        );
        assert!((report.clean_accuracy - 1.0).abs() < 1e-10);
        assert!((report.adversarial_accuracy - 1.0).abs() < 1e-10);
        assert!((report.mean_perturbation_norm - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_robustness_attack_flips() {
        let inputs = vec![array![0.0], array![1.0]];
        let labels = vec![0, 1];
        let report = empirical_robustness(
            &inputs,
            &labels,
            |x| if x[0] < 0.5 { 0 } else { 1 },
            |x| x.mapv(|v| 1.0 - v), // flip
        );
        assert!((report.clean_accuracy - 1.0).abs() < 1e-10);
        assert!((report.adversarial_accuracy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_robustness_empty() {
        let report = empirical_robustness(&[], &[], |_| 0, |x| x.clone());
        assert!((report.clean_accuracy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lipschitz_estimate_identity() {
        // Identity function has Lipschitz constant 1
        let lip = lipschitz_estimate(|x| x.clone(), &array![0.0, 0.0], 200, 1.0, 42);
        assert!((lip - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_lipschitz_estimate_scaled() {
        // f(x) = 3x has Lipschitz constant 3
        let lip = lipschitz_estimate(|x| x * 3.0, &array![0.0, 0.0], 200, 1.0, 42);
        assert!((lip - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_lipschitz_estimate_constant() {
        // constant function has Lipschitz constant 0
        let lip = lipschitz_estimate(|_| array![1.0], &array![0.0], 50, 1.0, 42);
        assert!(lip < 1e-10);
    }

    #[test]
    fn test_certified_radius_high_confidence() {
        // Very confident prediction -> larger certified radius
        let logits = array![0.99, 0.01];
        let r = certified_radius(&logits, 1.0);
        assert!(r > 0.0);
    }

    #[test]
    fn test_certified_radius_equal_logits() {
        let logits = array![0.5, 0.5];
        let r = certified_radius(&logits, 1.0);
        assert!((r - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_certified_radius_single_class() {
        let logits = array![0.9];
        let r = certified_radius(&logits, 1.0);
        assert!((r - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_probit_symmetry() {
        let a = probit(0.3);
        let b = probit(0.7);
        assert!((a + b).abs() < 0.01); // approximately symmetric
    }
}
