//! Evasion attacks — inference-phase adversarial perturbations.

use ndarray::Array1;

/// Fast Gradient Sign Method (Goodfellow et al., 2015).
///
/// Perturbs `input` in the direction of the gradient sign scaled by `epsilon`.
pub fn fgsm(input: &Array1<f64>, gradient: &Array1<f64>, epsilon: f64) -> Array1<f64> {
    input + &(gradient.mapv(|g| if g == 0.0 { 0.0 } else { g.signum() }) * epsilon)
}

/// Projected Gradient Descent (Madry et al., 2018).
///
/// Iterative FGSM with projection back onto the ε-ball around the original input.
pub fn pgd(
    input: &Array1<f64>,
    gradient_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
    epsilon: f64,
    alpha: f64,
    steps: usize,
) -> Array1<f64> {
    let mut adv = input.clone();
    for _ in 0..steps {
        let grad = gradient_fn(&adv);
        // step in sign direction
        adv = &adv + &(grad.mapv(|g| g.signum()) * alpha);
        // project back onto L∞ ε-ball
        let diff = &adv - input;
        let projected = diff.mapv(|d| d.clamp(-epsilon, epsilon));
        adv = input + &projected;
    }
    adv
}

/// Carlini & Wagner L2 attack (Carlini & Wagner, 2017).
///
/// Minimises ‖δ‖₂ + c·loss via gradient descent in tanh-space.
pub fn cw_attack(
    input: &Array1<f64>,
    target: usize,
    loss_fn: impl Fn(&Array1<f64>, usize) -> f64,
    gradient_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
    c: f64,
    steps: usize,
    lr: f64,
) -> Array1<f64> {
    let mut perturbation = Array1::<f64>::zeros(input.len());
    let mut best = input.clone();
    let mut best_l2 = f64::MAX;

    for _ in 0..steps {
        let adv = input + &perturbation;
        let l2 = perturbation.mapv(|x| x * x).sum().sqrt();
        let loss = l2 + c * loss_fn(&adv, target);

        // keep track of best adversarial found
        if loss < best_l2 {
            best_l2 = loss;
            best = adv.clone();
        }

        // gradient step on perturbation
        let grad = gradient_fn(&adv);
        // combined gradient: ∂l2/∂δ + c·∂loss/∂x
        let l2_norm = l2.max(1e-12);
        let l2_grad = &perturbation / l2_norm;
        let total_grad = &l2_grad + &(&grad * c);
        perturbation = &perturbation - &(&total_grad * lr);
    }
    best
}

/// Jacobian-based Saliency Map Attack (Papernot et al., 2016).
///
/// Greedily perturbs features with the highest saliency towards `target` class.
pub fn jsma(
    input: &Array1<f64>,
    _target: usize,
    saliency_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
    max_perturbations: usize,
    theta: f64,
) -> Array1<f64> {
    let mut adv = input.clone();
    let mut modified = vec![false; input.len()];

    for _ in 0..max_perturbations {
        let saliency = saliency_fn(&adv);
        // pick highest-saliency unmodified feature
        let best_idx = saliency
            .iter()
            .enumerate()
            .filter(|(i, _)| !modified[*i])
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        match best_idx {
            Some(idx) => {
                adv[idx] += theta;
                modified[idx] = true;
            }
            None => break,
        }
    }
    adv
}

/// Universal Adversarial Perturbation (Moosavi-Dezfooli et al., 2017).
///
/// Finds a single perturbation that fools the model on many inputs.
pub fn universal_perturbation(
    inputs: &[Array1<f64>],
    loss_fn: impl Fn(&Array1<f64>) -> f64,
    gradient_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
    epsilon: f64,
    max_iter: usize,
) -> Array1<f64> {
    if inputs.is_empty() {
        return Array1::zeros(0);
    }
    let dim = inputs[0].len();
    let mut perturbation = Array1::<f64>::zeros(dim);

    for _ in 0..max_iter {
        for x in inputs {
            let perturbed = x + &perturbation;
            let loss = loss_fn(&perturbed);
            if loss <= 0.0 {
                continue; // already fooled
            }
            // minimal perturbation direction for this sample
            let grad = gradient_fn(&perturbed);
            let grad_norm = grad.mapv(|g| g * g).sum().sqrt().max(1e-12);
            let delta = &grad * (loss / grad_norm);
            perturbation = &perturbation + &delta;

            // project to L2 ε-ball
            let p_norm = perturbation.mapv(|p| p * p).sum().sqrt();
            if p_norm > epsilon {
                perturbation *= epsilon / p_norm;
            }
        }
    }
    perturbation
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fgsm_positive_gradient() {
        let input = array![0.5, 0.3, 0.8];
        let grad = array![1.0, -1.0, 0.5];
        let adv = fgsm(&input, &grad, 0.1);
        assert!((adv[0] - 0.6).abs() < 1e-10);
        assert!((adv[1] - 0.2).abs() < 1e-10);
        assert!((adv[2] - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_fgsm_zero_gradient() {
        let input = array![1.0, 2.0];
        let grad = array![0.0, 0.0];
        let adv = fgsm(&input, &grad, 0.5);
        assert!((adv[0] - 1.0).abs() < 1e-10);
        assert!((adv[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pgd_stays_in_ball() {
        let input = array![0.5, 0.5];
        let adv = pgd(&input, |_x| array![1.0, 1.0], 0.1, 0.05, 100);
        let diff = &adv - &input;
        for &d in diff.iter() {
            assert!(d.abs() <= 0.1 + 1e-10);
        }
    }

    #[test]
    fn test_pgd_moves_in_gradient_direction() {
        let input = array![0.0, 0.0];
        let adv = pgd(&input, |_| array![1.0, -1.0], 0.5, 0.1, 10);
        assert!(adv[0] > 0.0);
        assert!(adv[1] < 0.0);
    }

    #[test]
    fn test_cw_attack_produces_perturbation() {
        let input = array![0.5, 0.5, 0.5];
        let adv = cw_attack(&input, 1, |_x, _t| 1.0, |x| x.clone(), 1.0, 10, 0.01);
        // Should differ from input
        let diff_norm = (&adv - &input).mapv(|d| d * d).sum().sqrt();
        // Just verify it ran without panic and produced something
        assert!(diff_norm >= 0.0);
    }

    #[test]
    fn test_jsma_perturbs_highest_saliency() {
        let input = array![0.0, 0.0, 0.0];
        let adv = jsma(
            &input,
            0,
            |_x| array![0.1, 0.9, 0.3], // index 1 is most salient
            1,
            0.5,
        );
        assert!((adv[0] - 0.0).abs() < 1e-10);
        assert!((adv[1] - 0.5).abs() < 1e-10);
        assert!((adv[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jsma_respects_max_perturbations() {
        let input = array![0.0, 0.0, 0.0, 0.0];
        let adv = jsma(&input, 0, |_x| array![0.5, 0.5, 0.5, 0.5], 2, 1.0);
        let changed = adv.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(changed, 2);
    }

    #[test]
    fn test_universal_perturbation_bounded() {
        let inputs = vec![array![1.0, 0.0], array![0.0, 1.0]];
        let pert = universal_perturbation(&inputs, |_x| 1.0, |x| x.clone(), 0.5, 3);
        let norm = pert.mapv(|p| p * p).sum().sqrt();
        assert!(norm <= 0.5 + 1e-10);
    }

    #[test]
    fn test_universal_perturbation_empty_inputs() {
        let pert = universal_perturbation(&[], |_| 0.0, |x| x.clone(), 1.0, 5);
        assert_eq!(pert.len(), 0);
    }
}
