//! Lyapunov exponents — quantify sensitivity to initial conditions.
//!
//! Positive => chaos, zero => edge of chaos, negative => stable.

/// Maximal Lyapunov exponent (MLE) for a 1D map f with derivative df.
///
/// Uses the direct method: average log|df/dx| along an orbit.
pub fn mle_1d<F, DF>(f: F, df: DF, x0: f64, iterations: usize, transient: usize) -> f64
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut x = x0;

    // Discard transient
    for _ in 0..transient {
        x = f(x);
    }

    let mut lyap_sum = 0.0;
    for _ in 0..iterations {
        let deriv = df(x).abs();
        if deriv < 1e-15 {
            // Superstable orbit
            return f64::NEG_INFINITY;
        }
        lyap_sum += deriv.ln();
        x = f(x);
    }

    lyap_sum / iterations as f64
}

/// Lyapunov exponent spectrum for an n-dimensional continuous system.
///
/// Uses the QR decomposition method (Benettin et al.).
/// `dynamics`: dx/dt = f(x, t) returns the derivative vector.
/// `jacobian`: J(x, t) returns the n×n Jacobian matrix (flattened row-major).
/// `x0`: initial state (n-dimensional).
/// `dt`: integration time step.
/// `steps`: number of integration steps.
/// `transient`: steps to discard.
pub fn lyapunov_spectrum(
    dynamics: &dyn Fn(&[f64], f64) -> Vec<f64>,
    jacobian: &dyn Fn(&[f64], f64) -> Vec<f64>,
    x0: &[f64],
    dt: f64,
    steps: usize,
    transient: usize,
) -> Vec<f64> {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut t = 0.0;

    // Initialize perturbation vectors as identity matrix (column-major)
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }

    let mut lyap_sums = vec![0.0; n];
    let total = transient + steps;

    for step in 0..total {
        // RK4 integration of state
        let k1 = dynamics(&x, t);
        let x_mid1: Vec<f64> = x
            .iter()
            .zip(k1.iter())
            .map(|(xi, ki)| xi + 0.5 * dt * ki)
            .collect();
        let k2 = dynamics(&x_mid1, t + 0.5 * dt);
        let x_mid2: Vec<f64> = x
            .iter()
            .zip(k2.iter())
            .map(|(xi, ki)| xi + 0.5 * dt * ki)
            .collect();
        let k3 = dynamics(&x_mid2, t + 0.5 * dt);
        let x_end: Vec<f64> = x
            .iter()
            .zip(k3.iter())
            .map(|(xi, ki)| xi + dt * ki)
            .collect();
        let k4 = dynamics(&x_end, t + dt);

        for i in 0..n {
            x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += dt;

        // Evolve perturbation vectors using Jacobian
        let jac = jacobian(&x, t);
        let mut new_q = vec![0.0; n * n];
        for col in 0..n {
            for row in 0..n {
                let val = q[col * n + row];
                let mut dq = 0.0;
                for k in 0..n {
                    dq += jac[row * n + k] * q[col * n + k];
                }
                new_q[col * n + row] = val + dt * dq;
            }
        }
        q = new_q;

        // Modified Gram-Schmidt QR decomposition
        let mut r_diag = vec![0.0; n];
        for j in 0..n {
            // Subtract projections of previous vectors
            for i in 0..j {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += q[j * n + k] * q[i * n + k];
                }
                for k in 0..n {
                    q[j * n + k] -= dot * q[i * n + k];
                }
            }
            // Normalize
            let mut norm = 0.0;
            for k in 0..n {
                norm += q[j * n + k] * q[j * n + k];
            }
            norm = norm.sqrt();
            r_diag[j] = norm;
            if norm > 1e-15 {
                for k in 0..n {
                    q[j * n + k] /= norm;
                }
            }
        }

        // Accumulate after transient
        if step >= transient {
            for i in 0..n {
                if r_diag[i] > 1e-15 {
                    lyap_sums[i] += r_diag[i].ln();
                }
            }
        }
    }

    let count = steps as f64 * dt;
    lyap_sums.iter().map(|&s| s / count).collect()
}

/// Classify a system based on its maximal Lyapunov exponent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynamicsType {
    /// All exponents negative — converges to fixed point.
    FixedPoint,
    /// MLE ≈ 0 — quasiperiodic or limit cycle.
    Periodic,
    /// MLE > 0 — chaotic.
    Chaotic,
    /// MLE → ∞ — divergent / unstable.
    Divergent,
}

pub fn classify_dynamics(mle: f64, threshold: f64) -> DynamicsType {
    if mle > threshold {
        if mle > 10.0 {
            DynamicsType::Divergent
        } else {
            DynamicsType::Chaotic
        }
    } else if mle > -threshold {
        DynamicsType::Periodic
    } else {
        DynamicsType::FixedPoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_map_lyapunov() {
        // Logistic map x_{n+1} = r*x*(1-x)
        // At r=4 (fully chaotic), MLE = ln(2) ≈ 0.693
        let r = 4.0;
        let f = |x: f64| r * x * (1.0 - x);
        let df = |x: f64| r * (1.0 - 2.0 * x);

        let mle = mle_1d(f, df, 0.1, 10_000, 1000);
        assert!(
            (mle - 2.0_f64.ln()).abs() < 0.05,
            "MLE at r=4 should be ln(2), got {}",
            mle
        );
    }

    #[test]
    fn test_logistic_map_stable() {
        // At r=3.2 the logistic map has a stable 2-cycle, MLE < 0
        let r = 3.2;
        let f = |x: f64| r * x * (1.0 - x);
        let df = |x: f64| r * (1.0 - 2.0 * x);

        let mle = mle_1d(f, df, 0.1, 10_000, 1000);
        assert!(mle < 0.0, "MLE at r=3.2 should be negative, got {}", mle);
    }

    #[test]
    fn test_classify() {
        assert_eq!(classify_dynamics(0.5, 0.01), DynamicsType::Chaotic);
        assert_eq!(classify_dynamics(-0.5, 0.01), DynamicsType::FixedPoint);
        assert_eq!(classify_dynamics(0.001, 0.01), DynamicsType::Periodic);
    }
}
