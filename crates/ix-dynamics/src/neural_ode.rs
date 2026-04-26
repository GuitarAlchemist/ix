//! Neural ODE solver: continuous-depth models via ODE integration.
//!
//! Models the dynamics dy/dt = f(y, t) where f is a parameterized function
//! (e.g., a neural network). Provides RK4 fixed-step and Dormand-Prince
//! adaptive-step integrators.
//!
//! # Examples
//!
//! ```
//! use ix_dynamics::neural_ode::NeuralOde;
//! use ndarray::array;
//!
//! // Simple exponential decay: dy/dt = -y
//! let ode = NeuralOde::new(|y, _t| -y.to_owned());
//!
//! let y0 = array![1.0];
//! let trajectory = ode.solve(&y0, 0.0, 1.0, 0.01);
//!
//! // y(1) ≈ e⁻¹ ≈ 0.3679
//! let y_final = &trajectory.last().unwrap().1;
//! assert!((y_final[0] - (-1.0_f64).exp()).abs() < 1e-4);
//! ```

use ndarray::Array1;

/// A Neural ODE system: dy/dt = f(y, t).
///
/// The dynamics function `f` maps (state, time) → state derivative.
pub struct NeuralOde<F>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    /// The dynamics function.
    pub f: F,
}

impl<F> NeuralOde<F>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    /// Create a new Neural ODE with the given dynamics function.
    pub fn new(f: F) -> Self {
        Self { f }
    }

    /// Solve the ODE using fixed-step RK4.
    ///
    /// Returns a trajectory of (time, state) pairs from `t0` to `t1`.
    pub fn solve(&self, y0: &Array1<f64>, t0: f64, t1: f64, dt: f64) -> Vec<(f64, Array1<f64>)> {
        let mut trajectory = Vec::new();
        let mut t = t0;
        let mut y = y0.clone();

        trajectory.push((t, y.clone()));

        while t < t1 - dt * 0.5 {
            let step = dt.min(t1 - t);
            y = rk4_step(&self.f, &y, t, step);
            t += step;
            trajectory.push((t, y.clone()));
        }

        trajectory
    }

    /// Solve the ODE using adaptive Dormand-Prince (RK45).
    ///
    /// Automatically adjusts step size to maintain the given tolerance.
    /// Returns a trajectory of (time, state) pairs.
    pub fn solve_adaptive(
        &self,
        y0: &Array1<f64>,
        t0: f64,
        t1: f64,
        dt_init: f64,
        atol: f64,
        rtol: f64,
    ) -> Vec<(f64, Array1<f64>)> {
        let mut trajectory = Vec::new();
        let mut t = t0;
        let mut y = y0.clone();
        let mut dt = dt_init;

        trajectory.push((t, y.clone()));

        let dt_min = dt_init * 1e-6;
        let dt_max = dt_init * 10.0;
        let max_steps = 1_000_000;
        let mut steps = 0;

        while t < t1 - dt_min * 0.5 && steps < max_steps {
            let step = dt.min(t1 - t);
            let (y_new, err) = dopri5_step(&self.f, &y, t, step);

            // Error norm (mixed absolute/relative tolerance)
            let err_norm = error_norm(&err, &y, &y_new, atol, rtol);

            if err_norm <= 1.0 {
                // Accept step
                t += step;
                y = y_new;
                trajectory.push((t, y.clone()));

                // Grow step size
                let factor = 0.9 * (1.0 / err_norm).powf(0.2);
                dt = (step * factor.min(5.0)).min(dt_max);
            } else {
                // Reject step, shrink
                let factor = 0.9 * (1.0 / err_norm).powf(0.25);
                dt = (step * factor.max(0.1)).max(dt_min);
            }

            steps += 1;
        }

        trajectory
    }
}

/// Single RK4 step: y(t + dt) given y(t).
pub fn rk4_step<F>(f: &F, y: &Array1<f64>, t: f64, dt: f64) -> Array1<f64>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    let k1 = f(y, t);
    let k2 = f(&(y + &(&k1 * (dt / 2.0))), t + dt / 2.0);
    let k3 = f(&(y + &(&k2 * (dt / 2.0))), t + dt / 2.0);
    let k4 = f(&(y + &(&k3 * dt)), t + dt);

    y + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (dt / 6.0))
}

/// Single Dormand-Prince step with error estimate.
///
/// Returns (y_new, error_estimate).
pub fn dopri5_step<F>(f: &F, y: &Array1<f64>, t: f64, dt: f64) -> (Array1<f64>, Array1<f64>)
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    // Dormand-Prince coefficients (Butcher tableau)
    let k1 = f(y, t) * dt;
    let k2 = f(&(y + &(&k1 * (1.0 / 5.0))), t + dt / 5.0) * dt;
    let k3 = f(
        &(y + &(&k1 * (3.0 / 40.0)) + &(&k2 * (9.0 / 40.0))),
        t + dt * 3.0 / 10.0,
    ) * dt;
    let k4 = f(
        &(y + &(&k1 * (44.0 / 45.0)) + &(&k2 * (-56.0 / 15.0)) + &(&k3 * (32.0 / 9.0))),
        t + dt * 4.0 / 5.0,
    ) * dt;
    let k5 = f(
        &(y + &(&k1 * (19372.0 / 6561.0))
            + &(&k2 * (-25360.0 / 2187.0))
            + &(&k3 * (64448.0 / 6561.0))
            + &(&k4 * (-212.0 / 729.0))),
        t + dt * 8.0 / 9.0,
    ) * dt;
    let k6 = f(
        &(y + &(&k1 * (9017.0 / 3168.0))
            + &(&k2 * (-355.0 / 33.0))
            + &(&k3 * (46732.0 / 5247.0))
            + &(&k4 * (49.0 / 176.0))
            + &(&k5 * (-5103.0 / 18656.0))),
        t + dt,
    ) * dt;

    // 5th order solution
    let y_new = y
        + &(&k1 * (35.0 / 384.0))
        + &(&k3 * (500.0 / 1113.0))
        + &(&k4 * (125.0 / 192.0))
        + &(&k5 * (-2187.0 / 6784.0))
        + &(&k6 * (11.0 / 84.0));

    // 4th order solution (for error estimate)
    let k7 = f(&y_new, t + dt) * dt;
    let y_hat = y
        + &(&k1 * (5179.0 / 57600.0))
        + &(&k3 * (7571.0 / 16695.0))
        + &(&k4 * (393.0 / 640.0))
        + &(&k5 * (-92097.0 / 339200.0))
        + &(&k6 * (187.0 / 2100.0))
        + &(&k7 * (1.0 / 40.0));

    let error = &y_new - &y_hat;
    (y_new, error)
}

/// Compute mixed error norm for step size control.
fn error_norm(
    err: &Array1<f64>,
    y_old: &Array1<f64>,
    y_new: &Array1<f64>,
    atol: f64,
    rtol: f64,
) -> f64 {
    let n = err.len() as f64;
    let sum: f64 = err
        .iter()
        .zip(y_old.iter())
        .zip(y_new.iter())
        .map(|((e, yo), yn)| {
            let scale = atol + rtol * yo.abs().max(yn.abs());
            (e / scale) * (e / scale)
        })
        .sum();
    (sum / n).sqrt()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rk4_exponential_decay() {
        // dy/dt = -y, y(0) = 1 → y(t) = e^{-t}
        let f = |y: &Array1<f64>, _t: f64| -y.to_owned();
        let y0 = array![1.0];
        let y1 = rk4_step(&f, &y0, 0.0, 0.01);
        let exact = (-0.01_f64).exp();
        assert!((y1[0] - exact).abs() < 1e-10);
    }

    #[test]
    fn test_rk4_linear_ode() {
        // dy/dt = 1, y(0) = 0 → y(t) = t
        let f = |_y: &Array1<f64>, _t: f64| array![1.0];
        let y0 = array![0.0];
        let y1 = rk4_step(&f, &y0, 0.0, 1.0);
        assert!((y1[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_exponential() {
        let ode = NeuralOde::new(|y: &Array1<f64>, _t: f64| -y.to_owned());
        let y0 = array![1.0];
        let traj = ode.solve(&y0, 0.0, 1.0, 0.001);

        let y_final = &traj.last().unwrap().1;
        let exact = (-1.0_f64).exp();
        assert!(
            (y_final[0] - exact).abs() < 1e-6,
            "y(1) = {} (expected {})",
            y_final[0],
            exact
        );
    }

    #[test]
    fn test_solve_2d_rotation() {
        // dy/dt = [[0, -1], [1, 0]] * y → rotation
        let ode = NeuralOde::new(|y: &Array1<f64>, _t: f64| array![-y[1], y[0]]);
        let y0 = array![1.0, 0.0];
        let traj = ode.solve(&y0, 0.0, std::f64::consts::PI / 2.0, 0.001);

        let y_final = &traj.last().unwrap().1;
        // After π/2 rotation: [1,0] → [0,1]
        assert!(y_final[0].abs() < 1e-3, "x ≈ 0, got {}", y_final[0]);
        assert!((y_final[1] - 1.0).abs() < 1e-3, "y ≈ 1, got {}", y_final[1]);
    }

    #[test]
    fn test_dopri5_step() {
        let f = |y: &Array1<f64>, _t: f64| -y.to_owned();
        let y0 = array![1.0];
        let (y_new, err) = dopri5_step(&f, &y0, 0.0, 0.01);

        let exact = (-0.01_f64).exp();
        assert!((y_new[0] - exact).abs() < 1e-10);
        assert!(err[0].abs() < 1e-12); // Error should be very small
    }

    #[test]
    fn test_solve_adaptive() {
        let ode = NeuralOde::new(|y: &Array1<f64>, _t: f64| -y.to_owned());
        let y0 = array![1.0];
        let traj = ode.solve_adaptive(&y0, 0.0, 1.0, 0.1, 1e-8, 1e-8);

        let y_final = &traj.last().unwrap().1;
        let exact = (-1.0_f64).exp();
        assert!(
            (y_final[0] - exact).abs() < 1e-6,
            "adaptive: y(1) = {} (expected {})",
            y_final[0],
            exact
        );
    }

    #[test]
    fn test_solve_adaptive_uses_fewer_steps() {
        let ode = NeuralOde::new(|y: &Array1<f64>, _t: f64| -y.to_owned());
        let y0 = array![1.0];
        let traj_fixed = ode.solve(&y0, 0.0, 1.0, 0.001);
        let traj_adaptive = ode.solve_adaptive(&y0, 0.0, 1.0, 0.1, 1e-6, 1e-6);

        // Adaptive should use fewer steps than fixed with dt=0.001
        assert!(
            traj_adaptive.len() < traj_fixed.len(),
            "adaptive: {} steps, fixed: {} steps",
            traj_adaptive.len(),
            traj_fixed.len()
        );
    }

    #[test]
    fn test_solve_preserves_norm() {
        // Rotation preserves norm: |y(t)| = |y(0)| for all t
        let ode = NeuralOde::new(|y: &Array1<f64>, _t: f64| array![-y[1], y[0]]);
        let y0 = array![1.0, 0.0];
        let traj = ode.solve(&y0, 0.0, 2.0 * std::f64::consts::PI, 0.01);

        let initial_norm = y0.dot(&y0).sqrt();
        for (_, y) in &traj {
            let norm = y.dot(y).sqrt();
            assert!(
                (norm - initial_norm).abs() < 1e-4,
                "norm drift: {} vs {}",
                norm,
                initial_norm
            );
        }
    }

    #[test]
    fn test_trajectory_time_ordering() {
        let ode = NeuralOde::new(|y: &Array1<f64>, _t: f64| -y.to_owned());
        let y0 = array![1.0];
        let traj = ode.solve(&y0, 0.0, 1.0, 0.1);

        for i in 1..traj.len() {
            assert!(traj[i].0 > traj[i - 1].0);
        }
    }
}
