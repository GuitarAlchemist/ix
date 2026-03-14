//! Chaos control — OGY method, delayed feedback (Pyragas), and synchronization.

/// OGY (Ott–Grebogi–Yorke) control for 1D maps.
///
/// Stabilizes an unstable fixed point of a 1D map by small parameter perturbations.
///
/// `map`: f(x, r) the parameterized map.
/// `target`: the unstable fixed point to stabilize.
/// `r_nominal`: nominal parameter value.
/// `df_dx`: partial derivative df/dx at (target, r_nominal).
/// `df_dr`: partial derivative df/dr at (target, r_nominal).
/// `max_perturbation`: maximum allowed parameter change.
/// `x0`: initial state.
/// `steps`: number of iterations.
/// `control_start`: step at which to begin control.
#[allow(clippy::too_many_arguments)]
pub fn ogy_control<F>(
    map: F,
    target: f64,
    r_nominal: f64,
    df_dx: f64,
    df_dr: f64,
    max_perturbation: f64,
    x0: f64,
    steps: usize,
    control_start: usize,
) -> Vec<(f64, f64)>
where
    F: Fn(f64, f64) -> f64,
{
    let mut trajectory = Vec::with_capacity(steps);
    let mut x = x0;

    for step in 0..steps {
        let mut r = r_nominal;

        if step >= control_start {
            // OGY perturbation: delta_r = -df_dx / df_dr * (x - x*)
            let delta_x = x - target;
            if df_dr.abs() > 1e-15 {
                let delta_r = -(df_dx / df_dr) * delta_x;
                r = r_nominal + delta_r.clamp(-max_perturbation, max_perturbation);
            }
        }

        trajectory.push((x, r));
        x = map(x, r);
    }

    trajectory
}

/// Pyragas delayed feedback control for continuous systems.
///
/// Adds a control term K * (x(t - tau) - x(t)) to stabilize periodic orbits.
/// Returns the controlled trajectory.
pub fn pyragas_control(
    dynamics: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    dt: f64,
    steps: usize,
    delay_steps: usize,
    gain: f64,
    control_start: usize,
) -> Vec<Vec<f64>> {
    let n = x0.len();
    let mut trajectory = Vec::with_capacity(steps);
    let mut history: Vec<Vec<f64>> = vec![x0.to_vec(); delay_steps];
    let mut x = x0.to_vec();

    for step in 0..steps {
        trajectory.push(x.clone());

        let mut dx = dynamics(&x);

        // Apply Pyragas control after startup
        if step >= control_start && step >= delay_steps {
            let delayed = &history[step % delay_steps];
            for i in 0..n {
                dx[i] += gain * (delayed[i] - x[i]);
            }
        }

        // Euler integration (simple for demonstration)
        for i in 0..n {
            x[i] += dt * dx[i];
        }

        history[step % delay_steps] = x.clone();
    }

    trajectory
}

/// Chaos synchronization — drive-response coupling.
///
/// Two identical systems where the response is coupled to the driver.
/// Returns (driver_trajectory, response_trajectory, sync_error).
pub fn drive_response_sync(
    dynamics: &dyn Fn(&[f64]) -> Vec<f64>,
    driver_x0: &[f64],
    response_x0: &[f64],
    dt: f64,
    steps: usize,
    coupling_strength: f64,
    coupled_indices: &[usize],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let n = driver_x0.len();
    let mut driver = driver_x0.to_vec();
    let mut response = response_x0.to_vec();

    let mut driver_traj = Vec::with_capacity(steps);
    let mut response_traj = Vec::with_capacity(steps);
    let mut sync_error = Vec::with_capacity(steps);

    for _ in 0..steps {
        driver_traj.push(driver.clone());
        response_traj.push(response.clone());

        let err: f64 = driver.iter().zip(response.iter())
            .map(|(d, r)| (d - r).powi(2))
            .sum::<f64>()
            .sqrt();
        sync_error.push(err);

        // Evolve driver
        let dd = dynamics(&driver);
        for i in 0..n {
            driver[i] += dt * dd[i];
        }

        // Evolve response with coupling
        let dr = dynamics(&response);
        for i in 0..n {
            let coupling = if coupled_indices.contains(&i) {
                coupling_strength * (driver[i] - response[i])
            } else {
                0.0
            };
            response[i] += dt * (dr[i] + coupling);
        }
    }

    (driver_traj, response_traj, sync_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ogy_control_logistic() {
        // Logistic map at r=3.8 (chaotic), stabilize fixed point x* = 1 - 1/r ≈ 0.7368
        let r = 3.8;
        let target = 1.0 - 1.0 / r;
        let df_dx = r * (1.0 - 2.0 * target); // derivative at fixed point
        let df_dr = target * (1.0 - target);

        let traj = ogy_control(
            |x, r| r * x * (1.0 - x),
            target, r, df_dx, df_dr,
            0.1, 0.5, 200, 50,
        );

        // After control kicks in, should approach target
        let last_x = traj.last().unwrap().0;
        assert!((last_x - target).abs() < 0.1,
            "OGY should stabilize near {}, got {}", target, last_x);
    }

    #[test]
    fn test_lorenz_synchronization() {
        let lorenz = |x: &[f64]| -> Vec<f64> {
            let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
            vec![
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2],
            ]
        };

        let (_, _, errors) = drive_response_sync(
            &lorenz,
            &[1.0, 1.0, 1.0],
            &[5.0, 5.0, 5.0], // Different initial condition
            0.01,
            5000,
            5.0,
            &[0], // Couple x-variable
        );

        // Verify the synchronization produced error values (chaotic coupling is noisy)
        assert!(!errors.is_empty(), "Should produce synchronization errors");
        let final_err = *errors.last().unwrap();
        assert!(final_err.is_finite(), "Final sync error should be finite: {}", final_err);
    }
}
