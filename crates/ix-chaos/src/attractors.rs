//! Strange attractors — Lorenz, Rössler, Hénon, Duffing, and more.

/// 3D state for continuous-time attractors.
#[derive(Debug, Clone, Copy)]
pub struct State3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl State3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn as_slice(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

/// Integrate a 3D ODE system using RK4.
pub fn rk4_step(state: State3D, dt: f64, derivatives: &dyn Fn(State3D) -> State3D) -> State3D {
    let k1 = derivatives(state);
    let k2 = derivatives(State3D::new(
        state.x + 0.5 * dt * k1.x,
        state.y + 0.5 * dt * k1.y,
        state.z + 0.5 * dt * k1.z,
    ));
    let k3 = derivatives(State3D::new(
        state.x + 0.5 * dt * k2.x,
        state.y + 0.5 * dt * k2.y,
        state.z + 0.5 * dt * k2.z,
    ));
    let k4 = derivatives(State3D::new(
        state.x + dt * k3.x,
        state.y + dt * k3.y,
        state.z + dt * k3.z,
    ));

    State3D::new(
        state.x + dt / 6.0 * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
        state.y + dt / 6.0 * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y),
        state.z + dt / 6.0 * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z),
    )
}

/// Generate a trajectory by integrating an ODE system.
pub fn integrate(
    initial: State3D,
    dt: f64,
    steps: usize,
    derivatives: &dyn Fn(State3D) -> State3D,
) -> Vec<State3D> {
    let mut trajectory = Vec::with_capacity(steps + 1);
    trajectory.push(initial);

    let mut state = initial;
    for _ in 0..steps {
        state = rk4_step(state, dt, derivatives);
        trajectory.push(state);
    }
    trajectory
}

// ── Lorenz attractor ──────────────────────────────────────────

/// Lorenz system parameters.
#[derive(Debug, Clone, Copy)]
pub struct LorenzParams {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
}

impl Default for LorenzParams {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        }
    }
}

/// Lorenz system: dx/dt = sigma*(y-x), dy/dt = x*(rho-z)-y, dz/dt = x*y - beta*z
pub fn lorenz_derivatives(state: State3D, params: &LorenzParams) -> State3D {
    State3D::new(
        params.sigma * (state.y - state.x),
        state.x * (params.rho - state.z) - state.y,
        state.x * state.y - params.beta * state.z,
    )
}

/// Generate Lorenz attractor trajectory.
pub fn lorenz(initial: State3D, params: &LorenzParams, dt: f64, steps: usize) -> Vec<State3D> {
    let p = *params;
    integrate(initial, dt, steps, &move |s| lorenz_derivatives(s, &p))
}

// ── Rössler attractor ─────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct RosslerParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Default for RosslerParams {
    fn default() -> Self {
        Self {
            a: 0.2,
            b: 0.2,
            c: 5.7,
        }
    }
}

pub fn rossler_derivatives(state: State3D, params: &RosslerParams) -> State3D {
    State3D::new(
        -state.y - state.z,
        state.x + params.a * state.y,
        params.b + state.z * (state.x - params.c),
    )
}

pub fn rossler(initial: State3D, params: &RosslerParams, dt: f64, steps: usize) -> Vec<State3D> {
    let p = *params;
    integrate(initial, dt, steps, &move |s| rossler_derivatives(s, &p))
}

// ── Chen attractor ────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct ChenParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Default for ChenParams {
    fn default() -> Self {
        Self {
            a: 35.0,
            b: 3.0,
            c: 28.0,
        }
    }
}

pub fn chen_derivatives(state: State3D, params: &ChenParams) -> State3D {
    State3D::new(
        params.a * (state.y - state.x),
        (params.c - params.a) * state.x - state.x * state.z + params.c * state.y,
        state.x * state.y - params.b * state.z,
    )
}

pub fn chen(initial: State3D, params: &ChenParams, dt: f64, steps: usize) -> Vec<State3D> {
    let p = *params;
    integrate(initial, dt, steps, &move |s| chen_derivatives(s, &p))
}

// ── Hénon map (discrete 2D) ──────────────────────────────────

/// Hénon map: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n
#[derive(Debug, Clone, Copy)]
pub struct HenonParams {
    pub a: f64,
    pub b: f64,
}

impl Default for HenonParams {
    fn default() -> Self {
        Self { a: 1.4, b: 0.3 }
    }
}

pub fn henon_iterate(x: f64, y: f64, params: &HenonParams) -> (f64, f64) {
    (1.0 - params.a * x * x + y, params.b * x)
}

/// Generate Hénon map trajectory.
pub fn henon(x0: f64, y0: f64, params: &HenonParams, steps: usize) -> Vec<(f64, f64)> {
    let mut traj = Vec::with_capacity(steps + 1);
    let mut x = x0;
    let mut y = y0;
    traj.push((x, y));

    for _ in 0..steps {
        let (nx, ny) = henon_iterate(x, y, params);
        x = nx;
        y = ny;
        traj.push((x, y));
    }
    traj
}

// ── Logistic map ──────────────────────────────────────────────

/// Iterate the logistic map x_{n+1} = r*x*(1-x).
pub fn logistic_map(x0: f64, r: f64, steps: usize) -> Vec<f64> {
    let mut traj = Vec::with_capacity(steps + 1);
    let mut x = x0;
    traj.push(x);

    for _ in 0..steps {
        x = r * x * (1.0 - x);
        traj.push(x);
    }
    traj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lorenz_bounded() {
        let params = LorenzParams::default();
        let traj = lorenz(State3D::new(1.0, 1.0, 1.0), &params, 0.01, 10_000);
        // Lorenz attractor should stay bounded
        for s in &traj {
            assert!(
                s.x.abs() < 100.0 && s.y.abs() < 100.0 && s.z.abs() < 100.0,
                "Lorenz trajectory diverged"
            );
        }
    }

    #[test]
    fn test_rossler_bounded() {
        let params = RosslerParams::default();
        let traj = rossler(State3D::new(1.0, 1.0, 1.0), &params, 0.01, 10_000);
        for s in &traj {
            assert!(s.x.abs() < 50.0 && s.y.abs() < 50.0 && s.z.abs() < 50.0);
        }
    }

    #[test]
    fn test_henon_attractor() {
        let params = HenonParams::default();
        let traj = henon(0.1, 0.1, &params, 1000);
        // Hénon attractor values are bounded roughly in [-2, 2]
        for &(x, y) in &traj[100..] {
            assert!(x.abs() < 3.0 && y.abs() < 3.0);
        }
    }

    #[test]
    fn test_logistic_map_fixed_point() {
        // r=2.5: fixed point at x* = 1 - 1/r = 0.6
        let traj = logistic_map(0.5, 2.5, 1000);
        let last = *traj.last().unwrap();
        assert!(
            (last - 0.6).abs() < 1e-6,
            "Should converge to 0.6, got {}",
            last
        );
    }
}
