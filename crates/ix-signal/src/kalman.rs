//! Kalman Filter for linear state estimation.
//!
//! Estimates hidden state from noisy observations.
//! Used in: tracking, navigation, sensor fusion, financial time series.

use ndarray::{Array1, Array2};

/// Linear Kalman Filter.
///
/// State model:   x_{k} = F * x_{k-1} + B * u_{k} + w_{k}
/// Observation:   z_{k} = H * x_{k} + v_{k}
///
/// where w ~ N(0, Q) and v ~ N(0, R)
pub struct KalmanFilter {
    /// State transition matrix (F).
    pub transition: Array2<f64>,
    /// Observation matrix (H).
    pub observation: Array2<f64>,
    /// Process noise covariance (Q).
    pub process_noise: Array2<f64>,
    /// Measurement noise covariance (R).
    pub measurement_noise: Array2<f64>,
    /// Control input matrix (B). Optional — set to zeros if unused.
    pub control: Array2<f64>,

    /// Current state estimate.
    pub state: Array1<f64>,
    /// Current error covariance (P).
    pub covariance: Array2<f64>,
}

impl KalmanFilter {
    /// Create a new Kalman filter.
    /// `state_dim`: dimension of the state vector.
    /// `obs_dim`: dimension of the observation vector.
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        Self {
            transition: Array2::eye(state_dim),
            observation: Array2::zeros((obs_dim, state_dim)),
            process_noise: Array2::eye(state_dim) * 0.01,
            measurement_noise: Array2::eye(obs_dim) * 1.0,
            control: Array2::zeros((state_dim, 1)),
            state: Array1::zeros(state_dim),
            covariance: Array2::eye(state_dim),
        }
    }

    /// Predict step: propagate state forward.
    pub fn predict(&mut self, control_input: Option<&Array1<f64>>) {
        // x = F * x + B * u
        self.state = self.transition.dot(&self.state);
        if let Some(u) = control_input {
            self.state = &self.state + &self.control.dot(u);
        }

        // P = F * P * F^T + Q
        self.covariance = self
            .transition
            .dot(&self.covariance)
            .dot(&self.transition.t())
            + &self.process_noise;
    }

    /// Update step: incorporate a measurement.
    pub fn update(&mut self, measurement: &Array1<f64>) {
        let h = &self.observation;
        let r = &self.measurement_noise;

        // Innovation: y = z - H * x
        let innovation = measurement - &h.dot(&self.state);

        // Innovation covariance: S = H * P * H^T + R
        let s = h.dot(&self.covariance).dot(&h.t()) + r;

        // Kalman gain: K = P * H^T * S^{-1}
        let s_inv = ix_math::linalg::inverse(&s).expect("Innovation covariance singular");
        let k = self.covariance.dot(&h.t()).dot(&s_inv);

        // State update: x = x + K * y
        self.state = &self.state + &k.dot(&innovation);

        // Covariance update: P = (I - K * H) * P
        let i = Array2::eye(self.state.len());
        self.covariance = (&i - &k.dot(h)).dot(&self.covariance);
    }

    /// Run predict + update in one step.
    pub fn step(
        &mut self,
        measurement: &Array1<f64>,
        control_input: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        self.predict(control_input);
        self.update(measurement);
        self.state.clone()
    }

    /// Process a sequence of measurements. Returns filtered states.
    pub fn filter(&mut self, measurements: &[Array1<f64>]) -> Vec<Array1<f64>> {
        measurements.iter().map(|z| self.step(z, None)).collect()
    }
}

/// Convenience: create a 1D constant-velocity Kalman filter.
/// State = [position, velocity], observation = [position].
pub fn constant_velocity_1d(process_noise: f64, measurement_noise: f64, dt: f64) -> KalmanFilter {
    let mut kf = KalmanFilter::new(2, 1);

    // State transition: [x, v] -> [x + v*dt, v]
    kf.transition = ndarray::array![[1.0, dt], [0.0, 1.0]];

    // Observation: we only measure position
    kf.observation = ndarray::array![[1.0, 0.0]];

    // Process noise
    let q = process_noise;
    kf.process_noise = ndarray::array![
        [q * dt.powi(3) / 3.0, q * dt.powi(2) / 2.0],
        [q * dt.powi(2) / 2.0, q * dt]
    ];

    // Measurement noise
    kf.measurement_noise = ndarray::array![[measurement_noise]];

    kf
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kalman_constant_position() {
        // Object at position 5.0, noisy measurements
        let mut kf = KalmanFilter::new(1, 1);
        kf.transition = array![[1.0]];
        kf.observation = array![[1.0]];
        kf.process_noise = array![[0.01]];
        kf.measurement_noise = array![[1.0]];
        kf.state = array![0.0]; // Start at 0

        let measurements: Vec<Array1<f64>> = vec![
            array![5.2],
            array![4.8],
            array![5.1],
            array![4.9],
            array![5.0],
            array![5.3],
            array![4.7],
            array![5.1],
            array![4.9],
            array![5.0],
        ];

        let states = kf.filter(&measurements);

        // After 10 measurements, should converge near 5.0
        let last = states.last().unwrap()[0];
        assert!(
            (last - 5.0).abs() < 0.5,
            "Should converge to ~5.0, got {}",
            last
        );
    }

    #[test]
    fn test_constant_velocity_tracking() {
        // Object moving at constant velocity: pos = 10 + 2*t
        let mut kf = constant_velocity_1d(0.1, 1.0, 1.0);

        let measurements: Vec<Array1<f64>> = (0..20)
            .map(|t| {
                let true_pos = 10.0 + 2.0 * t as f64;
                array![true_pos + 0.5 * (t as f64 % 3.0 - 1.0)] // Noisy
            })
            .collect();

        let states = kf.filter(&measurements);

        // Check velocity estimate converges near 2.0
        let last_vel = states.last().unwrap()[1];
        assert!(
            (last_vel - 2.0).abs() < 1.0,
            "Velocity should be ~2.0, got {}",
            last_vel
        );
    }
}
