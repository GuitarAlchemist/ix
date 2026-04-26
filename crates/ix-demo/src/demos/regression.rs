use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use ndarray::{Array1, Array2};

pub struct RegressionDemo {
    n_points: usize,
    noise: f64,
    data: Option<RegressionData>,
}

struct RegressionData {
    x: Vec<f64>,
    y: Vec<f64>,
    pred_x: Vec<f64>,
    pred_y: Vec<f64>,
    r2: f64,
    mse: f64,
    coeffs: Vec<f64>,
}

impl Default for RegressionDemo {
    fn default() -> Self {
        Self {
            n_points: 50,
            noise: 0.3,
            data: None,
        }
    }
}

impl RegressionDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Linear Regression (ix-supervised)");

        ui.horizontal(|ui| {
            ui.label("Points:");
            ui.add(egui::Slider::new(&mut self.n_points, 10..=200));
            ui.label("Noise:");
            ui.add(egui::Slider::new(&mut self.noise, 0.0..=2.0));
        });

        if ui.button("Generate & Fit").clicked() {
            self.run();
        }

        if let Some(d) = &self.data {
            ui.separator();
            ui.label(format!(
                "R² = {:.4}  |  MSE = {:.4}  |  y = {:.3}x + {:.3}",
                d.r2, d.mse, d.coeffs[0], d.coeffs[1]
            ));

            Plot::new("regression_plot")
                .height(400.0)
                .show(ui, |plot_ui| {
                    let pts: PlotPoints =
                        d.x.iter().zip(d.y.iter()).map(|(&x, &y)| [x, y]).collect();
                    plot_ui.points(Points::new(pts).radius(3.0).name("Data"));

                    let line: PlotPoints = d
                        .pred_x
                        .iter()
                        .zip(d.pred_y.iter())
                        .map(|(&x, &y)| [x, y])
                        .collect();
                    plot_ui.line(Line::new(line).name("Fit").width(2.0));
                });
        }
    }

    fn run(&mut self) {
        use rand::Rng;
        let mut rng = rand::rng();

        let x: Vec<f64> = (0..self.n_points)
            .map(|i| i as f64 / self.n_points as f64 * 10.0)
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.5 * xi + 1.0 + rng.random_range(-self.noise..self.noise) * 5.0)
            .collect();

        // Build X matrix [x, 1]
        let x_mat =
            Array2::from_shape_fn((self.n_points, 2), |(i, j)| if j == 0 { x[i] } else { 1.0 });
        let y_arr = Array1::from_vec(y.clone());

        // Least squares: (X^T X)^{-1} X^T y
        let xtx = x_mat.t().dot(&x_mat);
        let xty = x_mat.t().dot(&y_arr);

        // 2x2 inverse
        let a = xtx[[0, 0]];
        let b = xtx[[0, 1]];
        let c = xtx[[1, 0]];
        let d = xtx[[1, 1]];
        let det = a * d - b * c;
        if det.abs() < 1e-12 {
            return;
        }

        let coeffs = vec![
            (d * xty[0] - b * xty[1]) / det,
            (-c * xty[0] + a * xty[1]) / det,
        ];

        let pred_x: Vec<f64> = vec![0.0, 10.0];
        let pred_y: Vec<f64> = pred_x
            .iter()
            .map(|&xi| coeffs[0] * xi + coeffs[1])
            .collect();

        let y_pred: Vec<f64> = x.iter().map(|&xi| coeffs[0] * xi + coeffs[1]).collect();
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let ss_res: f64 = y
            .iter()
            .zip(y_pred.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let r2 = 1.0 - ss_res / ss_tot;
        let mse = ss_res / y.len() as f64;

        self.data = Some(RegressionData {
            x,
            y,
            pred_x,
            pred_y,
            r2,
            mse,
            coeffs,
        });
    }
}
