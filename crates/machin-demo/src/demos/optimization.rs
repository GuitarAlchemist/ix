use eframe::egui;
use egui_plot::{Plot, Line, Points, PlotPoints};

#[derive(PartialEq, Clone, Copy)]
enum OptMethod {
    Sgd,
    Adam,
    SimulatedAnnealing,
}

pub struct OptimizationDemo {
    method: OptMethod,
    learning_rate: f64,
    iterations: usize,
    trajectory: Vec<[f64; 2]>,
    loss_history: Vec<f64>,
    status: String,
}

impl Default for OptimizationDemo {
    fn default() -> Self {
        Self {
            method: OptMethod::Adam,
            learning_rate: 0.05,
            iterations: 100,
            trajectory: Vec::new(),
            loss_history: Vec::new(),
            status: "Ready — minimizing Rosenbrock function".into(),
        }
    }
}

impl OptimizationDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Optimization (machin-optimize)");
        ui.label("Minimizing Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.method, OptMethod::Sgd, "SGD");
            ui.radio_value(&mut self.method, OptMethod::Adam, "Adam");
            ui.radio_value(&mut self.method, OptMethod::SimulatedAnnealing, "Sim. Annealing");
        });
        ui.horizontal(|ui| {
            ui.label("LR:");
            ui.add(egui::Slider::new(&mut self.learning_rate, 0.001..=1.0).logarithmic(true));
            ui.label("Iterations:");
            ui.add(egui::Slider::new(&mut self.iterations, 10..=1000));
        });

        if ui.button("Optimize").clicked() {
            self.run();
        }

        ui.label(&self.status);

        if !self.trajectory.is_empty() {
            ui.columns(2, |cols| {
                // Trajectory plot
                Plot::new("opt_trajectory").height(300.0).show(&mut cols[0], |plot_ui| {
                    let data: Vec<[f64; 2]> = self.trajectory.clone();
                    let pts1: PlotPoints = data.iter().copied().collect();
                    let pts2: PlotPoints = data.iter().copied().collect();
                    plot_ui.line(Line::new(pts1).name("Path").width(1.5));
                    plot_ui.points(Points::new(pts2).radius(2.0).name("Steps"));

                    // Mark optimum
                    plot_ui.points(Points::new(PlotPoints::new(vec![[1.0, 1.0]]))
                        .radius(8.0).color(egui::Color32::GREEN).name("Optimum"));
                });

                // Loss curve
                Plot::new("opt_loss").height(300.0).show(&mut cols[1], |plot_ui| {
                    let pts: PlotPoints = self.loss_history.iter().enumerate()
                        .map(|(i, &l)| [i as f64, l.ln().max(-10.0)])
                        .collect();
                    plot_ui.line(Line::new(pts).name("log(loss)").width(2.0));
                });
            });
        }
    }

    fn run(&mut self) {
        // Rosenbrock gradient
        let rosenbrock = |x: f64, y: f64| -> f64 {
            (1.0 - x).powi(2) + 100.0 * (y - x*x).powi(2)
        };
        let grad = |x: f64, y: f64| -> [f64; 2] {
            [
                -2.0 * (1.0 - x) - 400.0 * x * (y - x*x),
                200.0 * (y - x*x),
            ]
        };

        let mut x = -1.0;
        let mut y = -1.0;
        self.trajectory.clear();
        self.loss_history.clear();
        self.trajectory.push([x, y]);
        self.loss_history.push(rosenbrock(x, y));

        match self.method {
            OptMethod::Sgd => {
                for _ in 0..self.iterations {
                    let g = grad(x, y);
                    x -= self.learning_rate * g[0];
                    y -= self.learning_rate * g[1];
                    self.trajectory.push([x, y]);
                    self.loss_history.push(rosenbrock(x, y));
                }
            }
            OptMethod::Adam => {
                let (beta1, beta2, eps) = (0.9, 0.999, 1e-8);
                let (mut m0, mut m1) = (0.0, 0.0);
                let (mut v0, mut v1) = (0.0, 0.0);
                for t in 1..=self.iterations {
                    let g = grad(x, y);
                    m0 = beta1 * m0 + (1.0 - beta1) * g[0];
                    m1 = beta1 * m1 + (1.0 - beta1) * g[1];
                    v0 = beta2 * v0 + (1.0 - beta2) * g[0] * g[0];
                    v1 = beta2 * v1 + (1.0 - beta2) * g[1] * g[1];
                    let mh0 = m0 / (1.0 - beta1.powi(t as i32));
                    let mh1 = m1 / (1.0 - beta1.powi(t as i32));
                    let vh0 = v0 / (1.0 - beta2.powi(t as i32));
                    let vh1 = v1 / (1.0 - beta2.powi(t as i32));
                    x -= self.learning_rate * mh0 / (vh0.sqrt() + eps);
                    y -= self.learning_rate * mh1 / (vh1.sqrt() + eps);
                    self.trajectory.push([x, y]);
                    self.loss_history.push(rosenbrock(x, y));
                }
            }
            OptMethod::SimulatedAnnealing => {
                use rand::Rng;
                let mut rng = rand::rng();
                let mut best_x = x;
                let mut best_y = y;
                let mut best_f = rosenbrock(x, y);
                let mut cur_f = best_f;

                for i in 0..self.iterations {
                    let temp = 10.0 * (1.0 - i as f64 / self.iterations as f64);
                    let nx = x + rng.random_range(-0.5..0.5);
                    let ny = y + rng.random_range(-0.5..0.5);
                    let nf = rosenbrock(nx, ny);
                    let df = nf - cur_f;
                    if df < 0.0 || rng.random_range(0.0..1.0) < (-df / temp.max(0.01)).exp() {
                        x = nx;
                        y = ny;
                        cur_f = nf;
                    }
                    if cur_f < best_f {
                        best_x = x;
                        best_y = y;
                        best_f = cur_f;
                    }
                    self.trajectory.push([x, y]);
                    self.loss_history.push(cur_f);
                }
                x = best_x;
                y = best_y;
            }
        }

        let final_f = rosenbrock(x, y);
        self.status = format!("Final: ({:.4}, {:.4}) = {:.6}", x, y, final_f);
    }
}
