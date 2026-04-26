use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};

#[derive(PartialEq, Clone, Copy)]
enum ChaosMode {
    Logistic,
    Lorenz,
    Mandelbrot,
}

pub struct ChaosDemo {
    mode: ChaosMode,
    // Logistic
    r_min: f64,
    r_max: f64,
    // Lorenz
    sigma: f64,
    rho: f64,
    beta: f64,
    lorenz_steps: usize,
    // Mandelbrot
    mandel_iters: usize,
    mandel_res: usize,
    plot_data: Vec<[f64; 2]>,
}

impl Default for ChaosDemo {
    fn default() -> Self {
        Self {
            mode: ChaosMode::Logistic,
            r_min: 2.5,
            r_max: 4.0,
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            lorenz_steps: 5000,
            mandel_iters: 50,
            mandel_res: 200,
            plot_data: Vec::new(),
        }
    }
}

impl ChaosDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Chaos Theory (ix-chaos)");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, ChaosMode::Logistic, "Bifurcation");
            ui.radio_value(&mut self.mode, ChaosMode::Lorenz, "Lorenz Attractor");
            ui.radio_value(&mut self.mode, ChaosMode::Mandelbrot, "Mandelbrot Set");
        });

        match self.mode {
            ChaosMode::Logistic => self.logistic_ui(ui),
            ChaosMode::Lorenz => self.lorenz_ui(ui),
            ChaosMode::Mandelbrot => self.mandelbrot_ui(ui),
        }
    }

    fn logistic_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("r range:");
            ui.add(egui::Slider::new(&mut self.r_min, 0.0..=3.5));
            ui.add(egui::Slider::new(&mut self.r_max, 3.0..=4.0));
        });

        if ui.button("Generate Bifurcation").clicked() {
            self.gen_bifurcation();
        }

        if !self.plot_data.is_empty() {
            Plot::new("bifurcation").height(500.0).show(ui, |plot_ui| {
                let pts: PlotPoints = self.plot_data.iter().copied().collect();
                plot_ui.points(
                    Points::new(pts)
                        .radius(0.5)
                        .color(egui::Color32::LIGHT_BLUE),
                );
            });
        }
    }

    fn lorenz_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("σ:");
            ui.add(egui::Slider::new(&mut self.sigma, 1.0..=20.0));
            ui.label("ρ:");
            ui.add(egui::Slider::new(&mut self.rho, 1.0..=50.0));
            ui.label("β:");
            ui.add(egui::Slider::new(&mut self.beta, 0.1..=10.0));
        });
        ui.add(egui::Slider::new(&mut self.lorenz_steps, 1000..=20000).text("Steps"));

        if ui.button("Simulate Lorenz").clicked() {
            self.gen_lorenz();
        }

        if !self.plot_data.is_empty() {
            Plot::new("lorenz")
                .height(500.0)
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    let pts: PlotPoints = self.plot_data.iter().copied().collect();
                    plot_ui.line(Line::new(pts).name("XZ projection").width(0.5));
                });
        }
    }

    fn mandelbrot_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Max iterations:");
            ui.add(egui::Slider::new(&mut self.mandel_iters, 10..=200));
            ui.label("Resolution:");
            ui.add(egui::Slider::new(&mut self.mandel_res, 50..=500));
        });

        if ui.button("Render Mandelbrot").clicked() {
            self.gen_mandelbrot();
        }

        if !self.plot_data.is_empty() {
            Plot::new("mandelbrot")
                .height(500.0)
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    let pts: PlotPoints = self.plot_data.iter().copied().collect();
                    plot_ui.points(Points::new(pts).radius(1.0).color(egui::Color32::WHITE));
                });
        }
    }

    fn gen_bifurcation(&mut self) {
        self.plot_data.clear();
        let n_r = 800;
        let warmup = 200;
        let show = 100;

        for i in 0..n_r {
            let r = self.r_min + (self.r_max - self.r_min) * i as f64 / n_r as f64;
            let mut x = 0.5;
            for _ in 0..warmup {
                x = r * x * (1.0 - x);
            }
            for _ in 0..show {
                x = r * x * (1.0 - x);
                self.plot_data.push([r, x]);
            }
        }
    }

    fn gen_lorenz(&mut self) {
        self.plot_data.clear();
        let dt = 0.005;
        let (mut x, mut y, mut z) = (1.0, 1.0, 1.0);

        for _ in 0..self.lorenz_steps {
            let dx = self.sigma * (y - x);
            let dy = x * (self.rho - z) - y;
            let dz = x * y - self.beta * z;
            x += dx * dt;
            y += dy * dt;
            z += dz * dt;
            self.plot_data.push([x, z]); // XZ projection
        }
    }

    fn gen_mandelbrot(&mut self) {
        self.plot_data.clear();
        let res = self.mandel_res;
        let max_iter = self.mandel_iters;

        for xi in 0..res {
            for yi in 0..res {
                let cr = -2.5 + 3.5 * xi as f64 / res as f64;
                let ci = -1.25 + 2.5 * yi as f64 / res as f64;
                let mut zr = 0.0;
                let mut zi = 0.0;
                let mut in_set = true;
                for _ in 0..max_iter {
                    let zr2 = zr * zr - zi * zi + cr;
                    let zi2 = 2.0 * zr * zi + ci;
                    zr = zr2;
                    zi = zi2;
                    if zr * zr + zi * zi > 4.0 {
                        in_set = false;
                        break;
                    }
                }
                if in_set {
                    self.plot_data.push([cr, ci]);
                }
            }
        }
    }
}
