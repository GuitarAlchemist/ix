use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};

#[derive(PartialEq, Clone, Copy)]
enum TopoMode {
    Persistence,
    BettiCurve,
    BettiAtRadius,
}

#[derive(PartialEq, Clone, Copy)]
enum PointCloudPreset {
    Triangle,
    Circle,
    TwoClusters,
}

pub struct TopologyDemo {
    mode: TopoMode,
    preset: PointCloudPreset,
    circle_n: usize,
    max_radius: f64,
    betti_radius: f64,
    betti_steps: usize,
    top_k: usize,
    // Computed results
    points: Vec<Vec<f64>>,
    persistence_scatter: Vec<[f64; 2]>,
    top_features: Vec<(usize, f64, f64, f64)>,
    betti0_curve: Vec<[f64; 2]>,
    betti1_curve: Vec<[f64; 2]>,
    betti_at_r: Vec<usize>,
}

impl Default for TopologyDemo {
    fn default() -> Self {
        Self {
            mode: TopoMode::Persistence,
            preset: PointCloudPreset::Triangle,
            circle_n: 12,
            max_radius: 2.0,
            betti_radius: 1.0,
            betti_steps: 20,
            top_k: 5,
            points: Vec::new(),
            persistence_scatter: Vec::new(),
            top_features: Vec::new(),
            betti0_curve: Vec::new(),
            betti1_curve: Vec::new(),
            betti_at_r: Vec::new(),
        }
    }
}

impl TopologyDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Topology / Persistent Homology (ix-topo)");

        ui.horizontal(|ui| {
            ui.radio_value(
                &mut self.mode,
                TopoMode::Persistence,
                "Point Cloud Persistence",
            );
            ui.radio_value(&mut self.mode, TopoMode::BettiCurve, "Betti Curve");
            ui.radio_value(&mut self.mode, TopoMode::BettiAtRadius, "Betti at Radius");
        });

        ui.separator();
        self.point_cloud_controls(ui);

        match self.mode {
            TopoMode::Persistence => self.persistence_ui(ui),
            TopoMode::BettiCurve => self.betti_curve_ui(ui),
            TopoMode::BettiAtRadius => self.betti_at_radius_ui(ui),
        }
    }

    fn point_cloud_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Preset:");
            ui.radio_value(&mut self.preset, PointCloudPreset::Triangle, "Triangle");
            ui.radio_value(&mut self.preset, PointCloudPreset::Circle, "Circle");
            ui.radio_value(
                &mut self.preset,
                PointCloudPreset::TwoClusters,
                "Two Clusters",
            );
        });

        if self.preset == PointCloudPreset::Circle {
            ui.add(egui::Slider::new(&mut self.circle_n, 4..=50).text("Circle points"));
        }
    }

    fn generate_points(&mut self) {
        self.points = match self.preset {
            PointCloudPreset::Triangle => vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866]],
            PointCloudPreset::Circle => {
                let n = self.circle_n;
                (0..n)
                    .map(|i| {
                        let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                        vec![theta.cos(), theta.sin()]
                    })
                    .collect()
            }
            PointCloudPreset::TwoClusters => vec![
                vec![0.0, 0.0],
                vec![0.1, 0.0],
                vec![0.0, 0.1],
                vec![0.1, 0.1],
                vec![5.0, 5.0],
                vec![5.1, 5.0],
                vec![5.0, 5.1],
                vec![5.1, 5.1],
            ],
        };
    }

    fn persistence_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.max_radius, 0.1..=10.0).text("Max radius"));
            ui.add(egui::Slider::new(&mut self.top_k, 1..=20).text("Top K features"));
        });

        if ui.button("Compute Persistence").clicked() {
            self.generate_points();
            let diagrams =
                ix_topo::pointcloud::persistence_from_points(&self.points, 2, self.max_radius);

            self.persistence_scatter.clear();
            for d in &diagrams {
                for &(birth, death) in &d.pairs {
                    if death.is_finite() {
                        self.persistence_scatter.push([birth, death]);
                    }
                }
            }

            self.top_features =
                ix_topo::pointcloud::most_persistent_features(&diagrams, self.top_k);
        }

        if !self.persistence_scatter.is_empty() {
            Plot::new("persistence_diagram")
                .height(400.0)
                .data_aspect(1.0)
                .x_axis_label("Birth")
                .y_axis_label("Death")
                .show(ui, |plot_ui| {
                    let pts: PlotPoints = self.persistence_scatter.iter().copied().collect();
                    plot_ui.points(
                        Points::new(pts)
                            .radius(4.0)
                            .color(egui::Color32::LIGHT_BLUE)
                            .name("(birth, death)"),
                    );
                    // Diagonal reference line
                    let diag: PlotPoints = [[0.0, 0.0], [self.max_radius, self.max_radius]]
                        .iter()
                        .copied()
                        .collect();
                    plot_ui.line(
                        Line::new(diag)
                            .color(egui::Color32::DARK_GRAY)
                            .name("diagonal"),
                    );
                });
        }

        if !self.top_features.is_empty() {
            ui.separator();
            ui.label("Most persistent features:");
            egui::Grid::new("top_features_grid")
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Dim");
                    ui.label("Birth");
                    ui.label("Death");
                    ui.label("Persistence");
                    ui.end_row();
                    for &(dim, birth, death, persistence) in &self.top_features {
                        ui.label(format!("H{}", dim));
                        ui.label(format!("{:.4}", birth));
                        ui.label(format!("{:.4}", death));
                        ui.label(format!("{:.4}", persistence));
                        ui.end_row();
                    }
                });
        }
    }

    fn betti_curve_ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.betti_steps, 5..=100).text("Radius steps"));

        if ui.button("Compute Betti Curve").clicked() {
            self.generate_points();
            let curve = ix_topo::pointcloud::betti_curve(&self.points, 2, self.betti_steps);

            self.betti0_curve.clear();
            self.betti1_curve.clear();
            for (r, betti) in &curve {
                let b0 = if !betti.is_empty() {
                    betti[0] as f64
                } else {
                    0.0
                };
                let b1 = if betti.len() > 1 {
                    betti[1] as f64
                } else {
                    0.0
                };
                self.betti0_curve.push([*r, b0]);
                self.betti1_curve.push([*r, b1]);
            }
        }

        if !self.betti0_curve.is_empty() {
            Plot::new("betti_curve")
                .height(400.0)
                .x_axis_label("Radius")
                .y_axis_label("Betti number")
                .show(ui, |plot_ui| {
                    let b0_pts: PlotPoints = self.betti0_curve.iter().copied().collect();
                    plot_ui.line(
                        Line::new(b0_pts)
                            .name("b0")
                            .width(2.0)
                            .color(egui::Color32::LIGHT_BLUE),
                    );
                    let b1_pts: PlotPoints = self.betti1_curve.iter().copied().collect();
                    plot_ui.line(
                        Line::new(b1_pts)
                            .name("b1")
                            .width(2.0)
                            .color(egui::Color32::LIGHT_RED),
                    );
                });
        }
    }

    fn betti_at_radius_ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.betti_radius, 0.01..=10.0).text("Radius"));

        if ui.button("Compute Betti Numbers").clicked() {
            self.generate_points();
            self.betti_at_r =
                ix_topo::pointcloud::betti_at_radius(&self.points, 2, self.betti_radius);
        }

        if !self.betti_at_r.is_empty() {
            ui.separator();
            let b0 = self.betti_at_r[0];
            let b1 = if self.betti_at_r.len() > 1 {
                self.betti_at_r[1]
            } else {
                0
            };
            ui.label(format!(
                "At radius {:.4}:  b0 = {} (connected components),  b1 = {} (loops)",
                self.betti_radius, b0, b1
            ));
        }
    }
}
