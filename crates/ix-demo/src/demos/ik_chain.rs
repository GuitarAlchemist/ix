use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};

#[derive(PartialEq, Clone, Copy)]
enum IKSolver {
    Ccd,
    Jacobian,
}

pub struct IKChainDemo {
    n_links: usize,
    link_length: f64,
    target_x: f64,
    target_y: f64,
    solver: IKSolver,
    joint_angles: Vec<f64>,
    joint_positions: Vec<[f64; 2]>,
    status: String,
    animate: bool,
    anim_angle: f64,
}

impl Default for IKChainDemo {
    fn default() -> Self {
        Self {
            n_links: 3,
            link_length: 1.0,
            target_x: 2.0,
            target_y: 1.0,
            solver: IKSolver::Ccd,
            joint_angles: Vec::new(),
            joint_positions: Vec::new(),
            status: "Set target and solve".into(),
            animate: false,
            anim_angle: 0.0,
        }
    }
}

impl IKChainDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("IK Chain (ix-dynamics)");

        ui.horizontal(|ui| {
            ui.label("Links:");
            ui.add(egui::Slider::new(&mut self.n_links, 2..=8));
            ui.label("Length:");
            ui.add(egui::Slider::new(&mut self.link_length, 0.3..=2.0));
        });
        ui.horizontal(|ui| {
            ui.label("Target X:");
            ui.add(egui::Slider::new(&mut self.target_x, -5.0..=5.0));
            ui.label("Target Y:");
            ui.add(egui::Slider::new(&mut self.target_y, -5.0..=5.0));
        });
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.solver, IKSolver::Ccd, "CCD");
            ui.radio_value(&mut self.solver, IKSolver::Jacobian, "Jacobian (DLS)");
            if ui.button("Solve").clicked() {
                self.solve();
            }
            ui.checkbox(&mut self.animate, "Animate");
        });

        if self.animate {
            self.anim_angle += 0.02;
            self.target_x = self.anim_angle.cos() * (self.n_links as f64 * self.link_length * 0.7);
            self.target_y =
                self.anim_angle.sin() * (self.n_links as f64 * self.link_length * 0.7) + 1.0;
            self.solve();
            ui.ctx().request_repaint();
        }

        ui.label(&self.status);

        let max_reach = self.n_links as f64 * self.link_length;
        Plot::new("ik_plot")
            .height(500.0)
            .data_aspect(1.0)
            .include_x(-max_reach - 1.0)
            .include_x(max_reach + 1.0)
            .include_y(-1.0)
            .include_y(max_reach + 1.0)
            .show(ui, |plot_ui| {
                // Chain links
                if !self.joint_positions.is_empty() {
                    let data = self.joint_positions.clone();
                    let pts1: PlotPoints = data.iter().copied().collect();
                    let pts2: PlotPoints = data.iter().copied().collect();
                    plot_ui.line(
                        Line::new(pts1)
                            .name("Chain")
                            .width(3.0)
                            .color(egui::Color32::from_rgb(100, 180, 255)),
                    );
                    plot_ui.points(
                        Points::new(pts2)
                            .radius(6.0)
                            .color(egui::Color32::from_rgb(100, 180, 255))
                            .name("Joints"),
                    );
                }

                // Target
                plot_ui.points(
                    Points::new(PlotPoints::new(vec![[self.target_x, self.target_y]]))
                        .radius(10.0)
                        .color(egui::Color32::RED)
                        .name("Target")
                        .shape(egui_plot::MarkerShape::Cross),
                );

                // Reach circle
                let circle: PlotPoints = (0..=64)
                    .map(|i| {
                        let a = 2.0 * std::f64::consts::PI * i as f64 / 64.0;
                        [a.cos() * max_reach, a.sin() * max_reach]
                    })
                    .collect();
                plot_ui.line(
                    Line::new(circle)
                        .name("Reach limit")
                        .style(egui_plot::LineStyle::dashed_dense())
                        .color(egui::Color32::from_rgba_premultiplied(150, 150, 150, 80)),
                );
            });
    }

    fn solve(&mut self) {
        use ix_dynamics::ik::{Chain, Joint};

        // Build a planar chain (all revolute around Z axis)
        let joints: Vec<Joint> = (0..self.n_links)
            .map(|_| Joint::revolute([0.0, 0.0, 1.0], [self.link_length, 0.0, 0.0]))
            .collect();
        let chain = Chain::new(joints);

        let target = [self.target_x, self.target_y, 0.0];
        let initial: Vec<f64> = vec![0.3; self.n_links]; // slight bend

        let result = match self.solver {
            IKSolver::Ccd => chain.solve_ccd(&target, &initial, 500, 0.01),
            IKSolver::Jacobian => chain.solve_jacobian(&target, &initial, 200, 0.01, 0.5),
        };

        match result {
            Ok(angles) => {
                // Compute joint positions for visualization (2D projection)
                let mut positions = vec![[0.0, 0.0]]; // base
                let mut cum_angle = 0.0;
                let mut x = 0.0;
                let mut y = 0.0;

                for &angle in angles.iter() {
                    cum_angle += angle;
                    x += self.link_length * cum_angle.cos();
                    y += self.link_length * cum_angle.sin();
                    positions.push([x, y]);
                }

                let ee = positions.last().unwrap();
                let dist =
                    ((ee[0] - self.target_x).powi(2) + (ee[1] - self.target_y).powi(2)).sqrt();
                self.status = format!("Solved! End-effector error: {:.4}", dist);
                self.joint_angles = angles;
                self.joint_positions = positions;
            }
            Err(e) => {
                self.status = format!("Error: {e}");
            }
        }
    }
}
