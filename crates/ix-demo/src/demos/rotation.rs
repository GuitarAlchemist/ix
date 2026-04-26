use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

#[derive(PartialEq, Clone, Copy)]
enum RotationMode {
    Slerp,
    Euler,
    RotationMatrix,
}

pub struct RotationDemo {
    mode: RotationMode,
    // SLERP
    axis1: [f64; 3],
    angle1: f64,
    axis2: [f64; 3],
    angle2: f64,
    slerp_samples: usize,
    slerp_data_w: Vec<[f64; 2]>,
    slerp_data_x: Vec<[f64; 2]>,
    slerp_data_y: Vec<[f64; 2]>,
    slerp_data_z: Vec<[f64; 2]>,
    // Euler
    roll: f64,
    pitch: f64,
    yaw: f64,
    euler_result: String,
    // Rotation Matrix
    rm_axis: [f64; 3],
    rm_angle: f64,
    rm_result: String,
}

impl Default for RotationDemo {
    fn default() -> Self {
        Self {
            mode: RotationMode::Slerp,
            axis1: [1.0, 0.0, 0.0],
            angle1: 0.0,
            axis2: [0.0, 1.0, 0.0],
            angle2: std::f64::consts::PI,
            slerp_samples: 50,
            slerp_data_w: Vec::new(),
            slerp_data_x: Vec::new(),
            slerp_data_y: Vec::new(),
            slerp_data_z: Vec::new(),
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            euler_result: String::new(),
            rm_axis: [1.0, 1.0, 0.0],
            rm_angle: 0.7,
            rm_result: String::new(),
        }
    }
}

impl RotationDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("3D Rotations (ix-rotation)");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, RotationMode::Slerp, "SLERP Animation");
            ui.radio_value(&mut self.mode, RotationMode::Euler, "Euler Angles");
            ui.radio_value(
                &mut self.mode,
                RotationMode::RotationMatrix,
                "Rotation Matrix",
            );
        });

        match self.mode {
            RotationMode::Slerp => self.slerp_ui(ui),
            RotationMode::Euler => self.euler_ui(ui),
            RotationMode::RotationMatrix => self.rotation_matrix_ui(ui),
        }
    }

    fn slerp_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Interpolate between two quaternions (defined via axis-angle) using SLERP.");

        ui.separator();
        ui.label("Quaternion A (axis-angle):");
        ui.horizontal(|ui| {
            ui.label("Axis X:");
            ui.add(egui::Slider::new(&mut self.axis1[0], -1.0..=1.0));
            ui.label("Y:");
            ui.add(egui::Slider::new(&mut self.axis1[1], -1.0..=1.0));
            ui.label("Z:");
            ui.add(egui::Slider::new(&mut self.axis1[2], -1.0..=1.0));
        });
        ui.add(
            egui::Slider::new(&mut self.angle1, 0.0..=std::f64::consts::TAU).text("Angle A (rad)"),
        );

        ui.separator();
        ui.label("Quaternion B (axis-angle):");
        ui.horizontal(|ui| {
            ui.label("Axis X:");
            ui.add(egui::Slider::new(&mut self.axis2[0], -1.0..=1.0));
            ui.label("Y:");
            ui.add(egui::Slider::new(&mut self.axis2[1], -1.0..=1.0));
            ui.label("Z:");
            ui.add(egui::Slider::new(&mut self.axis2[2], -1.0..=1.0));
        });
        ui.add(
            egui::Slider::new(&mut self.angle2, 0.0..=std::f64::consts::TAU).text("Angle B (rad)"),
        );

        ui.add(egui::Slider::new(&mut self.slerp_samples, 10..=200).text("Samples"));

        if ui.button("Interpolate").clicked() {
            self.compute_slerp();
        }

        if !self.slerp_data_w.is_empty() {
            Plot::new("slerp_plot").height(300.0).show(ui, |plot_ui| {
                let w: PlotPoints = self.slerp_data_w.iter().copied().collect();
                let x: PlotPoints = self.slerp_data_x.iter().copied().collect();
                let y: PlotPoints = self.slerp_data_y.iter().copied().collect();
                let z: PlotPoints = self.slerp_data_z.iter().copied().collect();
                plot_ui.line(
                    Line::new(w)
                        .name("w")
                        .width(2.0)
                        .color(egui::Color32::WHITE),
                );
                plot_ui.line(Line::new(x).name("x").width(2.0).color(egui::Color32::RED));
                plot_ui.line(
                    Line::new(y)
                        .name("y")
                        .width(2.0)
                        .color(egui::Color32::GREEN),
                );
                plot_ui.line(
                    Line::new(z)
                        .name("z")
                        .width(2.0)
                        .color(egui::Color32::from_rgb(80, 140, 255)),
                );
            });
        }
    }

    fn compute_slerp(&mut self) {
        use ix_rotation::quaternion::Quaternion;
        use ix_rotation::slerp::slerp_array;

        let q0 = Quaternion::from_axis_angle(self.axis1, self.angle1);
        let q1 = Quaternion::from_axis_angle(self.axis2, self.angle2);
        let samples = slerp_array(&q0, &q1, self.slerp_samples);

        self.slerp_data_w.clear();
        self.slerp_data_x.clear();
        self.slerp_data_y.clear();
        self.slerp_data_z.clear();

        let n = samples.len();
        for (i, q) in samples.iter().enumerate() {
            let t = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            self.slerp_data_w.push([t, q.w]);
            self.slerp_data_x.push([t, q.x]);
            self.slerp_data_y.push([t, q.y]);
            self.slerp_data_z.push([t, q.z]);
        }
    }

    fn euler_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Convert Euler angles (XYZ order) to quaternion and back.");

        ui.add(
            egui::Slider::new(&mut self.roll, -std::f64::consts::PI..=std::f64::consts::PI)
                .text("Roll (rad)"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.pitch,
                -std::f64::consts::FRAC_PI_2..=std::f64::consts::FRAC_PI_2,
            )
            .text("Pitch (rad)"),
        );
        ui.add(
            egui::Slider::new(&mut self.yaw, -std::f64::consts::PI..=std::f64::consts::PI)
                .text("Yaw (rad)"),
        );

        if ix_rotation::euler::gimbal_lock_check(self.pitch) {
            ui.colored_label(
                egui::Color32::YELLOW,
                "Warning: Pitch is near +/-90 deg — gimbal lock region!",
            );
        }

        if ui.button("Convert").clicked() {
            self.compute_euler();
        }

        if !self.euler_result.is_empty() {
            ui.separator();
            ui.monospace(&self.euler_result);
        }
    }

    fn compute_euler(&mut self) {
        use ix_rotation::euler::{from_quaternion, gimbal_lock_check, to_quaternion, EulerOrder};

        let q = to_quaternion(self.roll, self.pitch, self.yaw, EulerOrder::XYZ);
        let (r2, p2, y2) = from_quaternion(&q, EulerOrder::XYZ);

        let mut text = String::new();
        text.push_str(&format!(
            "Input:  roll={:.4}, pitch={:.4}, yaw={:.4}\n",
            self.roll, self.pitch, self.yaw
        ));
        text.push_str(&format!(
            "Quaternion: w={:.6}, x={:.6}, y={:.6}, z={:.6}\n",
            q.w, q.x, q.y, q.z
        ));
        text.push_str(&format!("Norm: {:.8}\n", q.norm()));
        text.push_str(&format!(
            "Round-trip: roll={:.6}, pitch={:.6}, yaw={:.6}\n",
            r2, p2, y2
        ));
        if gimbal_lock_check(self.pitch) {
            text.push_str("GIMBAL LOCK: pitch near +/-pi/2, yaw is degenerate.\n");
        }

        self.euler_result = text;
    }

    fn rotation_matrix_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Convert a quaternion (via axis-angle) to a 3x3 rotation matrix.");

        ui.horizontal(|ui| {
            ui.label("Axis X:");
            ui.add(egui::Slider::new(&mut self.rm_axis[0], -1.0..=1.0));
            ui.label("Y:");
            ui.add(egui::Slider::new(&mut self.rm_axis[1], -1.0..=1.0));
            ui.label("Z:");
            ui.add(egui::Slider::new(&mut self.rm_axis[2], -1.0..=1.0));
        });
        ui.add(
            egui::Slider::new(&mut self.rm_angle, 0.0..=std::f64::consts::TAU).text("Angle (rad)"),
        );

        if ui.button("Compute Matrix").clicked() {
            self.compute_rotation_matrix();
        }

        if !self.rm_result.is_empty() {
            ui.separator();
            ui.monospace(&self.rm_result);
        }
    }

    fn compute_rotation_matrix(&mut self) {
        use ix_rotation::quaternion::Quaternion;
        use ix_rotation::rotation_matrix::{from_quaternion, is_rotation_matrix};

        let q = Quaternion::from_axis_angle(self.rm_axis, self.rm_angle);
        let m = from_quaternion(&q);

        let mut text = String::new();
        text.push_str(&format!(
            "Quaternion: w={:.6}, x={:.6}, y={:.6}, z={:.6}\n\n",
            q.w, q.x, q.y, q.z
        ));

        text.push_str("Rotation Matrix:\n");
        for row in &m {
            text.push_str(&format!(
                "  [{:>9.6}, {:>9.6}, {:>9.6}]\n",
                row[0], row[1], row[2]
            ));
        }

        // Determinant
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        text.push_str(&format!("\ndet = {:.8}", det));
        if (det - 1.0).abs() < 1e-6 {
            text.push_str("  (det ~ 1 OK)");
        } else {
            text.push_str("  (WARNING: det != 1)");
        }

        let valid = is_rotation_matrix(&m, 1e-8);
        text.push_str(&format!(
            "\nOrthogonality check: {}",
            if valid {
                "PASS (orthogonal, det=1)"
            } else {
                "FAIL"
            }
        ));

        self.rm_result = text;
    }
}
