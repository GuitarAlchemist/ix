use eframe::egui;
use egui_plot::{Plot, Points, PlotPoints};

#[derive(PartialEq, Clone, Copy)]
enum GpuDemo {
    QuaternionRotate,
    Knn,
    PairwiseDistance,
}

pub struct GpuKernelsDemo {
    mode: GpuDemo,
    // Quaternion
    quat_angle: f64,
    quat_axis: usize, // 0=X, 1=Y, 2=Z
    quat_points: Vec<[f64; 3]>,
    quat_rotated: Vec<[f64; 3]>,
    // kNN
    knn_k: usize,
    knn_query: [f64; 2],
    knn_refs: Vec<[f64; 2]>,
    knn_neighbors: Vec<usize>,
    // Distance matrix
    dist_points: Vec<[f64; 2]>,
    dist_matrix: Vec<Vec<f64>>,
    status: String,
}

impl Default for GpuKernelsDemo {
    fn default() -> Self {
        Self {
            mode: GpuDemo::QuaternionRotate,
            quat_angle: 45.0,
            quat_axis: 2,
            quat_points: Vec::new(),
            quat_rotated: Vec::new(),
            knn_k: 3,
            knn_query: [0.0, 0.0],
            knn_refs: Vec::new(),
            knn_neighbors: Vec::new(),
            dist_points: Vec::new(),
            dist_matrix: Vec::new(),
            status: "GPU kernels (CPU fallback mode)".into(),
        }
    }
}

impl GpuKernelsDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("GPU Compute Kernels (machin-gpu)");
        ui.label("Using CPU fallback — same algorithms as WGSL shaders.");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, GpuDemo::QuaternionRotate, "Quaternion Rotate");
            ui.radio_value(&mut self.mode, GpuDemo::Knn, "k-NN");
            ui.radio_value(&mut self.mode, GpuDemo::PairwiseDistance, "Distance Matrix");
        });

        match self.mode {
            GpuDemo::QuaternionRotate => self.quaternion_ui(ui),
            GpuDemo::Knn => self.knn_ui(ui),
            GpuDemo::PairwiseDistance => self.distance_ui(ui),
        }
    }

    fn quaternion_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Angle (deg):"); ui.add(egui::Slider::new(&mut self.quat_angle, 0.0..=360.0));
            ui.label("Axis:");
            ui.radio_value(&mut self.quat_axis, 0, "X");
            ui.radio_value(&mut self.quat_axis, 1, "Y");
            ui.radio_value(&mut self.quat_axis, 2, "Z");
        });

        if ui.button("Generate & Rotate").clicked() {
            self.run_quaternion();
        }

        ui.label(&self.status);

        // Plot XY projection
        Plot::new("quat_plot").height(400.0).data_aspect(1.0).show(ui, |plot_ui| {
            if !self.quat_points.is_empty() {
                let pts: PlotPoints = self.quat_points.iter().map(|p| [p[0], p[1]]).collect();
                plot_ui.points(Points::new(pts).radius(4.0)
                    .color(egui::Color32::from_rgb(100, 100, 255)).name("Original"));
            }
            if !self.quat_rotated.is_empty() {
                let pts: PlotPoints = self.quat_rotated.iter().map(|p| [p[0], p[1]]).collect();
                plot_ui.points(Points::new(pts).radius(4.0)
                    .color(egui::Color32::from_rgb(255, 100, 100)).name("Rotated"));
            }
        });
    }

    fn run_quaternion(&mut self) {
        use machin_gpu::quaternion::batch_quaternion_rotate_cpu;

        // Generate a grid of points
        self.quat_points.clear();
        for i in -5..=5 {
            for j in -5..=5 {
                self.quat_points.push([i as f64 * 0.3, j as f64 * 0.3, 0.0]);
            }
        }

        let angle_rad = self.quat_angle.to_radians() / 2.0;
        let (s, c) = (angle_rad.sin() as f32, angle_rad.cos() as f32);
        let quat: [f32; 4] = match self.quat_axis {
            0 => [c, s, 0.0, 0.0],
            1 => [c, 0.0, s, 0.0],
            _ => [c, 0.0, 0.0, s],
        };

        let flat: Vec<f32> = self.quat_points.iter()
            .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();

        let result = batch_quaternion_rotate_cpu(&flat, &quat);

        self.quat_rotated = result.chunks(3)
            .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
            .collect();

        self.status = format!("Rotated {} points by {:.0}° around {} axis",
            self.quat_points.len(), self.quat_angle,
            ["X", "Y", "Z"][self.quat_axis]);
    }

    fn knn_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("k:"); ui.add(egui::Slider::new(&mut self.knn_k, 1..=10));
            ui.label("Query X:"); ui.add(egui::Slider::new(&mut self.knn_query[0], -5.0..=5.0));
            ui.label("Query Y:"); ui.add(egui::Slider::new(&mut self.knn_query[1], -5.0..=5.0));
        });

        if ui.button("Generate & Query").clicked() {
            self.run_knn();
        }

        ui.label(&self.status);

        Plot::new("knn_plot").height(400.0).data_aspect(1.0).show(ui, |plot_ui| {
            // All ref points
            if !self.knn_refs.is_empty() {
                let non_neighbor: PlotPoints = self.knn_refs.iter().enumerate()
                    .filter(|(i, _)| !self.knn_neighbors.contains(i))
                    .map(|(_, p)| *p)
                    .collect();
                plot_ui.points(Points::new(non_neighbor).radius(4.0)
                    .color(egui::Color32::GRAY).name("Ref points"));

                let neighbor_pts: PlotPoints = self.knn_neighbors.iter()
                    .filter(|&&i| i < self.knn_refs.len())
                    .map(|&i| self.knn_refs[i])
                    .collect();
                plot_ui.points(Points::new(neighbor_pts).radius(7.0)
                    .color(egui::Color32::GREEN).name("Neighbors"));
            }

            // Query point
            plot_ui.points(Points::new(PlotPoints::new(vec![self.knn_query]))
                .radius(10.0).color(egui::Color32::RED).name("Query")
                .shape(egui_plot::MarkerShape::Cross));
        });
    }

    fn run_knn(&mut self) {
        use rand::Rng;
        use machin_gpu::knn::batch_knn_cpu;
        let mut rng = rand::rng();

        // Generate random 2D ref points
        self.knn_refs = (0..50).map(|_| {
            [rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)]
        }).collect();

        let refs_flat: Vec<f32> = self.knn_refs.iter()
            .flat_map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        let query_flat = vec![self.knn_query[0] as f32, self.knn_query[1] as f32];

        let (indices, dists) = batch_knn_cpu(&refs_flat, &query_flat, 2, self.knn_k);

        self.knn_neighbors = indices.iter()
            .filter(|&&i| i != u32::MAX)
            .map(|&i| i as usize)
            .collect();

        let nearest_dist = dists.first().copied().unwrap_or(f32::NAN);
        self.status = format!("{} nearest neighbors found. Closest: {:.3}",
            self.knn_neighbors.len(), nearest_dist);
    }

    fn distance_ui(&mut self, ui: &mut egui::Ui) {
        if ui.button("Generate Points & Compute").clicked() {
            self.run_distance();
        }

        ui.label(&self.status);

        // Show distance matrix as colored grid
        if !self.dist_matrix.is_empty() {
            let n = self.dist_matrix.len();
            Plot::new("dist_plot").height(400.0).data_aspect(1.0).show(ui, |plot_ui| {
                // Plot points
                let pts: PlotPoints = self.dist_points.iter().copied().collect();
                plot_ui.points(Points::new(pts).radius(6.0).color(egui::Color32::WHITE).name("Points"));
            });

            // Distance matrix text
            ui.label("Distance matrix (first 8x8):");
            egui::Grid::new("dist_grid").striped(true).show(ui, |ui| {
                let show_n = n.min(8);
                ui.label("");
                for j in 0..show_n {
                    ui.label(format!("P{j}"));
                }
                ui.end_row();
                for i in 0..show_n {
                    ui.label(format!("P{i}"));
                    for j in 0..show_n {
                        ui.label(format!("{:.2}", self.dist_matrix[i][j]));
                    }
                    ui.end_row();
                }
            });
        }
    }

    fn run_distance(&mut self) {
        use rand::Rng;
        use machin_gpu::distance::pairwise_distance_cpu;
        let mut rng = rand::rng();

        let n = 10;
        self.dist_points = (0..n).map(|_| {
            [rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)]
        }).collect();

        let flat: Vec<f32> = self.dist_points.iter()
            .flat_map(|p| [p[0] as f32, p[1] as f32])
            .collect();

        let result = pairwise_distance_cpu(&flat, 2);

        self.dist_matrix = (0..n).map(|i| {
            (0..n).map(|j| result[i * n + j] as f64).collect()
        }).collect();

        self.status = format!("{n}x{n} pairwise distance matrix computed");
    }
}
