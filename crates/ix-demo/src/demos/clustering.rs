use eframe::egui;
use egui_plot::{Plot, PlotPoints, Points};
use ndarray::Array2;

pub struct ClusteringDemo {
    k: usize,
    n_points: usize,
    spread: f64,
    result: Option<ClusterResult>,
}

struct ClusterResult {
    points: Vec<[f64; 2]>,
    labels: Vec<usize>,
    centroids: Vec<[f64; 2]>,
    iterations: usize,
}

impl Default for ClusteringDemo {
    fn default() -> Self {
        Self {
            k: 3,
            n_points: 150,
            spread: 1.5,
            result: None,
        }
    }
}

impl ClusteringDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("K-Means Clustering (ix-unsupervised)");

        ui.horizontal(|ui| {
            ui.label("K:");
            ui.add(egui::Slider::new(&mut self.k, 2..=8));
            ui.label("Points:");
            ui.add(egui::Slider::new(&mut self.n_points, 30..=500));
            ui.label("Spread:");
            ui.add(egui::Slider::new(&mut self.spread, 0.5..=5.0));
        });

        if ui.button("Cluster").clicked() {
            self.run();
        }

        if let Some(r) = &self.result {
            ui.label(format!("Converged in {} iterations", r.iterations));

            let colors = [
                egui::Color32::from_rgb(230, 70, 70),
                egui::Color32::from_rgb(70, 130, 230),
                egui::Color32::from_rgb(70, 200, 70),
                egui::Color32::from_rgb(230, 180, 50),
                egui::Color32::from_rgb(180, 70, 230),
                egui::Color32::from_rgb(70, 200, 200),
                egui::Color32::from_rgb(230, 120, 50),
                egui::Color32::from_rgb(200, 70, 150),
            ];

            Plot::new("cluster_plot")
                .height(500.0)
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    for c in 0..self.k {
                        let pts: PlotPoints = r
                            .points
                            .iter()
                            .zip(r.labels.iter())
                            .filter(|(_, &l)| l == c)
                            .map(|(p, _)| *p)
                            .collect();
                        let color = colors[c % colors.len()];
                        plot_ui.points(
                            Points::new(pts)
                                .radius(4.0)
                                .color(color)
                                .name(format!("Cluster {c}")),
                        );
                    }

                    // Centroids
                    let cpts: PlotPoints = r.centroids.iter().copied().collect();
                    plot_ui.points(
                        Points::new(cpts)
                            .radius(10.0)
                            .shape(egui_plot::MarkerShape::Diamond)
                            .color(egui::Color32::WHITE)
                            .name("Centroids"),
                    );
                });
        }
    }

    fn run(&mut self) {
        use rand::Rng;
        let mut rng = rand::rng();

        // Generate clustered data
        let pts_per_cluster = self.n_points / self.k;
        let mut points = Vec::new();
        let centers: Vec<[f64; 2]> = (0..self.k)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / self.k as f64;
                [angle.cos() * 5.0, angle.sin() * 5.0]
            })
            .collect();

        for center in &centers {
            for _ in 0..pts_per_cluster {
                points.push([
                    center[0] + rng.random_range(-self.spread..self.spread),
                    center[1] + rng.random_range(-self.spread..self.spread),
                ]);
            }
        }

        let n = points.len();
        let data = Array2::from_shape_fn((n, 2), |(i, j)| points[i][j]);

        use ix_unsupervised::traits::Clusterer;
        let mut km = ix_unsupervised::kmeans::KMeans::new(self.k);
        let labels_arr = km.fit_predict(&data);
        let labels: Vec<usize> = labels_arr.to_vec();

        if let Some(ref centroids_arr) = km.centroids {
            let centroids: Vec<[f64; 2]> =
                centroids_arr.outer_iter().map(|r| [r[0], r[1]]).collect();

            self.result = Some(ClusterResult {
                points,
                labels,
                centroids,
                iterations: 100,
            });
        }
    }
}
