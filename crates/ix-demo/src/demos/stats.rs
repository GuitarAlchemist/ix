use eframe::egui;
use egui_plot::{Bar, BarChart, Plot};
use ndarray::Array1;

pub struct StatsDemo {
    data_text: String,
    results: Option<StatsResults>,
}

struct StatsResults {
    mean: f64,
    variance: f64,
    std_dev: f64,
    median: f64,
    min: f64,
    max: f64,
    sample_var: Option<f64>,
}

impl Default for StatsDemo {
    fn default() -> Self {
        Self {
            data_text: "3.2, 7.1, 5.5, 2.8, 9.0, 4.3, 6.7, 1.9, 8.4, 5.0".into(),
            results: None,
        }
    }
}

impl StatsDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Statistics (ix-math)");
        ui.label("Enter comma-separated values:");
        ui.text_edit_singleline(&mut self.data_text);

        if ui.button("Compute").clicked() {
            self.compute();
        }

        if let Some(r) = &self.results {
            ui.separator();
            egui::Grid::new("stats_grid").striped(true).show(ui, |ui| {
                ui.label("Mean:");
                ui.label(format!("{:.4}", r.mean));
                ui.end_row();
                ui.label("Variance:");
                ui.label(format!("{:.4}", r.variance));
                ui.end_row();
                ui.label("Std Dev:");
                ui.label(format!("{:.4}", r.std_dev));
                ui.end_row();
                ui.label("Median:");
                ui.label(format!("{:.4}", r.median));
                ui.end_row();
                ui.label("Min:");
                ui.label(format!("{:.4}", r.min));
                ui.end_row();
                ui.label("Max:");
                ui.label(format!("{:.4}", r.max));
                ui.end_row();
                if let Some(sv) = r.sample_var {
                    ui.label("Sample Var:");
                    ui.label(format!("{:.4}", sv));
                    ui.end_row();
                }
            });

            // Histogram
            let vals = self.parse_data();
            if vals.len() > 1 {
                let bars: Vec<Bar> = vals
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| Bar::new(i as f64, v))
                    .collect();
                Plot::new("stats_plot").height(250.0).show(ui, |plot_ui| {
                    plot_ui.bar_chart(BarChart::new(bars).name("Data"));
                });
            }
        }
    }

    fn parse_data(&self) -> Vec<f64> {
        self.data_text
            .split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect()
    }

    fn compute(&mut self) {
        let vals = self.parse_data();
        if vals.is_empty() {
            return;
        }
        let arr = Array1::from_vec(vals);

        let mean = ix_math::stats::mean(&arr).unwrap_or(0.0);
        let variance = ix_math::stats::variance(&arr).unwrap_or(0.0);
        let std_dev = ix_math::stats::std_dev(&arr).unwrap_or(0.0);
        let median = ix_math::stats::median(&arr).unwrap_or(0.0);
        let (min, max) = ix_math::stats::min_max(&arr).unwrap_or((0.0, 0.0));
        let sample_var = ix_math::stats::sample_variance(&arr).ok();

        self.results = Some(StatsResults {
            mean,
            variance,
            std_dev,
            median,
            min,
            max,
            sample_var,
        });
    }
}
