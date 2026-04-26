use eframe::egui;
use egui_plot::{Bar, BarChart, Plot};

pub struct ProbabilisticDemo {
    n_elements: usize,
    bloom_fp_rate: f64,
    bloom_result: Option<BloomResult>,
    hll_result: Option<HllResult>,
}

struct BloomResult {
    expected_elements: usize,
    actual_fp_rate: f64,
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
}

struct HllResult {
    actual_count: usize,
    estimated_count: f64,
    error_pct: f64,
}

impl Default for ProbabilisticDemo {
    fn default() -> Self {
        Self {
            n_elements: 1000,
            bloom_fp_rate: 0.01,
            bloom_result: None,
            hll_result: None,
        }
    }
}

impl ProbabilisticDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Probabilistic Data Structures (ix-probabilistic)");

        ui.add(egui::Slider::new(&mut self.n_elements, 100..=10000).text("Elements"));

        ui.separator();
        ui.subheading("Bloom Filter");
        ui.add(
            egui::Slider::new(&mut self.bloom_fp_rate, 0.001..=0.1)
                .text("Target FP rate")
                .logarithmic(true),
        );

        if ui.button("Test Bloom Filter").clicked() {
            self.test_bloom();
        }

        if let Some(r) = &self.bloom_result {
            egui::Grid::new("bloom_grid").show(ui, |ui| {
                ui.label("Elements inserted:");
                ui.label(format!("{}", r.expected_elements));
                ui.end_row();
                ui.label("True positives:");
                ui.label(format!("{}", r.true_positives));
                ui.end_row();
                ui.label("False positives:");
                ui.label(format!("{}", r.false_positives));
                ui.end_row();
                ui.label("True negatives:");
                ui.label(format!("{}", r.true_negatives));
                ui.end_row();
                ui.label("Actual FP rate:");
                ui.label(format!("{:.4}", r.actual_fp_rate));
                ui.end_row();
            });
        }

        ui.separator();
        ui.subheading("HyperLogLog Cardinality");

        if ui.button("Test HyperLogLog").clicked() {
            self.test_hll();
        }

        if let Some(r) = &self.hll_result {
            egui::Grid::new("hll_grid").show(ui, |ui| {
                ui.label("Actual unique:");
                ui.label(format!("{}", r.actual_count));
                ui.end_row();
                ui.label("Estimated:");
                ui.label(format!("{:.1}", r.estimated_count));
                ui.end_row();
                ui.label("Error:");
                ui.label(format!("{:.2}%", r.error_pct));
                ui.end_row();
            });

            let bars = vec![
                Bar::new(0.0, r.actual_count as f64).name("Actual"),
                Bar::new(1.0, r.estimated_count).name("Estimated"),
            ];
            Plot::new("hll_plot").height(200.0).show(ui, |plot_ui| {
                plot_ui.bar_chart(BarChart::new(bars).width(0.6));
            });
        }
    }

    fn test_bloom(&mut self) {
        use ix_probabilistic::bloom::BloomFilter;
        use std::collections::HashSet;

        let mut bf = BloomFilter::new(self.n_elements, self.bloom_fp_rate);
        let mut inserted = HashSet::new();

        // Insert elements
        for i in 0..self.n_elements {
            let key = format!("key-{i}");
            bf.insert(&key);
            inserted.insert(key);
        }

        // Test membership
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let test_count = self.n_elements;

        for i in 0..test_count * 2 {
            let key = format!("key-{i}");
            let in_set = inserted.contains(&key);
            let in_bloom = bf.contains(&key);

            if in_set && in_bloom {
                tp += 1;
            } else if !in_set && in_bloom {
                fp += 1;
            } else if !in_set && !in_bloom {
                tn += 1;
            }
        }

        let neg_count = (fp + tn) as f64;
        self.bloom_result = Some(BloomResult {
            expected_elements: self.n_elements,
            actual_fp_rate: if neg_count > 0.0 {
                fp as f64 / neg_count
            } else {
                0.0
            },
            true_positives: tp,
            false_positives: fp,
            true_negatives: tn,
        });
    }

    fn test_hll(&mut self) {
        use ix_probabilistic::hyperloglog::HyperLogLog;

        let mut hll = HyperLogLog::new(12); // 2^12 = 4096 registers
        let mut unique = std::collections::HashSet::new();

        for i in 0..self.n_elements {
            let key = format!("element-{i}");
            hll.add(&key);
            unique.insert(key);
        }
        // Add some duplicates
        for i in 0..self.n_elements / 2 {
            let key = format!("element-{i}");
            hll.add(&key);
        }

        let estimated = hll.count();
        let actual = unique.len();
        let error = ((estimated - actual as f64).abs() / actual as f64) * 100.0;

        self.hll_result = Some(HllResult {
            actual_count: actual,
            estimated_count: estimated,
            error_pct: error,
        });
    }
}

trait SubheadingExt {
    fn subheading(&mut self, text: &str);
}

impl SubheadingExt for egui::Ui {
    fn subheading(&mut self, text: &str) {
        self.label(egui::RichText::new(text).strong().size(16.0));
    }
}
