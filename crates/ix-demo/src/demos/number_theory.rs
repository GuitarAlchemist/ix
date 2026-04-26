use eframe::egui;
use egui_plot::{Plot, PlotPoints, Points};

#[derive(PartialEq, Clone, Copy)]
enum Mode {
    PrimeSieve,
    PrimeGaps,
    ModularArithmetic,
    PrimalityTest,
}

pub struct NumberTheoryDemo {
    mode: Mode,
    sieve_limit: usize,
    sieve_primes: Option<Vec<usize>>,
    gaps_limit: usize,
    gaps_result: Option<Vec<usize>>,
    mod_base: u64,
    mod_exp: u64,
    mod_modulus: u64,
    mod_result: Option<u64>,
    primality_input: u64,
    primality_result: Option<bool>,
}

impl Default for NumberTheoryDemo {
    fn default() -> Self {
        Self {
            mode: Mode::PrimeSieve,
            sieve_limit: 1000,
            sieve_primes: None,
            gaps_limit: 1000,
            gaps_result: None,
            mod_base: 2,
            mod_exp: 10,
            mod_modulus: 1000,
            mod_result: None,
            primality_input: 104729,
            primality_result: None,
        }
    }
}

impl NumberTheoryDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Number Theory (ix-number-theory)");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, Mode::PrimeSieve, "Prime Sieve");
            ui.radio_value(&mut self.mode, Mode::PrimeGaps, "Prime Gaps");
            ui.radio_value(
                &mut self.mode,
                Mode::ModularArithmetic,
                "Modular Arithmetic",
            );
            ui.radio_value(&mut self.mode, Mode::PrimalityTest, "Primality Test");
        });

        ui.separator();

        match self.mode {
            Mode::PrimeSieve => self.ui_prime_sieve(ui),
            Mode::PrimeGaps => self.ui_prime_gaps(ui),
            Mode::ModularArithmetic => self.ui_modular(ui),
            Mode::PrimalityTest => self.ui_primality(ui),
        }
    }

    fn ui_prime_sieve(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("Prime Sieve (Eratosthenes)")
                .strong()
                .size(16.0),
        );

        ui.add(
            egui::Slider::new(&mut self.sieve_limit, 100..=1_000_000)
                .text("Limit")
                .logarithmic(true),
        );

        if ui.button("Run Sieve").clicked() {
            let primes = ix_number_theory::sieve::sieve_of_eratosthenes(self.sieve_limit);
            self.sieve_primes = Some(primes);
        }

        if let Some(primes) = &self.sieve_primes {
            ui.label(format!("Primes found: {}", primes.len()));

            let points: PlotPoints = primes
                .iter()
                .enumerate()
                .map(|(i, &p)| [i as f64, p as f64])
                .collect();

            Plot::new("sieve_plot").height(300.0).show(ui, |plot_ui| {
                plot_ui.points(Points::new(points).radius(1.5).name("Primes"));
            });
        }
    }

    fn ui_prime_gaps(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Prime Gaps").strong().size(16.0));

        ui.add(
            egui::Slider::new(&mut self.gaps_limit, 100..=1_000_000)
                .text("Limit")
                .logarithmic(true),
        );

        if ui.button("Compute Gaps").clicked() {
            let gaps = ix_number_theory::primes::prime_gaps(self.gaps_limit);
            self.gaps_result = Some(gaps);
        }

        if let Some(gaps) = &self.gaps_result {
            ui.label(format!("Number of gaps: {}", gaps.len()));
            if let Some(&max_gap) = gaps.iter().max() {
                ui.label(format!("Largest gap: {}", max_gap));
            }

            let points: PlotPoints = gaps
                .iter()
                .enumerate()
                .map(|(i, &g)| [i as f64, g as f64])
                .collect();

            Plot::new("gaps_plot").height(300.0).show(ui, |plot_ui| {
                plot_ui.points(Points::new(points).radius(1.5).name("Gap size"));
            });
        }
    }

    fn ui_modular(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("Modular Exponentiation")
                .strong()
                .size(16.0),
        );

        egui::Grid::new("mod_inputs").show(ui, |ui| {
            ui.label("Base:");
            ui.add(egui::DragValue::new(&mut self.mod_base));
            ui.end_row();

            ui.label("Exponent:");
            ui.add(egui::DragValue::new(&mut self.mod_exp));
            ui.end_row();

            ui.label("Modulus:");
            ui.add(egui::DragValue::new(&mut self.mod_modulus).range(1..=u64::MAX));
            ui.end_row();
        });

        if ui.button("Compute mod_pow").clicked() {
            let result =
                ix_number_theory::modular::mod_pow(self.mod_base, self.mod_exp, self.mod_modulus);
            self.mod_result = Some(result);
        }

        if let Some(result) = self.mod_result {
            ui.label(
                egui::RichText::new(format!(
                    "{}^{} mod {} = {}",
                    self.mod_base, self.mod_exp, self.mod_modulus, result
                ))
                .strong()
                .size(18.0),
            );
        }
    }

    fn ui_primality(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("Miller-Rabin Primality Test")
                .strong()
                .size(16.0),
        );

        ui.horizontal(|ui| {
            ui.label("Number:");
            ui.add(egui::DragValue::new(&mut self.primality_input).range(0..=u64::MAX));
        });

        if ui.button("Test Primality").clicked() {
            let result =
                ix_number_theory::primality::is_prime_miller_rabin(self.primality_input, 10);
            self.primality_result = Some(result);
        }

        if let Some(is_prime) = self.primality_result {
            let text = if is_prime {
                format!("{} is PRIME", self.primality_input)
            } else {
                format!("{} is COMPOSITE", self.primality_input)
            };
            let color = if is_prime {
                egui::Color32::GREEN
            } else {
                egui::Color32::RED
            };
            ui.label(egui::RichText::new(text).strong().size(18.0).color(color));
        }
    }
}
