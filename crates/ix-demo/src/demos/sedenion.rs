use eframe::egui;

use ix_sedenion::cayley_dickson::double_multiply;
use ix_sedenion::octonion::Octonion;
use ix_sedenion::sedenion::Sedenion;

#[derive(PartialEq, Clone, Copy)]
enum SedenionMode {
    CayleyDickson,
    SedenionAlgebra,
    OctonionAssociativity,
}

pub struct SedenionDemo {
    mode: SedenionMode,
    // Cayley-Dickson chain
    cd_dimension: usize,
    cd_table: Vec<Vec<Vec<f64>>>,
    // Sedenion algebra
    sed_a: [f64; 4],
    sed_b: [f64; 4],
    sed_product: Option<[f64; 16]>,
    sed_conjugate: Option<[f64; 16]>,
    sed_norm: Option<f64>,
    // Octonion non-associativity
    oct_i: usize,
    oct_j: usize,
    oct_k: usize,
    oct_lhs: Option<[f64; 8]>,
    oct_rhs: Option<[f64; 8]>,
    oct_differ: Option<bool>,
}

impl Default for SedenionDemo {
    fn default() -> Self {
        Self {
            mode: SedenionMode::CayleyDickson,
            cd_dimension: 4,
            cd_table: Vec::new(),
            sed_a: [1.0, 0.0, 0.0, 0.0],
            sed_b: [0.0, 1.0, 0.0, 0.0],
            sed_product: None,
            sed_conjugate: None,
            sed_norm: None,
            oct_i: 1,
            oct_j: 2,
            oct_k: 4,
            oct_lhs: None,
            oct_rhs: None,
            oct_differ: None,
        }
    }
}

impl SedenionDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Sedenion Algebra (ix-sedenion)");

        ui.horizontal(|ui| {
            ui.radio_value(
                &mut self.mode,
                SedenionMode::CayleyDickson,
                "Cayley-Dickson Chain",
            );
            ui.radio_value(
                &mut self.mode,
                SedenionMode::SedenionAlgebra,
                "Sedenion Algebra",
            );
            ui.radio_value(
                &mut self.mode,
                SedenionMode::OctonionAssociativity,
                "Octonion Non-Associativity",
            );
        });

        match self.mode {
            SedenionMode::CayleyDickson => self.cayley_dickson_ui(ui),
            SedenionMode::SedenionAlgebra => self.sedenion_algebra_ui(ui),
            SedenionMode::OctonionAssociativity => self.octonion_assoc_ui(ui),
        }
    }

    fn cayley_dickson_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Multiplication tables for Cayley-Dickson algebras at various dimensions.");
        ui.horizontal(|ui| {
            ui.label("Dimension:");
            ui.radio_value(&mut self.cd_dimension, 2, "Complex (2)");
            ui.radio_value(&mut self.cd_dimension, 4, "Quaternion (4)");
            ui.radio_value(&mut self.cd_dimension, 8, "Octonion (8)");
            ui.radio_value(&mut self.cd_dimension, 16, "Sedenion (16)");
        });

        if ui.button("Compute Multiplication Table").clicked() {
            self.compute_cd_table();
        }

        if !self.cd_table.is_empty() {
            let dim = self.cd_table.len();
            ui.separator();
            ui.label(format!(
                "Basis product table (e_i * e_j) for dimension {}. Each cell shows the resulting basis element.",
                dim
            ));

            egui::ScrollArea::both().max_height(400.0).show(ui, |ui| {
                egui::Grid::new("cd_table")
                    .striped(true)
                    .min_col_width(50.0)
                    .show(ui, |ui| {
                        // Header row
                        ui.label("");
                        for j in 0..dim {
                            ui.label(format!("e{}", j));
                        }
                        ui.end_row();

                        for i in 0..dim {
                            ui.label(format!("e{}", i));
                            for j in 0..dim {
                                let product = &self.cd_table[i][j];
                                let label = format_basis_product(product);
                                ui.label(label);
                            }
                            ui.end_row();
                        }
                    });
            });
        }
    }

    fn compute_cd_table(&mut self) {
        let dim = self.cd_dimension;
        let mut table = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut row = Vec::with_capacity(dim);
            let mut ei = vec![0.0; dim];
            ei[i] = 1.0;
            for j in 0..dim {
                let mut ej = vec![0.0; dim];
                ej[j] = 1.0;
                let product = double_multiply(&ei, &ej);
                row.push(product);
            }
            table.push(row);
        }
        self.cd_table = table;
    }

    fn sedenion_algebra_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Input two sedenions (first 4 components, rest zero). Compute product, conjugate, and norm.");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Sedenion A:");
            for (idx, val) in self.sed_a.iter_mut().enumerate() {
                ui.add(
                    egui::DragValue::new(val)
                        .speed(0.1)
                        .prefix(format!("e{}=", idx)),
                );
            }
        });
        ui.horizontal(|ui| {
            ui.label("Sedenion B:");
            for (idx, val) in self.sed_b.iter_mut().enumerate() {
                ui.add(
                    egui::DragValue::new(val)
                        .speed(0.1)
                        .prefix(format!("e{}=", idx)),
                );
            }
        });

        if ui.button("Compute").clicked() {
            let mut ca = [0.0f64; 16];
            for (i, &v) in self.sed_a.iter().enumerate() {
                ca[i] = v;
            }
            let mut cb = [0.0f64; 16];
            for (i, &v) in self.sed_b.iter().enumerate() {
                cb[i] = v;
            }
            let sa = Sedenion::new(ca);
            let sb = Sedenion::new(cb);

            let product = Sedenion::mul(&sa, &sb);
            let conjugate = sa.conjugate();
            let norm = sa.norm();

            self.sed_product = Some(product.components);
            self.sed_conjugate = Some(conjugate.components);
            self.sed_norm = Some(norm);
        }

        if let Some(ref prod) = self.sed_product {
            ui.separator();
            ui.label("A * B =");
            ui.label(format_components(prod));

            if let Some(ref conj) = self.sed_conjugate {
                ui.label("conj(A) =");
                ui.label(format_components(conj));
            }
            if let Some(norm) = self.sed_norm {
                ui.label(format!("||A|| = {:.6}", norm));
            }
        }
    }

    fn octonion_assoc_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Demonstrate non-associativity: compare (e_i * e_j) * e_k vs e_i * (e_j * e_k).");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("i:");
            ui.add(egui::Slider::new(&mut self.oct_i, 0..=7));
            ui.label("j:");
            ui.add(egui::Slider::new(&mut self.oct_j, 0..=7));
            ui.label("k:");
            ui.add(egui::Slider::new(&mut self.oct_k, 0..=7));
        });

        if ui.button("Check Associativity").clicked() {
            let ei = Octonion::basis(self.oct_i);
            let ej = Octonion::basis(self.oct_j);
            let ek = Octonion::basis(self.oct_k);

            // (e_i * e_j) * e_k
            let ei_ej = Octonion::mul(&ei, &ej);
            let lhs = Octonion::mul(&ei_ej, &ek);

            // e_i * (e_j * e_k)
            let ej_ek = Octonion::mul(&ej, &ek);
            let rhs = Octonion::mul(&ei, &ej_ek);

            let differ = lhs
                .components
                .iter()
                .zip(rhs.components.iter())
                .any(|(a, b)| (a - b).abs() > 1e-10);

            self.oct_lhs = Some(lhs.components);
            self.oct_rhs = Some(rhs.components);
            self.oct_differ = Some(differ);
        }

        if let (Some(ref lhs), Some(ref rhs), Some(differ)) =
            (&self.oct_lhs, &self.oct_rhs, self.oct_differ)
        {
            ui.separator();
            ui.label(format!(
                "(e{} * e{}) * e{} = {}",
                self.oct_i,
                self.oct_j,
                self.oct_k,
                format_components_8(lhs)
            ));
            ui.label(format!(
                "e{} * (e{} * e{}) = {}",
                self.oct_i,
                self.oct_j,
                self.oct_k,
                format_components_8(rhs)
            ));

            if differ {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 100, 100),
                    "NON-ASSOCIATIVE: results differ!",
                );
            } else {
                ui.colored_label(
                    egui::Color32::from_rgb(100, 255, 100),
                    "Associative for this triple (results match).",
                );
            }
        }
    }
}

/// Format a basis product vector as a readable basis element string.
/// E.g. [0, 0, -1, 0] -> "-e2"
fn format_basis_product(components: &[f64]) -> String {
    let eps = 1e-10;
    let mut parts = Vec::new();
    for (idx, &val) in components.iter().enumerate() {
        if val.abs() > eps {
            if (val - 1.0).abs() < eps {
                if idx == 0 {
                    parts.push("1".to_string());
                } else {
                    parts.push(format!("e{}", idx));
                }
            } else if (val + 1.0).abs() < eps {
                if idx == 0 {
                    parts.push("-1".to_string());
                } else {
                    parts.push(format!("-e{}", idx));
                }
            } else {
                parts.push(format!("{:.1}e{}", val, idx));
            }
        }
    }
    if parts.is_empty() {
        "0".to_string()
    } else {
        parts.join("+")
    }
}

/// Format 16-component sedenion for display.
fn format_components(c: &[f64; 16]) -> String {
    let eps = 1e-10;
    let mut parts = Vec::new();
    for (i, &v) in c.iter().enumerate() {
        if v.abs() > eps {
            if i == 0 {
                parts.push(format!("{:.4}", v));
            } else {
                parts.push(format!("{:.4}e{}", v, i));
            }
        }
    }
    if parts.is_empty() {
        "0".to_string()
    } else {
        parts.join(" + ")
    }
}

/// Format 8-component octonion for display.
fn format_components_8(c: &[f64; 8]) -> String {
    let eps = 1e-10;
    let mut parts = Vec::new();
    for (i, &v) in c.iter().enumerate() {
        if v.abs() > eps {
            if i == 0 {
                parts.push(format!("{:.4}", v));
            } else {
                parts.push(format!("{:.4}e{}", v, i));
            }
        }
    }
    if parts.is_empty() {
        "0".to_string()
    } else {
        parts.join(" + ")
    }
}
