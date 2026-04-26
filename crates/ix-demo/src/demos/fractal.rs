use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};

#[derive(PartialEq, Clone, Copy)]
enum FractalMode {
    Takagi,
    Ifs,
    LSystem,
    SpaceFilling,
}

#[derive(PartialEq, Clone, Copy)]
enum IfsPreset {
    Sierpinski,
    BarnsleyFern,
    Koch,
}

#[derive(PartialEq, Clone, Copy)]
enum LSystemPreset {
    Dragon,
    Sierpinski,
    Koch,
}

#[derive(PartialEq, Clone, Copy)]
enum SpaceFillingPreset {
    Hilbert,
    Peano,
}

pub struct FractalDemo {
    mode: FractalMode,
    // Takagi
    takagi_terms: usize,
    // IFS
    ifs_preset: IfsPreset,
    ifs_iterations: usize,
    // L-System
    lsystem_preset: LSystemPreset,
    lsystem_iterations: usize,
    // Space-filling
    space_preset: SpaceFillingPreset,
    space_order: u32,
    // Shared plot data
    plot_points: Vec<[f64; 2]>,
}

impl Default for FractalDemo {
    fn default() -> Self {
        Self {
            mode: FractalMode::Takagi,
            takagi_terms: 20,
            ifs_preset: IfsPreset::Sierpinski,
            ifs_iterations: 10_000,
            lsystem_preset: LSystemPreset::Dragon,
            lsystem_iterations: 5,
            space_preset: SpaceFillingPreset::Hilbert,
            space_order: 3,
            plot_points: Vec::new(),
        }
    }
}

impl FractalDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Fractals (ix-fractal)");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, FractalMode::Takagi, "Takagi Curve");
            ui.radio_value(&mut self.mode, FractalMode::Ifs, "IFS Chaos Game");
            ui.radio_value(&mut self.mode, FractalMode::LSystem, "L-System");
            ui.radio_value(&mut self.mode, FractalMode::SpaceFilling, "Space-Filling");
        });

        match self.mode {
            FractalMode::Takagi => self.takagi_ui(ui),
            FractalMode::Ifs => self.ifs_ui(ui),
            FractalMode::LSystem => self.lsystem_ui(ui),
            FractalMode::SpaceFilling => self.space_filling_ui(ui),
        }
    }

    fn takagi_ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.takagi_terms, 1..=50).text("Terms"));

        if ui.button("Generate Takagi Curve").clicked() {
            self.gen_takagi();
        }

        if !self.plot_points.is_empty() {
            Plot::new("takagi").height(500.0).show(ui, |plot_ui| {
                let pts: PlotPoints = self.plot_points.iter().copied().collect();
                plot_ui.line(Line::new(pts).name("Takagi curve").width(1.5));
            });
        }
    }

    fn ifs_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.ifs_preset, IfsPreset::Sierpinski, "Sierpinski");
            ui.radio_value(
                &mut self.ifs_preset,
                IfsPreset::BarnsleyFern,
                "Barnsley Fern",
            );
            ui.radio_value(&mut self.ifs_preset, IfsPreset::Koch, "Koch");
        });
        ui.add(
            egui::Slider::new(&mut self.ifs_iterations, 1000..=100_000)
                .text("Iterations")
                .logarithmic(true),
        );

        if ui.button("Run Chaos Game").clicked() {
            self.gen_ifs();
        }

        if !self.plot_points.is_empty() {
            Plot::new("ifs")
                .height(500.0)
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    let pts: PlotPoints = self.plot_points.iter().copied().collect();
                    plot_ui.points(
                        Points::new(pts)
                            .radius(0.5)
                            .color(egui::Color32::LIGHT_GREEN),
                    );
                });
        }
    }

    fn lsystem_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.lsystem_preset, LSystemPreset::Dragon, "Dragon");
            ui.radio_value(
                &mut self.lsystem_preset,
                LSystemPreset::Sierpinski,
                "Sierpinski",
            );
            ui.radio_value(&mut self.lsystem_preset, LSystemPreset::Koch, "Koch");
        });
        ui.add(egui::Slider::new(&mut self.lsystem_iterations, 1..=8).text("Iterations"));

        if ui.button("Generate L-System").clicked() {
            self.gen_lsystem();
        }

        if !self.plot_points.is_empty() {
            Plot::new("lsystem")
                .height(500.0)
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    let pts: PlotPoints = self.plot_points.iter().copied().collect();
                    plot_ui.line(Line::new(pts).name("L-System path").width(1.0));
                });
        }
    }

    fn space_filling_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.radio_value(
                &mut self.space_preset,
                SpaceFillingPreset::Hilbert,
                "Hilbert",
            );
            ui.radio_value(&mut self.space_preset, SpaceFillingPreset::Peano, "Peano");
        });
        ui.add(egui::Slider::new(&mut self.space_order, 1..=6).text("Order"));

        if ui.button("Generate Curve").clicked() {
            self.gen_space_filling();
        }

        if !self.plot_points.is_empty() {
            Plot::new("space_filling")
                .height(500.0)
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    let pts: PlotPoints = self.plot_points.iter().copied().collect();
                    plot_ui.line(Line::new(pts).name("Space-filling curve").width(1.0));
                });
        }
    }

    fn gen_takagi(&mut self) {
        let n_points = 501;
        let curve = ix_fractal::takagi::takagi_series(n_points, self.takagi_terms);
        self.plot_points.clear();
        let step = 1.0 / (n_points - 1) as f64;
        for (i, &y) in curve.iter().enumerate() {
            self.plot_points.push([i as f64 * step, y]);
        }
    }

    fn gen_ifs(&mut self) {
        use rand::SeedableRng;
        let maps = match self.ifs_preset {
            IfsPreset::Sierpinski => ix_fractal::ifs::sierpinski_maps(),
            IfsPreset::BarnsleyFern => ix_fractal::ifs::barnsley_fern_maps(),
            IfsPreset::Koch => ix_fractal::ifs::koch_snowflake_maps(),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        self.plot_points = ix_fractal::ifs::ifs_iterate(&maps, self.ifs_iterations, &mut rng);
    }

    fn gen_lsystem(&mut self) {
        let (lsys, angle) = match self.lsystem_preset {
            LSystemPreset::Dragon => (ix_fractal::lsystem::dragon_curve(), 90.0),
            LSystemPreset::Sierpinski => (ix_fractal::lsystem::sierpinski_arrowhead(), 60.0),
            LSystemPreset::Koch => (ix_fractal::lsystem::koch_curve(), 60.0),
        };
        let expanded = lsys.expand(self.lsystem_iterations);
        self.plot_points = ix_fractal::lsystem::interpret(&expanded, angle, 1.0);
    }

    fn gen_space_filling(&mut self) {
        self.plot_points = match self.space_preset {
            SpaceFillingPreset::Hilbert => {
                ix_fractal::space_filling::hilbert_curve(self.space_order)
            }
            SpaceFillingPreset::Peano => ix_fractal::space_filling::peano_curve(self.space_order),
        };
    }
}
