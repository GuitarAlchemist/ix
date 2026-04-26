use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

pub struct EvolutionDemo {
    pop_size: usize,
    generations: usize,
    mutation_rate: f64,
    best_history: Vec<f64>,
    avg_history: Vec<f64>,
    status: String,
}

impl Default for EvolutionDemo {
    fn default() -> Self {
        Self {
            pop_size: 50,
            generations: 100,
            mutation_rate: 0.1,
            best_history: Vec::new(),
            avg_history: Vec::new(),
            status: "Evolving to maximize f(x,y) = -(x²+y²) (find the origin)".into(),
        }
    }
}

impl EvolutionDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Genetic Algorithm (ix-evolution)");
        ui.label("Maximize f(x,y) = -(x² + y²) — optimum at (0, 0)");

        ui.horizontal(|ui| {
            ui.label("Pop:");
            ui.add(egui::Slider::new(&mut self.pop_size, 10..=200));
            ui.label("Gens:");
            ui.add(egui::Slider::new(&mut self.generations, 10..=500));
            ui.label("Mutation:");
            ui.add(egui::Slider::new(&mut self.mutation_rate, 0.01..=0.5));
        });

        if ui.button("Evolve").clicked() {
            self.run();
        }

        ui.label(&self.status);

        if !self.best_history.is_empty() {
            Plot::new("evo_plot").height(350.0).show(ui, |plot_ui| {
                let best: PlotPoints = self
                    .best_history
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v])
                    .collect();
                let avg: PlotPoints = self
                    .avg_history
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v])
                    .collect();
                plot_ui.line(
                    Line::new(best)
                        .name("Best fitness")
                        .width(2.0)
                        .color(egui::Color32::GREEN),
                );
                plot_ui.line(
                    Line::new(avg)
                        .name("Avg fitness")
                        .width(1.5)
                        .color(egui::Color32::YELLOW),
                );
            });
        }
    }

    fn run(&mut self) {
        use rand::Rng;
        let mut rng = rand::rng();

        let fitness = |genes: &[f64]| -> f64 { -(genes[0] * genes[0] + genes[1] * genes[1]) };

        // Initialize population
        let mut pop: Vec<Vec<f64>> = (0..self.pop_size)
            .map(|_| vec![rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)])
            .collect();

        self.best_history.clear();
        self.avg_history.clear();

        for _ in 0..self.generations {
            let mut fits: Vec<(f64, usize)> = pop
                .iter()
                .enumerate()
                .map(|(i, g)| (fitness(g), i))
                .collect();
            fits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let best_fit = fits[0].0;
            let avg_fit = fits.iter().map(|f| f.0).sum::<f64>() / fits.len() as f64;
            self.best_history.push(best_fit);
            self.avg_history.push(avg_fit);

            // Selection: top 50%
            let survivors: Vec<Vec<f64>> = fits
                .iter()
                .take(self.pop_size / 2)
                .map(|&(_, i)| pop[i].clone())
                .collect();

            // Crossover + mutation
            let mut new_pop = survivors.clone();
            while new_pop.len() < self.pop_size {
                let p1 = &survivors[rng.random_range(0..survivors.len())];
                let p2 = &survivors[rng.random_range(0..survivors.len())];
                let mut child = vec![
                    if rng.random_bool(0.5) { p1[0] } else { p2[0] },
                    if rng.random_bool(0.5) { p1[1] } else { p2[1] },
                ];
                // Mutation
                for g in child.iter_mut() {
                    if rng.random_range(0.0..1.0) < self.mutation_rate {
                        *g += rng.random_range(-1.0..1.0);
                    }
                }
                new_pop.push(child);
            }
            pop = new_pop;
        }

        let best = pop
            .iter()
            .min_by(|a, b| {
                let fa = a[0] * a[0] + a[1] * a[1];
                let fb = b[0] * b[0] + b[1] * b[1];
                fa.partial_cmp(&fb).unwrap()
            })
            .unwrap();
        self.status = format!(
            "Best: ({:.4}, {:.4}), fitness: {:.6}",
            best[0],
            best[1],
            fitness(best)
        );
    }
}
