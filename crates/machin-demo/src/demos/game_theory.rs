use eframe::egui;
use egui_plot::{Plot, Line, PlotPoints};

pub struct GameTheoryDemo {
    rounds: usize,
    history: Vec<[f64; 2]>,
    p1_score: f64,
    p2_score: f64,
    status: String,
}

impl Default for GameTheoryDemo {
    fn default() -> Self {
        Self {
            rounds: 200,
            history: Vec::new(),
            p1_score: 0.0,
            p2_score: 0.0,
            status: "Prisoner's Dilemma — Tit-for-Tat vs Always Defect".into(),
        }
    }
}

impl GameTheoryDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Game Theory (machin-game)");
        ui.label("Iterated Prisoner's Dilemma: Tit-for-Tat (P1) vs Always Defect (P2)");

        ui.add(egui::Slider::new(&mut self.rounds, 10..=500).text("Rounds"));

        if ui.button("Simulate").clicked() {
            self.run();
        }

        ui.label(&self.status);

        if !self.history.is_empty() {
            Plot::new("game_plot").height(350.0).show(ui, |plot_ui| {
                let p1: PlotPoints = self.history.iter().enumerate()
                    .map(|(i, h)| [i as f64, h[0]]).collect();
                let p2: PlotPoints = self.history.iter().enumerate()
                    .map(|(i, h)| [i as f64, h[1]]).collect();
                plot_ui.line(Line::new(p1).name("Tit-for-Tat (cumulative)").width(2.0)
                    .color(egui::Color32::from_rgb(100, 200, 100)));
                plot_ui.line(Line::new(p2).name("Always Defect (cumulative)").width(2.0)
                    .color(egui::Color32::from_rgb(200, 100, 100)));
            });
        }
    }

    fn run(&mut self) {
        // Payoff matrix: (cooperate=true, defect=false)
        // Both cooperate: (3,3), Both defect: (1,1)
        // C vs D: (0,5), D vs C: (5,0)
        let payoff = |a: bool, b: bool| -> (f64, f64) {
            match (a, b) {
                (true, true) => (3.0, 3.0),
                (true, false) => (0.0, 5.0),
                (false, true) => (5.0, 0.0),
                (false, false) => (1.0, 1.0),
            }
        };

        self.history.clear();
        let mut cum1 = 0.0;
        let mut cum2 = 0.0;
        let mut last_p2 = true; // Tit-for-tat starts cooperative

        for _ in 0..self.rounds {
            let p1_action = last_p2; // Tit-for-tat: copy opponent's last move
            let p2_action = false;   // Always defect

            let (r1, r2) = payoff(p1_action, p2_action);
            cum1 += r1;
            cum2 += r2;
            last_p2 = p2_action;
            self.history.push([cum1, cum2]);
        }

        self.p1_score = cum1;
        self.p2_score = cum2;
        self.status = format!("Tit-for-Tat: {:.0} | Always Defect: {:.0} | {} wins",
            cum1, cum2, if cum1 > cum2 { "TfT" } else { "AD" });
    }
}
