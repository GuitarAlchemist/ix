use eframe::egui;
use egui_plot::{Plot, Line, PlotPoints, Bar, BarChart};

#[derive(PartialEq, Clone, Copy)]
enum BanditStrategy {
    EpsilonGreedy,
    Ucb1,
    Thompson,
}

pub struct ReinforcementDemo {
    n_arms: usize,
    n_rounds: usize,
    strategy: BanditStrategy,
    epsilon: f64,
    reward_history: Vec<f64>,
    arm_counts: Vec<f64>,
    arm_rewards: Vec<f64>,
    true_means: Vec<f64>,
    status: String,
}

impl Default for ReinforcementDemo {
    fn default() -> Self {
        Self {
            n_arms: 5,
            n_rounds: 1000,
            strategy: BanditStrategy::Ucb1,
            epsilon: 0.1,
            reward_history: Vec::new(),
            arm_counts: Vec::new(),
            arm_rewards: Vec::new(),
            true_means: Vec::new(),
            status: "Ready".into(),
        }
    }
}

impl ReinforcementDemo {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("RL Bandits (machin-rl)");
        ui.label("Multi-armed bandit — explore vs exploit.");

        ui.horizontal(|ui| {
            ui.label("Arms:"); ui.add(egui::Slider::new(&mut self.n_arms, 2..=10));
            ui.label("Rounds:"); ui.add(egui::Slider::new(&mut self.n_rounds, 100..=5000));
        });
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.strategy, BanditStrategy::EpsilonGreedy, "ε-Greedy");
            ui.radio_value(&mut self.strategy, BanditStrategy::Ucb1, "UCB1");
            ui.radio_value(&mut self.strategy, BanditStrategy::Thompson, "Thompson");
        });
        if self.strategy == BanditStrategy::EpsilonGreedy {
            ui.add(egui::Slider::new(&mut self.epsilon, 0.01..=0.5).text("ε"));
        }

        if ui.button("Run").clicked() {
            self.run();
        }

        ui.label(&self.status);

        if !self.reward_history.is_empty() {
            // Cumulative avg reward
            Plot::new("bandit_reward").height(250.0).show(ui, |plot_ui| {
                let mut cum = 0.0;
                let pts: PlotPoints = self.reward_history.iter().enumerate()
                    .map(|(i, &r)| {
                        cum += r;
                        [i as f64, cum / (i + 1) as f64]
                    }).collect();
                plot_ui.line(Line::new(pts).name("Avg Reward").width(2.0));
            });

            // Arm pull distribution
            ui.label("Arm pull counts:");
            Plot::new("bandit_arms").height(200.0).show(ui, |plot_ui| {
                let bars: Vec<Bar> = self.arm_counts.iter().enumerate()
                    .map(|(i, &c)| Bar::new(i as f64, c))
                    .collect();
                plot_ui.bar_chart(BarChart::new(bars).name("Pulls"));
            });
        }
    }

    fn run(&mut self) {
        use rand::Rng;
        let mut rng = rand::rng();

        // Random true arm means
        self.true_means = (0..self.n_arms)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();

        let mut counts = vec![0.0f64; self.n_arms];
        let mut values = vec![0.0f64; self.n_arms];
        // Thompson: beta params
        let mut alpha = vec![1.0f64; self.n_arms];
        let mut beta_param = vec![1.0f64; self.n_arms];

        self.reward_history.clear();

        for t in 0..self.n_rounds {
            let arm = match self.strategy {
                BanditStrategy::EpsilonGreedy => {
                    if rng.random_range(0.0..1.0) < self.epsilon {
                        rng.random_range(0..self.n_arms)
                    } else {
                        values.iter().enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(i, _)| i).unwrap_or(0)
                    }
                }
                BanditStrategy::Ucb1 => {
                    if (t as f64) < self.n_arms as f64 {
                        t % self.n_arms
                    } else {
                        let total = counts.iter().sum::<f64>();
                        values.iter().enumerate()
                            .map(|(i, &v)| (i, v + (2.0 * total.ln() / counts[i]).sqrt()))
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .map(|(i, _)| i).unwrap_or(0)
                    }
                }
                BanditStrategy::Thompson => {
                    // Sample from Beta(alpha, beta) — approximate with simple formula
                    let samples: Vec<(usize, f64)> = (0..self.n_arms).map(|i| {
                        let a = alpha[i];
                        let b = beta_param[i];
                        // Simple beta approximation: mean + noise scaled by variance
                        let mean = a / (a + b);
                        let var = (a * b) / ((a + b).powi(2) * (a + b + 1.0));
                        (i, mean + rng.random_range(-1.0..1.0) * var.sqrt() * 3.0)
                    }).collect();
                    samples.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(i, _)| *i).unwrap_or(0)
                }
            };

            // Pull arm
            let reward = if rng.random_range(0.0..1.0) < self.true_means[arm] { 1.0 } else { 0.0 };
            counts[arm] += 1.0;
            values[arm] += (reward - values[arm]) / counts[arm];
            alpha[arm] += reward;
            beta_param[arm] += 1.0 - reward;

            self.reward_history.push(reward);
        }

        self.arm_counts = counts;
        self.arm_rewards = values;

        let best_arm = self.true_means.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let most_pulled = self.arm_counts.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        self.status = format!("Best arm: {} (μ={:.3}), Most pulled: {} ({:.0} times)",
            best_arm, self.true_means[best_arm], most_pulled, self.arm_counts[most_pulled]);
    }
}
