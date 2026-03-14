//! MachinDeOuf Interactive Demo
//!
//! A tabbed egui application showcasing ML/math algorithms from
//! every crate in the workspace.

mod demos;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("MachinDeOuf — ML Algorithm Explorer"),
        ..Default::default()
    };
    eframe::run_native(
        "MachinDeOuf Demo",
        options,
        Box::new(|_cc| Ok(Box::new(MachinApp::default()))),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Stats,
    Regression,
    Clustering,
    NeuralNet,
    Optimization,
    Chaos,
    Signal,
    IKChain,
    Evolution,
    Reinforcement,
    Search,
    GameTheory,
    Probabilistic,
    GpuKernels,
    Transformer,
}

impl Tab {
    const ALL: &[Tab] = &[
        Tab::Stats,
        Tab::Regression,
        Tab::Clustering,
        Tab::NeuralNet,
        Tab::Optimization,
        Tab::Chaos,
        Tab::Signal,
        Tab::IKChain,
        Tab::Evolution,
        Tab::Reinforcement,
        Tab::Search,
        Tab::GameTheory,
        Tab::Probabilistic,
        Tab::GpuKernels,
        Tab::Transformer,
    ];

    fn label(self) -> &'static str {
        match self {
            Tab::Stats => "Stats",
            Tab::Regression => "Regression",
            Tab::Clustering => "Clustering",
            Tab::NeuralNet => "Neural Net",
            Tab::Optimization => "Optimization",
            Tab::Chaos => "Chaos",
            Tab::Signal => "Signal",
            Tab::IKChain => "IK Chain",
            Tab::Evolution => "Evolution",
            Tab::Reinforcement => "RL Bandits",
            Tab::Search => "Search",
            Tab::GameTheory => "Game Theory",
            Tab::Probabilistic => "Probabilistic",
            Tab::GpuKernels => "GPU Kernels",
            Tab::Transformer => "Transformer",
        }
    }
}

struct MachinApp {
    active_tab: Tab,
    stats_demo: demos::stats::StatsDemo,
    regression_demo: demos::regression::RegressionDemo,
    clustering_demo: demos::clustering::ClusteringDemo,
    nn_demo: demos::neural_net::NeuralNetDemo,
    optim_demo: demos::optimization::OptimizationDemo,
    chaos_demo: demos::chaos::ChaosDemo,
    signal_demo: demos::signal::SignalDemo,
    ik_demo: demos::ik_chain::IKChainDemo,
    evolution_demo: demos::evolution::EvolutionDemo,
    rl_demo: demos::reinforcement::ReinforcementDemo,
    search_demo: demos::search::SearchDemo,
    game_demo: demos::game_theory::GameTheoryDemo,
    prob_demo: demos::probabilistic::ProbabilisticDemo,
    gpu_demo: demos::gpu_kernels::GpuKernelsDemo,
    transformer_demo: demos::transformer::TransformerDemo,
}

impl Default for MachinApp {
    fn default() -> Self {
        Self {
            active_tab: Tab::Stats,
            stats_demo: demos::stats::StatsDemo::default(),
            regression_demo: demos::regression::RegressionDemo::default(),
            clustering_demo: demos::clustering::ClusteringDemo::default(),
            nn_demo: demos::neural_net::NeuralNetDemo::default(),
            optim_demo: demos::optimization::OptimizationDemo::default(),
            chaos_demo: demos::chaos::ChaosDemo::default(),
            signal_demo: demos::signal::SignalDemo::default(),
            ik_demo: demos::ik_chain::IKChainDemo::default(),
            evolution_demo: demos::evolution::EvolutionDemo::default(),
            rl_demo: demos::reinforcement::ReinforcementDemo::default(),
            search_demo: demos::search::SearchDemo::default(),
            game_demo: demos::game_theory::GameTheoryDemo::default(),
            prob_demo: demos::probabilistic::ProbabilisticDemo::default(),
            gpu_demo: demos::gpu_kernels::GpuKernelsDemo::default(),
            transformer_demo: demos::transformer::TransformerDemo::default(),
        }
    }
}

impl eframe::App for MachinApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("MachinDeOuf");
                ui.separator();
                for &tab in Tab::ALL {
                    ui.selectable_value(&mut self.active_tab, tab, tab.label());
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().show(ui, |ui| {
                match self.active_tab {
                    Tab::Stats => self.stats_demo.ui(ui),
                    Tab::Regression => self.regression_demo.ui(ui),
                    Tab::Clustering => self.clustering_demo.ui(ui),
                    Tab::NeuralNet => self.nn_demo.ui(ui),
                    Tab::Optimization => self.optim_demo.ui(ui),
                    Tab::Chaos => self.chaos_demo.ui(ui),
                    Tab::Signal => self.signal_demo.ui(ui),
                    Tab::IKChain => self.ik_demo.ui(ui),
                    Tab::Evolution => self.evolution_demo.ui(ui),
                    Tab::Reinforcement => self.rl_demo.ui(ui),
                    Tab::Search => self.search_demo.ui(ui),
                    Tab::GameTheory => self.game_demo.ui(ui),
                    Tab::Probabilistic => self.prob_demo.ui(ui),
                    Tab::GpuKernels => self.gpu_demo.ui(ui),
                    Tab::Transformer => self.transformer_demo.ui(ui),
                }
            });
        });
    }
}
