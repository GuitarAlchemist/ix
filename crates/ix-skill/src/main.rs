//! ix - Claude Code ML skill CLI.
//!
//! Exposes ix algorithms as CLI commands for use as Claude Code skills.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ix", version, about = "ML algorithms for Claude Code skills")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Optimization algorithms
    Optimize {
        /// Algorithm: sgd, adam, annealing, pso, genetic, differential
        #[arg(long)]
        algo: String,

        /// Benchmark function: sphere, rosenbrock, rastrigin
        #[arg(long, default_value = "sphere")]
        function: String,

        /// Number of dimensions
        #[arg(long, default_value = "2")]
        dim: usize,

        /// Maximum iterations
        #[arg(long, default_value = "1000")]
        max_iter: usize,
    },

    /// Supervised learning
    Train {
        /// Model: linear, logistic, knn, naive-bayes
        #[arg(long)]
        model: String,

        /// Path to CSV data file
        #[arg(long)]
        data: Option<String>,
    },

    /// Clustering
    Cluster {
        /// Algorithm: kmeans, dbscan
        #[arg(long)]
        algo: String,

        /// Number of clusters (for kmeans)
        #[arg(long, default_value = "3")]
        k: usize,
    },

    /// Probabilistic grammar operations
    Grammar {
        #[command(subcommand)]
        command: GrammarCommands,
    },

    /// Information about available algorithms
    List,
}

#[derive(Subcommand)]
enum GrammarCommands {
    /// Bayesian weight update and softmax query over grammar rules
    Weights {
        /// Path to JSON file with WeightedRule array
        #[arg(long)]
        rules: String,

        /// Apply an observation: rule_id:success|failure  (e.g. r1:success)
        #[arg(long)]
        observe: Option<String>,

        /// Softmax temperature
        #[arg(long, default_value = "1.0")]
        temperature: f64,
    },

    /// Replicator dynamics simulation over grammar species
    Evolve {
        /// Path to JSON file with GrammarSpecies array
        #[arg(long)]
        species: String,

        /// Number of simulation steps
        #[arg(long, default_value = "100")]
        steps: usize,

        /// Time step dt
        #[arg(long, default_value = "0.05")]
        dt: f64,

        /// Prune threshold
        #[arg(long, default_value = "0.000001")]
        prune_threshold: f64,
    },

    /// Grammar-guided MCTS derivation search
    Search {
        /// Path to EBNF grammar file
        #[arg(long)]
        grammar: String,

        /// MCTS iterations
        #[arg(long, default_value = "500")]
        iterations: usize,

        /// UCB1 exploration constant
        #[arg(long, default_value = "1.41")]
        exploration: f64,

        /// Max derivation depth
        #[arg(long, default_value = "20")]
        max_depth: usize,

        /// RNG seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Optimize { algo, function, dim, max_iter } => {
            println!("Running {} on {} (dim={}, max_iter={})", algo, function, dim, max_iter);
            run_optimize(&algo, &function, dim, max_iter);
        }
        Commands::Train { model, data } => {
            run_train(&model, data.as_deref());
        }
        Commands::Cluster { algo, k } => {
            run_cluster(&algo, k);
        }
        Commands::Grammar { command } => {
            run_grammar(command);
        }
        Commands::List => {
            print_algorithms();
        }
    }
}

fn run_train(model: &str, data: Option<&str>) {
    use ndarray::array;

    // Demo data if no file provided
    let (x, y) = if let Some(path) = data {
        let content = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => { eprintln!("Failed to read {}: {}", path, e); return; }
        };
        // Simple CSV: last column is target
        let rows: Vec<Vec<f64>> = content.lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .filter_map(|l| l.split(',').map(|v| v.trim().parse::<f64>().ok()).collect::<Option<Vec<_>>>())
            .collect();
        if rows.is_empty() { eprintln!("No valid data rows"); return; }
        let ncols = rows[0].len();
        if ncols < 2 { eprintln!("Need at least 2 columns"); return; }
        let x_data: Vec<f64> = rows.iter().flat_map(|r| &r[..ncols-1]).copied().collect();
        let y_data: Vec<f64> = rows.iter().map(|r| r[ncols-1]).collect();
        let n = rows.len();
        (ndarray::Array2::from_shape_vec((n, ncols-1), x_data).unwrap(),
         ndarray::Array1::from_vec(y_data))
    } else {
        println!("  No data file — using demo dataset");
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 7.0], [5.0, 9.0]];
        let y = ndarray::Array1::from_vec(vec![3.0, 5.0, 8.0, 11.0, 14.0]);
        (x, y)
    };

    match model {
        "linear" => {
            use ix_supervised::linear_regression::LinearRegression;
            use ix_supervised::traits::Regressor;
            let mut lr = LinearRegression::new();
            lr.fit(&x, &y);
            let pred = lr.predict(&x);
            println!("  Linear Regression trained on {} samples", x.nrows());
            println!("  Predictions: {:?}", pred.to_vec());
        }
        "logistic" => {
            use ix_supervised::logistic_regression::LogisticRegression;
            use ix_supervised::traits::Classifier;
            let mean_y = y.mean().unwrap_or(0.0);
            let labels = y.mapv(|v| if v > mean_y { 1usize } else { 0 });
            let mut lg = LogisticRegression::new();
            lg.fit(&x, &labels);
            let pred = lg.predict(&x);
            println!("  Logistic Regression trained on {} samples", x.nrows());
            println!("  Predictions: {:?}", pred.to_vec());
        }
        "knn" => {
            use ix_supervised::knn::KNN;
            use ix_supervised::traits::Classifier;
            let mean_y = y.mean().unwrap_or(0.0);
            let labels = y.mapv(|v| if v > mean_y { 1usize } else { 0 });
            let mut knn = KNN::new(3);
            knn.fit(&x, &labels);
            let pred = knn.predict(&x);
            println!("  KNN (k=3) trained on {} samples", x.nrows());
            println!("  Predictions: {:?}", pred.to_vec());
        }
        "naive-bayes" => {
            use ix_supervised::naive_bayes::GaussianNaiveBayes;
            use ix_supervised::traits::Classifier;
            let mean_y = y.mean().unwrap_or(0.0);
            let labels = y.mapv(|v| if v > mean_y { 1usize } else { 0 });
            let mut nb = GaussianNaiveBayes::new();
            nb.fit(&x, &labels);
            let pred = nb.predict(&x);
            println!("  Gaussian Naive Bayes trained on {} samples", x.nrows());
            println!("  Predictions: {:?}", pred.to_vec());
        }
        "decision-tree" => {
            use ix_supervised::decision_tree::DecisionTree;
            use ix_supervised::traits::Classifier;
            let mean_y = y.mean().unwrap_or(0.0);
            let labels = y.mapv(|v| if v > mean_y { 1usize } else { 0 });
            let mut dt = DecisionTree::new(5);
            dt.fit(&x, &labels);
            let pred = dt.predict(&x);
            println!("  Decision Tree trained on {} samples", x.nrows());
            println!("  Predictions: {:?}", pred.to_vec());
        }
        "svm" => {
            use ix_supervised::svm::LinearSVM;
            use ix_supervised::traits::Classifier;
            let mean_y = y.mean().unwrap_or(0.0);
            let labels = y.mapv(|v| if v > mean_y { 1usize } else { 0 });
            let mut svm = LinearSVM::new(1.0);
            svm.fit(&x, &labels);
            let pred = svm.predict(&x);
            println!("  Linear SVM trained on {} samples", x.nrows());
            println!("  Predictions: {:?}", pred.to_vec());
        }
        _ => {
            eprintln!("Unknown model: {}. Use: linear, logistic, knn, naive-bayes, decision-tree, svm", model);
        }
    }
}

fn run_cluster(algo: &str, k: usize) {
    use ndarray::array;
    use ix_unsupervised::traits::Clusterer;

    // Demo data
    let x = array![
        [0.0, 0.0], [0.5, 0.5], [1.0, 0.0],
        [10.0, 10.0], [10.5, 10.5], [11.0, 10.0],
        [5.0, 20.0], [5.5, 20.5], [6.0, 20.0]
    ];

    match algo {
        "kmeans" => {
            let mut km = ix_unsupervised::kmeans::KMeans::new(k);
            let labels = km.fit_predict(&x);
            println!("  K-Means (k={}) on {} samples", k, x.nrows());
            println!("  Labels: {:?}", labels.to_vec());
        }
        "dbscan" => {
            let mut db = ix_unsupervised::dbscan::DBSCAN::new(2.0, 2);
            let labels = db.fit_predict(&x);
            println!("  DBSCAN (eps=2.0, min_pts=2) on {} samples", x.nrows());
            println!("  Labels: {:?}", labels.to_vec());
        }
        "pca" => {
            use ix_unsupervised::traits::DimensionReducer;
            let mut pca = ix_unsupervised::pca::PCA::new(1);
            let reduced = pca.fit_transform(&x);
            println!("  PCA (1 component) on {} samples", x.nrows());
            println!("  Reduced shape: ({}, {})", reduced.nrows(), reduced.ncols());
            for i in 0..reduced.nrows() {
                println!("    [{:.4}]", reduced[[i, 0]]);
            }
        }
        "gmm" => {
            let mut gmm = ix_unsupervised::gmm::GMM::new(k);
            let labels = gmm.fit_predict(&x);
            println!("  GMM (k={}) on {} samples", k, x.nrows());
            println!("  Labels: {:?}", labels.to_vec());
        }
        _ => {
            eprintln!("Unknown algorithm: {}. Use: kmeans, dbscan, pca, gmm", algo);
        }
    }
}

fn run_grammar(cmd: GrammarCommands) {
    match cmd {
        GrammarCommands::Weights { rules, observe, temperature } => {
            let json_str = match std::fs::read_to_string(&rules) {
                Ok(s) => s,
                Err(e) => { eprintln!("Failed to read {}: {}", rules, e); return; }
            };
            let mut rule_list: Vec<ix_grammar::weighted::WeightedRule> =
                match serde_json::from_str(&json_str) {
                    Ok(r) => r,
                    Err(e) => { eprintln!("Failed to parse rules: {}", e); return; }
                };

            if let Some(obs) = observe {
                let parts: Vec<&str> = obs.splitn(2, ':').collect();
                if parts.len() == 2 {
                    let rule_id = parts[0];
                    let success = parts[1].eq_ignore_ascii_case("success");
                    for rule in &mut rule_list {
                        if rule.id == rule_id {
                            *rule = ix_grammar::weighted::bayesian_update(rule, success);
                            break;
                        }
                    }
                }
            }

            let probs = ix_grammar::weighted::softmax(&rule_list, temperature);
            println!("Updated rules:");
            for r in &rule_list {
                println!("  {} α={:.3} β={:.3} w={:.4}", r.id, r.alpha, r.beta, r.weight);
            }
            println!("\nSoftmax (temperature={}):", temperature);
            for (id, p) in &probs {
                println!("  {} → {:.4}", id, p);
            }
        }

        GrammarCommands::Evolve { species, steps, dt, prune_threshold } => {
            let json_str = match std::fs::read_to_string(&species) {
                Ok(s) => s,
                Err(e) => { eprintln!("Failed to read {}: {}", species, e); return; }
            };
            let initial: Vec<ix_grammar::replicator::GrammarSpecies> =
                match serde_json::from_str(&json_str) {
                    Ok(s) => s,
                    Err(e) => { eprintln!("Failed to parse species: {}", e); return; }
                };

            let result = ix_grammar::replicator::simulate(&initial, steps, dt, prune_threshold);

            println!("Final species (after {} steps):", steps);
            for s in &result.final_species {
                println!("  {} proportion={:.4} fitness={:.4} stable={}", s.id, s.proportion, s.fitness, s.is_stable);
            }
            if !result.ess.is_empty() {
                println!("\nEvolutionarily Stable Strategies:");
                for s in &result.ess {
                    println!("  {} (proportion={:.4})", s.id, s.proportion);
                }
            }
        }

        GrammarCommands::Search { grammar, iterations, exploration, max_depth, seed } => {
            let grammar_str = match std::fs::read_to_string(&grammar) {
                Ok(s) => s,
                Err(e) => { eprintln!("Failed to read {}: {}", grammar, e); return; }
            };
            let g = match ix_grammar::constrained::EbnfGrammar::from_str(&grammar_str) {
                Ok(g) => g,
                Err(e) => { eprintln!("Failed to parse grammar: {}", e); return; }
            };

            let result = ix_grammar::constrained::search_derivation(
                g, iterations, exploration, max_depth, seed,
            );

            println!("Best derivation (reward={:.4}):", result.reward);
            for (nt, alt) in &result.best_derivation {
                println!("  {} → {}", nt, alt.join(" "));
            }
        }
    }
}

fn run_optimize(algo: &str, function: &str, dim: usize, max_iter: usize) {
    use ix_optimize::traits::ClosureObjective;
    use ix_math::ndarray::Array1;

    #[allow(clippy::type_complexity)]
    let obj: ClosureObjective<Box<dyn Fn(&Array1<f64>) -> f64>> = match function {
        "sphere" => ClosureObjective {
            f: Box::new(|x: &Array1<f64>| x.mapv(|v| v * v).sum()),
            dimensions: dim,
        },
        "rosenbrock" => ClosureObjective {
            f: Box::new(|x: &Array1<f64>| {
                (0..x.len() - 1)
                    .map(|i| {
                        let xi: f64 = x[i];
                        let xi1: f64 = x[i + 1];
                        100.0 * (xi1 - xi.powi(2)).powi(2) + (1.0 - xi).powi(2)
                    })
                    .sum::<f64>()
            }),
            dimensions: dim,
        },
        "rastrigin" => ClosureObjective {
            f: Box::new(|x: &Array1<f64>| {
                let n = x.len() as f64;
                10.0 * n + x.iter().map(|&xi: &f64| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
            }),
            dimensions: dim,
        },
        _ => {
            eprintln!("Unknown function: {}. Use: sphere, rosenbrock, rastrigin", function);
            return;
        }
    };

    match algo {
        "annealing" => {
            let sa = ix_optimize::annealing::SimulatedAnnealing::new()
                .with_max_iterations(max_iter);
            let initial = Array1::from_elem(dim, 5.0);
            let result = sa.minimize(&obj, initial);
            print_result("Simulated Annealing", &result);
        }
        "pso" => {
            let pso = ix_optimize::pso::ParticleSwarm::new()
                .with_max_iterations(max_iter);
            let result = pso.minimize(&obj);
            print_result("Particle Swarm", &result);
        }
        "genetic" => {
            let ga = ix_evolution::genetic::GeneticAlgorithm::new()
                .with_generations(max_iter);
            let result = ga.minimize(&obj.f, dim);
            println!("\n  Genetic Algorithm:");
            println!("    Best fitness: {:.6}", result.best_fitness);
            println!("    Best params:  {:?}", result.best_genes.to_vec());
            println!("    Generations:  {}", result.generations);
        }
        "differential" => {
            let de = ix_evolution::differential::DifferentialEvolution::new()
                .with_generations(max_iter);
            let result = de.minimize(&obj.f, dim);
            println!("\n  Differential Evolution:");
            println!("    Best fitness: {:.6}", result.best_fitness);
            println!("    Best params:  {:?}", result.best_genes.to_vec());
            println!("    Generations:  {}", result.generations);
        }
        "sgd" | "adam" => {
            use ix_optimize::convergence::ConvergenceCriteria;
            let criteria = ConvergenceCriteria { max_iterations: max_iter, tolerance: 1e-8 };
            let initial = Array1::from_elem(dim, 5.0);

            let result = if algo == "adam" {
                let mut opt = ix_optimize::gradient::Adam::new(0.01);
                ix_optimize::gradient::minimize(&obj, &mut opt, initial, &criteria)
            } else {
                let mut opt = ix_optimize::gradient::SGD::new(0.01);
                ix_optimize::gradient::minimize(&obj, &mut opt, initial, &criteria)
            };
            print_result(algo, &result);
        }
        _ => {
            eprintln!("Unknown algorithm: {}. Use: sgd, adam, annealing, pso, genetic, differential", algo);
        }
    }
}

fn print_result(name: &str, result: &ix_optimize::traits::OptimizeResult) {
    println!("\n  {}:", name);
    println!("    Best value:   {:.6}", result.best_value);
    println!("    Best params:  {:?}", result.best_params.to_vec());
    println!("    Iterations:   {}", result.iterations);
    println!("    Converged:    {}", result.converged);
}

fn print_algorithms() {
    println!("ix - ML algorithms for Claude Code skills\n");
    println!("OPTIMIZATION:");
    println!("  sgd            - Stochastic Gradient Descent");
    println!("  adam           - Adam optimizer");
    println!("  annealing      - Simulated Annealing");
    println!("  pso            - Particle Swarm Optimization");
    println!("  genetic        - Genetic Algorithm");
    println!("  differential   - Differential Evolution");
    println!();
    println!("SUPERVISED LEARNING:");
    println!("  linear         - Linear Regression (OLS)");
    println!("  logistic       - Logistic Regression");
    println!("  knn            - k-Nearest Neighbors");
    println!("  naive-bayes    - Gaussian Naive Bayes");
    println!("  decision-tree  - Decision Tree (CART)");
    println!("  svm            - Linear SVM");
    println!();
    println!("UNSUPERVISED LEARNING:");
    println!("  kmeans         - K-Means clustering");
    println!("  dbscan         - DBSCAN density clustering");
    println!("  pca            - Principal Component Analysis");
    println!("  gmm            - Gaussian Mixture Model");
    println!();
    println!("NEURAL NETWORKS:");
    println!("  dense          - Dense layer + backprop");
    println!();
    println!("REINFORCEMENT LEARNING:");
    println!("  epsilon-greedy - Epsilon-Greedy bandit");
    println!("  ucb1           - Upper Confidence Bound");
    println!("  thompson       - Thompson Sampling");
}
