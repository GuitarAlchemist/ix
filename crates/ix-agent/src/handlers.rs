//! Tool handler implementations — each parses JSON params, calls the underlying crate, returns JSON.

use ndarray::{Array1, Array2};
use serde_json::{json, Value};

use ix_cache::{Cache, CacheConfig};

use std::sync::OnceLock;

/// Global cache instance shared across tool calls. Also used by
/// `ToolRegistry::run_pipeline` for R2 content-addressed step caching.
pub(crate) fn global_cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Cache::new(CacheConfig::default()))
}

// ── helpers ────────────────────────────────────────────────────

fn parse_f64_array(val: &Value, field: &str) -> Result<Vec<f64>, String> {
    val.get(field)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("Missing or invalid field '{}'", field))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| format!("Non-numeric value in '{}'", field)))
        .collect()
}

fn parse_f64_matrix(val: &Value, field: &str) -> Result<Vec<Vec<f64>>, String> {
    val.get(field)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("Missing or invalid field '{}'", field))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| format!("Non-array row in '{}'", field))?
                .iter()
                .map(|v| v.as_f64().ok_or_else(|| format!("Non-numeric value in '{}'", field)))
                .collect()
        })
        .collect()
}

fn vec_to_array1(v: &[f64]) -> Array1<f64> {
    Array1::from_vec(v.to_vec())
}

fn vecs_to_array2(rows: &[Vec<f64>]) -> Result<Array2<f64>, String> {
    if rows.is_empty() {
        return Err("Empty matrix".into());
    }
    let ncols = rows[0].len();
    if rows.iter().any(|r| r.len() != ncols) {
        return Err("Inconsistent row lengths in matrix".into());
    }
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), ncols), flat)
        .map_err(|e| format!("Matrix shape error: {}", e))
}

fn parse_usize(val: &Value, field: &str) -> Result<usize, String> {
    val.get(field)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| format!("Missing or invalid field '{}'", field))
}

fn parse_f64_matrix_to_ndarray(val: &Value, field: &str) -> Result<Array2<f64>, String> {
    let rows = parse_f64_matrix(val, field)?;
    vecs_to_array2(&rows)
}

fn parse_str<'a>(val: &'a Value, field: &str) -> Result<&'a str, String> {
    val.get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("Missing or invalid field '{}'", field))
}

// ── ix_stats ───────────────────────────────────────────────

pub fn stats(params: Value) -> Result<Value, String> {
    let data = parse_f64_array(&params, "data")?;
    if data.is_empty() {
        return Err("data must not be empty".into());
    }
    let arr = vec_to_array1(&data);

    let mean = ix_math::stats::mean(&arr).map_err(|e| format!("{}", e))?;
    let std_dev = ix_math::stats::std_dev(&arr).map_err(|e| format!("{}", e))?;
    let median = ix_math::stats::median(&arr).map_err(|e| format!("{}", e))?;
    let (min, max) = ix_math::stats::min_max(&arr).map_err(|e| format!("{}", e))?;
    let variance = ix_math::stats::variance(&arr).map_err(|e| format!("{}", e))?;

    Ok(json!({
        "mean": mean,
        "std_dev": std_dev,
        "variance": variance,
        "median": median,
        "min": min,
        "max": max,
        "count": data.len(),
    }))
}

// ── ix_distance ────────────────────────────────────────────

pub fn distance(params: Value) -> Result<Value, String> {
    let a = parse_f64_array(&params, "a")?;
    let b = parse_f64_array(&params, "b")?;
    let metric = parse_str(&params, "metric")?;

    let arr_a = vec_to_array1(&a);
    let arr_b = vec_to_array1(&b);

    let (distance, metric_name) = match metric {
        "euclidean" => {
            let d = ix_math::distance::euclidean(&arr_a, &arr_b)
                .map_err(|e| format!("{}", e))?;
            (d, "euclidean")
        }
        "cosine" => {
            let d = ix_math::distance::cosine_distance(&arr_a, &arr_b)
                .map_err(|e| format!("{}", e))?;
            (d, "cosine")
        }
        "manhattan" => {
            let d = ix_math::distance::manhattan(&arr_a, &arr_b)
                .map_err(|e| format!("{}", e))?;
            (d, "manhattan")
        }
        _ => return Err(format!("Unknown metric: {}", metric)),
    };

    Ok(json!({
        "distance": distance,
        "metric": metric_name,
    }))
}

// ── ix_optimize ────────────────────────────────────────────

pub fn optimize(params: Value) -> Result<Value, String> {
    let func_name = parse_str(&params, "function")?;
    let dimensions = parse_usize(&params, "dimensions")?;
    let method = parse_str(&params, "method")?;
    let max_iter = parse_usize(&params, "max_iter")?;

    if dimensions == 0 {
        return Err("dimensions must be >= 1".into());
    }

    #[allow(clippy::type_complexity)]
    let objective: ix_optimize::traits::ClosureObjective<Box<dyn Fn(&Array1<f64>) -> f64>> =
        match func_name {
            "sphere" => ix_optimize::traits::ClosureObjective {
                f: Box::new(|x: &Array1<f64>| x.mapv(|v| v * v).sum()),
                dimensions,
            },
            "rosenbrock" => ix_optimize::traits::ClosureObjective {
                f: Box::new(|x: &Array1<f64>| {
                    (0..x.len() - 1)
                        .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                        .sum()
                }),
                dimensions,
            },
            "rastrigin" => ix_optimize::traits::ClosureObjective {
                f: Box::new(|x: &Array1<f64>| {
                    let n = x.len() as f64;
                    10.0 * n
                        + x.iter()
                            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                            .sum::<f64>()
                }),
                dimensions,
            },
            _ => return Err(format!("Unknown function: {}", func_name)),
        };

    let result = match method {
        "sgd" => {
            let mut opt = ix_optimize::gradient::SGD::new(0.01);
            let criteria = ix_optimize::convergence::ConvergenceCriteria {
                max_iterations: max_iter,
                tolerance: 1e-8,
            };
            let init = Array1::from_elem(dimensions, 5.0);
            ix_optimize::gradient::minimize(&objective, &mut opt, init, &criteria)
        }
        "adam" => {
            let mut opt = ix_optimize::gradient::Adam::new(0.01);
            let criteria = ix_optimize::convergence::ConvergenceCriteria {
                max_iterations: max_iter,
                tolerance: 1e-8,
            };
            let init = Array1::from_elem(dimensions, 5.0);
            ix_optimize::gradient::minimize(&objective, &mut opt, init, &criteria)
        }
        "pso" => {
            let pso = ix_optimize::pso::ParticleSwarm::new()
                .with_max_iterations(max_iter)
                .with_bounds(-10.0, 10.0)
                .with_seed(42);
            pso.minimize(&objective)
        }
        "annealing" => {
            let sa = ix_optimize::annealing::SimulatedAnnealing::new()
                .with_max_iterations(max_iter)
                .with_seed(42);
            let init = Array1::from_elem(dimensions, 5.0);
            sa.minimize(&objective, init)
        }
        _ => return Err(format!("Unknown method: {}", method)),
    };

    Ok(json!({
        "best_params": result.best_params.to_vec(),
        "best_value": result.best_value,
        "iterations": result.iterations,
        "converged": result.converged,
        "method": method,
        "function": func_name,
    }))
}

// ── ix_linear_regression ───────────────────────────────────

pub fn linear_regression(params: Value) -> Result<Value, String> {
    use ix_supervised::traits::Regressor;

    let x_rows = parse_f64_matrix(&params, "x")?;
    let y_data = parse_f64_array(&params, "y")?;

    let x = vecs_to_array2(&x_rows)?;
    let y = vec_to_array1(&y_data);

    if x.nrows() != y.len() {
        return Err(format!(
            "x has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        ));
    }

    let mut model = ix_supervised::linear_regression::LinearRegression::new();
    model.fit(&x, &y);

    let predictions = model.predict(&x);
    let weights = model.weights.as_ref().map(|w| w.to_vec()).unwrap_or_default();

    Ok(json!({
        "weights": weights,
        "bias": model.bias,
        "predictions": predictions.to_vec(),
    }))
}

// ── ix_kmeans ──────────────────────────────────────────────

pub fn kmeans(params: Value) -> Result<Value, String> {
    use ix_unsupervised::traits::Clusterer;

    let data_rows = parse_f64_matrix(&params, "data")?;
    let k = parse_usize(&params, "k")?;
    let max_iter = parse_usize(&params, "max_iter")?;

    let data = vecs_to_array2(&data_rows)?;

    let mut km = ix_unsupervised::kmeans::KMeans::new(k);
    km.max_iterations = max_iter;
    km.seed = 42;

    let labels = km.fit_predict(&data);
    let centroids: Vec<Vec<f64>> = km
        .centroids
        .as_ref()
        .map(|c| (0..c.nrows()).map(|i| c.row(i).to_vec()).collect())
        .unwrap_or_default();

    let inertia = km.centroids.as_ref().map(|c| {
        ix_unsupervised::kmeans::inertia(&data, &labels, c)
    }).unwrap_or(0.0);

    Ok(json!({
        "labels": labels.to_vec(),
        "centroids": centroids,
        "inertia": inertia,
        "k": k,
    }))
}

// ── ix_fft ─────────────────────────────────────────────────

pub fn fft(params: Value) -> Result<Value, String> {
    let signal = parse_f64_array(&params, "signal")?;
    if signal.is_empty() {
        return Err("signal must not be empty".into());
    }

    let spectrum = ix_signal::fft::rfft(&signal);
    let magnitudes = ix_signal::fft::magnitude_spectrum(&spectrum);
    let n = spectrum.len();
    // Frequency bins assuming sample_rate=1.0 (normalized)
    let frequencies: Vec<f64> = (0..n).map(|k| k as f64 / n as f64).collect();

    Ok(json!({
        "frequencies": frequencies,
        "magnitudes": magnitudes,
        "fft_size": n,
    }))
}

// ── ix_markov ──────────────────────────────────────────────

pub fn markov(params: Value) -> Result<Value, String> {
    let tm_rows = parse_f64_matrix(&params, "transition_matrix")?;
    let steps = parse_usize(&params, "steps")?;

    let tm = vecs_to_array2(&tm_rows)?;
    let chain = ix_graph::markov::MarkovChain::new(tm)?;

    let stationary = chain.stationary_distribution(steps, 1e-10);
    let is_ergodic = chain.is_ergodic(100);

    Ok(json!({
        "stationary_distribution": stationary.to_vec(),
        "n_states": chain.n_states(),
        "is_ergodic": is_ergodic,
    }))
}

// ── ix_viterbi ─────────────────────────────────────────────

pub fn viterbi(params: Value) -> Result<Value, String> {
    let initial_data = parse_f64_array(&params, "initial")?;
    let transition_rows = parse_f64_matrix(&params, "transition")?;
    let emission_rows = parse_f64_matrix(&params, "emission")?;
    let observations: Vec<usize> = params
        .get("observations")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Missing or invalid field 'observations'".to_string())?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| "Non-integer in observations".to_string()).map(|n| n as usize))
        .collect::<Result<Vec<_>, _>>()?;

    let initial = vec_to_array1(&initial_data);
    let transition = vecs_to_array2(&transition_rows)?;
    let emission = vecs_to_array2(&emission_rows)?;

    let hmm = ix_graph::hmm::HiddenMarkovModel::new(initial, transition, emission)?;
    let (path, log_probability) = hmm.viterbi(&observations);

    Ok(json!({
        "path": path,
        "log_probability": log_probability,
    }))
}

// ── ix_search ──────────────────────────────────────────────

pub fn search_info(params: Value) -> Result<Value, String> {
    let algorithm = parse_str(&params, "algorithm")?;

    let info = match algorithm {
        "astar" => json!({
            "algorithm": "A*",
            "description": "A* is a best-first graph search algorithm that finds the shortest path using a heuristic function h(n) to estimate cost to goal. Combines Dijkstra's optimality with greedy best-first speed.",
            "time_complexity": "O(b^d) worst case, often much better with good heuristic",
            "space_complexity": "O(b^d)",
            "optimal": true,
            "complete": true,
            "requires_heuristic": true,
            "properties": ["optimal if h is admissible", "complete if branching factor is finite"]
        }),
        "bfs" => json!({
            "algorithm": "Breadth-First Search",
            "description": "BFS explores all neighbors at the current depth before moving to nodes at the next depth level. Guarantees shortest path in unweighted graphs.",
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V)",
            "optimal": "for unweighted graphs",
            "complete": true,
            "requires_heuristic": false,
            "properties": ["finds shortest path in unweighted graphs", "level-order traversal"]
        }),
        "dfs" => json!({
            "algorithm": "Depth-First Search",
            "description": "DFS explores as far as possible along each branch before backtracking. Memory efficient but does not guarantee shortest path.",
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V) worst case, O(d) with depth limit",
            "optimal": false,
            "complete": "only if graph is finite and no cycles (or with cycle detection)",
            "requires_heuristic": false,
            "properties": ["memory efficient", "good for topological sort", "can detect cycles"]
        }),
        _ => return Err(format!("Unknown algorithm: {}", algorithm)),
    };

    Ok(info)
}

// ── ix_game_nash ───────────────────────────────────────────

pub fn game_nash(params: Value) -> Result<Value, String> {
    let pa_rows = parse_f64_matrix(&params, "payoff_a")?;
    let pb_rows = parse_f64_matrix(&params, "payoff_b")?;

    let payoff_a = vecs_to_array2(&pa_rows)?;
    let payoff_b = vecs_to_array2(&pb_rows)?;

    let game = ix_game::nash::BimatrixGame::new(payoff_a, payoff_b);
    let equilibria = game.support_enumeration();

    let eq_json: Vec<Value> = equilibria
        .iter()
        .map(|e| {
            json!({
                "player_a": e.player_a.to_vec(),
                "player_b": e.player_b.to_vec(),
                "expected_payoff_a": e.expected_payoff_a(&game),
                "expected_payoff_b": e.expected_payoff_b(&game),
            })
        })
        .collect();

    Ok(json!({
        "equilibria": eq_json,
        "count": eq_json.len(),
    }))
}

// ── ix_chaos_lyapunov ──────────────────────────────────────

pub fn chaos_lyapunov(params: Value) -> Result<Value, String> {
    let map = parse_str(&params, "map")?;
    let parameter = params
        .get("parameter")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "Missing or invalid field 'parameter'".to_string())?;
    let iterations = parse_usize(&params, "iterations")?;

    match map {
        "logistic" => {
            let r = parameter;
            let f = move |x: f64| r * x * (1.0 - x);
            let df = move |x: f64| r * (1.0 - 2.0 * x);
            let mle = ix_chaos::lyapunov::mle_1d(f, df, 0.1, iterations, 1000);
            let dynamics = ix_chaos::lyapunov::classify_dynamics(mle, 0.01);

            Ok(json!({
                "lyapunov_exponent": mle,
                "dynamics": format!("{:?}", dynamics),
                "map": "logistic",
                "parameter": r,
                "iterations": iterations,
            }))
        }
        _ => Err(format!("Unknown map: {}", map)),
    }
}

// ── ix_adversarial_fgsm ────────────────────────────────────

pub fn adversarial_fgsm(params: Value) -> Result<Value, String> {
    let input = parse_f64_array(&params, "input")?;
    let gradient = parse_f64_array(&params, "gradient")?;
    let epsilon = params
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "Missing or invalid field 'epsilon'".to_string())?;

    let input_arr = vec_to_array1(&input);
    let grad_arr = vec_to_array1(&gradient);

    let adversarial = ix_adversarial::evasion::fgsm(&input_arr, &grad_arr, epsilon);
    let perturbation = &adversarial - &input_arr;
    let l_inf = perturbation.mapv(f64::abs).iter().cloned().fold(0.0_f64, f64::max);

    Ok(json!({
        "adversarial_input": adversarial.to_vec(),
        "perturbation": perturbation.to_vec(),
        "l_inf_norm": l_inf,
        "epsilon": epsilon,
    }))
}

// ── ix_bloom_filter ────────────────────────────────────────

pub fn bloom_filter(params: Value) -> Result<Value, String> {
    let items: Vec<String> = params
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Missing or invalid field 'items'".to_string())?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();

    let fp_rate = params
        .get("false_positive_rate")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "Missing or invalid field 'false_positive_rate'".to_string())?;

    let mut bf = ix_probabilistic::bloom::BloomFilter::new(items.len().max(1), fp_rate);
    for item in &items {
        bf.insert(item);
    }

    let operation = params
        .get("operation")
        .and_then(|v| v.as_str())
        .unwrap_or("create");

    match operation {
        "check" => {
            let query = parse_str(&params, "query")?;
            let probably_contains = bf.contains(&query.to_string());
            Ok(json!({
                "query": query,
                "probably_contains": probably_contains,
                "items_count": items.len(),
                "estimated_fp_rate": bf.estimated_fp_rate(),
                "bit_size": bf.bit_size(),
            }))
        }
        _ => {
            Ok(json!({
                "created": true,
                "items_count": items.len(),
                "bit_size": bf.bit_size(),
                "estimated_fp_rate": bf.estimated_fp_rate(),
            }))
        }
    }
}

// ── ix_grammar_weights ─────────────────────────────────────

pub fn grammar_weights(params: Value) -> Result<Value, String> {
    use ix_grammar::weighted::{bayesian_update, softmax, WeightedRule};

    // Parse rules array
    let rules_val = params
        .get("rules")
        .and_then(|v| v.as_array())
        .ok_or("Missing or invalid field 'rules'")?;

    let mut rules: Vec<WeightedRule> = rules_val
        .iter()
        .map(|r| {
            let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let alpha = r.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let beta = r.get("beta").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let weight = r.get("weight").and_then(|v| v.as_f64()).unwrap_or(alpha / (alpha + beta));
            let level = r.get("level").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let source = r.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string();
            WeightedRule { id, alpha, beta, weight, level, source }
        })
        .collect();

    // Apply Bayesian update if observation provided
    if let Some(obs) = params.get("observation") {
        let rule_id = obs.get("rule_id").and_then(|v| v.as_str())
            .ok_or("observation.rule_id required")?;
        let success = obs.get("success").and_then(|v| v.as_bool())
            .ok_or("observation.success required")?;

        for rule in &mut rules {
            if rule.id == rule_id {
                *rule = bayesian_update(rule, success);
                break;
            }
        }
    }

    let temperature = params.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let probs = softmax(&rules, temperature);

    let rules_json: Vec<Value> = rules
        .iter()
        .map(|r| json!({
            "id": r.id,
            "alpha": r.alpha,
            "beta": r.beta,
            "weight": r.weight,
            "level": r.level,
            "source": r.source,
        }))
        .collect();

    let probs_json: Vec<Value> = probs
        .iter()
        .map(|(id, p)| json!({ "rule_id": id, "probability": p }))
        .collect();

    Ok(json!({
        "updated_rules": rules_json,
        "probabilities": probs_json,
        "temperature": temperature,
    }))
}

// ── ix_grammar_evolve ──────────────────────────────────────

pub fn grammar_evolve(params: Value) -> Result<Value, String> {
    use ix_grammar::replicator::{simulate, GrammarSpecies};

    let species_val = params
        .get("species")
        .and_then(|v| v.as_array())
        .ok_or("Missing or invalid field 'species'")?;

    let species: Vec<GrammarSpecies> = species_val
        .iter()
        .map(|s| {
            let id = s.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let proportion = s.get("proportion").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let fitness = s.get("fitness").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let is_stable = s.get("is_stable").and_then(|v| v.as_bool()).unwrap_or(false);
            GrammarSpecies { id, proportion, fitness, is_stable }
        })
        .collect();

    let steps = parse_usize(&params, "steps")?;
    let dt = params.get("dt").and_then(|v| v.as_f64()).unwrap_or(0.05);
    let prune_threshold = params.get("prune_threshold").and_then(|v| v.as_f64()).unwrap_or(1e-6);

    let result = simulate(&species, steps, dt, prune_threshold);

    let species_to_json = |s: &GrammarSpecies| json!({
        "id": s.id,
        "proportion": s.proportion,
        "fitness": s.fitness,
        "is_stable": s.is_stable,
    });

    // Return trajectory sampled at most every 10 steps to keep payload manageable
    let sample_rate = (steps / 100).max(1);
    let trajectory_json: Vec<Value> = result.trajectory
        .iter()
        .step_by(sample_rate)
        .map(|step| step.iter().map(species_to_json).collect::<Vec<_>>().into())
        .collect();

    Ok(json!({
        "final_species": result.final_species.iter().map(species_to_json).collect::<Vec<_>>(),
        "trajectory": trajectory_json,
        "ess": result.ess.iter().map(species_to_json).collect::<Vec<_>>(),
        "steps": steps,
        "dt": dt,
    }))
}

// ── ix_grammar_search ──────────────────────────────────────

pub fn grammar_search(params: Value) -> Result<Value, String> {
    use ix_grammar::constrained::{search_derivation, EbnfGrammar};

    let grammar_str = parse_str(&params, "grammar_ebnf")?;
    let grammar = EbnfGrammar::from_str(grammar_str)?;

    let max_iterations = params.get("max_iterations").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
    let exploration = params.get("exploration").and_then(|v| v.as_f64()).unwrap_or(1.41);
    let max_depth = params.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

    let result = search_derivation(grammar, max_iterations, exploration, max_depth, seed);

    let derivation_json: Vec<Value> = result
        .best_derivation
        .iter()
        .map(|(nt, alt)| json!({ "nonterminal": nt, "alternative": alt }))
        .collect();

    Ok(json!({
        "best_derivation": derivation_json,
        "reward": result.reward,
        "iterations": result.iterations,
    }))
}

// ── ix_rotation ────────────────────────────────────────────

pub fn rotation(params: Value) -> Result<Value, String> {
    use ix_rotation::quaternion::Quaternion;

    let op = parse_str(&params, "operation")?;

    match op {
        "quaternion" => {
            let axis = parse_f64_array(&params, "axis")?;
            let angle = params.get("angle").and_then(|v| v.as_f64())
                .ok_or("Missing 'angle'")?;
            if axis.len() != 3 { return Err("axis must have 3 elements".into()); }
            let q = Quaternion::from_axis_angle([axis[0], axis[1], axis[2]], angle);
            Ok(json!({ "w": q.w, "x": q.x, "y": q.y, "z": q.z, "norm": q.norm() }))
        }
        "slerp" => {
            use ix_rotation::slerp::slerp;
            let axis1 = parse_f64_array(&params, "axis")?;
            let angle1 = params.get("angle").and_then(|v| v.as_f64()).ok_or("Missing 'angle'")?;
            let axis2 = parse_f64_array(&params, "axis2")?;
            let angle2 = params.get("angle2").and_then(|v| v.as_f64()).ok_or("Missing 'angle2'")?;
            let t = params.get("t").and_then(|v| v.as_f64()).ok_or("Missing 't'")?;
            if axis1.len() != 3 || axis2.len() != 3 { return Err("axes must have 3 elements".into()); }
            let q0 = Quaternion::from_axis_angle([axis1[0], axis1[1], axis1[2]], angle1);
            let q1 = Quaternion::from_axis_angle([axis2[0], axis2[1], axis2[2]], angle2);
            let r = slerp(&q0, &q1, t);
            Ok(json!({ "w": r.w, "x": r.x, "y": r.y, "z": r.z }))
        }
        "euler_to_quat" => {
            use ix_rotation::euler::{to_quaternion, EulerOrder};
            let roll = params.get("roll").and_then(|v| v.as_f64()).ok_or("Missing 'roll'")?;
            let pitch = params.get("pitch").and_then(|v| v.as_f64()).ok_or("Missing 'pitch'")?;
            let yaw = params.get("yaw").and_then(|v| v.as_f64()).ok_or("Missing 'yaw'")?;
            let q = to_quaternion(roll, pitch, yaw, EulerOrder::XYZ);
            Ok(json!({ "w": q.w, "x": q.x, "y": q.y, "z": q.z, "norm": q.norm() }))
        }
        "quat_to_euler" => {
            use ix_rotation::euler::{from_quaternion, EulerOrder, gimbal_lock_check};
            let qv = parse_f64_array(&params, "quaternion")?;
            if qv.len() != 4 { return Err("quaternion must have 4 elements [w,x,y,z]".into()); }
            let q = Quaternion::new(qv[0], qv[1], qv[2], qv[3]);
            let (roll, pitch, yaw) = from_quaternion(&q, EulerOrder::XYZ);
            Ok(json!({ "roll": roll, "pitch": pitch, "yaw": yaw, "gimbal_lock": gimbal_lock_check(pitch) }))
        }
        "rotate_point" => {
            let axis = parse_f64_array(&params, "axis")?;
            let angle = params.get("angle").and_then(|v| v.as_f64()).ok_or("Missing 'angle'")?;
            let point = parse_f64_array(&params, "point")?;
            if axis.len() != 3 || point.len() != 3 { return Err("axis and point must have 3 elements".into()); }
            let q = Quaternion::from_axis_angle([axis[0], axis[1], axis[2]], angle);
            let rotated = q.rotate_point([point[0], point[1], point[2]]);
            Ok(json!({ "rotated_point": rotated }))
        }
        "rotation_matrix" => {
            use ix_rotation::rotation_matrix::{from_quaternion, is_rotation_matrix};
            let axis = parse_f64_array(&params, "axis")?;
            let angle = params.get("angle").and_then(|v| v.as_f64()).ok_or("Missing 'angle'")?;
            if axis.len() != 3 { return Err("axis must have 3 elements".into()); }
            let q = Quaternion::from_axis_angle([axis[0], axis[1], axis[2]], angle);
            let m = from_quaternion(&q);
            let valid = is_rotation_matrix(&m, 1e-8);
            Ok(json!({ "matrix": m, "valid": valid, "quaternion": { "w": q.w, "x": q.x, "y": q.y, "z": q.z } }))
        }
        _ => Err(format!("Unknown rotation operation: {}", op)),
    }
}

// ── ix_number_theory ──────────────────────────────────────

pub fn number_theory(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;

    match op {
        "sieve" => {
            let limit = parse_usize(&params, "limit")?;
            let primes = ix_number_theory::sieve::sieve_of_eratosthenes(limit);
            Ok(json!({ "primes": primes, "count": primes.len(), "limit": limit }))
        }
        "is_prime" => {
            let n = params.get("n").and_then(|v| v.as_u64()).ok_or("Missing 'n'")?;
            let trial = ix_number_theory::primality::is_prime_trial(n);
            let miller_rabin = ix_number_theory::primality::is_prime_miller_rabin(n, 10);
            Ok(json!({ "n": n, "is_prime_trial": trial, "is_prime_miller_rabin": miller_rabin }))
        }
        "mod_pow" => {
            let base = params.get("base").and_then(|v| v.as_u64()).ok_or("Missing 'base'")?;
            let exp = params.get("exp").and_then(|v| v.as_u64()).ok_or("Missing 'exp'")?;
            let modulus = params.get("modulus").and_then(|v| v.as_u64()).ok_or("Missing 'modulus'")?;
            let result = ix_number_theory::modular::mod_pow(base, exp, modulus);
            Ok(json!({ "result": result, "expression": format!("{}^{} mod {}", base, exp, modulus) }))
        }
        "gcd" => {
            let a = params.get("a").and_then(|v| v.as_u64()).ok_or("Missing 'a'")?;
            let b = params.get("b").and_then(|v| v.as_u64()).ok_or("Missing 'b'")?;
            let g = ix_number_theory::modular::gcd(a, b);
            Ok(json!({ "gcd": g, "a": a, "b": b }))
        }
        "lcm" => {
            let a = params.get("a").and_then(|v| v.as_u64()).ok_or("Missing 'a'")?;
            let b = params.get("b").and_then(|v| v.as_u64()).ok_or("Missing 'b'")?;
            let l = ix_number_theory::modular::lcm(a, b);
            Ok(json!({ "lcm": l, "a": a, "b": b }))
        }
        "mod_inverse" => {
            let a = params.get("a").and_then(|v| v.as_u64()).ok_or("Missing 'a'")?;
            let modulus = params.get("modulus").and_then(|v| v.as_u64()).ok_or("Missing 'modulus'")?;
            let inv = ix_number_theory::modular::mod_inverse(a, modulus);
            Ok(json!({ "inverse": inv, "a": a, "modulus": modulus, "exists": inv.is_some() }))
        }
        "prime_gaps" => {
            let limit = parse_usize(&params, "limit")?;
            let primes = ix_number_theory::sieve::sieve_of_eratosthenes(limit);
            let gaps: Vec<usize> = primes.windows(2).map(|w| w[1] - w[0]).collect();
            let max_gap = gaps.iter().copied().max().unwrap_or(0);
            let avg_gap = if gaps.is_empty() { 0.0 } else { gaps.iter().sum::<usize>() as f64 / gaps.len() as f64 };
            Ok(json!({ "prime_count": primes.len(), "max_gap": max_gap, "avg_gap": avg_gap, "first_10_gaps": &gaps[..gaps.len().min(10)] }))
        }
        _ => Err(format!("Unknown number theory operation: {}", op)),
    }
}

// ── ix_fractal ────────────────────────────────────────────

pub fn fractal(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;

    match op {
        "takagi" => {
            let n_points = parse_usize(&params, "n_points")?;
            let terms = parse_usize(&params, "terms")?;
            let curve = ix_fractal::takagi::takagi_series(n_points, terms);
            let step = 1.0 / (n_points - 1).max(1) as f64;
            let points: Vec<[f64; 2]> = curve.iter().enumerate()
                .map(|(i, &y)| [i as f64 * step, y]).collect();
            Ok(json!({ "points": points, "n_points": n_points, "terms": terms }))
        }
        "hilbert" => {
            let order = params.get("order").and_then(|v| v.as_u64()).ok_or("Missing 'order'")? as u32;
            let points = ix_fractal::space_filling::hilbert_curve(order);
            Ok(json!({ "points": points, "order": order, "n_points": points.len() }))
        }
        "peano" => {
            let order = params.get("order").and_then(|v| v.as_u64()).ok_or("Missing 'order'")? as u32;
            let points = ix_fractal::space_filling::peano_curve(order);
            Ok(json!({ "points": points, "order": order, "n_points": points.len() }))
        }
        "morton_encode" => {
            let x = params.get("x").and_then(|v| v.as_u64()).ok_or("Missing 'x'")? as u32;
            let y = params.get("y").and_then(|v| v.as_u64()).ok_or("Missing 'y'")? as u32;
            let z = ix_fractal::space_filling::morton_encode(x, y);
            Ok(json!({ "z_order": z, "x": x, "y": y }))
        }
        "morton_decode" => {
            let z = params.get("z").and_then(|v| v.as_u64()).ok_or("Missing 'z'")?;
            let (x, y) = ix_fractal::space_filling::morton_decode(z);
            Ok(json!({ "x": x, "y": y, "z_order": z }))
        }
        _ => Err(format!("Unknown fractal operation: {}", op)),
    }
}

// ── ix_sedenion ───────────────────────────────────────────

pub fn sedenion(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;
    let a_vec = parse_f64_array(&params, "a")?;

    match op {
        "multiply" => {
            let b_vec = parse_f64_array(&params, "b")?;
            if a_vec.len() != b_vec.len() { return Err("a and b must have same length".into()); }
            let product = ix_sedenion::cayley_dickson::double_multiply(&a_vec, &b_vec);
            Ok(json!({ "product": product, "dimension": a_vec.len() }))
        }
        "conjugate" => {
            let conj = ix_sedenion::cayley_dickson::double_conjugate(&a_vec);
            Ok(json!({ "conjugate": conj, "dimension": a_vec.len() }))
        }
        "norm" => {
            let n = ix_sedenion::cayley_dickson::double_norm(&a_vec);
            Ok(json!({ "norm": n, "dimension": a_vec.len() }))
        }
        "cayley_dickson_multiply" => {
            let b_vec = parse_f64_array(&params, "b")?;
            let product = ix_sedenion::cayley_dickson::double_multiply(&a_vec, &b_vec);
            Ok(json!({ "product": product, "dimension": a_vec.len() }))
        }
        _ => Err(format!("Unknown sedenion operation: {}", op)),
    }
}

// ── ix_topo ───────────────────────────────────────────────

pub fn topo(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;
    let points_raw = parse_f64_matrix(&params, "points")?;

    let max_dim = params.get("max_dim").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let max_radius = params.get("max_radius").and_then(|v| v.as_f64()).unwrap_or(2.0);

    match op {
        "persistence" => {
            let diagrams = ix_topo::pointcloud::persistence_from_points(&points_raw, max_dim, max_radius);
            let diag_json: Vec<Value> = diagrams.iter().enumerate().map(|(dim, d)| {
                let pairs: Vec<Value> = d.pairs.iter().map(|p| json!({ "birth": p.0, "death": p.1 })).collect();
                json!({ "dimension": dim, "pairs": pairs })
            }).collect();
            Ok(json!({ "diagrams": diag_json, "max_dim": max_dim, "max_radius": max_radius }))
        }
        "betti_at_radius" => {
            let radius = params.get("radius").and_then(|v| v.as_f64()).ok_or("Missing 'radius'")?;
            let betti = ix_topo::pointcloud::betti_at_radius(&points_raw, max_dim, radius);
            Ok(json!({ "betti_numbers": betti, "radius": radius }))
        }
        "betti_curve" => {
            let n_steps = params.get("n_steps").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
            let curve = ix_topo::pointcloud::betti_curve(&points_raw, max_dim, n_steps);
            let curve_json: Vec<Value> = curve.iter().map(|(r, b)| json!({ "radius": r, "betti": b })).collect();
            Ok(json!({ "curve": curve_json, "n_steps": n_steps }))
        }
        _ => Err(format!("Unknown topo operation: {}", op)),
    }
}

// ── ix_category ───────────────────────────────────────────

pub fn category(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;

    match op {
        "monad_laws" => {
            use ix_category::monad::{OptionMonad, Monad};
            let a = params.get("value").and_then(|v| v.as_i64()).unwrap_or(5) as i32;
            let f = |x: i32| -> Option<i32> { Some(x + 1) };
            let g = |x: i32| -> Option<i32> { Some(x * 2) };

            // Left unit: bind(unit(a), f) == f(a)
            let lhs_left: Option<i32> = OptionMonad::bind(OptionMonad::unit(a), f);
            let rhs_left = f(a);
            let left_ok = lhs_left == rhs_left;

            // Right unit: bind(m, unit) == m
            let m = OptionMonad::unit(a);
            let lhs_right: Option<i32> = OptionMonad::bind(m, OptionMonad::unit);
            let right_ok = lhs_right == m;

            // Associativity
            let bind_m_f: Option<i32> = OptionMonad::bind(m, f);
            let lhs_assoc: Option<i32> = match bind_m_f {
                Some(v) => OptionMonad::bind(Some(v), g),
                None => None,
            };
            let rhs_assoc: Option<i32> = OptionMonad::bind(m, |x| {
                match f(x) { Some(v) => OptionMonad::bind(Some(v), g), None => None }
            });
            let assoc_ok = lhs_assoc == rhs_assoc;

            Ok(json!({
                "value": a,
                "left_unit": { "pass": left_ok, "lhs": format!("{:?}", lhs_left), "rhs": format!("{:?}", rhs_left) },
                "right_unit": { "pass": right_ok, "lhs": format!("{:?}", lhs_right), "rhs": format!("{:?}", m) },
                "associativity": { "pass": assoc_ok, "lhs": format!("{:?}", lhs_assoc), "rhs": format!("{:?}", rhs_assoc) },
                "all_pass": left_ok && right_ok && assoc_ok,
            }))
        }
        "free_forgetful" => {
            use ix_category::monad::FreeForgetfulAdj;
            let elements: Vec<i32> = params.get("elements")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|n| n as i32)).collect())
                .unwrap_or_else(|| vec![1, 2, 3]);
            let free = FreeForgetfulAdj::free(&elements);
            let forget = FreeForgetfulAdj::forget(&free);
            let round_trip_ok = elements == forget;
            Ok(json!({
                "input": elements,
                "free": free,
                "forget": forget,
                "round_trip_ok": round_trip_ok,
            }))
        }
        _ => Err(format!("Unknown category operation: {}", op)),
    }
}

// ── ix_nn_forward ─────────────────────────────────────────

pub fn nn_forward(params: Value) -> Result<Value, String> {
    let op = parse_str(&params, "operation")?;

    match op {
        "dense_forward" => {
            use ix_nn::layer::Layer;
            let input_rows = parse_f64_matrix(&params, "input")?;
            let output_size = parse_usize(&params, "output_size")?;
            let input = vecs_to_array2(&input_rows)?;
            let input_size = input.ncols();
            let mut layer = ix_nn::layer::Dense::new(input_size, output_size);
            let output = layer.forward(&input);
            let output_rows: Vec<Vec<f64>> = (0..output.nrows())
                .map(|i| output.row(i).to_vec()).collect();
            Ok(json!({ "output": output_rows, "input_size": input_size, "output_size": output_size }))
        }
        "mse_loss" => {
            let pred_rows = parse_f64_matrix(&params, "input")?;
            let target_rows = parse_f64_matrix(&params, "target")?;
            let pred = vecs_to_array2(&pred_rows)?;
            let target = vecs_to_array2(&target_rows)?;
            let loss = ix_nn::loss::mse_loss(&pred, &target);
            Ok(json!({ "mse_loss": loss }))
        }
        "bce_loss" => {
            let pred_rows = parse_f64_matrix(&params, "input")?;
            let target_rows = parse_f64_matrix(&params, "target")?;
            let pred = vecs_to_array2(&pred_rows)?;
            let target = vecs_to_array2(&target_rows)?;
            let loss = ix_nn::loss::binary_cross_entropy(&pred, &target);
            Ok(json!({ "bce_loss": loss }))
        }
        "sinusoidal_encoding" => {
            let max_len = parse_usize(&params, "max_len")?;
            let d_model = parse_usize(&params, "d_model")?;
            let enc = ix_nn::positional::sinusoidal_encoding(max_len, d_model);
            let rows: Vec<Vec<f64>> = (0..enc.nrows())
                .map(|i| enc.row(i).to_vec()).collect();
            Ok(json!({ "encoding": rows, "max_len": max_len, "d_model": d_model }))
        }
        _ => Err(format!("Unknown nn operation: {}", op)),
    }
}

// ── ix_bandit ─────────────────────────────────────────────

pub fn bandit(params: Value) -> Result<Value, String> {
    use rand::SeedableRng;
    use rand_distr::{Normal, Distribution};

    let algo = parse_str(&params, "algorithm")?;
    let true_means = parse_f64_array(&params, "true_means")?;
    let rounds = parse_usize(&params, "rounds")?;
    let n_arms = true_means.len();

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut total_reward = 0.0;
    let mut arm_pulls = vec![0usize; n_arms];

    match algo {
        "epsilon_greedy" => {
            let epsilon = params.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(0.1);
            let mut bandit = ix_rl::bandit::EpsilonGreedy::new(n_arms, epsilon, 42);
            for _ in 0..rounds {
                let arm = bandit.select_arm();
                let normal = Normal::new(true_means[arm], 1.0).unwrap();
                let reward = normal.sample(&mut rng);
                bandit.update(arm, reward);
                total_reward += reward;
                arm_pulls[arm] += 1;
            }
            Ok(json!({
                "algorithm": "epsilon_greedy",
                "q_values": bandit.q_values,
                "arm_pulls": arm_pulls,
                "total_reward": total_reward,
                "avg_reward": total_reward / rounds as f64,
            }))
        }
        "ucb1" => {
            let mut bandit = ix_rl::bandit::UCB1::new(n_arms);
            for _ in 0..rounds {
                let arm = bandit.select_arm();
                let normal = Normal::new(true_means[arm], 1.0).unwrap();
                let reward = normal.sample(&mut rng);
                bandit.update(arm, reward);
                total_reward += reward;
                arm_pulls[arm] += 1;
            }
            Ok(json!({
                "algorithm": "ucb1",
                "q_values": bandit.q_values,
                "arm_pulls": arm_pulls,
                "total_reward": total_reward,
                "avg_reward": total_reward / rounds as f64,
            }))
        }
        "thompson" => {
            let mut bandit = ix_rl::bandit::ThompsonSampling::new(n_arms, 42);
            for _ in 0..rounds {
                let arm = bandit.select_arm();
                let normal = Normal::new(true_means[arm], 1.0).unwrap();
                let reward = normal.sample(&mut rng);
                bandit.update(arm, reward);
                total_reward += reward;
                arm_pulls[arm] += 1;
            }
            Ok(json!({
                "algorithm": "thompson",
                "means": bandit.means,
                "arm_pulls": arm_pulls,
                "total_reward": total_reward,
                "avg_reward": total_reward / rounds as f64,
            }))
        }
        _ => Err(format!("Unknown bandit algorithm: {}", algo)),
    }
}

// ── ix_evolution ──────────────────────────────────────────

pub fn evolution(params: Value) -> Result<Value, String> {
    let algo = parse_str(&params, "algorithm")?;
    let func_name = parse_str(&params, "function")?;
    let dimensions = parse_usize(&params, "dimensions")?;
    let generations = parse_usize(&params, "generations")?;
    let pop_size = params.get("population_size").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

    #[allow(clippy::type_complexity)]
    let fitness_fn: Box<dyn Fn(&Array1<f64>) -> f64> = match func_name {
        "sphere" => Box::new(|x: &Array1<f64>| x.mapv(|v| v * v).sum()),
        "rosenbrock" => Box::new(|x: &Array1<f64>| {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                .sum()
        }),
        "rastrigin" => Box::new(|x: &Array1<f64>| {
            let n = x.len() as f64;
            10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
        }),
        _ => return Err(format!("Unknown function: {}", func_name)),
    };

    let result = match algo {
        "genetic" => {
            let mutation_rate = params.get("mutation_rate").and_then(|v| v.as_f64()).unwrap_or(0.1);
            let ga = ix_evolution::genetic::GeneticAlgorithm::new()
                .with_population_size(pop_size)
                .with_generations(generations)
                .with_mutation_rate(mutation_rate)
                .with_bounds(-10.0, 10.0)
                .with_seed(42);
            ga.minimize(&fitness_fn, dimensions)
        }
        "differential" => {
            let de = ix_evolution::differential::DifferentialEvolution::new()
                .with_population_size(pop_size)
                .with_generations(generations)
                .with_bounds(-10.0, 10.0)
                .with_seed(42);
            de.minimize(&fitness_fn, dimensions)
        }
        _ => return Err(format!("Unknown evolution algorithm: {}", algo)),
    };

    Ok(json!({
        "algorithm": algo,
        "function": func_name,
        "best_params": result.best_genes.to_vec(),
        "best_fitness": result.best_fitness,
        "generations": result.generations,
        "fitness_history_len": result.fitness_history.len(),
    }))
}

// ── ix_random_forest ──────────────────────────────────────

pub fn random_forest(params: Value) -> Result<Value, String> {
    use ix_ensemble::traits::EnsembleClassifier;

    let x_train_rows = parse_f64_matrix(&params, "x_train")?;
    let y_train_raw: Vec<usize> = params.get("y_train")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'y_train'")?
        .iter()
        .map(|v| v.as_u64().ok_or("Non-integer in y_train").map(|n| n as usize))
        .collect::<Result<Vec<_>, _>>()?;
    let x_test_rows = parse_f64_matrix(&params, "x_test")?;

    let x_train = vecs_to_array2(&x_train_rows)?;
    let y_train = Array1::from_vec(y_train_raw);
    let x_test = vecs_to_array2(&x_test_rows)?;

    let n_trees = params.get("n_trees").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let max_depth = params.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    let mut rf = ix_ensemble::random_forest::RandomForest::new(n_trees, max_depth).with_seed(42);
    rf.fit(&x_train, &y_train);
    let predictions = rf.predict(&x_test);
    let probas = rf.predict_proba(&x_test);
    let proba_rows: Vec<Vec<f64>> = (0..probas.nrows())
        .map(|i| probas.row(i).to_vec()).collect();

    Ok(json!({
        "predictions": predictions.to_vec(),
        "probabilities": proba_rows,
        "n_trees": n_trees,
        "max_depth": max_depth,
    }))
}

// ── ix_gradient_boosting ──────────────────────────────────

pub fn gradient_boosting(params: Value) -> Result<Value, String> {
    use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
    use ix_ensemble::traits::EnsembleClassifier;

    let x_train_rows = parse_f64_matrix(&params, "x_train")?;
    let y_train_raw: Vec<usize> = params.get("y_train")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'y_train'")?
        .iter()
        .map(|v| v.as_u64().ok_or("Non-integer in y_train").map(|n| n as usize))
        .collect::<Result<Vec<_>, _>>()?;
    let x_test_rows = parse_f64_matrix(&params, "x_test")?;

    let x_train = vecs_to_array2(&x_train_rows)?;
    let y_train = Array1::from_vec(y_train_raw);
    let x_test = vecs_to_array2(&x_test_rows)?;

    let n_estimators = params.get("n_estimators").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let learning_rate = params.get("learning_rate").and_then(|v| v.as_f64()).unwrap_or(0.1);

    let mut gbc = GradientBoostedClassifier::new(n_estimators, learning_rate);
    gbc.fit(&x_train, &y_train);
    let predictions = gbc.predict(&x_test);
    let probas = gbc.predict_proba(&x_test);
    let proba_rows: Vec<Vec<f64>> = (0..probas.nrows())
        .map(|i| probas.row(i).to_vec()).collect();

    Ok(json!({
        "predictions": predictions.to_vec(),
        "probabilities": proba_rows,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }))
}

// ── ix_supervised ──────────────────────────────────────────

pub fn supervised(params: Value) -> Result<Value, String> {
    use ix_supervised::linear_regression::LinearRegression;
    use ix_supervised::logistic_regression::LogisticRegression;
    use ix_supervised::svm::LinearSVM;
    use ix_supervised::knn::KNN;
    use ix_supervised::naive_bayes::GaussianNaiveBayes;
    use ix_supervised::decision_tree::DecisionTree;
    use ix_supervised::traits::{Classifier, Regressor};
    use ix_supervised::metrics;

    let operation = parse_str(&params, "operation")?;

    match operation {
        "linear_regression" | "logistic_regression" | "svm" | "knn" | "naive_bayes"
        | "decision_tree" => {
            let x_train = parse_f64_matrix_to_ndarray(&params, "x_train")?;
            let y_train_raw = parse_f64_array(&params, "y_train")?;
            let x_test = parse_f64_matrix_to_ndarray(&params, "x_test")?;

            match operation {
                "linear_regression" => {
                    let mut model = LinearRegression::new();
                    let y = Array1::from_vec(y_train_raw);
                    model.fit(&x_train, &y);
                    let preds = model.predict(&x_test);
                    Ok(json!({ "predictions": preds.to_vec(), "algorithm": "linear_regression" }))
                }
                "logistic_regression" => {
                    let mut model = LogisticRegression::new();
                    let y: Array1<usize> = Array1::from_vec(
                        y_train_raw.iter().map(|v| *v as usize).collect(),
                    );
                    model.fit(&x_train, &y);
                    let preds = model.predict(&x_test);
                    let probs = model.predict_proba(&x_test);
                    Ok(json!({
                        "predictions": preds.to_vec(),
                        "probabilities": probs.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                        "algorithm": "logistic_regression"
                    }))
                }
                "svm" => {
                    let c = params.get("c").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    let mut model = LinearSVM::new(c);
                    let y: Array1<usize> = Array1::from_vec(
                        y_train_raw.iter().map(|v| *v as usize).collect(),
                    );
                    model.fit(&x_train, &y);
                    let preds = model.predict(&x_test);
                    Ok(json!({ "predictions": preds.to_vec(), "algorithm": "svm", "c": c }))
                }
                "knn" => {
                    let k = params.get("k").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
                    let mut model = KNN::new(k);
                    let y: Array1<usize> = Array1::from_vec(
                        y_train_raw.iter().map(|v| *v as usize).collect(),
                    );
                    model.fit(&x_train, &y);
                    let preds = model.predict(&x_test);
                    let probs = model.predict_proba(&x_test);
                    Ok(json!({
                        "predictions": preds.to_vec(),
                        "probabilities": probs.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                        "algorithm": "knn", "k": k
                    }))
                }
                "naive_bayes" => {
                    let mut model = GaussianNaiveBayes::new();
                    let y: Array1<usize> = Array1::from_vec(
                        y_train_raw.iter().map(|v| *v as usize).collect(),
                    );
                    model.fit(&x_train, &y);
                    let preds = model.predict(&x_test);
                    Ok(json!({ "predictions": preds.to_vec(), "algorithm": "naive_bayes" }))
                }
                "decision_tree" => {
                    let max_depth = params.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                    let mut model = DecisionTree::new(max_depth);
                    let y: Array1<usize> = Array1::from_vec(
                        y_train_raw.iter().map(|v| *v as usize).collect(),
                    );
                    model.fit(&x_train, &y);
                    let preds = model.predict(&x_test);
                    let probs = model.predict_proba(&x_test);
                    Ok(json!({
                        "predictions": preds.to_vec(),
                        "probabilities": probs.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                        "algorithm": "decision_tree", "max_depth": max_depth
                    }))
                }
                _ => unreachable!(),
            }
        }
        "metrics" => {
            let y_true_raw = parse_f64_array(&params, "y_true")?;
            let y_pred_raw = parse_f64_array(&params, "y_pred")?;
            let metric_type = parse_str(&params, "metric_type")?;

            match metric_type {
                "mse" => {
                    let yt = Array1::from_vec(y_true_raw);
                    let yp = Array1::from_vec(y_pred_raw);
                    Ok(json!({ "mse": metrics::mse(&yt, &yp), "rmse": metrics::rmse(&yt, &yp), "r_squared": metrics::r_squared(&yt, &yp) }))
                }
                "accuracy" => {
                    let yt: Array1<usize> = Array1::from_vec(y_true_raw.iter().map(|v| *v as usize).collect());
                    let yp: Array1<usize> = Array1::from_vec(y_pred_raw.iter().map(|v| *v as usize).collect());
                    let class = params.get("class").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                    Ok(json!({
                        "accuracy": metrics::accuracy(&yt, &yp),
                        "precision": metrics::precision(&yt, &yp, class),
                        "recall": metrics::recall(&yt, &yp, class),
                        "f1": metrics::f1_score(&yt, &yp, class)
                    }))
                }
                _ => Err(format!("Unknown metric_type: {metric_type}. Use 'mse' or 'accuracy'")),
            }
        }
        "confusion_matrix" => {
            let y_true_raw = parse_f64_array(&params, "y_true")?;
            let y_pred_raw = parse_f64_array(&params, "y_pred")?;
            let yt: Array1<usize> = Array1::from_vec(y_true_raw.iter().map(|v| *v as usize).collect());
            let yp: Array1<usize> = Array1::from_vec(y_pred_raw.iter().map(|v| *v as usize).collect());
            let n_classes = params.get("n_classes").and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or_else(|| *yt.iter().chain(yp.iter()).max().unwrap() + 1);

            let cm = metrics::ConfusionMatrix::from_labels(&yt, &yp, n_classes);
            let (prec, rec, f1, support) = cm.classification_report();
            let matrix: Vec<Vec<usize>> = (0..n_classes)
                .map(|r| (0..n_classes).map(|c| cm.matrix()[[r, c]]).collect())
                .collect();

            Ok(json!({
                "confusion_matrix": matrix,
                "accuracy": cm.accuracy(),
                "per_class": {
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "support": support
                },
                "display": cm.display()
            }))
        }
        "roc_auc" => {
            let y_true_raw = parse_f64_array(&params, "y_true")?;
            let y_scores_raw = parse_f64_array(&params, "y_scores")?;
            let yt: Array1<usize> = Array1::from_vec(y_true_raw.iter().map(|v| *v as usize).collect());
            let ys = Array1::from_vec(y_scores_raw);

            let (fpr, tpr, thresholds) = metrics::roc_curve(&yt, &ys);
            let auc = metrics::roc_auc(&fpr, &tpr);

            Ok(json!({
                "auc": auc,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds
            }))
        }
        "cross_validate" => {
            use ix_supervised::validation::cross_val_score;

            let x = parse_f64_matrix_to_ndarray(&params, "x_train")?;
            let y_raw = parse_f64_array(&params, "y_train")?;
            let y: Array1<usize> = Array1::from_vec(y_raw.iter().map(|v| *v as usize).collect());
            let k = params.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
            let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);
            let model = params.get("model").and_then(|v| v.as_str()).unwrap_or("decision_tree");

            let scores = match model {
                "knn" => {
                    let knn_k = params.get("knn_k").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
                    cross_val_score(&x, &y, || KNN::new(knn_k), k, seed)
                }
                "decision_tree" => {
                    let max_depth = params.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                    cross_val_score(&x, &y, || DecisionTree::new(max_depth), k, seed)
                }
                "naive_bayes" => {
                    cross_val_score(&x, &y, GaussianNaiveBayes::new, k, seed)
                }
                "logistic_regression" => {
                    cross_val_score(&x, &y, LogisticRegression::new, k, seed)
                }
                _ => return Err(format!("Unknown model for cross_validate: {model}. Use knn, decision_tree, naive_bayes, or logistic_regression")),
            };

            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let std = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64).sqrt();

            Ok(json!({
                "fold_scores": scores,
                "mean_accuracy": mean,
                "std_accuracy": std,
                "k_folds": k,
                "model": model
            }))
        }
        _ => Err(format!("Unknown supervised operation: {operation}")),
    }
}

// ── ix_graph_ops ───────────────────────────────────────────

/// Parse a node index out of a JSON value, describing *why* the value
/// is unacceptable when it is. `field` names the slot the value occupies
/// (`"from"`, `"to"`, `"source"`, etc.) and `ctx` is a short label used
/// in the error (e.g. `"edge[3]"`).
///
/// The previous implementation just returned `"Invalid 'from'"` on any
/// failure, which could mean anything from "missing" to "wrong type" to
/// "negative float". This version names the exact problem and echoes
/// the offending value so the caller can see why the parse failed.
fn parse_node_index(value: Option<&Value>, field: &str, ctx: &str) -> Result<usize, String> {
    let v = value.ok_or_else(|| format!("ix_graph {ctx}: missing '{field}' (node index)"))?;
    // Integer-valued JSON numbers pass through cleanly.
    if let Some(u) = v.as_u64() {
        return Ok(u as usize);
    }
    // A negative integer is always a mistake — node ids are unsigned.
    if let Some(i) = v.as_i64() {
        return Err(format!(
            "ix_graph {ctx}: '{field}' must be a non-negative integer node index, got {i}"
        ));
    }
    // Floats might be an honest mistake (e.g. 1.0 that round-trips
    // through a typed array). Accept integral floats and report
    // fractional ones specifically.
    if let Some(f) = v.as_f64() {
        if f.is_finite() && f >= 0.0 && f.fract() == 0.0 {
            return Ok(f as usize);
        }
        return Err(format!(
            "ix_graph {ctx}: '{field}' must be an integer node index, got float {f}"
        ));
    }
    // Any other JSON kind (string, bool, null, object, array) is
    // definitely wrong — name the kind.
    let kind = match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
        Value::Number(_) => "number",
    };
    Err(format!(
        "ix_graph {ctx}: '{field}' must be an integer node index, got {kind} ({v})"
    ))
}

pub fn graph_ops(params: Value) -> Result<Value, String> {
    use ix_graph::graph::Graph;

    let operation = parse_str(&params, "operation")?;

    match operation {
        "dijkstra" | "shortest_path" | "pagerank" | "bfs" | "dfs" | "topological_sort" => {
            // Build graph from edges
            let edges = params
                .get("edges")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "ix_graph: missing 'edges' array".to_string())?;
            let n = params
                .get("n_nodes")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| "ix_graph: missing 'n_nodes' (positive integer)".to_string())?
                as usize;
            let directed = params.get("directed").and_then(|v| v.as_bool()).unwrap_or(true);

            let mut g = Graph::with_nodes(n);
            for (i, e) in edges.iter().enumerate() {
                let ctx = format!("edge[{i}]");
                let arr = e.as_array().ok_or_else(|| {
                    format!(
                        "ix_graph {ctx}: must be a [from, to, weight] array, got {e}"
                    )
                })?;
                if arr.len() < 2 {
                    return Err(format!(
                        "ix_graph {ctx}: must have at least [from, to], got {} elements",
                        arr.len()
                    ));
                }
                let from = parse_node_index(arr.first(), "from", &ctx)?;
                let to = parse_node_index(arr.get(1), "to", &ctx)?;
                if from >= n {
                    return Err(format!(
                        "ix_graph {ctx}: 'from' = {from} is out of range for n_nodes = {n}"
                    ));
                }
                if to >= n {
                    return Err(format!(
                        "ix_graph {ctx}: 'to' = {to} is out of range for n_nodes = {n}"
                    ));
                }
                let w = arr.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0);
                if directed {
                    g.add_edge(from, to, w);
                } else {
                    g.add_undirected_edge(from, to, w);
                }
            }

            match operation {
                "dijkstra" => {
                    let source = parse_node_index(params.get("source"), "source", "dijkstra")?;
                    let (dists, _preds) = g.dijkstra(source);
                    let dist_map: serde_json::Map<String, Value> = dists
                        .iter()
                        .map(|(k, v)| (k.to_string(), json!(*v)))
                        .collect();
                    Ok(json!({ "distances": dist_map, "source": source }))
                }
                "shortest_path" => {
                    let source =
                        parse_node_index(params.get("source"), "source", "shortest_path")?;
                    let target =
                        parse_node_index(params.get("target"), "target", "shortest_path")?;
                    match g.shortest_path(source, target) {
                        Some(path) => Ok(json!({ "path": path, "found": true })),
                        None => Ok(json!({ "path": [], "found": false })),
                    }
                }
                "pagerank" => {
                    let damping = params.get("damping").and_then(|v| v.as_f64()).unwrap_or(0.85);
                    let iters = params
                        .get("iterations")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(100) as usize;
                    let ranks = g.pagerank(damping, iters);
                    let rank_map: serde_json::Map<String, Value> = ranks
                        .iter()
                        .map(|(k, v)| (k.to_string(), json!(*v)))
                        .collect();
                    Ok(json!({ "pagerank": rank_map, "damping": damping, "iterations": iters }))
                }
                "bfs" => {
                    let source = parse_node_index(params.get("source"), "source", "bfs")?;
                    let dists = g.bfs(source);
                    let dist_map: serde_json::Map<String, Value> = dists
                        .iter()
                        .map(|(k, v)| (k.to_string(), json!(*v)))
                        .collect();
                    Ok(json!({ "distances": dist_map, "source": source }))
                }
                "dfs" => {
                    let source = parse_node_index(params.get("source"), "source", "dfs")?;
                    let order = g.dfs(source);
                    Ok(json!({ "visit_order": order, "source": source }))
                }
                "topological_sort" => match g.topological_sort() {
                    Some(order) => Ok(json!({ "order": order, "is_dag": true })),
                    None => Ok(json!({ "order": [], "is_dag": false })),
                },
                _ => unreachable!(),
            }
        }
        _ => Err(format!("ix_graph: unknown operation '{operation}' (expected one of: dijkstra, shortest_path, pagerank, bfs, dfs, topological_sort)")),
    }
}

// ── ix_hyperloglog ─────────────────────────────────────────

pub fn hyperloglog(params: Value) -> Result<Value, String> {
    use ix_probabilistic::hyperloglog::HyperLogLog;

    let operation = parse_str(&params, "operation")?;

    match operation {
        "estimate" => {
            let precision = params.get("precision").and_then(|v| v.as_u64()).unwrap_or(14) as usize;
            let items = params.get("items").and_then(|v| v.as_array())
                .ok_or_else(|| "Missing 'items' array".to_string())?;

            let mut hll = HyperLogLog::new(precision);
            for item in items {
                if let Some(s) = item.as_str() {
                    hll.add(&s);
                } else if let Some(n) = item.as_i64() {
                    hll.add(&n);
                } else if let Some(n) = item.as_f64() {
                    hll.add(&n.to_bits());
                }
            }

            Ok(json!({
                "estimated_cardinality": hll.count(),
                "actual_items": items.len(),
                "precision": precision,
                "error_rate": hll.error_rate(),
                "memory_bytes": hll.memory_bytes()
            }))
        }
        "merge" => {
            let precision = params.get("precision").and_then(|v| v.as_u64()).unwrap_or(14) as usize;
            let sets = params.get("sets").and_then(|v| v.as_array())
                .ok_or_else(|| "Missing 'sets' array of arrays".to_string())?;

            let mut merged = HyperLogLog::new(precision);
            let mut per_set = Vec::new();

            for set in sets {
                let items = set.as_array().ok_or("Each set must be an array")?;
                let mut hll = HyperLogLog::new(precision);
                for item in items {
                    if let Some(s) = item.as_str() {
                        hll.add(&s);
                    } else if let Some(n) = item.as_i64() {
                        hll.add(&n);
                    }
                }
                per_set.push(hll.count());
                merged.merge(&hll).map_err(|e| e.to_string())?;
            }

            Ok(json!({
                "merged_cardinality": merged.count(),
                "per_set_cardinality": per_set,
                "n_sets": sets.len(),
                "precision": precision
            }))
        }
        _ => Err(format!("Unknown hyperloglog operation: {operation}. Use 'estimate' or 'merge'")),
    }
}

// ── ix_pipeline_run placeholder ────────────────────────────

/// Placeholder handler for `ix_pipeline_run`. Real execution lives in
/// `ToolRegistry::run_pipeline` because the pipeline runner needs
/// access to the registry itself (to dispatch child tool calls) and
/// the `Tool::handler` fn-pointer has no such access. The dispatcher
/// in `ToolRegistry::call_with_ctx` intercepts `ix_pipeline_run`
/// before this handler ever runs. This stub exists only so that
/// `tools/list` can expose the tool name and schema.
pub fn pipeline_run_placeholder(_params: Value) -> Result<Value, String> {
    Err(
        "ix_pipeline_run must be invoked via the top-level MCP dispatcher, \
         which routes it to ToolRegistry::run_pipeline. It cannot be called \
         directly as a handler function."
            .to_string(),
    )
}

/// Placeholder handler for `ix_pipeline_compile`. The real execution
/// lives in `ToolRegistry::compile_pipeline` because the compiler
/// needs access to the registry (for the prompt's tool summary and
/// for post-generation validation) and the ServerContext (for MCP
/// sampling). `ToolRegistry::call_with_ctx` intercepts
/// `ix_pipeline_compile` before this handler ever runs.
pub fn pipeline_compile_placeholder(_params: Value) -> Result<Value, String> {
    Err(
        "ix_pipeline_compile must be invoked via the top-level MCP dispatcher, \
         which routes it to ToolRegistry::compile_pipeline. It cannot be called \
         directly as a handler function."
            .to_string(),
    )
}

// ── ix_git_log ─────────────────────────────────────────────

/// P1.1 — shell out to `git log` and return a normalized per-path
/// commit cadence time series. The primary consumer is the
/// adversarial refactor oracle, which previously baked its 90-day
/// commit counts as constants because ix had no git introspection
/// tool of its own.
///
/// Input format:
/// ```json
/// {
///   "path": "crates/ix-agent",
///   "since_days": 90,
///   "bucket": "day" | "week"
/// }
/// ```
///
/// Output:
/// ```json
/// {
///   "path": "crates/ix-agent",
///   "since_days": 90,
///   "bucket": "day",
///   "window_days": 90,
///   "n_buckets": 90,
///   "commits": 89,
///   "series": [0.0, 0.0, ..., 19.0],
///   "dates": ["2026-01-14", ..., "2026-04-13"]
/// }
/// ```
///
/// # Security
///
/// This handler spawns a `git` subprocess. Every argument is passed
/// via `Command::arg()` (NOT `arg_line()` or shell concatenation),
/// so shell metacharacters in `path` cannot escape the argument
/// boundary. In addition we whitelist-validate `path` against the
/// `is_safe_git_path` predicate so even a well-formed argument
/// containing a `.git/hooks/...` style injection vector is rejected.
pub fn git_log(params: Value) -> Result<Value, String> {
    use std::process::Command;

    let path = parse_str(&params, "path")?.to_string();
    if !is_safe_git_path(&path) {
        return Err(format!(
            "ix_git_log: 'path' must be a relative repo-internal path with no '..', \
             absolute prefix, or shell metacharacters; got {path:?}"
        ));
    }

    let since_days = params
        .get("since_days")
        .and_then(|v| v.as_u64())
        .unwrap_or(90);
    if since_days == 0 || since_days > 3650 {
        return Err(format!(
            "ix_git_log: 'since_days' must be in 1..=3650, got {since_days}"
        ));
    }

    let bucket = params
        .get("bucket")
        .and_then(|v| v.as_str())
        .unwrap_or("day");
    if bucket != "day" && bucket != "week" {
        return Err(format!(
            "ix_git_log: 'bucket' must be 'day' or 'week', got {bucket:?}"
        ));
    }

    // Build the argument list. Every arg is a fixed literal or a
    // validated value; there is no string concatenation of untrusted
    // input into a single argument.
    let since_arg = format!("--since={since_days} days ago");
    let output = Command::new("git")
        .arg("log")
        .arg(&since_arg)
        .arg("--format=%ad")
        .arg("--date=format:%Y-%m-%d")
        .arg("--") // end of option parsing — everything after this is a path
        .arg(&path)
        .output()
        .map_err(|e| format!("ix_git_log: failed to spawn git: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ix_git_log: git exited with {:?}: {}",
            output.status.code(),
            stderr.trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let raw_dates: Vec<&str> = stdout
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    let total_commits = raw_dates.len();

    // Bucket the commits into a dense per-day or per-week series.
    let (n_buckets, bucket_size_days) = match bucket {
        "day" => (since_days as usize, 1usize),
        "week" => ((since_days as usize + 6) / 7, 7usize),
        _ => unreachable!(),
    };

    // Today anchors the end of the window. We never call a real
    // date library here — dates from git are strings in
    // %Y-%m-%d and we just subtract whole days from today. The
    // cheapest correct impl: count days via Julian date number.
    let today = today_ymd_epoch_days()
        .ok_or_else(|| "ix_git_log: system clock is before the unix epoch".to_string())?;
    let window_start = today.saturating_sub(since_days as i64 - 1);

    let mut series = vec![0.0_f64; n_buckets];
    let mut dates: Vec<String> = Vec::with_capacity(n_buckets);
    for i in 0..n_buckets {
        let day = window_start + (i * bucket_size_days) as i64;
        dates.push(epoch_days_to_ymd(day));
    }
    for raw in &raw_dates {
        if let Some(day) = ymd_to_epoch_days(raw) {
            if day < window_start || day > today {
                continue;
            }
            let bucket_idx = ((day - window_start) as usize) / bucket_size_days;
            if bucket_idx < n_buckets {
                series[bucket_idx] += 1.0;
            }
        }
    }

    Ok(json!({
        "path": path,
        "since_days": since_days,
        "bucket": bucket,
        "window_days": since_days,
        "n_buckets": n_buckets,
        "commits": total_commits,
        "series": series,
        "dates": dates,
    }))
}

/// Accept only relative paths with no `..` segments, no absolute
/// prefixes, no shell metacharacters, and no null bytes. This is the
/// second layer of defence — the first is passing args through
/// `Command::arg()` instead of shell concatenation — but we keep
/// the whitelist so even a misuse of the handler cannot name a
/// path like `.git/hooks/pre-commit`.
fn is_safe_git_path(path: &str) -> bool {
    if path.is_empty() {
        return false;
    }
    if path.contains('\0') {
        return false;
    }
    if path.starts_with('/') || path.starts_with('\\') {
        return false;
    }
    // Windows drive prefixes.
    if path.len() >= 2 && path.as_bytes()[1] == b':' {
        return false;
    }
    // Shell metacharacters that have no business in a repo path.
    for c in path.chars() {
        if matches!(
            c,
            '|' | '&' | ';' | '<' | '>' | '$' | '`' | '(' | ')' | '{' | '}' | '*' | '?' | '\n' | '\r'
        ) {
            return false;
        }
    }
    // Reject any `..` segment.
    for segment in path.split(|c| c == '/' || c == '\\') {
        if segment == ".." {
            return false;
        }
    }
    true
}

/// Convert today's system time to a count of whole days since
/// 1970-01-01 (the Julian-day-adjacent representation we use for
/// cheap date arithmetic inside `git_log`).
fn today_ymd_epoch_days() -> Option<i64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    Some((now.as_secs() / 86_400) as i64)
}

/// Parse a `YYYY-MM-DD` string to days since epoch. Returns `None`
/// on any parse error — we just drop unparseable git dates, since
/// they would have been skipped by the bucket assignment anyway.
fn ymd_to_epoch_days(s: &str) -> Option<i64> {
    let mut parts = s.split('-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // Rata Die / proleptic Gregorian days since 0001-01-01 via the
    // classic Howard Hinnant formula, then rebase to 1970-01-01
    // (epoch day 0 ≡ rata die 719162).
    let year = if m <= 2 { y - 1 } else { y };
    let era = if year >= 0 { year / 400 } else { (year - 399) / 400 };
    let yoe = (year - era * 400) as i64; // [0, 399]
    let month_adj = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * month_adj + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146097 + doe - 719468)
}

/// Inverse of `ymd_to_epoch_days` — format a day count as
/// `YYYY-MM-DD`. Uses the same Hinnant algorithm in reverse.
fn epoch_days_to_ymd(days: i64) -> String {
    let z = days + 719468;
    let era = if z >= 0 { z / 146097 } else { (z - 146096) / 146097 };
    let doe = (z - era * 146097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ── ix_cargo_deps ──────────────────────────────────────────

/// P1.2 — walk a Rust workspace, parse every `crates/<name>/Cargo.toml`
/// for intra-workspace `ix-*` dependencies, and return a
/// {nodes, edges, n_nodes} structure that `ix_graph` can consume
/// directly.
///
/// Input format:
/// ```json
/// { "workspace_root": "C:\\Users\\spare\\source\\repos\\ix" }
/// ```
/// `workspace_root` is optional; when absent we walk from the CWD.
///
/// Output:
/// ```json
/// {
///   "workspace_root": "<abs path>",
///   "n_nodes": 52,
///   "nodes": [
///     { "id": 0, "name": "ix-agent", "sloc": 11429, "file_count": 21, "dep_count": 36 },
///     ...
///   ],
///   "edges": [ [0, 4, 1.0], [0, 2, 1.0], ... ]
/// }
/// ```
///
/// Deliberately uses a hand-rolled Cargo.toml parser (no `toml`
/// crate) because we only need to extract dep names matching
/// `^ix-[a-z0-9-]+$` inside the `[dependencies]`, `[dev-dependencies]`,
/// and `[build-dependencies]` tables. A full TOML parser would be
/// more robust but carries a new workspace dep for something this
/// narrow.
pub fn cargo_deps(params: Value) -> Result<Value, String> {
    let workspace_root = match params.get("workspace_root").and_then(|v| v.as_str()) {
        Some(s) => std::path::PathBuf::from(s),
        None => std::env::current_dir()
            .map_err(|e| format!("ix_cargo_deps: cwd: {e}"))?,
    };
    let crates_dir = workspace_root.join("crates");
    if !crates_dir.is_dir() {
        return Err(format!(
            "ix_cargo_deps: 'crates' directory not found under {}",
            workspace_root.display()
        ));
    }

    // First pass: enumerate every crate in `crates/<name>`. We only
    // register a node if the directory contains a Cargo.toml so we
    // don't pick up stray folders.
    let mut crate_entries: Vec<(String, std::path::PathBuf)> = Vec::new();
    let read = std::fs::read_dir(&crates_dir)
        .map_err(|e| format!("ix_cargo_deps: read_dir {}: {e}", crates_dir.display()))?;
    for entry in read.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let manifest = path.join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        crate_entries.push((name, manifest));
    }
    crate_entries.sort_by(|a, b| a.0.cmp(&b.0));

    // Assign stable node ids (alphabetical).
    let name_to_id: std::collections::HashMap<String, usize> = crate_entries
        .iter()
        .enumerate()
        .map(|(i, (n, _))| (n.clone(), i))
        .collect();
    let n_nodes = crate_entries.len();

    // Second pass: for each crate, compute (sloc, file_count) from
    // its `src/` tree and parse its Cargo.toml for intra-workspace
    // deps. Only deps whose name matches a directory under `crates/`
    // count as edges — so we correctly include non-`ix-*` workspace
    // members (e.g. `memristive-markov`) and exclude external
    // crates.io deps.
    let known_crates: std::collections::HashSet<&str> =
        name_to_id.keys().map(|s| s.as_str()).collect();

    let mut nodes: Vec<Value> = Vec::with_capacity(n_nodes);
    let mut edges: Vec<Value> = Vec::new();
    for (from_id, (name, manifest)) in crate_entries.iter().enumerate() {
        let src_dir = manifest.parent().unwrap().join("src");
        let (sloc, file_count) = measure_src_tree(&src_dir);

        let manifest_text = std::fs::read_to_string(manifest).map_err(|e| {
            format!(
                "ix_cargo_deps: read {}: {e}",
                manifest.display()
            )
        })?;
        let deps = extract_workspace_deps(&manifest_text, &known_crates);
        let dep_count = deps.len();

        for dep in &deps {
            if let Some(&to_id) = name_to_id.get(dep) {
                // Skip self-references (shouldn't happen but be safe).
                if to_id == from_id {
                    continue;
                }
                edges.push(json!([from_id, to_id, 1.0]));
            }
        }

        nodes.push(json!({
            "id": from_id,
            "name": name,
            "sloc": sloc,
            "file_count": file_count,
            "dep_count": dep_count,
        }));
    }

    Ok(json!({
        "workspace_root": workspace_root.display().to_string(),
        "n_nodes": n_nodes,
        "nodes": nodes,
        "edges": edges,
    }))
}

/// Walk `src/**/*.rs` recursively and return `(total_loc, file_count)`.
/// Non-Rust files are ignored. A missing directory returns `(0, 0)`.
fn measure_src_tree(dir: &std::path::Path) -> (u64, u64) {
    let mut total_loc = 0u64;
    let mut files = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let read = match std::fs::read_dir(&d) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for entry in read.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                files += 1;
                if let Ok(content) = std::fs::read_to_string(&p) {
                    total_loc += content.lines().count() as u64;
                }
            }
        }
    }
    (total_loc, files)
}

/// Extract every dependency name from a Cargo.toml body whose name
/// matches a known workspace crate. Scans the `[dependencies]`,
/// `[dev-dependencies]`, and `[build-dependencies]` tables. Handles
/// both inline-table values (`ix-math = { workspace = true }`) and
/// plain string values (`ix-math = "0.1"`). The package's own name
/// (from `[package]`) is excluded so we never emit self-loops.
///
/// `known_crates` is the set of directory names under `crates/`.
/// Only dep keys that appear in this set are kept; external crates
/// (serde, tokio, …) are silently dropped.
fn extract_workspace_deps(
    toml_body: &str,
    known_crates: &std::collections::HashSet<&str>,
) -> Vec<String> {
    #[derive(PartialEq)]
    enum Section {
        None,
        Package,
        Deps,
    }
    let mut section = Section::None;
    let mut pkg_name: Option<String> = None;
    let mut deps: Vec<String> = Vec::new();

    for raw in toml_body.lines() {
        let line = raw.trim();
        // Strip comments.
        let line = match line.find('#') {
            Some(i) => line[..i].trim(),
            None => line,
        };
        if line.is_empty() {
            continue;
        }
        if let Some(inner) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            let table_name = inner.trim();
            section = match table_name {
                "package" => Section::Package,
                "dependencies" | "dev-dependencies" | "build-dependencies" => Section::Deps,
                _ => Section::None,
            };
            continue;
        }

        match section {
            Section::Package => {
                if let Some(rest) = line.strip_prefix("name") {
                    let rest = rest.trim_start_matches('=').trim();
                    if let Some(n) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                        pkg_name = Some(n.to_string());
                    }
                }
            }
            Section::Deps => {
                // Dep line: `ix-math = ...` or `ix-math.workspace = true`.
                let key = line
                    .split_once('=')
                    .map(|(k, _)| k.trim())
                    .unwrap_or(line);
                // Strip table prefix like `ix-math.workspace` → `ix-math`.
                let key = key.split('.').next().unwrap_or(key);
                if known_crates.contains(key) {
                    deps.push(key.to_string());
                }
            }
            Section::None => {}
        }
    }

    // Filter self-reference and dedupe while preserving order.
    let self_name = pkg_name.unwrap_or_default();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for d in deps {
        if d == self_name {
            continue;
        }
        if seen.insert(d.clone()) {
            out.push(d);
        }
    }
    out
}

// ── ix_pipeline_list ───────────────────────────────────────

/// R1 companion to `ix_pipeline_run`: discover `pipeline.json` specs
/// under a conventional directory (default
/// `examples/canonical-showcase`) and return metadata for each one.
///
/// Input:
/// ```json
/// { "root": "examples/canonical-showcase" }
/// ```
///
/// `root` is optional; when absent the default relative path is used.
/// Relative paths are resolved against the process CWD.
///
/// Output:
/// ```json
/// {
///   "root": "<resolved absolute path>",
///   "pipelines": [
///     {
///       "path": "<abs path to pipeline.json>",
///       "folder": "<containing folder name>",
///       "name": "cost-anomaly-hunter",
///       "description": "...",
///       "version": "1.0",
///       "step_count": 3,
///       "tools": ["ix_stats", "ix_fft", "ix_kmeans"]
///     }
///   ]
/// }
/// ```
pub fn pipeline_list(params: Value) -> Result<Value, String> {
    let root_arg = params
        .get("root")
        .and_then(|v| v.as_str())
        .unwrap_or("examples/canonical-showcase");
    let root = std::path::PathBuf::from(root_arg);
    let root = if root.is_absolute() {
        root
    } else {
        std::env::current_dir()
            .map_err(|e| format!("ix_pipeline_list: cwd: {e}"))?
            .join(root)
    };

    if !root.exists() {
        return Ok(json!({
            "root": root.display().to_string(),
            "pipelines": [],
            "warning": format!("root '{}' does not exist", root.display()),
        }));
    }

    let entries = std::fs::read_dir(&root)
        .map_err(|e| format!("ix_pipeline_list: read_dir {}: {e}", root.display()))?;

    let mut pipelines: Vec<Value> = Vec::new();
    let mut child_dirs: Vec<std::path::PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            child_dirs.push(path);
        }
    }
    child_dirs.sort();

    for dir in child_dirs {
        let spec_path = dir.join("pipeline.json");
        if !spec_path.is_file() {
            continue;
        }
        let raw = match std::fs::read_to_string(&spec_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let parsed: Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let steps = parsed
            .get("steps")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let tools: Vec<String> = steps
            .iter()
            .filter_map(|s| s.get("tool").and_then(|t| t.as_str()).map(String::from))
            .collect();

        pipelines.push(json!({
            "path": spec_path.display().to_string(),
            "folder": dir.file_name().and_then(|n| n.to_str()).unwrap_or_default(),
            "name": parsed.get("name").and_then(|v| v.as_str()).unwrap_or(""),
            "description": parsed.get("description").and_then(|v| v.as_str()).unwrap_or(""),
            "version": parsed.get("version").and_then(|v| v.as_str()).unwrap_or(""),
            "step_count": steps.len(),
            "tools": tools,
        }));
    }

    Ok(json!({
        "root": root.display().to_string(),
        "pipelines": pipelines,
    }))
}

// ── ix_autograd_run ────────────────────────────────────────

/// R7 Week 2: run a differentiable tool via MCP and return forward
/// outputs + per-input gradients in a single call.
///
/// Supported tool names:
/// - `"linear_regression"` — requires `x`, `w`, `b`, `y` in inputs
/// - `"stats_variance"` — requires `x` in inputs
///
/// Input arrays are nested f64 arrays; the shape is inferred from
/// the nesting depth. Scalars use a 1-element 1-D array.
///
/// Returns:
/// ```json
/// {
///   "tool": "linear_regression",
///   "forward": { "y_hat": [[...]], "loss": 0.123 },
///   "gradients": { "x": [[...]], "w": [[...]], "b": [[...]] }
/// }
/// ```
pub fn autograd_run(params: Value) -> Result<Value, String> {
    use ix_autograd::prelude::*;
    use ix_autograd::tools::{linear_regression::LinearRegressionTool, stats_variance::StatsVarianceTool};

    let tool_name = params
        .get("tool")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "ix_autograd_run: missing 'tool'".to_string())?;

    let inputs_json = params
        .get("inputs")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "ix_autograd_run: missing 'inputs' object".to_string())?;

    // Parse each input into an ArrayD<f64> via generic helpers.
    let mut in_map = ValueMap::new();
    for (key, val) in inputs_json {
        let array = parse_array_d(val)
            .map_err(|e| format!("ix_autograd_run: input '{key}': {e}"))?;
        in_map.insert(key.clone(), Tensor::from_array_with_grad(array));
    }

    let mut ctx = DiffContext::new(ExecutionMode::Train);

    let (forward_map, grad_map) = match tool_name {
        "linear_regression" => {
            let tool = LinearRegressionTool;
            let fwd = tool
                .forward(&mut ctx, &in_map)
                .map_err(|e| format!("linear_regression forward: {e}"))?;
            let dummy = ValueMap::new();
            let grads = tool
                .backward(&mut ctx, &dummy)
                .map_err(|e| format!("linear_regression backward: {e}"))?;
            (fwd, grads)
        }
        "stats_variance" => {
            let tool = StatsVarianceTool;
            let fwd = tool
                .forward(&mut ctx, &in_map)
                .map_err(|e| format!("stats_variance forward: {e}"))?;
            let dummy = ValueMap::new();
            let grads = tool
                .backward(&mut ctx, &dummy)
                .map_err(|e| format!("stats_variance backward: {e}"))?;
            (fwd, grads)
        }
        other => {
            return Err(format!(
                "ix_autograd_run: unknown tool '{other}'. Supported: \
                 linear_regression, stats_variance"
            ))
        }
    };

    let forward_serialized = serialize_value_map(&forward_map);
    let grads_serialized = serialize_value_map(&grad_map);

    Ok(json!({
        "tool": tool_name,
        "forward": forward_serialized,
        "gradients": grads_serialized,
    }))
}

/// Parse an arbitrary nested JSON array (or scalar) into a dynamic
/// `ndarray::ArrayD<f64>`. The shape is inferred from the nesting.
fn parse_array_d(v: &Value) -> Result<ndarray::ArrayD<f64>, String> {
    use ndarray::{Array, IxDyn};
    match v {
        Value::Number(n) => {
            let x = n.as_f64().ok_or("non-f64 number")?;
            Ok(Array::from_elem(IxDyn(&[]), x))
        }
        Value::Array(a) => {
            if a.is_empty() {
                return Ok(Array::from_shape_vec(IxDyn(&[0]), vec![]).unwrap());
            }
            let first_child_is_array = matches!(a[0], Value::Array(_));
            if !first_child_is_array {
                let values: Vec<f64> = a
                    .iter()
                    .map(|x| x.as_f64().ok_or_else(|| "non-f64 element".to_string()))
                    .collect::<Result<_, _>>()?;
                Array::from_shape_vec(IxDyn(&[values.len()]), values)
                    .map_err(|e: ndarray::ShapeError| e.to_string())
            } else {
                let rows = a.len();
                let first_row = a[0].as_array().ok_or("first row not array")?;
                let cols = first_row.len();
                let mut flat: Vec<f64> = Vec::with_capacity(rows * cols);
                for (i, row) in a.iter().enumerate() {
                    let row_arr = row
                        .as_array()
                        .ok_or_else(|| format!("row {i} not array"))?;
                    if row_arr.len() != cols {
                        return Err(format!(
                            "row {i} has {} cols, expected {cols}",
                            row_arr.len()
                        ));
                    }
                    for x in row_arr {
                        flat.push(x.as_f64().ok_or_else(|| {
                            format!("row {i} contains non-f64 element")
                        })?);
                    }
                }
                Array::from_shape_vec(IxDyn(&[rows, cols]), flat)
                    .map_err(|e: ndarray::ShapeError| e.to_string())
            }
        }
        _ => Err("expected array or number".into()),
    }
}

/// Turn a `ValueMap` of tensors into a serde_json::Value object.
fn serialize_value_map(map: &ix_autograd::tool::ValueMap) -> Value {
    let mut out = serde_json::Map::new();
    for (k, tensor) in map {
        let arr = tensor.as_f64();
        let shape = arr.shape().to_vec();
        if shape.is_empty() {
            // rank-0 scalar
            out.insert(k.clone(), json!(arr.iter().next().copied().unwrap_or(0.0)));
        } else if shape.len() == 1 {
            let v: Vec<f64> = arr.iter().copied().collect();
            out.insert(k.clone(), json!(v));
        } else if shape.len() == 2 {
            let rows = shape[0];
            let cols = shape[1];
            let mut rows_out: Vec<Vec<f64>> = Vec::with_capacity(rows);
            let data: Vec<f64> = arr.iter().copied().collect();
            for r in 0..rows {
                rows_out.push(data[r * cols..(r + 1) * cols].to_vec());
            }
            out.insert(k.clone(), json!(rows_out));
        } else {
            // Higher-rank: flatten + emit shape.
            let flat: Vec<f64> = arr.iter().copied().collect();
            out.insert(
                k.clone(),
                json!({ "shape": shape, "data": flat }),
            );
        }
    }
    Value::Object(out)
}

// ── ix_pipeline_exec ───────────────────────────────────────

pub fn pipeline_exec(params: Value) -> Result<Value, String> {
    use ix_pipeline::dag::Dag;

    let operation = parse_str(&params, "operation")?;

    match operation {
        "info" => {
            // Build a DAG from step definitions and return structure info
            let steps = params.get("steps").and_then(|v| v.as_array())
                .ok_or_else(|| "Missing 'steps' array".to_string())?;

            let mut dag: Dag<String> = Dag::new();
            for step in steps {
                let id = step.get("id").and_then(|v| v.as_str())
                    .ok_or("Each step needs an 'id'")?;
                let desc = step.get("description").and_then(|v| v.as_str()).unwrap_or("");
                dag.add_node(id, desc.to_string()).map_err(|e| e.to_string())?;
            }

            // Add edges from dependencies
            for step in steps {
                let id = step.get("id").and_then(|v| v.as_str()).unwrap();
                if let Some(deps) = step.get("depends_on").and_then(|v| v.as_array()) {
                    for dep in deps {
                        if let Some(dep_id) = dep.as_str() {
                            dag.add_edge(dep_id, id).map_err(|e| e.to_string())?;
                        }
                    }
                }
            }

            let levels = dag.parallel_levels();
            let level_ids: Vec<Vec<&str>> = levels.iter()
                .map(|level| level.iter().map(|id| id.as_str()).collect())
                .collect();

            Ok(json!({
                "node_count": dag.node_count(),
                "edge_count": dag.edge_count(),
                "roots": dag.roots(),
                "leaves": dag.leaves(),
                "topological_order": dag.topological_sort(),
                "parallel_levels": level_ids,
                "max_parallelism": levels.iter().map(|l| l.len()).max().unwrap_or(0)
            }))
        }
        _ => Err(format!("Unknown pipeline operation: {operation}. Use 'info'")),
    }
}

// ── ix_cache ───────────────────────────────────────────────

pub fn cache_op(params: Value) -> Result<Value, String> {
    let operation = parse_str(&params, "operation")?;
    let cache = global_cache();

    match operation {
        "set" => {
            let key = parse_str(&params, "key")?;
            let value = params
                .get("value")
                .ok_or_else(|| "Missing field 'value'".to_string())?;
            cache.set(key, value);
            Ok(json!({ "ok": true, "key": key }))
        }
        "get" => {
            let key = parse_str(&params, "key")?;
            let value: Option<Value> = cache.get(key);
            match value {
                Some(v) => Ok(json!({ "key": key, "value": v, "found": true })),
                None => Ok(json!({ "key": key, "value": null, "found": false })),
            }
        }
        "delete" => {
            let key = parse_str(&params, "key")?;
            let deleted = cache.delete(key);
            Ok(json!({ "key": key, "deleted": deleted }))
        }
        "keys" => {
            let pattern = params
                .get("key")
                .and_then(|v| v.as_str())
                .unwrap_or("*");
            let keys = cache.keys(pattern);
            Ok(json!({ "keys": keys, "count": keys.len() }))
        }
        _ => Err(format!("Unknown cache operation: {}", operation)),
    }
}

// ── governance helpers ─────────────────────────────────────

fn workspace_root() -> std::path::PathBuf {
    // Try CARGO_MANIFEST_DIR (available during `cargo run`)
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return std::path::PathBuf::from(manifest).join("../..");
    }
    // Try IX_ROOT env var (for standalone binary)
    if let Ok(root) = std::env::var("IX_ROOT") {
        return std::path::PathBuf::from(root);
    }
    // Try to find workspace root by looking for Cargo.toml with [workspace]
    let mut dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    for _ in 0..5 {
        if dir.join("Cargo.toml").exists() && dir.join("governance").exists() {
            return dir;
        }
        if !dir.pop() { break; }
    }
    std::path::PathBuf::from(".")
}

fn governance_dir() -> std::path::PathBuf {
    workspace_root().join("governance/demerzel")
}

// ── ix_governance_check ────────────────────────────────────

pub fn governance_check(params: Value) -> Result<Value, String> {
    let action = parse_str(&params, "action")?;
    let _context = params.get("context").and_then(|v| v.as_str()).unwrap_or("");

    let constitution_path = governance_dir().join("constitutions/default.constitution.md");
    let constitution = ix_governance::Constitution::load(&constitution_path)
        .map_err(|e| format!("Failed to load constitution: {}", e))?;

    let result = constitution.check_action(action);

    let mut response = json!({
        "compliant": result.compliant,
        "relevant_articles": result.relevant_articles.iter().map(|a| json!({
            "number": a.number,
            "name": a.name,
            "relevance": a.relevance,
        })).collect::<Vec<_>>(),
        "warnings": result.warnings,
        "constitution_version": constitution.version,
        "total_articles": constitution.articles.len(),
    });

    // R2 Phase 2: pipeline lineage audit trail. When the caller passes a
    // `lineage` map emitted by `ix_pipeline_run`, summarise it alongside
    // the compliance verdict so auditors can see which upstream
    // assets/steps contributed to the decision. We don't change the
    // verdict based on lineage — that's a Phase 3 concern. This hook
    // exists so the lineage DAG is surfaced in governance output and
    // can be walked back to concrete cache keys.
    if let Some(lineage) = params.get("lineage").and_then(|v| v.as_object()) {
        let mut summary: Vec<Value> = Vec::with_capacity(lineage.len());
        for (step_id, entry) in lineage {
            summary.push(json!({
                "step_id": step_id,
                "tool": entry.get("tool").cloned().unwrap_or(Value::Null),
                "asset_name": entry.get("asset_name").cloned().unwrap_or(Value::Null),
                "cache_key": entry.get("cache_key").cloned().unwrap_or(Value::Null),
                "depends_on": entry.get("depends_on").cloned().unwrap_or_else(|| json!([])),
                "upstream_cache_keys": entry
                    .get("upstream_cache_keys")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            }));
        }
        response["lineage_audit"] = json!({
            "step_count": summary.len(),
            "steps": summary,
        });
    }

    Ok(response)
}

// ── ix_governance_persona ──────────────────────────────────

pub fn governance_persona(params: Value) -> Result<Value, String> {
    let name = parse_str(&params, "persona")?;

    let personas_dir = governance_dir().join("personas");
    let persona = ix_governance::Persona::load_by_name(&personas_dir, name)
        .map_err(|e| format!("Failed to load persona '{}': {}", name, e))?;

    let mut result = json!({
        "name": persona.name,
        "version": persona.version,
        "description": persona.description,
        "role": persona.role,
        "domain": persona.domain,
        "capabilities": persona.capabilities,
        "constraints": persona.constraints,
        "voice": {
            "tone": persona.voice.tone,
            "verbosity": persona.voice.verbosity,
            "style": persona.voice.style,
        },
    });

    if let Some(patterns) = &persona.interaction_patterns {
        result["interaction_patterns"] = json!({
            "with_humans": patterns.with_humans,
            "with_agents": patterns.with_agents,
        });
    }

    if let Some(prov) = &persona.provenance {
        result["provenance"] = json!({
            "source": prov.source,
            "extraction_date": prov.extraction_date,
            "archetype": prov.archetype,
        });
    }

    Ok(result)
}

// ── ix_governance_belief ───────────────────────────────────

pub fn governance_belief(params: Value) -> Result<Value, String> {
    let operation = parse_str(&params, "operation")?;
    let proposition = parse_str(&params, "proposition")?;

    fn parse_truth_value(s: &str) -> Result<ix_governance::TruthValue, String> {
        match s {
            "T" => Ok(ix_governance::TruthValue::True),
            "F" => Ok(ix_governance::TruthValue::False),
            "U" => Ok(ix_governance::TruthValue::Unknown),
            "C" => Ok(ix_governance::TruthValue::Contradictory),
            _ => Err(format!("Invalid truth value '{}': use T, F, U, or C", s)),
        }
    }

    let tv_str = params.get("truth_value").and_then(|v| v.as_str()).unwrap_or("U");
    let tv = parse_truth_value(tv_str)?;
    let confidence = params.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5);

    let supporting: Vec<String> = params
        .get("supporting")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let contradicting: Vec<String> = params
        .get("contradicting")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    match operation {
        "create" | "update" => {
            let mut belief = ix_governance::BeliefState::new(proposition, tv, confidence);
            for claim in &supporting {
                belief.add_supporting(ix_governance::EvidenceItem {
                    source: "user".to_string(),
                    claim: claim.clone(),
                });
            }
            for claim in &contradicting {
                belief.add_contradicting(ix_governance::EvidenceItem {
                    source: "user".to_string(),
                    claim: claim.clone(),
                });
            }

            Ok(json!({
                "proposition": belief.proposition,
                "truth_value": format!("{}", belief.truth_value),
                "confidence": belief.confidence,
                "supporting_count": belief.supporting.len(),
                "contradicting_count": belief.contradicting.len(),
                "resolved_action": format!("{:?}", belief.resolve()),
            }))
        }
        "resolve" => {
            let belief = ix_governance::BeliefState::new(proposition, tv, confidence);
            let action = belief.resolve();
            Ok(json!({
                "proposition": belief.proposition,
                "truth_value": format!("{}", belief.truth_value),
                "resolved_action": format!("{:?}", action),
                "explanation": match action {
                    ix_governance::ResolvedAction::Proceed => "Belief is verified — safe to proceed",
                    ix_governance::ResolvedAction::DoNotProceed => "Belief is refuted — do not proceed",
                    ix_governance::ResolvedAction::GatherEvidence => "Insufficient evidence — gather more before deciding",
                    ix_governance::ResolvedAction::Escalate => "Contradictory evidence — escalate to human",
                },
            }))
        }
        _ => Err(format!("Unknown belief operation: {}. Use 'create', 'update', or 'resolve'", operation)),
    }
}

// ── ix_governance_policy ───────────────────────────────────

pub fn governance_policy(params: Value) -> Result<Value, String> {
    let policy_name = parse_str(&params, "policy")?;
    let query = params.get("query").and_then(|v| v.as_str());

    let policies_dir = governance_dir().join("policies");

    let filename = match policy_name {
        "alignment" => "alignment-policy.yaml",
        "rollback" => "rollback-policy.yaml",
        "self-modification" => "self-modification-policy.yaml",
        _ => return Err(format!("Unknown policy: {}. Use 'alignment', 'rollback', or 'self-modification'", policy_name)),
    };

    let path = policies_dir.join(filename);

    // For alignment, use strongly-typed parsing
    if policy_name == "alignment" {
        let ap = ix_governance::AlignmentPolicy::load(&path)
            .map_err(|e| format!("Failed to load alignment policy: {}", e))?;

        return match query {
            Some("thresholds") => Ok(json!({
                "policy": "alignment",
                "thresholds": {
                    "proceed_autonomously": ap.confidence_thresholds.proceed_autonomously,
                    "proceed_with_note": ap.confidence_thresholds.proceed_with_note,
                    "ask_for_confirmation": ap.confidence_thresholds.ask_for_confirmation,
                    "escalate_to_human": ap.confidence_thresholds.escalate_to_human,
                },
            })),
            Some("triggers") => Ok(json!({
                "policy": "alignment",
                "escalation_triggers": ap.escalation_triggers,
            })),
            _ => Ok(json!({
                "policy": "alignment",
                "name": ap.name,
                "version": ap.version,
                "description": ap.description,
                "thresholds": {
                    "proceed_autonomously": ap.confidence_thresholds.proceed_autonomously,
                    "proceed_with_note": ap.confidence_thresholds.proceed_with_note,
                    "ask_for_confirmation": ap.confidence_thresholds.ask_for_confirmation,
                    "escalate_to_human": ap.confidence_thresholds.escalate_to_human,
                },
                "escalation_triggers": ap.escalation_triggers,
            })),
        };
    }

    // For other policies, use generic parsing
    let policy = ix_governance::Policy::load(&path)
        .map_err(|e| format!("Failed to load policy '{}': {}", policy_name, e))?;

    match query {
        Some("thresholds") | Some("triggers") | Some("allowed") => {
            // Return the extra fields which contain policy-specific data
            Ok(json!({
                "policy": policy_name,
                "name": policy.name,
                "version": policy.version,
                "query": query,
                "data": policy.extra,
            }))
        }
        _ => Ok(json!({
            "policy": policy_name,
            "name": policy.name,
            "version": policy.version,
            "description": policy.description,
            "data": policy.extra,
        })),
    }
}

// ── ix_federation_discover ─────────────────────────────────

pub fn federation_discover(params: Value) -> Result<Value, String> {
    let domain_filter = params.get("domain").and_then(|v| v.as_str());
    let query_filter = params.get("query").and_then(|v| v.as_str());

    let registry_path = workspace_root().join("governance/demerzel/schemas/capability-registry.json");
    let content = std::fs::read_to_string(&registry_path)
        .map_err(|e| format!("Failed to read capability registry: {}", e))?;
    let registry: Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse capability registry: {}", e))?;

    let repos = registry.get("repos").and_then(|v| v.as_object())
        .ok_or_else(|| "Invalid registry: missing 'repos'".to_string())?;

    let mut results = Vec::new();

    for (repo_name, repo_info) in repos {
        let description = repo_info.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let domains: Vec<&str> = repo_info.get("domains")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        // Filter by domain
        if let Some(d) = domain_filter {
            if !domains.iter().any(|dom| dom.contains(d)) {
                continue;
            }
        }

        let tools = repo_info.get("tools").and_then(|v| v.as_object());

        // Filter by query (search in tool names)
        let mut matching_tools: Value = json!(tools);
        if let Some(q) = query_filter {
            let q_lower = q.to_lowercase();
            if let Some(tool_map) = tools {
                let filtered: serde_json::Map<String, Value> = tool_map.iter()
                    .filter(|(cat, tools_arr)| {
                        cat.contains(&q_lower) || tools_arr.as_array()
                            .map(|arr| arr.iter().any(|t| {
                                t.as_str().map(|s| s.to_lowercase().contains(&q_lower)).unwrap_or(false)
                            }))
                            .unwrap_or(false)
                    })
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                if filtered.is_empty() {
                    continue;
                }
                matching_tools = Value::Object(filtered);
            }
        }

        results.push(json!({
            "repo": repo_name,
            "description": description,
            "domains": domains,
            "tools": matching_tools,
        }));
    }

    let mut response = json!({
        "results": results,
        "total_repos": results.len(),
    });

    // Include roadblock resolution strategies if present
    if domain_filter.is_none() && query_filter.is_none() {
        if let Some(strategies) = registry.get("roadblock_resolution") {
            response["roadblock_resolution"] = strategies.clone();
        }
    }

    Ok(response)
}

// ── Trace ingest ─────────────────────────────────────────────────

/// Ingest GA traces from a directory and return statistics.
///
/// Params (all optional):
/// - `dir`: path to trace directory (default: `~/.ga/traces/`)
pub fn trace_ingest(params: Value) -> Result<Value, String> {
    use ix_io::trace_bridge;
    use std::path::PathBuf;

    let dir = params
        .get("dir")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(trace_bridge::default_trace_dir);

    if !dir.exists() {
        return Ok(json!({
            "error": format!("Trace directory does not exist: {}", dir.display()),
            "hint": "Create ~/.ga/traces/ and place JSON trace files there, or pass a 'dir' parameter."
        }));
    }

    let traces = trace_bridge::load_traces(&dir).map_err(|e| format!("Failed to load traces: {e}"))?;

    if traces.is_empty() {
        return Ok(json!({
            "total_traces": 0,
            "message": "No valid trace files found in directory."
        }));
    }

    let stats = trace_bridge::compute_stats(&traces);
    let csv_rows = trace_bridge::traces_to_csv_rows(&traces);

    Ok(json!({
        "total_traces": stats.total_traces,
        "success_count": stats.success_count,
        "failure_count": stats.failure_count,
        "avg_duration_ms": stats.avg_duration_ms,
        "p50_duration_ms": stats.p50_duration_ms,
        "p95_duration_ms": stats.p95_duration_ms,
        "event_type_counts": stats.event_type_counts,
        "csv_row_count": csv_rows.len() - 1,
        "csv_preview": csv_rows.iter().take(6).collect::<Vec<_>>(),
    }))
}

// ── ix_fuzzy_eval ─────────────────────────────────────────────
//
// Primitive #5: fuzzy distribution operations. Takes a hexavalent
// distribution and an operation name and returns the result plus
// derived diagnostics (argmax, escalation flag, sharpen result).
//
// Params:
//   distribution: { "T": f, "P": f, "U": f, "D": f, "F": f, "C": f }
//     — every field optional; missing variants default to 0.0
//   operation:   "info" | "not" | "and" | "or"
//   other:       another distribution (required for "and" and "or")
pub fn fuzzy_eval(params: Value) -> Result<Value, String> {
    use ix_fuzzy::hexavalent::{
        escalation_triggered, hexavalent_argmax, hexavalent_from_tpudfc, hexavalent_not,
        try_sharpen, ESCALATION_THRESHOLD, SHARPEN_THRESHOLD,
    };
    use ix_fuzzy::HexavalentDistribution;

    fn parse_dist(v: &Value) -> Result<HexavalentDistribution, String> {
        let get = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0);
        hexavalent_from_tpudfc(
            get("T"),
            get("P"),
            get("U"),
            get("D"),
            get("F"),
            get("C"),
        )
        .map_err(|e| format!("invalid distribution: {e}"))
    }

    let op = params
        .get("operation")
        .and_then(|v| v.as_str())
        .unwrap_or("info");

    let dist_value = params
        .get("distribution")
        .ok_or_else(|| "'distribution' is required".to_string())?;
    let dist = parse_dist(dist_value)?;

    let result: HexavalentDistribution = match op {
        "info" => dist.clone(),
        "not" => hexavalent_not(&dist).map_err(|e| format!("hexavalent_not: {e}"))?,
        "and" => {
            let other = params
                .get("other")
                .ok_or_else(|| "'other' required for 'and'".to_string())?;
            let b = parse_dist(other)?;
            dist.and(&b).map_err(|e| format!("and: {e}"))?
        }
        "or" => {
            let other = params
                .get("other")
                .ok_or_else(|| "'other' required for 'or'".to_string())?;
            let b = parse_dist(other)?;
            dist.or(&b).map_err(|e| format!("or: {e}"))?
        }
        other => return Err(format!("unknown operation: {other}")),
    };

    let argmax = hexavalent_argmax(&result);
    let argmax_mass = result.get(&argmax);
    let escalation = escalation_triggered(&result);
    let sharpen = try_sharpen(&result);

    Ok(json!({
        "distribution": result,
        "argmax": argmax.to_string(),
        "argmax_mass": argmax_mass,
        "escalation_triggered": escalation,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "sharpen": sharpen.map(|v| v.to_string()),
        "sharpen_threshold": SHARPEN_THRESHOLD,
    }))
}

// ── ix_session_flywheel_export ────────────────────────────────
//
// Primitive #6: the trace flywheel. Converts a persisted
// `ix_session::SessionLog` into a GA-flavored Trace JSON file that
// `ix_trace_ingest` can immediately consume, closing the
// self-improvement loop.
//
// Params:
// - `session_log` (required): path to the JSONL session log file
// - `trace_dir`   (optional): destination directory for the Trace
//                             JSON file. Defaults to
//                             `ix_io::trace_bridge::default_trace_dir()`
//                             (typically `~/.ga/traces`).
// - `trace_id`    (optional): explicit trace id; defaults to the
//                             session log filename stem.
pub fn session_flywheel_export(params: Value) -> Result<Value, String> {
    use crate::flywheel;
    use ix_session::SessionLog;
    use std::path::PathBuf;

    let log_path = params
        .get("session_log")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "'session_log' is required".to_string())?;
    let trace_dir: PathBuf = params
        .get("trace_dir")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(ix_io::trace_bridge::default_trace_dir);
    let trace_id = params
        .get("trace_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let log = SessionLog::open(log_path).map_err(|e| format!("open session log: {e}"))?;
    let written = flywheel::export_session_to_trace_dir(&log, &trace_dir, trace_id)
        .map_err(|e| format!("export trace: {e}"))?;

    let trace_count = log.events()
        .map(|it| it.count())
        .unwrap_or(0);

    Ok(json!({
        "written": written.display().to_string(),
        "trace_dir": trace_dir.display().to_string(),
        "event_count": trace_count,
    }))
}

// ── ix_ml_pipeline ────────────────────────────────────────────

pub fn ml_pipeline(params: Value) -> Result<Value, String> {
    let config: crate::ml_pipeline::PipelineConfig = serde_json::from_value(params)
        .map_err(|e| format!("Invalid pipeline config: {}", e))?;
    crate::ml_pipeline::run_pipeline(config)
}

// ── ix_ml_predict ─────────────────────────────────────────────

pub fn ml_predict(params: Value) -> Result<Value, String> {
    let key = parse_str(&params, "persist_key")?;
    let data = parse_f64_matrix(&params, "data")?;
    crate::ml_pipeline::run_predict(key, &data)
}

// ── ix_code_analyze ─────────────────────────────────────────────────

pub fn code_analyze(params: Value) -> Result<Value, String> {
    use ix_code::analyze::{Language, analyze_source, analyze_file};
    use std::path::Path;

    // Option 1: analyze a file by path
    if let Some(path_str) = params.get("path").and_then(|v| v.as_str()) {
        let path = Path::new(path_str);
        let metrics = analyze_file(path)
            .ok_or_else(|| format!("Could not analyze file: {} (unsupported language or read error)", path_str))?;
        return Ok(serde_json::to_value(&metrics).unwrap());
    }

    // Option 2: analyze source code string
    let source = params.get("source").and_then(|v| v.as_str())
        .ok_or_else(|| "Either 'source' or 'path' is required".to_string())?;
    let lang_str = params.get("language").and_then(|v| v.as_str())
        .ok_or_else(|| "'language' is required when using 'source'".to_string())?;

    let lang = Language::from_extension(lang_str)
        .or(match lang_str {
            "rust" => Some(Language::Rust),
            "python" => Some(Language::Python),
            "javascript" => Some(Language::JavaScript),
            "typescript" => Some(Language::TypeScript),
            "cpp" | "c" => Some(Language::Cpp),
            "java" => Some(Language::Java),
            "go" => Some(Language::Go),
            "csharp" => Some(Language::CSharp),
            "fsharp" => Some(Language::FSharp),
            "php" => Some(Language::Php),
            "ruby" => Some(Language::Ruby),
            _ => None,
        })
        .ok_or_else(|| format!("Unsupported language: {}", lang_str))?;

    let path = Path::new(match lang {
        Language::Rust => "input.rs",
        Language::Python => "input.py",
        Language::JavaScript => "input.js",
        Language::TypeScript => "input.ts",
        Language::Cpp => "input.cpp",
        Language::Java => "input.java",
        Language::Go => "input.go",
        Language::CSharp => "input.cs",
        Language::FSharp => "input.fs",
        Language::Php => "input.php",
        Language::Ruby => "input.rb",
    });

    let metrics = analyze_source(source, lang, path);

    // Add feature vector info
    let mut result = serde_json::to_value(&metrics).unwrap();
    result["feature_names"] = json!(ix_code::metrics::CodeMetrics::feature_names());
    result["file_features"] = json!(metrics.file_scope.to_features().to_vec());
    if !metrics.functions.is_empty() {
        let fn_features: Vec<Vec<f64>> = metrics.functions.iter()
            .map(|f| f.to_features().to_vec())
            .collect();
        result["function_features"] = json!(fn_features);
    }

    Ok(result)
}

// ── ix_tars_bridge ──────────────────────────────────────────────────

pub fn tars_bridge(params: Value) -> Result<Value, String> {
    let action = parse_str(&params, "action")?;

    match action {
        "prepare_traces" => {
            use ix_io::trace_bridge;
            use std::path::PathBuf;

            let dir = params.get("trace_dir")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
                .unwrap_or_else(trace_bridge::default_trace_dir);

            if !dir.exists() {
                return Ok(json!({
                    "error": "Trace directory not found",
                    "dir": dir.display().to_string(),
                    "hint": "Create ~/.ga/traces/ and place trace JSON files there"
                }));
            }

            let traces = trace_bridge::load_traces(&dir)
                .map_err(|e| format!("Failed to load traces: {e}"))?;
            let stats = trace_bridge::compute_stats(&traces);

            Ok(json!({
                "action": "prepare_traces",
                "tars_tool": "ingest_ga_traces",
                "payload": {
                    "Count": stats.total_traces,
                    "MinOccurrences": params.get("min_frequency").and_then(|v| v.as_i64()).unwrap_or(3)
                },
                "stats": {
                    "total_traces": stats.total_traces,
                    "success_count": stats.success_count,
                    "failure_count": stats.failure_count,
                    "avg_duration_ms": stats.avg_duration_ms,
                    "p95_duration_ms": stats.p95_duration_ms,
                    "event_types": stats.event_type_counts
                },
                "instruction": "Call TARS tool 'ingest_ga_traces' with the payload above to trigger pattern promotion"
            }))
        }
        "prepare_patterns" => {
            Ok(json!({
                "action": "prepare_patterns",
                "description": "Prepare pattern data for TARS promotion pipeline",
                "workflow": [
                    {"step": 1, "ix_tool": "ix_grammar_weights", "description": "Get current grammar rules with Bayesian weights"},
                    {"step": 2, "ix_tool": "ix_trace_ingest", "description": "Analyze traces to find recurring tool-call sequences"},
                    {"step": 3, "tars_tool": "run_promotion_pipeline", "description": "Run 7-step promotion (Inspect→Extract→Classify→Propose→Validate→Persist→Govern)"},
                    {"step": 4, "tars_tool": "promotion_index", "description": "View ranked promotion results"}
                ],
                "min_frequency": params.get("min_frequency").and_then(|v| v.as_i64()).unwrap_or(3),
                "instruction": "First call ix_grammar_weights to get current state, then call TARS run_promotion_pipeline"
            }))
        }
        "export_grammar" => {
            Ok(json!({
                "action": "export_grammar",
                "description": "Export ix grammar state for TARS synchronization",
                "workflow": [
                    {"step": 1, "ix_tool": "ix_grammar_weights", "description": "Get current Bayesian-weighted rules from ix"},
                    {"step": 2, "tars_tool": "grammar_weights", "description": "View current TARS grammar weights"},
                    {"step": 3, "tars_tool": "grammar_update", "description": "Update TARS rules with ix weights (per-rule: PatternId + Success)"}
                ],
                "tars_grammar_tools": ["grammar_weights", "grammar_update", "grammar_evolve", "grammar_search"],
                "instruction": "Call ix_grammar_weights first, then sync to TARS via grammar_update for each rule"
            }))
        }
        _ => Err(format!("Unknown tars_bridge action: {}. Use: prepare_traces, prepare_patterns, export_grammar", action)),
    }
}

// ── ix_ga_bridge ────────────────────────────────────────────────────

pub fn ga_bridge(params: Value) -> Result<Value, String> {
    let action = parse_str(&params, "action")?;

    match action {
        "chord_features" => {
            let chords = params.get("chords")
                .and_then(|v| v.as_array())
                .ok_or("'chords' array required for chord_features")?;

            Ok(json!({
                "action": "chord_features",
                "description": "Convert GA chord data to ML feature vectors",
                "workflow": [
                    {"step": 1, "tool": "GaParseChord", "description": "Parse each chord symbol to get intervals, root, quality"},
                    {"step": 2, "tool": "GaChordIntervals", "description": "Get interval names (P1, m3, P5, m7)"},
                    {"step": 3, "tool": "GaChordToSet", "description": "Get pitch-class set, ICV, prime form for atonal features"},
                    {"step": 4, "tool": "ix_stats", "description": "Compute statistics on interval vectors"},
                    {"step": 5, "tool": "ix_kmeans", "description": "Cluster chords by interval/ICV similarity"}
                ],
                "feature_encoding": {
                    "interval_vector": "12-bit binary pitch class presence (e.g. C=1,Db=0,D=0,Eb=1,E=0,...)",
                    "icv": "6-element interval class vector from GaChordToSet",
                    "quality_onehot": "One-hot: major=0, minor=1, dim=2, aug=3, dom7=4, maj7=5, min7=6",
                    "root_chromatic": "Root note as chromatic index 0-11 (C=0, C#=1, ..., B=11)"
                },
                "chords_to_analyze": chords,
                "ga_tools_needed": ["GaParseChord", "GaChordIntervals", "GaChordToSet"],
                "ix_tools_needed": ["ix_stats", "ix_kmeans", "ix_ml_pipeline"]
            }))
        }
        "progression_features" => {
            let progression = params.get("progression")
                .and_then(|v| v.as_str())
                .unwrap_or("C Am F G");

            Ok(json!({
                "action": "progression_features",
                "description": "Convert chord progression to ML feature matrix",
                "workflow": [
                    {"step": 1, "tool": "GaAnalyzeProgression", "description": "Detect key, annotate with Roman numerals"},
                    {"step": 2, "tool": "GaCommonTones", "description": "Compute common tones between adjacent chords (voice-leading cost)"},
                    {"step": 3, "tool": "GaChordSubstitutions", "description": "Find possible substitutions for style analysis"},
                    {"step": 4, "tool": "ix_ml_pipeline", "description": "Train classifier on progression features (style detection)"}
                ],
                "feature_encoding": {
                    "roman_numerals": "Sequence of Roman numeral degrees as integers (I=1, ii=2, ...)",
                    "common_tone_count": "Number of common tones between adjacent chords",
                    "root_motion": "Chromatic interval between successive roots",
                    "harmonic_tension": "Sum of tritone intervals in progression"
                },
                "progression": progression,
                "ga_tools_needed": ["GaAnalyzeProgression", "GaCommonTones", "GaChordSubstitutions"],
                "ix_tools_needed": ["ix_stats", "ix_ml_pipeline"]
            }))
        }
        "scale_features" => {
            Ok(json!({
                "action": "scale_features",
                "description": "Convert scale data to ML feature vectors",
                "workflow": [
                    {"step": 1, "tool": "GetAvailableScales", "description": "List all scales with binary IDs"},
                    {"step": 2, "tool": "GaScaleById", "description": "Get scale details by 12-bit pitch-class bitmask"},
                    {"step": 3, "tool": "ix_distance", "description": "Compute Hamming distance between scale bitmasks"},
                    {"step": 4, "tool": "ix_kmeans", "description": "Cluster scales by pitch-class similarity"}
                ],
                "feature_encoding": {
                    "pitch_class_set": "12-bit binary vector (1=note present, 0=absent)",
                    "interval_pattern": "Sequence of semitone intervals between scale degrees",
                    "cardinality": "Number of notes in the scale (5=pentatonic, 7=diatonic, etc.)"
                },
                "ga_tools_needed": ["GetAvailableScales", "GaScaleById", "GetScaleDegrees"],
                "ix_tools_needed": ["ix_distance", "ix_kmeans", "ix_ml_pipeline"]
            }))
        }
        "workflow_guide" => {
            Ok(json!({
                "action": "workflow_guide",
                "description": "Complete guide to GA→ix federation workflows",
                "workflows": {
                    "chord_clustering": {
                        "description": "Cluster chords by timbral/harmonic similarity",
                        "steps": "GaParseChord → GaChordToSet → ix_kmeans",
                        "use_case": "Find harmonically similar chord substitutions"
                    },
                    "style_classification": {
                        "description": "Classify chord progressions by musical style",
                        "steps": "GaAnalyzeProgression → ix_ml_pipeline (classify)",
                        "use_case": "Detect jazz vs pop vs classical progressions"
                    },
                    "scale_recommendation": {
                        "description": "Recommend scales for improvisation over a progression",
                        "steps": "GaAnalyzeProgression → GaArpeggioSuggestions → ix_supervised",
                        "use_case": "Suggest optimal scales for each chord in a progression"
                    },
                    "harmonic_complexity_analysis": {
                        "description": "Quantify harmonic complexity of a piece",
                        "steps": "GaChordToSet → ix_stats → ix_chaos_lyapunov",
                        "use_case": "Measure harmonic unpredictability and compare pieces"
                    },
                    "voice_leading_optimization": {
                        "description": "Find optimal voice-leading paths between chords",
                        "steps": "GaCommonTones → ix_search (A*) → ix_optimize",
                        "use_case": "Minimize voice-leading distance in chord progressions"
                    },
                    "trace_feedback_loop": {
                        "description": "Self-improving loop across GA→ix→TARS",
                        "steps": "GA traces → ix_trace_ingest → ix_tars_bridge → TARS promotion",
                        "use_case": "Learn from past tool-call patterns to improve future suggestions"
                    }
                }
            }))
        }
        _ => Err(format!("Unknown ga_bridge action: {}. Use: chord_features, progression_features, scale_features, workflow_guide", action)),
    }
}

// ── ix_explain_algorithm ────────────────────────────────────────────
//
// Delegates algorithm selection to the client's LLM via MCP
// `sampling/createMessage` (spec 2025-06-18). The bidirectional
// dispatcher in `main.rs` routes server-initiated requests through
// [`ServerContext`]; see `server_context.rs`.
//
// The static catalog below is now shipped to the client LLM as part of
// the user prompt so the model can pick from the actual ix surface area
// rather than inventing crate names.

/// Back-compat stub: this tool MUST be routed through the context-aware
/// path ([`explain_algorithm_with_ctx`]). The plain `fn(Value)` handler
/// is retained to keep the [`crate::tools::Tool`] struct shape intact for
/// all other tools.
pub fn explain_algorithm(_params: Value) -> Result<Value, String> {
    Err(
        "ix_explain_algorithm must be dispatched via ServerContext \
         (bidirectional JSON-RPC). This codepath should be intercepted by \
         ToolRegistry::call_with_ctx."
            .into(),
    )
}

/// Context-aware implementation. Builds an algorithm catalog, asks the
/// client LLM to pick one via `sampling/createMessage`, and returns the
/// model's recommendation text.
pub fn explain_algorithm_with_ctx(
    params: Value,
    ctx: &crate::server_context::ServerContext,
) -> Result<Value, String> {
    let problem = parse_str(&params, "problem")?;

    // Curated static catalog of ix algorithms grouped by task family.
    // Each entry: crate path, when-to-use, complexity, key hyperparameters.
    let catalog = json!({
        "clustering": [
            {
                "name": "ix_unsupervised::KMeans",
                "use_when": "Known k, roughly spherical clusters, low noise",
                "complexity": "O(n * k * i * d)",
                "hyperparameters": ["k", "max_iter", "tol", "init (kmeans++)"]
            },
            {
                "name": "ix_unsupervised::DBSCAN",
                "use_when": "Unknown cluster count, noisy data, arbitrary shapes",
                "complexity": "O(n log n) with spatial index",
                "hyperparameters": ["eps", "min_samples", "metric"]
            },
            {
                "name": "ix_unsupervised::HierarchicalClustering",
                "use_when": "Want a dendrogram or nested cluster structure",
                "complexity": "O(n^2 log n)",
                "hyperparameters": ["linkage", "distance_metric"]
            }
        ],
        "supervised": [
            {
                "name": "ix_supervised::LinearRegression",
                "use_when": "Linear relationship, interpretable coefficients",
                "complexity": "O(n * d^2)",
                "hyperparameters": ["regularization (ridge/lasso)", "alpha"]
            },
            {
                "name": "ix_supervised::RandomForest",
                "use_when": "Tabular, mixed feature types, robust baseline",
                "complexity": "O(n log n * d * trees)",
                "hyperparameters": ["n_trees", "max_depth", "min_samples_split"]
            },
            {
                "name": "ix_supervised::GradientBoosting",
                "use_when": "Top accuracy on tabular, willing to tune",
                "complexity": "O(n log n * d * rounds)",
                "hyperparameters": ["learning_rate", "n_estimators", "max_depth"]
            }
        ],
        "optimization": [
            {
                "name": "ix_optimize::GradientDescent",
                "use_when": "Differentiable objective, convex or smooth",
                "complexity": "O(iters * grad_cost)",
                "hyperparameters": ["learning_rate", "momentum", "max_iter"]
            },
            {
                "name": "ix_optimize::BFGS",
                "use_when": "Smooth objective, fast convergence on medium problems",
                "complexity": "O(iters * d^2)",
                "hyperparameters": ["line_search", "tol"]
            },
            {
                "name": "ix_evolution::GeneticAlgorithm",
                "use_when": "Non-differentiable, combinatorial, or rugged landscapes",
                "complexity": "O(gens * pop * fitness_cost)",
                "hyperparameters": ["population", "mutation_rate", "crossover_rate"]
            }
        ],
        "signal_and_sequence": [
            {
                "name": "ix_signal::FFT",
                "use_when": "Frequency-domain analysis of evenly sampled signal",
                "complexity": "O(n log n)",
                "hyperparameters": ["window", "n_points"]
            },
            {
                "name": "ix_probabilistic::HMM (viterbi)",
                "use_when": "Most likely hidden state sequence given observations",
                "complexity": "O(n * s^2)",
                "hyperparameters": ["states", "transition", "emission"]
            }
        ],
        "graph": [
            {
                "name": "ix_graph::Dijkstra",
                "use_when": "Shortest path, non-negative weights",
                "complexity": "O((V + E) log V)",
                "hyperparameters": []
            },
            {
                "name": "ix_graph::PageRank",
                "use_when": "Node importance by link structure",
                "complexity": "O(iters * E)",
                "hyperparameters": ["damping", "tol"]
            }
        ],
        "approximate_data_structures": [
            {
                "name": "ix_probabilistic::BloomFilter",
                "use_when": "Set membership with tolerable false positives, low memory",
                "complexity": "O(k) per op",
                "hyperparameters": ["capacity", "false_positive_rate"]
            },
            {
                "name": "ix_probabilistic::HyperLogLog",
                "use_when": "Cardinality estimation on huge streams",
                "complexity": "O(1) per op",
                "hyperparameters": ["precision"]
            }
        ],
        "chaos_and_dynamics": [
            {
                "name": "ix_chaos::Lyapunov",
                "use_when": "Quantify sensitivity to initial conditions in a time series",
                "complexity": "O(n)",
                "hyperparameters": ["embedding_dim", "delay"]
            }
        ]
    });

    let catalog_pretty = serde_json::to_string_pretty(&catalog)
        .unwrap_or_else(|_| "<catalog unavailable>".into());

    let user_text = format!(
        "Given this problem: \"{}\"\n\n\
         Pick the best ix algorithm from the catalog below and justify the choice. \
         State suggested hyperparameters and one alternative.\n\n\
         ix algorithm catalog (JSON):\n{}",
        problem, catalog_pretty
    );

    let system_prompt = "You are an ML algorithm selector for the ix crate family. \
                         Recommend one primary algorithm and one alternative from the \
                         supplied catalog, with a brief rationale and suggested \
                         hyperparameters. Do not invent crate names that are not in \
                         the catalog.";

    let recommendation = ctx.sample(&user_text, system_prompt, 512)?;

    Ok(json!({
        "status": "ok",
        "problem": problem,
        "recommendation": recommendation,
        "catalog": catalog,
    }))
}

// ─── ix_triage_session ─────────────────────────────────────────────
//
// End-to-end harness scenario tool. Exercises every shipped primitive
// in one call:
//   #1 Context DAG (indirectly, if LLM proposes ix_context_walk)
//   #2 Loop detector (via dispatch_action middleware chain)
//   #2b Substrate (AgentAction construction, SessionEvent emission)
//   #3 Approval / blast-radius (auto-approves Tier 1/2, blocks Tier 3+)
//   #4 Session log (reads recent events, writes dispatched outcomes)
//   #5 Fuzzy / HexavalentDistribution (plan confidence aggregation,
//      escalation_triggered check)
//   #6 Trace flywheel (optional export → ix_trace_ingest round-trip)
//   #7 MCP sampling (ctx.sample is the whole triage decision)
//
// Design doc: docs/brainstorms/2026-04-11-triage-session-scenario.md

/// Back-compat stub for the plain handler table. The real
/// implementation lives in [`triage_session_with_ctx`] and is routed
/// there by [`crate::tools::ToolRegistry::call_with_ctx`].
pub fn triage_session(_params: Value) -> Result<Value, String> {
    Err(
        "ix_triage_session must be dispatched via ServerContext \
         (bidirectional JSON-RPC). This codepath should be intercepted by \
         ToolRegistry::call_with_ctx."
            .into(),
    )
}

/// Context-aware implementation of `ix_triage_session`.
///
/// 1. Reads the last N events from the installed [`ix_session::SessionLog`]
///    (returns an error if no log is installed — triage without history
///    has no input to work with).
/// 2. Builds a compact summary of recent tool calls, blocks, and
///    observations.
/// 3. Asks the client LLM (via MCP sampling) to propose up to
///    `max_actions` ix tool invocations as a JSON array. Parses and
///    validates the response via [`crate::triage::parse_plan`].
/// 4. Builds a [`ix_fuzzy::HexavalentDistribution`] from the plan's
///    aggregated confidences. If the plan-level contradiction mass
///    exceeds the escalation threshold (0.3), returns without
///    dispatching — the plan is too uncertain to run autonomously.
/// 5. Otherwise sorts the plan by hexavalent priority
///    (`C > U > D > P > T > F`) and dispatches each item through
///    [`crate::registry_bridge::dispatch_action`], collecting
///    per-item results. Blocked or failed actions are captured, not
///    thrown — the caller gets a full picture.
/// 6. If `learn == true`, exports the session log via the flywheel
///    and invokes `ix_trace_ingest` on the resulting directory for
///    self-improvement statistics.
/// 7. Returns a synthesis: the plan, the dispatched outcomes, whether
///    escalation triggered, and (optionally) the ingested trace stats.
pub fn triage_session_with_ctx(
    params: Value,
    ctx: &crate::server_context::ServerContext,
) -> Result<Value, String> {
    use crate::projection::events_to_observations;
    use crate::registry_bridge;
    use crate::triage::{
        build_distribution, parse_plan, sort_plan_by_priority, TriagePlanItem,
    };
    use ix_agent_core::{AgentAction, ReadContext, SessionEvent};
    use ix_fuzzy::escalation_triggered;
    use ix_fuzzy::observations::{merge_with_default_staleness, HexObservation};

    // ── 1. Inputs ─────────────────────────────────────────────────
    let focus = params
        .get("focus")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let max_actions = params
        .get("max_actions")
        .and_then(|v| v.as_u64())
        .unwrap_or(3)
        .clamp(1, 8);
    let learn = params
        .get("learn")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let round = params
        .get("round")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    // Parse optional `prior_observations` — typically populated by
    // the main-agent shuttle with tars-emitted diagnosis observations
    // from the previous remediation round. Format: JSON array of
    // HexObservation-shaped objects. Invalid entries are silently
    // dropped to keep the triage call robust against partial data.
    let prior_observations: Vec<HexObservation> = params
        .get("prior_observations")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(parse_prior_observation)
                .collect()
        })
        .unwrap_or_default();

    // ── 2. Read the installed session log ─────────────────────────
    let log = registry_bridge::current_session_log().ok_or_else(|| {
        "ix_triage_session requires an installed SessionLog. \
         Set IX_SESSION_LOG=<path> or call install_session_log() \
         before dispatching this tool."
            .to_string()
    })?;

    // Flush any buffered writes so we see our own recent history.
    if let Err(e) = log.flush() {
        return Err(format!("failed to flush session log before read: {e}"));
    }

    let events: Vec<SessionEvent> = log
        .events()
        .map_err(|e| format!("failed to read session log: {e}"))?
        .filter_map(Result::ok)
        .collect();

    // Take the last 20 events as the "recent history" the LLM reasons
    // about. Older events are out of scope for a single triage call.
    let recent: Vec<&SessionEvent> = events.iter().rev().take(20).collect::<Vec<_>>();
    let recent: Vec<&SessionEvent> = recent.into_iter().rev().collect();
    let summary = summarize_events(&recent);

    // ── 3. Sample the LLM for a structured plan ───────────────────
    let system_prompt =
        "You are the ix harness triage agent. Given recent session events, propose up to \
         MAX_ACTIONS ix tool invocations that would advance the current investigation. \
         Respond with ONLY a JSON array (no prose, no code fences) matching this schema:\n\
         [\n  {\n    \"tool_name\": \"ix_<tool>\",\n    \"params\": { ... },\n    \
         \"confidence\": one of T/P/U/D/F/C,\n    \"reason\": \"short justification\"\n  }\n]\n\
         \n\
         Confidence legend (Demerzel hexavalent logic):\n\
         - T (True): you are verified this action will help\n\
         - P (Probable): evidence leans toward helpful\n\
         - U (Unknown): insufficient evidence — action would be exploratory\n\
         - D (Doubtful): evidence leans against usefulness\n\
         - F (False): refuted — do NOT propose F-confidence actions\n\
         - C (Contradictory): conflicting signals — the plan should be escalated if most items are C\n\
         \n\
         HARD CONSTRAINT: do NOT propose ix_triage_session. That would recurse.";

    let user_text = format!(
        "Recent session events (up to 20, oldest first):\n\
         {summary}\n\
         \n\
         Focus hint: {focus}\n\
         Max actions in plan: {max_actions}\n\
         \n\
         Emit the JSON array now.",
        summary = summary,
        focus = if focus.is_empty() { "none" } else { &focus },
        max_actions = max_actions,
    );

    let plan_text = ctx
        .sample(&user_text, system_prompt, 1024)
        .map_err(|e| format!("sampling failed: {e}"))?;

    // ── 4. Parse, validate, rank ──────────────────────────────────
    let mut plan: Vec<TriagePlanItem> = match parse_plan(&plan_text) {
        Ok(p) => p,
        Err(e) => {
            // Non-destructive failure: return the raw LLM text plus
            // the parse error so the caller can see what went wrong.
            return Ok(json!({
                "status": "parse_failed",
                "error": e.to_string(),
                "raw_response": plan_text,
                "events_read": events.len(),
            }));
        }
    };

    // Clamp to max_actions in case the LLM emitted more than
    // requested (prompt is a hint, not a contract).
    plan.truncate(max_actions as usize);
    sort_plan_by_priority(&mut plan);

    // ── 5. Build unified observation set ──────────────────────────
    //
    // Three streams feed the merge:
    //   1. Plan observations — projected from the parsed plan items
    //      with source="triage-plan" and claim_key="<tool>::valuable"
    //   2. Session history observations — projected from the events
    //      we already read above, so ix's own recent execution
    //      outcomes count toward the escalation decision
    //   3. Prior observations from the params — typically
    //      cross-repo observations delivered by the main-agent
    //      shuttle (e.g., tars diagnosis from the previous round)
    //
    // All three streams merge through the G-Set CRDT defined in
    // ix-fuzzy::observations (see demerzel/logic/hex-merge.md).
    // Contradictions are synthesized automatically; the merged
    // distribution feeds escalation_triggered.
    let plan_diagnosis_id = format!("triage-plan:{focus}");
    let plan_observations: Vec<HexObservation> = plan
        .iter()
        .enumerate()
        .map(|(i, item)| plan_item_to_observation(item, &plan_diagnosis_id, round, i as u32))
        .collect();

    let session_observations = events_to_observations(&events, "ix", "ix-session-log", round);

    let mut all_observations = Vec::with_capacity(
        plan_observations.len() + session_observations.len() + prior_observations.len(),
    );
    all_observations.extend(plan_observations.iter().cloned());
    all_observations.extend(session_observations);
    all_observations.extend(prior_observations.iter().cloned());

    let merged = merge_with_default_staleness(&all_observations, round)
        .map_err(|e| format!("observation merge failed: {e}"))?;

    // Preserve the plan-only distribution for backward-compat output
    // so callers that used to read the plan distribution still see
    // it. The merged distribution is what drives escalation.
    let plan_only_distribution = build_distribution(&plan)
        .map_err(|e| format!("plan distribution build failed: {e}"))?;
    let escalate = escalation_triggered(&merged.distribution);

    if escalate {
        return Ok(json!({
            "status": "escalated",
            "reason": if merged.contradictions.is_empty() {
                "plan-level contradiction mass exceeds 0.3 threshold"
            } else {
                "cross-source contradiction detected"
            },
            "plan": plan_as_json(&plan),
            "distribution": distribution_as_json(&plan_only_distribution),
            "merged_distribution": distribution_as_json(&merged.distribution),
            "contradictions": contradictions_as_json(&merged.contradictions),
            "observation_counts": {
                "plan": plan_observations.len(),
                "session": merged.observations.iter().filter(|o| o.source == "ix").count(),
                "prior": prior_observations.len(),
                "synthesized": merged.contradictions.len(),
            },
            "events_read": events.len(),
        }));
    }

    // ── 6. Dispatch each item through the governed chain ─────────
    let cx = ReadContext::synthetic_for_legacy();
    let mut dispatched_results: Vec<Value> = Vec::with_capacity(plan.len());
    for item in &plan {
        let action = AgentAction::InvokeTool {
            tool_name: item.tool_name.clone(),
            params: item.params.clone(),
            ordinal: 0, // dispatch_action assigns
            target_hint: item
                .params
                .get("target")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        };

        let outcome = registry_bridge::dispatch_action(&cx, action);
        let result_entry = match outcome {
            Ok(action_outcome) => json!({
                "tool_name": item.tool_name,
                "ok": true,
                "confidence": hexavalent_label(&item.confidence),
                "value": action_outcome.value,
                "reason": item.reason,
            }),
            Err(err) => json!({
                "tool_name": item.tool_name,
                "ok": false,
                "confidence": hexavalent_label(&item.confidence),
                "error": err.to_string(),
                "reason": item.reason,
            }),
        };
        dispatched_results.push(result_entry);
    }

    // ── 7. Optional self-learning via the flywheel ───────────────
    let (trace_dir_value, trace_ingest_value) = if learn {
        // Ensure our own dispatch events are visible to the export.
        let _ = log.flush();

        // Export to a sibling `traces/` directory next to the log
        // file. Using a deterministic directory makes the trace
        // discoverable by other tools.
        let trace_dir = log
            .path()
            .parent()
            .map(|p| p.join("traces"))
            .unwrap_or_else(|| std::path::PathBuf::from("traces"));

        match crate::flywheel::export_session_to_trace_dir(&log, &trace_dir, None) {
            Ok(written_path) => {
                // Now invoke ix_trace_ingest on the directory through
                // the legacy dispatch path (which also runs through
                // the middleware chain — governed recursion is a
                // feature).
                let ingest = registry_bridge::dispatch(
                    "ix_trace_ingest",
                    json!({ "dir": trace_dir.display().to_string() }),
                );
                let ingest_value = match ingest {
                    Ok(v) => json!({ "ok": true, "stats": v }),
                    Err(e) => json!({ "ok": false, "error": e }),
                };
                (
                    Value::String(written_path.display().to_string()),
                    ingest_value,
                )
            }
            Err(e) => (
                Value::Null,
                json!({ "ok": false, "error": format!("flywheel export failed: {e}") }),
            ),
        }
    } else {
        (Value::Null, Value::Null)
    };

    // ── 8. Synthesis ──────────────────────────────────────────────
    Ok(json!({
        "status": "dispatched",
        "focus": focus,
        "max_actions": max_actions,
        "round": round,
        "events_read": events.len(),
        "plan": plan_as_json(&plan),
        "distribution": distribution_as_json(&plan_only_distribution),
        "merged_distribution": distribution_as_json(&merged.distribution),
        "contradictions": contradictions_as_json(&merged.contradictions),
        "observation_counts": {
            "plan": plan_observations.len(),
            "session": merged.observations.iter().filter(|o| o.source == "ix").count(),
            "prior": prior_observations.len(),
            "synthesized": merged.contradictions.len(),
        },
        "escalated": escalate,
        "dispatched": dispatched_results,
        "trace_dir": trace_dir_value,
        "trace_ingest": trace_ingest_value,
    }))
}

/// Parse a JSON value into a [`HexObservation`]. Silently drops
/// malformed entries — the triage session prefers partial data
/// over a hard failure when the caller's observations list has
/// typos or stale fields.
fn parse_prior_observation(
    value: &Value,
) -> Option<ix_fuzzy::observations::HexObservation> {
    use ix_fuzzy::observations::HexObservation;
    let obj = value.as_object()?;
    let variant_label = obj.get("variant").and_then(|v| v.as_str())?;
    let variant = crate::triage::parse_hexavalent_label(variant_label)?;
    Some(HexObservation {
        source: obj.get("source").and_then(|v| v.as_str())?.to_string(),
        diagnosis_id: obj
            .get("diagnosis_id")
            .and_then(|v| v.as_str())?
            .to_string(),
        round: obj.get("round").and_then(|v| v.as_u64())? as u32,
        ordinal: obj.get("ordinal").and_then(|v| v.as_u64())? as u32,
        claim_key: obj.get("claim_key").and_then(|v| v.as_str())?.to_string(),
        variant,
        weight: obj.get("weight").and_then(|v| v.as_f64())?,
        evidence: obj.get("evidence").and_then(|v| v.as_str()).map(String::from),
    })
}

/// Project a triage plan item into a [`HexObservation`]. The
/// source is always `"triage-plan"` so the merge function can
/// distinguish these from ix session observations and tars prior
/// observations. The claim_key is always `<tool>::valuable` — plan
/// items assert the value of calling a tool, not its safety.
fn plan_item_to_observation(
    item: &crate::triage::TriagePlanItem,
    diagnosis_id: &str,
    round: u32,
    ordinal: u32,
) -> ix_fuzzy::observations::HexObservation {
    use ix_fuzzy::observations::HexObservation;
    // Derive target_hint from item.params.target if present, matching
    // the existing dispatch path in the handler above.
    let target_hint = item
        .params
        .get("target")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let action_key = match target_hint {
        Some(t) => format!("{}:{t}", item.tool_name),
        None => item.tool_name.clone(),
    };
    HexObservation {
        source: "triage-plan".to_string(),
        diagnosis_id: diagnosis_id.to_string(),
        round,
        ordinal,
        claim_key: format!("{action_key}::valuable"),
        variant: item.confidence,
        weight: 1.0,
        evidence: if item.reason.is_empty() {
            None
        } else {
            Some(item.reason.clone())
        },
    }
}

/// Render a [`HexavalentDistribution`] as a JSON object with
/// per-variant keys. Extracted so both the escalation-path and the
/// dispatch-path output share one formatting rule.
fn distribution_as_json(dist: &ix_fuzzy::HexavalentDistribution) -> Value {
    json!({
        "T": dist.get(&ix_types::Hexavalent::True),
        "P": dist.get(&ix_types::Hexavalent::Probable),
        "U": dist.get(&ix_types::Hexavalent::Unknown),
        "D": dist.get(&ix_types::Hexavalent::Doubtful),
        "F": dist.get(&ix_types::Hexavalent::False),
        "C": dist.get(&ix_types::Hexavalent::Contradictory),
    })
}

/// Render a list of synthesized contradictions as JSON objects.
/// Used in both the escalation-path and dispatch-path output so
/// callers can always see why the merge flagged a disagreement.
fn contradictions_as_json(
    contradictions: &[ix_fuzzy::observations::HexObservation],
) -> Value {
    Value::Array(
        contradictions
            .iter()
            .map(|o| {
                json!({
                    "claim_key": o.claim_key,
                    "weight": o.weight,
                    "evidence": o.evidence,
                })
            })
            .collect(),
    )
}

/// Compact string summary of a slice of session events, used inside
/// the triage sampling prompt. One line per event, keeping only the
/// fields the LLM needs for triage decisions.
fn summarize_events(events: &[&ix_agent_core::SessionEvent]) -> String {
    use ix_agent_core::SessionEvent;
    if events.is_empty() {
        return "(no events in session — this is a fresh start)".to_string();
    }
    let mut out = String::new();
    for (i, event) in events.iter().enumerate() {
        use std::fmt::Write;
        let line = match event {
            SessionEvent::ActionProposed { ordinal, action } => match action {
                ix_agent_core::AgentAction::InvokeTool { tool_name, .. } => {
                    format!("#{ordinal} proposed invoke_tool({tool_name})")
                }
                _ => format!("#{ordinal} proposed {:?}", action),
            },
            SessionEvent::ActionCompleted { ordinal, .. } => {
                format!("#{ordinal} completed")
            }
            SessionEvent::ActionBlocked {
                ordinal,
                code,
                reason,
                emitted_by,
            } => format!(
                "#{ordinal} BLOCKED by {emitted_by} ({code:?}): {reason}"
            ),
            SessionEvent::ActionReplaced {
                ordinal, emitted_by, ..
            } => format!("#{ordinal} replaced by {emitted_by}"),
            SessionEvent::ActionFailed { ordinal, error } => {
                format!("#{ordinal} FAILED: {error}")
            }
            SessionEvent::MetadataMounted {
                ordinal,
                path,
                emitted_by,
                ..
            } => format!("#{ordinal} metadata {path} (from {emitted_by})"),
            SessionEvent::BeliefChanged {
                ordinal,
                proposition,
                ..
            } => format!("#{ordinal} belief changed: {proposition}"),
            SessionEvent::ObservationAdded {
                ordinal,
                source,
                claim_key,
                variant,
                weight,
                ..
            } => format!(
                "#{ordinal} observation from {source}: {claim_key} = {variant:?} (w={weight:.2})"
            ),
        };
        let _ = writeln!(out, "{i:>3}. {line}");
    }
    out
}

/// Serialize a plan to a JSON-friendly shape for the synthesis output.
fn plan_as_json(plan: &[crate::triage::TriagePlanItem]) -> Value {
    Value::Array(
        plan.iter()
            .map(|item| {
                json!({
                    "tool_name": item.tool_name,
                    "params": item.params,
                    "confidence": hexavalent_label(&item.confidence),
                    "reason": item.reason,
                })
            })
            .collect(),
    )
}

/// Map a [`ix_types::Hexavalent`] to its single-letter label for
/// readable output.
fn hexavalent_label(h: &ix_types::Hexavalent) -> &'static str {
    match h {
        ix_types::Hexavalent::True => "T",
        ix_types::Hexavalent::Probable => "P",
        ix_types::Hexavalent::Unknown => "U",
        ix_types::Hexavalent::Doubtful => "D",
        ix_types::Hexavalent::False => "F",
        ix_types::Hexavalent::Contradictory => "C",
    }
}

// ── ix_render_audit ──────────────────────────────────────────────
//
// MCP tool that navigates the GA Prime Radiant to a target body,
// captures a screenshot via the GA API (SignalR → browser), and
// uses MCP sampling (sampling/createMessage with image content)
// to ask the client LLM to analyze the rendering for correctness.
//
// This is the "QA agent" the v3 ocean spec calls for: the harness
// looks at what the Prime Radiant is showing, forms beliefs about
// rendering quality, and emits structured observations.
//
// Wire:
//   ix MCP client → ix_render_audit (this tool)
//     → HTTP GET ga-api/navigate-and-capture
//       → SignalR → browser → screenshot
//     → sampling/createMessage { image + text prompt }
//       → client LLM (Claude vision) → structured analysis
//     → returns observations as JSON

/// Back-compat stub for the rendering audit tool. Must be routed
/// through the context-aware path.
pub fn render_audit(_params: Value) -> Result<Value, String> {
    Err(
        "ix_render_audit must be dispatched via ServerContext \
         (bidirectional JSON-RPC for MCP sampling). This codepath should \
         be intercepted by ToolRegistry::call_with_ctx."
            .into(),
    )
}

/// Context-aware implementation of `ix_render_audit`.
///
/// 1. Calls the GA API `POST /api/governance/navigate-and-capture`
///    to navigate the Prime Radiant and capture a screenshot.
/// 2. Sends the screenshot to the client LLM via MCP sampling
///    (`sampling/createMessage` with image content) alongside
///    a structured prompt asking for rendering observations.
/// 3. Returns the LLM's analysis as structured JSON.
pub fn render_audit_with_ctx(
    params: Value,
    ctx: &crate::server_context::ServerContext,
) -> Result<Value, String> {
    let target = params
        .get("target")
        .and_then(|v| v.as_str())
        .unwrap_or("earth");
    let wait_ms = params
        .get("wait_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(2000);
    let ga_api_url = params
        .get("ga_api_url")
        .and_then(|v| v.as_str())
        .unwrap_or("http://localhost:5001");

    // ── 1. Navigate + Capture via GA API ─────────────────────────
    let url = format!(
        "{}/api/governance/navigate-and-capture",
        ga_api_url.trim_end_matches('/')
    );

    let body = json!({
        "target": target,
        "waitMs": wait_ms,
    });

    let client = std::sync::OnceLock::<reqwest::blocking::Client>::new();
    let http = client.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .danger_accept_invalid_certs(true) // dev localhost
            .build()
            .expect("build HTTP client")
    });

    let response = http
        .post(&url)
        .json(&body)
        .send()
        .map_err(|e| format!("GA API request failed: {e}. Is the GA server running?"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().unwrap_or_default();
        return Err(format!(
            "GA API returned {status}: {text}. Is a Prime Radiant client connected?"
        ));
    }

    // Response is a PNG image
    let image_bytes = response
        .bytes()
        .map_err(|e| format!("read response body: {e}"))?;
    let image_base64 = base64_encode(&image_bytes);

    // ── 2. Ask Claude to analyze via MCP sampling ────────────────
    let system_prompt = format!(
        "You are a rendering QA agent for the Prime Radiant 3D solar system viewer. \
         You are analyzing a screenshot of '{target}' captured from a live Prime Radiant instance. \
         Your job is to identify rendering issues — shading errors, missing effects, \
         visual artifacts, incorrect lighting, missing dark faces, distorted geometry, \
         or any other visual problem. \
         \
         Respond with a JSON object (no markdown fencing) containing: \
         {{ \
           \"target\": \"{target}\", \
           \"quality\": <number 0-100>, \
           \"issues\": [{{ \"id\": \"...\", \"severity\": \"critical|major|minor\", \"description\": \"...\" }}], \
           \"observations\": [{{ \"claim_key\": \"render:...\", \"variant\": \"T|P|U|D|F|C\", \"reason\": \"...\" }}] \
         }}"
    );

    let user_prompt = format!(
        "Analyze this screenshot of '{target}' from the Prime Radiant solar system viewer. \
         Check for: \
         1. Does the body have correct day/night shading (lit side facing the sun, dark side away)? \
         2. Is the body's shape correct (no visible distortion, bloating, or puffiness)? \
         3. Are textures sharp enough for the zoom level? \
         4. Is the atmosphere (if applicable) rendering correctly? \
         5. Are there any visible artifacts, Z-order issues, or shader errors? \
         \
         Return your analysis as a JSON object."
    );

    let analysis = ctx
        .sample_with_image(
            &user_prompt,
            &image_base64,
            "image/png",
            &system_prompt,
            2000,
        )
        .map_err(|e| format!("MCP sampling failed: {e}"))?;

    // ── 3. Parse and return ──────────────────────────────────────
    // Try to parse as JSON; if the LLM returned markdown-fenced
    // JSON, strip the fences first.
    let clean = analysis
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let parsed: Value = serde_json::from_str(clean).unwrap_or_else(|_| {
        // Fallback: wrap the raw text as a best-effort response
        json!({
            "target": target,
            "quality": 0,
            "issues": [],
            "raw_response": analysis,
            "parse_error": "LLM response was not valid JSON"
        })
    });

    Ok(json!({
        "target": target,
        "screenshot_size_bytes": image_bytes.len(),
        "analysis": parsed,
    }))
}

/// Simple base64 encoder (no external dep for this one function).
fn base64_encode(data: &[u8]) -> String {
    use std::io::Write as _;
    let mut buf = Vec::with_capacity(data.len() * 4 / 3 + 4);
    {
        let mut encoder = base64_writer(&mut buf);
        encoder.write_all(data).expect("base64 encode");
    }
    String::from_utf8(buf).expect("base64 is valid UTF-8")
}

/// Minimal base64 encoder writer using the standard alphabet.
fn base64_writer(output: &mut Vec<u8>) -> impl std::io::Write + '_ {
    struct B64Writer<'a> {
        out: &'a mut Vec<u8>,
        buf: [u8; 3],
        pos: usize,
    }
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    impl<'a> std::io::Write for B64Writer<'a> {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            for &b in data {
                self.buf[self.pos] = b;
                self.pos += 1;
                if self.pos == 3 {
                    let n = ((self.buf[0] as u32) << 16)
                        | ((self.buf[1] as u32) << 8)
                        | (self.buf[2] as u32);
                    self.out.push(ALPHA[((n >> 18) & 0x3F) as usize]);
                    self.out.push(ALPHA[((n >> 12) & 0x3F) as usize]);
                    self.out.push(ALPHA[((n >> 6) & 0x3F) as usize]);
                    self.out.push(ALPHA[(n & 0x3F) as usize]);
                    self.pos = 0;
                }
            }
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            match self.pos {
                1 => {
                    let n = (self.buf[0] as u32) << 16;
                    self.out.push(ALPHA[((n >> 18) & 0x3F) as usize]);
                    self.out.push(ALPHA[((n >> 12) & 0x3F) as usize]);
                    self.out.push(b'=');
                    self.out.push(b'=');
                    self.pos = 0;
                }
                2 => {
                    let n = ((self.buf[0] as u32) << 16) | ((self.buf[1] as u32) << 8);
                    self.out.push(ALPHA[((n >> 18) & 0x3F) as usize]);
                    self.out.push(ALPHA[((n >> 12) & 0x3F) as usize]);
                    self.out.push(ALPHA[((n >> 6) & 0x3F) as usize]);
                    self.out.push(b'=');
                    self.pos = 0;
                }
                _ => {}
            }
            Ok(())
        }
    }
    impl<'a> Drop for B64Writer<'a> {
        fn drop(&mut self) {
            let _ = std::io::Write::flush(self);
        }
    }
    B64Writer {
        out: output,
        buf: [0; 3],
        pos: 0,
    }
}
