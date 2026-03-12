//! Tool handler implementations — each parses JSON params, calls the underlying crate, returns JSON.

use ndarray::{Array1, Array2};
use serde_json::{json, Value};

use machin_cache::{Cache, CacheConfig};

use std::sync::OnceLock;

/// Global cache instance shared across tool calls.
fn global_cache() -> &'static Cache {
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

fn parse_str<'a>(val: &'a Value, field: &str) -> Result<&'a str, String> {
    val.get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("Missing or invalid field '{}'", field))
}

// ── machin_stats ───────────────────────────────────────────────

pub fn stats(params: Value) -> Result<Value, String> {
    let data = parse_f64_array(&params, "data")?;
    if data.is_empty() {
        return Err("data must not be empty".into());
    }
    let arr = vec_to_array1(&data);

    let mean = machin_math::stats::mean(&arr).map_err(|e| format!("{}", e))?;
    let std_dev = machin_math::stats::std_dev(&arr).map_err(|e| format!("{}", e))?;
    let median = machin_math::stats::median(&arr).map_err(|e| format!("{}", e))?;
    let (min, max) = machin_math::stats::min_max(&arr).map_err(|e| format!("{}", e))?;
    let variance = machin_math::stats::variance(&arr).map_err(|e| format!("{}", e))?;

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

// ── machin_distance ────────────────────────────────────────────

pub fn distance(params: Value) -> Result<Value, String> {
    let a = parse_f64_array(&params, "a")?;
    let b = parse_f64_array(&params, "b")?;
    let metric = parse_str(&params, "metric")?;

    let arr_a = vec_to_array1(&a);
    let arr_b = vec_to_array1(&b);

    let (distance, metric_name) = match metric {
        "euclidean" => {
            let d = machin_math::distance::euclidean(&arr_a, &arr_b)
                .map_err(|e| format!("{}", e))?;
            (d, "euclidean")
        }
        "cosine" => {
            let d = machin_math::distance::cosine_distance(&arr_a, &arr_b)
                .map_err(|e| format!("{}", e))?;
            (d, "cosine")
        }
        "manhattan" => {
            let d = machin_math::distance::manhattan(&arr_a, &arr_b)
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

// ── machin_optimize ────────────────────────────────────────────

pub fn optimize(params: Value) -> Result<Value, String> {
    let func_name = parse_str(&params, "function")?;
    let dimensions = parse_usize(&params, "dimensions")?;
    let method = parse_str(&params, "method")?;
    let max_iter = parse_usize(&params, "max_iter")?;

    if dimensions == 0 {
        return Err("dimensions must be >= 1".into());
    }

    let objective: machin_optimize::traits::ClosureObjective<Box<dyn Fn(&Array1<f64>) -> f64>> =
        match func_name {
            "sphere" => machin_optimize::traits::ClosureObjective {
                f: Box::new(|x: &Array1<f64>| x.mapv(|v| v * v).sum()),
                dimensions,
            },
            "rosenbrock" => machin_optimize::traits::ClosureObjective {
                f: Box::new(|x: &Array1<f64>| {
                    (0..x.len() - 1)
                        .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                        .sum()
                }),
                dimensions,
            },
            "rastrigin" => machin_optimize::traits::ClosureObjective {
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
            let mut opt = machin_optimize::gradient::SGD::new(0.01);
            let criteria = machin_optimize::convergence::ConvergenceCriteria {
                max_iterations: max_iter,
                tolerance: 1e-8,
            };
            let init = Array1::from_elem(dimensions, 5.0);
            machin_optimize::gradient::minimize(&objective, &mut opt, init, &criteria)
        }
        "adam" => {
            let mut opt = machin_optimize::gradient::Adam::new(0.01);
            let criteria = machin_optimize::convergence::ConvergenceCriteria {
                max_iterations: max_iter,
                tolerance: 1e-8,
            };
            let init = Array1::from_elem(dimensions, 5.0);
            machin_optimize::gradient::minimize(&objective, &mut opt, init, &criteria)
        }
        "pso" => {
            let pso = machin_optimize::pso::ParticleSwarm::new()
                .with_max_iterations(max_iter)
                .with_bounds(-10.0, 10.0)
                .with_seed(42);
            pso.minimize(&objective)
        }
        "annealing" => {
            let sa = machin_optimize::annealing::SimulatedAnnealing::new()
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

// ── machin_linear_regression ───────────────────────────────────

pub fn linear_regression(params: Value) -> Result<Value, String> {
    use machin_supervised::traits::Regressor;

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

    let mut model = machin_supervised::linear_regression::LinearRegression::new();
    model.fit(&x, &y);

    let predictions = model.predict(&x);
    let weights = model.weights.as_ref().map(|w| w.to_vec()).unwrap_or_default();

    Ok(json!({
        "weights": weights,
        "bias": model.bias,
        "predictions": predictions.to_vec(),
    }))
}

// ── machin_kmeans ──────────────────────────────────────────────

pub fn kmeans(params: Value) -> Result<Value, String> {
    use machin_unsupervised::traits::Clusterer;

    let data_rows = parse_f64_matrix(&params, "data")?;
    let k = parse_usize(&params, "k")?;
    let max_iter = parse_usize(&params, "max_iter")?;

    let data = vecs_to_array2(&data_rows)?;

    let mut km = machin_unsupervised::kmeans::KMeans::new(k);
    km.max_iterations = max_iter;
    km.seed = 42;

    let labels = km.fit_predict(&data);
    let centroids: Vec<Vec<f64>> = km
        .centroids
        .as_ref()
        .map(|c| (0..c.nrows()).map(|i| c.row(i).to_vec()).collect())
        .unwrap_or_default();

    let inertia = km.centroids.as_ref().map(|c| {
        machin_unsupervised::kmeans::inertia(&data, &labels, c)
    }).unwrap_or(0.0);

    Ok(json!({
        "labels": labels.to_vec(),
        "centroids": centroids,
        "inertia": inertia,
        "k": k,
    }))
}

// ── machin_fft ─────────────────────────────────────────────────

pub fn fft(params: Value) -> Result<Value, String> {
    let signal = parse_f64_array(&params, "signal")?;
    if signal.is_empty() {
        return Err("signal must not be empty".into());
    }

    let spectrum = machin_signal::fft::rfft(&signal);
    let magnitudes = machin_signal::fft::magnitude_spectrum(&spectrum);
    let n = spectrum.len();
    // Frequency bins assuming sample_rate=1.0 (normalized)
    let frequencies: Vec<f64> = (0..n).map(|k| k as f64 / n as f64).collect();

    Ok(json!({
        "frequencies": frequencies,
        "magnitudes": magnitudes,
        "fft_size": n,
    }))
}

// ── machin_markov ──────────────────────────────────────────────

pub fn markov(params: Value) -> Result<Value, String> {
    let tm_rows = parse_f64_matrix(&params, "transition_matrix")?;
    let steps = parse_usize(&params, "steps")?;

    let tm = vecs_to_array2(&tm_rows)?;
    let chain = machin_graph::markov::MarkovChain::new(tm)?;

    let stationary = chain.stationary_distribution(steps, 1e-10);
    let is_ergodic = chain.is_ergodic(100);

    Ok(json!({
        "stationary_distribution": stationary.to_vec(),
        "n_states": chain.n_states(),
        "is_ergodic": is_ergodic,
    }))
}

// ── machin_viterbi ─────────────────────────────────────────────

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

    let hmm = machin_graph::hmm::HiddenMarkovModel::new(initial, transition, emission)?;
    let (path, log_probability) = hmm.viterbi(&observations);

    Ok(json!({
        "path": path,
        "log_probability": log_probability,
    }))
}

// ── machin_search ──────────────────────────────────────────────

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

// ── machin_game_nash ───────────────────────────────────────────

pub fn game_nash(params: Value) -> Result<Value, String> {
    let pa_rows = parse_f64_matrix(&params, "payoff_a")?;
    let pb_rows = parse_f64_matrix(&params, "payoff_b")?;

    let payoff_a = vecs_to_array2(&pa_rows)?;
    let payoff_b = vecs_to_array2(&pb_rows)?;

    let game = machin_game::nash::BimatrixGame::new(payoff_a, payoff_b);
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

// ── machin_chaos_lyapunov ──────────────────────────────────────

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
            let mle = machin_chaos::lyapunov::mle_1d(f, df, 0.1, iterations, 1000);
            let dynamics = machin_chaos::lyapunov::classify_dynamics(mle, 0.01);

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

// ── machin_adversarial_fgsm ────────────────────────────────────

pub fn adversarial_fgsm(params: Value) -> Result<Value, String> {
    let input = parse_f64_array(&params, "input")?;
    let gradient = parse_f64_array(&params, "gradient")?;
    let epsilon = params
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "Missing or invalid field 'epsilon'".to_string())?;

    let input_arr = vec_to_array1(&input);
    let grad_arr = vec_to_array1(&gradient);

    let adversarial = machin_adversarial::evasion::fgsm(&input_arr, &grad_arr, epsilon);
    let perturbation = &adversarial - &input_arr;
    let l_inf = perturbation.mapv(f64::abs).iter().cloned().fold(0.0_f64, f64::max);

    Ok(json!({
        "adversarial_input": adversarial.to_vec(),
        "perturbation": perturbation.to_vec(),
        "l_inf_norm": l_inf,
        "epsilon": epsilon,
    }))
}

// ── machin_bloom_filter ────────────────────────────────────────

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

    let mut bf = machin_probabilistic::bloom::BloomFilter::new(items.len().max(1), fp_rate);
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
        "create" | _ => {
            Ok(json!({
                "created": true,
                "items_count": items.len(),
                "bit_size": bf.bit_size(),
                "estimated_fp_rate": bf.estimated_fp_rate(),
            }))
        }
    }
}

// ── machin_cache ───────────────────────────────────────────────

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
