//! Batch 2 — 28 primitive MCP tools migrated to the capability registry.
//!
//! Each wrapper delegates to the pre-existing `handlers::*` fn to preserve
//! MCP behavior byte-for-byte; only the registration surface changes. Skill
//! names map 1:1 to the original MCP tool names after dot→underscore
//! expansion so `ToolRegistry::merge_registry_tools` replaces manual entries
//! cleanly (verified by the strict 43-count parity test).
//!
//! Composite / orchestration tools (`ml_pipeline`, `ml_predict`, `pipeline`,
//! `cache`, `code_analyze`, `federation_discover`, `trace_ingest`,
//! `ga_bridge`, `tars_bridge`) remain manually registered — their schemas
//! are complex nested shapes that gain little from registry migration.

use crate::handlers;
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

// ---- optimize ------------------------------------------------------------
fn optimize_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "function": {"type": "string", "enum": ["sphere", "rosenbrock", "rastrigin"]},
            "dimensions": {"type": "integer", "minimum": 1},
            "method": {"type": "string", "enum": ["sgd", "adam", "pso", "annealing"]},
            "max_iter": {"type": "integer", "minimum": 1}
        },
        "required": ["function", "dimensions", "method", "max_iter"]
    })
}
/// Minimize a benchmark function via SGD / Adam / PSO / simulated annealing.
#[ix_skill(
    domain = "optimize",
    name = "optimize",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::optimize_schema"
)]
pub fn optimize(p: Value) -> Result<Value, String> {
    handlers::optimize(p)
}

// ---- markov --------------------------------------------------------------
fn markov_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "transition_matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "steps": {"type": "integer", "minimum": 1}
        },
        "required": ["transition_matrix", "steps"]
    })
}
/// Markov chain stationary distribution via power iteration.
#[ix_skill(
    domain = "graph",
    name = "markov",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::markov_schema"
)]
pub fn markov(p: Value) -> Result<Value, String> {
    handlers::markov(p)
}

// ---- viterbi -------------------------------------------------------------
fn viterbi_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "initial": {"type": "array", "items": {"type": "number"}},
            "transition": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "emission": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "observations": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["initial", "transition", "emission", "observations"]
    })
}
/// HMM Viterbi decoding — most-likely hidden state sequence.
#[ix_skill(
    domain = "graph",
    name = "viterbi",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::viterbi_schema"
)]
pub fn viterbi(p: Value) -> Result<Value, String> {
    handlers::viterbi(p)
}

// ---- search --------------------------------------------------------------
fn search_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "algorithm": {"type": "string", "enum": ["astar", "bfs", "dfs"]},
            "description": {"type": "boolean"}
        },
        "required": ["algorithm"]
    })
}
/// Search algorithm catalog (A*, BFS, DFS): descriptions and complexity.
#[ix_skill(
    domain = "search",
    name = "search",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::search_schema"
)]
pub fn search(p: Value) -> Result<Value, String> {
    handlers::search_info(p)
}

// ---- game.nash -----------------------------------------------------------
fn game_nash_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "payoff_a": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "payoff_b": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
        },
        "required": ["payoff_a", "payoff_b"]
    })
}
/// Nash equilibria of a 2-player bimatrix game via support enumeration.
#[ix_skill(
    domain = "game",
    name = "game.nash",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::game_nash_schema"
)]
pub fn game_nash(p: Value) -> Result<Value, String> {
    handlers::game_nash(p)
}

// ---- chaos.lyapunov ------------------------------------------------------
fn chaos_lyapunov_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "map": {"type": "string", "enum": ["logistic"]},
            "parameter": {"type": "number"},
            "iterations": {"type": "integer", "minimum": 1}
        },
        "required": ["map", "parameter", "iterations"]
    })
}
/// Maximal Lyapunov exponent of the logistic map at parameter r.
#[ix_skill(
    domain = "chaos",
    name = "chaos.lyapunov",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch2::chaos_lyapunov_schema"
)]
pub fn chaos_lyapunov(p: Value) -> Result<Value, String> {
    handlers::chaos_lyapunov(p)
}

// ---- adversarial.fgsm ----------------------------------------------------
fn adversarial_fgsm_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "input": {"type": "array", "items": {"type": "number"}},
            "gradient": {"type": "array", "items": {"type": "number"}},
            "epsilon": {"type": "number"}
        },
        "required": ["input", "gradient", "epsilon"]
    })
}
/// Fast Gradient Sign Method adversarial perturbation.
#[ix_skill(
    domain = "adversarial",
    name = "adversarial.fgsm",
    governance = "safety",
    schema_fn = "crate::skills::batch2::adversarial_fgsm_schema"
)]
pub fn adversarial_fgsm(p: Value) -> Result<Value, String> {
    handlers::adversarial_fgsm(p)
}

// ---- bloom_filter --------------------------------------------------------
fn bloom_filter_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["create", "check"]},
            "items": {"type": "array", "items": {"type": "string"}},
            "query": {"type": "string"},
            "false_positive_rate": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["items", "false_positive_rate"]
    })
}
/// Bloom filter: probabilistic set membership.
#[ix_skill(
    domain = "probabilistic",
    name = "bloom_filter",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::bloom_filter_schema"
)]
pub fn bloom_filter(p: Value) -> Result<Value, String> {
    handlers::bloom_filter(p)
}

// ---- grammar.weights -----------------------------------------------------
fn grammar_weights_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "rules": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "alpha": {"type": "number"}, "beta": {"type": "number"},
                "weight": {"type": "number"}, "level": {"type": "integer"}, "source": {"type": "string"}
            }, "required": ["id"]}},
            "observation": {"type": "object", "properties": {
                "rule_id": {"type": "string"}, "success": {"type": "boolean"}
            }, "required": ["rule_id", "success"]},
            "temperature": {"type": "number", "minimum": 0}
        },
        "required": ["rules"]
    })
}
/// Bayesian (Beta-Binomial) update + softmax query of grammar rule weights.
#[ix_skill(
    domain = "grammar",
    name = "grammar.weights",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::grammar_weights_schema"
)]
pub fn grammar_weights(p: Value) -> Result<Value, String> {
    handlers::grammar_weights(p)
}

// ---- grammar.evolve ------------------------------------------------------
fn grammar_evolve_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "species": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "proportion": {"type": "number"},
                "fitness": {"type": "number"}, "is_stable": {"type": "boolean"}
            }, "required": ["id", "proportion", "fitness"]}},
            "steps": {"type": "integer", "minimum": 1},
            "dt": {"type": "number", "minimum": 0},
            "prune_threshold": {"type": "number", "minimum": 0}
        },
        "required": ["species", "steps"]
    })
}
/// Grammar rule competition via replicator dynamics.
#[ix_skill(
    domain = "grammar",
    name = "grammar.evolve",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::grammar_evolve_schema"
)]
pub fn grammar_evolve(p: Value) -> Result<Value, String> {
    handlers::grammar_evolve(p)
}

// ---- grammar.search ------------------------------------------------------
fn grammar_search_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "grammar_ebnf": {"type": "string"},
            "max_iterations": {"type": "integer", "minimum": 1},
            "exploration": {"type": "number", "minimum": 0},
            "max_depth": {"type": "integer", "minimum": 1},
            "seed": {"type": "integer", "minimum": 0}
        },
        "required": ["grammar_ebnf"]
    })
}
/// Grammar-guided MCTS derivation search.
#[ix_skill(
    domain = "grammar",
    name = "grammar.search",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::grammar_search_schema"
)]
pub fn grammar_search(p: Value) -> Result<Value, String> {
    handlers::grammar_search(p)
}

// ---- rotation ------------------------------------------------------------
fn rotation_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["quaternion", "slerp", "euler_to_quat", "quat_to_euler", "rotate_point", "rotation_matrix"]},
            "axis": {"type": "array", "items": {"type": "number"}},
            "angle": {"type": "number"},
            "axis2": {"type": "array", "items": {"type": "number"}},
            "angle2": {"type": "number"},
            "t": {"type": "number"},
            "roll": {"type": "number"}, "pitch": {"type": "number"}, "yaw": {"type": "number"},
            "point": {"type": "array", "items": {"type": "number"}},
            "quaternion": {"type": "array", "items": {"type": "number"}}
        },
        "required": ["operation"]
    })
}
/// 3D rotation ops: quaternions, SLERP, Euler conversions, rotation matrices.
#[ix_skill(
    domain = "rotation",
    name = "rotation",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::rotation_schema"
)]
pub fn rotation(p: Value) -> Result<Value, String> {
    handlers::rotation(p)
}

// ---- number_theory -------------------------------------------------------
fn number_theory_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["sieve", "is_prime", "mod_pow", "gcd", "lcm", "mod_inverse", "prime_gaps"]},
            "limit": {"type": "integer", "minimum": 2},
            "n": {"type": "integer"}, "base": {"type": "integer"},
            "exp": {"type": "integer"}, "modulus": {"type": "integer"},
            "a": {"type": "integer"}, "b": {"type": "integer"}
        },
        "required": ["operation"]
    })
}
/// Number theory: primes, modular arithmetic, gcd/lcm.
#[ix_skill(
    domain = "number_theory",
    name = "number_theory",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::number_theory_schema"
)]
pub fn number_theory(p: Value) -> Result<Value, String> {
    handlers::number_theory(p)
}

// ---- fractal -------------------------------------------------------------
fn fractal_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["takagi", "hilbert", "peano", "morton_encode", "morton_decode"]},
            "n_points": {"type": "integer", "minimum": 2},
            "terms": {"type": "integer", "minimum": 1},
            "order": {"type": "integer", "minimum": 1},
            "x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"}
        },
        "required": ["operation"]
    })
}
/// Fractals: Takagi curve, Hilbert/Peano curves, Morton encoding.
#[ix_skill(
    domain = "fractal",
    name = "fractal",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::fractal_schema"
)]
pub fn fractal(p: Value) -> Result<Value, String> {
    handlers::fractal(p)
}

// ---- sedenion ------------------------------------------------------------
fn sedenion_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["multiply", "conjugate", "norm", "cayley_dickson_multiply"]},
            "a": {"type": "array", "items": {"type": "number"}},
            "b": {"type": "array", "items": {"type": "number"}}
        },
        "required": ["operation", "a"]
    })
}
/// Sedenion / octonion algebra: multiplication, conjugate, norm, Cayley-Dickson.
#[ix_skill(
    domain = "sedenion",
    name = "sedenion",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::sedenion_schema"
)]
pub fn sedenion(p: Value) -> Result<Value, String> {
    handlers::sedenion(p)
}

// ---- topo ----------------------------------------------------------------
fn topo_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["persistence", "betti_at_radius", "betti_curve"]},
            "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "max_dim": {"type": "integer", "minimum": 0},
            "max_radius": {"type": "number"},
            "radius": {"type": "number"},
            "n_steps": {"type": "integer"}
        },
        "required": ["operation", "points"]
    })
}
/// Topological data analysis: persistent homology + Betti numbers.
#[ix_skill(
    domain = "topo",
    name = "topo",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::topo_schema"
)]
pub fn topo(p: Value) -> Result<Value, String> {
    handlers::topo(p)
}

// ---- category ------------------------------------------------------------
fn category_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["monad_laws", "free_forgetful"]},
            "monad": {"type": "string", "enum": ["option", "result"]},
            "value": {"type": "integer"},
            "elements": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["operation"]
    })
}
/// Category theory: verify monad laws; free-forgetful adjunction.
#[ix_skill(
    domain = "category",
    name = "category",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::category_schema"
)]
pub fn category(p: Value) -> Result<Value, String> {
    handlers::category(p)
}

// ---- nn.forward ----------------------------------------------------------
fn nn_forward_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["dense_forward", "mse_loss", "bce_loss", "sinusoidal_encoding", "attention"]},
            "input": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "target": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "output_size": {"type": "integer"},
            "max_len": {"type": "integer"}, "d_model": {"type": "integer"},
            "seed": {"type": "integer"}
        },
        "required": ["operation"]
    })
}
/// Neural network forward pass: dense / loss / attention / positional encoding.
#[ix_skill(
    domain = "nn",
    name = "nn.forward",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::nn_forward_schema"
)]
pub fn nn_forward(p: Value) -> Result<Value, String> {
    handlers::nn_forward(p)
}

// ---- bandit --------------------------------------------------------------
fn bandit_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "algorithm": {"type": "string", "enum": ["epsilon_greedy", "ucb1", "thompson"]},
            "n_arms": {"type": "integer", "minimum": 1},
            "true_means": {"type": "array", "items": {"type": "number"}},
            "rounds": {"type": "integer", "minimum": 1},
            "epsilon": {"type": "number"}
        },
        "required": ["algorithm", "true_means", "rounds"]
    })
}
/// Multi-armed bandit simulation (epsilon-greedy / UCB1 / Thompson).
#[ix_skill(
    domain = "rl",
    name = "bandit",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::bandit_schema"
)]
pub fn bandit(p: Value) -> Result<Value, String> {
    handlers::bandit(p)
}

// ---- evolution -----------------------------------------------------------
fn evolution_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "algorithm": {"type": "string", "enum": ["genetic", "differential"]},
            "function": {"type": "string", "enum": ["sphere", "rosenbrock", "rastrigin"]},
            "dimensions": {"type": "integer", "minimum": 1},
            "generations": {"type": "integer", "minimum": 1},
            "population_size": {"type": "integer"},
            "mutation_rate": {"type": "number"}
        },
        "required": ["algorithm", "function", "dimensions", "generations"]
    })
}
/// Evolutionary optimization: genetic algorithm / differential evolution.
#[ix_skill(
    domain = "evolution",
    name = "evolution",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::evolution_schema"
)]
pub fn evolution(p: Value) -> Result<Value, String> {
    handlers::evolution(p)
}

// ---- random_forest -------------------------------------------------------
fn random_forest_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "x_train": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "y_train": {"type": "array", "items": {"type": "integer"}},
            "x_test": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "n_trees": {"type": "integer", "minimum": 1},
            "max_depth": {"type": "integer", "minimum": 1}
        },
        "required": ["x_train", "y_train", "x_test"]
    })
}
/// Random forest classifier: train and predict.
#[ix_skill(
    domain = "ensemble",
    name = "random_forest",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::random_forest_schema"
)]
pub fn random_forest(p: Value) -> Result<Value, String> {
    handlers::random_forest(p)
}

// ---- gradient_boosting ---------------------------------------------------
fn gradient_boosting_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "x_train": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "y_train": {"type": "array", "items": {"type": "integer"}},
            "x_test": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "n_estimators": {"type": "integer", "minimum": 1},
            "learning_rate": {"type": "number", "minimum": 0.001},
            "max_depth": {"type": "integer", "minimum": 1}
        },
        "required": ["x_train", "y_train", "x_test"]
    })
}
/// Gradient boosted trees classifier (binary + multiclass).
#[ix_skill(
    domain = "ensemble",
    name = "gradient_boosting",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::gradient_boosting_schema"
)]
pub fn gradient_boosting(p: Value) -> Result<Value, String> {
    handlers::gradient_boosting(p)
}

// ---- supervised ----------------------------------------------------------
fn supervised_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["linear_regression", "logistic_regression", "svm", "knn", "naive_bayes", "decision_tree", "metrics", "cross_validate", "confusion_matrix", "roc_auc"]},
            "x_train": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "y_train": {"type": "array", "items": {"type": "number"}},
            "x_test": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "k": {"type": "integer"}, "c": {"type": "number"},
            "max_depth": {"type": "integer"},
            "y_true": {"type": "array", "items": {"type": "number"}},
            "y_pred": {"type": "array", "items": {"type": "number"}},
            "y_scores": {"type": "array", "items": {"type": "number"}},
            "metric_type": {"type": "string", "enum": ["mse", "accuracy"]},
            "model": {"type": "string", "enum": ["knn", "decision_tree", "naive_bayes", "logistic_regression"]},
            "n_classes": {"type": "integer"}, "seed": {"type": "integer"}
        },
        "required": ["operation"]
    })
}
/// Supervised learning dispatcher: train / predict / metrics / cross-validate.
#[ix_skill(
    domain = "supervised",
    name = "supervised",
    governance = "empirical",
    schema_fn = "crate::skills::batch2::supervised_schema"
)]
pub fn supervised(p: Value) -> Result<Value, String> {
    handlers::supervised(p)
}

// ---- graph ---------------------------------------------------------------
fn graph_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["dijkstra", "shortest_path", "pagerank", "bfs", "dfs", "topological_sort"]},
            "n_nodes": {"type": "integer"},
            "edges": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "directed": {"type": "boolean"},
            "source": {"type": "integer"}, "target": {"type": "integer"},
            "damping": {"type": "number"}, "iterations": {"type": "integer"}
        },
        "required": ["operation", "n_nodes", "edges"]
    })
}
/// Graph algorithms: Dijkstra, PageRank, BFS/DFS, topological sort.
#[ix_skill(
    domain = "graph",
    name = "graph",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::graph_schema"
)]
pub fn graph(p: Value) -> Result<Value, String> {
    handlers::graph_ops(p)
}

// ---- hyperloglog ---------------------------------------------------------
fn hyperloglog_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["estimate", "merge"]},
            "items": {"type": "array"},
            "sets": {"type": "array", "items": {"type": "array"}},
            "precision": {"type": "integer", "minimum": 4, "maximum": 18}
        },
        "required": ["operation"]
    })
}
/// HyperLogLog cardinality estimation.
#[ix_skill(
    domain = "probabilistic",
    name = "hyperloglog",
    governance = "deterministic",
    schema_fn = "crate::skills::batch2::hyperloglog_schema"
)]
pub fn hyperloglog(p: Value) -> Result<Value, String> {
    handlers::hyperloglog(p)
}

// ---- governance.check ----------------------------------------------------
fn governance_check_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "context": {"type": "string"}
        },
        "required": ["action"]
    })
}
/// Check a proposed action against the Demerzel constitution.
#[ix_skill(
    domain = "governance",
    name = "governance.check",
    governance = "safety,reversible",
    schema_fn = "crate::skills::batch2::governance_check_schema"
)]
pub fn governance_check(p: Value) -> Result<Value, String> {
    handlers::governance_check(p)
}

// ---- governance.persona --------------------------------------------------
fn governance_persona_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "persona": {"type": "string", "enum": ["default", "kaizen-optimizer", "reflective-architect", "skeptical-auditor", "system-integrator"]}
        },
        "required": ["persona"]
    })
}
/// Load a Demerzel persona by name.
#[ix_skill(
    domain = "governance",
    name = "governance.persona",
    governance = "safety",
    schema_fn = "crate::skills::batch2::governance_persona_schema"
)]
pub fn governance_persona(p: Value) -> Result<Value, String> {
    handlers::governance_persona(p)
}

// ---- governance.policy ---------------------------------------------------
fn governance_policy_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "policy": {"type": "string", "enum": ["alignment", "rollback", "self-modification"]},
            "query": {"type": "string", "enum": ["thresholds", "triggers", "allowed"]}
        },
        "required": ["policy"]
    })
}
/// Query Demerzel governance policies.
#[ix_skill(
    domain = "governance",
    name = "governance.policy",
    governance = "safety",
    schema_fn = "crate::skills::batch2::governance_policy_schema"
)]
pub fn governance_policy(p: Value) -> Result<Value, String> {
    handlers::governance_policy(p)
}
