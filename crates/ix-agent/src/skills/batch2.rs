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
use crate::schema::{object, Prop};
use ix_skill_macros::ix_skill;
use serde_json::Value;

// ---- optimize ------------------------------------------------------------
fn optimize_schema() -> Value {
    object(
        vec![
            ("function", Prop::string().enum_of(&["sphere", "rosenbrock", "rastrigin"])),
            ("dimensions", Prop::integer().minimum(1)),
            ("method", Prop::string().enum_of(&["sgd", "adam", "pso", "annealing"])),
            ("max_iter", Prop::integer().minimum(1)),
        ],
        &["function", "dimensions", "method", "max_iter"],
    )
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
    object(
        vec![
            ("transition_matrix", Prop::num_matrix()),
            ("steps", Prop::integer().minimum(1)),
        ],
        &["transition_matrix", "steps"],
    )
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
    object(
        vec![
            ("initial", Prop::num_array()),
            ("transition", Prop::num_matrix()),
            ("emission", Prop::num_matrix()),
            ("observations", Prop::int_array()),
        ],
        &["initial", "transition", "emission", "observations"],
    )
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
    object(
        vec![
            ("algorithm", Prop::string().enum_of(&["astar", "bfs", "dfs"])),
            ("description", Prop::boolean()),
        ],
        &["algorithm"],
    )
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
    object(
        vec![
            ("payoff_a", Prop::num_matrix()),
            ("payoff_b", Prop::num_matrix()),
        ],
        &["payoff_a", "payoff_b"],
    )
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
    object(
        vec![
            ("map", Prop::string().enum_of(&["logistic"])),
            ("parameter", Prop::number()),
            ("iterations", Prop::integer().minimum(1)),
        ],
        &["map", "parameter", "iterations"],
    )
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
    object(
        vec![
            ("input", Prop::num_array()),
            ("gradient", Prop::num_array()),
            ("epsilon", Prop::number()),
        ],
        &["input", "gradient", "epsilon"],
    )
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
    object(
        vec![
            ("operation", Prop::string().enum_of(&["create", "check"])),
            ("items", Prop::str_array()),
            ("query", Prop::string()),
            ("false_positive_rate", Prop::number().minimum(0).maximum(1)),
        ],
        &["items", "false_positive_rate"],
    )
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
    object(
        vec![
            (
                "rules",
                Prop::array_of(Prop::object(
                    vec![
                        ("id", Prop::string()),
                        ("alpha", Prop::number()),
                        ("beta", Prop::number()),
                        ("weight", Prop::number()),
                        ("level", Prop::integer()),
                        ("source", Prop::string()),
                    ],
                    &["id"],
                )),
            ),
            (
                "observation",
                Prop::object(
                    vec![("rule_id", Prop::string()), ("success", Prop::boolean())],
                    &["rule_id", "success"],
                ),
            ),
            ("temperature", Prop::number().minimum(0)),
        ],
        &["rules"],
    )
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
    object(
        vec![
            (
                "species",
                Prop::array_of(Prop::object(
                    vec![
                        ("id", Prop::string()),
                        ("proportion", Prop::number()),
                        ("fitness", Prop::number()),
                        ("is_stable", Prop::boolean()),
                    ],
                    &["id", "proportion", "fitness"],
                )),
            ),
            ("steps", Prop::integer().minimum(1)),
            ("dt", Prop::number().minimum(0)),
            ("prune_threshold", Prop::number().minimum(0)),
        ],
        &["species", "steps"],
    )
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
    object(
        vec![
            ("grammar_ebnf", Prop::string()),
            ("max_iterations", Prop::integer().minimum(1)),
            ("exploration", Prop::number().minimum(0)),
            ("max_depth", Prop::integer().minimum(1)),
            ("seed", Prop::integer().minimum(0)),
        ],
        &["grammar_ebnf"],
    )
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
    object(
        vec![
            (
                "operation",
                Prop::string().enum_of(&[
                    "quaternion",
                    "slerp",
                    "euler_to_quat",
                    "quat_to_euler",
                    "rotate_point",
                    "rotation_matrix",
                ]),
            ),
            ("axis", Prop::num_array()),
            ("angle", Prop::number()),
            ("axis2", Prop::num_array()),
            ("angle2", Prop::number()),
            ("t", Prop::number()),
            ("roll", Prop::number()),
            ("pitch", Prop::number()),
            ("yaw", Prop::number()),
            ("point", Prop::num_array()),
            ("quaternion", Prop::num_array()),
        ],
        &["operation"],
    )
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
    object(
        vec![
            (
                "operation",
                Prop::string().enum_of(&[
                    "sieve",
                    "is_prime",
                    "mod_pow",
                    "gcd",
                    "lcm",
                    "mod_inverse",
                    "prime_gaps",
                ]),
            ),
            ("limit", Prop::integer().minimum(2)),
            ("n", Prop::integer()),
            ("base", Prop::integer()),
            ("exp", Prop::integer()),
            ("modulus", Prop::integer()),
            ("a", Prop::integer()),
            ("b", Prop::integer()),
        ],
        &["operation"],
    )
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
    object(
        vec![
            (
                "operation",
                Prop::string().enum_of(&["takagi", "hilbert", "peano", "morton_encode", "morton_decode"]),
            ),
            ("n_points", Prop::integer().minimum(2)),
            ("terms", Prop::integer().minimum(1)),
            ("order", Prop::integer().minimum(1)),
            ("x", Prop::integer()),
            ("y", Prop::integer()),
            ("z", Prop::integer()),
        ],
        &["operation"],
    )
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
    object(
        vec![
            (
                "operation",
                Prop::string().enum_of(&["multiply", "conjugate", "norm", "cayley_dickson_multiply"]),
            ),
            ("a", Prop::num_array()),
            ("b", Prop::num_array()),
        ],
        &["operation", "a"],
    )
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
    object(
        vec![
            (
                "operation",
                Prop::string().enum_of(&["persistence", "betti_at_radius", "betti_curve"]),
            ),
            ("points", Prop::num_matrix()),
            ("max_dim", Prop::integer().minimum(0)),
            ("max_radius", Prop::number()),
            ("radius", Prop::number()),
            ("n_steps", Prop::integer()),
        ],
        &["operation", "points"],
    )
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
    object(
        vec![
            ("operation", Prop::string().enum_of(&["monad_laws", "free_forgetful"])),
            ("monad", Prop::string().enum_of(&["option", "result"])),
            ("value", Prop::integer()),
            ("elements", Prop::int_array()),
        ],
        &["operation"],
    )
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
    object(
        vec![
            (
                "operation",
                Prop::string().enum_of(&[
                    "dense_forward",
                    "mse_loss",
                    "bce_loss",
                    "sinusoidal_encoding",
                    "attention",
                ]),
            ),
            ("input", Prop::num_matrix()),
            ("target", Prop::num_matrix()),
            ("output_size", Prop::integer()),
            ("max_len", Prop::integer()),
            ("d_model", Prop::integer()),
            ("seed", Prop::integer()),
        ],
        &["operation"],
    )
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
    object(
        vec![
            (
                "algorithm",
                Prop::string().enum_of(&["epsilon_greedy", "ucb1", "thompson"]),
            ),
            ("n_arms", Prop::integer().minimum(1)),
            ("true_means", Prop::num_array()),
            ("rounds", Prop::integer().minimum(1)),
            ("epsilon", Prop::number()),
        ],
        &["algorithm", "true_means", "rounds"],
    )
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
    object(
        vec![
            ("algorithm", Prop::string().enum_of(&["genetic", "differential"])),
            ("function", Prop::string().enum_of(&["sphere", "rosenbrock", "rastrigin"])),
            ("dimensions", Prop::integer().minimum(1)),
            ("generations", Prop::integer().minimum(1)),
            ("population_size", Prop::integer()),
            ("mutation_rate", Prop::number()),
        ],
        &["algorithm", "function", "dimensions", "generations"],
    )
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
    object(
        vec![
            ("x_train", Prop::num_matrix()),
            ("y_train", Prop::int_array()),
            ("x_test", Prop::num_matrix()),
            ("n_trees", Prop::integer().minimum(1)),
            ("max_depth", Prop::integer().minimum(1)),
        ],
        &["x_train", "y_train", "x_test"],
    )
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
    object(
        vec![
            ("x_train", Prop::num_matrix()),
            ("y_train", Prop::int_array()),
            ("x_test", Prop::num_matrix()),
            ("n_estimators", Prop::integer().minimum(1)),
            ("learning_rate", Prop::number().minimum(0.001)),
            ("max_depth", Prop::integer().minimum(1)),
        ],
        &["x_train", "y_train", "x_test"],
    )
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
    object(
        vec![
            (
                "operation",
                Prop::string().enum_of(&[
                    "linear_regression",
                    "logistic_regression",
                    "svm",
                    "knn",
                    "naive_bayes",
                    "decision_tree",
                    "metrics",
                    "cross_validate",
                    "confusion_matrix",
                    "roc_auc",
                ]),
            ),
            ("x_train", Prop::num_matrix()),
            ("y_train", Prop::num_array()),
            ("x_test", Prop::num_matrix()),
            ("k", Prop::integer()),
            ("c", Prop::number()),
            ("max_depth", Prop::integer()),
            ("y_true", Prop::num_array()),
            ("y_pred", Prop::num_array()),
            ("y_scores", Prop::num_array()),
            ("metric_type", Prop::string().enum_of(&["mse", "accuracy"])),
            (
                "model",
                Prop::string().enum_of(&["knn", "decision_tree", "naive_bayes", "logistic_regression"]),
            ),
            ("n_classes", Prop::integer()),
            ("seed", Prop::integer()),
        ],
        &["operation"],
    )
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
    object(
        vec![
            (
                "operation",
                Prop::string().enum_of(&[
                    "dijkstra",
                    "shortest_path",
                    "pagerank",
                    "bfs",
                    "dfs",
                    "topological_sort",
                ]),
            ),
            ("n_nodes", Prop::integer()),
            ("edges", Prop::num_matrix()),
            ("directed", Prop::boolean()),
            ("source", Prop::integer()),
            ("target", Prop::integer()),
            ("damping", Prop::number()),
            ("iterations", Prop::integer()),
        ],
        &["operation", "n_nodes", "edges"],
    )
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
    object(
        vec![
            ("operation", Prop::string().enum_of(&["estimate", "merge"])),
            ("items", Prop::array_any()),
            ("sets", Prop::array_of(Prop::array_any())),
            ("precision", Prop::integer().minimum(4).maximum(18)),
        ],
        &["operation"],
    )
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
    object(
        vec![("action", Prop::string()), ("context", Prop::string())],
        &["action"],
    )
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
    object(
        vec![(
            "persona",
            Prop::string().enum_of(&[
                "default",
                "kaizen-optimizer",
                "reflective-architect",
                "skeptical-auditor",
                "system-integrator",
            ]),
        )],
        &["persona"],
    )
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
    object(
        vec![
            (
                "policy",
                Prop::string().enum_of(&["alignment", "rollback", "self-modification"]),
            ),
            ("query", Prop::string().enum_of(&["thresholds", "triggers", "allowed"])),
        ],
        &["policy"],
    )
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
