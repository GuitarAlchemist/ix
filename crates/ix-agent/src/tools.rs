//! Tool registry — defines MCP tools and dispatches calls.

use serde_json::{json, Value};

use crate::handlers;

/// An MCP tool definition.
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
    pub handler: fn(Value) -> Result<Value, String>,
}

/// Registry of all available tools.
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut reg = Self { tools: Vec::new() };
        reg.register_all();
        reg
    }

    /// List all tools as MCP tool definitions.
    pub fn list(&self) -> Value {
        let tools: Vec<Value> = self
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                })
            })
            .collect();
        json!({ "tools": tools })
    }

    /// Call a tool by name with the given arguments.
    pub fn call(&self, name: &str, arguments: Value) -> Result<Value, String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("Unknown tool: {}", name))?;
        (tool.handler)(arguments)
    }

    fn register_all(&mut self) {
        self.tools.push(Tool {
            name: "ix_stats",
            description: "Compute statistics (mean, std, min, max, median) on a list of numbers.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "List of numbers to compute statistics on"
                    }
                },
                "required": ["data"]
            }),
            handler: handlers::stats,
        });

        self.tools.push(Tool {
            name: "ix_distance",
            description: "Compute distance between two vectors (euclidean, cosine, or manhattan).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "a": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "First vector"
                    },
                    "b": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Second vector"
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["euclidean", "cosine", "manhattan"],
                        "description": "Distance metric to use"
                    }
                },
                "required": ["a", "b", "metric"]
            }),
            handler: handlers::distance,
        });

        self.tools.push(Tool {
            name: "ix_optimize",
            description: "Minimize a benchmark function (sphere, rosenbrock, rastrigin) using SGD, Adam, PSO, or simulated annealing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "function": {
                        "type": "string",
                        "enum": ["sphere", "rosenbrock", "rastrigin"],
                        "description": "Benchmark function to minimize"
                    },
                    "dimensions": {
                        "type": "integer",
                        "description": "Number of dimensions",
                        "minimum": 1
                    },
                    "method": {
                        "type": "string",
                        "enum": ["sgd", "adam", "pso", "annealing"],
                        "description": "Optimization method"
                    },
                    "max_iter": {
                        "type": "integer",
                        "description": "Maximum iterations",
                        "minimum": 1
                    }
                },
                "required": ["function", "dimensions", "method", "max_iter"]
            }),
            handler: handlers::optimize,
        });

        self.tools.push(Tool {
            name: "ix_linear_regression",
            description: "Fit ordinary least squares linear regression and return weights, bias, and predictions.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Feature matrix (rows=samples, cols=features)"
                    },
                    "y": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Target values"
                    }
                },
                "required": ["x", "y"]
            }),
            handler: handlers::linear_regression,
        });

        self.tools.push(Tool {
            name: "ix_kmeans",
            description: "K-Means clustering with K-Means++ initialization.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Data matrix (rows=samples, cols=features)"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of clusters",
                        "minimum": 1
                    },
                    "max_iter": {
                        "type": "integer",
                        "description": "Maximum iterations",
                        "minimum": 1
                    }
                },
                "required": ["data", "k", "max_iter"]
            }),
            handler: handlers::kmeans,
        });

        self.tools.push(Tool {
            name: "ix_fft",
            description: "Compute the Fast Fourier Transform of a real-valued signal. Returns frequency bins and magnitudes.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Real-valued signal samples"
                    }
                },
                "required": ["signal"]
            }),
            handler: handlers::fft,
        });

        self.tools.push(Tool {
            name: "ix_markov",
            description: "Analyze a Markov chain: compute stationary distribution after a number of steps.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "transition_matrix": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Row-stochastic transition matrix (rows sum to 1)"
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of power-iteration steps for stationary distribution",
                        "minimum": 1
                    }
                },
                "required": ["transition_matrix", "steps"]
            }),
            handler: handlers::markov,
        });

        self.tools.push(Tool {
            name: "ix_viterbi",
            description: "HMM Viterbi decoding: find the most likely hidden state sequence given observations.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "initial": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Initial state distribution (sums to 1)"
                    },
                    "transition": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "State transition matrix (row-stochastic)"
                    },
                    "emission": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Emission probability matrix (row-stochastic)"
                    },
                    "observations": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Observation sequence (indices into emission columns)"
                    }
                },
                "required": ["initial", "transition", "emission", "observations"]
            }),
            handler: handlers::viterbi,
        });

        self.tools.push(Tool {
            name: "ix_search",
            description: "Get information about search algorithms (A*, BFS, DFS) including descriptions and complexity.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["astar", "bfs", "dfs"],
                        "description": "Search algorithm to describe"
                    },
                    "description": {
                        "type": "boolean",
                        "description": "Whether to include a description"
                    }
                },
                "required": ["algorithm"]
            }),
            handler: handlers::search_info,
        });

        self.tools.push(Tool {
            name: "ix_game_nash",
            description: "Find Nash equilibria of a 2-player bimatrix game via support enumeration.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "payoff_a": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Player A payoff matrix"
                    },
                    "payoff_b": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Player B payoff matrix"
                    }
                },
                "required": ["payoff_a", "payoff_b"]
            }),
            handler: handlers::game_nash,
        });

        self.tools.push(Tool {
            name: "ix_chaos_lyapunov",
            description: "Compute the maximal Lyapunov exponent of the logistic map for a given parameter r.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "map": {
                        "type": "string",
                        "enum": ["logistic"],
                        "description": "Map type (currently only 'logistic')"
                    },
                    "parameter": {
                        "type": "number",
                        "description": "Map parameter (r for logistic map, 0 < r <= 4)"
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "Number of iterations for Lyapunov computation",
                        "minimum": 1
                    }
                },
                "required": ["map", "parameter", "iterations"]
            }),
            handler: handlers::chaos_lyapunov,
        });

        self.tools.push(Tool {
            name: "ix_adversarial_fgsm",
            description: "Fast Gradient Sign Method: compute adversarial perturbation of an input given its loss gradient.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Original input vector"
                    },
                    "gradient": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Loss gradient w.r.t. input"
                    },
                    "epsilon": {
                        "type": "number",
                        "description": "Perturbation magnitude"
                    }
                },
                "required": ["input", "gradient", "epsilon"]
            }),
            handler: handlers::adversarial_fgsm,
        });

        self.tools.push(Tool {
            name: "ix_bloom_filter",
            description: "Create a Bloom filter from items and check membership. Returns whether query items are (probably) in the set.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "check"],
                        "description": "Operation: 'create' inserts items, 'check' tests membership"
                    },
                    "items": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Items to insert into the Bloom filter"
                    },
                    "query": {
                        "type": "string",
                        "description": "Item to check membership for (used with 'check')"
                    },
                    "false_positive_rate": {
                        "type": "number",
                        "description": "Desired false positive rate (e.g. 0.01)",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["items", "false_positive_rate"]
            }),
            handler: handlers::bloom_filter,
        });

        self.tools.push(Tool {
            name: "ix_grammar_weights",
            description: "Bayesian (Beta-Binomial) update of grammar rule weights and softmax probability query. \
                Returns updated rule weights and selection probabilities.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "rules": {
                        "type": "array",
                        "description": "Grammar rules with weights",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":     { "type": "string" },
                                "alpha":  { "type": "number" },
                                "beta":   { "type": "number" },
                                "weight": { "type": "number" },
                                "level":  { "type": "integer" },
                                "source": { "type": "string" }
                            },
                            "required": ["id"]
                        }
                    },
                    "observation": {
                        "type": "object",
                        "description": "Optional: apply a Bayesian update to one rule",
                        "properties": {
                            "rule_id": { "type": "string" },
                            "success": { "type": "boolean" }
                        },
                        "required": ["rule_id", "success"]
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Softmax temperature (default 1.0)",
                        "minimum": 0
                    }
                },
                "required": ["rules"]
            }),
            handler: handlers::grammar_weights,
        });

        self.tools.push(Tool {
            name: "ix_grammar_evolve",
            description: "Simulate grammar rule competition via replicator dynamics. \
                Returns the final species proportions, full trajectory, and detected ESS.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "species": {
                        "type": "array",
                        "description": "Initial grammar species",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":          { "type": "string" },
                                "proportion":  { "type": "number" },
                                "fitness":     { "type": "number" },
                                "is_stable":   { "type": "boolean" }
                            },
                            "required": ["id", "proportion", "fitness"]
                        }
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of simulation steps",
                        "minimum": 1
                    },
                    "dt": {
                        "type": "number",
                        "description": "Time step (default 0.05)",
                        "minimum": 0
                    },
                    "prune_threshold": {
                        "type": "number",
                        "description": "Proportion below which species are pruned (default 1e-6)",
                        "minimum": 0
                    }
                },
                "required": ["species", "steps"]
            }),
            handler: handlers::grammar_evolve,
        });

        self.tools.push(Tool {
            name: "ix_grammar_search",
            description: "Grammar-guided MCTS derivation search. \
                Finds the best sentence derivation from an EBNF grammar using Monte Carlo Tree Search.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "grammar_ebnf": {
                        "type": "string",
                        "description": "Grammar in EBNF notation (one rule per line: name ::= alt | alt)"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "MCTS iterations (default 500)",
                        "minimum": 1
                    },
                    "exploration": {
                        "type": "number",
                        "description": "UCB1 exploration constant (default 1.41)",
                        "minimum": 0
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max grammar expansion depth (default 20)",
                        "minimum": 1
                    },
                    "seed": {
                        "type": "integer",
                        "description": "RNG seed for reproducibility (default 42)",
                        "minimum": 0
                    }
                },
                "required": ["grammar_ebnf"]
            }),
            handler: handlers::grammar_search,
        });

        // ── New crate tools (Phase 4) ──────────────────────────────

        self.tools.push(Tool {
            name: "ix_rotation",
            description: "3D rotation operations: quaternion from axis-angle, SLERP interpolation, Euler angle conversion, rotation matrix.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["quaternion", "slerp", "euler_to_quat", "quat_to_euler", "rotate_point", "rotation_matrix"],
                        "description": "Rotation operation to perform"
                    },
                    "axis": { "type": "array", "items": { "type": "number" }, "description": "Rotation axis [x,y,z]" },
                    "angle": { "type": "number", "description": "Rotation angle in radians" },
                    "axis2": { "type": "array", "items": { "type": "number" }, "description": "Second rotation axis (for SLERP)" },
                    "angle2": { "type": "number", "description": "Second angle (for SLERP)" },
                    "t": { "type": "number", "description": "Interpolation parameter 0..1 (for SLERP)" },
                    "roll": { "type": "number", "description": "Roll in radians (for Euler)" },
                    "pitch": { "type": "number", "description": "Pitch in radians (for Euler)" },
                    "yaw": { "type": "number", "description": "Yaw in radians (for Euler)" },
                    "point": { "type": "array", "items": { "type": "number" }, "description": "Point [x,y,z] to rotate" },
                    "quaternion": { "type": "array", "items": { "type": "number" }, "description": "Quaternion [w,x,y,z]" }
                },
                "required": ["operation"]
            }),
            handler: handlers::rotation,
        });

        self.tools.push(Tool {
            name: "ix_number_theory",
            description: "Number theory: prime sieve, primality testing, modular arithmetic (mod_pow, gcd, lcm, mod_inverse).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["sieve", "is_prime", "mod_pow", "gcd", "lcm", "mod_inverse", "prime_gaps"],
                        "description": "Number theory operation"
                    },
                    "limit": { "type": "integer", "description": "Upper limit for sieve", "minimum": 2 },
                    "n": { "type": "integer", "description": "Number to test for primality" },
                    "base": { "type": "integer", "description": "Base for mod_pow" },
                    "exp": { "type": "integer", "description": "Exponent for mod_pow" },
                    "modulus": { "type": "integer", "description": "Modulus for mod_pow/mod_inverse" },
                    "a": { "type": "integer", "description": "First number for gcd/lcm" },
                    "b": { "type": "integer", "description": "Second number for gcd/lcm" }
                },
                "required": ["operation"]
            }),
            handler: handlers::number_theory,
        });

        self.tools.push(Tool {
            name: "ix_fractal",
            description: "Generate fractal data: Takagi curve, Hilbert/Peano space-filling curves, Morton encoding.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["takagi", "hilbert", "peano", "morton_encode", "morton_decode"],
                        "description": "Fractal operation"
                    },
                    "n_points": { "type": "integer", "description": "Number of points for Takagi curve", "minimum": 2 },
                    "terms": { "type": "integer", "description": "Number of terms for Takagi", "minimum": 1 },
                    "order": { "type": "integer", "description": "Order for space-filling curves", "minimum": 1 },
                    "x": { "type": "integer", "description": "X coordinate for Morton encode" },
                    "y": { "type": "integer", "description": "Y coordinate for Morton encode" },
                    "z": { "type": "integer", "description": "Z-order value for Morton decode" }
                },
                "required": ["operation"]
            }),
            handler: handlers::fractal,
        });

        self.tools.push(Tool {
            name: "ix_sedenion",
            description: "Hypercomplex algebra: sedenion/octonion multiplication, conjugate, norm, Cayley-Dickson construction.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["multiply", "conjugate", "norm", "cayley_dickson_multiply"],
                        "description": "Sedenion operation"
                    },
                    "a": { "type": "array", "items": { "type": "number" }, "description": "First element components (16 for sedenion)" },
                    "b": { "type": "array", "items": { "type": "number" }, "description": "Second element components (for multiply)" }
                },
                "required": ["operation", "a"]
            }),
            handler: handlers::sedenion,
        });

        self.tools.push(Tool {
            name: "ix_topo",
            description: "Topological data analysis: persistent homology, Betti numbers, Betti curves from point clouds.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["persistence", "betti_at_radius", "betti_curve"],
                        "description": "TDA operation"
                    },
                    "points": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Point cloud (rows=points, cols=dimensions)"
                    },
                    "max_dim": { "type": "integer", "description": "Maximum homology dimension (default 1)", "minimum": 0 },
                    "max_radius": { "type": "number", "description": "Maximum filtration radius (default 2.0)" },
                    "radius": { "type": "number", "description": "Radius for betti_at_radius" },
                    "n_steps": { "type": "integer", "description": "Number of steps for betti_curve (default 50)" }
                },
                "required": ["operation", "points"]
            }),
            handler: handlers::topo,
        });

        self.tools.push(Tool {
            name: "ix_category",
            description: "Category theory: verify monad laws for Option/Result monads with sample functions.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["monad_laws", "free_forgetful"],
                        "description": "Category theory operation"
                    },
                    "monad": {
                        "type": "string",
                        "enum": ["option", "result"],
                        "description": "Which monad to verify laws for"
                    },
                    "value": { "type": "integer", "description": "Value to test monad laws with" },
                    "elements": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Elements for free-forgetful adjunction"
                    }
                },
                "required": ["operation"]
            }),
            handler: handlers::category,
        });

        self.tools.push(Tool {
            name: "ix_nn_forward",
            description: "Neural network forward pass: dense layer, MSE/BCE loss, attention, positional encodings.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["dense_forward", "mse_loss", "bce_loss", "sinusoidal_encoding", "attention"],
                        "description": "Neural network operation"
                    },
                    "input": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Input matrix (rows=batch, cols=features)"
                    },
                    "target": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Target matrix (for loss computation)"
                    },
                    "output_size": { "type": "integer", "description": "Output size for dense layer" },
                    "max_len": { "type": "integer", "description": "Max sequence length for positional encoding" },
                    "d_model": { "type": "integer", "description": "Model dimension for positional encoding" },
                    "seed": { "type": "integer", "description": "RNG seed (default 42)" }
                },
                "required": ["operation"]
            }),
            handler: handlers::nn_forward,
        });

        self.tools.push(Tool {
            name: "ix_bandit",
            description: "Multi-armed bandit simulation: run epsilon-greedy, UCB1, or Thompson sampling for N rounds.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["epsilon_greedy", "ucb1", "thompson"],
                        "description": "Bandit algorithm"
                    },
                    "n_arms": { "type": "integer", "description": "Number of arms", "minimum": 1 },
                    "true_means": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "True mean rewards for each arm (for simulation)"
                    },
                    "rounds": { "type": "integer", "description": "Number of rounds to simulate", "minimum": 1 },
                    "epsilon": { "type": "number", "description": "Epsilon for epsilon-greedy (default 0.1)" }
                },
                "required": ["algorithm", "true_means", "rounds"]
            }),
            handler: handlers::bandit,
        });

        self.tools.push(Tool {
            name: "ix_evolution",
            description: "Evolutionary optimization: genetic algorithm or differential evolution on benchmark functions.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["genetic", "differential"],
                        "description": "Evolution algorithm"
                    },
                    "function": {
                        "type": "string",
                        "enum": ["sphere", "rosenbrock", "rastrigin"],
                        "description": "Benchmark function to minimize"
                    },
                    "dimensions": { "type": "integer", "description": "Number of dimensions", "minimum": 1 },
                    "generations": { "type": "integer", "description": "Number of generations", "minimum": 1 },
                    "population_size": { "type": "integer", "description": "Population size (default 50)" },
                    "mutation_rate": { "type": "number", "description": "Mutation rate for GA (default 0.1)" }
                },
                "required": ["algorithm", "function", "dimensions", "generations"]
            }),
            handler: handlers::evolution,
        });

        self.tools.push(Tool {
            name: "ix_random_forest",
            description: "Random forest classifier: train on data and predict class labels with probability estimates.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "x_train": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Training feature matrix"
                    },
                    "y_train": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Training labels (class indices)"
                    },
                    "x_test": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Test feature matrix to predict"
                    },
                    "n_trees": { "type": "integer", "description": "Number of trees (default 10)", "minimum": 1 },
                    "max_depth": { "type": "integer", "description": "Max tree depth (default 5)", "minimum": 1 }
                },
                "required": ["x_train", "y_train", "x_test"]
            }),
            handler: handlers::random_forest,
        });

        self.tools.push(Tool {
            name: "ix_supervised",
            description: "Supervised learning: train and predict with linear/logistic regression, SVM, KNN, naive Bayes, decision tree. Also compute classification/regression metrics.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["linear_regression", "logistic_regression", "svm", "knn", "naive_bayes", "decision_tree", "metrics"],
                        "description": "Algorithm or 'metrics' for evaluation"
                    },
                    "x_train": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Training features matrix"
                    },
                    "y_train": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Training labels (class indices for classification, values for regression)"
                    },
                    "x_test": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Test features matrix"
                    },
                    "k": { "type": "integer", "description": "K for KNN (default 3)" },
                    "c": { "type": "number", "description": "Regularization for SVM (default 1.0)" },
                    "max_depth": { "type": "integer", "description": "Max depth for decision tree (default 5)" },
                    "y_true": { "type": "array", "items": { "type": "number" }, "description": "True labels (for metrics)" },
                    "y_pred": { "type": "array", "items": { "type": "number" }, "description": "Predicted labels (for metrics)" },
                    "metric_type": { "type": "string", "enum": ["mse", "accuracy"], "description": "Metric type: 'mse' for regression, 'accuracy' for classification" }
                },
                "required": ["operation"]
            }),
            handler: handlers::supervised,
        });

        self.tools.push(Tool {
            name: "ix_graph",
            description: "Graph algorithms: Dijkstra shortest path, BFS, DFS, PageRank, topological sort on weighted directed/undirected graphs.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["dijkstra", "shortest_path", "pagerank", "bfs", "dfs", "topological_sort"],
                        "description": "Graph algorithm"
                    },
                    "n_nodes": { "type": "integer", "description": "Number of nodes in the graph" },
                    "edges": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Edges as [from, to, weight] triples"
                    },
                    "directed": { "type": "boolean", "description": "Whether graph is directed (default true)" },
                    "source": { "type": "integer", "description": "Source node for path/traversal algorithms" },
                    "target": { "type": "integer", "description": "Target node for shortest_path" },
                    "damping": { "type": "number", "description": "Damping factor for PageRank (default 0.85)" },
                    "iterations": { "type": "integer", "description": "Iterations for PageRank (default 100)" }
                },
                "required": ["operation", "n_nodes", "edges"]
            }),
            handler: handlers::graph_ops,
        });

        self.tools.push(Tool {
            name: "ix_hyperloglog",
            description: "HyperLogLog cardinality estimation: estimate unique item count with configurable precision, or merge multiple sketches.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["estimate", "merge"],
                        "description": "'estimate' for single set, 'merge' for union of multiple sets"
                    },
                    "items": {
                        "type": "array",
                        "description": "Items to count (strings or numbers) — for 'estimate'"
                    },
                    "sets": {
                        "type": "array",
                        "items": { "type": "array" },
                        "description": "Array of item arrays — for 'merge'"
                    },
                    "precision": { "type": "integer", "description": "HLL precision 4-18 (default 14, ~0.81% error)", "minimum": 4, "maximum": 18 }
                },
                "required": ["operation"]
            }),
            handler: handlers::hyperloglog,
        });

        self.tools.push(Tool {
            name: "ix_pipeline",
            description: "DAG pipeline analysis: define steps with dependencies, get topological order, parallel execution levels, and critical path info.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["info"],
                        "description": "Pipeline operation"
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "Step identifier" },
                                "description": { "type": "string", "description": "Step description" },
                                "depends_on": { "type": "array", "items": { "type": "string" }, "description": "IDs of prerequisite steps" }
                            },
                            "required": ["id"]
                        },
                        "description": "Pipeline step definitions"
                    }
                },
                "required": ["operation", "steps"]
            }),
            handler: handlers::pipeline_exec,
        });

        self.tools.push(Tool {
            name: "ix_cache",
            description: "In-memory cache operations: set, get, delete, or list keys.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["set", "get", "delete", "keys"],
                        "description": "Cache operation to perform"
                    },
                    "key": {
                        "type": "string",
                        "description": "Cache key (required for set/get/delete)"
                    },
                    "value": {
                        "description": "Value to store (required for set, any JSON value)"
                    }
                },
                "required": ["operation"]
            }),
            handler: handlers::cache_op,
        });

        // ── Governance & federation tools ─────────────────────────────

        self.tools.push(Tool {
            name: "ix_governance_check",
            description: "Check a proposed action against the Demerzel constitution for compliance",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The proposed action to check for compliance"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context for the action"
                    }
                },
                "required": ["action"]
            }),
            handler: handlers::governance_check,
        });

        self.tools.push(Tool {
            name: "ix_governance_persona",
            description: "Load a Demerzel persona by name — returns capabilities, constraints, voice, interaction patterns",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "persona": {
                        "type": "string",
                        "enum": ["default", "kaizen-optimizer", "reflective-architect", "skeptical-auditor", "system-integrator"],
                        "description": "Persona name to load"
                    }
                },
                "required": ["persona"]
            }),
            handler: handlers::governance_persona,
        });

        self.tools.push(Tool {
            name: "ix_governance_belief",
            description: "Manage beliefs with tetravalent logic (True/False/Unknown/Contradictory)",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "update", "resolve"],
                        "description": "Belief operation"
                    },
                    "proposition": {
                        "type": "string",
                        "description": "The proposition to evaluate"
                    },
                    "truth_value": {
                        "type": "string",
                        "enum": ["T", "F", "U", "C"],
                        "description": "Initial truth value (True/False/Unknown/Contradictory)"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level 0.0–1.0"
                    },
                    "supporting": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Supporting evidence claims"
                    },
                    "contradicting": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Contradicting evidence claims"
                    }
                },
                "required": ["operation", "proposition"]
            }),
            handler: handlers::governance_belief,
        });

        self.tools.push(Tool {
            name: "ix_governance_policy",
            description: "Query Demerzel governance policies — alignment thresholds, rollback triggers, self-modification rules",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "policy": {
                        "type": "string",
                        "enum": ["alignment", "rollback", "self-modification"],
                        "description": "Policy to query"
                    },
                    "query": {
                        "type": "string",
                        "enum": ["thresholds", "triggers", "allowed"],
                        "description": "What aspect of the policy to query"
                    }
                },
                "required": ["policy"]
            }),
            handler: handlers::governance_policy,
        });

        self.tools.push(Tool {
            name: "ix_federation_discover",
            description: "Discover capabilities across the GuitarAlchemist ecosystem (ix, tars, ga)",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Filter by domain (e.g. 'math', 'grammar', 'music-theory')"
                    },
                    "query": {
                        "type": "string",
                        "description": "Free-text search across tool names and descriptions"
                    }
                }
            }),
            handler: handlers::federation_discover,
        });
    }
}
