# MachinDeOuf

[![CI](https://github.com/anthropics/MachinDeOuf/actions/workflows/ci.yml/badge.svg)](https://github.com/anthropics/MachinDeOuf/actions/workflows/ci.yml)

A Rust workspace of composable ML/math algorithms, designed to be exposed as **Claude Code skills** via an MCP server and CLI.

27 crates. Pure Rust. No external ML frameworks.

## Quick Start

```bash
# Build everything
cargo build --workspace

# Run tests
cargo test --workspace

# Run the CLI
cargo run -p machin-skill -- optimize --algo pso --function sphere --dim 10

# Start the MCP server (for Claude Code integration)
cargo run -p machin-agent
```

## Crates

### Core Math & Optimization
| Crate | Description |
|-------|-------------|
| **machin-math** | Linear algebra, statistics, distances, activations, calculus, random, hyperbolic geometry |
| **machin-optimize** | SGD, Adam, simulated annealing, particle swarm optimization |

### Supervised Learning
| Crate | Description |
|-------|-------------|
| **machin-supervised** | Linear/logistic regression, decision trees (CART), KNN, Naive Bayes, SVM, metrics |
| **machin-ensemble** | Random forest with bootstrap aggregation and random feature subsets |

### Unsupervised Learning
| Crate | Description |
|-------|-------------|
| **machin-unsupervised** | K-Means clustering, DBSCAN, PCA (power iteration) |

### Deep Learning & RL
| Crate | Description |
|-------|-------------|
| **machin-nn** | Neural network layers, loss functions, backpropagation |
| **machin-rl** | Multi-armed bandits (epsilon-greedy, UCB1, Thompson sampling), Q-learning |
| **machin-evolution** | Genetic algorithms, differential evolution |

### Search & Graphs
| Crate | Description |
|-------|-------------|
| **machin-search** | A\*, Q\* (learned heuristic), MCTS, minimax/alpha-beta, BFS/DFS, hill climbing, tabu search |
| **machin-graph** | Graph algorithms, Markov chains, HMM/Viterbi/Baum-Welch, state spaces, agent routing |
| **machin-game** | Nash equilibria, Shapley value, auctions, evolutionary dynamics, mean field games |

### Signal & Chaos
| Crate | Description |
|-------|-------------|
| **machin-signal** | FFT, wavelets, FIR/IIR filters, Kalman filter, spectral analysis, DCT |
| **machin-chaos** | Lyapunov exponents, bifurcation diagrams, strange attractors, fractal dimensions, delay embedding, chaos control |

### Security & Privacy
| Crate | Description |
|-------|-------------|
| **machin-adversarial** | FGSM, PGD, C&W attacks, adversarial training, poisoning detection, differential privacy, robustness evaluation |

### Advanced Math
| Crate | Description |
|-------|-------------|
| **machin-dynamics** | Inverse kinematics chains, Lie groups/algebras (SO(3), SE(3)), neural ODEs |
| **machin-topo** | Persistent homology, simplicial complexes, Betti numbers, topological data analysis |
| **machin-ktheory** | Graph K-theory, Grothendieck K0/K1, Mayer-Vietoris sequences |
| **machin-category** | Functors, natural transformations, monads, category theory primitives |
| **machin-grammar** | Formal grammars: context-free grammars, Earley parser, CYK, Chomsky normal form |

### Infrastructure
| Crate | Description |
|-------|-------------|
| **machin-gpu** | WGPU compute shaders for cosine similarity, matrix multiply, batch vector search (Vulkan/DX12/Metal) |
| **machin-cache** | Embedded Redis-like cache with sharded concurrency, TTL, LRU eviction, pub/sub, RESP protocol server |
| **machin-pipeline** | DAG executor with topological sort, parallel branch execution, memoization, critical path analysis |
| **machin-probabilistic** | Bloom filter, Count-Min sketch, HyperLogLog, Cuckoo filter |
| **machin-io** | CSV, JSON, file watcher, named pipes, TCP, HTTP client, WebSocket |

### Integration
| Crate | Description |
|-------|-------------|
| **machin-agent** | MCP server exposing all algorithms as Claude Code tools via JSON-RPC over stdio |
| **machin-skill** | CLI binary (`machin`) for direct command-line access to all algorithms |
| **machin-demo** | egui desktop app with 16 interactive demo tabs (stats, regression, clustering, neural nets, chaos, GPU, transformer, etc.) |

## Claude Code Integration

### MCP Server

Register MachinDeOuf as a Claude Code MCP server in `.mcp.json`:

```json
{
  "mcpServers": {
    "machin": {
      "command": "cargo",
      "args": ["run", "-p", "machin-agent"]
    }
  }
}
```

Claude can then call tools like `machin_kmeans`, `machin_viterbi`, `machin_optimize`, etc. directly during conversations.

### Skills

10 Claude Code skills are included in `.claude/skills/`:

- `/machin-optimize` — Function minimization
- `/machin-cluster` — Data clustering
- `/machin-search` — Pathfinding and game search
- `/machin-chaos` — Dynamical systems analysis
- `/machin-hmm` — Hidden Markov Model decoding
- `/machin-adversarial` — Robustness testing
- `/machin-game` — Game theory analysis
- `/machin-pipeline` — DAG orchestration
- `/machin-signal` — Signal processing
- `/machin-benchmark` — Performance profiling

## Examples

16 runnable examples organized by domain:

| # | Example | Domain | Source |
|---|---------|--------|--------|
| 1 | **PSO Rosenbrock** — Minimize a 10D cost function | Optimization | [`examples/optimization/pso_rosenbrock.rs`](examples/optimization/pso_rosenbrock.rs) |
| 2 | **Decision Tree** — CART classification with probabilities | Supervised | [`examples/supervised/decision_tree.rs`](examples/supervised/decision_tree.rs) |
| 3 | **K-Means Clustering** — Segment data into k groups | Unsupervised | [`examples/unsupervised/kmeans_clustering.rs`](examples/unsupervised/kmeans_clustering.rs) |
| 4 | **DBSCAN Anomaly** — Density-based clustering + noise detection | Unsupervised | [`examples/unsupervised/dbscan_anomaly.rs`](examples/unsupervised/dbscan_anomaly.rs) |
| 5 | **Viterbi HMM** — Decode hidden states, Baum-Welch training | Sequence | [`examples/sequence/viterbi_hmm.rs`](examples/sequence/viterbi_hmm.rs) |
| 6 | **Nash Equilibrium** — Prisoner's Dilemma analysis | Game Theory | [`examples/game-theory/nash_equilibrium.rs`](examples/game-theory/nash_equilibrium.rs) |
| 7 | **A\* & Q\* Search** — Hand-crafted vs learned heuristics | Search | [`examples/search/astar_qstar.rs`](examples/search/astar_qstar.rs) |
| 8 | **Logistic Map** — Lyapunov exponents and chaos detection | Chaos | [`examples/chaos/logistic_map.rs`](examples/chaos/logistic_map.rs) |
| 9 | **DAG Pipeline** — Parallel data flow with memoization | Pipeline | [`examples/pipeline/dag_pipeline.rs`](examples/pipeline/dag_pipeline.rs) |
| 10 | **Robustness Test** — FGSM/PGD attacks and defenses | Adversarial | [`examples/adversarial/robustness_test.rs`](examples/adversarial/robustness_test.rs) |
| 11 | **FFT Analysis** — Frequency decomposition of signals | Signal | [`examples/signal/fft_analysis.rs`](examples/signal/fft_analysis.rs) |
| 12 | **Auctions** — First-price, second-price, English, Dutch | Game Theory | [`examples/game-theory/auctions.rs`](examples/game-theory/auctions.rs) |
| 13 | **Bandits** — Thompson sampling for A/B testing | RL | [`examples/reinforcement-learning/bandits.rs`](examples/reinforcement-learning/bandits.rs) |
| 14 | **Bloom Filter** — Probabilistic membership + HyperLogLog | Probabilistic | [`examples/probabilistic/bloom_filter.rs`](examples/probabilistic/bloom_filter.rs) |
| 15 | **GPU Similarity** — WGPU cosine similarity search | GPU | [`examples/gpu/similarity_search.rs`](examples/gpu/similarity_search.rs) |
| 16 | **Embedded Cache** — TTL, LRU, pub/sub, Redis-style ops | Cache | [`examples/cache/embedded_cache.rs`](examples/cache/embedded_cache.rs) |

Run any example:
```bash
cargo run --example pso_rosenbrock
cargo run --example viterbi_hmm
cargo run --example bloom_filter
# etc.
```

## Architecture

```
MachinDeOuf/
├── Cargo.toml                 # Workspace root
├── CLAUDE.md                  # Project conventions for Claude Code
├── .mcp.json                  # MCP server configuration
├── .claude/
│   ├── skills/                # 10 Claude Code skills
│   ├── hooks/                 # Pipeline validation, cache lifecycle
│   └── agents/                # Compound engineering agents
└── crates/
    ├── machin-math/           # Core math primitives
    ├── machin-optimize/       # Optimization algorithms
    ├── machin-supervised/     # Regression, classification
    ├── machin-unsupervised/   # Clustering, dimensionality reduction
    ├── machin-ensemble/       # Random forest
    ├── machin-nn/             # Neural networks
    ├── machin-rl/             # Reinforcement learning
    ├── machin-evolution/      # Evolutionary algorithms
    ├── machin-graph/          # Graphs, Markov, HMM
    ├── machin-search/         # Search algorithms
    ├── machin-game/           # Game theory
    ├── machin-chaos/          # Chaos theory
    ├── machin-signal/         # Signal processing
    ├── machin-adversarial/    # Adversarial ML
    ├── machin-probabilistic/  # Probabilistic data structures
    ├── machin-gpu/            # GPU compute (WGPU)
    ├── machin-cache/          # Embedded cache
    ├── machin-pipeline/       # DAG executor
    ├── machin-io/             # Data I/O
    ├── machin-dynamics/       # IK, Lie groups, neural ODEs
    ├── machin-topo/           # Persistent homology
    ├── machin-ktheory/        # Graph K-theory
    ├── machin-category/       # Category theory
    ├── machin-grammar/        # Formal grammars
    ├── machin-agent/          # MCP server
    ├── machin-skill/          # CLI binary
    └── machin-demo/           # egui demo app
```

## Key Dependencies

- **ndarray** 0.17 — Matrix operations (`f64`)
- **rand** 0.9 + **rand_distr** 0.5 — Random number generation
- **wgpu** 28 — Cross-platform GPU compute
- **parking_lot** 0.12 — Fast concurrent locks
- **tokio** 1 — Async runtime (I/O, cache server)
- **thiserror** 2 — Error types
- **clap** 4 — CLI parsing

## Testing

```bash
# Unit + integration tests
cargo test --workspace

# Clippy (zero warnings policy)
cargo clippy --workspace -- -D warnings

# Property-based tests (proptest)
cargo test --workspace -- --include-ignored

# Benchmarks (criterion)
cargo bench -p machin-math
```

- **proptest** for math invariants (commutativity, associativity, norm preservation)
- **criterion** for performance-critical paths (FFT, matrix ops, GPU kernels)

## Conventions

- Pure Rust (except WGPU for GPU compute)
- CPU algorithms use `f64`; GPU uses `f32` via WGSL shaders
- Builder pattern for algorithm configuration
- Seeded RNG for reproducibility
- Each crate defines domain traits (`Regressor`, `Classifier`, `Clusterer`, `Optimizer`, etc.)
- MSRV: Rust 1.80+ (due to wgpu 28)

## License

MIT
