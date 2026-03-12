# MachinDeOuf

A Rust workspace of composable ML/math algorithms, designed to be exposed as **Claude Code skills** via an MCP server and CLI.

22 crates. Pure Rust. No external ML frameworks.

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
    ├── machin-agent/          # MCP server
    └── machin-skill/          # CLI binary
```

## Key Dependencies

- **ndarray** 0.17 — Matrix operations (`f64`)
- **rand** 0.9 + **rand_distr** 0.5 — Random number generation
- **wgpu** 28 — Cross-platform GPU compute
- **parking_lot** 0.12 — Fast concurrent locks
- **tokio** 1 — Async runtime (I/O, cache server)
- **thiserror** 2 — Error types
- **clap** 4 — CLI parsing

## Conventions

- Pure Rust (except WGPU for GPU compute)
- CPU algorithms use `f64`; GPU uses `f32` via WGSL shaders
- Builder pattern for algorithm configuration
- Seeded RNG for reproducibility
- Each crate defines domain traits (`Regressor`, `Classifier`, `Clusterer`, `Optimizer`, etc.)

## License

MIT
