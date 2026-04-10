# ix - ML Algorithms + Governance for Claude Code Skills

## Project Overview
Rust workspace (32 crates) implementing foundational ML/math algorithms and AI governance as composable crates, exposed as Claude Code skills via MCP server (`ix-agent`) and CLI (`ix-skill`). Part of the GuitarAlchemist ecosystem (ix + tars + ga + Demerzel).

## Architecture

### Core Math & Optimization
- `crates/ix-math` - Core math: linalg, stats, distance, activation, calculus, random, hyperbolic (Poincaré)
- `crates/ix-optimize` - Optimization: SGD, Adam, simulated annealing, PSO

### Supervised & Unsupervised Learning
- `crates/ix-supervised` - Regression, classification, metrics (confusion matrix, ROC/AUC, log loss), cross-validation (KFold, StratifiedKFold), resampling (SMOTE, random undersampling), text vectorization (TF-IDF, CountVectorizer)
- `crates/ix-unsupervised` - Clustering (K-Means, DBSCAN), PCA, t-SNE, GMM
- `crates/ix-ensemble` - Random forest, gradient boosting (GradientBoostedClassifier)

### Deep Learning, RL & Evolution
- `crates/ix-nn` - Trainable transformers (full backprop, GPU attention), dense layers, Sequential, loss functions, normalization (LayerNorm, RMSNorm, BatchNorm)
- `crates/ix-rl` - Bandits (epsilon-greedy, UCB1, Thompson), Q-learning, GridWorld
- `crates/ix-evolution` - Genetic algorithms, differential evolution

### Search, Graphs & Game Theory
- `crates/ix-graph` - Graph algorithms, Markov chains, HMM/Viterbi, state spaces, agent routing
- `crates/ix-search` - Search: A*, Q*, MCTS, minimax, alpha-beta, BFS/DFS, data structure search
- `crates/ix-game` - Game theory: Nash equilibria, Shapley value, auctions, evolutionary, mean field

### Signal, Chaos & Adversarial
- `crates/ix-signal` - Signal processing: FFT, wavelets, filters, Kalman, spectral analysis, time series (rolling stats, lag features, temporal split, EWMA)
- `crates/ix-chaos` - Chaos theory: Lyapunov exponents, bifurcation, attractors, fractals, embedding
- `crates/ix-adversarial` - Adversarial ML: evasion (FGSM, PGD, C&W), defense, poisoning detection, privacy

### Advanced Math
- `crates/ix-dynamics` - Inverse kinematics, Lie groups/algebras, neural ODEs
- `crates/ix-topo` - Persistent homology, simplicial complexes, Betti numbers
- `crates/ix-ktheory` - Graph K-theory, Mayer-Vietoris sequences
- `crates/ix-category` - Functors, natural transformations, monads
- `crates/ix-grammar` - Formal grammars: CFG, Earley parser, CYK, Chomsky normal form
- `crates/ix-rotation` - Quaternions, SLERP, Euler angles, rotation matrices
- `crates/ix-sedenion` - Hypercomplex algebra: sedenions, octonions, Cayley-Dickson
- `crates/ix-fractal` - Takagi curves, IFS, L-systems, space-filling curves
- `crates/ix-number-theory` - Prime sieving, primality tests, modular arithmetic, elliptic curves

### Probabilistic & Infrastructure
- `crates/ix-probabilistic` - Bloom filters, Count-Min sketch, HyperLogLog, Cuckoo filter
- `crates/ix-io` - Data I/O: CSV, JSON, file watcher, named pipes, TCP, HTTP, WebSocket, trace bridge
- `crates/ix-gpu` - GPU compute via WGPU: cosine similarity, matrix multiply, batch vector search, BSP kNN
- `crates/ix-cache` - Embedded Redis-like cache: concurrent sharded store, TTL, LRU, pub/sub, RESP server
- `crates/ix-pipeline` - DAG pipeline executor: skill orchestration, parallel branches, memoized data flow

### Governance
- `crates/ix-governance` - Demerzel governance: tetravalent logic (T/F/U/C), constitution parser, persona loader, policy engine
- `governance/demerzel` - Git submodule: 12 personas, 11-article constitution, 3 policies, behavioral tests

### Integration
- `crates/ix-agent` - MCP server: 37 tools via JSON-RPC over stdio (algorithms + governance + federation)
- `crates/ix-skill` - CLI binary for direct command-line access
- `crates/ix-demo` - egui desktop app with 22+ interactive demo tabs

## MCP Federation
Three MCP servers registered in `.mcp.json`:
- **ix** (Rust): 37 tools — math, ML, governance, federation
- **tars** (F#): grammar weighting, pattern promotion, metacognition
- **ga** (C#): music theory, chord analysis, trace export

Capability registry at `governance/demerzel/schemas/capability-registry.json`.

## Build
```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo doc --workspace --no-deps
```

## Testing
- **Unit tests**: `#[test]` for every public function
- **Property tests**: `proptest` for math invariants
- **Benchmarks**: `criterion` for performance-critical paths
- **Doc tests**: All `///` examples must compile and run
- **Behavioral tests**: Demerzel thought experiments in `governance/demerzel/tests/`
- **CI**: GitHub Actions on stable + nightly Rust, Linux + Windows

## Key Dependencies
- `ndarray` 0.17 - Matrix operations
- `rand` 0.9 + `rand_distr` 0.5 - Random number generation
- `thiserror` 2 - Error types
- `clap` 4 - CLI parsing
- `wgpu` 28 - Cross-platform GPU compute (Vulkan/DX12/Metal)
- `tokio` 1 - Async runtime (I/O crate)
- `serde_yaml` 0.9 - Governance artifact parsing
- `proptest` 1 - Property-based testing
- `criterion` 0.5 - Benchmarking

## MSRV
Rust 1.80+ (due to wgpu 28)

## Conventions
- Pure Rust, no external ML frameworks (except wgpu for GPU compute)
- CPU algorithms use `f64` and `ndarray::Array{1,2}<f64>`; GPU uses `f32` via WGPU shaders
- Each crate defines traits (Regressor, Classifier, Clusterer, Optimizer, etc.)
- Builder pattern for algorithm configuration
- Seeded RNG for reproducibility
- Governance: all agent actions subject to Demerzel constitution (11 articles)
- Federation: cross-repo tool calls via MCP protocol

## Graph Theory Coverage

IX ships a deep, cross-crate graph theory stack. **Before adding a new graph
algorithm, new graph crate, or pulling in `petgraph`/`daggy`/`graph-rs`:**
check this list first. The primitives you need are almost certainly already
here.

Full landing page: [`docs/guides/graph-theory-in-ix.md`](docs/guides/graph-theory-in-ix.md).

| Module | What it provides | Use when |
|---|---|---|
| `crates/ix-graph` | Generic graphs, Markov chains, HMM/Viterbi, state spaces, agent routing | You need generic graph data structures, probabilistic transition models, or state-space search |
| `crates/ix-pipeline::dag::Dag<N>` | Generic cycle-checked DAG with topological sort | You need an acyclic DAG substrate — cycle detection is enforced on every edge insertion |
| `crates/ix-search` | A*, Q*, MCTS, minimax, alpha-beta, BFS/DFS, adversarial search | You need shortest-path, optimal play, tree search, or goal-directed traversal |
| `crates/ix-topo` | Persistent homology, simplicial complexes, Betti numbers | You need topological invariants of a graph (connected components, cycles, cavities) |
| `crates/ix-ktheory` | Algebraic K-theory over graphs, Mayer-Vietoris sequences, Grothendieck K₀/K₁ | You need algebraic invariants for classification/equivalence of graph structures |
| `crates/ix-code::semantic` | Tree-sitter call graph extraction + rich `CalleeHint` variants | You need static call graphs from Rust source |
| `crates/ix-code::topology` | Persistent homology applied to call graphs | You need structural summaries of codebases (code shape, cycles, islands) |
| `crates/ix-code::advanced` | Hyperbolic embeddings, BSP spatial search, spectral methods over graphs | You need embeddings of large graphs or spectral clustering |
| `crates/ix-code::physics` | Laplacian spectrum of call graphs (via `ix-math::eigen`) | You need spectral graph theory — eigenvalues, community detection, connectivity |
| `crates/ix-context` *(WIP)* | Walker over AST + call + import + git-trajectory DAG | You need agent-facing context retrieval by graph walks |

**Rules:**
1. **Never add `petgraph`, `daggy`, `graph-rs`, or similar as a new dependency** without explicitly documenting why IX's existing primitives are insufficient. Default answer: they aren't.
2. **For cycle-checked DAGs, use `ix-pipeline::dag::Dag<N>`.** It is generic over node data, enforces acyclicity on every edge insertion, and has topological sort built in.
3. **For shortest paths, use `ix-search::astar` or `ix-search::uninformed`** (BFS/DFS). Do not hand-roll a BFS.
4. **For topological analysis of graph structure, use `ix-topo`.** Persistent homology is already shipped.
5. **For any new graph-shaped primitive, consult [`docs/guides/graph-theory-in-ix.md`](docs/guides/graph-theory-in-ix.md) first** to confirm it doesn't already live somewhere in the stack.
6. **`ix-code::semantic::CallGraph` is per-file.** Cross-file/project-wide call graphs are the job of `ix-context` (in progress). Do not try to bolt cross-file resolution into `ix-code`.

<!-- BEGIN DEMERZEL GOVERNANCE -->
# Demerzel Governance Integration

This repo participates in the Demerzel governance framework.

## Governance Framework

All agents in this repo are governed by the Demerzel constitutional hierarchy:

- **Root constitution:** governance/demerzel/constitutions/asimov.constitution.md (Articles 0-5: Laws of Robotics + LawZero principles)
- **Governance coordinator:** Demerzel (see governance/demerzel/constitutions/demerzel-mandate.md)
- **Operational ethics:** governance/demerzel/constitutions/default.constitution.md (Articles 1-7)
- **Harm taxonomy:** governance/demerzel/constitutions/harm-taxonomy.md

## Policy Compliance

Agents must comply with all Demerzel policies:

- **Alignment:** Verify actions serve user intent (confidence thresholds: 0.9 autonomous, 0.7 with note, 0.5 confirm, 0.3 escalate)
- **Rollback:** Revert failed changes automatically; pause autonomous changes after automatic rollback
- **Self-modification:** Never modify constitutional articles, disable audit logging, or remove safety checks
- **Kaizen:** Follow PDCA cycle for improvements; classify as reactive/proactive/innovative before acting
- **Reconnaissance:** Respond to Demerzel reconnaissance requests with belief snapshots and compliance reports
- **Scientific objectivity:** Tag evidence as empirical/inferential/subjective; generator/estimator accountability
- **Streeling:** Accept knowledge transfers from Seldon; report comprehension via belief state assessment

## Galactic Protocol

This repo communicates with Demerzel via the Galactic Protocol:

- **Inbound (from Demerzel):** Governance directives, knowledge packages
- **Outbound (to Demerzel):** Compliance reports, belief snapshots, learning outcomes
- **Message formats:** See governance/demerzel/schemas/contracts/

## Belief State Persistence

This repo maintains a `state/` directory for belief persistence:

- `state/beliefs/` — Tetravalent belief states (*.belief.json)
- `state/pdca/` — PDCA cycle tracking (*.pdca.json)
- `state/knowledge/` — Knowledge transfer records (*.knowledge.json)
- `state/snapshots/` — Belief snapshots for reconnaissance (*.snapshot.json)

File naming: `{date}-{short-description}.{type}.json`

## Agent Requirements

Every persona in this repo must include:

- `affordances` — Explicit list of permitted actions
- `goal_directedness` — One of: none, task-scoped, session-scoped
- `estimator_pairing` — Neutral evaluator persona (typically skeptical-auditor)
- All fields required by governance/demerzel/schemas/persona.schema.json
<!-- END DEMERZEL GOVERNANCE -->
