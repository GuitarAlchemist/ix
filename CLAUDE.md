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
- `crates/ix-signal` - Signal processing: FFT, wavelets, filters, Kalman, spectral analysis
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
