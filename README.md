# ix

[![CI](https://github.com/GuitarAlchemist/ix/actions/workflows/ci.yml/badge.svg)](https://github.com/GuitarAlchemist/ix/actions/workflows/ci.yml)

> **Read this first:** [`docs/MANUAL.md`](docs/MANUAL.md) — the canonical user manual. Install, tool inventory, pipeline format, governance integration, extension guide, troubleshooting. Start there.

A Rust workspace of composable ML/math algorithms and AI governance, designed to be exposed as **Claude Code skills** via an MCP server and CLI. Part of the [GuitarAlchemist](https://github.com/GuitarAlchemist) ecosystem (ix + [tars](https://github.com/GuitarAlchemist/tars) + [ga](https://github.com/GuitarAlchemist/ga) + [Demerzel](https://github.com/GuitarAlchemist/Demerzel)).

77 crates. 93 MCP tools. 80+ Claude Code skills. Pure Rust. No external ML frameworks.

## Quick Start

```bash
# Build everything
cargo build --workspace

# Run tests
cargo test --workspace

# Run the CLI
cargo run -p ix-skill -- optimize --algo pso --function sphere --dim 10

# Start the MCP server (for Claude Code integration)
cargo run -p ix-agent
```

## Crate Maturity & Stability

### Maturity Tiers

| Tier | Meaning | Downstream-Safe? |
|------|---------|:---:|
| **Stable** | API commitments, well-tested, used in production MCP tools | Yes |
| **Beta** | Feature-complete, actively used, API may change between minor versions | With caution |
| **Experimental** | Research-grade, novel math, no stability guarantees | No |
| **Internal** | Tooling/infra, not intended for direct external use | No |

### Stable Subset (safe for ga, tars, Demerzel consumption via MCP)

| Crate | Tier | Notes |
|-------|------|-------|
| ix-math | Beta | Core linear algebra, statistics, activations — everything depends on this. Demoted from Stable 2026-06-21 while the new `inference` surface (two-sample tests, divergences, moments) settles |
| ix-optimize | Stable | SGD, Adam, PSO, simulated annealing |
| ix-supervised | Stable | Regression, trees, KNN, SVM, metrics |
| ix-ensemble | Stable | Random forest, gradient boosted trees |
| ix-unsupervised | Stable | KMeans, DBSCAN, PCA, t-SNE, GMM |
| ix-search | Stable | A*, MCTS, minimax, BFS/DFS |
| ix-graph | Stable | Markov chains, HMM/Viterbi, agent routing |
| ix-signal | Stable | FFT, wavelets, Kalman, spectral analysis |
| ix-cache | Stable | Embedded in-process cache (TTL, LRU, pub/sub) — promoted 2026-05-02 |
| ix-probabilistic | Stable | Bloom, HLL, Count-Min, Cuckoo — promoted 2026-05-02 |
| ix-game | Stable | Nash, Shapley, auctions, mechanism design — promoted 2026-05-02 |
| ix-rl | Stable | Bandits (EpsilonGreedy, UCB1, Thompson), Q-learning — promoted 2026-05-02 |

### Stability contract

"Stable" is not just documentation — **CI prevents silent breakage**. The
[`stable-surface.yml`](.github/workflows/stable-surface.yml) workflow runs on
every PR: it hashes the public-API surface of each Stable-tier crate on
`main`, then again on the PR branch, and **fails** the check if any hash
changed. Non-stable crates produce a warning only.

Source of truth: [`crate-maturity.toml`](crate-maturity.toml) at the workspace
root. The CLI subcommand `ix stable-surface` is the reference implementation
used by CI and is runnable locally:

```bash
cargo run -p ix-skill -- stable-surface --format=json
```

The hash is a BLAKE3 over the sorted, deduplicated set of `pub` declarations
discovered in each crate's `src/**.rs`. It catches **added / removed /
renamed exported symbols** — enough to stop accidental breaks. It does NOT
catch finer-grained changes (function signature edits, trait-bound
additions, variance shifts). Those are out of scope for v1; revisit with
`cargo public-api` integration when a real break slips through.

**To intentionally break a Stable API**, do one of the following in the
SAME PR that introduces the change:

1. **Demote the crate** to `beta` in `crate-maturity.toml`. The guard then
   downgrades the diff to a warning. Document the demotion in the PR body.
2. **Bump the crate's major version** in its `Cargo.toml`. (v1 of this
   guard still flags the hash change — bumping the major version is the
   semver-correct signal to downstream consumers, but the guard does not
   yet auto-detect it. Use the demotion path for the CI to go green, then
   restore the tier in a follow-up release PR.)

The current 12 Stable crates (see table above) cover the safe-to-consume
subset for `ga`, `tars`, `Demerzel`, and `agent-blackbox`.

### Beta Crates

| Crate | Tier | Notes |
|-------|------|-------|
| ix-nn | Beta | Transformers, backprop — complex but actively developed |
| ix-pipeline | Beta | DAG executor — critical infrastructure, API stabilizing |
| ix-agent | Beta | MCP server (93 tools) — production-facing integration point |
| ix-governance | Beta | Demerzel governance bridge — consumed by ga/tars |
| ix-io | Beta | I/O utilities (CSV, JSON, TCP, WebSocket) |
| ix-grammar | Beta | Earley, CYK parsers, EBNF/ABNF parsers, ~30-entry grammar catalog |
| ix-catalog-core | Beta | Shared `Catalog` trait substrate used by ix-code, ix-grammar, ix-net catalogs |
| ix-net | Beta | Curated catalog of ~70 IETF RFCs with obsolescence-graph queries |

### Experimental Crates

| Crate | Tier | Notes |
|-------|------|-------|
| ix-evolution | Experimental | Genetic algorithms, differential evolution |
| ix-chaos | Experimental | Lyapunov, bifurcation, strange attractors |
| ix-adversarial | Experimental | FGSM, PGD, differential privacy |
| ix-dynamics | Experimental | Lie groups/algebras, neural ODEs |
| ix-topo | Experimental | Persistent homology, simplicial complexes |
| ix-ktheory | Experimental | Graph K-theory, Grothendieck K0/K1 |
| ix-category | Experimental | Functors, monads, category theory |
| ix-rotation | Experimental | Quaternions, SLERP, Plücker coordinates |
| ix-sedenion | Experimental | Hypercomplex algebra, Cayley-Dickson |
| ix-fractal | Experimental | Takagi curves, IFS, L-systems |
| ix-number-theory | Experimental | Prime sieving, elliptic curves |
| ix-gpu | Experimental | WGPU compute shaders (Vulkan/DX12/Metal) |
| memristive-markov | Experimental | Research prototype |
| ix-acoustic-tune | Experimental | Acoustic auto-tuning: CMA-ES / ask-tell optimizers + spectral features and losses for synth sound-matching |

### Internal Crates

| Crate | Tier | Notes |
|-------|------|-------|
| ix-skill | Internal | CLI binary for direct command-line access |
| ix-skill-macros | Internal | Proc macros for skill registration |
| ix-demo | Internal | egui desktop app with interactive tabs |
| ix-dashboard | Internal | Dashboard utilities |
| ix-registry | Internal | Crate registry metadata |
| ix-code | Internal | Code analysis (cyclomatic, Halstead, catalog of external tools) |
| ix-types | Internal | Shared type definitions |

### Infrastructure & Harness Crates

The 16 crates below were added during the agent-harness and pipeline work. They are internal-tier: consumed by `ix-agent` and the session runtime, not intended as direct deps for downstream repos.

| Crate | Tier | Notes |
|-------|------|-------|
| ix-agent-core | Internal | Shared agent substrate: SessionEvent, middleware chain, action dispatcher |
| ix-approval | Internal | Approval / blast-radius middleware for agent actions |
| ix-autograd | Internal | R7 reverse-mode autograd tape over ndarray with DifferentiableTool trait |
| ix-context | Internal | Context DAG — AST + call + import + git-trajectory graph for agents |
| ix-fuzzy | Internal | Fuzzy-enum evaluation (hexavalent confidence distributions) |
| ix-harness-cargo | Internal | Harness adapter: cargo build / test / clippy / audit |
| ix-harness-clippy | Internal | Harness adapter: clippy warning parser + remediation |
| ix-harness-ga | Internal | Harness adapter: ga music-theory MCP bridge |
| ix-harness-github-actions | Internal | Harness adapter: GitHub Actions workflow inspector |
| ix-harness-signing | Internal | Harness adapter: cosign / SLSA signing integration |
| ix-harness-tars | Internal | Harness adapter: tars F# cognition MCP bridge |
| ix-loop-detect | Internal | Session-level loop detection middleware |
| ix-memory | Internal | Agent memory system (belief persistence, context window management) |
| ix-registry-check | Internal | R3 Buf-style breaking-change detector for capability-registry.json |
| ix-sentinel | Internal | Governed reactive reasoner — autonomous session-level monitor |
| ix-session | Internal | Session log, event stream, trace flywheel export |

### Music, voicings & OPTIC-K crates

Music set-theory and the GA voicing-index (OPTIC-K) pipeline. Internal-tier — consumed by `ga` / the analyst tooling, not direct downstream deps.

| Crate | Tier | Notes |
|-------|------|-------|
| ix-bracelet | Internal | Dihedral group D₁₂ operator algebra on 12-bit pitch-class sets: unified Tₙ/TₙI action, bracelet prime forms, twelve-tone serialism |
| ix-voicings | Internal | Voicings study pipeline: enumerate + featurize chord voicings from the GA FretboardVoicingsCLI |
| ix-optick | Internal | Memory-mapped OPTIC-K index reader with brute-force cosine search |
| ix-optick-invariants | Internal | Corpus-reading invariant checker for the OPTIC-K mmap index |
| ix-optick-sae | Internal | OPTIC-K Sparse Autoencoder orchestrator (Python SAE trainer + artifact validation) |
| ix-embedding-diagnostics | Internal | Baseline diagnostics for OPTIC-K voicing embeddings — leak detection, retrieval consistency, topology |
| ga-chatbot | Internal | Domain-specific voicing chatbot MCP server with a deterministic QA harness |

### Governance-instrumentation & analyst crates

| Crate | Tier | Notes |
|-------|------|-------|
| ix-duck | Internal | In-process DuckDB analyst bench over IX telemetry, with IX algorithms as SQL UDFs (optional `duck` feature) |
| ix-manifold | Internal | Pure-Rust manifold learning: t-SNE for 2D/3D embedding visualization |
| ix-sanitize | Internal | Injection-regex stripper + source-envelope wrapper + Confidential/Unknown verdict gate for governance pipelines |
| ix-ai-annotations | Internal | Extracts in-source `@ai:` semantic annotations (hexavalent T/P/U/D/F/C) and emits the ai-annotation contract stream |
| ix-assumption-graph | Internal | Graph of assumptions from `@ai:` annotations + contradiction detection (temporal assumption graph, Phase 1) |
| ix-sentrux-annotations | Internal | Bridge converting sentrux structural-rule violations into ai-annotation records |
| ix-blast-radius | Internal | Path-based blast-radius analyzer matching the `qa-verdict.blast_radius` contract |
| ix-autoresearch | Internal | Karpathy-style edit-eval-iterate kernel + target adapters for IX subsystems |
| ix-quality-trend | Internal | Aggregates quality snapshots over time into an executive markdown trend report |
| ix-invariant-coverage | Internal | Measures invariant-catalog optimality: rank, redundancy, orphans, coverage gaps |
| ix-quality-validate | Internal | CI validator: every `state/quality/*.json` against the dashboard-envelope schema |
| ix-streeling | Internal | Streeling University — federated learnings catalog + campus index over the ecosystem's docs |
| ix-value | Internal | Business-Value Scorecard — federate per-repo value manifests into a RICE→stars catalog |

(A loadable DuckDB C-API extension, `ix-duck-ext`, lives under `crates/` but is **workspace-excluded** so the default build never compiles DuckDB — build it explicitly via its `build.ps1`.)

### MCP Tool Surface

`ix-agent` exposes **93 MCP tools** covering core math, supervised / unsupervised ML, neural networks + autograd, signal + chaos, graph + topology, adversarial + evolution, game theory, probabilistic data structures, grammar, governance, federation, source adapters (`ix_git_log`, `ix_cargo_deps`, `ix_code_analyze`, `ix_code_catalog`, `ix_ast_query`, `ix_code_smells`), OPTIC-K voicing search, and pipeline orchestration (`ix_pipeline_run`, `ix_pipeline_compile`, `ix_pipeline_list`).

See [`docs/MANUAL.md §4`](docs/MANUAL.md#4-the-64-mcp-tools--by-category) for the full categorized inventory and [`crates/ix-agent/src/tools.rs`](crates/ix-agent/src/tools.rs) for the authoritative schemas.

### Pipelines, compiler, and canonical showcases

- **Pipelines** — Submit a DAG of tool calls in one request via `ix_pipeline_run`. Content-addressed cache hits on replay (R2 Phase 1). Every run emits a lineage DAG consumable by `ix_governance_check` (R2 Phase 2).
- **Natural-language compiler** — `ix_pipeline_compile` turns a sentence into a validated `pipeline.json` DAG via MCP sampling + registry validation.
- **Canonical showcases** at [`examples/canonical-showcase/`](examples/canonical-showcase/):
  - `01-cost-anomaly-hunter` (3 tools, FinOps)
  - `02-chaos-detective` (4 tools, signal + chaos + topology)
  - `03-governance-gauntlet` (5 tools, constitutional audit)
  - `04-sprint-oracle` (4 tools, forecasting)
  - `05-adversarial-refactor-oracle` (14 tools, self-referential live-data audit of ix's own workspace — see its [`FINDINGS.md`](examples/canonical-showcase/05-adversarial-refactor-oracle/FINDINGS.md))

## Crates

### Core Math & Optimization
| Crate | Description |
|-------|-------------|
| **ix-math** | Linear algebra, statistics, distances, activations, calculus, random, hyperbolic geometry |
| **ix-optimize** | SGD, Adam, simulated annealing, particle swarm optimization |

### Supervised Learning
| Crate | Description |
|-------|-------------|
| **ix-supervised** | Linear/logistic regression, decision trees, KNN, Naive Bayes, SVM, metrics (confusion matrix, ROC/AUC), cross-validation, SMOTE, TF-IDF |
| **ix-ensemble** | Random forest, gradient boosted trees |

### Unsupervised Learning
| Crate | Description |
|-------|-------------|
| **ix-unsupervised** | K-Means, DBSCAN, PCA, t-SNE (Barnes-Hut), GMM (EM algorithm) |

### Deep Learning & RL
| Crate | Description |
|-------|-------------|
| **ix-nn** | Neural network layers (Dense, LayerNorm, BatchNorm, Dropout), loss functions, backprop, transformers |
| **ix-rl** | Multi-armed bandits (epsilon-greedy, UCB1, Thompson), Q-learning, GridWorld |
| **ix-evolution** | Genetic algorithms, differential evolution |

### Search & Graphs
| Crate | Description |
|-------|-------------|
| **ix-search** | A\*, Q\* (learned heuristic), MCTS, minimax/alpha-beta, BFS/DFS, hill climbing, tabu search |
| **ix-graph** | Graph algorithms, Markov chains, HMM/Viterbi/Baum-Welch, state spaces, agent routing |
| **ix-game** | Nash equilibria, Shapley value, auctions, evolutionary dynamics, mean field games |

### Signal & Chaos
| Crate | Description |
|-------|-------------|
| **ix-signal** | FFT, wavelets, FIR/IIR filters, Kalman filter, spectral analysis, DCT |
| **ix-chaos** | Lyapunov exponents, bifurcation diagrams, strange attractors, fractal dimensions, delay embedding, chaos control |

### Security & Privacy
| Crate | Description |
|-------|-------------|
| **ix-adversarial** | FGSM, PGD, C&W attacks, adversarial training, poisoning detection, differential privacy, robustness evaluation |

### Advanced Math
| Crate | Description |
|-------|-------------|
| **ix-dynamics** | Inverse kinematics chains, Lie groups/algebras (SO(3), SE(3)), neural ODEs |
| **ix-topo** | Persistent homology, simplicial complexes, Betti numbers, topological data analysis |
| **ix-ktheory** | Graph K-theory, Grothendieck K0/K1, Mayer-Vietoris sequences |
| **ix-category** | Functors, natural transformations, monads, category theory primitives |
| **ix-grammar** | Formal grammars: context-free grammars, Earley parser, CYK, Chomsky normal form |
| **ix-rotation** | Quaternions, SLERP, Euler angles, axis-angle, rotation matrices, Plücker coordinates |
| **ix-sedenion** | Hypercomplex algebra: sedenions, octonions, Cayley-Dickson construction, BSP trees |
| **ix-fractal** | Takagi curves, IFS (Sierpinski, fern), L-systems, Hilbert/Peano/Morton space-filling curves |
| **ix-number-theory** | Prime sieving, Miller-Rabin, modular arithmetic, CRT, elliptic curves |

### Infrastructure
| Crate | Description |
|-------|-------------|
| **ix-gpu** | WGPU compute shaders for cosine similarity, matrix multiply, batch vector search (Vulkan/DX12/Metal) |
| **ix-cache** | Embedded Redis-like cache with sharded concurrency, TTL, LRU eviction, pub/sub, RESP protocol server |
| **ix-pipeline** | DAG executor with topological sort, parallel branch execution, memoization, critical path analysis |
| **ix-probabilistic** | Bloom filter, Count-Min sketch, HyperLogLog, Cuckoo filter |
| **ix-io** | CSV, JSON, file watcher, named pipes, TCP, HTTP, WebSocket, trace bridge |
| **ix-catalog-core** | Shared `Catalog` trait + helpers — substrate for ix-code / ix-grammar / ix-net catalogs, exposed via `ix_catalog_list` meta-tool |
| **ix-net** | Curated IETF RFC catalog (~70 entries) with obsolescence graph, queryable by number / topic / status via `ix_rfc_catalog` |

### Governance
| Crate | Description |
|-------|-------------|
| **ix-governance** | Demerzel governance: tetravalent logic (T/F/U/C), constitution parser, 12 persona loader, policy engine |

### Integration
| Crate | Description |
|-------|-------------|
| **ix-agent** | MCP server: 93 tools via JSON-RPC over stdio (algorithms + governance + federation + pipeline orchestration) |
| **ix-skill** | CLI binary for direct command-line access to all algorithms |
| **ix-demo** | egui desktop app with 22+ interactive demo tabs including governance explorer |

## Claude Code Integration

### MCP Server

Register ix as a Claude Code MCP server in `.mcp.json`:

```json
{
  "mcpServers": {
    "ix": {
      "command": "cargo",
      "args": ["run", "-p", "ix-agent"]
    }
  }
}
```

Claude can then call tools like `ix_kmeans`, `ix_viterbi`, `ix_optimize`, etc. directly during conversations.

### Skills

80+ Claude Code skills organized by domain:

**Algorithm skills** (26): ix-optimize, ix-cluster, ix-search, ix-chaos, ix-hmm, ix-adversarial, ix-game, ix-pipeline, ix-signal, ix-benchmark, ix-nn, ix-bandit, ix-evolution, ix-random-forest, ix-supervised, ix-topo, ix-category, ix-dynamics, ix-ktheory, ix-gpu, ix-cache, ix-grammar, ix-rotation, ix-sedenion, ix-fractal, ix-number-theory

**Governance skills** (3): ix-governance-check, ix-governance-persona, ix-governance-belief

**Federation skills** (4): federation-discover, federation-grammar, federation-music, federation-traces

**Ecosystem skills** (4): governed-execute, ecosystem-audit, roadblock-resolver, delegate-cli

### MCP Federation

ix connects to tars (F# reasoning) and ga (music theory) via MCP:

```json
{
  "mcpServers": {
    "ix": { "command": "cargo", "args": ["run", "--release", "-p", "ix-agent"] },
    "tars": { "command": "dotnet", "args": ["run", "--project", "path/to/Tars.Interface.Cli", "--", "mcp", "server"] },
    "ga": { "command": "dotnet", "args": ["run", "--project", "path/to/GaMcpServer"] }
  }
}
```

### Demerzel Governance

Agents operate under the [Demerzel](https://github.com/GuitarAlchemist/Demerzel) constitution (11 articles, 12 personas, tetravalent logic). Named after [R. Daneel Olivaw](https://asimov.fandom.com/wiki/R._Daneel_Olivaw) — consistent with Asimov's Zeroth Law.

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

## Documentation

**[Learning Path — 60+ tutorials](docs/INDEX.md)** from foundations to advanced topics, with runnable Rust examples.

**[🇫🇷 Parcours d'apprentissage en français](docs/fr/INDEX.md)** — 11 tutoriels d'algorithmes + 4 cas pratiques traduits en français.

Topics: linear algebra, optimization, supervised/unsupervised learning, neural networks, reinforcement learning, game theory, signal processing, chaos theory, adversarial ML, GPU computing, and more.

## DuckDB Integration (analyst bench)

`ix-duck` is an **in-process DuckDB "analyst bench"** — *not* a production engine and *not* a source of truth (see [`docs/DUCKDB.md`](docs/DUCKDB.md)). It does two things:

1. **IX algorithms as SQL UDFs.** Call IX primitives directly from SQL — `ix_cosine` / `ix_euclidean`, `ix_pca_project` / `ix_kmeans` / `ix_dbscan` / `ix_silhouette`, ranking metrics (`ix_ndcg`, …), probabilistic sketches (`ix_bloom_*`, `ix_hll_*`, `ix_cms_*`, `ix_cuckoo_*`), music set-theory (`ix_forte_number`, `ix_icv`, `ix_prime_form`, the twelve-tone serialism + Grothendieck UDFs), graph/signal (`ix_pagerank`, `ix_rfft`), and code metrics (`ix_code_complexity`, `ix_ast_query`).
2. **Lenses over ecosystem telemetry.** Read-only analyzers over the sibling repos' `state/quality/**` JSON-on-disk artifacts (cross-repo by **format contract, not runtime coupling**), each graceful-degrading when a directory is absent:

| Lens | Reads | Surfaces |
|---|---|---|
| `chatbot` | GA golden-traces | flight recorder + **canonical-diff regression gate** (routed-agent drift) |
| `routing` | GA `routing-eval-*.json` | per-intent F1 trend, weakest intents, run-over-run regressions |
| `ood` | GA query embeddings | out-of-domain queries (mean top-k cosine) |
| `loops` | AFK loop iterations | failure clustering — recurring worst-items, oscillating loops |
| `maintain` | all of the above + `hits.jsonl` | the **maintain-gate**: one hexavalent (T/P/U/D/F/C) RSI verdict (metric↑ ∧ guardrail held ∧ converging ∧ in-distribution, never an average) |

The shared **`telemetry` ingestion seam** concentrates the *absent → skip / unreadable → fail-closed* contract every lens depends on. DuckDB code lives behind the optional **`duck`** feature (a bundled C++ build); the default `cargo build --workspace` and CI never compile it. Exposed to agents through the feature-gated `ix_maintain_gate` MCP tool. IXQL (Demerzel's governance DSL) and `ix-duck` are complementary layers joined only by the file-format contract — see [`docs/adr/0001-ixql-duckdb-integration-via-mcp-seam.md`](docs/adr/0001-ixql-duckdb-integration-via-mcp-seam.md).

```bash
# the analyst bench, with every IX UDF registered:
cargo run -p ix-duck --features duck --example ix_chatbot_lens -- check ../ga/state/quality/chatbot-qa
```

## Architecture

ix is a Rust workspace of **77 crates** organised into six rough layers, plus a governance submodule. The top-level shape:

```
ix/
├── Cargo.toml             # Workspace root
├── CLAUDE.md              # Project conventions (load-bearing for Claude Code sessions)
├── README.md              # This file
├── docs/
│   ├── MANUAL.md          # ← canonical user manual (start here)
│   ├── INDEX.md           # 60+ tutorial learning path
│   ├── FEDERATION.md      # Cross-repo (ix + tars + ga) federation
│   └── guides/            # graph-theory-in-ix, code-analysis-tools, ...
├── examples/
│   └── canonical-showcase/# 5 reproducible demo pipelines + roadmap + findings
├── governance/
│   └── demerzel/          # Git submodule: constitution + personas + policies
└── crates/                # 77 crates — see maturity tables above
```

For the per-crate inventory grouped by concern, see [`docs/MANUAL.md §4`](docs/MANUAL.md#4-the-64-mcp-tools--by-category). The source of truth for crate dependencies is each crate's `Cargo.toml`; for a live workspace dep graph, run the `ix_cargo_deps` MCP tool against this repo.

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
cargo bench -p ix-math
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
