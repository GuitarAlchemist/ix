---
title: "feat: Code Observatory - 5-Layer Analysis System"
type: feat
status: active
date: 2026-04-09
deepened: 2026-04-09
origin: docs/brainstorms/2026-04-09-ix-code-observatory-brainstorm.md
---

# feat: Code Observatory - 5-Layer Analysis System

## Enhancement Summary

**Deepened on:** 2026-04-09
**Research agents used:** 8 (tree-sitter best practices, git2 best practices, architecture strategist, performance oracle, security sentinel, governance auditor, pattern recognition, agent-native reviewer)

### Key Improvements
1. Concrete tree-sitter S-expression queries for Rust/Python/TS call-graph extraction
2. git2 file-history walk pattern with pathspec optimization and thread-safety model
3. Security hardening: file size limits, path canonicalization, tree-sitter cancellation flags
4. Performance: rayon parallelism, string interning, Rips dimension cap at H1
5. Governance: thresholds in YAML (not hardcoded), C-resolution procedure, provenance with input hash
6. Two new MCP tools discovered: `code.diff_scope` (PR scoping) and `code.explain` (verdict rationale)
7. Feature gates on ALL layers (not just semantic/trajectory) following memristive-markov pattern

### New Considerations Discovered
- Layer 3 (trajectory) should work with Layer 1 alone (no semantic dependency) for maximum flexibility
- Rips complex must be capped at dimension 1 (H0+H1 only) to avoid O(n^3) triangle explosion
- Symmetrize directed call-graph distance matrix before feeding to Rips
- tree-sitter `Parser` is `!Send` -- use thread-local instances for rayon parallelism
- git2 `Repository` is `Send` but not `Sync` -- one repo handle per thread
- Use `dep:` syntax for optional dependencies (memristive-markov pattern)

## Overview

Evolve ix-code from a 20-feature keyword-based code analyzer into a **5-layer Code Observatory** -- ML-powered, temporally aware, topologically informed, with governance-grade uncertainty quantification via hexavalent verdicts.

Multi-AI brainstorm (Codex GPT-5.4 + Gemini 2.5 + Claude Opus 4.6) converged on: keep the lightweight engine as universal fast path, add tree-sitter for selective depth, invest in temporal analysis and topological structure as the unique differentiators. (see brainstorm: docs/brainstorms/2026-04-09-ix-code-observatory-brainstorm.md)

## Problem Statement / Motivation

ix-code currently extracts 20 static features via keyword counting. This is fast and works across 11 languages, but:
- **No structural understanding** -- can't extract call graphs, can't see function dependencies
- **No temporal awareness** -- a function at complexity 15 and falling is healthier than one at 8 and rising
- **No topological insight** -- 10 simple functions in a cycle are more dangerous than 2 complex functions in a chain
- **No governance integration** -- metrics exist but don't feed into hexavalent policy verdicts
- **Java/C#/C++ function extraction disabled** -- keyword-based approach can't detect functions without clear `fn` keywords

The Code Observatory addresses all five by layering semantic, temporal, topological, and governance capabilities on top of the existing lightweight engine.

## Architecture

```
Layer 1: ix-code (DONE)           -> fast 20-feature vectors, 11 languages
Layer 2: ix-code (semantic mod)   -> tree-sitter AST + call-graph for Rust/Python/TS
Layer 3: ix-code (trajectory mod) -> git-history x metrics -> EWMA, velocity, acceleration
Layer 4: ix-code (topology mod)   -> call-graph -> ix-topo persistent homology -> Betti numbers
Layer 5: ix-code (gates mod)      -> Risk delta scorecards -> hexavalent verdicts (T/P/U/D/F/C)
Layer 6: ix-code (advanced mod)   -> Poincare embeddings, K-theory, spectral, BSP, fractals
Layer 7: ix-code (physics mod)    -> Chaos, Kalman, wavelets, Markov, Lie, Laplacian, Ricci, sheaves
```

All layers live in the **existing `ix-code` crate** as feature-gated modules. No new crates -- keeps the dependency graph flat and avoids workspace bloat.

### Cargo Feature Gates

```toml
[features]
default = []
semantic = ["dep:tree-sitter", "dep:tree-sitter-rust", "dep:tree-sitter-python", "dep:tree-sitter-typescript"]
trajectory = ["dep:git2"]
topology = ["dep:ix-topo", "dep:ix-graph"]
gates = ["dep:ix-governance"]
full = ["semantic", "trajectory", "topology", "gates"]
```

> **Research insight (architecture review):** Feature-gate ALL layers, not just semantic/trajectory. ix-agent should not compile persistent homology just to run basic code metrics. Follow the `memristive-markov` pattern with `dep:` syntax for optional deps and a `full` composite feature.

> **Research insight (pattern review):** Only 2 crates in the workspace use feature gates (memristive-markov, ix-cache). Follow memristive-markov's `dep:` syntax pattern exactly.

## Proposed Solution

### Layer 2: Semantic Analysis (`crates/ix-code/src/semantic.rs`)

**Purpose:** AST-based feature extraction + call-graph construction for Rust, Python, TypeScript.

**tree-sitter integration:**
- Use `tree-sitter` crate (Rust bindings) + language grammars as optional deps behind `semantic` feature
- Parse source into concrete syntax tree
- Walk tree with S-expression queries to extract:
  - Function boundaries (fixes Java/C#/C++ gap)
  - Call-graph edges (function A calls function B)
  - Nesting depth (precise, not estimated)
  - Error-handling density (`Result`/`try`/`except`/`catch` nodes)
  - Unsafe blocks (Rust), type annotation coverage
  - AST node-type distribution (as additional feature vector)

**Call-graph extraction:**
```rust
pub struct CallEdge {
    pub caller: String,      // qualified function name
    pub callee: String,      // qualified function name
    pub call_site_line: usize,
    pub weight: f64,         // 1.0 default, can be annotated
}

pub struct CallGraph {
    pub edges: Vec<CallEdge>,
    pub nodes: Vec<String>,  // all function names
}

pub fn extract_call_graph(source: &str, lang: Language) -> Option<CallGraph>;
```

**Fallback strategy:** When tree-sitter parse fails (syntax errors, generated code, unsupported language), fall back to Layer 1 silently. The returned `SemanticMetrics` includes a `parse_quality: f64` (0.0 = keyword-only fallback, 1.0 = full AST). Downstream layers use this as a confidence multiplier.

**Languages:**
- Phase A: Rust (highest-quality grammar, dogfood target)
- Phase B: Python + TypeScript (broad coverage)
- Future: Java/C#/Go via additional grammar crates

**New struct:**
```rust
pub struct SemanticMetrics {
    pub parse_quality: f64,         // 0.0 = fallback, 1.0 = full AST
    pub ast_node_count: usize,
    pub nesting_depth_max: usize,   // precise from AST
    pub nesting_depth_mean: f64,
    pub error_handling_density: f64, // catch/Result nodes per SLOC
    pub unsafe_blocks: usize,       // Rust only
    pub type_annotation_ratio: f64, // typed params / total params
    pub call_graph: CallGraph,
}
```

#### Research Insights: tree-sitter Implementation

**S-Expression Queries (concrete, tested patterns):**

Rust -- function definitions and call expressions:
```scheme
;; Function definitions
(function_item name: (identifier) @func.def)
;; Free function calls: foo(...)
(call_expression function: (identifier) @func.call)
;; Method calls: obj.method(...)
(call_expression function: (field_expression field: (field_identifier) @func.call))
;; Scoped calls: module::func(...)
(call_expression function: (scoped_identifier name: (identifier) @func.call))
```

Python:
```scheme
(function_definition name: (identifier) @func.def)
(call function: (identifier) @func.call)
(call function: (attribute attribute: (identifier) @func.call))
```

TypeScript:
```scheme
(function_declaration name: (identifier) @func.def)
(call_expression function: (identifier) @func.call)
(call_expression function: (member_expression property: (property_identifier) @func.call))
```

**Query execution pattern:**
```rust
let query = Query::new(&language, query_source).expect("bad query");
let mut cursor = QueryCursor::new();
for m in cursor.matches(&query, tree.root_node(), source.as_bytes()) {
    for cap in m.captures {
        let name = &query.capture_names()[cap.index as usize];
        let text = &source[cap.node.byte_range()];
        // Build CallEdge from @func.def / @func.call context
    }
}
```

**Parse quality computation:**
```rust
let tree = parser.parse(source, None);
match tree {
    None => { /* Timeout/cancellation -- fall back to Layer 1 */ }
    Some(tree) => {
        let root = tree.root_node();
        if root.has_error() {
            let total = count_nodes(root);
            let errors = count_error_nodes(root);
            let parse_quality = 1.0 - (errors as f64 / total as f64);
        }
    }
}
```

**Performance optimization (from ast-grep):** Filter by `node.kind_id()` early. Maintain a `BitSet` of relevant kind IDs to skip irrelevant nodes -- 40% speedup reported.

**Parallelism:** `Parser` is `!Send`. For rayon batch scanning, use `thread_local!` or create a new `Parser` per task. `Query` objects should be compiled once and cloned per thread.

### Layer 3: Metric Metabolism (`crates/ix-code/src/trajectory.rs`)

**Purpose:** Temporal derivatives of code metrics over git history.

**Design:**
- Use `git2` crate (libgit2 bindings) behind `trajectory` feature
- Walk git log for a given file path, extract blob at each commit
- Run Layer 1 analysis on each historical version
- Compute time series of each metric, then apply ix-signal functions:
  - `ewma()` -- smoothed trend (alpha configurable, default 0.3)
  - `difference()` -- first derivative (velocity)
  - `difference(difference())` -- second derivative (acceleration)
  - `rolling_std()` -- volatility window

**Minimum history threshold:** Require >= 5 commits touching the file for meaningful trajectory. Below that, return metrics with `confidence: 0.0` and `reason: "insufficient_history"`.

**Discontinuity detection:** After squash merges or rebases, metric jumps > 3 sigma from rolling mean are flagged as `discontinuity: true` and excluded from velocity/acceleration calculation.

```rust
pub struct MetricTrajectory {
    pub file_path: String,
    pub metric_name: String,        // e.g., "cyclomatic"
    pub current: f64,
    pub ewma: f64,
    pub velocity: f64,              // first derivative
    pub acceleration: f64,          // second derivative
    pub volatility: f64,            // rolling std
    pub trend: Trend,               // Rising, Falling, Stable, Volatile
    pub n_commits: usize,
    pub confidence: f64,            // 0.0-1.0 based on history depth
    pub discontinuities: Vec<usize>, // commit indices with jumps
}

pub enum Trend { Rising, Falling, Stable, Volatile }

pub fn compute_trajectory(
    repo_path: &Path,
    file_path: &str,
    metric: &str,
    max_commits: usize,     // default 50
) -> Result<MetricTrajectory, TrajectoryError>;
```

#### Research Insights: git2 Implementation

**File history walk pattern (pathspec-optimized):**
```rust
use git2::{Repository, Sort, DiffOptions};

fn file_history(repo: &Repository, path: &str, max: usize) -> Vec<git2::Oid> {
    let mut revwalk = repo.revwalk().unwrap();
    revwalk.set_sorting(Sort::TIME | Sort::TOPOLOGICAL).unwrap();
    revwalk.push_head().unwrap();

    let mut opts = DiffOptions::new();
    opts.pathspec(path);  // libgit2 optimizes internally

    let mut result = Vec::new();
    for oid in revwalk.flatten() {
        let commit = repo.find_commit(oid).unwrap();
        let tree = commit.tree().unwrap();
        let parent_tree = commit.parents().next().map(|p| p.tree().unwrap());
        let diff = repo.diff_tree_to_tree(
            parent_tree.as_ref(), Some(&tree), Some(&mut opts)
        ).unwrap();
        if diff.deltas().len() > 0 {
            result.push(oid);
            if result.len() >= max { break; }
        }
    }
    result
}
```

**Blob extraction (no checkout needed):**
```rust
fn file_at_commit(repo: &Repository, oid: git2::Oid, path: &str) -> Option<Vec<u8>> {
    let commit = repo.find_commit(oid).ok()?;
    let tree = commit.tree().ok()?;
    let entry = tree.get_path(std::path::Path::new(path)).ok()?;
    let blob = repo.find_blob(entry.id()).ok()?;
    Some(blob.content().to_vec())
}
```

**Thread safety:** `Repository` is `Send` but not `Sync`. Open one per rayon thread:
```rust
paths.par_iter().for_each(|path| {
    let repo = Repository::open(".").unwrap(); // one per thread
    let history = file_history(&repo, path, 50);
});
```

**Edge cases:**
- Renames: `diff.find_similar(None)` detects renames but conflicts with pathspec -- diff without pathspec, call `find_similar`, then filter manually
- Merge commits: skip or diff only against first parent (`--first-parent` behavior)
- Windows: always use forward slashes in pathspecs (`"src/main.rs"` not `"src\\main.rs"`)

> **Architecture insight:** Layer 3 should work with Layer 1 metrics alone (no semantic dependency). When `semantic` feature is also enabled, trajectory can track richer metrics. This makes `trajectory` independently useful.

### Layer 4: Topological Code Structure (`crates/ix-code/src/topology.rs`)

**Purpose:** Structural complexity via persistent homology on call graphs.

**Design:**
- Take `CallGraph` from Layer 2 (or keyword-based approximation from Layer 1)
- Convert to `ix-graph::Graph` adjacency list
- Build Vietoris-Rips filtration:
  - **Filtration function:** Edge weight = inverse of call frequency (or 1.0 for static analysis). Low weight = tight coupling = appears early in filtration.
  - Birth time for edge (A,B) = 1.0 / call_count(A->B) (minimum 0.01)
  - Vertices born at t=0
- Feed `SimplexStream` to `ix-topo::compute_persistence()`
- Extract Betti numbers:
  - **beta_0:** Connected components (independent subsystems)
  - **beta_1:** Loops / cycles (circular dependencies)
  - **beta_2:** Voids (higher-order structural holes)

**Keyword-based call-graph fallback:** When tree-sitter is unavailable, extract edges by scanning for function name mentions in other function bodies. Noisy but workable for Rust (explicit paths). Set `parse_quality` accordingly.

```rust
pub struct CodeTopology {
    pub betti_0: usize,             // connected components
    pub betti_1: usize,             // cycles
    pub betti_2: usize,             // voids
    pub persistence_pairs: Vec<(f64, f64)>,  // (birth, death) for H1
    pub max_persistence: f64,       // longest-lived feature
    pub total_persistence: f64,     // sum of all lifetimes
    pub n_nodes: usize,
    pub n_edges: usize,
    pub parse_quality: f64,         // inherited from Layer 2
}

pub fn compute_code_topology(call_graph: &CallGraph) -> CodeTopology;
```

#### Research Insights: Persistent Homology on Call Graphs

**Filtration implementation (connecting to ix-topo API):**
```rust
use ix_topo::simplicial::rips_complex;
use ix_topo::persistence::compute_persistence;

pub fn compute_code_topology(cg: &CallGraph) -> CodeTopology {
    let n = cg.nodes.len();
    // Build symmetric distance matrix: d(A,B) = 1/call_count, min 0.01
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    for edge in &cg.edges {
        let i = node_index(&cg.nodes, &edge.caller);
        let j = node_index(&cg.nodes, &edge.callee);
        let d = 1.0 / edge.weight.max(0.01);
        dist[i][j] = dist[i][j].min(d);
        dist[j][i] = dist[i][j]; // MUST symmetrize for Rips
    }
    let stream = rips_complex(&dist, 10.0, 2); // cap at dim 2 for H1
    let pairs = compute_persistence(&stream);
    // Extract Betti numbers from persistence pairs...
}
```

> **Performance insight:** Cap Rips complex at dimension 1 (H0 + H1 only). Full Rips on 200 nodes generates O(n^2) edges and O(n^3) triangles. With dim cap at 1: 10-100ms for 200-node graphs. Add guard rejecting n > 500 nodes.

> **Critical:** Symmetrize the distance matrix before Rips. Call graphs are directed, but Vietoris-Rips requires a metric space. Use `min(d(A,B), d(B,A))`. Directed flag complexes are more complex -- defer to future phase.

**Long-lived H1 features** (large death-birth) indicate genuine circular dependency cycles. Short-lived features are incidental -- use `significant()` with persistence threshold to filter noise.

### Layer 5: Risk Delta Governance Gates (`crates/ix-code/src/gates.rs`)

**Purpose:** Hexavalent verdicts on PRs/diffs via scorecard indirection.

**Scorecard indirection pattern** (from Codex brainstorm):
```
Metric -> Signal -> Scorecard -> Policy -> Decision
```

- **Metric:** Raw measurement from Layers 1-4 (e.g., cyclomatic=15.0)
- **Signal:** Interpreted measurement (e.g., "high_nesting_in_changed_function")
- **Scorecard:** Grouped signals for an entity (file, module, PR)
- **Policy:** Governance rule over scorecard (e.g., "fail if risk_delta > 0.3")
- **Decision:** Hexavalent verdict + confidence + explanation

**Hexavalent extension:** ix-types already has the 6-valued `Hexavalent` enum (T/P/U/D/F/C). ix-governance's `TruthValue` is the 4-valued subset. Layer 5 uses `Hexavalent` directly from ix-types.

**Risk delta computation:**
```rust
pub struct RiskDelta {
    pub entity: String,             // file path or function name
    pub metric_deltas: Vec<MetricDelta>,
    pub composite_risk_delta: f64,  // weighted sum of normalized deltas
    pub verdict: Hexavalent,
    pub confidence: f64,
    pub signals: Vec<Signal>,
    pub provenance: Provenance,
}

pub struct MetricDelta {
    pub name: String,
    pub before: f64,
    pub after: f64,
    pub delta: f64,
    pub normalized_delta: f64,      // delta / baseline_std
}

pub struct Signal {
    pub name: String,               // e.g., "complexity_spike"
    pub severity: f64,              // 0.0-1.0
    pub description: String,
}

pub struct Provenance {
    pub analyzer_version: String,
    pub feature_schema_version: u32,
    pub model_version: Option<String>,
    pub timestamp: String,
}

pub struct Scorecard {
    pub entity: String,
    pub signals: Vec<Signal>,
    pub risk_score: f64,
    pub verdict: Hexavalent,
    pub confidence: f64,
}
```

**Verdict mapping:**
| Condition | Verdict |
|-----------|---------|
| risk_delta <= 0.0 | T (True = safe) |
| risk_delta < 0.15 and high confidence | P (Probable = likely safe) |
| insufficient data or mixed signals | U (Unknown) |
| risk_delta 0.15-0.30 and some concerns | D (Doubtful) |
| risk_delta > 0.30 | F (False = unsafe, block) |
| High complexity but high test coverage | C (Contradictory = needs human review) |

**Escalation integration:** Map verdicts to ix-governance's `EscalationLevel`:
- T/P -> Autonomous
- U -> ProceedWithNote
- D -> AskConfirmation
- F/C -> Escalate

#### Research Insights: Governance Hardening

> **Governance audit (PASS with recommendations):**

1. **Thresholds in YAML, not code.** The alignment policy already defines configurable confidence bands. Hardcoding 0.15/0.30 creates a self-modification risk (Article 5). Extract to `governance/demerzel/policies/code-quality-gate.yaml`.

2. **Define C-resolution procedure.** Escalating on C is necessary but insufficient. Document: who arbitrates after escalation, what evidence breaks the tie, whether the system blocks or proceeds with warning.

3. **Calibrate P/D boundary.** The 0.15 gap between P and D is narrow. Log early decisions for PDCA review per the Kaizen policy. Consider widening to 0.20 initially.

4. **Provenance must include timestamp + input hash** for full Article 7 (Auditability) compliance.

**Updated Provenance struct:**
```rust
pub struct Provenance {
    pub analyzer_version: String,
    pub feature_schema_version: u32,
    pub model_version: Option<String>,
    pub timestamp: String,
    pub input_hash: String,           // FNV-1a hash of input source
    pub governance_persona: String,   // "code-quality-auditor"
}
```

### Layer 6: Advanced Math Features (`crates/ix-code/src/advanced.rs`)

**Purpose:** Leverage the full ix math stack as feature extractors -- hyperbolic embeddings, K-theory invariants, fractal dimensions, spectral analysis, and spatial partitioning. This is the layer that makes ix-code genuinely unique: no other code analysis tool has these capabilities.

**Philosophy:** LLMs embed code in flat Euclidean vector spaces. Code structure is inherently *hierarchical* and *non-Euclidean*. ix already has the math for this -- we just need to connect the pipes.

#### 6a. Hyperbolic Code Embeddings (ix-math/poincare_hierarchy)

Call trees are *trees*. Trees embed naturally in hyperbolic space with zero distortion, while Euclidean space requires O(sqrt(n)) distortion. The Poincare disk is the right geometry.

```rust
use ix_math::poincare_hierarchy::{PoincareEmbedder, HierarchyDecoder};

pub struct HyperbolicCodeMap {
    pub embeddings: Vec<Array1<f64>>,  // one per function, in Poincare ball
    pub root_functions: Vec<String>,    // closest to origin = highest in hierarchy
    pub depth_scores: Vec<f64>,         // distance from origin = depth in call tree
    pub hierarchy_map_score: f64,       // how tree-like is the code structure?
}

pub fn embed_call_hierarchy(cg: &CallGraph, dim: usize) -> HyperbolicCodeMap;
```

**Key insight:** `hierarchy_map_score` (already in ix-math) tells you how tree-like vs how tangled your code is. Low score = spaghetti. High score = clean hierarchy. This is a *single number* that captures architectural quality better than any combination of cyclomatic/Halstead metrics.

**Advantage over LLM embeddings:** Poincare distance between parent-child is small everywhere in the tree, regardless of branching factor. Euclidean embeddings distort wide trees. Code with 50 utility functions under one module embeds naturally in hyperbolic space but crowds in Euclidean space.

#### 6b. K-Theory Invariants (ix-ktheory)

K-groups detect structural properties invisible to topology alone:

```rust
use ix_ktheory::graph_k::{k0_from_adjacency, k1_from_adjacency};

pub struct KTheoryInvariants {
    pub k0_rank: usize,     // "stable equivalence classes" -- interchangeable modules
    pub k0_torsion: Vec<i64>, // cyclic constraints on resource flow
    pub k1_rank: usize,     // feedback cycles (eigenvalue-1 detection)
    pub k1_torsion: Vec<i64>,
}

pub fn compute_k_invariants(cg: &CallGraph) -> KTheoryInvariants;
```

**What K0 tells you:** Functions with equivalent K0 generators are *structurally interchangeable* -- they have the same "resource balance" in the call graph. This identifies refactoring candidates that human review misses.

**What K1 tells you:** Non-trivial K1 means the call graph has feedback cycles that can't be eliminated by reordering. This is a stronger statement than just "there are cycles" (which Betti-1 already detects) -- K1 says the cycles are *algebraically essential*.

#### 6c. Mayer-Vietoris Decomposition (ix-ktheory/mayer_vietoris)

Split a codebase into two overlapping subsets (e.g., "module A" and "module B"), compute K-groups of each piece and the overlap, then verify consistency via the Mayer-Vietoris exact sequence.

```rust
use ix_ktheory::mayer_vietoris::check_exactness;

pub struct ModuleDecomposition {
    pub module_a: KTheoryInvariants,
    pub module_b: KTheoryInvariants,
    pub overlap: KTheoryInvariants,
    pub exact_sequence_valid: bool,  // do the pieces fit together consistently?
    pub boundary_tension: f64,       // how much the interface constrains the modules
}
```

**What this detects:** If `exact_sequence_valid` is false, the module boundary is *algebraically wrong* -- the two modules don't compose cleanly. This catches architectural smells that no metric or lint can see.

#### 6d. Spectral Analysis of Metric Time Series (ix-signal/fft)

Apply DFT/FFT to code metric trajectories (from Layer 3) to detect periodic patterns:

```rust
use ix_signal::fft::fft;

pub struct MetricSpectrum {
    pub dominant_frequency: f64,    // commits^-1
    pub dominant_period: f64,       // commits (e.g., 14 = every 2 weeks)
    pub spectral_entropy: f64,      // high = noisy, low = periodic
    pub power_spectrum: Vec<f64>,
}

pub fn analyze_metric_spectrum(trajectory: &[f64]) -> MetricSpectrum;
```

**What this detects:** Sprint cycles (complexity spikes every 2 weeks then drops), tech debt accumulation (low-frequency trend), and chaotic churn (high spectral entropy). The `dominant_period` maps directly to team process patterns.

#### 6e. BSP Spatial Partitioning for Code Similarity (ix-math/bsp, ix-gpu/bsp_knn)

Embed function metrics as points in 20-D space (Layer 1 feature vectors), build a BSP tree, then use nearest-neighbor queries for:

```rust
use ix_math::bsp::BspTree;

pub struct CodeSimilarityIndex {
    tree: BspTree<20>,             // 20-D feature space
    function_names: Vec<String>,
}

impl CodeSimilarityIndex {
    /// Build index from all function-level feature vectors in a codebase
    pub fn build(metrics: &[CodeMetrics]) -> Self;

    /// Find k functions most similar to query function
    pub fn similar_functions(&self, query: &CodeMetrics, k: usize) -> Vec<(String, f64)>;

    /// Find functions within radius (detect clones / near-duplicates)
    pub fn region_query(&self, center: &CodeMetrics, radius: f64) -> Vec<String>;

    /// Cluster functions by structural similarity
    pub fn detect_clusters(&self) -> Vec<Vec<String>>;
}
```

**What BSP gives you:**
- **Clone detection** -- functions with near-identical feature vectors are structural clones even if syntactically different
- **Anomaly detection** -- functions far from all neighbors in BSP space are outliers worth reviewing
- **Refactoring targets** -- tight clusters of similar functions suggest abstraction opportunities
- **GPU acceleration** via ix-gpu `bsp_knn` for large codebases (>10K functions)

#### 6f. Fractal Dimension of Code Change Patterns (ix-fractal)

Measure the fractal dimension of how code changes distribute across the codebase:

```rust
pub struct ChangePatternFractal {
    pub box_counting_dim: f64,     // fractal dimension of change distribution
    pub lacunarity: f64,           // gap structure -- how clustered are changes?
    pub self_similarity_ratio: f64, // do changes at file level mirror module level?
}

pub fn compute_change_fractal(change_history: &[(String, usize)]) -> ChangePatternFractal;
```

**What fractal dimension tells you:** If changes follow a Cantor-set-like pattern (high dimension but concentrated), the codebase has "hot zones" that dominate maintenance. If changes are uniformly distributed (dimension ~ 1.0), maintenance effort is spread evenly. Low self-similarity indicates different refactoring patterns at different scales -- a possible organizational smell.

#### 6g. Sedenion Zero-Divisor Analysis (ix-sedenion)

The most exotic feature: encode 16-dimensional module interaction vectors as sedenions. Sedenions have zero divisors -- non-zero elements whose product is zero.

```rust
use ix_sedenion::sedenion::Sedenion;

pub struct CouplingFragility {
    pub zero_divisor_pairs: Vec<(String, String)>,  // modules whose interaction "cancels"
    pub fragility_score: f64,                        // how many near-zero-divisor pairs exist
}
```

**What zero divisors mean for code:** Two modules whose interaction vector is a near-zero-divisor pair exhibit "coupling fragility" -- they interact in a way that looks stable but is algebraically degenerate. Small perturbations to either module can cause disproportionate effects. This is a completely novel code quality signal.

#### Layer 6 Feature Gate

```toml
[features]
advanced = ["dep:ix-ktheory", "dep:ix-fractal", "dep:ix-sedenion"]
full = ["semantic", "trajectory", "topology", "gates", "advanced"]
```

#### Layer 6 MCP Tools

| Tool | Input | Output |
|------|-------|--------|
| `code.hyperbolic_map` | call graph | Poincare embeddings + hierarchy score |
| `code.k_invariants` | call graph adjacency matrix | K0/K1 groups + torsion |
| `code.metric_spectrum` | metric trajectory | FFT dominant frequency + spectral entropy |
| `code.similarity_index` | directory path | BSP-indexed function clusters + anomalies |
| `code.change_fractal` | git repo + path glob | Box-counting dimension + lacunarity |

> **Updated tool count:** 14 new tools total (9 from Layers 2-5 + 5 from Layer 6). Parity test target: **59** (45 existing + 14 new).

### Layer 7: Physics-Inspired Features (`crates/ix-code/src/physics.rs`)

**Purpose:** Apply physics and mechanics analogies to code analysis. Code evolves like a physical system -- it has energy, entropy, symmetry, vibration modes, and phase transitions. These features are unique to ix and impossible for traditional static analysis tools.

#### 7a. Chaos-Theoretic Code Stability (ix-chaos)

**Lyapunov exponents** measure how sensitive a system is to initial conditions. Applied to code: how much does a small change to one function ripple through the codebase?

```rust
use ix_chaos::lyapunov::lyapunov_exponent;

pub struct CodeStability {
    pub lyapunov_exponent: f64,     // positive = chaotic (small changes amplify)
    pub stability_class: StabilityClass,
    pub bifurcation_parameter: Option<f64>, // if near a phase transition
}

pub enum StabilityClass {
    Stable,          // lambda < 0: changes dampen
    Marginal,        // lambda ~ 0: neutral stability
    Chaotic,         // lambda > 0: small changes amplify
    Bifurcating,     // near a qualitative phase transition
}
```

**Bifurcation detection:** Track how code metrics respond to incremental changes. When a small PR causes a disproportionate metric shift, the codebase is near a bifurcation point -- a phase transition where architectural decisions become critical.

#### 7b. Kalman-Filtered Code Quality (ix-signal)

Raw code metrics are noisy. The **Kalman filter** provides optimal estimation of "true" code quality from noisy observations:

```rust
use ix_signal::kalman::KalmanFilter;

pub struct FilteredQuality {
    pub estimated_quality: f64,      // Kalman-filtered quality score
    pub uncertainty: f64,            // estimation uncertainty (P matrix diagonal)
    pub innovation: f64,             // surprise of latest observation vs prediction
    pub is_anomaly: bool,            // innovation > 3*sqrt(S) = anomalous commit
}

pub fn kalman_filter_metrics(
    metric_history: &[f64],
    process_noise: f64,      // Q: how much does true quality drift per commit?
    measurement_noise: f64,  // R: how noisy are our metric observations?
) -> Vec<FilteredQuality>;
```

**What this gives you:** The Kalman innovation (surprise) is a better anomaly detector than raw sigma thresholds. A commit that looks normal in raw metrics but has high Kalman innovation means the codebase is behaving unexpectedly.

#### 7c. Wavelet Multi-Scale Analysis (ix-signal)

FFT decomposes into global frequencies. **Wavelets** give time-localized frequency analysis -- see *when* complexity spikes happened, not just *that* they exist:

```rust
use ix_signal::wavelet;

pub struct MultiScaleAnalysis {
    pub coarse_trend: Vec<f64>,      // long-term quality trajectory
    pub fine_detail: Vec<f64>,       // per-commit noise/spikes
    pub scale_energy: Vec<f64>,      // energy at each scale (which timescale dominates?)
    pub dominant_scale: usize,       // most active timescale (sprint? quarter? year?)
}

pub fn wavelet_decompose_metrics(trajectory: &[f64], levels: usize) -> MultiScaleAnalysis;
```

**What wavelets reveal:** If `dominant_scale` = 10 commits (~2 weeks), complexity follows sprint cycles. If `dominant_scale` = 50 commits (~quarter), there's a deeper organizational pattern. High energy at fine scale = noisy/chaotic development.

#### 7d. Markov Code Evolution (ix-graph)

Model code quality transitions as a **Markov chain**:

```rust
use ix_graph::markov::MarkovChain;

pub struct CodeEvolutionModel {
    pub transition_matrix: Vec<Vec<f64>>,  // P[i][j] = prob of quality state i -> j
    pub states: Vec<String>,               // "healthy", "degrading", "critical", "recovering"
    pub steady_state: Vec<f64>,            // long-run distribution
    pub mean_time_to_critical: f64,        // expected commits until "critical" state
    pub current_state: String,
}

pub fn model_code_evolution(
    quality_history: &[f64],
    n_states: usize,
) -> CodeEvolutionModel;
```

**What this predicts:** `mean_time_to_critical` tells you how many commits until the code likely reaches a critical quality state. The steady-state distribution tells you the *equilibrium* quality level -- where the code naturally settles given current development patterns.

#### 7e. Lie Group Symmetry Analysis (ix-dynamics)

Refactorings are **group actions** on code. Some preserve structure (renaming = identity up to labels), others change it (extracting a function = non-trivial transformation). Lie groups formalize which transformations are "safe":

```rust
use ix_dynamics::lie::{LieGroup, LieAlgebra};

pub struct RefactoringSymmetry {
    pub symmetry_group_dim: usize,  // dimension of the symmetry group
    pub invariant_metrics: Vec<String>,  // metrics preserved by the refactoring
    pub broken_symmetries: Vec<String>,  // metrics that change
    pub is_isometry: bool,              // does the refactoring preserve all distances?
}

pub fn analyze_refactoring_symmetry(
    before: &CodeMetrics,
    after: &CodeMetrics,
) -> RefactoringSymmetry;
```

**What symmetry tells you:** If a PR breaks many symmetries (changes metrics that should be invariant under the stated refactoring type), it's likely not a pure refactoring -- it's smuggling in behavioral changes.

#### 7f. Graph Laplacian Spectral Analysis (NEW -- needs implementation)

The **Laplacian matrix** of the call graph encodes its "vibrational modes" -- like how a drumhead vibrates:

```rust
pub struct LaplacianSpectrum {
    pub eigenvalues: Vec<f64>,          // sorted ascending
    pub algebraic_connectivity: f64,    // lambda_2: how well-connected is the graph?
    pub spectral_gap: f64,             // lambda_2 / lambda_max: mixing time proxy
    pub cheeger_bound: f64,            // lower bound on min-cut / graph partitioning
    pub n_connected_components: usize,  // count of zero eigenvalues
    pub fiedler_vector: Vec<f64>,      // eigenvector of lambda_2: natural bisection
}

pub fn compute_laplacian_spectrum(cg: &CallGraph) -> LaplacianSpectrum;
```

**What eigenvalues tell you:**
- **lambda_2 (algebraic connectivity):** How hard it is to split the codebase into two disconnected halves. Low = fragile, high = well-connected.
- **Fiedler vector:** The natural way to bisect the codebase into two modules. This is *the* mathematically optimal module boundary.
- **Spectral gap:** How quickly information (bugs, changes) propagate through the code. Small gap = information bottleneck.

**Implementation:** Needs eigenvalue decomposition of the graph Laplacian L = D - A. Use `ndarray-linalg` or implement power iteration for the few smallest eigenvalues.

#### 7g. Ollivier-Ricci Curvature on Call Graphs (NEW -- needs implementation)

**Ricci curvature** from Riemannian geometry, adapted for graphs by Ollivier. Measures local geometry of each edge:

```rust
pub struct EdgeCurvature {
    pub edge: (String, String),
    pub ricci_curvature: f64,  // -1 to 1
    // positive = well-connected neighborhood (redundant paths)
    // zero = tree-like (no redundancy)
    // negative = bottleneck (bridge between clusters)
}

pub struct GraphCurvatureProfile {
    pub edge_curvatures: Vec<EdgeCurvature>,
    pub mean_curvature: f64,
    pub bottleneck_edges: Vec<(String, String)>,  // most negative curvature
    pub robust_edges: Vec<(String, String)>,       // most positive curvature
}

pub fn compute_ricci_curvature(cg: &CallGraph) -> GraphCurvatureProfile;
```

**What curvature tells you:**
- **Negative curvature edges** are bridges/bottlenecks -- if they break, the graph disconnects. These are the functions where bugs have maximum blast radius.
- **Positive curvature** regions have redundant paths -- resilient to local failures.
- **Mean curvature** of the whole graph is a single-number measure of architectural robustness.

**Implementation:** Ollivier-Ricci curvature uses optimal transport (Wasserstein distance) between probability distributions on node neighborhoods. Compute via linear programming or Sinkhorn iterations.

#### 7h. Sheaf Cohomology for Interface Consistency (NEW -- needs implementation)

A **sheaf** assigns data to open sets (code modules) with consistency conditions on overlaps (interfaces). Sheaf cohomology detects *where consistency fails*:

```rust
pub struct InterfaceConsistency {
    pub h0: usize,  // connected components of consistent behavior
    pub h1: usize,  // independent inconsistencies across module boundaries
    pub inconsistent_interfaces: Vec<InterfaceViolation>,
}

pub struct InterfaceViolation {
    pub module_a: String,
    pub module_b: String,
    pub shared_type: String,    // the type/function both modules use
    pub violation: String,      // how they disagree
}

pub fn check_sheaf_consistency(modules: &[ModuleMetrics]) -> InterfaceConsistency;
```

**What H1 > 0 means:** There are interfaces where two modules have *incompatible assumptions* about shared types or behaviors. This is the algebraic topology version of "integration bugs that unit tests can't catch."

#### 7i. Entropy Production / Thermodynamic Code Degradation (NEW)

Inspired by the **second law of thermodynamics**: code entropy increases over time without active maintenance.

```rust
pub struct ThermodynamicProfile {
    pub entropy_rate: f64,           // bits/commit: rate of disorder increase
    pub negentropy_ratio: f64,       // fraction of commits that reduce entropy (refactoring)
    pub equilibrium_entropy: f64,    // predicted steady-state entropy
    pub time_to_heat_death: f64,     // commits until max entropy (unmaintainable)
    pub free_energy: f64,            // available capacity for new features
}

pub fn compute_thermodynamic_profile(metric_history: &[Vec<f64>]) -> ThermodynamicProfile;
```

**What this means:** `free_energy` is the capacity to add features without increasing disorder. When it approaches zero, every new feature creates as much mess as it adds value -- the codebase is thermodynamically dead.

#### 7j. Renormalization Group for Multi-Scale Structure (NEW)

From quantum field theory: analyze code at progressively coarser scales, tracking which features are **relevant** (survive scaling) vs **irrelevant** (disappear):

```rust
pub struct RenormalizationFlow {
    pub scales: Vec<ScaleLevel>,  // function -> file -> module -> crate
    pub relevant_metrics: Vec<String>,    // metrics that matter at all scales
    pub irrelevant_metrics: Vec<String>,  // metrics that vanish at coarse scale
    pub fixed_point: Option<Vec<f64>>,    // scale-invariant metric profile (if exists)
    pub is_scale_invariant: bool,         // does the code "look the same" at all scales?
}

pub struct ScaleLevel {
    pub name: String,          // "function", "file", "module", "crate"
    pub n_entities: usize,
    pub metric_distribution: Vec<f64>,
}

pub fn renormalization_flow(metrics_by_level: &[Vec<CodeMetrics>]) -> RenormalizationFlow;
```

**What this tells you:** If `is_scale_invariant` is true, the codebase has self-similar structure -- the same patterns repeat at every level. This is usually healthy. If certain metrics are "relevant" (survive coarsening), they capture real architectural properties. "Irrelevant" metrics are just noise at the function level.

#### Layer 7 Feature Gate & Dependencies

```toml
[features]
physics = ["dep:ix-chaos", "dep:ix-dynamics"]
full = ["semantic", "trajectory", "topology", "gates", "advanced", "physics"]
```

Laplacian spectrum, Ricci curvature, sheaf cohomology, thermodynamics, and renormalization are new implementations within ix-code itself (using ndarray for linear algebra). No new external crates needed.

#### Layer 7 MCP Tools

| Tool | Input | Output |
|------|-------|--------|
| `code.stability` | file path + git range | Lyapunov exponent + bifurcation detection |
| `code.kalman_quality` | metric trajectory | Filtered quality + anomaly detection |
| `code.wavelet_analysis` | metric trajectory | Multi-scale decomposition + dominant timescale |
| `code.evolution_model` | quality history | Markov transition matrix + mean time to critical |
| `code.symmetry` | before/after source | Refactoring symmetry analysis |
| `code.laplacian_spectrum` | call graph | Eigenvalues + algebraic connectivity + Fiedler bisection |
| `code.curvature` | call graph | Ollivier-Ricci per edge + bottleneck detection |
| `code.sheaf_consistency` | module metrics | Interface consistency + H0/H1 cohomology |
| `code.thermodynamics` | metric history | Entropy rate + free energy + time to heat death |
| `code.renormalization` | metrics by scale | Scale flow + relevant/irrelevant metric classification |

> **Updated total:** 24 new tools (9 from L2-5 + 5 from L6 + 10 from L7). Parity test target: **69** (45 existing + 24 new).

## MCP Tools

### New Tools (6 tools, all in domain "code")

| Tool | Layer | Skill Name | Governance |
|------|-------|-----------|------------|
| Semantic analysis | 2 | `code.semantic` | deterministic |
| Metric trajectory | 3 | `code.trajectory` | empirical |
| Code topology | 4 | `code.topology` | deterministic |
| Risk score | 5 | `code.risk_score` | safety |
| Hotspots | 3+4 | `code.hotspots` | empirical |
| Quality gate | 5 | `code.quality_gate` | safety |

**Every scoring tool returns:**
```json
{
  "result": { ... },
  "confidence": 0.85,
  "top_features": ["cyclomatic_velocity", "betti_1"],
  "provenance": {
    "analyzer_version": "0.2.1",
    "feature_schema_version": 2,
    "layers_used": [1, 3, 4]
  }
}
```

**Unsupported input handling:** When language is unsupported or file is binary, return:
```json
{
  "result": null,
  "confidence": 0.0,
  "reason": "unsupported_language",
  "provenance": { ... }
}
```

### Tool Schemas

**`code.semantic`**
- Input: `{ source: string, language: string }` or `{ path: string }`
- Output: SemanticMetrics + call_graph edges

**`code.trajectory`**
- Input: `{ repo_path: string, file_path: string, metric?: string, max_commits?: int }`
- Output: MetricTrajectory per metric (or all metrics if not specified)

**`code.topology`**
- Input: `{ source: string, language: string }` or `{ path: string }` or `{ call_graph: edges[] }`
- Output: CodeTopology with Betti numbers + persistence pairs

**`code.risk_score`**
- Input: `{ before: string, after: string, language: string }` (two source versions)
- Output: RiskDelta with verdict + signals + provenance

**`code.hotspots`**
- Input: `{ repo_path: string, glob?: string, top_n?: int }`
- Output: Ranked list of risky files/functions with trajectory + topology data

**`code.quality_gate`**
- Input: `{ before: string, after: string, language: string, policy?: string }`
- Output: Scorecard with pass/warn/fail/escalate + violated rules

### Batch Scanning Tool

**`code.scan`** -- orchestration tool for dogfooding:
- Input: `{ directory: string, glob?: string, recursive?: bool }`
- Output: Summary statistics across all files (aggregated scorecards)
- Walks directory, detects languages, runs Layers 1-4, rolls up

### Agent-Native Tools (discovered during deepening)

> **Agent-native review** identified 2 missing tools critical for real-world agent workflows:

**`code.diff_scope`** -- PR/commit range scoping:
- Input: `{ repo_path: string, base_ref: string, head_ref: string }` or `{ file_list: string[] }`
- Output: List of changed files/functions with before/after snapshots
- **Why:** Agents reviewing PRs need to scope analysis to changed code only. Without this, they must `scan` the entire directory and manually filter.

**`code.explain`** -- Human-readable verdict rationale:
- Input: `{ risk_delta: RiskDelta }` or `{ scorecard: Scorecard }`
- Output: Natural-language explanation with evidence links
- **Why:** After `quality_gate` returns "fail", agents need to explain WHY to the user.

> **Tool composability:** Tools should accept either a path OR a prior result payload to avoid redundant computation. Consider a `scan_id` or structured result reference for cross-tool threading.

**Updated tool count:** 9 new tools (was 7), parity test target: 54 (was 52).

## Aggregation Strategy

Function-level is the **canonical unit**. Roll up to file/module preserving distribution shape:

```rust
pub struct AggregatedMetrics {
    pub entity: String,             // file or module path
    pub n_functions: usize,
    pub mean_cyclomatic: f64,
    pub p90_cyclomatic: f64,        // 90th percentile
    pub max_cyclomatic: f64,
    pub n_high_risk: usize,         // functions above threshold
    pub n_medium_risk: usize,
    pub n_low_risk: usize,
    pub gini_complexity: f64,       // inequality of complexity distribution
    pub entropy_complexity: f64,    // Shannon entropy of complexity bins
}
```

**Gini coefficient** catches the "one catastrophic function hidden among trivial helpers" pattern that averaging masks.

## Technical Considerations

### Dependencies

```toml
# New workspace deps (Cargo.toml)
[workspace.dependencies]
tree-sitter = { version = "0.24", optional = true }
tree-sitter-rust = { version = "0.24", optional = true }
tree-sitter-python = { version = "0.24", optional = true }
tree-sitter-typescript = { version = "0.24", optional = true }
git2 = { version = "0.19", optional = true }
```

### Performance

- Layer 1: ~microseconds per file (no change)
- Layer 2: ~1-10ms per file (tree-sitter parsing)
- Layer 3: ~50-500ms per file (git log walk, bounded by max_commits)
- Layer 4: ~1-10ms per module (persistence computation, H0+H1 only)
- Layer 5: ~microseconds (arithmetic on pre-computed scores)

#### Performance Audit Findings

**Batch scanning (200+ files):** Use **rayon, not async**. This is CPU-bound. Rayon work-stealing handles uneven file sizes naturally. Expected: ~400ms on 8 cores (vs ~3s single-threaded).

**String interning:** Use `lasso` crate or `Arc<str>` for function names shared across layers. Names repeat heavily in call graphs, metrics, and persistence diagrams. Per-analysis-run interner avoids thousands of redundant allocations.

**Rips complex guard:** Cap at 500 nodes. For 200 nodes with dim-1 cap: ~19,900 edges max, 10-100ms. Without the dim cap, O(n^3) triangle generation becomes the bottleneck.

**git2 optimization:** Use `revwalk` + `commit.tree().get_path()` OID comparison between consecutive commits to detect changes, rather than full `diff_tree_to_tree` per commit. O(k) where k = commits walked.

**Tree-sitter query caching:** Compile `Query` objects once per language, reuse across files. Query compilation is microseconds but adds up across 200+ files.

### Architecture Impacts

- **No new crates** -- all modules in ix-code behind feature gates
- ix-code Cargo.toml grows from 3 deps to ~8 (with features enabled)
- ix-agent Cargo.toml adds ix-code feature flags for MCP exposure
- Parity test count increases by 7 (from 45 to 52 registry skills)

### Security

- `git2` opens local repos read-only -- no remote operations
- tree-sitter parses untrusted source -- bounded by stack depth and timeout
- No file writes, no network calls, no shell execution

#### Security Audit Findings (8 agents)

**HIGH: Path traversal.** File path inputs must be canonicalized with `std::fs::canonicalize()` then verified against an allowlist of directories. Reject symlinks or resolve before allowlist check. Without this, attackers can pass `../../../etc/shadow`.

**HIGH: Denial of service via crafted source.**
- Enforce file size limit: 1MB max
- tree-sitter cancellation flag: `parser.set_timeout_micros(5_000_000)` (5 second cap)
- Stack overflow guard: parse in a thread with explicit stack size via `std::thread::Builder::new().stack_size(8 * 1024 * 1024)`
- Cap call-graph node count at 500

**MEDIUM: git2 repo escape.** Validate `repo.path()` post-open. Disable submodule recursion. Use `Repository::open_ext` with `NO_SEARCH` to prevent `.git` reference following.

**MEDIUM: Information disclosure.** Map all internal errors (git2, tree-sitter, std::fs) to opaque error codes before JSON-RPC serialization. Never leak absolute paths in error messages.

**MEDIUM: Inline source bypasses file-path checks.** Apply same size limits and parsing timeouts to `source: string` inputs, not just file reads.

## System-Wide Impact

### Interaction Graph

- `ix_code_analyze` (Layer 1) -> consumed by Layer 3 (trajectory), Layer 5 (gates)
- `extract_call_graph` (Layer 2) -> consumed by Layer 4 (topology)
- `compute_trajectory` (Layer 3) -> consumed by Layer 5 (risk delta)
- `compute_code_topology` (Layer 4) -> consumed by Layer 5 (risk delta)
- `ix-governance::AlignmentPolicy` -> consumed by Layer 5 (escalation mapping)
- `ix-signal::ewma/difference` -> consumed by Layer 3 (temporal analysis)
- `ix-topo::compute_persistence` -> consumed by Layer 4 (Betti numbers)

### Error Propagation

- tree-sitter parse failure -> graceful fallback to Layer 1, `parse_quality: 0.0`
- git2 repo not found -> `TrajectoryError::NotAGitRepo`, MCP returns structured error
- Insufficient git history -> `confidence: 0.0`, `reason: "insufficient_history"`
- Empty file -> MI=171.0 (current behavior), trajectory returns constant series with `confidence: 0.0`
- Binary file / unsupported lang -> `None` from `Language::from_path`, MCP returns `reason: "unsupported_language"`

### API Surface Parity

- CLI (`ix-skill run code.semantic`) -- all 7 new tools accessible
- MCP (JSON-RPC) -- all 7 new tools in ix-agent tools list
- Pipeline (ix.yaml) -- all 7 tools usable as pipeline stages
- egui demo -- new "Code Observatory" tab showing all layers

## Acceptance Criteria

### Functional Requirements

- [ ] `code.semantic` returns AST metrics + call-graph for Rust source
- [ ] `code.semantic` returns AST metrics + call-graph for Python source
- [ ] `code.semantic` returns AST metrics + call-graph for TypeScript source
- [ ] `code.semantic` falls back to Layer 1 for unsupported languages with `parse_quality: 0.0`
- [ ] `code.trajectory` returns EWMA/velocity/acceleration for a file in a git repo
- [ ] `code.trajectory` returns `confidence: 0.0` with < 5 commits
- [ ] `code.trajectory` detects and excludes discontinuities (> 3 sigma jumps)
- [ ] `code.topology` returns Betti numbers from a call graph
- [ ] `code.topology` correctly identifies cycles (beta_1 > 0)
- [ ] `code.risk_score` returns hexavalent verdict comparing two source versions
- [ ] `code.risk_score` returns Contradictory when high complexity + high test coverage
- [ ] `code.hotspots` scans a directory and ranks risky files
- [ ] `code.quality_gate` maps verdicts to escalation levels
- [ ] `code.scan` batch-analyzes ix repo itself (dogfood)
- [ ] `code.diff_scope` scopes analysis to a git ref range
- [ ] `code.explain` produces human-readable rationale from any verdict
- [ ] Path traversal rejected for file inputs outside allowlist
- [ ] File size > 1MB rejected with structured error
- [ ] tree-sitter parse timeout enforced (5s cap)

### Non-Functional Requirements

- [ ] Layer 1 performance unchanged (< 1ms per file)
- [ ] Layer 2 < 10ms per file for Rust/Python/TS
- [ ] Layer 3 < 500ms per file (50-commit window)
- [ ] All tools return provenance (analyzer version, schema version)
- [ ] `cargo clippy --workspace -- -D warnings` clean
- [ ] Zero unsafe blocks in new code

### Quality Gates

- [ ] Unit tests for every public function
- [ ] Property tests for metric invariants (cyclomatic >= 1, MI in [0,171])
- [ ] Integration test: round-trip analyze ix-code's own source through all 5 layers
- [ ] Integration test: verify Betti numbers stable across comment-only changes
- [ ] Parity test updated (69 tools -- 45 existing + 24 new)
- [ ] Doc tests on all public API examples

## Implementation Phases

### Phase 1: Foundation (Layer 2 - Semantic)

**Files:**
- `crates/ix-code/Cargo.toml` -- add tree-sitter deps behind `semantic` feature
- `crates/ix-code/src/semantic.rs` -- SemanticMetrics, CallGraph, extract_call_graph()
- `crates/ix-code/src/lib.rs` -- `#[cfg(feature = "semantic")] pub mod semantic;`
- `crates/ix-agent/src/handlers.rs` -- code_semantic handler
- `crates/ix-agent/src/skills/batch3.rs` -- `#[ix_skill]` for code.semantic
- `Cargo.toml` -- workspace deps for tree-sitter

**Deliverables:**
- tree-sitter parsing for Rust with call-graph extraction
- SemanticMetrics struct with parse_quality
- Fallback path when tree-sitter unavailable
- MCP tool: `code.semantic`
- Tests: Rust source -> call-graph edges verified

### Phase 2: Temporal (Layer 3 - Trajectory)

**Files:**
- `crates/ix-code/Cargo.toml` -- add git2 behind `trajectory` feature
- `crates/ix-code/src/trajectory.rs` -- MetricTrajectory, compute_trajectory()
- `crates/ix-agent/src/handlers.rs` -- code_trajectory handler
- `crates/ix-agent/src/skills/batch3.rs` -- `#[ix_skill]` for code.trajectory

**Deliverables:**
- Git history walk -> metric time series
- EWMA, velocity, acceleration via ix-signal
- Discontinuity detection (3-sigma threshold)
- Minimum history check (5 commits)
- MCP tool: `code.trajectory`
- Dogfood test: trajectory of ix-code/src/analyze.rs

### Phase 3: Structural (Layer 4 - Topology)

**Files:**
- `crates/ix-code/src/topology.rs` -- CodeTopology, compute_code_topology()
- `crates/ix-agent/src/handlers.rs` -- code_topology handler
- `crates/ix-agent/src/skills/batch3.rs` -- `#[ix_skill]` for code.topology

**Deliverables:**
- CallGraph -> ix-graph::Graph conversion
- Vietoris-Rips filtration with inverse-frequency weights
- ix-topo persistent homology -> Betti numbers
- MCP tool: `code.topology`
- Test: synthetic call graph with known cycle -> beta_1 = 1

### Phase 4: Governance (Layer 5 - Gates)

**Files:**
- `crates/ix-code/src/gates.rs` -- RiskDelta, Signal, Scorecard, Provenance, verdict mapping
- `crates/ix-code/src/aggregate.rs` -- AggregatedMetrics, Gini coefficient, percentile rollup
- `crates/ix-agent/src/handlers.rs` -- code_risk_score, code_quality_gate, code_hotspots handlers
- `crates/ix-agent/src/skills/batch3.rs` -- 3 `#[ix_skill]` registrations

**Deliverables:**
- Scorecard indirection: Metric -> Signal -> Scorecard -> Policy -> Decision
- Hexavalent verdict mapping with confidence
- Risk delta computation (before/after source comparison)
- Aggregation with Gini coefficient + percentiles
- MCP tools: `code.risk_score`, `code.quality_gate`, `code.hotspots`
- `code.scan` batch tool for dogfooding

### Phase 5: Advanced Math (Layer 6)

**Files:**
- `crates/ix-code/src/advanced.rs` -- HyperbolicCodeMap, KTheoryInvariants, MetricSpectrum, CodeSimilarityIndex, ChangePatternFractal, CouplingFragility
- `crates/ix-code/Cargo.toml` -- add `advanced` feature with deps on ix-ktheory, ix-fractal, ix-sedenion, ix-math (for BSP + Poincare)
- `crates/ix-agent/src/handlers.rs` -- 5 new handlers
- `crates/ix-agent/src/skills/batch3.rs` -- 5 `#[ix_skill]` registrations

**Deliverables:**
- Poincare hierarchy embedding of call graphs + hierarchy_map_score
- K0/K1 invariants from call-graph adjacency matrix
- FFT spectral analysis of metric trajectories
- BSP-indexed code similarity with clone detection + anomaly detection
- Fractal dimension of change patterns
- MCP tools: `code.hyperbolic_map`, `code.k_invariants`, `code.metric_spectrum`, `code.similarity_index`, `code.change_fractal`
- Test: synthetic tree-structured call graph embeds cleanly in Poincare disk (hierarchy_map_score > 0.8)

### Phase 6: Physics-Inspired (Layer 7)

**Files:**
- `crates/ix-code/src/physics.rs` -- CodeStability, FilteredQuality, MultiScaleAnalysis, CodeEvolutionModel, RefactoringSymmetry, LaplacianSpectrum, GraphCurvatureProfile, InterfaceConsistency, ThermodynamicProfile, RenormalizationFlow
- `crates/ix-code/Cargo.toml` -- add `physics` feature with deps on ix-chaos, ix-dynamics
- `crates/ix-agent/src/handlers.rs` -- 10 new handlers
- `crates/ix-agent/src/skills/batch3.rs` -- 10 `#[ix_skill]` registrations

**Sub-phases:**
- **6a: Already-in-ix** (chaos Lyapunov, Kalman filter, wavelets, Markov, Lie symmetry) -- connect existing crate APIs
- **6b: New implementations** (graph Laplacian, Ollivier-Ricci curvature, sheaf cohomology, thermodynamics, renormalization) -- implement within ix-code using ndarray

**Deliverables:**
- 10 MCP tools covering chaos, filtering, multi-scale, evolution, symmetry, spectral, curvature, consistency, thermodynamics, renormalization
- Tests: synthetic systems with known Lyapunov exponents, known graph Laplacian eigenvalues
- Dogfood: run full 7-layer analysis on ix-code itself

### Phase 7: Integration & Polish

**Files:**
- `crates/ix-agent/tests/parity.rs` -- update to 52 tools
- `examples/showcase/advanced/code-observatory.yaml` -- showcase pipeline
- `crates/ix-demo/src/` -- Code Observatory egui tab
- `docs/tutorials/code-observatory.md` + `docs/fr/observatoire-de-code.md`

**Deliverables:**
- Updated parity tests
- Showcase pipeline chaining all 5 layers
- egui demo tab
- EN + FR documentation
- Full dogfood run on ix repo

## Alternative Approaches Considered

1. **rust-code-analysis instead of tree-sitter** -- Rejected: less flexible for custom ML features, limited language coverage, opinionated metrics that may conflict with ix-code's own. (brainstorm consensus)
2. **Separate crates per layer** -- Rejected: would add 4 crates to workspace, increase build times, complicate cross-layer data flow. Feature gates on one crate are simpler.
3. **Replace lightweight engine entirely** -- Rejected by all three brainstorm providers: the lightweight engine is the moat, not the limitation. It's the universal fast path.
4. **Absolute thresholds for governance** -- Rejected: risk deltas are more actionable than absolutes. Legacy code with high complexity shouldn't block PRs; only increasing complexity should.

## Error Handling Pattern

> **Pattern review:** The workspace uses two error patterns: `thiserror` enums in algorithm crates, `String` errors at MCP boundary. Follow the same split.

```rust
// crates/ix-code/src/error.rs -- dedicated thiserror enum
#[derive(Debug, thiserror::Error)]
pub enum CodeError {
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),
    #[error("file too large: {size} bytes (max {max})")]
    FileTooLarge { size: usize, max: usize },
    #[error("parse timeout after {0}ms")]
    ParseTimeout(u64),
    #[error("not a git repository: {0}")]
    NotAGitRepo(String),
    #[error("insufficient history: {commits} commits (min {min})")]
    InsufficientHistory { commits: usize, min: usize },
    #[error("node limit exceeded: {nodes} (max {max})")]
    NodeLimitExceeded { nodes: usize, max: usize },
    #[error("path outside allowlist: {0}")]
    PathTraversal(String),
}
```

MCP handlers convert `CodeError` to structured JSON with opaque error codes (no path leakage).

## Dependencies & Prerequisites

- `tree-sitter` 0.24+ crate available on crates.io (verified)
- `git2` 0.19+ crate available on crates.io (verified)
- ix-topo `compute_persistence()` and `rips_complex()` working (verified in repo)
- ix-signal `ewma()`, `difference()`, `rolling_std()` working (verified in repo)
- ix-governance `AlignmentPolicy` and `EscalationLevel` working (verified in repo)
- ix-types `Hexavalent` enum with T/P/U/D/F/C (verified in repo)

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| tree-sitter grammar quality varies | Noisy call-graphs | parse_quality score, graceful fallback to Layer 1 |
| git2 on Windows linking issues | Build failures | Feature-gate, test in CI on windows-latest |
| Large repos slow trajectory | Timeout | max_commits cap (default 50), async option |
| Noisy keyword-based call-graph | False topology | Clearly mark parse_quality, prefer tree-sitter when available |
| Hexavalent thresholds need tuning | Over/under-gating | Configurable thresholds in policy, start conservative |

## Documentation Plan

- `docs/tutorials/code-observatory.md` -- EN tutorial walking through all 5 layers
- `docs/fr/observatoire-de-code.md` -- FR translation
- Doc comments on all public structs and functions
- Showcase pipeline: `examples/showcase/advanced/code-observatory.yaml`
- Updated CLAUDE.md architecture section

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-04-09-ix-code-observatory-brainstorm.md](../brainstorms/2026-04-09-ix-code-observatory-brainstorm.md)
- Key decisions carried forward: two-lane architecture, metric metabolism, topological analysis, risk delta gates, scorecard indirection

### Internal References

- ix-code current: `crates/ix-code/src/analyze.rs` (Layer 1)
- ix-signal EWMA: `crates/ix-signal/src/timeseries.rs:335` (`ewma()`)
- ix-topo persistence: `crates/ix-topo/src/persistence.rs:74` (`compute_persistence()`)
- ix-graph: `crates/ix-graph/src/graph.rs:8` (`Graph` struct)
- ix-governance policy: `crates/ix-governance/src/policy.rs:10` (`AlignmentPolicy`)
- ix-types hexavalent: `crates/ix-types/src/lib.rs:35` (`Hexavalent` enum)
- Skill registration: `crates/ix-skill-macros/src/lib.rs:45` (`#[ix_skill]` macro)
- MCP handler: `crates/ix-agent/src/handlers.rs:2051` (`code_analyze`)

### Learnings Applied

- Path resolution: use `governance_dir()` walk-up helper, not `CARGO_MANIFEST_DIR` (from docs/solutions/)
- API design: every builder field must map to behavior (from docs/solutions/)
- Performance: avoid heap allocations in hot loops, hoist loop-invariant computation (from docs/solutions/)
- Windows LLD: sentinel element in IX_SKILLS, filter in public APIs (from Phase 1 delivery)
