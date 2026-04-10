# Graph theory in IX — where everything lives

**Purpose:** IX ships graph-theoretic primitives across ten crates and modules.
This guide is the single place to find them. Before adding a new algorithm,
pulling in `petgraph`, or hand-rolling a BFS, read this page first.

**Audience:** humans onboarding to the workspace, and Claude Code sessions
selecting a graph primitive for a task. If you are a Claude Code session
reading this: the CLAUDE.md at the repo root points you here — the six rules
at the bottom of that section are load-bearing.

**Not on this page:** performance benchmarks, theoretical tutorials on any of
the underlying algorithms, or anything that duplicates `cargo doc`. For
background, see Deo's *Graph Theory with Applications* and Skiena's
*Algorithm Design Manual* (chapters 5–9 cover the ground IX implements).

---

## 1. Inventory

IX's graph theory coverage spans three concerns that most projects split across
separate ecosystems:

1. **Classical graph algorithms** — search, shortest paths, DAGs, Markov chains
2. **Topological / algebraic invariants** — persistent homology, K-theory,
   spectral analysis
3. **Applied code analysis** — static call graphs, git-trajectory graphs,
   agent-facing walkers

### 1.1 Classical graph algorithms

#### `crates/ix-graph`

Modules: `graph`, `markov`, `hmm`, `state_space`, `routing`.

- `graph` — generic graph data structures and basic algorithms
- `markov` — discrete-time Markov chains, transition matrices, stationary
  distributions
- `hmm` — Hidden Markov Models with Viterbi decoding, forward-backward,
  Baum-Welch training
- `state_space` — state-space exploration primitives for RL and planning
- `routing` — agent routing for token-efficient skill dispatch

Use when: you need a generic graph, a probabilistic transition model, or
sequence inference over latent states.

#### `crates/ix-pipeline::dag::Dag<N>`

Generic cycle-checked DAG with topological sort and parallel execution.

Key properties:
- Generic over node data type `N`
- Cycle detection enforced on **every edge insertion** — `add_edge` returns
  `DagError::CycleDetected` rather than allowing you to corrupt the graph
- Reverse adjacency maintained for upstream walks
- Deterministic iteration order (insertion order)
- Executor layer in `ix-pipeline::executor` runs independent branches in
  parallel with memoization

Use when: you need a DAG substrate for pipelines, dependency graphs, or any
workflow with acyclicity guarantees.

```rust
use ix_pipeline::dag::Dag;

let mut g: Dag<&str> = Dag::new();
g.add_node("parse", "tree-sitter")?;
g.add_node("analyze", "metrics")?;
g.add_edge("parse", "analyze")?;
// g.add_edge("analyze", "parse")? would return DagError::CycleDetected
```

#### `crates/ix-search`

Modules: `astar`, `qstar`, `mcts`, `adversarial`, `local`, `data_search`,
`uninformed`.

- `uninformed` — BFS and DFS for graph traversal
- `astar` — A* shortest-path search with configurable heuristic
- `qstar` — Q* reinforcement-guided search
- `mcts` — Monte Carlo Tree Search (UCT, progressive widening)
- `adversarial` — minimax and alpha-beta for two-player games
- `local` — local search (hill climbing, simulated annealing adaptations)
- `data_search` — search over typed data structures

Use when: you need shortest paths, tree search, goal-directed traversal, or
adversarial planning.

### 1.2 Topological and algebraic invariants

#### `crates/ix-topo`

Modules: `persistence`, `simplex`, `pointcloud`.

- `persistence` — persistent homology: compute birth/death pairs for H₀
  (connected components) and H₁ (cycles) across a filtration
- `simplex` — simplicial complexes, boundary matrices, reduction
- `pointcloud` — Vietoris-Rips filtrations from distance matrices

Use when: you need to summarize a graph's *shape* — number of components,
independent cycles, how persistent each feature is across filtration scales.

This is the "topological data analysis" stack. It sits on top of any
distance/adjacency matrix and produces Betti numbers and persistence diagrams.

#### `crates/ix-ktheory`

Modules: `graph_k`, `mayer_vietoris`.

- `graph_k` — Grothendieck K₀ and K₁ for graphs
- `mayer_vietoris` — Mayer-Vietoris long exact sequences for decomposable graphs

Use when: you need algebraic invariants that classify graphs up to equivalence,
or you want to decompose a large graph along cut vertices and compute
invariants compositionally.

Rare in practice. Ship only when you need categorical/algebraic machinery —
for day-to-day analysis, `ix-topo` is the right tool.

#### `crates/ix-code::physics` — Laplacian spectrum

Spectral graph theory applied to call graphs. The call-graph Laplacian
`L = D - A` is eigendecomposed via `ix-math::eigen`; the eigenvalues give
you:

- `λ₂` (Fiedler value) — algebraic connectivity, how "tightly coupled" the
  graph is
- Eigenvector of `λ₂` — Fiedler vector, canonical graph-bisection signal
- Spectrum shape — community structure indicators

Use when: you need spectral clustering, community detection, or connectivity
metrics on call graphs.

#### `crates/ix-code::advanced`

Modules in `advanced` for cross-layer analysis of call graphs:

- Hyperbolic embeddings (Poincaré ball) — embed large graphs into hyperbolic
  space where tree-like structure is preserved with low distortion
- BSP (binary space partitioning) spatial search — approximate nearest
  neighbor queries on embedded nodes
- Spectral methods — complementary to `physics` module

Use when: the graph is too large for dense spectral methods, or you need a
geometric embedding for downstream ML.

### 1.3 Applied code analysis

#### `crates/ix-code::semantic`

Tree-sitter-backed extraction of call graphs from Rust source.

Public types:
- `CallGraph` — `{ nodes: Vec<String>, edges: Vec<CallEdge> }`
- `CallEdge` — `{ caller, callee_hint, call_site_line, weight }`
- `CalleeHint` — discriminated `Bare` / `Scoped(segments)` / `MethodCall`

**Important scope limit:** `CallGraph` here is **per translation unit**
(per file). It does not resolve cross-file calls. Cross-file and project-wide
resolution is the job of `ix-context` (WIP).

Use when: you have Rust source text and want a file-local call graph.

#### `crates/ix-code::topology`

Persistent homology applied to call graphs. Wraps `ix-topo` with a
call-graph-specific input format.

Returns:
- `betti_0` — number of connected components in the call graph (islands of
  unrelated functions)
- `betti_1` — number of independent cycles (recursion loops, mutual
  recursion clusters)
- Persistence pairs — how long each feature survives across filtration scales

Use when: you want a quantitative summary of a codebase's structural shape —
how fragmented, how cyclic, how tangled.

#### `crates/ix-context` *(in progress — see [`docs/brainstorms/2026-04-10-context-dag.md`](../brainstorms/2026-04-10-context-dag.md))*

Walker over AST + call graph + import graph + git trajectory, framed as a
deterministic retrieval system for agents. Not yet shipped — the
Prerequisite B refactor (`CalleeHint`) landed in commit `6ef8462`, the rest
is blocked on Windows Application Control for test verification.

When it ships, it will be the **public entry point for agent-facing graph
retrieval** — the walker that turns IX's graph primitives into a Claude-
callable context provider.

---

## 2. Selection matrix

| If you need... | Use | Not |
|---|---|---|
| Generic graph data structure | `ix-graph::graph` | Roll your own adjacency list |
| Cycle-checked DAG with topo sort | `ix-pipeline::dag::Dag<N>` | `petgraph::Graph` + manual check |
| BFS / DFS traversal | `ix-search::uninformed` | Hand-rolled `VecDeque` loop |
| Shortest path with heuristic | `ix-search::astar` | `petgraph::algo::astar` |
| Monte Carlo tree search | `ix-search::mcts` | External MCTS crate |
| Two-player game search | `ix-search::adversarial` | Hand-rolled minimax |
| Discrete-time Markov chain | `ix-graph::markov` | Raw matrix math |
| Viterbi decoding over HMM | `ix-graph::hmm` | External HMM crate |
| Connected components count | `ix-topo::persistence` (Betti 0) | Union-find reinvention |
| Cycle count (graph topology) | `ix-topo::persistence` (Betti 1) | Tarjan reinvention |
| Spectral clustering | `ix-code::physics::laplacian` + `ix-math::eigen` | External spectral library |
| Hyperbolic embedding of large graph | `ix-code::advanced` | External hyperbolic crate |
| Static call graph from Rust source | `ix-code::semantic::extract_call_graph` | Hand-parse with regex |
| Code structure summary (islands, cycles) | `ix-code::topology::compute_code_topology` | Rebuild from scratch |
| Cross-file / project-wide call graph | `ix-context` (WIP) | Do NOT force `ix-code::semantic` to do this |
| Agent-facing context walker over code | `ix-context` (WIP) | Do NOT build a separate walker |

---

## 3. What NOT to do

1. **Do not add `petgraph`, `daggy`, `graph-rs`, or any other generic graph
   crate** as a dependency. IX's primitives cover all common graph shapes.
   If you believe you need one, document the specific shortfall in
   `docs/brainstorms/` first and escalate before the PR.

2. **Do not hand-roll BFS/DFS.** `ix-search::uninformed` is shipped and
   tested. Importing `std::collections::VecDeque` to build a BFS is a
   smell — use the crate.

3. **Do not mix up `ix-pipeline::dag` with `ix-graph::graph`.**
   - `ix-pipeline::dag::Dag<N>` is specifically a *cycle-checked* DAG — use
     it when acyclicity must be guaranteed.
   - `ix-graph::graph` is a general graph — use it when cycles are allowed
     or required.

4. **Do not add cross-file resolution into `ix-code::semantic`.** That crate
   is per-file by design. Cross-file is `ix-context`'s job.

5. **Do not hand-write an adjacency matrix when you can describe the graph
   as nodes + edges.** `ix-pipeline::dag` and `ix-graph::graph` both take
   nodes + edges directly.

6. **Do not reach for embeddings (vector similarity) when structural walks
   would do.** See `docs/brainstorms/2026-04-10-context-dag.md` — the
   whole brainstorm reframes structural retrieval as *the* answer to
   "RAG is a trap for code."

---

## 4. Worked examples

### 4.1 Cycle-checked pipeline with `ix-pipeline::dag`

```rust
use ix_pipeline::dag::{Dag, DagError};

let mut pipeline: Dag<String> = Dag::new();
pipeline.add_node("parse", "tree-sitter parse".to_string())?;
pipeline.add_node("metrics", "compute metrics".to_string())?;
pipeline.add_node("topology", "persistent homology".to_string())?;

pipeline.add_edge("parse", "metrics")?;
pipeline.add_edge("parse", "topology")?;

// Adding this would form a cycle back to parse:
match pipeline.add_edge("metrics", "parse") {
    Err(DagError::CycleDetected(_, _)) => {
        // handled — the DAG substrate protects the invariant
    }
    _ => unreachable!(),
}
```

### 4.2 Call graph + topology summary

```rust
use ix_code::semantic::extract_call_graph;
use ix_code::topology::compute_code_topology;

let source = r#"
    fn a() { b(); c(); }
    fn b() { c(); }
    fn c() { a(); }  // cycle: a -> b -> c -> a
"#;

let call_graph = extract_call_graph(source).expect("parse ok");
// call_graph.edges now carry richer CalleeHint variants (Bare / Scoped / MethodCall)

// Convert to the topology module's input format and analyze:
// (See ix-code::topology::CallGraph for the expected shape.)
```

### 4.3 A* search over an explicit graph

```rust
use ix_search::astar::{astar, AStarProblem};

// Define your AStarProblem trait impl with successors + heuristic,
// then call astar(&problem, start, goal).
// See crates/ix-search/src/astar.rs for the trait definition and
// examples in the tests module.
```

---

## 5. References

**Background reading** (none of these replace reading the actual crate docs):

- Deo, *Graph Theory with Applications to Engineering and Computer Science*
  — classical algorithms, matrix representations, planarity
- Skiena, *Algorithm Design Manual*, chapters 5–9 — practical graph
  problem-solving patterns
- Edelsbrunner & Harer, *Computational Topology: An Introduction* — the
  persistent homology stack used by `ix-topo`
- Chung, *Spectral Graph Theory* — the Laplacian-spectrum methods used by
  `ix-code::physics`
- Bonchi & Pous, *Bisimulation and Coinduction Enhancements* — background
  for graph equivalence classes underpinning `ix-ktheory`

**In-workspace references:**

- `docs/brainstorms/2026-04-10-context-dag.md` — design for the
  `ix-context` walker; sections on node schema, walk strategies, and
  unresolved-edge preservation are the canonical guide to how IX treats
  graphs as structured context
- `docs/brainstorms/2026-04-10-ix-harness-primitives.md` — strategic
  context: why IX's graph primitives are positioned as a "structural
  oracle" backend for agent harnesses
- `crates/ix-code/src/topology.rs` — reference implementation of how to
  bridge a call graph to `ix-topo`'s persistence stack
- `crates/ix-pipeline/src/dag.rs` — canonical cycle-checked DAG with
  commentary on invariants

---

## 6. Adding something genuinely new

If after reading this page you still believe IX needs a graph-theory
primitive that isn't here, the process is:

1. **Write a brainstorm doc** in `docs/brainstorms/YYYY-MM-DD-<topic>.md`
   explaining the gap and the closest existing primitive.
2. **Link the gap to a concrete use case.** "Graph theory completeness"
   is not a use case.
3. **Check whether the primitive belongs in an existing crate** (extending
   `ix-graph` or `ix-topo`) or deserves a new crate. New crates are
   expensive — the workspace is already at 33+ crates.
4. **Escalate via `/octo:brainstorm`** if the design space is open.
5. **Update this guide** in the same PR that ships the new primitive, so
   the next reader of this page knows it exists.

French translation stub: [`docs/fr/recherche-et-graphes/theorie-des-graphes-dans-ix.md`](../fr/recherche-et-graphes/theorie-des-graphes-dans-ix.md)
