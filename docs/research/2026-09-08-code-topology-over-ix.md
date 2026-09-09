# Topological code analysis over IX's own crates

Gap-matrix row **F1** (*topological code analysis — persistent homology over
code structure*), from
[`tars-v1-advanced-math-ix-gap-matrix.md`](tars-v1-advanced-math-ix-gap-matrix.md)
(ix#202). The companion row **B1b** (*persistent homology over a DuckDB column*)
is covered in [§7](#7-b1b-persistent-homology-over-a-duckdb-column).

| Item | Value |
| --- | --- |
| Measured on | `GuitarAlchemist/ix` @ `fa9852c` (`origin/main`, 2026-09-08) |
| Crates measured | 82 — every directory under `crates/` with a `src/` |
| Reproduce | `cargo run -p ix-agent --example code_topology_corpus` |
| Raw data | [`data/2026-09-08-code-topology-ix.csv`](data/2026-09-08-code-topology-ix.csv) |
| Cost incurred | 0 USD — local computation only |

---

## 1. What the matrix said, and what was actually true

F1 is classified `implemented_needs_exposure`, with the note *"Both halves are
built and already exposed separately; nothing joins them."* That classification
**holds, but it understates the situation in one direction and overstates it in
another.** Both corrections were found by reading the code before writing any,
and both changed what this PR had to do.

**It understates how much was already built.** The matrix describes F1 as a
composition still to be made. In fact `crates/ix-code/src/topology.rs` already
existed on `main` — 296 lines, five unit tests, a Vietoris–Rips filtration over
an inverted-weight call graph, feeding `ix_topo::persistence::compute_persistence`.
It arrived with the Code Observatory work (`8c0573c`, "7-layer analysis system")
and has been sitting there since. Nothing in this PR reimplements it.

**It overstates how reachable it was.** `topology.rs` is behind ix-code's
`topology` feature, and at `fa9852c` **no crate in the workspace enabled that
feature**:

| Consumer | ix-code features enabled |
| --- | --- |
| `ix-agent` | `semantic` |
| `ix-context` | `semantic`, `trajectory`, `gates` |
| `ix-duck` | (optional dep; `code-semantic` → `semantic`) |

CI runs `cargo clippy --workspace --all-targets -- -D warnings` and
`cargo test --workspace` — neither passes `--all-features`. So `topology.rs`
was not merely unexposed: **it was never compiled or tested by CI at all.**
rust-analyzer says so independently, reporting the file as *"not included in any
crates"*. This is the same failure class that `crates/ix-duck/sql/pareto_frontier.sql`
documents for ix-duck's `duck`/`udf` features, arrived at by a different route —
there it is a deliberately excluded crate, here it is a feature nobody turned on.

**And the seam was genuinely broken.** `topology.rs` defines its own `CallGraph`
with the comment *"Defined locally … so Phase 3 can build and ship independently
… can be swapped for an alias once Phase 1 is merged."* Phase 1 **is** merged —
`ix_code::semantic::CallGraph` exists and is produced by tree-sitter — but the
swap never happened, and no converter was ever written. `compute_code_topology`
had zero callers outside its own test module. The two `CallGraph` types are not
even structurally compatible: the semantic one carries `CallEdge` records with
syntactic callee hints, the topology one carries `(String, String, f64)`.

So the honest restatement of F1 is: *the mathematics ships, the extractor ships,
the converter between them does not exist, and the mathematics is invisible to
CI.* That is what this PR fixes.

## 2. What was built

Three things, no new topology:

1. **The seam** — `ix_code::semantic::extract_definitions` (new; the existing
   `CallGraph::nodes` mixes definitions with call targets and so cannot answer
   "which file owns this name?"), plus `ix_code::topology::{Unit,
   module_call_graph, call_graph_from_semantic, undirected_edge_count}`.
2. **The exposure** — MCP tool `ix_code_topology` on `ix-agent`, which is also
   what puts `ix-code/topology` on the workspace build for the first time.
3. **The measurement** — `cargo run -p ix-agent --example code_topology_corpus`,
   the runner that produced every number below.

### The module-level graph

One node per source file; an edge `a → b` when a call site in `a` names a
function defined in `b`, weighted by the number of such call sites. Resolution is
by bare name against the union of the definition sets, and is deliberately
conservative — a callee defined in **no** unit is *external* and dropped, a
callee defined in **more than one** is *ambiguous* and dropped. Both counts are
reported rather than swallowed, because they bound what the graph is worth
(§5).

`compute_code_topology` then inverts each weight into a distance (`d = 1/w`,
floored at 0.01), symmetrizes, and builds a Rips filtration capped at
dimension 1.

## 3. The central caveat: what homology adds over McCabe, and what it does not

The premise for F1 is that homology sees cycles and voids that cyclomatic
complexity cannot express. Half of that is true here, and the other half is
worth stating plainly rather than letting a Betti number carry an implication it
cannot support.

**β₁ is not new information.** The filtration stops at dimension 1, so no
triangle ever fills a loop and every 1-cycle survives to infinity. For a graph
with `V` vertices, `E` undirected edges and `β₀` components, that makes

```
β₁ = E − V + β₀
```

which is exactly McCabe's cyclomatic number, applied to the dependency graph
instead of a control-flow graph. This is not a conjecture: the corpus runner
checks it on every row, and it **held on all 82 crates**. It is also pinned as a
unit test (`betti_1_is_the_circuit_rank_while_the_filtration_stops_at_dimension_1`)
so that if the filtration ever gains triangles — at which point β₁ *would* start
meaning something stronger — the claim breaks loudly instead of going stale.

**β₁ is also weaker than it sounds in a second way.** The graph is symmetrized
before the filtration is built, so β₁ counts *undirected* cycles, not circular
dependencies. A fan-in diamond (`a→b`, `a→c`, `b→c`) registers as one cycle
despite being a perfectly acyclic dependency structure; and adding a genuine
back-edge `c→a` on top of an existing `a→c` link changes **nothing**, because it
lands on an edge that is already there. Both behaviours are pinned in
`betti_1_counts_undirected_tangle_not_dependency_cycles`. If you want circular
dependencies, this is the wrong instrument — use `ix-graph`'s SCC machinery.

**What is genuinely beyond McCabe is the H0 persistence.** Two graphs with
identical `V`, `E` and β₁ — identical cyclomatic numbers — separate cleanly on
the *filtration value at which components merge*. That value is the inverse
coupling strength: two files joined by one call site merge at distance `1/1 = 1.0`;
two joined by ten call sites merge at `1/10 = 0.1`. A crate's H0 diagram is
therefore its minimum spanning tree in coupling space, and `total_persistence` is
that tree's total weight — how loosely the crate is held together, not how many
branches it contains. This is pinned in
`h0_persistence_separates_graphs_with_identical_cyclomatic_numbers`.

Because the smallest possible weight is one call site, **`max_persistence`
saturates at 1.0**. A crate reporting exactly 1.000 has at least one module that
joins the rest through a single call. That turns out to be very common (§4).

## 4. Measured results

`units` = source files; `undEdg` = distinct undirected module edges; `b0`/`b1` =
β₀/β₁; `maxPers`/`totPers` = maximum and total H0 persistence; `resolved`,
`external`, `amb` = call-site resolution accounting.

| crate | units | undEdg | b0 | b1 | maxPers | totPers | resolved | external | amb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ga-chatbot` | 8 | 12 | 1 | 5 | 1.000 | 2.001 | 229 | 1293 | 2 |
| `ix-acoustic-tune` | 8 | 8 | 2 | 2 | 1.000 | 3.676 | 117 | 634 | 30 |
| `ix-adversarial` | 6 | 0 | 6 | 0 | 0.000 | 0.000 | 52 | 258 | 0 |
| `ix-agent` | 30 | 32 | 11 | 13 | 1.000 | 8.175 | 599 | 4496 | 328 |
| `ix-agent-core` | 8 | 7 | 3 | 2 | 0.500 | 1.458 | 75 | 213 | 66 |
| `ix-ai-annotations` | 8 | 14 | 1 | 7 | 1.000 | 3.017 | 95 | 339 | 0 |
| `ix-approval` | 4 | 2 | 2 | 0 | 1.000 | 1.125 | 46 | 47 | 0 |
| `ix-assumption-graph` | 12 | 28 | 2 | 18 | 1.000 | 2.457 | 249 | 467 | 6 |
| `ix-autograd` | 12 | 17 | 5 | 10 | 0.143 | 0.684 | 159 | 310 | 6 |
| `ix-autoresearch` | 11 | 17 | 2 | 8 | 0.333 | 1.886 | 225 | 897 | 98 |
| `ix-baml` | 2 | 1 | 1 | 0 | 0.167 | 0.167 | 47 | 141 | 27 |
| `ix-blast-radius` | 2 | 1 | 1 | 0 | 1.000 | 1.000 | 13 | 99 | 0 |
| `ix-bracelet` | 11 | 22 | 2 | 13 | 0.250 | 1.031 | 357 | 229 | 255 |
| `ix-cache` | 6 | 7 | 2 | 3 | 0.200 | 0.366 | 311 | 263 | 55 |
| `ix-catalog-core` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 7 | 16 | 0 |
| `ix-category` | 4 | 0 | 4 | 0 | 0.000 | 0.000 | 50 | 86 | 18 |
| `ix-chaos` | 10 | 5 | 5 | 0 | 1.000 | 2.736 | 94 | 300 | 0 |
| `ix-code` | 12 | 23 | 2 | 13 | 0.500 | 1.486 | 446 | 1101 | 3 |
| `ix-context` | 7 | 11 | 2 | 6 | 0.333 | 0.619 | 255 | 766 | 107 |
| `ix-dashboard` | 5 | 4 | 2 | 1 | 0.500 | 0.664 | 50 | 189 | 0 |
| `ix-demo` | 28 | 27 | 3 | 2 | 1.000 | 4.639 | 453 | 2804 | 123 |
| `ix-duck` | 28 | 80 | 1 | 53 | 1.000 | 6.657 | 801 | 3135 | 226 |
| `ix-duck-ext` | 2 | 0 | 2 | 0 | 0.000 | 0.000 | 0 | 79 | 0 |
| `ix-dynamics` | 5 | 1 | 4 | 0 | 0.500 | 0.500 | 109 | 256 | 31 |
| `ix-embedding-diagnostics` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 22 | 225 | 0 |
| `ix-ensemble` | 4 | 0 | 4 | 0 | 0.000 | 0.000 | 8 | 134 | 44 |
| `ix-evolution` | 7 | 4 | 4 | 1 | 1.000 | 1.278 | 30 | 283 | 47 |
| `ix-fractal` | 6 | 0 | 6 | 0 | 0.000 | 0.000 | 73 | 128 | 16 |
| `ix-fuzzy` | 7 | 5 | 3 | 1 | 0.111 | 0.287 | 210 | 308 | 32 |
| `ix-game` | 8 | 0 | 8 | 0 | 0.000 | 0.000 | 100 | 324 | 18 |
| `ix-governance` | 12 | 3 | 9 | 0 | 1.000 | 2.071 | 249 | 856 | 88 |
| `ix-gpu` | 12 | 11 | 3 | 2 | 0.333 | 1.633 | 189 | 526 | 0 |
| `ix-grammar` | 7 | 4 | 3 | 0 | 1.000 | 2.343 | 76 | 477 | 113 |
| `ix-graph` | 6 | 1 | 5 | 0 | 0.500 | 0.500 | 133 | 416 | 66 |
| `ix-harness-cargo` | 2 | 1 | 1 | 0 | 0.500 | 0.500 | 38 | 146 | 0 |
| `ix-harness-clippy` | 2 | 1 | 1 | 0 | 1.000 | 1.000 | 42 | 160 | 0 |
| `ix-harness-ga` | 2 | 1 | 1 | 0 | 1.000 | 1.000 | 15 | 75 | 0 |
| `ix-harness-github-actions` | 2 | 1 | 1 | 0 | 1.000 | 1.000 | 40 | 145 | 0 |
| `ix-harness-signing` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 52 | 92 | 0 |
| `ix-harness-tars` | 2 | 1 | 1 | 0 | 1.000 | 1.000 | 52 | 167 | 0 |
| `ix-invariant-coverage` | 7 | 7 | 2 | 2 | 1.000 | 3.833 | 79 | 323 | 36 |
| `ix-io` | 11 | 11 | 3 | 3 | 1.000 | 3.072 | 119 | 406 | 34 |
| `ix-ixql` | 8 | 11 | 2 | 5 | 0.500 | 1.477 | 322 | 595 | 94 |
| `ix-ktheory` | 4 | 0 | 4 | 0 | 0.000 | 0.000 | 23 | 92 | 0 |
| `ix-loop-detect` | 2 | 1 | 1 | 0 | 0.200 | 0.200 | 59 | 64 | 15 |
| `ix-manifold` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 61 | 143 | 0 |
| `ix-math` | 22 | 26 | 6 | 10 | 0.500 | 3.940 | 574 | 1121 | 228 |
| `ix-memory` | 5 | 0 | 5 | 0 | 0.000 | 0.000 | 113 | 177 | 34 |
| `ix-net` | 2 | 0 | 2 | 0 | 0.000 | 0.000 | 20 | 92 | 0 |
| `ix-nn` | 12 | 14 | 4 | 6 | 1.000 | 2.196 | 278 | 1172 | 221 |
| `ix-number-theory` | 8 | 6 | 3 | 1 | 0.500 | 1.433 | 78 | 62 | 0 |
| `ix-optick` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 65 | 215 | 0 |
| `ix-optick-invariants` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 10 | 132 | 0 |
| `ix-optick-sae` | 3 | 2 | 1 | 0 | 0.500 | 1.000 | 27 | 176 | 0 |
| `ix-optimize` | 6 | 3 | 3 | 0 | 0.500 | 1.167 | 20 | 62 | 17 |
| `ix-pipeline` | 8 | 8 | 2 | 2 | 1.000 | 3.267 | 231 | 780 | 136 |
| `ix-probabilistic` | 5 | 1 | 4 | 0 | 1.000 | 1.000 | 22 | 91 | 68 |
| `ix-quality-trend` | 11 | 26 | 2 | 17 | 1.000 | 2.484 | 380 | 1156 | 86 |
| `ix-quality-validate` | 2 | 1 | 1 | 0 | 0.500 | 0.500 | 7 | 43 | 0 |
| `ix-registrar` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 16 | 56 | 0 |
| `ix-registry` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 7 | 31 | 0 |
| `ix-registry-check` | 2 | 1 | 1 | 0 | 0.333 | 0.333 | 6 | 85 | 0 |
| `ix-rl` | 5 | 1 | 4 | 0 | 0.250 | 0.250 | 24 | 93 | 12 |
| `ix-rotation` | 8 | 6 | 3 | 1 | 0.200 | 0.541 | 125 | 94 | 86 |
| `ix-router-spike` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 40 | 207 | 0 |
| `ix-sanitize` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 21 | 31 | 0 |
| `ix-search` | 8 | 7 | 2 | 1 | 0.250 | 0.698 | 172 | 335 | 105 |
| `ix-sedenion` | 5 | 0 | 5 | 0 | 0.000 | 0.000 | 77 | 198 | 106 |
| `ix-sentinel` | 3 | 2 | 1 | 0 | 1.000 | 2.000 | 20 | 215 | 0 |
| `ix-sentrux-annotations` | 8 | 9 | 2 | 3 | 1.000 | 3.833 | 87 | 482 | 0 |
| `ix-session` | 4 | 1 | 3 | 0 | 0.100 | 0.100 | 19 | 58 | 1 |
| `ix-signal` | 12 | 5 | 7 | 0 | 1.000 | 2.309 | 105 | 384 | 11 |
| `ix-skill` | 19 | 41 | 3 | 25 | 1.000 | 4.577 | 406 | 1663 | 126 |
| `ix-skill-macros` | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 1 | 85 | 0 |
| `ix-streeling` | 7 | 4 | 3 | 0 | 1.000 | 2.500 | 36 | 261 | 0 |
| `ix-supervised` | 13 | 4 | 9 | 0 | 1.000 | 2.833 | 128 | 655 | 121 |
| `ix-topo` | 5 | 3 | 3 | 1 | 0.167 | 0.191 | 190 | 162 | 42 |
| `ix-types` | 2 | 1 | 1 | 0 | 0.053 | 0.053 | 53 | 91 | 0 |
| `ix-unsupervised` | 11 | 0 | 11 | 0 | 0.000 | 0.000 | 34 | 658 | 110 |
| `ix-value` | 5 | 2 | 3 | 0 | 0.500 | 0.667 | 14 | 69 | 0 |
| `ix-voicings` | 6 | 6 | 1 | 1 | 0.333 | 1.079 | 150 | 1209 | 0 |
| `memristive-markov` | 13 | 8 | 8 | 3 | 0.500 | 1.218 | 79 | 285 | 122 |

### 4.1 Aggregate shape

* 82 crates, **571 source files, 570 undirected module edges** — the workspace's
  intra-crate dependency graph is, on average, exactly tree-density. That is not
  a coincidence of design so much as a consequence of most crates being small.
* 71 crates have more than one source file. **33 of those carry at least one
  undirected cycle**; the other 38 are trees or forests.
* **22 crates have zero internal edges at all.** Nine of them have more than
  three files: `ix-unsupervised` (11 files), `ix-game` (8), `ix-adversarial` (6),
  `ix-fractal` (6), `ix-memory` (5), `ix-sedenion` (5), `ix-category` (4),
  `ix-ensemble` (4), `ix-ktheory` (4). These are *algorithm collections* — a bag
  of mutually independent modules sharing a crate boundary and nothing else.
  β₀ = file count is the signature, and it is arguably the single most useful
  number this measure produces: it says "this crate is a namespace, not a
  design".

### 4.2 The tangle ranking

| crate | units | undEdg | β₀ | β₁ |
| --- | ---: | ---: | ---: | ---: |
| `ix-duck` | 28 | 80 | 1 | **53** |
| `ix-skill` | 19 | 41 | 3 | 25 |
| `ix-assumption-graph` | 12 | 28 | 2 | 18 |
| `ix-quality-trend` | 11 | 26 | 2 | 17 |
| `ix-code` | 12 | 23 | 2 | 13 |
| `ix-bracelet` | 11 | 22 | 2 | 13 |
| `ix-agent` | 30 | 32 | 11 | 13 |

`ix-duck` is the outlier by a wide margin: 28 files, one component, 80 edges
where a tree would need 27. Its module graph carries **twice as many
cycle-forming edges as it has files**. Read together with §3, that is a
statement about mutual reachability, not about circular dependencies — but 53
independent undirected cycles across 28 files means essentially every module can
reach essentially every other by several distinct routes, which is the
structural signature of a crate whose modules share a large body of common
helpers.

`ix-agent` is the interesting contrast: it has more files than `ix-duck` (30 vs
28) and comparable β₁ (13), but β₀ = 11. It is not one tangled object; it is a
tangled core plus ten islands. Its `totPers` of 8.175 is the highest in the
workspace — the loosest spanning structure measured — which is what you would
expect from a crate that is mostly an aggregation point.

### 4.3 The persistence dimension

Mean merge distance (`totPers / (units − β₀)`) is the average inverse coupling of
the spanning tree — the statistic McCabe has no analogue for.

| Tightest coupling | mean merge | | Loosest coupling | mean merge |
| --- | ---: | --- | --- | ---: |
| `ix-types` | 0.053 | | `ix-invariant-coverage` | 0.767 |
| `ix-fuzzy` | 0.072 | | `ix-supervised` | 0.708 |
| `ix-cache` | 0.092 | | `ix-governance` | 0.690 |
| `ix-topo` | 0.095 | | `ix-sentrux-annotations` | 0.639 |
| `ix-autograd` | 0.098 | | `ix-streeling` | 0.625 |

A mean near 0.05–0.1 means modules are joined by ten to twenty call sites each:
the crate is one machine. A mean near 0.7 means modules are joined by roughly
one or two calls each: the crate is a set of parts that happen to be adjacent.
`ix-types` and `ix-supervised` have similar file counts and both have β₁ = 0 —
cyclomatic complexity rates them identically — and they sit at opposite ends of
this column.

## 5. What these numbers do *not* establish

Stated plainly, because the matrix rates F1's testability `medium` for exactly
this reason.

1. **Only 20.4% of call sites resolve inside their own crate.** Across the
   corpus: 10,466 resolved, 37,189 external, 3,616 ambiguous. The external
   majority is expected and correct — those are `std`, dependencies, trait
   methods — but it means the module graph explains a fifth of the call traffic,
   not all of it. The 7.1% ambiguous slice is the one that biases the result
   *downward*: those are real intra-crate calls dropped because a bare name is
   defined in two files. `ix-agent` (328), `ix-bracelet` (255), `ix-math` (228)
   and `ix-duck` (226) are the worst affected, so their true edge counts are
   higher than reported. **`ix-duck`'s 53 is a lower bound.**
2. **Resolution is by bare name, not by path.** `extract_definitions` collects
   `function_item` names; `mod` paths, `impl` blocks and trait dispatch are not
   modelled. Two unrelated `new()` functions in different files are ambiguous
   and dropped rather than merged, which is the safe direction, but it is still
   an approximation of the real dependency graph.
3. **Only files at the root of `src/` are units.** The walker is recursive, but
   a crate with `src/foo/bar.rs` gets a unit named `foo/bar.rs`, which is right;
   what it does *not* do is aggregate to `mod` boundaries. For crates with deep
   module trees, "file" and "module" diverge.
4. **No outcome variable.** Nothing here shows that a high β₁ or a low mean merge
   distance predicts defects, churn, or review difficulty. That would need
   labelled outcomes; the honest baseline is `ga/state/quality/`, and it has not
   been attempted. Everything in §4 is *description of structure*, not evidence
   of consequence. Treating these numbers as a quality gate today would be
   Goodharting an unvalidated metric.
5. **Rust only.** `extract_call_graph`/`extract_definitions` use the Rust
   tree-sitter grammar. The C#/TypeScript/F# grammars ix-code carries are not
   wired into this path.
6. **`MAX_NODES = 300`.** No IX crate comes close (largest is `ix-agent` at 30
   root files), so the cap was never exercised in this measurement. It would bite
   on a monorepo-scale walk.

## 6. One bug found by running it

Running the module walk over the real corpus panicked immediately:

```
byte index 60 is not a char boundary; it is inside '→' (bytes 58..61)
```

`ix_code::semantic::classify_call_target` truncated long method-call receivers
with `&s[..60]`, a byte slice that lands mid-character whenever the receiver
contains a multi-byte character. IX's own sources contain plenty — the failing
receiver came from a `crates/ix-grammar` doc example containing `→`. Any caller
passing such a file to `extract_call_graph` or `extract_semantic_metrics` would
panic, and that is a path `ix_ast_query` and `ix-context`'s indexer already use.

Fixed to truncate on chars, with the failing receiver pinned verbatim as a
regression test. Worth noting as a data point about the exposure gap itself: the
crash was in `semantic`, which *is* CI-compiled — it simply had never been run
over enough real source to hit it.

## 7. B1b: persistent homology over a DuckDB column

Row B1b is `implemented_needs_exposure`, `prio P2`, with the note *"a persistence
diagram is not a scalar — needs a `LIST(STRUCT(dim,birth,death))` return, or a
pairwise `ix_bottleneck(a,b)` scalar form. The scalar form is the cheap version."*

Delivered as **a SQL macro, not a UDF**, and the reason is recorded in
`crates/ix-duck/sql/pareto_frontier.sql`: ix-duck's `duck`/`udf` features are
never compiled by `cargo build --workspace` or by CI, so logic placed in a UDF is
invisible to every CI job. `crates/ix-duck/sql/persistence_h0.sql` runs on the
stock `duckdb` CLI with no build step.

**Scope: H0 only, and that is a property of the input rather than a shortcut.**
One DuckDB column is a point cloud in ℝ¹. A Rips complex over ℝ¹ capped at
dimension 1 does report H1 classes, but every one of them is an artefact of the
cap — there are no triangles in the filtration to fill the loops the complete
edge set creates. H1 and above need ≥ 2 coordinates, which is a different input
shape and needs the general engine, not a window function. The `ix_bottleneck`
scalar form the matrix suggests is **not** delivered: `ix_topo::bottleneck_distance`
is documented in its own source as *"Simple approximation: sort by persistence
and match greedily"*, so a SQL surface over it would pin an approximation as a
contract. That is deferred rather than shipped.

The mathematics is closed-form: over ℝ¹ the H0 deaths of the Rips filtration are
the minimum spanning tree's edge weights, and the MST of a 1-D point set is the
path through the sorted points. So the finite H0 pairs are `(0, gap)` for each
consecutive gap of the sorted **distinct** values, plus one essential `(0, ∞)`
class — a `lag()` window, `O(n log n)`, against the engine's boundary-matrix
reduction.

The argument above is *why* the two should agree; what proves it is
`crates/ix-duck/tests/persistence_h0_golden.rs`, which follows the two-surface
shape ix#294 established. The general engine (`rips_complex` +
`compute_persistence`, which knows nothing about the input being
one-dimensional) and the SQL closed form are pinned to the same frozen golden,
`tests/fixtures/persistence/golden-h0.csv`. Neither is the source of truth for
the other. They share no code and no algorithm, which is what makes the
agreement mean anything.

Verified by mutation, not assumed: perturbing one digit of the golden fails
three tests; changing the SQL's `gap` to `gap/2` fails the DuckDB CLI test.

**The honest limit on B1b's coverage:** the CLI test is *opportunistic*. It runs
only where a `duckdb` binary is on PATH and returns without asserting otherwise,
exactly like the Pareto one. On a CI machine without duckdb, a SQL-only
regression would pass silently — the engine half is what CI actually guards.
Locally, with duckdb v1.5.3 present, both halves ran and agreed.

## 8. Suggested matrix amendments

| Row | Current | Should read |
| --- | --- | --- |
| F1 | `implemented_needs_exposure` · exposure `duckdb: partial`, `skill: no`, `pipeline: candidate` | Still `implemented_needs_exposure` **before** this PR, but the note should say the composition *already existed* in `ix-code/src/topology.rs` and was invisible to CI because no crate enabled the `topology` feature. After this PR: MCP `ix_code_topology`, CI-compiled. |
| B1b | `implemented_needs_exposure` · `duckdb: no` | `duckdb: yes (SQL macro)` for H0. The `ix_bottleneck` scalar form remains undelivered and should be re-scoped: `ix_topo::bottleneck_distance` is a greedy approximation, so exposing it would freeze an approximation as a contract. |

The matrix's own §"exposure columns" convention counts `duckdb` as *"a UDF
registered in `ix-duck`/`ix-duck-ext`"*. B1b is delivered as a SQL macro, which
that definition does not cover — and given what `pareto_frontier.sql` records
about UDF invisibility, the convention is the thing that should change, not the
implementation.

## 9. What would make F1 worth more than description

In dependency order, none of it attempted here:

1. **Path-aware resolution.** Use `mod` paths and `impl` receivers so the
   ambiguous 7.1% resolves. This is the cheapest change with the largest effect
   on fidelity, and it moves `ix-duck`'s 53 off its lower bound.
2. **An outcome variable.** Join the per-crate topology against churn
   (`ix_git_churn`) or the quality snapshots in `ga/state/quality/` and ask
   whether mean merge distance carries signal that SLOC and McCabe do not. Until
   that is answered, §4 is a description and nothing more.
3. **Triangles in the filtration.** Only worth doing *after* (2) shows the
   structure predicts something. It would make β₁ mean more than the circuit
   rank, at O(n³) — and the honest prior is that H0 persistence is the part
   carrying the information.
