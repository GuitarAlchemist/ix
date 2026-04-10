---
date: 2026-04-09
topic: tars-v1-inspirations
source: https://github.com/GuitarAlchemist/tars (v1 directory, especially HyperComplexGeometricDSL.fs and TarsSedenionPartitioner.fs)
---

# TARS v1 Inspirations for ix

## Context

TARS v2 is undergoing a pragmatic re-architecture that explicitly **defers** the
exotic math components (sedenions, hyperbolic geometry, Hurwitz quaternions,
CUDA kernels) to v3+. Per `v2/docs/4_Research/V1_Insights/v1_component_reusability_analysis.md`:

> "Defer to v3+: Hyperbolic embeddings, Sedenions & exotic math DSLs"
>
> "Advanced Mathematics -- DEFER -- Explicitly v3+ per v2 docs"

That is exactly ix's territory. ix already has these pieces as first-class
crates (ix-sedenion, ix-math::poincare_hierarchy, ix-math::bsp, ix-rotation,
ix-gpu). The gap is in unified APIs that combine them.

## High-Value Picks (very IX-spirited)

### 1. Unified `GeometricSpace` enum + `distance()` function

**Source:** `v1/src/TarsEngine.FSharp.Core/DSL/HyperComplexGeometricDSL.fs`

TARS v1 has a single `GeometricSpace` discriminated union wrapping 10 distance
metrics under one API:

```fsharp
type GeometricSpace =
    | Euclidean
    | Hyperbolic of curvature: float32
    | Spherical of radius: float32
    | Minkowski of signature: int * int * int * int
    | Mahalanobis
    | Wasserstein
    | Manhattan
    | Chebyshev
    | Hamming
    | Jaccard

let distance (space: GeometricSpace) (a: float32 array) (b: float32 array) : float32
```

**ix gap:** ix-math has `distance.rs` but it is a bag of functions, not a unified
enum. ix-math::hyperbolic has Poincare distance, but it is unrelated to the
Euclidean helpers. ix-math::poincare_hierarchy uses hyperbolic distance but
cannot easily swap spaces.

**Port plan:** Add `crates/ix-math/src/geometric_space.rs` with:
- `GeometricSpace` enum with parameterized variants
- `distance(space, a, b)` function dispatching over the enum
- Reuse existing `ix_math::hyperbolic::poincare_distance` for the hyperbolic case
- Add missing metrics: Mahalanobis, Wasserstein (1D), Chebyshev, Hamming, Jaccard
- Builder pattern with seedable parameters

**Why IX-spirited:**
- Pure math, no external deps
- Small (~200 LOC)
- Composable with existing crates (ix-unsupervised KMeans can take a space arg)
- Foundational for downstream work

### 2. Sedenion `exp` and `log` via scalar+vector decomposition

**Source:** `v1/src/TarsEngine.FSharp.Core/DSL/HyperComplexGeometricDSL.fs::SedenionOps`

TARS implements sedenion exp/log using the classic quaternion trick extended
to 16D: decompose into scalar component (e_0) and 15-component "vector" part,
then use `exp(s+v) = e^s * (cos|v| + (v/|v|) sin|v|)`.

```fsharp
let exp (s: Sedenion) : Sedenion =
    let scalar = s.Components.[0]
    let vector = s.Components.[1..15]
    let vectorNorm = vector |> Array.sumBy (fun x -> x * x) |> sqrt
    if vectorNorm < 1e-6f then
        { Components = [| exp scalar; 0.0f; ... |] }
    else
        let expScalar = exp scalar
        let cosVN = cos vectorNorm
        let sinVN = sin vectorNorm
        let factor = expScalar * sinVN / vectorNorm
        // Components = [expScalar * cosVN, vector .* factor]
```

**ix gap:** ix-sedenion has add/sub/mul/conjugate but not exp/log. These are
non-trivial because sedenions are non-associative and have zero divisors --
the scalar+vector decomposition sidesteps those issues when the vector part
has small norm.

**Port plan:** Add `exp` and `log` to `crates/ix-sedenion/src/sedenion.rs`.
Same algorithm, Rust idioms. Add property tests: `log(exp(s)) ≈ s` for small
`|vector|`, `exp(0) = 1`, `exp(s).norm() > 0`.

**Why IX-spirited:**
- Pure math extension to existing crate
- ~50 LOC
- Enables: sedenion-based Lie group operations, exponential map for
  hypercomplex gradient descent

### 3. 16D Sedenion BSP Partitioner with hyperplane normals

**Source:** `v1/src/TarsEngine.FSharp.Core/TarsSedenionPartitioner.fs`

TARS builds BSP trees where split planes are sedenion-valued normals, not
axis-aligned. Each hyperplane has a `Significance` score, enabling
importance-weighted partitioning.

```fsharp
type Hyperplane = {
    Normal: Sedenion
    Distance: float
    Significance: float
}

type BspNode = {
    Hyperplane: Hyperplane option
    LeftChild: BspNode option
    RightChild: BspNode option
    Points: Sedenion list
    Significance: float
}
```

**ix gap:** ix-math::bsp has `BspTree<D>` with **axis-aligned** splits (cycling
through dimensions). ix-sedenion/src/bsp.rs has arbitrary-axis splits but not
hyperplane-normal splits in sedenion space.

**Port plan:** Add `crates/ix-sedenion/src/hyper_bsp.rs` with:
- `Hyperplane` struct (sedenion normal, offset, significance)
- `HyperBspNode` with arbitrary-orientation splits
- `build(points, max_depth)` -- at each node, pick the hyperplane that best
  separates points (PCA on residual, or variance-maximizing direction)
- `nearest_neighbor(query)` with hyperplane-aware branch pruning

**Why IX-spirited:**
- Combines two existing ix crates (bsp + sedenion) into something neither has alone
- Opens door to clustering in 16D hypercomplex space
- ~300 LOC

## Medium-Value Picks (worth considering later)

### 4. `TrsxHypergraph` -- hypergraph with semantic diffs

TARS tracks file versions as a hypergraph where edges connect multiple nodes
(not just pairs) and carry semantic vectors + quaternion embeddings. ix-graph
has only regular graphs. A hypergraph module would enable richer call-graph
analysis in the Code Observatory (a function that calls 5 others is one
hyperedge, not 5 separate edges).

**Effort:** ~500 LOC, new module in ix-graph
**Risk:** Hypergraphs have many API design choices (incidence matrix, edge
lists, bipartite representation); pick carefully.

### 5. AgenticTraceCapture -- structured event logging

TARS has a 1230-line structured event logger with types for: AgentEvent,
InterAgentCommunication, GrammarEvolutionEvent, TarsArchitectureSnapshot,
WebRequest, TripleStoreQuery, VectorStoreOperation, LLMAPICall.

**Effort:** ~300 LOC, would live in ix-agent or a new ix-trace crate
**Risk:** Overlaps with existing ix observability; check first.

## Deferred / Not a Fit

- **Hurwitz Quaternions** -- ix-rotation has quaternions; Hurwitz are integer
  lattice quaternions, useful for number theory but a narrower use case.
- **CUDA kernels** -- ix-gpu uses WGPU, which is cleaner cross-platform.
- **Metascript executor** -- ix-pipeline covers this.
- **FLUX multi-language DSL** -- too F#-specific, not IX-spirited.
- **FractalGrammar** -- ix-grammar already has CFG/Earley/CYK; fractal grammars
  are niche.

## Implementation Order

1. **GeometricSpace enum** -- foundational, ~200 LOC, one commit
2. **Sedenion exp/log** -- small extension, ~50 LOC, one commit
3. **HyperBspNode** -- combines existing crates, ~300 LOC, one commit
4. Deferred: TrsxHypergraph, AgenticTraceCapture -- fresh session

Each commit should build, test, and clippy-clean independently.
