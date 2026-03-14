---
date: 2026-03-14
topic: ix-full-vision
---

# ix Full Vision: Algorithms + Demo + MCP + Infrastructure

## What We're Building

A comprehensive expansion of the ix workspace across four parallel workstreams:

1. **New Math Crates** — All 9 TARS math domains as pure-Rust crates
2. **Demo App Evolution** — More tabs + richer interactions + better UX + potential web version
3. **MCP & Skill Integration** — Wire algorithms into Claude Code as callable tools
4. **Infrastructure Hardening** — CI, tests, README, stub completion

The goal is a polished, deeply integrated workspace where every algorithm is implemented, tested, demoed, and exposed as a Claude Code skill.

## Current State Baseline

| Metric | Count | Notes |
|--------|-------|-------|
| Crates | 27 | CLAUDE.md lists 22; 5 newer (grammar, dynamics, ktheory, topo, category) |
| MCP tools | 17 | In ix-agent (stats, distance, kmeans, fft, lyapunov, fgsm, grammar, cache, etc.) |
| Claude Code skills | 10 | In .claude/skills/ix-*/ |
| Demo tabs | 15 | egui app: stats through transformer |
| TODO/stubs | 10 | t-SNE, GMM, network.rs, RL env, CLI data loading |
| CI pipelines | 0 | No GitHub Actions or other CI |
| License | MIT | Already in workspace Cargo.toml |

### Existing Code That Overlaps New Crate Proposals

**These implementations already exist and need extraction/migration, not rewriting:**

| Existing file | Domain | Proposed crate | Action needed |
|--------------|--------|----------------|---------------|
| `ix-gpu/src/quaternion.rs` | Batch quaternion rotation + WGSL shader | ix-rotation | Extract CPU math to ix-rotation; ix-gpu keeps GPU kernel, depends on ix-rotation |
| `ix-gpu/src/sedenion.rs` | Batch sedenion multiply + Cayley-Dickson | ix-sedenion | Extract algebra to ix-sedenion; ix-gpu keeps GPU kernel |
| `ix-chaos/src/takagi.rs` | Takagi/Blancmange curve | ix-fractal | Migrate to ix-fractal, re-export from ix-chaos for compat |
| `ix-chaos/src/de_rham.rs` | de Rham fractal interpolation | ix-fractal | Migrate to ix-fractal |
| `ix-chaos/src/fractal.rs` | Box-counting, correlation dimension | ix-fractal | Keep in ix-chaos (these are chaos measures, not curve generators) |

## Workstream 1: New Math Crates (9 TARS Domains)

### Already Implemented (deepen these)
- **ix-topo** — Persistent homology, simplicial complexes. Expand: filtrations, Betti numbers visualization, Vietoris-Rips from point clouds.
- **ix-dynamics** — IK chains, Lie groups/algebras, neural ODEs. Expand: Neural PDE solvers, more Lie group instances (SO(3), SE(3)), symplectic integrators.
- **ix-ktheory** — Graph K-theory, Mayer-Vietoris. Expand: Grothendieck K0/K1 computation, spectral sequences.
- **ix-category** — Functors, natural transformations. Expand: Monads, adjunctions, Kan extensions, Free/Forgetful functor pairs.

### New Crates (Extract + Expand)

#### Phase 1: Geometric & Number Theory
- **ix-rotation** — Extract quaternion CPU math from ix-gpu. Add: dual quaternions, SLERP, Plucker coordinates, Euler angles, rotation matrices, axis-angle, rigid body transforms, screw theory. ix-gpu retains GPU kernel, depends on ix-rotation for types.
- **ix-number-theory** — Net-new. Prime sieve (Eratosthenes, Atkin), primality tests (Miller-Rabin), prime triplets/constellations, prime gaps, totient, modular arithmetic, Chinese Remainder Theorem, elliptic curves (note: elliptic curves may need a big-integer dependency).

#### Phase 2: Fractals & Higher Algebra
- **ix-fractal** — Extract Takagi + de Rham from ix-chaos. Add: IFS (iterated function systems), L-systems, space-filling curves (Hilbert, Peano). ix-chaos keeps box-counting/correlation dimension (chaos measures).
- **ix-sedenion** — Extract sedenion algebra from ix-gpu. Add: full Cayley-Dickson construction chain, BSP partitioning for spatial queries. ix-gpu retains GPU kernel.

### Crate Dependency Graph (new crates)
```
ix-math ← ix-rotation (linalg primitives)
ix-rotation ← ix-gpu (GPU kernels for batch rotation)
ix-math ← ix-number-theory (no external deps unless elliptic curves need bigint)
ix-chaos ← ix-fractal (share types, re-exports for compat)
ix-sedenion ← ix-gpu (GPU kernels for batch sedenion ops)
```

### Phasing Strategy
- Phase 1 (rotation + number-theory) first — high visual impact for demo, useful for GPU kernels
- Phase 2 (fractal + sedenion) second — builds on chaos crate and GPU work
- Deepen existing crates (topo, dynamics, ktheory, category) continuously alongside new work

## Workstream 2: Demo App Evolution

### Current: 15 Tabs
Stats, Regression, Clustering, Neural Net, Optimization, Chaos, Signal, IK Chain, Evolution, RL Bandits, Search, Game Theory, Probabilistic, GPU Kernels, Transformer.

### New Tabs (target: 21+)
- Rotation tab: 3D rotation visualization, SLERP interpolation animation, dual quaternion screw motion, Euler angle gimbal lock demo
- Number Theory tab: Ulam spiral, prime gap distribution, sieve performance comparison, modular arithmetic explorer
- Fractal tab: Takagi curve parameter explorer, IFS fern/triangle, L-system renderer
- Sedenion tab: Higher-dimensional algebra visualization (projection to 2D/3D)
- Topology tab: Point cloud -> persistent homology barcode/persistence diagram
- Category tab: Functor diagram visualization, composition chains

### Richer Interactions
- **Cross-crate demos**: Pipeline that chains algorithms (e.g., signal FFT -> clustering -> visualization)
- **Parameter presets**: "Interesting configurations" buttons for each demo
- **Animation**: Time-stepping for dynamical systems, SLERP paths, evolutionary convergence
- **Side-by-side**: Compare algorithms (e.g., SGD vs Adam, A* vs Q*)
- **Export**: Save plots as PNG, export data as CSV

### UX Polish
- **Category navigation**: Group tabs into sections (Math, ML, Search, Infrastructure) with collapsible sidebar
- **Search/filter**: Find demos by keyword
- **Tooltips**: Explain parameters on hover
- **Theme**: Dark/light mode toggle
- **Status bar**: Show computation time, memory usage

### Web Version (Future)
- Use eframe's built-in WASM support first (minimal effort) — egui already compiles to WASM
- Crates are pure Rust and WASM-compatible (except ix-gpu, ix-io, ix-cache)
- Consider Leptos/Yew only if eframe WASM proves insufficient

## Workstream 3: MCP & Skill Integration

### MCP Server (ix-agent)
- **Already has 17 tools** — stats, distance, kmeans, fft, lyapunov, fgsm, grammar (3 tools), cache, bloom, etc.
- **Action**: Register ix-agent in `.mcp.json` (currently only has context7)
- Ensure all tool schemas are complete with parameter descriptions and examples
- Test end-to-end: Claude Code invokes skill -> MCP -> ix-agent -> result

### Coverage Gap Analysis

| Crate | MCP tool? | Claude Code skill? | Demo tab? |
|-------|-----------|-------------------|-----------|
| ix-math | Yes (stats, distance) | No | Yes (Stats) |
| ix-optimize | Yes | Yes | Yes |
| ix-supervised | No | No | Yes (Regression) |
| ix-unsupervised | Yes (kmeans) | Yes (cluster) | Yes |
| ix-nn | No | No | Yes (Neural Net, Transformer) |
| ix-rl | No | No | Yes (RL Bandits) |
| ix-evolution | No | No | Yes |
| ix-graph | No | Yes (hmm) | No |
| ix-search | No | Yes | Yes |
| ix-signal | Yes (fft) | Yes | Yes |
| ix-chaos | Yes (lyapunov) | Yes | Yes |
| ix-game | No | Yes | Yes |
| ix-probabilistic | Yes (bloom) | No | Yes |
| ix-gpu | No | No | Yes |
| ix-cache | Yes (cache) | No | No |
| ix-adversarial | Yes (fgsm) | Yes | No |
| ix-grammar | Yes (3 tools) | No | No |
| ix-pipeline | No | Yes | No |
| ix-ensemble | No | No | No |
| ix-dynamics | No | No | Yes (IK Chain) |
| ix-topo | No | No | No |
| ix-ktheory | No | No | No |
| ix-category | No | No | No |
| ix-io | No | No | No |
| ix-rotation | — | — | — |
| ix-number-theory | — | — | — |
| ix-fractal | — | — | — |
| ix-sedenion | — | — | — |

**Target**: Every crate has at least one MCP tool, one Claude Code skill, and one demo tab.

## Workstream 4: Infrastructure Hardening

### CI/CD (GitHub Actions)
- `cargo build --workspace` on every push
- `cargo clippy --workspace -- -D warnings`
- `cargo test --workspace`
- `cargo doc --workspace --no-deps` to catch doc-test failures
- Matrix: stable + nightly Rust, Linux + Windows
- Cache cargo registry + target dir for speed
- `criterion` benchmarks: run separately (nightly only, not blocking CI)
- MSRV: Declare in Cargo.toml, test in CI (likely Rust 1.80+ due to wgpu 28)

### Testing
- **Unit tests**: Every public function gets at least one `#[test]`. Many crates already have `#[cfg(test)]` modules.
- **Integration tests**: `tests/` directories for cross-crate scenarios
- **Doc tests**: Every public API example in `//!` and `///` must compile and run
- **Property tests**: `proptest` for math invariants (commutativity, associativity, norm preservation, rotation orthogonality)
- **Benchmarks**: `criterion` for performance-critical paths (FFT, matrix ops, GPU kernels)

### Stub Completion (10 items)
- `ix-unsupervised`: t-SNE, GMM
- `ix-nn/network.rs`: Composable network (TODO)
- `ix-rl/q_learning.rs`: Implement after Environment trait (TODO)
- `ix-rl/env.rs`: GridWorld environment (TODO)
- `ix-skill/src/main.rs:140,144`: Data loading for Train/Cluster commands (TODO)
- `ix-skill/src/main.rs:350-356`: Wire up decision-tree, SVM, DBSCAN, PCA (TODO)

### README & Docs
- Update to reflect all 27+ crates (currently documents 22)
- Add badges: CI status, crate count, Rust version, license
- Document MCP setup instructions (`.mcp.json` entry)
- Add architecture diagram showing crate dependencies
- Per-crate README with examples
- Documentation strategy: rustdoc (primary) + workspace-level mdbook (future, for tutorials)

### WASM Compatibility Matrix

| WASM-safe | Needs feature gate | Not WASM-compatible |
|-----------|--------------------|---------------------|
| ix-math | ix-gpu (no WGPU in WASM yet) | ix-io (TCP, named pipes) |
| ix-optimize | | ix-cache (tokio, networking) |
| ix-supervised | | |
| ix-unsupervised | | |
| ix-nn | | |
| ix-rotation | | |
| ix-number-theory | | |
| ix-fractal | | |
| ix-sedenion | | |
| ix-chaos | | |
| ix-signal | | |
| ix-game | | |
| ix-search | | |
| ix-graph | | |
| ix-probabilistic | | |

## Key Decisions

1. **Fractal placement**: New ix-fractal crate. Extract Takagi/de Rham from ix-chaos. Keep box-counting in ix-chaos (it's a chaos measure).
2. **Demo architecture**: Keep single binary with tabs. It's the showcase.
3. **Web version approach**: Use eframe's built-in WASM support first (minimal effort).
4. **Phasing**: Infrastructure hardening runs in parallel with new crate development.
5. **Sedenion scope**: Start with sedenions + Cayley-Dickson, generalize later if useful.
6. **Crate naming**: `ix-rotation` (broad rotation math) over `ix-quaternion`.
7. **Number theory scope**: Full `ix-number-theory` (primes + modular arithmetic + elliptic curves).
8. **GPU from day one**: Rotation and sedenion get WGPU kernels alongside CPU implementations.
9. **crates.io publishing**: Plan for it now. Clean APIs, semver, CI-driven publish on git tag.
10. **Test framework**: All three — `#[test]` + `proptest` + `criterion`.
11. **New crate strategy**: Extract + expand (not rewrite) for rotation, sedenion, fractal.
12. **License**: MIT (already set in workspace Cargo.toml).
13. **MSRV**: Declare and test in CI (likely 1.80+ due to wgpu 28).
14. **Criterion in CI**: Run separately on nightly, not blocking main CI pipeline.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU tests need hardware/software renderer in CI | CI failures on GPU crates | Use `wgpu` software adapter or skip GPU tests in CI with feature flag |
| Elliptic curves in number theory is a rabbit hole | Scope creep, delays Phase 1 | Start with primes + modular arithmetic; elliptic curves as a follow-up module |
| ix-gpu is a bottleneck (8+ kernel files) | Extraction complexity | Extract one crate at a time; keep GPU kernels in ix-gpu, move CPU algebra out |
| Breaking changes during extraction | Downstream crates break | All crates at 0.1.0, semver allows breaking changes. Add deprecation re-exports in old locations |
| WASM feature gates add complexity | Build matrix grows | Only gate ix-gpu and ix-io; most crates are naturally WASM-safe |

## Success Criteria

- All 9 TARS math domains implemented as crates with tests + proptest + criterion
- Demo app has 21+ tabs with category navigation
- Every crate has at least one MCP tool AND one Claude Code skill
- CI passes on every push (build + clippy + test + doc) on stable + nightly
- README accurately reflects the full workspace (27+ crates)
- Zero clippy warnings, zero TODO stubs remaining
- All crates structured for crates.io publishing (clean APIs, docs, semver)
- WASM build works for algorithm crates via eframe
