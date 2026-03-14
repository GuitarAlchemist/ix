---
date: 2026-03-14
topic: machineouf-full-vision
---

# MachinDeOuf Full Vision: Algorithms + Demo + MCP + Infrastructure

## What We're Building

A comprehensive expansion of the MachinDeOuf workspace across four parallel workstreams:

1. **New Math Crates** — All 9 TARS math domains as pure-Rust crates
2. **Demo App Evolution** — More tabs + richer interactions + better UX + potential web version
3. **MCP & Skill Integration** — Wire algorithms into Claude Code as callable tools
4. **Infrastructure Hardening** — CI, tests, README, stub completion

The goal is a polished, deeply integrated workspace where every algorithm is implemented, tested, demoed, and exposed as a Claude Code skill.

## Current State Baseline

| Metric | Count | Notes |
|--------|-------|-------|
| Crates | 27 | CLAUDE.md lists 22; 5 newer (grammar, dynamics, ktheory, topo, category) |
| MCP tools | 17 | In machin-agent (stats, distance, kmeans, fft, lyapunov, fgsm, grammar, cache, etc.) |
| Claude Code skills | 10 | In .claude/skills/machin-*/ |
| Demo tabs | 15 | egui app: stats through transformer |
| TODO/stubs | 10 | t-SNE, GMM, network.rs, RL env, CLI data loading |
| CI pipelines | 0 | No GitHub Actions or other CI |
| License | MIT | Already in workspace Cargo.toml |

### Existing Code That Overlaps New Crate Proposals

**These implementations already exist and need extraction/migration, not rewriting:**

| Existing file | Domain | Proposed crate | Action needed |
|--------------|--------|----------------|---------------|
| `machin-gpu/src/quaternion.rs` | Batch quaternion rotation + WGSL shader | machin-rotation | Extract CPU math to machin-rotation; machin-gpu keeps GPU kernel, depends on machin-rotation |
| `machin-gpu/src/sedenion.rs` | Batch sedenion multiply + Cayley-Dickson | machin-sedenion | Extract algebra to machin-sedenion; machin-gpu keeps GPU kernel |
| `machin-chaos/src/takagi.rs` | Takagi/Blancmange curve | machin-fractal | Migrate to machin-fractal, re-export from machin-chaos for compat |
| `machin-chaos/src/de_rham.rs` | de Rham fractal interpolation | machin-fractal | Migrate to machin-fractal |
| `machin-chaos/src/fractal.rs` | Box-counting, correlation dimension | machin-fractal | Keep in machin-chaos (these are chaos measures, not curve generators) |

## Workstream 1: New Math Crates (9 TARS Domains)

### Already Implemented (deepen these)
- **machin-topo** — Persistent homology, simplicial complexes. Expand: filtrations, Betti numbers visualization, Vietoris-Rips from point clouds.
- **machin-dynamics** — IK chains, Lie groups/algebras, neural ODEs. Expand: Neural PDE solvers, more Lie group instances (SO(3), SE(3)), symplectic integrators.
- **machin-ktheory** — Graph K-theory, Mayer-Vietoris. Expand: Grothendieck K0/K1 computation, spectral sequences.
- **machin-category** — Functors, natural transformations. Expand: Monads, adjunctions, Kan extensions, Free/Forgetful functor pairs.

### New Crates (Extract + Expand)

#### Phase 1: Geometric & Number Theory
- **machin-rotation** — Extract quaternion CPU math from machin-gpu. Add: dual quaternions, SLERP, Plucker coordinates, Euler angles, rotation matrices, axis-angle, rigid body transforms, screw theory. machin-gpu retains GPU kernel, depends on machin-rotation for types.
- **machin-number-theory** — Net-new. Prime sieve (Eratosthenes, Atkin), primality tests (Miller-Rabin), prime triplets/constellations, prime gaps, totient, modular arithmetic, Chinese Remainder Theorem, elliptic curves (note: elliptic curves may need a big-integer dependency).

#### Phase 2: Fractals & Higher Algebra
- **machin-fractal** — Extract Takagi + de Rham from machin-chaos. Add: IFS (iterated function systems), L-systems, space-filling curves (Hilbert, Peano). machin-chaos keeps box-counting/correlation dimension (chaos measures).
- **machin-sedenion** — Extract sedenion algebra from machin-gpu. Add: full Cayley-Dickson construction chain, BSP partitioning for spatial queries. machin-gpu retains GPU kernel.

### Crate Dependency Graph (new crates)
```
machin-math ← machin-rotation (linalg primitives)
machin-rotation ← machin-gpu (GPU kernels for batch rotation)
machin-math ← machin-number-theory (no external deps unless elliptic curves need bigint)
machin-chaos ← machin-fractal (share types, re-exports for compat)
machin-sedenion ← machin-gpu (GPU kernels for batch sedenion ops)
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
- Crates are pure Rust and WASM-compatible (except machin-gpu, machin-io, machin-cache)
- Consider Leptos/Yew only if eframe WASM proves insufficient

## Workstream 3: MCP & Skill Integration

### MCP Server (machin-agent)
- **Already has 17 tools** — stats, distance, kmeans, fft, lyapunov, fgsm, grammar (3 tools), cache, bloom, etc.
- **Action**: Register machin-agent in `.mcp.json` (currently only has context7)
- Ensure all tool schemas are complete with parameter descriptions and examples
- Test end-to-end: Claude Code invokes skill -> MCP -> machin-agent -> result

### Coverage Gap Analysis

| Crate | MCP tool? | Claude Code skill? | Demo tab? |
|-------|-----------|-------------------|-----------|
| machin-math | Yes (stats, distance) | No | Yes (Stats) |
| machin-optimize | Yes | Yes | Yes |
| machin-supervised | No | No | Yes (Regression) |
| machin-unsupervised | Yes (kmeans) | Yes (cluster) | Yes |
| machin-nn | No | No | Yes (Neural Net, Transformer) |
| machin-rl | No | No | Yes (RL Bandits) |
| machin-evolution | No | No | Yes |
| machin-graph | No | Yes (hmm) | No |
| machin-search | No | Yes | Yes |
| machin-signal | Yes (fft) | Yes | Yes |
| machin-chaos | Yes (lyapunov) | Yes | Yes |
| machin-game | No | Yes | Yes |
| machin-probabilistic | Yes (bloom) | No | Yes |
| machin-gpu | No | No | Yes |
| machin-cache | Yes (cache) | No | No |
| machin-adversarial | Yes (fgsm) | Yes | No |
| machin-grammar | Yes (3 tools) | No | No |
| machin-pipeline | No | Yes | No |
| machin-ensemble | No | No | No |
| machin-dynamics | No | No | Yes (IK Chain) |
| machin-topo | No | No | No |
| machin-ktheory | No | No | No |
| machin-category | No | No | No |
| machin-io | No | No | No |
| machin-rotation | — | — | — |
| machin-number-theory | — | — | — |
| machin-fractal | — | — | — |
| machin-sedenion | — | — | — |

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
- `machin-unsupervised`: t-SNE, GMM
- `machin-nn/network.rs`: Composable network (TODO)
- `machin-rl/q_learning.rs`: Implement after Environment trait (TODO)
- `machin-rl/env.rs`: GridWorld environment (TODO)
- `machin-skill/src/main.rs:140,144`: Data loading for Train/Cluster commands (TODO)
- `machin-skill/src/main.rs:350-356`: Wire up decision-tree, SVM, DBSCAN, PCA (TODO)

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
| machin-math | machin-gpu (no WGPU in WASM yet) | machin-io (TCP, named pipes) |
| machin-optimize | | machin-cache (tokio, networking) |
| machin-supervised | | |
| machin-unsupervised | | |
| machin-nn | | |
| machin-rotation | | |
| machin-number-theory | | |
| machin-fractal | | |
| machin-sedenion | | |
| machin-chaos | | |
| machin-signal | | |
| machin-game | | |
| machin-search | | |
| machin-graph | | |
| machin-probabilistic | | |

## Key Decisions

1. **Fractal placement**: New machin-fractal crate. Extract Takagi/de Rham from machin-chaos. Keep box-counting in machin-chaos (it's a chaos measure).
2. **Demo architecture**: Keep single binary with tabs. It's the showcase.
3. **Web version approach**: Use eframe's built-in WASM support first (minimal effort).
4. **Phasing**: Infrastructure hardening runs in parallel with new crate development.
5. **Sedenion scope**: Start with sedenions + Cayley-Dickson, generalize later if useful.
6. **Crate naming**: `machin-rotation` (broad rotation math) over `machin-quaternion`.
7. **Number theory scope**: Full `machin-number-theory` (primes + modular arithmetic + elliptic curves).
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
| machin-gpu is a bottleneck (8+ kernel files) | Extraction complexity | Extract one crate at a time; keep GPU kernels in machin-gpu, move CPU algebra out |
| Breaking changes during extraction | Downstream crates break | All crates at 0.1.0, semver allows breaking changes. Add deprecation re-exports in old locations |
| WASM feature gates add complexity | Build matrix grows | Only gate machin-gpu and machin-io; most crates are naturally WASM-safe |

## Success Criteria

- All 9 TARS math domains implemented as crates with tests + proptest + criterion
- Demo app has 21+ tabs with category navigation
- Every crate has at least one MCP tool AND one Claude Code skill
- CI passes on every push (build + clippy + test + doc) on stable + nightly
- README accurately reflects the full workspace (27+ crates)
- Zero clippy warnings, zero TODO stubs remaining
- All crates structured for crates.io publishing (clean APIs, docs, semver)
- WASM build works for algorithm crates via eframe
