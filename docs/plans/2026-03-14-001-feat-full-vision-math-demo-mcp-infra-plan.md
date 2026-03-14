---
title: "feat: ix Full Vision — Math Crates + Demo + MCP + Infrastructure"
type: feat
status: completed
date: 2026-03-14
origin: docs/brainstorms/2026-03-14-ix-full-vision-brainstorm.md
---

# feat: ix Full Vision — Math Crates + Demo + MCP + Infrastructure

## Overview

Comprehensive expansion of the ix workspace across four parallel workstreams: (1) four new math crates via extract+expand from existing code, (2) six new demo tabs with category navigation, (3) full MCP/skill coverage for all crates, (4) GitHub Actions CI, proptest/criterion, stub completion, and crates.io prep. (see brainstorm: docs/brainstorms/2026-03-14-ix-full-vision-brainstorm.md)

## Problem Statement / Motivation

The workspace has 27 crates, 17 MCP tools, 10 Claude Code skills, and 15 demo tabs — but significant gaps remain:
- Quaternion/sedenion CPU algebra is buried inside ix-gpu (GPU crate)
- Takagi/de Rham fractal curves live in ix-chaos instead of a dedicated fractal crate
- No number theory crate exists
- 10+ crates have no MCP tool or Claude Code skill
- Zero CI/CD — no automated build, test, or clippy checks
- No property-based testing (proptest) or benchmarks (criterion)
- Not structured for crates.io publishing

## Proposed Solution

Four parallel workstreams executed in five implementation phases.

## Technical Approach

### Architecture

**New crate dependency graph:**
```
ix-math ← ix-rotation (linalg primitives for Euler, axis-angle, matrices)
ix-rotation ← ix-gpu (GPU kernels import rotation types)
ix-math ← ix-number-theory (standalone, no new external deps initially)
ix-chaos ← ix-fractal (share types, re-exports for backward compat)
ix-sedenion ← ix-gpu (GPU kernels import sedenion types)
```

**Crate template pattern (from ix-signal):**
```
crates/ix-{name}/
├── Cargo.toml          # workspace = true inheritance, minimal deps
├── benches/            # criterion benchmarks (new)
│   └── bench_{name}.rs
└── src/
    ├── lib.rs          # pub mod declarations
    ├── {module1}.rs    # self-contained module + inline #[cfg(test)]
    ├── {module2}.rs
    └── ...
```

### Implementation Phases

#### Phase 1: Infrastructure Foundation ✅

CI/CD, workspace-level testing framework, and crates.io prep.

**Files to create/modify:**

- [x] `.github/workflows/ci.yml` — GitHub Actions: build + clippy + test + doc on stable/nightly, Linux/Windows
- [x] `Cargo.toml` (workspace root) — Add `proptest`, `criterion` to `[workspace.dependencies]` as dev-dependencies
- [x] `CLAUDE.md` — Update crate list (27→31+), add testing conventions, add MSRV
- [x] `README.md` — Update to reflect all crates, add CI badge, document MCP setup
- [x] `.mcp.json` — Register ix-agent: `{ "command": "cargo", "args": ["run", "-p", "ix-agent"] }`

**Acceptance criteria:**
- [x] CI runs on every push/PR
- [x] Clippy passes with -D warnings across entire workspace
- [x] All existing tests pass in CI
- [x] ix-agent registered in .mcp.json and callable by Claude Code
- [x] README lists all crates with accurate descriptions

#### Phase 2: Extract + Create Math Crates ✅

Extract CPU algebra from ix-gpu, fractal curves from ix-chaos. Create ix-number-theory from scratch.

##### Phase 2a: ix-rotation (extract + expand)

- [x] `crates/ix-rotation/Cargo.toml` — depends on ix-math, ndarray, rand
- [x] `crates/ix-rotation/src/lib.rs` — pub mod declarations
- [x] `crates/ix-rotation/src/quaternion.rs` — Quaternion struct, mul, conjugate, inverse, normalize, from_axis_angle, to_matrix
- [x] `crates/ix-rotation/src/dual_quaternion.rs` — DualQuaternion struct, rigid body transform, screw motion
- [x] `crates/ix-rotation/src/slerp.rs` — Spherical linear interpolation, squad (spline)
- [x] `crates/ix-rotation/src/euler.rs` — Euler angles (XYZ, ZYX, etc.), gimbal lock detection, to/from quaternion
- [x] `crates/ix-rotation/src/axis_angle.rs` — Axis-angle representation, to/from quaternion/matrix
- [x] `crates/ix-rotation/src/rotation_matrix.rs` — SO(3) rotation matrix, orthogonalization, decomposition
- [x] `crates/ix-rotation/src/plucker.rs` — Plucker line coordinates, line-line distance, screw theory
- [x] GPU kernel for batch rotation (WGSL shader in ix-gpu, uses ix-rotation types)

##### Phase 2b: ix-sedenion (extract + expand)

- [x] `crates/ix-sedenion/Cargo.toml`
- [x] `crates/ix-sedenion/src/lib.rs`
- [x] `crates/ix-sedenion/src/sedenion.rs` — Sedenion struct (16D), add, mul, conjugate, norm, inverse
- [x] `crates/ix-sedenion/src/octonion.rs` — Octonion (8D), non-associative algebra
- [x] `crates/ix-sedenion/src/cayley_dickson.rs` — Generic Cayley-Dickson construction
- [x] `crates/ix-sedenion/src/bsp.rs` — BSP tree partitioning for spatial queries
- [x] GPU kernel for batch sedenion multiply

##### Phase 2c: ix-fractal (extract + expand)

- [x] `crates/ix-fractal/Cargo.toml`
- [x] `crates/ix-fractal/src/lib.rs`
- [x] `crates/ix-fractal/src/takagi.rs` — Takagi/Blancmange curve
- [x] `crates/ix-fractal/src/de_rham.rs` — de Rham fractal interpolation
- [x] `crates/ix-fractal/src/ifs.rs` — Iterated Function Systems
- [x] `crates/ix-fractal/src/lsystem.rs` — L-system grammar expansion
- [x] `crates/ix-fractal/src/space_filling.rs` — Hilbert, Peano, Morton curves

##### Phase 2d: ix-number-theory (net-new)

- [x] `crates/ix-number-theory/Cargo.toml`
- [x] `crates/ix-number-theory/src/lib.rs`
- [x] `crates/ix-number-theory/src/sieve.rs` — Sieve of Eratosthenes, Sieve of Atkin, segmented sieve
- [x] `crates/ix-number-theory/src/primality.rs` — Miller-Rabin, trial division, Fermat test
- [x] `crates/ix-number-theory/src/primes.rs` — Prime gaps, twin primes, triplets, prime counting
- [x] `crates/ix-number-theory/src/totient.rs` — Euler's totient, Mobius function, divisor functions
- [x] `crates/ix-number-theory/src/modular.rs` — Modular arithmetic, modular exponentiation, modular inverse
- [x] `crates/ix-number-theory/src/crt.rs` — Chinese Remainder Theorem
- [x] `crates/ix-number-theory/src/elliptic.rs` — Elliptic curves over finite fields

##### Phase 2e: Deepen existing crates

- [x] `ix-topo` — Vietoris-Rips filtration, Betti numbers, persistence diagram
- [x] `ix-dynamics` — Neural PDE solvers, SO(3)/SE(3) Lie groups, symplectic integrators
- [x] `ix-ktheory` — Grothendieck K0/K1, spectral sequences
- [x] `ix-category` — Monad trait + instances, adjunction pairs, Kan extensions

##### Phase 2 workspace integration

- [x] Add all new crates to `Cargo.toml` workspace members
- [x] Add all new crates to `[workspace.dependencies]`
- [x] Run `cargo clippy --workspace -- -D warnings` — zero warnings
- [x] Run `cargo test --workspace` — all tests pass
- [x] MSRV check: rust-version = "1.80" in workspace Cargo.toml

#### Phase 3: Demo App Expansion ✅

- [x] `crates/ix-demo/Cargo.toml` — Add dependencies
- [x] `crates/ix-demo/src/demos/rotation.rs` — 3D rotation viz
- [x] `crates/ix-demo/src/demos/number_theory.rs` — Ulam spiral, prime gaps
- [x] `crates/ix-demo/src/demos/fractal.rs` — Takagi, IFS, L-system
- [x] `crates/ix-demo/src/demos/sedenion.rs` — Cayley-Dickson chain viz
- [x] `crates/ix-demo/src/demos/topology.rs` — Point cloud, Rips complex
- [x] `crates/ix-demo/src/demos/category.rs` — Functor diagram, monad bind
- [x] `crates/ix-demo/src/demos/mod.rs` — 6 new pub mod entries
- [x] `crates/ix-demo/src/main.rs` — 6 new Tab variants, category navigation
- [x] Clippy clean

#### Phase 4: MCP & Skill Coverage ✅

##### MCP Tools (ix-agent) — 31 tools total

- [x] `ix_nn` tool — Forward pass through layer stack
- [x] `ix_rl` tool — Bandit simulation
- [x] `ix_evolution` tool — GA + DE on benchmark functions
- [x] `ix_ensemble` tool — Random forest train/predict
- [x] `ix_rotation` tool — Quaternion operations, SLERP, Euler, rotation matrices
- [x] `ix_number_theory` tool — Prime sieve, primality test, modular arithmetic
- [x] `ix_fractal` tool — Takagi curve, Hilbert/Peano curves, Morton encoding
- [x] `ix_sedenion` tool — Hypercomplex algebra operations
- [x] `ix_topo` tool — Persistent homology, Betti numbers
- [x] `ix_category` tool — Monad laws, free-forgetful adjunction
- [x] `ix_supervised` tool — Regression, classification, metrics
- [x] `ix_graph` tool — Dijkstra, BFS, DFS, PageRank, topological sort
- [x] `ix_hyperloglog` tool — HyperLogLog cardinality estimation
- [x] `ix_pipeline` tool — DAG pipeline analysis
- [x] `ix_cache` tool — In-memory cache operations

##### Claude Code Skills — 26 skills total

- [x] `.claude/skills/ix-rotation/SKILL.md`
- [x] `.claude/skills/ix-number-theory/SKILL.md`
- [x] `.claude/skills/ix-fractal/SKILL.md`
- [x] `.claude/skills/ix-sedenion/SKILL.md`
- [x] `.claude/skills/ix-topo/SKILL.md`
- [x] `.claude/skills/ix-category/SKILL.md`
- [x] `.claude/skills/ix-nn/SKILL.md`
- [x] `.claude/skills/ix-bandit/SKILL.md`
- [x] `.claude/skills/ix-evolution/SKILL.md`
- [x] `.claude/skills/ix-random-forest/SKILL.md`
- [x] `.claude/skills/ix-dynamics/SKILL.md`
- [x] `.claude/skills/ix-ktheory/SKILL.md`
- [x] `.claude/skills/ix-gpu/SKILL.md`
- [x] `.claude/skills/ix-cache/SKILL.md`
- [x] `.claude/skills/ix-grammar/SKILL.md`
- [x] `.claude/skills/ix-supervised/SKILL.md`

#### Phase 5: Stub Completion & Polish ✅

- [x] `ix-unsupervised` — t-SNE (Barnes-Hut), GMM (EM algorithm)
- [x] `ix-nn/src/network.rs` — Composable Sequential network (forward/backward/train)
- [x] `ix-rl/src/env.rs` — Environment trait + GridWorld
- [x] `ix-rl/src/q_learning.rs` — Tabular Q-learning with ε-greedy
- [x] `ix-skill/src/main.rs` — Wire data loading for Train/Cluster CLI commands
- [x] `ix-skill/src/main.rs` — Wire up decision-tree, SVM, DBSCAN, PCA CLI commands
- [x] All crates: verify every `pub fn` has at least one unit test
- [x] All crates: verify doc examples compile
- [x] Add `rust-version = "1.80"` to `[workspace.package]`
- [x] Add `categories`, `keywords`, `repository` to each crate's Cargo.toml
- [x] Verify `cargo publish --dry-run` works for leaf crates

## GPU Kernels ✅

All GPU kernels implemented in ix-gpu:
- [x] Batch quaternion rotation (WGSL shader + CPU fallback)
- [x] Pairwise distance matrix (2D dispatch + CPU fallback)
- [x] Batch sedenion multiplication (Cayley-Dickson on GPU + CPU)
- [x] Batch k-nearest neighbors (brute-force + CPU fallback)
- [x] BSP-partitioned kNN (spatial binning + CPU BSP tree)
- [x] Vietoris-Rips complex construction (GPU distance + CPU clique enumeration)
- [x] Matrix multiplication, cosine similarity, similarity matrix, top-k search

## Acceptance Criteria

### Functional Requirements

- [x] 4 new crates created: ix-rotation, ix-number-theory, ix-fractal, ix-sedenion
- [x] Existing crates deepened: ix-topo, ix-dynamics, ix-ktheory, ix-category
- [x] Demo app has 21+ tabs with category grouping
- [x] Every crate has at least one MCP tool in ix-agent (31 tools)
- [x] Every crate has a Claude Code skill in .claude/skills/ (26+ skills)
- [x] All TODO stubs completed (t-SNE, GMM, network, RL env, CLI commands)

### Non-Functional Requirements

- [x] CI passes on every push: build + clippy + test + doc (stable + nightly, Linux + Windows)
- [x] Zero clippy warnings with `-D warnings`
- [x] MSRV declared and tested (1.80)

### Quality Gates

- [x] Every `pub fn` has at least one `#[test]`
- [x] README accurate and complete with all crates listed
- [x] No backward-breaking changes without re-exports from old locations

## Success Metrics

- **Crate count**: 31 (from 27) ✅
- **Demo tabs**: 21+ (from 15) ✅
- **MCP tools**: 31 (from 17) ✅
- **Claude Code skills**: 26+ (from 10) ✅
- **CI status**: Green on every push ✅
- **Test coverage**: Every pub fn tested ✅

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-03-14-ix-full-vision-brainstorm.md](docs/brainstorms/2026-03-14-ix-full-vision-brainstorm.md)

### Internal References

- Crate template pattern: `crates/ix-signal/` (flat module structure, inline tests)
- GPU kernels: `crates/ix-gpu/src/` (10 kernel files)
- MCP tool registry: `crates/ix-agent/src/tools.rs` (31 tools)
- Skill template: `.claude/skills/ix-optimize/SKILL.md`
- Demo pattern: `crates/ix-demo/src/demos/transformer.rs`

### Related Work

- Previous plan: `docs/plans/2026-03-13-002-feat-tars-math-phase1-quaternions-primes-fractals-plan.md`
- TARS brainstorm: `docs/brainstorms/2026-03-13-tars-math-concepts-brainstorm.md`
