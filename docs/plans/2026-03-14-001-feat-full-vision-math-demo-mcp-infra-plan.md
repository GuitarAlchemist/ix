---
title: "feat: MachinDeOuf Full Vision — Math Crates + Demo + MCP + Infrastructure"
type: feat
status: active
date: 2026-03-14
origin: docs/brainstorms/2026-03-14-machineouf-full-vision-brainstorm.md
---

# feat: MachinDeOuf Full Vision — Math Crates + Demo + MCP + Infrastructure

## Overview

Comprehensive expansion of the MachinDeOuf workspace across four parallel workstreams: (1) four new math crates via extract+expand from existing code, (2) six new demo tabs with category navigation, (3) full MCP/skill coverage for all crates, (4) GitHub Actions CI, proptest/criterion, stub completion, and crates.io prep. (see brainstorm: docs/brainstorms/2026-03-14-machineouf-full-vision-brainstorm.md)

## Problem Statement / Motivation

The workspace has 27 crates, 17 MCP tools, 10 Claude Code skills, and 15 demo tabs — but significant gaps remain:
- Quaternion/sedenion CPU algebra is buried inside machin-gpu (GPU crate)
- Takagi/de Rham fractal curves live in machin-chaos instead of a dedicated fractal crate
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
machin-math ← machin-rotation (linalg primitives for Euler, axis-angle, matrices)
machin-rotation ← machin-gpu (GPU kernels import rotation types)
machin-math ← machin-number-theory (standalone, no new external deps initially)
machin-chaos ← machin-fractal (share types, re-exports for backward compat)
machin-sedenion ← machin-gpu (GPU kernels import sedenion types)
```

**Crate template pattern (from machin-signal):**
```
crates/machin-{name}/
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

#### Phase 1: Infrastructure Foundation

CI/CD, workspace-level testing framework, and crates.io prep.

**Files to create/modify:**

- [ ] `.github/workflows/ci.yml` — GitHub Actions: build + clippy + test + doc on stable/nightly, Linux/Windows
- [ ] `Cargo.toml` (workspace root) — Add `proptest`, `criterion` to `[workspace.dependencies]` as dev-dependencies
- [ ] `CLAUDE.md` — Update crate list (27→31+), add testing conventions, add MSRV
- [ ] `README.md` — Update to reflect all crates, add CI badge, document MCP setup
- [ ] `.mcp.json` — Register machin-agent: `{ "command": "cargo", "args": ["run", "-p", "machin-agent"] }`

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        rust: [stable, nightly]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
          components: clippy
      - uses: Swatinem/rust-cache@v2
      - run: cargo build --workspace
      - run: cargo clippy --workspace -- -D warnings
      - run: cargo test --workspace
      - run: cargo doc --workspace --no-deps
        if: matrix.rust == 'stable'
```

**Acceptance criteria:**
- [ ] CI runs on every push/PR
- [ ] Clippy passes with -D warnings across entire workspace
- [ ] All existing tests pass in CI
- [ ] machin-agent registered in .mcp.json and callable by Claude Code
- [ ] README lists all crates with accurate descriptions

#### Phase 2: Extract + Create Math Crates

Extract CPU algebra from machin-gpu, fractal curves from machin-chaos. Create machin-number-theory from scratch.

##### Phase 2a: machin-rotation (extract + expand)

**Extract from machin-gpu/src/quaternion.rs:**
- `batch_quaternion_rotate_cpu()` (lines 125-148) — pure CPU quaternion rotation

**New modules to implement:**

- [ ] `crates/machin-rotation/Cargo.toml` — depends on machin-math, ndarray, rand
- [ ] `crates/machin-rotation/src/lib.rs` — pub mod declarations
- [ ] `crates/machin-rotation/src/quaternion.rs` — Quaternion struct, mul, conjugate, inverse, normalize, from_axis_angle, to_matrix
- [ ] `crates/machin-rotation/src/dual_quaternion.rs` — DualQuaternion struct, rigid body transform, screw motion
- [ ] `crates/machin-rotation/src/slerp.rs` — Spherical linear interpolation, squad (spline)
- [ ] `crates/machin-rotation/src/euler.rs` — Euler angles (XYZ, ZYX, etc.), gimbal lock detection, to/from quaternion
- [ ] `crates/machin-rotation/src/axis_angle.rs` — Axis-angle representation, to/from quaternion/matrix
- [ ] `crates/machin-rotation/src/rotation_matrix.rs` — SO(3) rotation matrix, orthogonalization, decomposition
- [ ] `crates/machin-rotation/src/plucker.rs` — Plucker line coordinates, line-line distance, screw theory
- [ ] Update `machin-gpu/Cargo.toml` — Add dependency on machin-rotation
- [ ] Update `machin-gpu/src/quaternion.rs` — Import types from machin-rotation, keep only GPU kernel + dispatch
- [ ] Add proptest tests for: quaternion norm preservation, rotation composition associativity, SLERP interpolation bounds
- [ ] Add criterion benches for: batch rotation, SLERP throughput
- [ ] GPU kernel for batch rotation (WGSL shader in machin-gpu, uses machin-rotation types)

##### Phase 2b: machin-sedenion (extract + expand)

**Extract from machin-gpu/src/sedenion.rs:**
- `sedenion_mul()`, `octonion_mul()`, `quat_mul()` and conjugate helpers (lines 475-537) — Cayley-Dickson construction

**New modules:**

- [ ] `crates/machin-sedenion/Cargo.toml`
- [ ] `crates/machin-sedenion/src/lib.rs`
- [ ] `crates/machin-sedenion/src/sedenion.rs` — Sedenion struct (16D), add, mul, conjugate, norm, inverse
- [ ] `crates/machin-sedenion/src/octonion.rs` — Octonion (8D), non-associative algebra
- [ ] `crates/machin-sedenion/src/cayley_dickson.rs` — Generic Cayley-Dickson construction: complex→quaternion→octonion→sedenion
- [ ] `crates/machin-sedenion/src/bsp.rs` — BSP tree partitioning for spatial queries using sedenion subspaces
- [ ] Update `machin-gpu/src/sedenion.rs` — Import algebra from machin-sedenion, keep GPU kernel
- [ ] Add proptest: non-commutativity of octonions, norm multiplicativity, Cayley-Dickson consistency
- [ ] Add criterion: sedenion multiplication throughput
- [ ] GPU kernel for batch sedenion multiply (already exists in machin-gpu, update imports)

##### Phase 2c: machin-fractal (extract + expand)

**Extract from machin-chaos:**
- `crates/machin-chaos/src/takagi.rs` → `crates/machin-fractal/src/takagi.rs`
- `crates/machin-chaos/src/de_rham.rs` → `crates/machin-fractal/src/de_rham.rs`

**New modules:**

- [ ] `crates/machin-fractal/Cargo.toml` — depends on ndarray, rand, rand_distr
- [ ] `crates/machin-fractal/src/lib.rs`
- [ ] `crates/machin-fractal/src/takagi.rs` — (migrated) Takagi/Blancmange curve
- [ ] `crates/machin-fractal/src/de_rham.rs` — (migrated) de Rham fractal interpolation
- [ ] `crates/machin-fractal/src/ifs.rs` — Iterated Function Systems (Sierpinski, Barnsley fern, custom affine maps)
- [ ] `crates/machin-fractal/src/lsystem.rs` — L-system grammar expansion + turtle graphics interpretation
- [ ] `crates/machin-fractal/src/space_filling.rs` — Hilbert curve, Peano curve, Z-order (Morton) curve
- [ ] Update `machin-chaos/src/lib.rs` — Remove takagi/de_rham modules, add re-exports from machin-fractal for backward compat
- [ ] Update `machin-chaos/Cargo.toml` — Add machin-fractal dependency for re-exports
- [ ] Add proptest: IFS attractor convergence, L-system determinism, space-filling curve bijectivity
- [ ] Add criterion: Hilbert curve generation throughput

##### Phase 2d: machin-number-theory (net-new)

- [ ] `crates/machin-number-theory/Cargo.toml` — depends on machin-math
- [ ] `crates/machin-number-theory/src/lib.rs`
- [ ] `crates/machin-number-theory/src/sieve.rs` — Sieve of Eratosthenes, Sieve of Atkin, segmented sieve
- [ ] `crates/machin-number-theory/src/primality.rs` — Miller-Rabin, trial division, Fermat test
- [ ] `crates/machin-number-theory/src/primes.rs` — Prime gaps, twin primes, triplets/constellations, prime counting
- [ ] `crates/machin-number-theory/src/totient.rs` — Euler's totient, Mobius function, divisor functions
- [ ] `crates/machin-number-theory/src/modular.rs` — Modular arithmetic, modular exponentiation, modular inverse
- [ ] `crates/machin-number-theory/src/crt.rs` — Chinese Remainder Theorem
- [ ] `crates/machin-number-theory/src/elliptic.rs` — Elliptic curves over finite fields (point addition, scalar multiplication, ECDH skeleton)
- [ ] Add proptest: Euler's product formula, CRT correctness, Miller-Rabin false positive rates
- [ ] Add criterion: sieve throughput for 10^6/10^7/10^8

##### Phase 2e: Deepen existing crates

- [ ] `machin-topo` — Add Vietoris-Rips filtration from point clouds, Betti number computation, persistence diagram output
- [ ] `machin-dynamics` — Add Neural PDE solvers, SO(3)/SE(3) Lie group instances, symplectic integrators
- [ ] `machin-ktheory` — Add Grothendieck K0/K1 computation examples, spectral sequence basics
- [ ] `machin-category` — Add Monad trait + instances, adjunction pairs, Kan extensions

##### Phase 2 workspace integration

- [ ] Add all new crates to `Cargo.toml` workspace members
- [ ] Add all new crates to `[workspace.dependencies]` with `path = "crates/machin-{name}"`
- [ ] Run `cargo clippy --workspace -- -D warnings` — zero warnings
- [ ] Run `cargo test --workspace` — all tests pass
- [ ] MSRV check: determine minimum Rust version, add `rust-version` to workspace Cargo.toml

#### Phase 3: Demo App Expansion

Add 6 new tabs + category navigation + UX improvements.

- [ ] `crates/machin-demo/Cargo.toml` — Add dependencies: machin-rotation, machin-number-theory, machin-fractal, machin-sedenion, machin-topo, machin-category
- [ ] `crates/machin-demo/src/demos/rotation.rs` — 3D rotation viz (SLERP animation, Euler gimbal lock, quaternion interpolation)
- [ ] `crates/machin-demo/src/demos/number_theory.rs` — Ulam spiral, prime gap histogram, sieve performance comparison, modular arithmetic explorer
- [ ] `crates/machin-demo/src/demos/fractal.rs` — Takagi curve parameter explorer, IFS renderer (fern, Sierpinski), L-system expander
- [ ] `crates/machin-demo/src/demos/sedenion.rs` — Cayley-Dickson chain visualization, sedenion multiplication table, norm preservation demo
- [ ] `crates/machin-demo/src/demos/topology.rs` — Point cloud input, Rips complex construction, persistence barcode/diagram
- [ ] `crates/machin-demo/src/demos/category.rs` — Functor diagram, composition chain visualization, monad bind demo
- [ ] `crates/machin-demo/src/demos/mod.rs` — Add 6 new pub mod entries
- [ ] `crates/machin-demo/src/main.rs` — Add 6 new Tab variants, update ALL/label/struct/Default/update
- [ ] `crates/machin-demo/src/main.rs` — Add category navigation: group tabs into sections (Core Math, ML, Advanced Math, Infrastructure) using `egui::CollapsingHeader` or horizontal separators
- [ ] Add parameter presets ("Interesting configurations" buttons) to at least 5 existing tabs
- [ ] Add computation time display to status bar for each demo
- [ ] Clippy clean: `cargo clippy -p machin-demo -- -D warnings`

#### Phase 4: MCP & Skill Coverage

Fill gaps in MCP tools and Claude Code skills for all crates.

##### MCP Tools (machin-agent)

Add tools for crates that lack them (see coverage gap table in brainstorm):

- [ ] `machin_supervised` tool — Linear/logistic regression, SVM, KNN prediction
- [ ] `machin_nn` tool — Forward pass through layer stack, transformer attention
- [ ] `machin_rl` tool — Bandit arm selection, Q-learning step
- [ ] `machin_evolution` tool — GA run with fitness function, DE optimization
- [ ] `machin_graph` tool — Shortest path, PageRank, connected components
- [ ] `machin_probabilistic_hll` tool — HyperLogLog cardinality estimation
- [ ] `machin_pipeline` tool — DAG execution with step definitions
- [ ] `machin_ensemble` tool — Random forest train/predict
- [ ] `machin_rotation` tool — Quaternion operations, SLERP, Euler conversion
- [ ] `machin_number_theory` tool — Prime sieve, primality test, modular arithmetic
- [ ] `machin_fractal` tool — Takagi curve, IFS generation, L-system expansion
- [ ] `machin_sedenion` tool — Sedenion algebra operations
- [ ] `machin_topo` tool — Persistent homology from point cloud
- [ ] `machin_category` tool — Functor mapping, monad bind
- [ ] Update `crates/machin-agent/src/tools.rs` — Register all new tools with JSON schemas
- [ ] Update `crates/machin-agent/src/handlers.rs` — Implement handlers for all new tools

##### Claude Code Skills

Create SKILL.md files for crates that lack them:

- [ ] `.claude/skills/machin-rotation/SKILL.md`
- [ ] `.claude/skills/machin-number-theory/SKILL.md`
- [ ] `.claude/skills/machin-fractal/SKILL.md`
- [ ] `.claude/skills/machin-sedenion/SKILL.md`
- [ ] `.claude/skills/machin-topo/SKILL.md`
- [ ] `.claude/skills/machin-category/SKILL.md`
- [ ] `.claude/skills/machin-dynamics/SKILL.md`
- [ ] `.claude/skills/machin-ktheory/SKILL.md`
- [ ] `.claude/skills/machin-gpu/SKILL.md`
- [ ] `.claude/skills/machin-cache/SKILL.md`
- [ ] `.claude/skills/machin-grammar/SKILL.md`
- [ ] `.claude/skills/machin-nn/SKILL.md`
- [ ] `.claude/skills/machin-supervised/SKILL.md`
- [ ] `.claude/skills/machin-ensemble/SKILL.md`
- [ ] `.claude/skills/machin-rl/SKILL.md`
- [ ] `.claude/skills/machin-evolution/SKILL.md`

Each skill follows the existing pattern (see `.claude/skills/machin-optimize/SKILL.md`): when to use, method selection guidance, execution examples, output format.

#### Phase 5: Stub Completion & Polish

Complete all remaining TODO items and prepare for crates.io.

- [ ] `machin-unsupervised` — Implement t-SNE (Barnes-Hut approximation), GMM (EM algorithm)
- [ ] `machin-nn/src/network.rs` — Composable Sequential network (layer stack with forward/backward)
- [ ] `machin-rl/src/env.rs` — Environment trait + GridWorld implementation
- [ ] `machin-rl/src/q_learning.rs` — Complete Q-learning using Environment trait
- [ ] `machin-skill/src/main.rs` — Wire data loading for Train/Cluster CLI commands
- [ ] `machin-skill/src/main.rs` — Wire up decision-tree, SVM, DBSCAN, PCA CLI commands
- [ ] All crates: verify every `pub fn` has at least one unit test
- [ ] All crates: verify doc examples compile (`cargo doc --workspace --no-deps`)
- [ ] Add `rust-version = "1.80"` (or appropriate MSRV) to `[workspace.package]`
- [ ] Add `categories`, `keywords`, `repository` to each crate's Cargo.toml for crates.io
- [ ] Verify `cargo publish --dry-run` works for leaf crates (no workspace-only deps)

## System-Wide Impact

### Interaction Graph

- New crates (rotation, sedenion) become dependencies of machin-gpu → GPU kernel compilation depends on new crate APIs stabilizing first
- machin-fractal depends on ndarray/rand (no cross-crate interaction beyond re-exports from machin-chaos)
- machin-demo depends on ALL crates → any API change triggers demo recompile
- machin-agent (MCP) depends on algorithm crates → new tools require matching handler implementations

### Error & Failure Propagation

- All math crates use `Result<T, E>` with thiserror — consistent error handling
- GPU crates can fail on device/adapter availability → CPU fallback pattern already established
- MCP handlers catch panics and convert to JSON error responses
- Demo app is fire-and-forget UI — panics in computation show as error status strings

### State Lifecycle Risks

- No persistent state beyond files — all computation is stateless
- machin-cache has in-memory state but is not used by other crates
- Risk: extracting modules from machin-gpu/machin-chaos could break downstream code if re-exports are incomplete → mitigate with deprecation re-exports

### API Surface Parity

Three interfaces per crate:
1. **Rust API** (lib.rs public functions) — primary
2. **MCP tool** (machin-agent JSON-RPC) — secondary
3. **CLI** (machin-skill) — tertiary

All three must expose equivalent functionality. Currently, MCP covers 17/27 crates and CLI covers ~10/27.

### Integration Test Scenarios

1. **GPU extraction**: After moving quaternion types to machin-rotation, verify machin-gpu batch rotation still produces identical results
2. **Fractal migration**: After moving takagi.rs to machin-fractal, verify machin-chaos re-export produces identical output
3. **MCP end-to-end**: Register machin-agent in .mcp.json, call a tool via Claude Code, verify JSON response
4. **Demo compilation**: After adding 6 new tabs + 4 new crate dependencies, verify demo compiles and runs
5. **CI pipeline**: Push a commit with intentional clippy warning, verify CI catches it

## Acceptance Criteria

### Functional Requirements

- [ ] 4 new crates created: machin-rotation, machin-number-theory, machin-fractal, machin-sedenion
- [ ] Existing crates deepened: machin-topo, machin-dynamics, machin-ktheory, machin-category
- [ ] Demo app has 21+ tabs with category grouping
- [ ] Every crate has at least one MCP tool in machin-agent
- [ ] Every crate has a Claude Code skill in .claude/skills/
- [ ] All TODO stubs completed (t-SNE, GMM, network, RL env, CLI commands)

### Non-Functional Requirements

- [ ] CI passes on every push: build + clippy + test + doc (stable + nightly, Linux + Windows)
- [ ] Zero clippy warnings with `-D warnings`
- [ ] proptest coverage for all math crates (invariant properties)
- [ ] criterion benchmarks for performance-critical operations
- [ ] MSRV declared and tested
- [ ] `cargo publish --dry-run` succeeds for leaf crates

### Quality Gates

- [ ] Every `pub fn` has at least one `#[test]`
- [ ] Every public API has a doc example that compiles
- [ ] README accurate and complete with all crates listed
- [ ] No backward-breaking changes without re-exports from old locations

## Dependencies & Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPU tests fail in CI (no hardware) | High | Medium | Use wgpu software adapter or feature-gate GPU tests |
| Elliptic curves scope creep | Medium | High | Implement basic point addition first; full EC crypto as follow-up |
| machin-gpu extraction breaks consumers | Medium | High | Add deprecation re-exports; keep GPU kernel in machin-gpu |
| CI is slow (31+ crates) | High | Low | Rust build caching, workspace-level incremental compilation |
| WASM incompatibility for some crates | Low | Medium | Feature-gate machin-gpu, machin-io, machin-cache for WASM |

## Success Metrics

- **Crate count**: 31+ (from 27)
- **Demo tabs**: 21+ (from 15)
- **MCP tools**: 31+ (from 17)
- **Claude Code skills**: 26+ (from 10)
- **CI status**: Green on every push
- **Test coverage**: Every pub fn tested, proptest for math invariants
- **crates.io ready**: dry-run publish succeeds for all leaf crates

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-03-14-machineouf-full-vision-brainstorm.md](docs/brainstorms/2026-03-14-machineouf-full-vision-brainstorm.md) — Key decisions carried forward: machin-rotation (broad scope), GPU from day one, all three test frameworks, crates.io publishing soon.

### Internal References

- Crate template pattern: `crates/machin-signal/` (flat module structure, inline tests)
- GPU extraction points: `crates/machin-gpu/src/quaternion.rs:125-148`, `crates/machin-gpu/src/sedenion.rs:475-537`
- Fractal migration: `crates/machin-chaos/src/takagi.rs`, `crates/machin-chaos/src/de_rham.rs`
- MCP tool registry: `crates/machin-agent/src/tools.rs` (17 existing tools)
- Skill template: `.claude/skills/machin-optimize/SKILL.md`
- Demo pattern: `crates/machin-demo/src/demos/transformer.rs` (latest tab added)

### Related Work

- Previous plan: `docs/plans/2026-03-13-002-feat-tars-math-phase1-quaternions-primes-fractals-plan.md`
- TARS brainstorm: `docs/brainstorms/2026-03-13-tars-math-concepts-brainstorm.md`
