# ix — ML Algorithms + Governance for Claude Code Skills

Rust workspace (32 crates) implementing foundational ML/math algorithms and AI governance as composable crates, exposed via MCP server (`ix-agent`) and CLI (`ix-skill`). Part of the GuitarAlchemist ecosystem (ix + tars + ga + Demerzel).

**Crate map**: see `README.md` for the full list of 32 crates grouped by domain.

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

MSRV: Rust 1.80+ (due to wgpu 28).

## Conventions

- Pure Rust; no external ML frameworks except `wgpu` for GPU compute.
- CPU algorithms use `f64` and `ndarray::Array{1,2}<f64>`; GPU uses `f32` via WGPU shaders.
- Each crate defines traits (`Regressor`, `Classifier`, `Clusterer`, `Optimizer`, etc.).
- Builder pattern for algorithm configuration; seeded RNG for reproducibility.
- Governance: all agent actions subject to Demerzel constitution (see `governance/demerzel`).
- Before adding a new graph primitive, check `docs/guides/graph-theory-in-ix.md` — IX already has 10 graph-theory modules.
- Do NOT add `petgraph`/`daggy`/`graph-rs` as new dependencies. Use `ix-graph`, `ix-pipeline::dag::Dag<N>`, `ix-search`, or `ix-topo`.

## MCP Federation

Three MCP servers in `.mcp.json`: **ix** (Rust, algorithms + governance), **tars** (F#, grammar + metacognition), **ga** (C#, music theory). Capability registry: `governance/demerzel/schemas/capability-registry.json`.

## Belief State

State lives in `state/` — beliefs, PDCA cycles, knowledge packages, snapshots. File naming: `{date}-{short-description}.{type}.json`.

For governance details, Demerzel policies, Galactic Protocol contracts, or agent persona requirements, use the `demerzel-*` skills.
