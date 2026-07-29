---
review_agents: [code-simplicity-reviewer, security-sentinel, performance-oracle, architecture-strategist]
plan_review_agents: [code-simplicity-reviewer]
---

# Review Context

Add project-specific review instructions here.
These notes are passed to all review agents during /workflows:review and /workflows:work.

## This repo (ix)

- Rust workspace, 78 crates — pure Rust, no external ML frameworks except `wgpu` for GPU compute. CPU algorithms use `f64` + `ndarray`; GPU uses `f32` via WGPU. MSRV 1.80+.
- CI bar is exact: `cargo clippy --workspace --all-targets -D warnings` and `cargo fmt --check`. Run the CI-exact invocation before claiming green (`--bins --lib` misses lints).
- Crate maturity is tiered (`crate-maturity.toml`); a stable-surface hash gate hashes `pub `-prefixed lines — adding a `pub fn` to a stable crate trips it. `experimental`-tier crates don't.
- Do NOT add `petgraph`/`daggy`/`graph-rs` — use `ix-graph`, `ix-pipeline::dag`, `ix-search`, `ix-topo` (ix already has 10 graph modules).
- Governance: agent actions are subject to the Demerzel constitution; hexavalent (6-valued T/P/U/D/F/C) logic, not tetravalent. Invariants/assumptions carry `@ai:` annotations with a live binding + confidence.
- Beware "green-but-dead": a passing gate on a placeholder or a metric that emits `0` on a *missing* input (distinguish "cannot evaluate here" from "evaluated and bad" — the maintain-gate #238/#244 trap).
- IX is a pure ML layer; musical/domain semantics live in GA. The IX↔GA bridge is on the interval-class vector (ICV) — never bridge on Forte number.
- Some subtrees are Python (`crates/ix-optick-sae/python`, `.agent-blackbox`) — stdlib-only test discipline where possible (no torch import in contract tests).

Examples of the kind of note to add:
- "This endpoint is performance-critical: 10k req/s."
- "Public API — extra scrutiny on input validation."
