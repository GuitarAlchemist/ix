---
title: "feat: R4 Meta-MCP Gateway (demerzel-gateway)"
type: feat
status: draft
date: 2026-04-14
origin: examples/canonical-showcase/ix-roadmap-plan-v1.md §R4
depends_on:
  - R2 Phase 2 (asset naming + lineage DAG — shipped)
  - R3 registry CI (schema stability gate — shipped)
---

# feat: R4 Meta-MCP Gateway (`demerzel-gateway`)

## Overview

R4 introduces a thin in-process MCP server — `demerzel-gateway` — that sits
between a client (Claude Code, a human, a CI job) and the three downstream
MCP servers in the GuitarAlchemist ecosystem: `ix` (Rust math/ML, 61 tools),
`tars` (F# reasoning, ~18 tools), and `ga` (C# music theory, ~50 tools).
The gateway aggregates their `tools/list` responses under a single prefixed
namespace (`ix__tool_name`, `tars__tool_name`, `ga__tool_name`), routes every
`tools/call` to the right downstream, and intercepts every call for Demerzel
audit via `ix_governance_check` before forwarding.

The goal is **one MCP endpoint** for consumers, mechanical name-collision
handling, and a single audit trail that spans the three repos. Today's
federation is discovery-only (`ix_federation_discover` reads a static
capability registry). R4 is the routing layer that actually makes that
discovery useful.

## Problem Statement

### Today's federation is discovery, not routing.

- `.mcp.json` lists three separate servers. Every client has to register
  all three and manage three connections.
- `ix_federation_discover` reads `capability-registry.json` and answers
  "what tools exist in the ecosystem?" It does **not** call any of them.
- The existing `ix_tars_bridge` and `ix_ga_bridge` tools format payloads
  for cross-repo hand-off but require the caller to already have a
  connection to the target server.
- Tool-name collisions are not handled. `grammar_weights` exists in both
  `ix` and `tars` with different signatures, and there is no mechanism
  to disambiguate them short of picking one by hand.
- Audit coverage is per-server. A pipeline that touches `ix_stats`,
  `ga_trace_stats`, and `ingest_ga_traces` produces three separate audit
  records with no cross-repo lineage.

### What breaks as we scale.

- **Client fan-out.** A CI job that wants to run a bracket pipeline needs
  stdin/stdout to three child processes, lifecycle management for each,
  and error correlation. The bracket showcase already wants this; it
  currently punts by keeping everything in `ix`.
- **Name collisions will get worse.** R6 (adversarial loops) and R7
  (autograd) both propose adding more grammar / pattern / gradient tools
  that overlap in name with TARS. Without a gateway-level namespace, the
  capability registry becomes an ordering-dependent mess.
- **Audit trail is fragmented.** Demerzel governance assumes every
  agent action is traced to one constitutional check. When an ix call
  triggers a tars call via a bridge, there is no unified audit record
  that ties them together — the governance articles that apply to
  cross-repo lineage can't be evaluated.

## Goals

1. **Single MCP endpoint.** A client registers `demerzel-gateway` in
   `.mcp.json` and reaches every tool in the ecosystem through it.
2. **Mechanical name disambiguation.** Every tool is exposed with a
   `<repo>__<tool>` prefix at the gateway boundary. No silent shadowing.
3. **Demerzel audit interception.** Every `tools/call` routed through the
   gateway produces a single audit record that records the upstream
   caller, the downstream server, the tool name, the arguments hash, and
   the Demerzel verdict from `ix_governance_check`.
4. **Pass-through emergency mode.** A gateway config flag disables the
   audit interception and turns the gateway into a naked proxy. Lets us
   roll back instantly if the audit layer misfires.
5. **Bit-identical output parity.** Running the bracket showcase through
   the gateway must produce the same `result["spec"]` and execution
   output as running it directly against `ix`, byte-for-byte (within the
   lineage DAG's new gateway-adds-a-hop distinction).

## Non-goals

- **Not a message bus.** No pub/sub, no queue, no streaming. Request →
  response, same shape as MCP.
- **Not a multi-tenant auth surface.** Single-user, single-machine. AuthN
  is out of scope for R4; R4 assumes the client is trusted.
- **Not a schema validator.** R3 (registry CI) already enforces that
  downstream schemas are stable. The gateway trusts R3; it does not
  re-validate every call's arguments against the downstream schema.
- **Not a caching layer.** R2's lineage DAG + blake3 content-addressed
  cache sits **below** the gateway, inside `ix`. The gateway is a thin
  proxy, not a cache.
- **Not a remote gateway.** Local stdio only. A remote WebSocket / HTTP
  variant is a follow-on (R4.1) gated on a real need.

## Current state

### Federation today: three moving parts.

| Piece | Role | Where |
|---|---|---|
| `capability-registry.json` | Static JSON listing every tool per repo | `governance/demerzel/schemas/capability-registry.json` |
| `ix_federation_discover` | Query the registry by domain/keyword | `crates/ix-agent/src/handlers.rs::federation_discover` |
| `ix_tars_bridge` | Format ix payloads for TARS ingestion | `crates/ix-agent/src/handlers.rs::tars_bridge` |
| `ix_ga_bridge` | Format ix payloads for GA ingestion | `crates/ix-agent/src/handlers.rs::ga_bridge` |
| `.mcp.json` (root) | Client-side server list — three entries | `/.mcp.json` |

None of this routes traffic. Everything runs through the client.

### Dependencies for R4.

- **R2 Phase 2 (shipped)** — asset naming (`$step.asset_name`) means the
  lineage DAG has stable identifiers the gateway can reference in audit
  records. Without stable names we'd be logging anonymous hops.
- **R3 registry CI (shipped)** — the gateway's tool-list aggregation is
  mostly a JOIN across three downstream `tools/list` responses. R3
  guarantees those responses are stable between versions. Without R3,
  every downstream rebuild would re-break the gateway's name cache.

Both are on `main`. R4 is unblocked.

## Architecture

### Crate layout

New crate `crates/demerzel-gateway/`. Conventions:

```
crates/demerzel-gateway/
├── Cargo.toml
├── src/
│   ├── lib.rs            # public API: GatewayConfig, Gateway, Router
│   ├── downstream.rs     # per-server stdio connection management
│   ├── aggregator.rs     # tools/list merge + prefix handling
│   ├── router.rs         # tools/call dispatch to the right downstream
│   ├── audit.rs          # Demerzel audit interception
│   └── bin/
│       └── demerzel-gateway.rs   # stdio MCP server binary
└── tests/
    ├── aggregation.rs    # three fake downstreams → merged namespace
    ├── routing.rs        # call routing + name collision handling
    ├── audit.rs          # every call produces an audit record
    └── pass_through.rs   # pass-through mode bypasses audit
```

Why in the `ix` workspace and not its own repo: two reasons.

1. **R4 depends on `ix_governance_check` for audit interception.** Keeping
   them in the same workspace lets us call it as a Rust dependency
   instead of spawning `ix` just to ask it a governance question.
2. **The roadmap allows later relocation.** R4 notes that the gateway
   "could move to the demerzel repo later". Landing it in `ix` first is
   the low-friction option; we can lift it out if we need a process
   boundary between `ix` the tool server and `ix` the audit authority.

### Transport: stdio-to-stdio, spawn children.

The gateway is itself an MCP server over stdio. On startup it spawns the
three downstream servers as child processes and holds open stdio pipes to
each. The path to each downstream binary is read from `GatewayConfig`
(which the gateway binary loads from its own `.demerzel-gateway.json` or
from command-line arguments), **not** from the client's `.mcp.json`. The
client only knows about `demerzel-gateway`.

```
┌─────────────┐        stdio        ┌──────────────────┐       stdio      ┌──────────┐
│   client    │ ───────────────────▶│ demerzel-gateway │ ────────────────▶│   ix     │
│ (CC, CI...) │ ◀─────────────────  │  (routes + audit)│ ◀──────────────  │          │
└─────────────┘                     │                  │                  └──────────┘
                                    │                  │       stdio      ┌──────────┐
                                    │                  │ ────────────────▶│   tars   │
                                    │                  │ ◀──────────────  │          │
                                    │                  │                  └──────────┘
                                    │                  │       stdio      ┌──────────┐
                                    │                  │ ────────────────▶│   ga     │
                                    │                  │ ◀──────────────  │          │
                                    └──────────────────┘                  └──────────┘
```

Children are spawned on first request (lazy) or on startup (eager),
configurable. Lazy start reduces boot time when only some downstreams
are needed; eager start catches misconfiguration at launch.

Sampling (`sampling/createMessage`) requests from a downstream server
are **forwarded upstream** to the client verbatim, with the envelope
`id` rewritten so the gateway can correlate the response and deliver
it back to the right downstream. This is non-trivial and is its own
implementation subsection (§4.5 below).

### Tool namespacing: mechanical prefixing

Every downstream tool is exposed at the gateway with a fixed prefix:

| Downstream | Prefix | Example |
|---|---|---|
| `ix`   | `ix__`   | `ix__stats`, `ix__pipeline_run` |
| `tars` | `tars__` | `tars__grammar_weights`, `tars__tars_compile_plan` |
| `ga`   | `ga__`   | `ga__GaParseChord`, `ga__GaAnalyzeProgression` |

Note the **double underscore** separator, deliberately chosen because
single `_` is ambiguous (tools are already named `ix_stats`, so `ix_`
as a prefix would be invisible). Double `_` is visually distinct and
reserved.

When the gateway receives `tools/list`, it:
1. Queries each downstream's `tools/list` in parallel.
2. Prepends the prefix to every tool name.
3. Leaves the schema untouched.
4. Merges into a single flat array in stable repo order: `ix__*`,
   `tars__*`, `ga__*`.
5. Returns the merged list.

Name collisions between repos are now impossible by construction.
Collisions *within* a repo are the downstream server's problem (R3
already enforces this on `ix`; `tars` and `ga` self-validate).

### Routing

On `tools/call name=foo__bar`, the gateway:

1. Parses the prefix. Unknown prefix → `-32601 Method not found`.
2. Looks up the downstream connection for that prefix. Dead connection
   → either restart (if lazy + within retry budget) or hard-fail with
   `-32603 Internal error: downstream <repo> unavailable`.
3. Rewrites `name` from `foo__bar` → `bar` (strips the prefix).
4. Runs the request through the **audit gate** (§4.4) unless in
   pass-through mode.
5. Forwards the rewritten envelope to the downstream.
6. Awaits the response envelope on the downstream's outbound pipe.
7. **Does not rewrite** the response's tool name — the response shape
   is standard MCP `tools/call` result with no tool name in it.
8. Records the audit trailer (§4.4) and returns the response upstream.

Request IDs are remapped per-downstream (the gateway maintains three
id pools, one per downstream) so the gateway can distinguish upstream
ids from downstream ids. This is standard proxy plumbing.

### Audit interception

Before forwarding a `tools/call`, the gateway synthesizes a governance
check payload:

```rust
let check_args = json!({
    "action": format!("gateway_call:{}__{}", repo_prefix, tool_name),
    "context": {
        "gateway_hop": true,
        "upstream_request_id": upstream_id,
        "downstream_server": repo_prefix,
        "downstream_tool": tool_name,
        "arguments_blake3": blake3_hash(&serde_json::to_vec(&args)?),
        "constitution_version": gateway_config.constitution_version,
    }
});
let verdict = ix_governance::check(check_args)?;
```

The check is a **direct in-process Rust call** to `ix_governance::check`,
not an MCP round-trip. That's the reason the gateway lives in the `ix`
workspace: `ix-governance` is a regular Rust crate and the gateway
depends on it as a library.

Every verdict is written to a per-session audit log at
`state/gateway-audit/<session-id>.jsonl`, one line per call, containing:

```json
{
  "timestamp": "2026-04-14T18:12:34.567Z",
  "session_id": "...",
  "upstream_id": 42,
  "downstream_id": 17,
  "downstream_server": "tars",
  "downstream_tool": "grammar_weights",
  "arguments_blake3": "a3f9d2...",
  "verdict": {
    "compliant": true,
    "constitution_version": "2.2.0",
    "relevant_articles": ["Art.4-Traceability"],
    "warnings": []
  },
  "duration_ms": 83
}
```

Enforcement policy: `compliant=false` with **any** warning is a hard
block (the gateway returns `-32000 Blocked by Demerzel governance` to
the upstream caller and does not forward to the downstream). `compliant=
true` with warnings is logged but forwarded. Pass-through mode bypasses
both.

### Sampling pass-through

`sampling/createMessage` is the one genuinely non-trivial piece. A
downstream server (for example, `ix_pipeline_compile`) may ask the
client to sample an LLM mid-call. The gateway must:

1. Intercept the downstream's outbound `sampling/createMessage` envelope.
2. Rewrite its `id` to a gateway-unique id.
3. Forward the rewritten envelope upstream.
4. Wait for the client's response.
5. Reverse-map the client's response `id` back to the downstream's
   original id.
6. Forward the response back to the originating downstream.

This is the second id-pool problem. The gateway needs **two** id maps
per downstream: one for client → downstream request pairs, and one for
downstream → client sampling-request pairs. Pair correlation is by
per-pool monotonic counter.

Sampling pass-through **must** work or `ix_pipeline_compile` breaks
through the gateway — that's a gating acceptance test.

## API

### From the client's perspective

The gateway is a plain MCP server. The client sees:

- `tools/list` — returns a merged array, prefix-namespaced.
- `tools/call name="ix__stats" arguments={...}` — routed to `ix`.
- `sampling/createMessage` — emitted by the gateway when a downstream
  requests sampling.

There are no new gateway-specific tools in the first cut. All observability
lives in the audit log.

### From the downstream's perspective

Downstreams see a normal MCP client talking to them. They don't know
they're behind a gateway. The only thing that changes is:

- Request `id` values are in a different range than they would be in
  a direct client connection. Most servers don't care.
- Responses flow back on the same pipe they arrived on, same as before.

### Config shape

`demerzel-gateway.config.json` at the gateway binary's working directory:

```json
{
  "constitution_version": "2.2.0",
  "downstreams": [
    {
      "prefix": "ix",
      "command": "C:/Users/spare/source/repos/ix/target/release/ix-mcp.exe",
      "args": [],
      "spawn": "eager"
    },
    {
      "prefix": "tars",
      "command": "C:/Users/spare/source/repos/tars/...",
      "args": ["mcp", "server"],
      "spawn": "lazy"
    },
    {
      "prefix": "ga",
      "command": "dotnet",
      "args": ["run", "--project", "..."],
      "spawn": "lazy"
    }
  ],
  "audit": {
    "enabled": true,
    "log_dir": "state/gateway-audit",
    "fail_on_warning": false
  },
  "pass_through": false
}
```

## Implementation plan

Five phases, each shippable independently. Phase N+1 depends on Phase N.

### Phase 1 — Skeleton + eager downstream (2 days)

- Create `crates/demerzel-gateway/` with empty `Gateway` struct and the
  binary entrypoint.
- Add `ix-governance` as a Rust dependency.
- Spawn **one** downstream (`ix`) eagerly on startup. Hold its stdio
  pipes in the gateway struct.
- Implement request/response plumbing: read JSON-RPC lines from
  upstream, forward unchanged to downstream, forward responses back.
  No prefixing, no audit — raw proxy with one downstream.
- Test: a tools/list call returns the ix tools list verbatim. A
  tools/call runs ix_stats end-to-end through the gateway.

**Exit criterion:** `demerzel-gateway --config gateway.json` acts as a
transparent proxy to a single ix server.

### Phase 2 — Multi-downstream + prefixing (2-3 days)

- Extend downstream to a `Vec<DownstreamConnection>`, keyed by prefix.
- On `tools/list`, call each downstream in parallel via `tokio::join!`.
- Prefix every returned tool name with `<repo>__`.
- On `tools/call`, parse the prefix, strip it, dispatch to the right
  downstream.
- Test: all three prefixes work; a deliberate collision (a fake
  downstream exposing a tool named `ix__stats`) is handled cleanly.

**Exit criterion:** a client registers only `demerzel-gateway` and can
call `ix__stats`, `tars__grammar_weights`, and `ga__GaParseChord` in a
single session.

### Phase 3 — Audit interception (2 days)

- Add `audit.rs`. Synthesize the governance check payload for every
  call. Call `ix_governance::check` in-process.
- Write audit records to `state/gateway-audit/<session-id>.jsonl`.
- Implement the `fail_on_warning` policy knob.
- Test: every tools/call produces exactly one audit record with the
  expected fields; a deliberate violation (e.g., calling a tool whose
  name matches a constitutional-trigger keyword) produces `compliant=
  false` and the gateway blocks.

**Exit criterion:** audit log shows full coverage of a bracket showcase
run through the gateway.

### Phase 4 — Sampling pass-through (2 days)

- Implement the two-id-pool machinery per downstream.
- Wire `sampling/createMessage` forwarding in both directions.
- Test: `ix__ix_pipeline_compile` through the gateway compiles a
  sentence end-to-end, with the sampling call round-tripping through
  the gateway to the real upstream client.

**Exit criterion:** bracket showcase's `ix_pipeline_compile` step
runs to completion when executed via the gateway.

### Phase 5 — Parity test + rollback switch (1-2 days)

- Add `pass_through: true` config flag that bypasses audit
  interception entirely.
- Build a parity test: run the bracket showcase through the direct
  ix connection and through the gateway, diff the JSON outputs. Must
  be bit-identical (except audit metadata).
- Add the `examples/canonical-showcase/04-catia-bracket-generative/via-
  gateway/` variant with `pipeline.json` pointing at gateway-prefixed
  names.

**Exit criterion:** gateway matches the direct-ix path bit-for-bit on
a real showcase. Pass-through mode verified to skip audit.

Total: **9-11 dev-days**, matching the roadmap's 10-12 estimate.

## Test strategy

### Unit tests (per-module)

- `downstream::tests::spawn_and_handshake` — mock child process,
  verify stdio framing.
- `aggregator::tests::merge_preserves_order` — three canned
  `tools/list` responses → correct merged order.
- `aggregator::tests::prefix_is_deterministic` — same downstream
  list → same prefixed output.
- `router::tests::unknown_prefix_returns_method_not_found`.
- `router::tests::strips_prefix_before_forwarding`.
- `audit::tests::records_every_call`.
- `audit::tests::fail_on_warning_blocks_forwarding`.

### Integration tests (against fake downstreams)

- `tests/three_fake_downstreams.rs` — spawn three Rust binaries that
  speak minimal MCP, register them as downstreams, run a multi-call
  session, verify routing + audit.
- `tests/name_collision.rs` — two downstreams both expose a tool
  named `foo`. Verify both are reachable via their prefixes, no
  collision.
- `tests/sampling_roundtrip.rs` — fake downstream emits
  `sampling/createMessage`, fake upstream client delivers a response,
  verify the downstream sees the correct reply.

### Acceptance tests (against real downstreams)

- `tests/bracket_showcase_parity.rs` — run the bracket showcase
  through `ix` directly and through the gateway. Diff JSON. Must be
  bit-identical on `result["spec"]`, `result["execution_order"]`,
  and `result["lineage"]`.
- `tests/pipeline_compile_via_gateway.rs` — run
  `ix_pipeline_compile` on a real sentence through the gateway;
  verify the compiled spec is structurally equivalent to the
  direct path.

### Regression gate

- Add `tests/gateway_audit_coverage_budget.rs`: every call in a
  showcase run must produce exactly one audit record. Fails if audit
  coverage drops below 100%.

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Sampling pass-through id-mapping bug corrupts a compile call | Medium | High | Phase 4 is gated on the sampling integration test. Every sampling id uses a per-pool monotonic counter with assertion. |
| Downstream schema drift breaks the prefix-lookup cache | Medium | Medium | R3 registry CI already locks downstream schemas. Gateway re-reads `tools/list` on reconnect. |
| Audit write perf hits the hot path | Low | Medium | Audit logs are buffered line-writes to `state/gateway-audit/`; fsync only on session end. |
| In-process `ix_governance::check` creates a circular dep | Low | Low | `ix-governance` is a library crate with no server-side imports; the gateway depends on it, not on `ix-agent`. |
| Windows file locks on audit log under concurrent sessions | Low | Medium | One log file per session id; no cross-session contention. |
| Gateway binary locks itself on rebuild (same WDAC memory issue) | Certain | Low | Build to `target/release-next/release/` when the running gateway holds `target/release/`. Standard workaround. |
| `fail_on_warning=true` surprises users with hard blocks | Medium | Medium | Default to `false` in the shipped config. Flip to `true` only after a two-week soak. |
| Pass-through mode becomes a permanent backdoor | Low | High | Audit log still records *that pass-through was on* in a startup preamble line, so after-the-fact review can see the gap. |

## Open questions (resolve before Phase 1)

1. **Which crate owns the audit log schema?** Candidate: extend
   `ix-governance::audit` with a new `GatewayAuditRecord` struct, or
   define it inside `demerzel-gateway::audit`. Prefer the former so
   other consumers can read gateway logs with the governance crate's
   existing de/serializer.

2. **Do we version the prefix scheme?** `ix__stats` is currently
   hardcoded to double underscore. Should the prefix separator be a
   config field? Decision: **no** for v1. Double underscore is the
   convention; config fields are a forever commitment.

3. **Does the gateway propagate downstream errors verbatim or wrap
   them?** Candidate: wrap them in `{downstream_error: {...}}` so
   upstream clients can distinguish gateway failures from downstream
   failures. Decision needed: adds one layer of unwrapping but makes
   failure attribution tractable.

4. **Where does the gateway's own MCP protocol version get pinned?**
   The gateway speaks MCP to the upstream and MCP to downstreams — in
   principle at different protocol versions. For v1, assume they
   match (MCP 2025-11-05 or whatever the shipped version is) and fail
   loudly if a downstream declares a different version in its
   handshake.

5. **Capability registry vs live tools/list.** The
   `capability-registry.json` is the static index; live `tools/list`
   is the runtime source of truth. These can diverge. Decision: the
   gateway trusts live `tools/list` exclusively and treats the
   registry as a bootstrap hint only. `ix_federation_discover` keeps
   using the registry (it's a discovery tool, not a routing layer).

6. **Does R4 block on an MCP SDK for Rust?** The gateway needs to be
   both an MCP server and an MCP client. Current `ix-agent` is only
   a server. We need a minimal client implementation (stdio
   framing, JSON-RPC envelope, request/response correlation).
   Candidate: write it ourselves in ~200 LOC inside
   `demerzel-gateway::downstream`. Pulling in `rmcp` or similar is
   tempting but adds dependency surface. Decision: **write our own**,
   reuse the `ServerContext` pattern from `ix-agent`.

## Exit from R4

R4 is done when:

1. `cargo test -p demerzel-gateway` is green on `main`, all five
   phases worth of tests included.
2. `examples/canonical-showcase/04-catia-bracket-generative/via-
   gateway/` runs end-to-end and matches the direct-ix path bit-for-
   bit on output.
3. `docs/MANUAL.md` has a new `§13 Gateway` section describing how
   to point `.mcp.json` at the gateway binary and how to read the
   audit log.
4. `.mcp.json` in the showcase is rewritten to point at the gateway
   as its **only** entry.
5. The regression gate for audit coverage passes on every showcase
   run.

After exit: R5 (Arrow IPC side-channel) and R6 (adversarial loops) can
start, both of which assume the gateway's audit + routing layer exists.

## Out of scope for this doc

- WebSocket / HTTP transport for remote gateway deployments. Follow-on
  R4.1, gated on a real multi-machine use case.
- Fine-grained AuthN/AuthZ. Single-user assumption holds for R4.
- Hot reload of the downstream list. Gateway restart is cheap enough
  that reloading config requires a restart in v1.
- Multi-tenant session isolation. One gateway process per user.
- A GUI for audit log browsing. JSONL + `jq` is enough for v1.
