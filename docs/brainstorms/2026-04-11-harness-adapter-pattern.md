# 2026-04-11 — Harness Adapter Pattern

**Status:** Design doc, partial implementation.
**Companion:** `demerzel/docs/brainstorms/2026-04-11-path-c-and-memory-spaces.md` (governance synthesis), `ix/docs/brainstorms/2026-04-11-triage-session-scenario.md` (in-process end-to-end scenario).
**Triggered by:** Martin Fowler's "Harness Engineering" framing applied to the Phase 1 Path C substrate.

## Thesis

The harness substrate we built for AI agent governance (SessionLog coordination, Belnap-extended CRDT merge, hexavalent escalation gate, trace flywheel) is directly applicable to **cross-repo harness engineering**: surrounding code deployments, CI runs, diagnostic tooling, and governance signals with the same middleware chain that surrounds LLM tool calls.

A **harness adapter** is any component that reads an external tool's native output (a `diagnose_and_remediate` markdown dump, a GitHub Actions run summary, a Prometheus alert JSON, a security scan report) and emits `SessionEvent::ObservationAdded` records into the shared substrate.

Once a tool has a harness adapter, its signals merge through the same G-Set CRDT as every other source. Contradictions between CI and security scan? Synthesized automatically. Conflict between a tars diagnosis and an ix execution trace? Synthesized automatically. The governance layer is uniform; the producers are many.

## What we already have (Phase 1 Path C Steps 1–5)

| Primitive | Crate | Role in the adapter pattern |
|---|---|---|
| `SessionEvent::ObservationAdded` | `ix-agent-core` | The universal carrier type |
| `HexObservation` struct + merge | `ix-fuzzy::observations` | The G-Set merge semantics |
| `projection::events_to_observations` | `ix-agent` | The SessionEvent → HexObservation projection |
| `ix_triage_session prior_observations` | `ix-agent` | The consumption endpoint |
| `session-event.schema.json` | `demerzel/schemas` | The authoritative wire format |
| `hex-merge.md` | `demerzel/logic` | The governance spec for merging |

**What's missing: the producer side.** Nothing exists today that takes an external tool's output and emits `observation_added` records. That's the harness adapter.

## The adapter contract

Every adapter, regardless of source, must satisfy:

1. **Read native format** — whatever the source produces (JSON, markdown, XML, logs, stdout)
2. **Validate against a source-specific schema** — each adapter owns the schema for its source
3. **Map to HexObservation shape** — apply a documented projection rule per source
4. **Emit as JSONL matching `session-event.schema.json`** — stdout, a file, or an installed SessionLog
5. **Content-hash the native input as the `diagnosis_id`** — deterministic correlation across rounds
6. **Stamp `source`, `round`, `ordinal` correctly** — these are the CRDT dedup key fields
7. **Be a pure function from (input_bytes, round) → Vec<SessionEvent>** — no global state, trivially testable

The last point is critical: adapters must be deterministic. Two independent runs on the same input must produce identical output (modulo wall-clock fields, which adapters should avoid). This is what makes the CRDT merge idempotent across re-ingestions.

## Three deployment shapes

### Shape 1 — Native Rust library

The adapter is a Rust module that lives in a crate named `ix-harness-<source>`. It takes `&[u8]` or `&str`, returns `Result<Vec<SessionEvent>, AdapterError>`. Callers link the crate and call the function directly.

**Use when:** the source is already emitted by Rust code OR consumed by Rust code. Maximum performance, compile-time type safety.

**Example:** `ix-harness-cargo` that reads `cargo test` JSON output and emits observations.

### Shape 2 — CLI binary (language-agnostic)

The adapter is a standalone executable. Reads native input from a file or stdin, writes SessionEvent JSONL to stdout or a file. External tooling invokes it as a pipe stage.

```bash
tars diagnose_and_remediate | ix-harness-tars --round 3 > session.jsonl
cat session.jsonl | ix-mcp-triage --prior-observations -
```

**Use when:** the source is a non-Rust tool and you don't want to write a native adapter library. Zero foreign-function-interface cost.

**Example:** `ix-harness-github-actions` that reads a GitHub Actions workflow run JSON and emits observations.

### Shape 3 — Foreign-language library

The adapter is a library in the source tool's native language (TypeScript, Python, Go). It writes SessionEvents directly without going through Rust.

**Use when:** the source tool is high-volume and you can't afford the CLI startup cost per event. Requires publishing the schema as a language-specific package.

**Example:** `@ix/harness-adapter` npm package that tars's `diagnose_and_remediate` TypeScript handler uses to emit observations inline.

## Projection rules are governance artifacts

The most important design decision: **the projection from native format to HexObservation is a governance question, not a code question.** Different sources have different epistemological shapes, and choosing which variant and weight to emit for each native signal is a policy decision.

For tars:
- Diagnostic signal `overall_health_score < 50%` → `<focus>::valuable` = `D`, weight 0.7
- `gpu_util > 95% sustained 60s` → `<gpu_service>::safe` = `F`, weight 0.8
- `disk_free < 1GB` → `<cleanup_action>::valuable` = `T`, weight 0.9

These mappings live in Demerzel under `logic/harness-<source>.md`. The Rust adapter is an *implementation of* the projection spec. This is the same pattern as `hex-merge.md` owning the merge rules while `ix-fuzzy::observations` implements them.

Every new adapter ships with:
1. A Demerzel governance doc describing the projection rules and their justification
2. A Rust/TS/Python implementation of the spec
3. Round-trip tests: "given this canonical input, the adapter emits these observations"
4. An entry in a `harness-adapter-catalog.md` index

## The trust model (honest about the gap)

**Today:** observations are trusted by `source` field at face value. A `source: "tars"` entry is trusted because we wrote tars.

**For internal repos:** this model is fine. Shared ops, shared keys, shared deploy infra. If the tars binary is compromised, bigger problems than the harness.

**For external repos:** trust-by-field fails. Any adversary who can write to the SessionLog can forge `source: "trusted-ci"` observations and push the merge toward any desired conclusion.

**The fix (future work):** signed observations. Each source holds a private key; every observation carries a detached signature over the canonical serialization of its fields. The merge function drops observations whose signatures don't verify against a known public key registry. The registry is itself a governance artifact.

**Scope for this doc:** we specify the trust gap but don't fix it. First-party harness adapters (tars, ga, ix-observatory) ship without signatures; external adapters are out of scope until the signature layer exists.

## Proposed adapter catalog (starter set)

Ordered by value × feasibility:

| Adapter | Source | Shape | Status | Value |
|---|---|---|---|---|
| `ix-harness-tars` | tars `diagnose_and_remediate` JSON | CLI binary or TS library | Step 6 of Phase 1 Path C | **Highest** — closes the loop |
| `ix-harness-cargo` | `cargo test --format=json` output | Rust library | Design only | High — ix's own test substrate |
| `ix-harness-ga` | Guitar Alchemist governance events | Rust library (via the submodule) | Design only | High — ga is already Demerzel-aware |
| `ix-harness-github-actions` | GitHub Actions run summary JSON | CLI binary | Design only | Medium — depends on signing |
| `ix-harness-prometheus` | Prometheus alert webhook payload | CLI binary | Design only | Medium — external system, signing blocker |
| `ix-harness-semgrep` | Semgrep scan report | CLI binary | Design only | Medium — security signals |
| `ix-harness-sentry` | Sentry incident webhook | CLI binary | Design only | Low — mostly redundant with alerts |

**Starting point: `ix-harness-tars`.** It unblocks Phase 1 Path C Step 6 AND serves as the reference implementation for future adapters. One commit, two wins.

## Concrete next steps

### Step 6a — Demerzel governance doc for tars projection

`demerzel/logic/harness-tars.md` — specifies the projection rules from tars's `diagnose_and_remediate` output to `HexObservation` instances. Must cover:

- How to derive `claim_key` from tars's issue categories
- Which hexavalent variant to emit per severity level (CRITICAL → F, WARNING → D, INFO → U)
- Weight rules (remediation confidence × severity multiplier)
- Content-hashing the input to produce `diagnosis_id`
- Round number handling (caller-supplied, default 0)

### Step 6b — `ix-harness-tars` CLI binary

A new crate `crates/ix-harness-tars/` with:

- `src/main.rs` — stdin/file → stdout JSONL
- `src/lib.rs` — pure function `tars_to_observations(input: &str, round: u32) -> Result<Vec<SessionEvent>, AdapterError>`
- Tests: round-trip (canonical input → expected observations)

Binary CLI pattern first because:
1. Doesn't require touching tars's TypeScript source (faster to land)
2. Demonstrates the language-agnostic adapter shape (useful for future external repos)
3. Can be composed as a pipe stage in shell scripts AND called programmatically from ix's main-agent shuttle
4. TypeScript native adapter can ship later as an optimization

### Step 6c — Triage integration hook

Document the main-agent-shuttle pattern for passing adapter output into `ix_triage_session`:

```bash
# Round N diagnosis
tars diagnose --json > /tmp/tars-round-N.json

# Adapter projects to observations
ix-harness-tars \
  --input /tmp/tars-round-N.json \
  --round N \
  --source tars > /tmp/observations-round-N.jsonl

# Triage merges observations with its own plan
jq -s '{prior_observations: .}' /tmp/observations-round-N.jsonl | \
  ix-mcp-call ix_triage_session --params -
```

Three shell commands, no new MCP plumbing, full cross-repo observation merge.

### Step 6d — Round-trip tests with canonical fixtures

A test fixtures directory with:

- `fixtures/tars-healthy.json` → expected `[observations with T variant on ::safe aspect]`
- `fixtures/tars-critical-disk.json` → expected `[F observation on disk-cleanup::valuable]`
- `fixtures/tars-gpu-overheating.json` → expected `[F observation on gpu_service::safe]`

Each fixture is committed with its expected output. CI re-runs the adapter and asserts bit-identical output — catches projection-rule drift.

## Non-goals for this round

- **Performance.** Adapter throughput doesn't matter at ix scale (governance decisions are per-round, not per-microsecond).
- **Signing.** Documented as a gap, not fixed. Internal use only until the signature layer exists.
- **Hot-reload adapter definitions.** Each adapter is a fixed binary; updating the projection rules requires a rebuild. Fine for governance-scale change velocity.
- **Multi-adapter pipelines.** Each adapter is invoked independently; composition is the caller's problem. We don't build a "pipeline orchestrator" crate.
- **Adapter discovery.** Callers must know which adapter to invoke. No registry, no reflection. The catalog doc is the source of truth.

## Open questions

1. **Should ix-harness-tars live in ix, in tars, or in a new top-level directory?**
   Probably `ix/crates/ix-harness-tars/` — colocated with the substrate that consumes it, so schema evolution stays coupled. Revisit if we get 4+ adapters and want a `harness-adapters/` workspace.

2. **Should the CLI output be plain JSONL or a wrapped format?**
   Plain JSONL matching `session-event.schema.json`. Simpler to compose, easier to audit, directly appendable to any SessionLog file. No magic wrapper.

3. **Who owns round-number allocation?**
   The caller. Adapters take `--round N` as a required flag. This keeps the adapter stateless; round tracking is the orchestrator's job.

4. **What happens when the native format is corrupt?**
   Adapter returns `Err(AdapterError::ParseError)` with a clear message. No partial output. The caller decides whether to skip the round or halt.

5. **Should we hash the native input as `diagnosis_id` or hash the derived observations?**
   Hash the native input. That way two adapters on the same input produce the same `diagnosis_id`, which matches the "content-addressable diagnosis" story from the Path C brainstorm.

## Why this is the right direction

Three reasons to invest in the harness-adapter pattern rather than one-off shims per source:

1. **The merge math is solid.** CRDT-correct, all proof obligations verified, tested. Adding producers doesn't risk the core.
2. **The schema is authoritative.** SessionEvent's governance lives in Demerzel; adapters implement that spec. Schema evolution has one owner.
3. **Each adapter has a clear boundary.** Read input → validate → project → emit. No creeping scope, no hidden state, no cross-adapter coupling.

Harness engineering across repos is a 10-year problem. The substrate we built for 2 repos (ix, tars) over 2 sessions happens to be shaped like the substrate needed for N repos. That's not luck — it's what happens when you pick CRDT + schema-first design for a coordination problem.

## Version

1.0 — 2026-04-11 — initial design. Starter catalog, tars-first implementation plan, trust-gap documented.
