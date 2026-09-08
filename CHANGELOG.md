# Changelog

All notable changes are tracked in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses workspace-unified semver (all crates share one version).

## [Unreleased]

### Added — `dark-features` doctor check (2026-09-08)

- `ix doctor` gains a `dark-features` check (`crates/ix-skill/src/doctor/dark_features.rs`,
  ix#315): a cargo feature that **no workspace member enables** is never built by
  `cargo {build,clippy,test} --workspace` — the three invocations CI runs — so the
  code behind it is neither compiled nor tested, and a skipped crate adds zero to
  both counters, so nothing in the output says so. The dark set is derived from
  `cargo metadata` (declared features minus the resolved set for the default
  workspace build), never hardcoded.
- The check reports **size, not just existence**: crate, feature, gated modules,
  lines and tests behind the gate. It fails only when a dark feature is
  significant (at least one test, or 100+ lines) and has no allowlist entry — so
  meta-features, implicit optional-dependency features and two-line stubs fall
  out as noise with no special-casing.
- `state/registry/dark-features.allow.json`: reasoned exemptions in the shape
  `orphan-traits.allow.json` established — a reasonless entry fails, an entry
  whose feature is no longer dark is reported stale. `kind` is `environment`
  (a native toolchain, an external binary, a toolchain newer than the workspace
  MSRV) or `tracked` (nothing environmental stops it, it is just not wired up —
  requires an `issue` and is counted as debt in the summary).
- Baseline at landing: **17 dark features, 13,719 lines and 219 tests that CI
  has never compiled**, 10 of them significant. The motivating case is
  `ix-code/topology` — 296 lines and five tests, merged and reviewed, never
  built once; the code is healthy (`cargo test -p ix-code --features full`
  passes), which is exactly why nobody noticed.
- Checklists: "Adding a feature-gated module" in `docs/CHECKLISTS.md`
  ([FR](docs/fr/CHECKLISTS.md)).

### Added — Stable Surface guard (2026-05-17)

- `crate-maturity.toml` workspace-root file: single source of truth mapping
  every crate to one of `stable` / `beta` / `experimental` / `internal`.
- `ix stable-surface` CLI subcommand: prints each Stable crate's public-API
  hash (BLAKE3 over sorted `pub` declarations). `--all-tiers` includes the
  rest. JSON output is the wire format for CI.
- `.github/workflows/stable-surface.yml`: diffs the report between `main`
  and the PR. Stable hash changes → fail. Non-stable hash changes → warn.
- README "Stability contract" section documenting the demotion escape
  hatch.
- Unit tests in `ix-skill::verbs::stable_surface::tests` cover pub-line
  extraction, internal-vs-external change detection, and the diff
  partitioning across tiers (4 tests, all green).

### Added — Phase 1 delivery (Weeks 1–8)

**New foundation crates**

- `ix-types` — shared value lattice (`Value` / `SocketType` / `Hexavalent` /
  `FromValue` / `IntoValue`) with `IxVector` / `IxMatrix` ndarray newtypes.
  Six-valued hexavalent logic (T/P/U/D/F/C) with involutive NOT, De Morgan
  OR/AND, and wire-format symbols matching Demerzel's
  `hexavalent-state.schema.json`.
- `ix-registry` — link-time capability registry via
  `#[linkme::distributed_slice]` with a Windows-LLD workaround sentinel.
  Public API: `by_name` / `by_domain` / `search` / `all` / `count` /
  `invoke`.
- `ix-skill-macros` — `#[ix_skill(domain=..., governance=..., name=...,
  schema_fn=...)]` proc-macro that generates adapter fns and registers
  descriptors at link time.

**43/43 MCP tools migrated to registry**

- `ix-agent` ships three batches (`skills/batch1.rs` / `batch2.rs` /
  `batch3.rs`) covering every one of the 43 pre-existing MCP tools.
  Hand-written handlers retained under `handlers.rs`; wrappers delegate
  to them while registering via `#[ix_skill]`.
- `ToolRegistry::merge_registry_tools` drops manual entries colliding
  with registry names, guaranteeing a single source of truth per MCP
  tool name.
- Parity test (`crates/ix-agent/tests/parity.rs`) enforces strict
  43-name equality between pre- and post-migration exposure.

**New 7-verb `ix` CLI grammar**

The old 5-subcommand CLI was replaced with a noun-verb grammar:

- `ix run <skill> [--input-file ...] [--input ...]` — invoke any
  registered skill with JSON input from stdin / `--input` / `--input-file`.
- `ix list {skills,domains,personas}` — discover capabilities with
  `--domain` and `--query` filters.
- `ix describe {skill,persona,policy} <name>` — introspect signature,
  schema, governance tags, or governance artifact.
- `ix check {doctor,action}` — environment diagnostics + constitutional
  compliance review of a proposed action.
- `ix beliefs {show,get,set,snapshot}` — hexavalent belief state
  management under `state/beliefs/*.belief.json` and
  `state/snapshots/{YYYY-MM-DD}-{slug}.snapshot.json`.
- `ix pipeline {new,validate,dag,run}` — scaffold, validate, render,
  and execute `ix.yaml` pipelines with NDJSON event streaming.
- `ix serve {mcp,repl}` — stub pointing at the `ix-mcp` binary for now.

Global flags: `--format {auto,table,json,jsonl,yaml}` (TTY-autodetect),
`--quiet`, `--verbose`, `--no-color`.

**Hexavalent process exit codes**

`ix check` maps verdicts to exit codes for CI integration:

| Exit | Symbol | Meaning |
|-----:|:------:|---|
| 0 | T | True — proceed autonomously |
| 1 | P | Probable — proceed with note |
| 2 | U | Unknown — gather evidence |
| 3 | D | Doubtful — hold action |
| 4 | F | False — do not proceed |
| 5 | C | Contradictory — escalate |
| 10 | — | Runtime error |
| 64 | — | Usage error |

**`ix.yaml` pipeline format + `ix.lock` reproducibility manifest**

- `PipelineSpec` YAML schema with `version`, `params`, `stages`, and
  opaque `x-editor` metadata. Stages reference registry skills by name
  with free-form `args` JSON and `deps` for dependency ordering.
- `{"from": "stage[.path]"}` references inside `args` let downstream
  stages consume upstream outputs (dotted path-walking supported).
- `lower()` validates skill names against `ix-registry`, checks for
  cycles, and produces an executable `Dag<PipelineNode>`.
- `ix.lock` written alongside `ix.yaml` on every `ix pipeline run`
  (write-only in Phase 1; verification deferred). Uses stable
  canonical-JSON hashing (FNV-1a 64-bit) so structurally equivalent
  args produce identical hashes.

**Visual pipeline editor (`ix-demo` → Pipeline tab)**

An `egui_snarl`-backed DAG editor with:

- 10 typed node variants: CsvRead, CsvWrite, Constant, Normalize, KMeans,
  LinearReg, FFT, PolicyGate, Belief, Plot, plus a generic `Skill`
  carrier for YAML-imported stages.
- 8 typed sockets (Any / Scalar / Vector / Matrix / Dataset / Model /
  Belief / Text) with distinct colors and compatibility-plus-widening
  rules (`Scalar→Vector`, `Vector→Matrix`, `Any↔*`).
- Per-frame palette-search filter on the right-click graph menu.
- Wire-aware live execution: topological sort + per-node invocation of
  the registered skill with upstream outputs merged into args by socket
  name. Status dots (green/red/grey) render on every node header.
- Collapsible results panels — one for Snarl execution (keyed by
  NodeId), one for `ix.yaml` execution (keyed by stage id).
- Static validation flags ML nodes missing an upstream PolicyGate.
- JSON round-trip (canonical Snarl format) + YAML export (Snarl → ix.yaml).
- **YAML import**: reconstructs the Snarl graph from any `ix.yaml`
  using generic Skill nodes whose sockets come from the registry.
- **Run ix.yaml**: executes an `ix.yaml` in-place via `ix-pipeline`
  and shows per-stage outputs — closes the authoring loop.
- Interactive 2x3 hexavalent quadrant selector on Belief nodes with
  color-coded T/P/U/D/F/C cells.

**Governance CLI wrappers**

- `ix describe persona <name>` — full YAML-loaded persona details
  (role, capabilities, constraints, voice, interaction patterns).
- `ix describe policy <name>` — tries `<name>-policy.yaml` then
  `<name>.yaml`, emits full `Policy.extra` payload.
- `ix beliefs set <key> <proposition> --truth T|P|U|D|F|C --confidence N`
  with hexavalent validation and [0,1] clamping.
- `ix beliefs snapshot <description>` — captures every belief file
  into a timestamped snapshot conforming to Demerzel's
  `reconnaissance-profile.schema.json`.

**Workspace hygiene**

- `cargo clippy --workspace -- -D warnings` passes clean.
- ~150 new tests across 6 new/modified crates:
  - 12 `ix-types` + 12 `ix-registry` + 10 smoke tests
  - 29 `ix-pipeline` (14 new: spec, lower, lock)
  - 7 `ix-agent` (43-tool parity)
  - 21 `ix-skill` integration (12 cli + 9 pipeline + 9 governance)
  - 34 `ix-demo` pipeline-editor unit tests
- Three force-link declarations (`ix-skill`, `ix-demo`, `ix-agent/lib`)
  document the LTO dead-code-stripping workaround for
  `linkme::distributed_slice`.

**Showcase artifacts** (`examples/showcase/`)

- `pipeline.yaml` — 5-stage diamond DAG (stats + fft + 2× number_theory
  → governance.check leaf).
- `ml-classification.yaml` — 3 classifiers compared + deployment gate.
- `signal-chain.yaml` — FFT + Lyapunov + envelope stats under
  constitutional review.
- `demo.sh` — 9-step bash walkthrough in a self-cleaning tempdir.
- `README.md` — narrative walkthrough with expected output.

### Changed

- `Tetravalent` (4-valued) → `Hexavalent` (6-valued) throughout the new
  crates to match the ecosystem-wide convention in
  `governance/demerzel/logic/hexavalent-logic.md`.
- `ix-skill` CLI was fully rewritten; the old 5-command structure
  (optimize/train/cluster/grammar/list) is replaced by the 7-verb
  grammar (no back-compat aliases — pre-1.0 API).

### Added workspace dependencies

- `linkme = "0.3"`, `schemars = "0.8"` for registry infrastructure
- `syn = "2"`, `quote = "1"`, `proc-macro2 = "1"` for the skill macro
- `egui-snarl = "0.7"` for the visual pipeline editor
- `assert_cmd = "2"`, `predicates = "3"` for CLI integration tests

### Three new workspace members

```
crates/ix-types/          ← ~400 LOC
crates/ix-registry/       ← ~250 LOC
crates/ix-skill-macros/   ← ~300 LOC
```

Plus substantial additions to `ix-skill`, `ix-pipeline`, `ix-agent`,
`ix-demo`, and new modules under each.

## Prior releases

No public release tags yet — this is the first CHANGELOG entry.
