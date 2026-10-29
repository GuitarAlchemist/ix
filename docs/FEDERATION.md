# Cross-repo federation: ix + tars + ga

How the three GuitarAlchemist repos compose through MCP + the ix
capability registry.

## The three repos

| Repo | Language | Role |
|---|---|---|
| **ix** | Rust | ML algorithms + governance substrate (this repo) |
| **tars** | F# | Cognition — grammar weighting, pattern promotion, metacognition |
| **ga** | C# | Music theory — chord analysis, scales, progression features |

Each repo publishes an MCP server exposing its capabilities as typed
tools. The three servers are registered together in `.mcp.json` and
discoverable via the capability registry at
`governance/demerzel/schemas/capability-registry.json`.

## How a federated call flows

```
                 ┌──────────────────┐
  user / agent → │  MCP client      │  (Claude, Cursor, custom)
                 └────────┬─────────┘
                          │ tools/call
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
       ┌───────┐     ┌───────┐     ┌───────┐
       │  ix   │     │ tars  │     │  ga   │
       │ (Rust)│     │  (F#) │     │  (C#) │
       └───────┘     └───────┘     └───────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                ┌────────────────────┐
                │ Demerzel governance│
                │  (submodule in ix) │
                └────────────────────┘
```

Any MCP client sees the union of all three servers' tools. Each tool
call is dispatched to the owning server, runs in that server's
runtime, and returns JSON.

## The bridge tools

`ix` ships three bridge tools in `ix-agent` that pre-format ix's
outputs for consumption by the other repos:

- **`tars_bridge`** — prepares trace stats / patterns / grammar
  weights for TARS ingestion. Actions: `prepare_traces`,
  `prepare_patterns`, `export_grammar`.
- **`ga_bridge`** — converts GA music theory data into ML-ready
  feature vectors. Actions: `chord_features`, `progression_features`,
  `scale_features`, `workflow_guide`.
- **`federation.discover`** — queries the capability registry to find
  which server owns a given capability, filtered by domain or query.

The bridge tools are themselves registry tools — callable from
`ix run ga_bridge`, embeddable in `ix.yaml` stages, and invokable from
agents.

## Federated pipeline pattern

A three-stage federated flow:

```yaml
# Step 1: GA emits chord-features for a progression
chord_vectors:
  skill: ga_bridge
  args:
    action: chord_features
    chords: ["Cmaj7", "Am7", "Dm7", "G7"]

# Step 2: ix clusters the feature vectors
harmonic_clusters:
  skill: kmeans
  args:
    data:
      from: chord_vectors.features   # cross-stage data flow
    k: 2
    max_iter: 50

# Step 3: TARS promotes the harmonic pattern to the grammar
promote_pattern:
  skill: tars_bridge
  args:
    action: prepare_patterns
    min_frequency: 3
  deps: [harmonic_clusters]
```

The `{from: chord_vectors.features}` reference is resolved by
ix-pipeline's `lower()` at execution time, so GA's output flows
straight into ix's clustering without any code in between.

## Belief + governance propagation

Demerzel's governance artifacts live in `governance/demerzel/` as a
git submodule. All three repos consume the same constitution + policies:

- **Belief state** is hexavalent (T/P/U/D/F/C) per
  `governance/demerzel/logic/hexavalent-logic.md`. Propagates across
  repo boundaries through JSON wire format with single-letter symbols.
- **Confidence thresholds** come from
  `governance/demerzel/policies/alignment-policy.yaml`:
  ≥0.9 autonomous · ≥0.7 with note · ≥0.5 confirm · ≥0.3 escalate.
- **Personas** are shared YAML files in
  `governance/demerzel/personas/*.persona.yaml` — any repo can load a
  persona by name and apply its affordances.
- **Audit trails** from `governance.check` calls conform to the same
  schema across repos, enabling cross-repo incident replay.

## Galactic Protocol

The three repos communicate via the Galactic Protocol (see
`governance/demerzel/contracts/`):

- **Directives flow down** from Demerzel to consumer repos
  (compliance requirements, policy updates, reconnaissance requests).
- **Belief snapshots flow up** from consumer repos to Demerzel
  (state/snapshots/*.snapshot.json).
- **Knowledge packages** (from Seldon) flow peer-to-peer for
  cross-repo learning transfer.

## Discovery example

```bash
# What music-theory capabilities exist across the ecosystem?
ix run federation.discover --input '{"domain":"music-theory"}'

# Fuzzy search for anything grammar-related:
ix run federation.discover --input '{"query":"grammar"}'
```

## Status (2026-04)

- **ix**: 43 tools across 29 domains, full registry + CLI + visual editor
- **tars**: `ingest_ga_traces`, `run_promotion_pipeline`, grammar weighting
- **ga**: chord/scale/progression analysis, trace export
- **Demerzel**: 11-article constitution + 39 policies + 14 personas

The ix bridge tools (`tars_bridge`, `ga_bridge`, `federation.discover`)
produce the data shapes that TARS and GA expect — they are the
narrow-waist interfaces between the three runtimes.

## See also

- `examples/showcase/advanced/music-theory.yaml` — uses ga_bridge for
  3 different feature extractions
- `governance/demerzel/schemas/capability-registry.json` — the
  canonical cross-repo capability index
- `governance/demerzel/contracts/` — Galactic Protocol message schemas
- `.mcp.json` — local MCP server registration for ix + tars + ga
