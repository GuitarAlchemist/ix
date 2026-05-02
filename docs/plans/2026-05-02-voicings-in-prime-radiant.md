---
date: 2026-05-02
reversibility: one-way-door (schema published to GA + governance registry)
revisit-trigger: GA users want to interact with voicings beyond view-and-pan, OR voicing-positions schema needs a breaking change, OR Prime Radiant adds a second non-governance dataset (need a generic loader pattern)
status: phase 1 shipped 2026-05-02 (ix_voicings_payload MCP tool + serve_viz --bind); phases 2-4 pending GA-side work
---

# Push the voicing cloud to GA's Prime Radiant in a separate scene area

## Problem

The local 3D viewer (`ix-voicings serve_viz /3d`, shipped 2026-05-02 in 209c9f4) renders all 688K voicing positions in a Three.js Points cloud. It works, but it lives only on `localhost:8765` — invisible to the rest of the GA ecosystem and to the user when they're in the GA app. Meanwhile GA has Prime Radiant — a 3D scene that already renders the 196-node governance graph at origin and is remote-controllable via MCP→SignalR (per `reference_ga_remote_ui_control.md` memory).

The user's ask (2026-05-02): push the voicing cloud into Prime Radiant **in a separate area of the scene** so it sits next to the governance graph rather than on top of it. Two clouds, one viewer, switchable focus.

## Background facts that shape the design

### 1. The 3D positions are degenerate by cluster

`viz_precompute.rs` produces `voicing-layout.json` with one 3D position per voicing, but the cluster precompute it inherits is degenerate: 99.94% of guitar voicings land in `guitar-C0`, 96.6% of bass in `bass-C0`, 95.2% of ukulele in `ukulele-C0`. Positions sit in tight knots around each cluster centroid, with sub-1-unit jitter. At default zoom this collapses to ~15 sub-pixel blobs.

The local viewer mitigates with a runtime "spread" slider (triangular noise per axis). Prime Radiant should support the same control, OR we should separately address the upstream cluster-imbalance bug in viz_precompute / its dependencies. **Recommendation:** ship the spread control in Prime Radiant for now, file a separate task to investigate the cluster precompute. Do not pre-jitter the data on disk — that loses the original signal.

### 2. Only ~7K of 688K voicings have detail metadata

`viz_precompute.rs:337` caps `voicing-details.json` at `DETAIL_SAMPLE_CAP = 1800` per cluster via stride sampling. With 15 wildly-imbalanced clusters this yields ~7,141 entries — confirmed by measurement and by re-deriving the math:

| Cluster | Voicings | Stride | Sampled |
|---|---|---|---|
| guitar-C0 | 666,712 | 370 | ~1,801 |
| guitar-C1..4 | 413 total | 1 | 413 |
| bass-C0 | 12,184 | 6 | ~2,031 |
| bass-C1..4 | 430 total | 1 | 430 |
| ukulele-C0 | 8,197 | 4 | ~2,049 |
| ukulele-C1..4 | 415 total | 1 | 415 |
| **total** | | | **~7,139** |

Comment in source: *"Stride-based sampling keeps deterministic global_ids aligned with what the renderer is likely to show. Full 688k details would be ~87MB; sampled file lands around 3MB."*

This means the local click-region panel says "no metadata in this region" for ~98% of voicings. **Implication for Prime Radiant:** if the user clicks a voicing in the cloud, we cannot serve full chord details from the precomputed file alone. Two options:

- **(a)** Lazy-fetch from `state/voicings/{instrument}-corpus.json` (full 688K, ~50MB per instrument) on click — needs a new MCP tool or HTTP endpoint
- **(b)** Raise `DETAIL_SAMPLE_CAP` and accept a larger detail file (10K cap → ~40K entries, ~6MB)
- **(c)** Ship without metadata for v1 and add it after we know what users actually click

**Recommendation:** **(c) for v1**, then **(a)** as the proper fix. Avoid (b) — it solves nothing structurally, just shifts the cap.

### 3. Prime Radiant is a single Three.js scene, not a multi-app shell

Per `project_prime_radiant_integration.md`, the governance graph already renders at origin. Adding a second dataset means either:

- A second `THREE.Group` with a translation offset (cheap, what the user asked for)
- A scene-switcher with two cameras (harder, not asked for)

**Recommendation:** translation offset. Render voicings at `[200, 0, 0]` or wherever the bounds clear the governance graph by ≥3× its diameter. Add a HUD toggle "focus voicings / focus governance / show both" that re-targets the camera.

## Proposed approach

### Schema: shared between ix and ga

```json
// voicings.payload.v1
{
  "schema": "voicings.payload.v1",
  "format": "binary-positions+meta",
  "positions_url": "http://127.0.0.1:8765/data/voicing-positions.bin",
  "meta_url": "http://127.0.0.1:8765/data/voicing-positions.meta.json",
  "scene_offset": [200, 0, 0],
  "default_spread": 1.5,
  "default_point_size": 0.3
}
```

The binary format is exactly what the local viewer already consumes (`voicing-positions.bin` + `voicing-positions.meta.json` from serve_viz's pre-derivation). Prime Radiant fetches both URLs, applies the offset, renders.

### ix side

1. **New MCP tool** in `ix-agent`: `ix_voicings_payload(scene_offset?, default_spread?, default_point_size?)` returns the JSON payload above. Reuses the binary already on disk.
2. **Make serve_viz optionally bind to 0.0.0.0** (or document that the user starts it manually) so Prime Radiant running on the same host can fetch. Alternative: have the MCP tool inline the binary as base64 (~11MB base64 of an 8MB binary — wasteful but avoids the cross-process HTTP step).

### ga side (Prime Radiant)

1. **New MCP-callable action** `loadVoicingCloud(payload)` — adds a `THREE.Group` at the offset, fetches the binary, builds a single Points object (same approach as `crates/ix-voicings/web/3d.html`).
2. **HUD additions** — instrument toggles, point-size slider, spread slider, focus toggle (governance / voicings / both). The UI patterns from `web/3d.html` translate directly.
3. **Camera focus action** `focusOn(target)` for the focus toggle.

### governance side

1. Register the `voicings.payload.v1` schema in `governance/demerzel/schemas/capability-registry.json` so both peers know the shape.
2. Add a Demerzel constitutional check that `loadVoicingCloud` only accepts payloads matching the registered schema (defense against drift).

## Phases

| Phase | Scope | Estimate | Reversible? |
|---|---|---|---|
| 0 | This plan, sign-off | – | yes |
| 1 | ix side: MCP tool + serve_viz CORS/bind | 1 session | yes |
| 2 | ga side: `loadVoicingCloud` action + HUD | 1–2 sessions | yes |
| 3 | governance: register schema, write check | 1 session | **no** (schema is now public contract) |
| 4 | Cross-repo smoke test (ix MCP → ga SignalR → Prime Radiant render) | 0.5 session | yes |
| 5 (later) | Lazy-fetch metadata on click — addresses the 7K-of-688K detail gap | 1 session | yes |

**Order:** 1 → 2 → 4 (smoke test before locking schema) → 3.

## Open questions

- Is `serve_viz` the right host for the binary, or should `ix-agent` serve it directly? Bundling it into ix-agent removes a moving part but bloats the agent binary by ~8MB on disk every rebuild.
- Should Prime Radiant share the spread/jitter implementation with `web/3d.html` via a tiny vendored JS module, or duplicate? Duplicating is fine for v1; share if Prime Radiant gets a third dataset later.
- The `scene_offset` in the payload is a one-way door: once GA users learn that voicings live "to the right" of governance, moving it later is disorienting. Pick the offset deliberately.

## Non-goals for v1

- Editing voicings from Prime Radiant
- Per-voicing chord-name labels in the cloud (use click → metadata panel instead)
- Real-time updates as the corpus rebuilds (one-shot fetch is fine)
- Integration with the `ga_voicing_telemetry` / OPTIC-K v1.8 telemetry stream

## Cross-references

- Local viewer: `crates/ix-voicings/web/3d.html`, `crates/ix-voicings/src/bin/serve_viz.rs`
- Detail-cap source: `crates/ix-voicings/src/viz_precompute.rs:337`
- Governance schema dir: `governance/demerzel/schemas/`
- Memory: `project_prime_radiant_integration.md`, `reference_ga_remote_ui_control.md`
