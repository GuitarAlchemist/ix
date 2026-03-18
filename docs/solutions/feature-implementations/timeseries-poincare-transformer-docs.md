---
title: "Time series module, Poincaré hierarchy extraction, transformer & time series tutorials (EN+FR)"
category: feature-implementation
date: 2026-03-17
tags: [time-series, poincare, hyperbolic-embeddings, hierarchy, transformer, tutorials, bilingual-docs]
components:
  - crates/ix-signal   # timeseries module
  - crates/ix-math     # poincare_hierarchy module
  - docs/neural-networks  # transformer tutorial
  - docs/signal-processing  # time series tutorial
  - docs/fr  # French translations
related_prs: [17, 18]
---

# Time Series + Poincaré Hierarchy + Transformer Docs

## What Was Built

### 1. Time Series Module (`ix-signal/timeseries.rs`) — PR #17

11 functions for time-ordered data:

| Function | Purpose |
|----------|---------|
| `rolling_mean`, `rolling_std` | Sliding window statistics (O(n) incremental) |
| `rolling_min`, `rolling_max` | Sliding window extrema |
| `ewma` | Exponentially weighted moving average |
| `difference` | First/higher-order differencing (stationarity) |
| `pct_change` | Percentage returns |
| `lag_features` | Convert time series → supervised learning matrix |
| `lag_features_with_stats` | Lag features + rolling mean/std appended |
| `temporal_split` | Train/test split respecting time order |
| `expanding_mean` | Cumulative mean |

26 tests (13 unit + 13 doc). Key design: `rolling_mean` uses O(n) incremental sum, not O(n*w) naive.

### 2. Poincaré Hierarchy Extraction (`ix-math/poincare_hierarchy.rs`) — PR #18

Learns tree structure from edge lists using hyperbolic geometry:

| Component | Purpose |
|-----------|---------|
| `PoincareEmbedder` | Train embeddings via Riemannian SGD + negative sampling + radial positioning |
| `HierarchyDecoder` | Extract tree: `root()`, `depth_ranking()`, `infer_parents()`, `infer_tree()`, `leaves()` |
| `hierarchy_map_score` | Evaluate: do inferred parents match ground truth? |

Key design decisions:
- **Radial positioning** (not just attraction/repulsion): BFS computes depth from roots, then each node is pushed toward its target radius (depth 0 → 0.1, max depth → 0.9). This was the breakthrough that made tests pass — pure distance-based training didn't converge for hierarchical structure.
- **Analytical gradient**: closed-form ∂d/∂u for Poincaré distance, replacing numerical finite-difference (which was too noisy).

12 tests (11 unit + 1 doc).

### 3. Tutorials (EN + FR)

| Tutorial | EN | FR |
|----------|----|----|
| Transformers | `docs/neural-networks/transformers.md` | `docs/fr/reseaux-neurones/transformers.md` |
| Time Series | `docs/signal-processing/time-series.md` | `docs/fr/traitement-signal/series-temporelles.md` |

Both include PSAP/first responder examples (response time monitoring with NFPA compliance).

## Issues Encountered

### Poincaré training convergence (fixed)

Initial approach (numerical gradient + distance attraction/repulsion) failed to produce correct hierarchy. Root nodes ended up further from origin than leaves.

**Root cause**: Poincaré distance gradients via finite differences were too noisy (high curvature near boundary), and pure distance-based attraction doesn't encode directionality (parent vs child).

**Fix**: Two changes:
1. Analytical Poincaré distance gradient (exact, not approximated)
2. Radial positioning: BFS from roots computes node depth, then each node is explicitly pushed toward `radius = 0.1 + 0.8 * depth/max_depth`

This makes the hierarchy explicit in the geometry rather than hoping it emerges from distance optimization alone.

## Prevention Strategies

- **Hyperbolic ML requires radial structure**: distance-only objectives don't encode hierarchy. Always include a radial component that encodes depth.
- **Use analytical gradients in curved spaces**: numerical gradients fail in high-curvature regions of hyperbolic space. Derive the closed form.
- **Test with known hierarchies**: always include a test with a known tree where you can verify root detection and parent inference.

## Cross-References

- [ML toolkit expansion compound doc](./ml-modules-tutorials-and-i18n.md)
- [ix-math/hyperbolic.rs](../../crates/ix-math/src/hyperbolic.rs) — foundation (distance, Möbius ops, exp/log maps)
- [PR #17](https://github.com/GuitarAlchemist/ix/pull/17), [PR #18](https://github.com/GuitarAlchemist/ix/pull/18)
