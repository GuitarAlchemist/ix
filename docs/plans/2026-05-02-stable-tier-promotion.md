---
date: 2026-05-02
reversibility: one-way door (downstream consumers will rely on the Stable label)
revisit-trigger: any of the 4 promoted crates needs a breaking API change, OR external consumer reports the promotion was premature, OR the workspace adopts independent crate versioning
status: shipped 2026-05-02
---

# Promote ix-cache, ix-probabilistic, ix-game, ix-rl from Beta to Stable

Audited the 4 Beta crates that the README's own descriptions already characterised as "stable patterns" / "well-defined algorithms" / "solid math" and confirmed they meet the Stable bar:

> **Stable** — API commitments, well-tested, used in production MCP tools (downstream-safe)

## Audit results

| Crate | Tests | Production consumer | Public surface | Verdict |
|---|---|---|---|---|
| `ix-cache` | 27 | `ix-agent::handlers`, `ix-agent::ml_pipeline` (Cache, CacheConfig) | 4 modules + 7 typed re-exports | promote |
| `ix-probabilistic` | 14 | `ix-agent::handlers` (BloomFilter, HyperLogLog) | 4 modules: bloom, count_min, cuckoo, hyperloglog | promote |
| `ix-game` | 25 | `ix-agent::handlers` (nash::BimatrixGame), MCP tool `ix_game_nash` | 7 modules: auction, cooperative, evolutionary, grammar_replicator, mean_field, mechanism, nash | promote |
| `ix-rl` | 10 | `ix-agent::handlers` (EpsilonGreedy, UCB1, ThompsonSampling) | 4 modules: bandit, env, q_learning, traits | promote |

All 4 pass `cargo test` (76 tests total in the 4 lib + integration runs above). All 4 are reachable from MCP tools today, so they're already exercised by the canonical downstream consumer.

## What "Stable" means in this workspace (clarified during the audit)

The README's tier definitions are explicit enough but worth restating since this is the first promotion:

- **Public types are now contract.** Removing or reshape-breaking a `pub` item in any of these 4 crates is a breaking change; a `cargo update` from a downstream consumer must continue to compile.
- **Internal helpers should be `pub(crate)`.** The audit didn't surface any glaring leaks but a quarterly hygiene pass is cheap and worth doing.
- **The shared `version.workspace = true` constraint is unchanged.** Promotion is a README label, not a Cargo-enforced barrier — when the workspace bumps a minor version, every Stable crate's commitments still apply, and any breaking change in any Stable crate forces a major-version bump for the whole workspace.

## What stays Beta and why

| Crate | Reason for staying Beta |
|---|---|
| ix-nn | Transformer features still landing (dropout, GPU, mini-batch — recent) |
| ix-pipeline | DAG semantics still being refined |
| ix-agent | Public type surface entangled with ga/tars/demerzel evolution |
| ix-governance | Bridge to Demerzel which itself evolves |
| ix-io | Catch-all utility crate, surface still growing |
| ix-grammar | Catalog growing; new entries may add public types |
| ix-catalog-core | Consumers (ix-code, ix-grammar, ix-net) still proliferating |
| ix-net | RFC catalog still growing |

## Revisit triggers

- **Any breaking change** to a now-Stable crate. The promotion was based on today's API surface; a forced break means the audit was wrong and we need to either revert (back to Beta) or commit to a workspace major-version bump.
- **External consumer reports premature promotion.** ga, tars, or another downstream repo finding that "Stable" was misleading.
- **Workspace adopts independent crate versioning.** Switching off `version.workspace = true` would let Stable and Beta crates evolve at different cadences, at which point the audit semantics tighten.

## Cross-references

- README §Maturity Tiers: definitions
- README §Stable Subset: where the 4 now appear
- CLAUDE.md: "non-trivial decisions go in `docs/plans/YYYY-MM-DD-*.md`" — this doc satisfies that
