# ix — Agent guide

This repository's conventions, build/test commands, and collaboration discipline are documented in **[CLAUDE.md](CLAUDE.md)** at the repo root. AI tools that look for `AGENTS.md` (Codex, Gemini, others) should read CLAUDE.md as the source of truth.

## Cross-repo context

`ix` is one of three repositories in the GuitarAlchemist ecosystem:

| Repo | Stack | Role |
|---|---|---|
| `ix` | Rust workspace (35 crates) | ML/math primitives, governance, structural code analysis, MCP server |
| [`ga`](https://github.com/GuitarAlchemist/ga) | .NET 10 / C# / React / F# DSL | Music theory engine, OPTIC-K embedding, chatbot, web app |
| [`tars`](https://github.com/GuitarAlchemist/tars) | F# | Grammar evolution, metacognition, knowledge graph |

Plus a shared submodule: `governance/demerzel` carries the constitution, policies, IxQL pipelines, and persona definitions that all three repos enforce.

## Cross-repo contracts ix participates in

When changing any of these files, treat the change as a one-way door (per [CLAUDE.md](CLAUDE.md) collaboration discipline) and verify the consumer side hasn't drifted:

| Contract | Producer | Consumer | Schema |
|---|---|---|---|
| `optick-sae-artifact` | `crates/ix-optick-sae` (this repo) | `ga/Apps/GaQaMcp/Tools/qa_score_quality_drift` | `ga/docs/contracts/optick-sae-artifact.schema.json` |
| `qa-verdict` | Demerzel `qa-architect-cycle.ixql` + `crates/ix-blast-radius` | (consumer in flight, see `docs/plans/2026-05-04-chatbot-autonomy-action-layer.md`) | `ga/docs/contracts/qa-verdict.schema.json` |
| `voicings.payload.v1` | `crates/ix-agent` (`ix_voicings_payload`) | `ga` Prime Radiant viewer | `governance/demerzel/schemas/contracts/voicings-payload-v1.schema.json` |
| `chatbot-baseline` (planned) | `ga` GaChatbot.Api | `crates/ga-chatbot` coverage tool | TBD per `docs/plans/2026-05-04-chatbot-autonomy-action-layer.md` |

The capability registry that ties producers to consumers across all three repos lives at `governance/demerzel/schemas/capability-registry.json`.
