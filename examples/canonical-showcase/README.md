# IX Canonical Showcase

The flagship set of demos showing IX composing its 46 MCP tools against four distinct real-world problem classes. Every artifact in this folder was produced by orchestrating IX tools through a language model; none were hand-written by a traditional developer.

This folder serves three purposes:

1. **Marketing** — prove IX can handle real aerospace, cyber, finops, and signal-processing problems.
2. **Onboarding tutorial** — traverse the demos from 3-tool to 13-tool pipelines to learn the IX mental model.
3. **Regression harness** — each demo replays through `ix_pipeline_run` (after R1 lands) and fails CI on any bit-level drift.

See `ix-roadmap-plan-v1.md` for the roadmap that turns this ad-hoc collection into a CI-gated first-class project artifact.

## Tool × demo matrix

| Demo | Domain | Tools | Size | Reports |
|---|---|---|---|---|
| [cost-anomaly-hunter.html](cost-anomaly-hunter.html) | FinOps | `ix_stats`, `ix_fft`, `ix_kmeans` (3) | 9 KB | — |
| [cyber-intrusion-triage.html](cyber-intrusion-triage.html) | Security | 12 tools: `stats`, `fft`, `chaos_lyapunov`, `bloom_filter`, `kmeans`, `viterbi`, `markov`, `linreg`, `rf`, `topo`, `nash`, `bandit`, `gov_check` | 19 KB | — |
| [catia-bracket-generative.html](catia-bracket-generative.html) | Aerospace CAD | 13 tools: the full flagship pipeline (`stats`, `fft`, `kmeans`, `linreg`, `rf`, `adam`, `ga`, `topo`, `chaos`, `nash`, `viterbi`, `markov`, `gov`) | 29 KB | [v1 FR](rapport-bracket-genetique-a350.md) · [v2 FR](rapport-bracket-genetique-a350-v2.md) |
| [catia-bracket-3d.html](catia-bracket-3d.html) | Aerospace CAD | Three.js r160 PBR render of the optimized bracket with stress map | 15 KB | — |
| [catia-toolpath-3d.html](catia-toolpath-3d.html) | Aerospace CAD | Isometric 5-axis toolpath from the 32-step Viterbi output | 12 KB | — |
| [catia-bracket-context-a350.html](catia-bracket-context-a350.html) | Aerospace CAD | Schematic A350 side/top/pylon context | 17 KB | — |
| [catia-bracket-context-realistic.html](catia-bracket-context-realistic.html) | Aerospace CAD | Engineering-drawing-style 4-plate technical package + Three.js r170 WebGPU + TSL | 41 KB | — |

## Reports and investigations

| File | Scope | Words |
|---|---|---|
| [rapport-bracket-genetique-a350.md](rapport-bracket-genetique-a350.md) | 50-page French technical report v1 | ~21 000 |
| [rapport-bracket-genetique-a350-v2.md](rapport-bracket-genetique-a350-v2.md) | Revised v2: math-normalized for Zed/GitHub, case studies, risk register, MCP JSON-RPC annex, bilingual glossary | ~26 000 |
| [improvement-investigation-ix-v1.md](improvement-investigation-ix-v1.md) | R1-R6 recommendations + R6 adversarial addendum + canonical showcase packaging proposal | ~4 500 |
| [brainstorm-next-horizon-ix.md](brainstorm-next-horizon-ix.md) | R7-R9 next-horizon brainstorm (Autograd, QED-MCP, PyO3+WASM) | ~3 000 |
| [r7-autograd-codex-review.md](r7-autograd-codex-review.md) | Codex + Claude code-first review of R7, merged 5-day schedule | ~2 200 |
| [ix-roadmap-plan-v1.md](ix-roadmap-plan-v1.md) | Consolidated R1-R9 + showcase roadmap with dependency graph, phasing, week-by-week plan, risk register, definition of done gates | ~6 000 |

## How to reproduce the demos

**Today (manual tool chaining):** each demo was produced by orchestrating `ix_pipeline` (and other MCP tools) through Claude Code. To reproduce, open the demo's `.html` source, read the tool call sequence embedded in the comments, and replay each call with the recorded inputs.

**After R1 ships:** each demo will have a sibling `pipeline.yaml` file. A single MCP call to `ix_pipeline_run` with the YAML spec will reproduce the demo bit-identically. See `ix-roadmap-plan-v1.md` §5 Week 2 for the migration schedule.

## Planned restructure

Per the roadmap §10, this folder will be reorganized as a progressive tutorial:

```
examples/canonical-showcase/
├── README.md
├── 01-cost-anomaly-hunter/     — 3 tools, finops, simplest
├── 02-chaos-detective/         — 4 tools, signal processing
├── 03-cyber-intrusion-triage/  — 12 tools, multi-domain
├── 04-catia-bracket-generative/— 13 tools, aerospace flagship
│   ├── pipeline.yaml
│   ├── dashboard.html
│   ├── bracket-3d.html
│   ├── toolpath-3d.html
│   ├── context-realistic.html
│   ├── rapport-v1.md
│   └── rapport-v2.md
├── 05-ix-self-improvement/     — the investigation documents
│   ├── improvement-investigation-ix-v1.md
│   ├── brainstorm-next-horizon-ix.md
│   ├── r7-autograd-codex-review.md
│   └── ix-roadmap-plan-v1.md
├── 06-autograd/                — R7 demo (TBD, Phase 1)
├── 07-python-bridge/           — R9a demo (TBD, Phase 3)
└── 08-wasm-playground/         — R9b demo (TBD, Phase 4)
```

The restructure is scheduled for Week 2 alongside the R1 migration so that every demo gets its `pipeline.yaml` at the same time as it gets its directory.

## CI gate (planned)

Starting Week 2, a GitHub Actions job regenerates every demo on every push to `main` and diffs against golden outputs. Drift fails the build. This makes the showcase a load-bearing regression harness — changes to core IX crates must preserve every demo's output or explicitly bless the new output.
