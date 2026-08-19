# IX Exploratory Showcase

These demos explore how IX algorithms can be composed against several problem classes. They are synthetic, model-orchestrated artifacts, not validated engineering analyses or production benchmarks.

> [!CAUTION]
> The aerospace/CATIA artifacts do not use a native CAD kernel, structural topology optimizer, FEA/CFD solver, collision-aware toolpath planner, or certification workflow. `ix_topo` computes topological-data-analysis quantities rather than structural topology optimization; `ix_game_nash` does not compute a Pareto front; and Viterbi does not establish a collision-free 5-axis toolpath. All bracket dimensions, stresses, frequencies, mass reductions, safety factors, toolpaths, certification statements, and commercial comparisons are illustrative and unvalidated. They must not be used for design, manufacture, safety, procurement, or certification decisions.

This folder serves three purposes:

1. **Exploration** — identify promising compositions and the adapters or solvers still missing.
2. **Onboarding tutorial** — traverse the demos from 3-tool to 13-tool pipelines to learn the IX mental model.
3. **Candidate regression corpus** — a demo becomes a regression harness only after an executable, independently validated replay exists.

See `ix-roadmap-plan-v1.md` for the roadmap that turns this ad-hoc collection into a CI-gated first-class project artifact.

## Tool × demo matrix

| Demo | Domain | Tools | Size | Reports |
|---|---|---|---|---|
| [cost-anomaly-hunter.html](cost-anomaly-hunter.html) | FinOps | `ix_stats`, `ix_fft`, `ix_kmeans` (3) | 9 KB | — |
| [cyber-intrusion-triage.html](cyber-intrusion-triage.html) | Security | 12 tools: `stats`, `fft`, `chaos_lyapunov`, `bloom_filter`, `kmeans`, `viterbi`, `markov`, `linreg`, `rf`, `topo`, `nash`, `bandit`, `gov_check` | 19 KB | — |
| [catia-bracket-generative.html](catia-bracket-generative.html) | Synthetic aerospace exploration | 13 mathematical stand-ins; no CAD/FEA validation | 29 KB | [v1 FR](rapport-bracket-genetique-a350.md) · [v2 FR](rapport-bracket-genetique-a350-v2.md) |
| [catia-bracket-3d.html](catia-bracket-3d.html) | Illustrative rendering | Three.js r160 rendering with a synthetic stress visualization | 15 KB | — |
| [catia-toolpath-3d.html](catia-toolpath-3d.html) | Illustrative sequence | Isometric visualization of a Viterbi state sequence; not a collision-aware toolpath | 12 KB | — |
| [catia-bracket-context-a350.html](catia-bracket-context-a350.html) | Illustrative context — unvalidated | Schematic A350 side/top/pylon context; no CAD kernel, no FEA | 17 KB | — |
| [catia-bracket-context-realistic.html](catia-bracket-context-realistic.html) | Illustrative context — unvalidated | Engineering-drawing-style synthetic context plates + Three.js r170 WebGPU + TSL; no CAD kernel, no FEA, no certification | 41 KB | — |

## Reports and investigations

| File | Scope | Words |
|---|---|---|
| [rapport-bracket-genetique-a350.md](rapport-bracket-genetique-a350.md) | 50-page French technical report v1 | ~21 000 |
| [rapport-bracket-genetique-a350-v2.md](rapport-bracket-genetique-a350-v2.md) | Revised v2: math-normalized for Zed/GitHub, case studies, risk register, MCP JSON-RPC annex, bilingual glossary | ~26 000 |
| [improvement-investigation-ix-v1.md](improvement-investigation-ix-v1.md) | R1-R6 recommendations + R6 adversarial addendum + canonical showcase packaging proposal | ~4 500 |
| [brainstorm-next-horizon-ix.md](brainstorm-next-horizon-ix.md) | R7-R9 next-horizon brainstorm (Autograd, QED-MCP, PyO3+WASM) | ~3 000 |
| [r7-autograd-codex-review.md](r7-autograd-codex-review.md) | Codex + Claude code-first review of R7, merged 5-day schedule | ~2 200 |
| [ix-roadmap-plan-v1.md](ix-roadmap-plan-v1.md) | Consolidated R1-R9 + showcase roadmap with dependency graph, phasing, week-by-week plan, risk register, definition of done gates | ~6 000 |

## Phase 1 deliverables (R7 Week 1)

The first phase of the R7 Autograd-IX work shipped Days 1-3 of Week 1 plus the full validation pass described in `ix-roadmap-plan-v1.md` §5. Phase 1 artifacts:

| File | Purpose |
|---|---|
| [ix-roadmap-plan-v1.md](ix-roadmap-plan-v1.md) | Consolidated R1-R9 + canonical showcase roadmap with dependency graph, phasing, risk register, definition-of-done gates |
| [r7-autograd-codex-review.md](r7-autograd-codex-review.md) | Initial Codex + Claude code-first review that locked in the build-scratch-over-ndarray decision, the `ExecutionMode` enum, and the Hermitian-mirror FFT fix |
| [r7-day2-review.md](r7-day2-review.md) | Four-provider Day 2 multi-LLM review (Codex + Gemini + Mistral Large + Claude) that reordered Day 3 to put cleanup before new ops |
| [r7-phase1-benchmarks.txt](r7-phase1-benchmarks.txt) | `cargo bench` output for 6 benchmarks: add/mul/matmul/variance primitives + full linreg MSE + full Adam step |
| [r7-phase1-security.txt](r7-phase1-security.txt) | `cargo audit` results: ix-autograd dep cone is CLEAN (0 advisories); 1 workspace-level vulnerability + 4 warnings triaged and deferred |
| [r7-phase1-retrospective.md](r7-phase1-retrospective.md) | Week 1 actual vs planned, go/no-go verdict (PASS, 242× speedup), what went better/worse, Phase 2 readiness, Day 4/5 TODO list |

**Phase 1 verdict:** PASS. R7 Week 1 go/no-go criterion met (gradient Adam converges 242× faster than evolutionary baseline, 15/15 finite-diff tests pass at 1e-5 tolerance). Proceed to Day 4 hardening and Day 5 FFT-behind-feature-flag.

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
