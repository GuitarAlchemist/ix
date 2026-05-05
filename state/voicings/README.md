# `state/voicings/`

Per-instrument voicing artifacts consumed by `ix-voicings`, `ga-chatbot`,
and the OPTIC-K embedding pipeline.

## File classes

| File pattern              | Tracked?  | Purpose                                                |
|---------------------------|-----------|--------------------------------------------------------|
| `*-clusters.json`         | yes       | Small (~6 KB) cluster summaries, kept as canonical     |
| `*-progressions.json`     | yes       | Tiny progression seeds                                 |
| `*-topology.json`         | yes       | Topology graphs                                        |
| `*-transitions.json`      | yes       | Transition tables                                      |
| `{guitar,bass,ukulele}-corpus.json` | **yes** | Per-instrument corpus fixtures (~80 KB / 2 MB / 1.3 MB) — adversarial QA needs them for Layer 1 grounding, GA Nightly Quality reads them via build_ci_index fallback |
| `*-features.json`         | no        | OPTIC-K feature vectors — regenerable, can be huge     |
| `raw/*.jsonl`             | no        | Raw GA CLI dumps (production-only; CI falls back to `*-corpus.json`) |

`*-features.json` is gitignored because the full-corpus rebuild
produces files >100 MB, which exceeds GitHub's single-file push limit.
**Exception**: the three `*-corpus.json` files stay tracked because
`.github/workflows/adversarial-qa.yml` loads them for Layer 1
grounding — without them, every prompt that cites a `guitar_v###` /
`bass_v###` / `ukulele_v###` ID fails as "hallucinated voicing ID".
GA Nightly Quality's `build_ci_index` step also falls back to these
when `raw/*.jsonl` dumps are absent (the typical PR scenario). The
gitignore re-includes them via `!state/voicings/{name}-corpus.json`
lines.

If a future rebuild bombs any `*-corpus.json` past ~10 MB, restore the
canonical small version with:

```pwsh
git checkout HEAD -- state/voicings/guitar-corpus.json
```

`*-features.json` files are freely regenerable; the adversarial QA
path does not consume them.

## Regenerating from GA

The canonical producer is GA's `FretboardVoicingsCLI`. From the `ga`
checkout (sibling of this `ix` checkout):

```pwsh
# Stop GaApi + GaMcpServer first — the optick.index is mmap-locked.
Stop-Process -Name GaApi, GaMcpServer -ErrorAction SilentlyContinue

# Regenerate per-instrument corpus + features (~140s for guitar @ 313k voicings).
dotnet run --project Apps/GaCli/FretboardVoicingsCLI -- `
  --instrument guitar `
  --output ../ix/state/voicings/
```

Repeat with `--instrument bass` and `--instrument ukulele` as needed.

Full procedure: `ga/.claude/skills/optic-k-rebuild/SKILL.md`.

## What works without regeneration

`ga-chatbot` tests, `ix-voicings` unit tests, and `ix-optick-*` tests
do not open these files at runtime — they are referenced as path
metadata inside fixtures (string literals only). A fresh clone passes
`cargo test --workspace` without regenerating anything in this
directory.
