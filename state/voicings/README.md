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
| `*-corpus.json`           | **no**    | Per-instrument voicing corpus — regenerable, can be huge |
| `*-features.json`         | **no**    | OPTIC-K feature vectors — regenerable, can be huge       |
| `raw/*.jsonl`             | no        | Raw GA CLI dumps                                        |

`*-corpus.json` and `*-features.json` are gitignored because the
full-corpus rebuild produces files >100 MB, which exceeds GitHub's
single-file push limit. Earlier `guitar-corpus.json` and
`guitar-features.json` were tracked at small sample sizes (~80 KB and
~250 KB) and got bombed by a corpus rebuild on 2026-05-04 (working tree
hit ~475 MB total). Untracked since `62ab038`+1 to prevent recurrence.

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
