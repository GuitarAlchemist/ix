# Voicings Study — ix + GA Plan

**Status:** Draft
**Date:** 2026-04-15
**Owner:** spareilleux
**Depends on:** `ix-pipeline` builder API (shipped w/ `ix-friday-brief`), GA `FretboardVoicingsCLI --export` (shipped), `GA.Domain.Core.Instruments.Tuning` presets (shipped)

## Goal

Produce a grounded study of chord voicings on guitar, bass, and ukulele by building a new `ix-voicings` Rust crate that drives GA's voicing enumerator, runs ix structural analyses over the output, and renders a Markdown book. The book doubles as the evidence base and design input for a future `ga-chatbot` agent spec (plan 003) — it is not a tutorial.

## Deliverables

1. New crate `crates/ix-voicings` (Rust, library + binary, same shape as `crates/ix-friday-brief`).
2. Voicing corpus JSONL/JSON per instrument at `state/voicings/{guitar|bass|ukulele}-corpus.json` (JSONL → compacted JSON after feature extraction; raw JSONL kept under `state/voicings/raw/`).
3. Analysis artifacts per instrument at:
   - `state/voicings/{instrument}-features.json` (feature matrix + column schema)
   - `state/voicings/{instrument}-clusters.json` (cluster assignments + centroids)
   - `state/voicings/{instrument}-topology.json` (Betti numbers, persistence diagram)
   - `state/voicings/{instrument}-transitions.json` (A* minimal-movement paths between representative voicings)
   - `state/voicings/{instrument}-progressions.json` (CFG-derived progression grammar + parse counts)
4. Study at `docs/books/chord-voicings-study.md` — the grounded artifact book.
5. Draft agent spec at `docs/plans/2026-04-15-003-feat-ga-chatbot-agent-spec.md` (stub placeholder created by this plan; real content in follow-up session).
6. Smoke test that runs the full DAG against a 500-voicing cap per instrument and asserts all artifact files land.

## No-slop rules (applies to the crate AND the book)

1. **Every section of the study must cite a computed artifact** — a JSON file, table row, diagram, or dataset row produced by the pipeline. No prose without a grounding artifact.
2. **No music-theory 101 prose.** Reader knows what a chord is. No "a chord is…" sentences.
3. **No marketing adjectives** — strike "powerful", "comprehensive", "revolutionary", "innovative".
4. **No filler chapters.** Chapters are content-length driven, not page-count driven. Honest target: 12–18 pages but length is whatever the data fills.
5. **No hypothetical examples** — every voicing ID / cluster label / fingering shown in the study must come from a real pipeline output, not made up.

These rules bind the book generator too. If a chapter's grounding artifact is empty or degenerate (e.g. a single cluster), the generator omits the chapter rather than padding it.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ ix-voicings binary                                                   │
│                                                                      │
│  per instrument in {guitar, bass, ukulele}:                          │
│                                                                      │
│    ┌─────────────────────┐                                           │
│    │ GA FretboardVoicingsCLI                                         │
│    │   --export --tuning={guitar|bass|ukulele} --export-max N        │
│    │   stdout: JSONL {diagram,frets,midiNotes,minFret,...}           │
│    └────────────┬────────┘                                           │
│                 │ Command::new(...).stdout(piped) line reader        │
│                 ▼                                                    │
│    ┌─────────────────────┐   writes state/voicings/raw/{i}.jsonl     │
│    │ enumerate node      │   writes state/voicings/{i}-corpus.json   │
│    └────────────┬────────┘                                           │
│                 ▼                                                    │
│    ┌─────────────────────┐                                           │
│    │ featurize node      │   -> {instrument}-features.json           │
│    └────────────┬────────┘                                           │
│        ┌────────┼──────────┬──────────┐                              │
│        ▼        ▼          ▼          ▼                              │
│     cluster  topology  transitions  progressions                     │
│        │        │          │          │                              │
│        └────────┴────┬─────┴──────────┘                              │
│                      ▼                                               │
│              render_book node                                        │
│                      │                                               │
│                      ▼                                               │
│        docs/books/chord-voicings-study.md                            │
└──────────────────────────────────────────────────────────────────────┘
```

### Decision: shell out to GA CLI vs extend `ix_ga_bridge`

**I picked shell-out to `FretboardVoicingsCLI --export`.** Reasons:

1. The CLI already ships JSONL export (`RunExportAsync` in `ga/Demos/Music Theory/FretboardVoicingsCLI/Program.cs` lines 507–541). No GA-side work needed for the enumerate step beyond adding a `--tuning` flag (currently hardcoded `Fretboard.Default`). Extending `ix_ga_bridge` would mean writing C#-side MCP handlers, wiring serialization through the bridge's F#-style action dispatcher, and shipping a new MCP tool version — weeks of coordinated changes across repos for one function call.
2. Shell-out is restartable and debuggable. A broken run drops a `raw/*.jsonl` on disk the user can diff. Bridge calls are ephemeral.
3. Streaming matters. A full guitar enumeration at standard tuning is six-figure counts; JSONL-on-stdout is backpressure-friendly, MCP JSON-RPC is not.

**When you'd want the bridge instead:** if this pipeline ever needs to run from inside an MCP-only client (Claude Desktop, a sandboxed agent with no shell exec), or if we want voicings live-queryable from other agents. At that point wrap the ix-voicings crate as an MCP tool rather than extending `ix_ga_bridge` — don't push the C# enumerator into the bridge just to avoid a `Command::new`.

## Pipeline DAG

The crate mirrors `crates/ix-friday-brief/src/lib.rs`: `PipelineBuilder::new().node("id", |b| b.input(...).compute(...)).build()`. The DAG runs **once per instrument**, so `run()` loops over `["guitar", "bass", "ukulele"]` and invokes `build_pipeline(instrument)` for each.

| # | Node | Inputs | ix crate called | Artifact written |
|---|------|--------|-----------------|------------------|
| 1 | `enumerate` | (source) | none — `std::process::Command` on GA CLI | `state/voicings/raw/{i}.jsonl`, `{i}-corpus.json` |
| 2 | `featurize` | `enumerate` | `ix-math` (ndarray build) | `{i}-features.json` |
| 3 | `cluster` | `featurize` | `ix-unsupervised::kmeans` (primary) + `dbscan` (fallback when silhouette < 0.15) | `{i}-clusters.json` |
| 4 | `topology` | `featurize` | `ix-topo` persistent homology on feature point cloud | `{i}-topology.json` |
| 5 | `transitions` | `cluster` | `ix-graph` + `ix-search::astar`; edge cost = finger-movement heuristic (fret delta + string delta + barre toggle) | `{i}-transitions.json` |
| 6 | `progressions` | `transitions` | `ix-grammar` Earley parser over transition-symbol sequences, CFG authored by hand and checked in at `crates/ix-voicings/grammars/progressions.cfg` | `{i}-progressions.json` |
| 7 | `preference` | `featurize` | `ix-ml-pipeline` supervised — **marked phase 2, skipped in MVP** (see §MVP scope) | (skipped) |
| 8 | `render_book` | all upstream | none — Markdown templater | `docs/books/chord-voicings-study.md` (written once per `run()`, after all three instruments finish) |

### Feature vector for `featurize`

Fixed schema, checked into `features.schema.json`:

- `fret_span` (int, max - min played fret, muted strings excluded)
- `frets_used` (vec<int>, length = StringCount, -1 for muted)
- `string_count_played` (int)
- `is_barre` (bool — derived from repeated fret across ≥3 adjacent played strings at minFret)
- `min_fret`, `max_fret` (int)
- `midi_note_count` (int)
- `lowest_midi`, `highest_midi` (int)
- `interval_class_vector` (vec<int>, length 6 — **only populated for guitar** where GA's analysis pass runs; bass/ukulele get zeros until we extend the CLI's export to include analysis output. Documented gap.)
- `chord_quality_onehot` (vec<f64>, length = catalog of GA-recognized qualities; zero vector where GA didn't label the voicing)

All numeric fields are z-scored before entering KMeans/DBSCAN; categorical one-hots are left as-is.

## Chapter outline for `docs/books/chord-voicings-study.md`

Maximum 9 chapters. Each line names its grounding artifact. Chapters whose artifact is thin (single cluster, empty grammar) are **dropped by the generator**, not padded.

1. **Corpus at a glance** — row counts, fret-span histograms per instrument. Source: `{i}-corpus.json` + `{i}-features.json`.
2. **Voicing families (clustering)** — per-instrument cluster table: centroid fret span, representative voicing diagram, member count. Source: `{i}-clusters.json`. **Drop if any instrument yields only 1 cluster** (feature engineering failed; see kill criteria).
3. **Topology of the voicing space** — Betti_0/Betti_1 per instrument, persistence diagram as an embedded PNG rendered from JSON. Source: `{i}-topology.json`.
4. **Shortest physical paths between representatives** — A* path tables between cluster representatives; finger-movement cost distribution. Source: `{i}-transitions.json`.
5. **Grammar of progressions** — CFG parse counts over known progressions (I-IV-V, ii-V-I, 12-bar, I-vi-IV-V), ambiguity report. Source: `{i}-progressions.json` + `grammars/progressions.cfg`.
6. **Cross-instrument comparison** — same grid, three columns. Which cluster families exist on all three? Which are guitar-exclusive? Source: joined `{i}-clusters.json`. **Drop if any instrument's cluster schema is incomparable** (e.g. ukulele has < 3 distinct clusters).
7. **Known gaps** — what `featurize` couldn't compute (ICV for bass/ukulele), what the grammar doesn't cover, sample sizes, parameter sweeps not run. Source: the crate's own `run_manifest.json`.

(Chapters 8–9 left as reserve slots. Do not pre-allocate. If the data fills 7, ship 7.)

## ga-chatbot agent spec preview

The chatbot calls ix-voicings-derived artifacts (not the live enumerator — too slow) to answer grounded questions about voicings. Example: "give me three drop-2 voicings for Cmaj7 on guitar that transition cleanly to an Fmaj7 drop-2" — the agent looks up the Cmaj7 cluster representatives, walks the `transitions.json` A* paths to Fmaj7 representatives, ranks by movement cost, returns three with diagrams. Persona: **domain-specific assistant**, not a generalist. Affordances: `mcp:ix:ix_voicings_query` (a new MCP tool to be added in plan 003), plus read-only filesystem access to `state/voicings/`. Governance hooks: `estimator_pairing: skeptical-auditor` (must cross-check that any voicing it cites resolves to a real row in the corpus); `goal_directedness: task-scoped` (one question, one answer, no multi-turn planning). The real spec — schemas, tool shapes, test harness — is plan 003.

## MVP scope (what ships in THIS plan)

**Ship:**

- `crates/ix-voicings` library + binary
- `enumerate` node with real GA CLI shell-out (GA-side: add `--tuning {guitar|bass|ukulele}` flag to `Program.cs` — see open question 1)
- `featurize` node with real feature extraction
- `cluster`, `topology`, `transitions`, `progressions` nodes calling real ix crates (not stubs — this is not Friday Brief phase 1)
- `render_book` producing a real Markdown book from real artifacts
- Smoke test at `--export-max 500` per instrument (fast enough for CI; full corpus is a manual run)
- Stub file for plan 003 so the link isn't dead

**Cut:**

- `preference` node (ix-ml-pipeline supervised). **Labeled data is the weak link.** No playability labels exist in GA today, and hand-labeling even 200 voicings is multi-hour work with debatable ground truth. Mark in the book as "phase 2" under Known gaps.
- MCP tool wrapping (`mcp__ix__ix_voicings_query`) — deferred to plan 003 where the chatbot needs it.
- Full-corpus run — MVP asserts the pipeline works, not that it scales. Full runs are manual with `--export-max 0` (unlimited) once the 500-cap run is green.
- Cross-instrument embedding (hyperbolic / ix-code::advanced) — interesting, not load-bearing for chapter 6, which can use set intersection over cluster features.

## Kill criteria

- **GA CLI JSON is not parseable or `--tuning` flag can't be added cleanly** → abandon the shell-out path and pivot to extending `ix_ga_bridge` with a `voicings` action. This is the only scenario where the bridge wins.
- **Clustering produces only 1 cluster per instrument** → feature vectors are wrong. Do not ship the book. Fix `featurize` (likely missing normalization or the one-hot quality vector is swamping numeric features) or abandon clustering and restructure the book around topology + transitions only.
- **Topology produces Betti_0 = N (every point its own component) and Betti_1 = 0** → the metric is wrong (probably raw Euclidean on un-normalized features). Re-pick the metric before shipping, or drop chapter 3.
- **Progression grammar parses nothing** → the CFG is mismatched with the transition alphabet. Either rewrite the CFG or drop chapter 5. Don't ship a book with a chapter titled "Grammar" and an empty parse table.
- **`enumerate` for guitar-standard produces < 10k voicings at `--export-max 0`** → GA enumerator regressed; stop and file a GA bug rather than build on broken corpus.

## Open questions

1. **Does `FretboardVoicingsCLI` accept a `--tuning` flag?** Reconnaissance answer: **no.** `Program.cs:509` hardcodes `Fretboard.Default` in `RunExportAsync`. Adding the flag is a ~30-line change: parse `--tuning guitar|bass|ukulele`, swap the `Tuning` passed to `Fretboard.Create(...)`, thread through. `Tuning.Default`/`Tuning.Bass`/`Tuning.Ukulele` already exist at `ga/Common/GA.Domain.Core/Instruments/Tuning.cs:23-33`. This is GA-side work that must land before ix-voicings can run bass/ukulele. Decision needed: do I file a GA PR from this plan, or scope it as a ga/ commit by the same engineer?
2. **Is the GA `Fretboard` ctor even parameterized by `Tuning`?** Need to grep `ga/Common/GA.Domain.Core/Instruments/Fretboard/` — `Fretboard.Default` exists, but `Fretboard.Create(Tuning, int frets)` or equivalent must exist for bass (4 strings, typically 21 frets) and ukulele (4 strings, 15 frets). If it doesn't, the GA-side change grows.
3. **Does `VoicingGenerator.GenerateAllVoicingsAsync` accept a non-default fretboard?** Program.cs:516 passes `fretboard` as a positional arg — assume yes, verify before starting.
4. **Feature vector dimensionality for bass vs guitar** — 4-string instruments vs 6-string. Either pad to 6 with `-1` (what the plan currently assumes) or run three independent per-instrument clusterings with instrument-specific feature schemas. Padding is simpler and lets chapter 6's cross-instrument comparison work on a common vector. I lean pad — flag for review.
5. **Grammar file format** — `ix-grammar` supports CFG / Earley / CYK. Which concrete syntax does it parse? Need to grep `crates/ix-grammar/` before hand-writing `progressions.cfg`.
6. **`state/voicings/` retention** — these files can get large (full guitar corpus is megabytes of JSONL). `.gitignore` or check in a 500-cap sample?

## Estimated effort

**One engineer, honest range: 4–7 dev-days.**

- GA-side `--tuning` flag + Fretboard constructor wiring: 0.5–1 day (includes smoke-testing bass/ukulele enumeration actually terminates)
- `crates/ix-voicings` skeleton + enumerate node + featurize: 1 day
- cluster + topology nodes (real ix crate calls, not stubs): 1 day
- transitions + progressions (A* cost function design is the hard part; CFG is small): 1–1.5 days
- render_book + chapter generator that can drop degenerate chapters: 0.5–1 day
- smoke test + end-to-end run + fixing whatever the data says is broken: 1–1.5 days

Add 1 day contingency if open questions 2 or 5 turn out badly.

## References

- `crates/ix-friday-brief/src/lib.rs` — pipeline-builder idiom to mirror
- `ga/Demos/Music Theory/FretboardVoicingsCLI/Program.cs` lines 507–541 — existing JSONL export mode
- `ga/Common/GA.Domain.Core/Instruments/Tuning.cs` lines 23–33 — the three required tuning presets
- `docs/plans/2026-04-15-001-feat-friday-brief-mvp-plan.md` — tone/structure reference
- `docs/guides/graph-theory-in-ix.md` — for transitions-node crate selection (CLAUDE.md mandate: no new graph deps)
