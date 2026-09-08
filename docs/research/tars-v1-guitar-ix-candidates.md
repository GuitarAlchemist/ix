# TARS V1 Music/Guitar Algorithm Candidates for IX

Issue: [GuitarAlchemist/ix#195](https://github.com/GuitarAlchemist/ix/issues/195) — parent epic
[#189](https://github.com/GuitarAlchemist/ix/issues/189), depends on
[#190](https://github.com/GuitarAlchemist/ix/issues/190). Companion to the advanced-math matrix
[`tars-v1-advanced-math-ix-gap-matrix.md`](tars-v1-advanced-math-ix-gap-matrix.md) (#202) and built to
the same shape so the two are comparable.

This document answers one question per row: **for this music/guitar idea that TARS V1 raised, what
does IX have today, and does the idea even belong to IX?** It is a salvage assessment, not an
adoption decision. Nothing has been ported, and no claim from the TARS source text is treated as
validated.

## Evidence base

| Item | Value |
| --- | --- |
| TARS source repository | `GuitarAlchemist/tars` (local sibling checkout `../tars`) |
| Pinned ref | `69cf427eccb25514728eacfd3530218df3975259` (`2026-07-30`) — same pin as #190 and #202 |
| How TARS paths were resolved | one full `git ls-tree -r <SHA>` (15,090 paths) for the path sweep, `git grep <pattern> <SHA>` for two content sweeps, `git show <SHA>:<path>` for every read |
| IX side | this worktree, branch `research/ix-195-tars-v1-music-candidates`, off `main` @ `fa9852c` |
| Survey date | 2026-09-08 |
| Cost incurred | 0 USD — local reads only (`free-local`), no model pass over any transcript |

### Verification rule

Every `source_doc` below was proven to exist at the pinned SHA **before** its row was written, and
every `current_ix_surface` cites a path in this worktree that was opened and read. The previous
inventory revision cited 14 F# files that do not exist at this SHA; that failure mode is the reason
this section exists. No path here was taken from another document on faith — including from #190 and
#202, both of which are corrected below on points of fact.

### Source-document existence check

All 13 candidate sources resolve at the pinned SHA. Line counts are recorded because they are the
cheapest independent corroboration that the right file was read.

| source_doc (relative to tars root) | Exists @ SHA | Lines | What it actually is |
| --- | --- | ---: | --- |
| `v1/guitar_fretboard_analysis.tars.md` | yes | 240 | autonomous-instruction **spec** — requirements, no code |
| `v1/harmonic_progression_analyzer.tars.md` | yes | 236 | autonomous-instruction **spec** — requirements, no code |
| `v1/sample_guitar_analysis.tars.md` | yes | 60 | **not music** — a spec for profiling the GA *codebase* |
| `v1/GUITAR_ALCHEMIST_TARS_INTEGRATION_PLAN.md` | yes | 320 | integration plan, quaternion-framed |
| `v1/src/TarsEngine.FSharp.Core/GuitarAlchemistIntegration.fs` | yes | 393 | **not music** — keyword classifier over GA source |
| `v1/src/TarsEngine.FSharp.Core/HurwitzQuaternions.fs` (`MusicalQuaternions`, L314–347) | yes | 34 (module) | the only compiled music arithmetic; see §F |
| `v1/src/TarsEngine.FSharp.Core/Tier10MetaLearning.fs` | yes | 342 | hardcoded music-concept prerequisite graph |
| `TarsEngine.FSharp.FLUX.Tests/VexFlowMusicTests.fs` | yes | 383 | VexFlow HTML-generation test; 2 lines of triad math |
| `v1/parked_legacy/autonomous_guitar_alchemist_cycle.fsx` | yes | 533 | `printfn` demo, hardcoded scores |
| `v1/parked_legacy/tars-guitar-alchemist-demo.fsx` | yes | 279 | `printfn` demo |
| `v1/parked_legacy/production/tars-guitar-alchemist-integration.md` | yes | 13 | stub |
| `v1/docs/Explorations/v1/Chats/ChatGPT-DSLs for Visualization Engines.md` | yes | not measured | music sub-DSL section (notation/tablature/MIDI) |
| `.tars/reports/sred-2024/GuitarAlchemist_SRED_Report_2024.md` | yes | 271 | SR&ED tax filing about GA, not a TARS artifact |

`v1/autonomous_backups/cycle_0001_20250906_151540/src/TarsEngine.FSharp.Core/GuitarAlchemistIntegration.fs`
also exists and is a backup copy of the same file; it is not treated as a separate source.

**Two sources were found only by the second content sweep** (`ChatGPT-DSLs for Visualization
Engines.md`, `Tier10MetaLearning.fs`) — the first sweep's vocabulary missed them. Sweep vocabularies
are recorded in [Method](#method-and-its-limits) so the negative results can be re-run and widened.

## The headline

**TARS V1 contains essentially no music algorithms.** Not "few" — the compiled, executable music
arithmetic in the entire V1 tree is:

1. `MusicalQuaternions.encodeMusicalInterval` / `harmonicRelationship` — 34 lines, and
   [demonstrably degenerate](#f-the-quaternion-harmonic-framework) (§F).
2. Two lines in a test file: `(rootInt + 4) % 12` and `(rootInt + 7) % 12`, a major-triad generator
   (`VexFlowMusicTests.fs:293-294`).

That is the complete inventory. Everything else is a **specification** (what a system should do), a
**demo** (`printfn` with hardcoded scores), or **not about music at all** despite its filename.

Meanwhile IX ships a substantial music surface that the #202 advanced-math matrix never saw, because
that matrix was scoped to advanced math and music was out of frame. Both #190 (which classified
exactly 2 documents as `music`) and #202 therefore understate what IX already owns.

So the honest answer to "what music/guitar algorithms should IX salvage from TARS V1" is:
**none of the implementations, and one named algorithm mentioned in passing in a spec** —
Krumhansl–Schmuckler key finding, which is not TARS's invention and whose input IX already computes.
That single row is the prototype proposal in §9.

## What IX already has (measured in this worktree)

Recorded first, because a salvage matrix that does not know the current surface will re-propose it.

### `crates/ix-bracelet` — 3,095 SLOC, 117 tests, 9 modules

Not "a dihedral `Group` trait". The full public surface, read from source:

| Module | Lines | Public surface |
| --- | ---: | --- |
| `prime_form.rs` | 197 | `necklace_prime_form`, `bracelet_prime_form` |
| `forte.rs` | 598 | `ForteNumber`, `forte_number`, `all_forte_numbers` |
| `grothendieck.rs` | 597 | `Icv`, `Delta`, `icv`, `grothendieck_delta`, `find_nearby`, `z_related_pairs`, `find_shortest_path` |
| `neo_riemannian.rs` | 179 | `TriadKind`, `classify_triad`, and the `P`/`L`/`R`/`S`/`N`/`H` transforms |
| `serial.rs` | 408 | `ToneRow`, `RowForm`, `SerialError` (twelve-tone operations) |
| `fourier.rs` | 370 | `dft`, `dft_magnitudes`, `dft_phases`, `phase_aligned_similarity` |
| `orbit.rs` | 130 | `orbit`, `orbit_unique`, `all_prime_forms` |
| `pc_set.rs` | 187 | `PcSet` |
| `dihedral.rs` / `action.rs` | — | `trait Group`, `DihedralElement`, `trait Action` |

`forte.rs:30` carries the doc-tested assertion `forte_number(major_triad).unwrap().to_string() ==
"3-11"`, which matches the behaviour #195's framing cites.

### DuckDB exposure — 10 music UDFs, each mapped to its registration file

| UDF | Registered in |
| --- | --- |
| `ix_prime_form`, `ix_forte_number`, `ix_classify_triad`, `ix_icv` | `crates/ix-duck/src/bracelet.rs` (193 L, 4 tests) |
| `ix_icv_l1`, `ix_z_related` | `crates/ix-duck/src/grothendieck.rs` |
| `ix_row_matrix`, `ix_row_invert`, `ix_row_retrograde`, `ix_row_combinatoriality` | `crates/ix-duck/src/serial.rs` |

### `crates/ix-voicings` — 3,210 SLOC, and the boundary already drawn

`lib.rs` (2,036 L) + `viz_precompute.rs` (974 L) + `main.rs` (200 L). Its module header states the
architecture plainly:

> This crate shells out to GA's `FretboardVoicingsCLI --export` with a `--tuning
> {guitar|bass|ukulele}` flag and turns the JSONL stream into two on-disk artifacts per instrument.

**IX does not generate fretboard geometry. GA does, and IX consumes it.** `fret_span`, `min_fret`,
`max_fret`, `is_barre` are *deserialized fields* on `VoicingRow`, not computed here. What IX adds is
the reusable pipeline over that corpus: `featurize` → `cluster` → `topology` → `transitions` →
`progressions`, plus `silhouette_score`, `predict_cluster`, `ShortestPath`, and `movement_cost`.

`movement_cost(&VoicingRow, &VoicingRow) -> f64` (`lib.rs:932`) is a physical voice-leading /
hand-movement distance — per-string fret delta, a mute-toggle penalty, a barre-toggle penalty — and
it is unit-tested (`movement_cost_same_voicing_is_zero`, `movement_cost_includes_fret_deltas`).

### `crates/ix-acoustic-tune` — 2,400 SLOC, the audio front end

`chroma(signal, sample_rate, f_min, n_octaves) -> [f64; 12]`
(`transforms.rs:121`) is a **12-element pitch-class profile**, octave-invariant by construction and
tested (`chroma_is_octave_invariant`). Also present: `cqt`, `cqt_frequencies`, `mfcc`, `mel_spectrum`,
`autocorrelation_f0`, `inharmonicity_b`, `fit_inharmonicity`, `spectral_centroid`, `hilbert`,
`envelope`.

This matters more than it looks: the harmonic-analyzer spec's Phase 2 Step 1 asks for "chromatic
pitch class profile analysis". IX already has it.

### MCP exposure — the music gap

`crates/ix-agent/tests/parity.rs` `EXPECTED` holds **94** tool names (measured by parsing the array
with comments stripped; no duplicates, all `ix_`-prefixed). Only four are music-adjacent:
`ix_ga_bridge`, `ix_analyze_reference`, `ix_optick_search`, `ix_voicings_payload`.

**There is no `ix_bracelet` MCP tool.** The richest music crate in IX — 3,095 SLOC, 117 tests — is
reachable from DuckDB and from Rust, and is invisible to every agent in the federation. This is the
music-side instance of the exposure deficit #202 found for `ix-dynamics`.

> **Two corrections to sibling documents, verified here.**
> #202 describes "the 96-entry `EXPECTED` array". `EXPECTED` has **94** entries at both `4331cf7`
> (the commit #202 surveyed) and `fa9852c`. The 96 figure is the count of distinct `ix_` names
> *file-wide*; the extra two are `ix_approval` and `ix_maintain_gate`, which appear outside the
> array. Separately, `parity.rs:124`'s own doc comment says "the default surface (93) and the
> `--features maintain-gate` surface (94)" — that prose is **stale by one**. Neither error changes
> any #202 conclusion, but the array is a count oracle, so its stated size should be right.

## The matrix

Conventions follow #202. `current_ix_surface` cites a path in this worktree; `—` means a
case-insensitive regex over `crates/**/*.rs` found nothing, with the pattern recorded in the row.
`owner` is the [boundary verdict](#the-ga-ix-boundary): `IX`, `GA`, `GS` (guitar-singularity), or
`none` (build nowhere).

### A. Fretboard geometry and voicing generation

Source: `v1/guitar_fretboard_analysis.tars.md` (240 L, a specification).

| # | candidate | owner | current_ix_surface | status | action | prio | cx | testability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Fretboard position map (tuning + fret → pitch) | **GA** | — (no `fret_to_note`/`open_string` anywhere) | out_of_scope | document | — | XS | high | Absent from IX *by design*. `ix-voicings` shells out to GA's `FretboardVoicingsCLI`. Deterministic and trivially testable, but it is the instrument-model layer GA owns; duplicating it in IX would create the divergence hazard #202's C3 row found for `takagi.rs`. |
| A2 | Chord voicing enumeration (12 roots × chord types × positions) | **GA** | consumed via `ix_voicings::enumerate` | out_of_scope | document | — | M | high | Same boundary. GA's live corpus is 313,047 voicings; the spec's target was "500+". |
| A3 | Fingering ergonomics / playability (span ≤ 4 frets) | **GA** | `VoicingRow::fret_span`, `is_barre` (consumed) | out_of_scope | document | — | S | high | The *constraint* is an instrument-physics fact, not a reusable algorithm. GA computes it. |
| A4 | Difficulty rating from span, position, stretch | **GA** | `FEATURE_COLUMNS` includes `fret_span`, `frets_used_*`, `is_barre` | partially_implemented | document | P3 | S | medium | IX already *features* these columns for clustering. A scalar "difficulty" is a product-facing judgement — GA/GS own the rubric. `ga_easier_voicings` already exists. |
| A5 | Drop-2 voicing generator | **GA** | — (`drop2`/`drop_2` appear only in prompt strings: `ix-voicings/src/lib.rs:1608`, `ga-chatbot/src/main.rs:1501`) | out_of_scope | document | — | S | high | **Tempting but not IX.** Drop-2 is a deterministic transform (drop the 2nd voice from the top by an octave) with an exact oracle — it passes boundary test 1 and 2, and fails test 3: it operates on GA's voicing/instrument model. It belongs beside GA's enumerator. Recorded because it is the strongest false-positive candidate in this matrix. |
| A6 | Slash chords / bass inversions | **GA** | — | out_of_scope | document | — | XS | high | GA owns this; per project convention bass lives in `SlashSuffix`, not in PC-set ranking. |
| A7 | Polychord / upper-structure-triad identification | **GA** | `ix-bracelet` `PcSet`, `forte_number` (the PC-set primitives) | partially_implemented | document | P3 | S | high | GA exposes `ga_polychord`. The set-theoretic substrate is in `ix-bracelet`; the naming/spelling layer is GA's. |
| A8 | Voice-leading cost between voicings | **IX** | `crates/ix-voicings/src/lib.rs:932` `movement_cost` | **already_done** | document | P2 | XS | high | The spec's "analyze finger movement efficiency between chord positions", already implemented and tested. Also `transitions` / `ShortestPath` over the voicing graph. |
| A9 | Scale ↔ chord compatibility mapping | **GA** | `ix-bracelet` subset/orbit primitives | partially_implemented | document | P3 | S | high | GA has `ga_diatonic_chords`, `ga_set_class_subs`. IX owns the PC-set algebra beneath. |

### B. Harmonic analysis

Source: `v1/harmonic_progression_analyzer.tars.md` (236 L, a specification).

| # | candidate | owner | current_ix_surface | status | action | prio | cx | testability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | **Krumhansl–Schmuckler key finding** | **IX** | — (no `krumhansl`/`schmuckler`/`key_profile`/`temperley` in `crates/`) | **missing_algorithm** | **prototype** | **P1** | **XS** | **high** | **The one genuine net-new IX candidate in this survey.** See §9. Input `[f64; 12]` is already produced by `ix_acoustic_tune::transforms::chroma`. The algorithm is published correlation against fixed profiles — an explicit rubric, so it clears #195's "no subjective scoring" non-goal. |
| B2 | Roman-numeral / functional harmony analysis | **GA** | — | out_of_scope | document | — | M | medium | Key-relative *spelling* is music-domain product logic (enharmonics, secondary dominants, notation conventions). GA owns it. |
| B3 | Modal interchange / borrowed-chord detection | **GA** | `ix-bracelet` PC-set + `neo_riemannian` transforms | partially_implemented | document | P3 | M | medium | Substrate in IX, interpretation in GA. |
| B4 | Chord substitution (tritone, chromatic mediant) | **GA** | `ix_z_related`, `find_nearby`, `grothendieck_delta` | partially_implemented | document | P3 | S | medium | GA exposes `ga_chord_substitutions`. IX's `find_nearby` over ICV space is the general "similar set-class" engine. |
| B5 | Counterpoint checks (parallel fifths/octaves) | **IX** | — (no `parallel_fifth`/`counterpoint` in `crates/`) | missing_algorithm | defer | P3 | S | high | Passes all three boundary tests — pure interval arithmetic over voice pairs, exact oracle, no product state. Deferred only because no IX consumer exists: `ix-voicings` models fretboard positions, not independent voices. Revisit if a voice-separated representation appears. |
| B6 | ML next-chord prediction | **GA** | `ix-graph` `markov.rs`, `hmm.rs`; MCP `ix_markov`, `ix_viterbi` | implemented_needs_exposure | document | P3 | M | medium | The *sequence model* is IX and already exists. The *corpus and musical acceptability* are GA's; `ga_progression_completion` exists. Nothing to salvage from TARS, which proposes this as future work. |
| B7 | Style/genre classification from progressions | **GA** | `ix-supervised` classifiers | out_of_scope | defer | — | M | low | Generic classification is IX and present; labels and taxonomy are product. No rubric in the source → #195 non-goal. |

### C. Audio → symbol

Source: `v1/harmonic_progression_analyzer.tars.md` Phase 2.

| # | candidate | owner | current_ix_surface | status | action | prio | cx | testability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | Chromatic pitch-class profile from audio | **IX** | `crates/ix-acoustic-tune/src/transforms.rs:121` `chroma` | **already_done** | document | P1 | XS | high | Octave-invariant, tested. The spec asks for exactly this. |
| C2 | Chord template matching over a chroma vector | **IX** | — (no `chord_template` in `crates/`) | missing_algorithm | prototype | P2 | XS | high | Sibling of B1 and the same shape: correlate a 12-vector against fixed binary templates. Should ride B1's harness — build B1 first, then this is a template-set swap. |
| C3 | Temporal smoothing of a chord sequence | **IX** | `ix-graph` `hmm.rs`, `viterbi`; `ix-signal` filters | **already_done** | document | P2 | XS | high | Viterbi over a chord-state HMM is the textbook method and IX has it exposed (`ix_viterbi`). |
| C4 | Real-time <100 ms latency budget | **GS** | n/a | out_of_scope | document | — | — | n/a | A product performance requirement, not an algorithm. |

### D. Pedagogy and practice paths

Source: `v1/src/TarsEngine.FSharp.Core/Tier10MetaLearning.fs` (342 L).

| # | candidate | owner | current_ix_surface | status | action | prio | cx | testability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | Ordering a prerequisite DAG of concepts | **IX** | `crates/ix-pipeline/src/dag.rs:201` and `crates/ix-graph/src/graph.rs:147` — both `topological_sort` | **already_done** | document | P2 | XS | high | TARS hardcodes a graph (`pitch → interval → chord → scale → voice_leading`, edges `Requires`, `Difficulty` weights) but never traverses it: the file contains **zero** occurrences of `% 12`, `mod 12`, `semitone` or `midi`. IX has the traversal twice over. |
| D2 | Difficulty-weighted practice-path generation | **GS** | `ix-graph::routing`, `ShortestPath` | partially_implemented | document | P3 | S | medium | The *algorithm* is weighted shortest path — IX has it. The *content* (which concepts, whose difficulty, what a learner has mastered) is per-user product state. Guitar-singularity owns it. Clean split: IX supplies the traversal, GS supplies the graph. |

### E. Notation and DSL

Source: `v1/docs/Explorations/v1/Chats/ChatGPT-DSLs for Visualization Engines.md` (music sub-DSL
section, L294–360); `TarsEngine.FSharp.FLUX.Tests/VexFlowMusicTests.fs` (383 L).

| # | candidate | owner | current_ix_surface | status | action | prio | cx | testability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | Music notation / tablature DSL (VexFlow, ABC) | **GA/GS** | — | reject | document | — | M | low | Rendering and notation syntax. #195's non-goals forbid UI/rendering work outright. `VexFlowMusicTests.fs` is an HTML-generation test, not a music algorithm. |
| E2 | MIDI sequencing DSL / DAW export | **GA/GS** | — | reject | document | — | M | low | Product I/O. |
| E3 | Major-triad construction `(root+4)%12`, `(root+7)%12` | **IX** | `crates/ix-bracelet/src/neo_riemannian.rs` `classify_triad`, `TriadKind`; UDF `ix_classify_triad` | **already_done** | document | P3 | XS | high | The entire compiled music arithmetic of TARS V1 outside §F is these two lines, inside a test file (`VexFlowMusicTests.fs:293-294`). IX's `ix-bracelet` supersedes it comprehensively. |

### F. The quaternion harmonic framework

Source: `v1/src/TarsEngine.FSharp.Core/HurwitzQuaternions.fs` L314–347 (`MusicalQuaternions`), plus
`v1/src/TarsEngine.FSharp.Core/GuitarAlchemistIntegration.fs`.

Both `.tars.md` specifications name the "Hurwitz quaternion framework" as a hard **dependency**
("Integration with Guitar Alchemist's quaternion harmonic framework" is a success criterion of the
harmonic analyzer). #190 flagged it as "unsubstantiated". It is worse than unsubstantiated: the code
exists, and it is arithmetically degenerate.

`encodeMusicalInterval` maps a frequency to four quaternion coefficients:

```fsharp
let a = int (logFreq * 10.0) % 20 - 10
let b = int (frequency / 100.0) % 20 - 10
let c = int (frequency * 0.01) % 20 - 10     // identical expression to b
let d = int (frequency * 0.001) % 20 - 10
Harmonic = int (frequency / 440.0)
```

Evaluated over the standard 24-fret guitar range (E2 = 82.41 Hz to E6 = 1318.51 Hz, all 49
semitones):

| Property | Measured result |
| --- | --- |
| `b` vs `c` | **identical for every input** — `frequency / 100.0` and `frequency * 0.01` are the same expression. One of four dimensions is wasted. |
| `d` | takes **2 distinct values** (`-10`, `-9`) across all 49 semitones — effectively constant. |
| Distinct 4-tuples for 49 distinct pitches | **30** → **19 collisions**; 39% of the range is not uniquely encoded. |
| `Harmonic = int(freq/440)` | takes values `{0, 1, 2}` over the whole fretboard; 28% of the range maps to `0`. |
| Continuity in pitch | `a` jumps from `9` to `-10` between MIDI 67 and 68 (G4→G♯4) — a modular wraparound, so two adjacent semitones land at opposite ends of the coefficient range. |

`harmonicRelationship` then declares two intervals related iff the **norm of their quaternion product
is prime**, returning the string `"Prime harmonic relationship: ..."`. No reference, no oracle, no
validation anywhere in the corpus.

| # | candidate | owner | status | action | notes |
| --- | --- | --- | --- | --- | --- |
| F1 | Quaternion encoding of musical intervals | **none** | **reject** | document | Encoding is non-injective (19/49 collisions), discontinuous in pitch, and has two redundant dimensions. Not a foundation to build on. |
| F2 | Primality of quaternion norm as a harmonic relation | **none** | **reject** | document | Numerology. No music-theoretic referent, no oracle, no cited source. |
| F3 | The `.tars.md` specs' dependency on F1/F2 | n/a | n/a | document | **Consequence:** both specs must be read with their quaternion integration points struck out. That removes a success criterion from each, but nothing else in them depends on it. |
| F4 | `GuitarAlchemistIntegration.fs` as a music module | **none** | **reject** | document | Contains **no** music arithmetic. `classifyFile` is `List.exists contentLower.Contains` over `["chord"; "scale"; "note"; ...]`; `calculateComplexity` is `content.Split("let ").Length`. Its `enhanceHarmonicAnalysis` / `optimizeChordProgression` live inside `sprintf """..."""` **string literals** — suggested code text, never compiled, never run. Its `ExpectedImprovement = 0.25` is a hardcoded field, not a measurement. |

## The GA/IX boundary

#195 says the separation is the hard part. It is, and the useful result is that **IX has already
drawn this boundary in code** — `ix-voicings` shells out to GA's `FretboardVoicingsCLI` rather than
reimplementing fretboard geometry. What follows makes that precedent explicit so it can be applied to
new candidates rather than re-litigated.

**A candidate is IX-owned only if all three hold:**

1. **Deterministic and specifiable without product context.** It can be stated as a function on
   plain data (vectors, pitch-class sets, graphs) with no reference to a user, a session, a catalog,
   or a rendering.
2. **It has a machine oracle.** Correctness is decidable by a test — a published constant, a closed
   form, an algebraic invariant — not by a musician's judgement. (#195's non-goal: no subjective
   musical scoring without an explicit rubric.)
3. **It needs no product state.** No user history, no instrument model, no notation convention, no
   preference.

Fail (1) or (3) → **GA** if it is music-domain logic or instrument modelling; **guitar-singularity**
if it is pedagogy, personalisation, or UX. Fail (2) with no rubric → build nothing.

Applying it to the sharp cases:

| Candidate | (1) stateless | (2) oracle | (3) no product state | Verdict |
| --- | --- | --- | --- | --- |
| Krumhansl–Schmuckler key finding (B1) | yes — `[f64;12] → (tonic, mode)` | yes — published profiles + a transposition invariant | yes | **IX** |
| Drop-2 voicing generator (A5) | yes | yes — exact | **no** — needs GA's voicing/instrument model | **GA** |
| Voice-leading cost (A8) | yes | yes | yes | **IX** (already built) |
| Practice-path generation (D2) | algorithm yes, content no | no — "best path" is pedagogical | **no** — learner state | **GS** (IX supplies traversal) |
| Roman-numeral analysis (B2) | no — enharmonic spelling is convention | partial | no | **GA** |
| Counterpoint checks (B5) | yes | yes — rule-exact | yes | **IX**, but no consumer → defer |

The two documents that most need this rule are the `.tars.md` specs: `guitar_fretboard_analysis`
sits almost entirely on the GA side of it, and `harmonic_progression_analyzer` straddles it — its
Phase 2 (audio → chroma → template) is IX, its Phases 3–5 (Roman numerals, substitutions, style) are
GA.

## 9. Prototype candidate — Krumhansl–Schmuckler key finding

The one cheap, testable, genuinely-missing IX algorithm this survey found.

```
ix-bracelet (or a new ix-tonal module)
    pub fn key_from_pcp(pcp: &[f64; 12]) -> KeyEstimate
    pub struct KeyEstimate { tonic: u8, mode: Mode, r: f64, runner_up: (u8, Mode, f64) }
```

**What it is.** Correlate a 12-element pitch-class profile against 24 candidate profiles (12
rotations × major/minor) built from the published Krumhansl–Kessler constants, and return the argmax
with its Pearson `r`. Roughly 60 lines.

**Why IX owns it.** Passes all three boundary tests: it is a function from a 12-vector to a label,
its oracle is published constants, and it touches no product state. It is vector correlation with a
fixed basis — the same shape as work already in `ix-math`.

**Why it is cheap.** Its input already exists: `ix_acoustic_tune::transforms::chroma` returns exactly
`[f64; 12]`, octave-invariant and tested. No new dependency, no data acquisition, no corpus.

**Why it is not "subjective musical scoring".** The profiles are published experimental constants and
the decision rule is argmax of a correlation — an explicit rubric, which is what #195's non-goal
requires.

**It does not duplicate GA.** GA's `ga_key_from_progression` infers key from a *symbolic chord
progression*. This infers key from a *weighted pitch-class distribution*, which is a different input
and the one an audio pipeline actually produces. GA would be a consumer, not a competitor.

### Test fixture plan

Three tiers, in build order. The metamorphic tier is the important one — it needs no ground-truth
corpus at all, which is what keeps this `free-local`.

| Tier | Fixture | Oracle | Cost |
| --- | --- | --- | --- |
| 1. Analytic | A synthetic PCP with unit weight on `{0,2,4,5,7,9,11}` (C major scale), zero elsewhere | Must return C major. Same set rotated by `k` must return the key `k` semitones up. | free |
| 2. **Metamorphic** | Any PCP `v`, and its rotation `rot(v, k)` for all `k ∈ 0..11` | **Transposition equivariance**: `key(rot(v,k)).tonic == (key(v).tonic + k) mod 12`, and mode unchanged. Holds for *every* input, so it can be property-tested over random vectors with no labels. | free |
| 3. End-to-end | A synthesised C-major triad arpeggio → `chroma()` → `key_from_pcp` | Returns C major with `r` above the runner-up by a stated margin. Wires the two crates together — the tracer-bullet slice. | free |

Tier 2 is the strongest guard: it is an exact algebraic invariant of the algorithm, it cannot be
satisfied by a lookup table keyed on the test inputs, and it would catch an off-by-one in the
rotation logic, which is the realistic bug here.

**Baseline and guardrail** (per CLAUDE.md's "instrument before you ship"): there is no current IX
key-finder, so the baseline is *absent*, not zero — the honest declaration is that tiers 1–3 must
pass at 100%, and the guardrail is that tier 2 holds for all 12 rotations on randomised input. No
accuracy claim against real music should be made without a labelled corpus, which this proposal does
not include and does not need.

## Follow-up tasks, by repo

Proposed, **not opened** — #195 asks for candidates and its non-goals forbid product implementation.

### IX

| # | Proposed issue | Rows | Oracle | cx | prio |
| --- | --- | --- | --- | --- | --- |
| 1 | Krumhansl–Schmuckler `key_from_pcp` + the three-tier fixture set | B1 | published profiles; transposition equivariance | XS | P1 |
| 2 | Expose `ix-bracelet` through MCP (`ix_bracelet`) | §What IX already has | `EXPECTED[]` count assertion in `parity.rs` + per-op smoke tests | S | P1 |
| 3 | Chord-template matching over a chroma vector | C2 | fixed binary templates; exact on synthetic input | XS | P2 |
| 4 | Correct `parity.rs:124`'s stale "(93)" doc comment | — | the array count itself | XS | P3 |
| 5 | Counterpoint interval checks — **hold until a consumer exists** | B5 | rule-exact | S | P3 |
| 6 | Bring `docs/research/` into the `ix-streeling` indexer's scope | — | catalog contains an entry per research doc | XS | P3 |

Issue 2 is the highest-leverage item in this document that is not the prototype: IX's largest music
crate is agent-invisible. Note the parity cascade — `EXPECTED[]` must be bumped in the same PR, and
stacked tool PRs cannot merge in parallel.

Issue 6 comes from a discoverability gap found while writing this: `state/streeling/catalog.jsonl`
holds 102 entries covering `docs/solutions` (61), `docs/plans` (27) and `docs/brainstorms` (14), and
**nothing** from `docs/research/`. All eight research documents — including #190's inventory, #202's
matrix, #192's ToT packet and this one — are absent from the catalog and therefore invisible to
`streeling search`. That is a scope gap in the indexer, not an omission by any one PR, which is why
this document does not add a catalog entry of its own.

### GA (`GuitarAlchemist/ga`, roadmap [#482](https://github.com/GuitarAlchemist/ga/issues/482))

| # | Proposed | Rows |
| --- | --- | --- |
| 6 | Drop-2 / drop-3 voicing generator beside the existing enumerator | A5 |
| 7 | Consume IX `key_from_pcp` once issue 1 lands, for audio-derived key detection | B1 |

### guitar-singularity

| # | Proposed | Rows |
| --- | --- | --- |
| 8 | Difficulty-weighted practice-path generation over a concept prerequisite graph, calling IX's `topological_sort` / weighted shortest path rather than reimplementing them | D1, D2 |

> **Broken reference in #195.** The issue lists `Related: GuitarAlchemist/guitar-singularity#487`.
> That repository contains exactly **one** issue — `#1`, "[PI][Guitar Singularity] Music intelligence
> roadmap and issue hierarchy", currently **closed**. `#487` has never existed. Route
> guitar-singularity follow-ups to `#1` or open a fresh issue; do not propagate `#487`.
> `GuitarAlchemist/ga#482` and `GuitarAlchemist/tars#71` both resolve and are open.

### TARS

No follow-up. Nothing in the V1 music corpus is worth porting, and §F recommends that the two
`.tars.md` specs be annotated to strike their quaternion dependency rather than deleted — they remain
useful as requirement checklists for GA.

## Cost notes

| Item | Value |
| --- | --- |
| Budget from #195 | `free-local`, `max_cost_usd: 0`, `max_runner_minutes: 30` |
| Actual spend | **0 USD** |
| Method | local `git ls-tree` / `git grep` / `git show` against the pinned tars checkout; local `grep` / `wc` / `python` over this worktree; `gh issue view` and `gh repo view` for cross-repo reference checks |
| Hosted passes | none — no embedding, summarization or model call over any transcript |
| Compute | one full `cargo +nightly-2026-08-23 clippy --workspace --all-targets -- -D warnings` (6m58s, **exit 0**); no `cargo test`. This change adds no Rust and touches no `.github/workflows/**` |

## Method, and its limits

1. Read #195, #190's inventory, and #202's matrix in full before writing any row.
2. Confirmed the pinned SHA exists as a commit object in the sibling checkout. That checkout's `HEAD`
   is on an unrelated branch (`refactor/reason-feedback-seam` @ `9490f73`), so **every** TARS read
   went through explicit `<SHA>:<path>` addressing — no row can have picked up post-pin content.
3. **Path sweep**: full `git ls-tree -r <SHA>` (15,090 paths) filtered on
   `guitar|music|fretboard|fret|chord|scale|harmon|voicing|tablature|midi|note|pitch|interval|tuning|arpegg|melod|rhythm`.
4. **Content sweep 1** (path-agnostic): `git grep -l` for
   `fretboard|voice.leading|voicing|pitch.class|interval.vector|chord progression|roman numeral|pentatonic|diatonic|modal interchange|tritone|circle of fifths`.
5. **Content sweep 2** (different vocabulary):
   `arpegg|cadence|counterpoint|twelve.tone|dodecaphon|set class|forte number|neo.riemannian|tonnetz|drop2|capo|strum|tablature|fingering|voice.lead`.
   This found two sources sweep 1 missed, which is why it is reported as a separate step.
6. Read every surviving candidate in full or, for the long transcripts, read the matched regions.
7. Verified TARS's own internal references rather than trusting them — that is what established that
   `MusicalQuaternions` is real (unlike #202's phantom closures) and that the `enhance*` functions
   citing it are string literals.
8. Re-derived the §F arithmetic independently in Python over the actual 49-semitone guitar range,
   rather than reasoning about the F# by eye.
9. Mapped candidates onto IX by reading the `pub` surface of every file cited, and cross-checked
   exposure against the UDF registrations under `crates/ix-duck*/src/` and the `EXPECTED` array in
   `crates/ix-agent/tests/parity.rs`.

### Unresolved and not reached

Stated explicitly, because a survey that hides its own gaps is worse than no survey.

- **`ChatGPT-DSLs for Visualization Engines.md` was not read end to end.** Only its music sub-DSL
  section (L294–360) was read, and its line count was not measured. Its verdict (§E, reject on
  #195's UI/rendering non-goal) does not depend on the remainder, but a buried algorithm could have
  been missed.
- **`GUITAR_ALCHEMIST_TARS_INTEGRATION_PLAN.md` and the SR&ED report were skimmed by grep**, not read
  in full. Both are plan/claim documents with no code; #190 already classified the former as
  superseded by the live MCP federation.
- **GA's internals were not inspected.** Every `owner: GA` verdict rests on the boundary rule plus
  the GA MCP tool surface visible from this session (`ga_polychord`, `ga_chord_substitutions`,
  `ga_easier_voicings`, `ga_key_from_progression`, `ga_progression_completion`,
  `ga_diatonic_chords`, `ga_set_class_subs`) and on `ix-voicings`' documented shell-out. I did not
  read GA source, so "GA already has it" is an inference from an exposed tool name, not from code.
- **The DuckDB UDFs were not executed.** Their existence and registration files are verified by
  reading `crates/ix-duck/src/{bracelet,grothendieck,serial}.rs`; their runtime behaviour is
  corroborated by `ix-bracelet`'s 117 tests (e.g. `forte.rs:30`, `forte.rs:490` asserting `"3-11"`),
  not by running a query. `crates/ix-duck` is excluded from the workspace, so exercising them would
  need a separate build that this `free-local` budget did not cover.
- **`governance/demerzel/schemas/capability-registry.json` was not read** — the submodule is
  uninitialized in this worktree (`git submodule status` shows `-e4e4273…`), so the federation
  capability registry could not be cross-checked against the ownership verdicts.
- **Negative results are sweep-bounded.** Every `—` means the recorded regexes found nothing across
  `crates/**/*.rs`. Sweep 2 finding two files sweep 1 missed demonstrates that this risk is real, not
  hypothetical, on the TARS side too. An implementation under an unguessable name would read as a
  false gap.
- **No claim in any TARS source document has been validated**, and no IX algorithm cited here was
  re-verified for correctness. `already_done` means *the code exists and is under test*, not *it is
  right*.
- **`cargo test` was not run.** The pinned clippy gate
  (`cargo +nightly-2026-08-23 clippy --workspace --all-targets -- -D warnings`) *was* run in this
  worktree and passed clean (exit 0). The test suite was not, since this change adds no Rust; note
  that `ix-agent/tests/showcase_r1_migrations.rs` fails in a fresh worktree regardless, because
  `governance/demerzel` is an uninitialized submodule here.
