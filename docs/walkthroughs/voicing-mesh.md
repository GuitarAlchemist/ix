# `ix_voicing_mesh` — a 100+ node pipeline mesh over the guitar voicing corpus

A runnable, real-world demonstration of the **pipeline-mesh substrate**
([ADR-0004](../adr/0004-duckdb-sql-pipeline-mesh.md)): compose ~120 IX "pipelines"
over the GA guitar voicing corpus on the DuckDB analyst bench and correlate their
outputs to find structure.

> _Version française : [`docs/fr/pas-a-pas/maillage-voicings.md`](../fr/pas-a-pas/maillage-voicings.md)._

Run it:

```bash
cargo run -p ix-duck --example ix_voicing_mesh --features duck
```

## The question

> **Which chord set-classes occupy the same fretboard regions, and which set-class
> is the structural hub of the guitar's voicing geometry?**

The corpus (`state/voicings/raw/guitar.jsonl`, ~667 k fingerings, of which 558 k are
classifiable into 136 Forte set-classes) is read directly with `read_json_auto` — no
ingest step, no separate database.

## The mesh is "both layered"

This is the maximal-scope shape of ADR-0004: a graph of **~460 operators** whose
**~120 stream** outputs feed an N×N correlation mesh.

- A **stream** is one Forte set-class. The aligned axis is **fret position**
  (`minFret`), so each stream is that set-class's *fretboard-position profile* —
  how its voicings distribute across the neck.
- The **operator graph** is reified as an `ix_pipeline::dag::Dag` and printed:

  ```text
  read_json_auto ─▶ ix_forte_number (annotate) ─▶ GROUP BY minFret (bin)   [shared head]
    ─▶ profile:k ─▶ normalize:k ─▶ residual:k ─▶ smooth:k                    [×120 streams]
                         └────────▶ common-mode ────────┘                    [barrier]
    ─▶ ix_pearson (N×N) ─▶ |r| ≥ τ ─▶ ix_connected_components ─▶ ix_centrality [shared tail]
  ```

  A typical run reports **464 operators, 803 edges, depth 12, fan-out 114** — a
  genuine 100+ node pipeline graph, not a conceptual one.

Every stage is an IX UDF on the DuckDB bench: `ix_forte_number`, `ix_wavelet_denoise`,
`ix_pearson`, `ix_connected_components`, `ix_centrality`. No statistics are
reimplemented in the demo.

## The load-bearing insight: common-mode removal

The first (tracer-bullet) run with 8 streams exposed a confound that the full run
would have hidden: **every** common set-class crowds the low frets, so the raw
profiles all correlate ~1.0 — one giant clique, betweenness 0 for every node, no
hub. The shared "most voicings live low on the neck" trend swamps the signal.

The fix is standard common-mode removal (as in market-model residuals or fMRI
global-signal regression), and it lives in the **stream-construction pipeline**, not
the generic mesh:

1. normalize each profile to a positional **distribution** (so a common set-class
   doesn't outweigh a rare one — Pearson is already scale-invariant, but this makes
   step 2 clean);
2. subtract the **cross-set-class mean** distribution per fret, leaving each
   set-class's *anomaly* — where it over- or under-indexes versus the typical
   set-class.

Pearson on these residuals correlates *distinctive* co-location, which is what the
question actually asks.

> This is the tracer-bullet discipline working as intended: build the thinnest
> end-to-end slice first, let it surface the unknown, then scale.

## The finding

At `τ = 0.8` over the 114 best-supported set-classes (≥ 1000 voicings each, so every
fret bin is populated and a sparse profile can't fake a bridge):

- The set-classes form **one connected fretboard-geometry web** — no isolated chord
  families; they are all positionally inter-related.
- **5-29 is the structural hub** — betweenness **≈ 112**, about 3.5× the runner-up,
  and well-supported (8 568 voicings), so it is not a sparse-profile artifact. It is
  the set-class whose fretboard-position residual most bridges the others.

The operating lever is the `|r|` threshold `τ` (ADR-0004): at `τ = 0.4` the de-trended
mesh is still one near-complete clique with degenerate betweenness ties; raising `τ`
sharpens the hub.

## Scope and caveats

- **Advisory analysis only.** Per ADR-0004 the mesh is never a binding gate or a
  source of truth. Binding verdicts go through the governed `maintain-gate`
  (ADR-0002).
- The composition language here is **DuckDB SQL**, not IXQL — they stay complementary
  (ADR-0001 + ADR-0004).
- `minFret` ≥ 18 voicings are dropped (a small tail); the demo uses `FRETS = 18`.
- Tuning knobs live as constants at the top of the example: `FRETS`, `MIN_SUPPORT`,
  `TOP_K`, `THRESHOLD`.
