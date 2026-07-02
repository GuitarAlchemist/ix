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

> **Which chord set-classes behave alike along the neck, and which set-class is the
> structural hub of that geometry?**

The demo answers this for **two axes**, one mesh each:

- **Position** (`minFret`) — *where* on the neck a set-class's voicings sit.
- **Stretch** (`fretSpan`) — *how ergonomically wide* its shapes are (a difficulty lens).

The corpus (`state/voicings/raw/guitar.jsonl`, ~667 k fingerings, of which 558 k are
classifiable into 136 Forte set-classes) is read directly with `read_json_auto` — no
ingest step, no separate database.

> **Corpus availability.** The full raw dump is gitignored (110 MB), so it exists only
> on a machine that has run `ix-voicings`. On a fresh checkout the demo automatically
> falls back to the tracked 500-voicing sample (`state/voicings/guitar-corpus.json`,
> same schema) and scales its thresholds down — so it always runs, but the headline
> 5-29 hub result below needs the full corpus. To produce it: `cargo run -p ix-voicings`.

## The mesh is "both layered"

This is the maximal-scope shape of ADR-0004: a graph of **~460 operators** whose
**~120 stream** outputs feed an N×N correlation mesh.

- A **stream** is one Forte set-class. The aligned axis is **fret position**
  (`minFret`), so each stream is that set-class's *fretboard-position profile* —
  how its voicings distribute across the neck.
- The **operator graph** is reified as an `ix_pipeline::dag::Dag` and printed:

  ```text
  read_json_auto ─▶ ix_forte_number (annotate) ─▶ GROUP BY <axis> (bin)    [shared head]
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

## The raw mesh output

Over the best-supported set-classes (≥ 1000 voicings each), the two axes report
**different** betweenness-leading set-classes — suggesting the geometry of *where*
chords sit differs from the geometry of *how wide* they stretch:

| Axis | Streams | Betweenness-leader (τ = 0.8) | Betweenness |
|---|---|---|---|
| **Position** (`minFret`) | 114 | **5-29** | ≈ 112 (3.5× runner-up) |
| **Stretch** (`fretSpan`) | 120 | **2-3** | ≈ 185 (2.2× runner-up) |

These are the mesh's *raw* outputs, not yet conclusions. Both leaders are well-supported
(5-29: 8 568 voicings; 2-3: 2 024), so neither is a *sparse-profile* artifact — but
"well-supported" is not "significant". The **Validation** section below subjects both to
a null model, and only one survives.

### The τ-sweep: one web fracturing into named regions

The `|r|` threshold `τ` is the operating lever (ADR-0004). At `τ = 0.8` each axis is a
single connected web; raising `τ` fractures it, and the demo names each surviving
region by the fret-band its members *distinctively over-index on* (the argmax of their
mean residual — what the mesh actually clustered on, not the raw peak, which is "open"
for almost every set-class):

```text
POSITION:  τ=0.80 → 1 region   τ=0.90 → 2   τ=0.95 → 3   τ=0.98 → 3
STRETCH:   τ=0.80 → 1 region   τ=0.90 → 3   τ=0.95 → 3   τ=0.98 → 5
```

On the stretch axis the split is the more interesting: a dominant **wide-band** region
peels off small **moderate-band** regions (e.g. `{3-10, 4-9, 6-33, 6-32}`), i.e.
clusters of set-classes that prefer a moderate stretch where the bulk prefer wide.

## Validation (null model)

A betweenness "hub" is only meaningful if the real data's hub-concentration exceeds what
the pipeline manufactures from structureless input. We test each axis against **500 null
meshes**, each built by **shuffling every bin's counts across set-classes** — a
permutation that *keeps* the realistic common-mode trend (low frets / wide spans are
crowded) but *destroys* any genuine per-set-class co-variation. The same pipeline
(normalize → common-mode removal → |Pearson| → τ = 0.8 → betweenness) runs on real and
null alike; the statistic is the top betweenness score, and `p` is the fraction of nulls
that match or beat the real value. Reproduce with:

```bash
cargo run -p ix-duck --example ix_voicing_mesh_nullcheck --features duck
```

| Axis | Real top betweenness | Null (mean / 95th / max) | `p`(real ≤ null) | Verdict |
|---|---|---|---|---|
| **Position** | 119.9 | 0.1 / 1.0 / 3.0 | **0.002** | ✅ **signal** |
| **Stretch** | 122.1 | 259 / 458 / 805 | **0.996** | ❌ **artifact** |

Three honest conclusions:

1. **Position structure is real.** Shuffling collapses betweenness to ≈ 0 (the null graph
   has no bridges), while the real positional residuals form a genuine cluster-and-bridge
   backbone (betweenness ≈ 120, `p` = 0.002). Guitar voicing set-classes really do have
   **non-random positional co-variation**.
2. **The single-hub *identity* is fragile.** The harness (which omits the wavelet-smoothing
   stage) names **4-17**, not 5-29, with the top two nearly tied. So *"there is real
   positional structure"* is supported; *"5-29 is **the** hub"* is **overclaimed** — the
   named leader shifts with small pipeline changes.
3. **The stretch result is an artifact.** Real stretch betweenness sits *below* the null
   median (`p` = 0.996): with only 5 `fretSpan` bins the residual space is too coarse to
   carry signal, and random data produces *more* hub structure than the real corpus. The
   `2-3` stretch "hub" does **not** survive — treat it as noise.

The lesson generalises: **this mesh, like any correlation method, needs a null model before
its hubs become claims.** The position finding clears that bar; the stretch finding does not.

## Scope and caveats

- **Advisory analysis only.** Per ADR-0004 the mesh is never a binding gate or a
  source of truth. Binding verdicts go through the governed `maintain-gate`
  (ADR-0002).
- The composition language here is **DuckDB SQL**, not IXQL — they stay complementary
  (ADR-0001 + ADR-0004).
- `minFret` ≥ 18 voicings are dropped (a small tail); the demo uses `FRETS = 18`.
- Tuning knobs live as constants at the top of the example: `FRETS`, `MIN_SUPPORT`,
  `TOP_K`, `THRESHOLD`.
