# OPTIC-K Search Latency Baseline — 2026-04-16

First real-scale measurement of `ix-optick` brute-force cosine search on
the post-dedup v4 index. Establishes the baseline against which any
future index structure (HNSW, IVF, quantization) must compete.

## Index under test

- **Format**: OPTK v4 (112-dim vectors, metadata offset table, sorted by instrument)
- **Corpus**: 313,047 unique voicings, built from GA's `FretboardVoicingsCLI --export-embeddings`
- **File size**: 165 MB
- **Dedup key**: `(relativeFretSignature, pitchClassBitmask)` per instrument, lowest `CheapPlayabilityCost` tiebreaker
- **Source raw count before dedup**: 688,351

### Per-instrument partition sizes

| Instrument | Unique voicings | % of corpus |
|---|---:|---:|
| Guitar | 297,910 | 95.2% |
| Bass | 7,795 | 2.5% |
| Ukulele | 7,342 | 2.3% |

## Results

Benchmarks: `cargo bench -p ix-optick --bench search` with
`OPTICK_INDEX_PATH` pointing at the index file. Criterion default
iteration budgets, 10–20 samples per bench.

| Bench | Mean | 95% CI | Notes |
|---|---:|---|---|
| `optick_open` | **27.8 µs** | [27.6, 27.9] | mmap + header parse, N-independent |
| `search_unfiltered top-10` | **17.3 ms** | [17.2, 17.5] | full corpus scan (313k × 112 dots) |
| `search_guitar top-10` | **16.8 ms** | [16.5, 16.9] | 297,910 × 112 dots |
| `search_bass top-10` | **220 µs** | [219, 222] | 7,795 × 112 dots |
| `search_ukulele top-10` | **211 µs** | [209, 212] | 7,342 × 112 dots |

### Key takeaways

1. **Brute-force is fast enough at this scale.** Full-corpus search is 17 ms —
   well under the 50 ms interactive threshold.
2. **Instrument pre-filtering gives ~80× speedup for small partitions.** The
   sorted layout + offset table means a bass query scans only 2.5% of the
   file at ~220 µs.
3. **Open cost is negligible** (27.8 µs) and N-independent because mmap is
   lazy and the header is a fixed ~568 bytes.
4. **Ranking cost dominates, not top-k selection.** Top-10 and top-100 should
   be essentially indistinguishable (the min-heap is cheap vs the dot-product
   scan).

## ANN decision — not yet

The brute-force numbers don't justify investment in HNSW / IVF-PQ at this
scale. Approximate nearest neighbor indexes start paying off when:

- The corpus grows to **5–10 M voicings** (projected 17 ms × 15 ≈ 250 ms),
- Or we need **p99 < 10 ms** for concurrent user queries, or
- We add a **re-ranking cascade** where embedding similarity is only the
  first stage.

Until then: brute-force mmap + instrument pre-filter.

## Reproducing

```bash
# Regenerate the real index (takes ~55 seconds)
cd ~/source/repos/ga
dotnet run --project "Demos/Music Theory/FretboardVoicingsCLI" -c Release -- \
  --export-embeddings --output state/voicings/optick.index

# Run benchmarks against it
cd ~/source/repos/ix
OPTICK_INDEX_PATH="$HOME/source/repos/ga/state/voicings/optick.index" \
  cargo bench -p ix-optick --bench search
```

HTML reports land in `target/criterion/`.

## Follow-ups worth measuring when relevant

- **Before a v5 quantization (int8/f16) lands** — rerun this suite and compare
  ranking quality + speed. Acceptance gate: top-10 overlap ≥ 0.9 vs f32, and
  ≥ 2× throughput improvement.
- **Before introducing an ANN index** — rerun and confirm the corpus has
  grown past ~5 M, or that p99 on concurrent queries exceeds targets.
- **After changing the embedding schema (v5+)** — rerun open/search to
  detect regressions from wider headers or larger vectors.
