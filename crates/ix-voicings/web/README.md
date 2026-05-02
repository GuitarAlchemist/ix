# ix-voicings t-SNE viewer

Single-page Canvas viewer for the OPTIC-K voicing embedding projected to 2D
via t-SNE (`crates/ix-voicings/src/bin/tsne_voicings.rs`).

## Run

Generate the data once, then start the viewer:

```sh
# 1. Project the voicing index to 2D (Barnes-Hut, ~5 min for 313K points).
cargo run --release -p ix-voicings --bin tsne_voicings -- \
    --index ../ga/state/voicings/optick.index \
    --output state/viz/voicings-tsne-313k-perp80.json \
    --algorithm barnes-hut \
    --sample 313047 \
    --perplexity 80 \
    --iterations 2000

# 2. (Optional) Precompute cluster + neighbor metadata for tooltips.
cargo run --release -p ix-voicings -- viz-precompute

# 3. Serve the viewer (default port 8765, default data dir state/viz/).
cargo run -p ix-voicings --bin serve_viz
# → open http://127.0.0.1:8765/
```

The viewer fetches three files from `state/viz/`:

| File                              | Required | Notes                                        |
|-----------------------------------|----------|----------------------------------------------|
| `voicings-tsne-313k-perp80.json`  | yes      | t-SNE points (~38 MB for 313K voicings)      |
| `voicing-details.json`            | yes      | Chord metadata for tooltips (~800 KB subset) |
| `cluster-layout.json`             | optional | Cluster labels — placed at each cluster's representative voicing |

## Controls

- **scroll** to zoom (cursor stays anchored)
- **drag** to pan
- **hover** to see chord name, frets, MIDI, pitch-classes
- **checkboxes** filter by instrument or hide cluster labels

## Notes

- All 313K points render to a single Canvas every frame — this is fine on
  modern hardware but may stutter during pan on integrated GPUs. Filter
  out one instrument if it gets sluggish.
- `state/viz/` is gitignored. Regenerate with the t-SNE bin above.
