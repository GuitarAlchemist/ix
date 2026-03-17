---
name: federation-music
description: Combine GA music theory tools with ix ML algorithms for music analysis
---

# Federation Music Analysis

Cross-repo music analysis combining GA's 50+ music theory tools with ix's ML capabilities.

## When to Use
When the user needs ML-powered music analysis — clustering chord progressions, spectral analysis, pattern detection in music data.

## Quick Start
Call `ix_ga_bridge` with action=workflow_guide for a complete list of GA-to-ix workflows.

## Capabilities
- **GA** provides: chord parsing (`GaParseChord`), progression analysis (`GaAnalyzeProgression`), atonal set theory (`GaChordToSet`), scale data (`GetAvailableScales`), voice leading (`GaCommonTones`), arpeggio suggestions (`GaArpeggioSuggestions`)
- **ix** provides: FFT (`ix_fft`), K-Means (`ix_kmeans`), PCA, supervised classification (`ix_supervised`), ML pipeline (`ix_ml_pipeline`), code analysis (`ix_code_analyze`)

## Bridge Tool
`ix_ga_bridge` converts GA data to ML-ready features:
- `chord_features` — Chord symbols to interval/pitch-class vectors
- `progression_features` — Progression to feature matrix
- `scale_features` — Scale data to binary pitch-class sets
- `workflow_guide` — Show all available GA-to-ix workflows

## Example Workflows

### Cluster chord voicings by harmonic similarity
1. `GaParseChord` — parse chord symbols
2. `GaChordToSet` — get pitch-class set + ICV
3. `ix_kmeans` — cluster by ICV vectors

### Classify progressions by style
1. `GaAnalyzeProgression` — detect key, get Roman numerals
2. `GaCommonTones` — compute voice-leading distances
3. `ix_ml_pipeline` — train style classifier

### Analyze harmonic complexity
1. `GaChordToSet` — get pitch-class analysis
2. `ix_stats` — statistical summary
3. `ix_chaos_lyapunov` — measure harmonic unpredictability

### Voice-leading optimization
1. `GaCommonTones` — common tones between chords
2. `ix_search` — A* shortest path in voice-leading space
3. `ix_optimize` — minimize total voice-leading distance

### Scale recommendation
1. `GaAnalyzeProgression` — detect key
2. `GaArpeggioSuggestions` — per-chord arpeggios and modes
3. `ix_supervised` — train scale recommender
