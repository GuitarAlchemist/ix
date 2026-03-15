---
name: federation-music
description: Combine GA music theory tools with ix ML algorithms for music analysis
---

# Federation Music Analysis

Cross-repo music analysis combining GA's music theory tools with ix's ML capabilities.

## When to Use
When the user needs ML-powered music analysis — clustering chord progressions, spectral analysis, pattern detection in music data.

## Capabilities
- **GA** provides: scale intervals, chord voicings, atonal analysis, fretboard mapping
- **ix** provides: FFT (spectral analysis), K-Means (clustering), PCA (dimensionality reduction), supervised classification

## Example Workflows

### Cluster chord voicings by timbral similarity
1. Use `ga_chord` to get voicing data
2. Use `ix_fft` to extract spectral features
3. Use `ix_kmeans` to cluster voicings

### Classify chord progressions by style
1. Use `ga_dsl` to parse progression notation
2. Extract feature vectors (intervals, voice leading)
3. Use `ix_supervised` to train a style classifier

### Analyze harmonic complexity
1. Use `ga_chord_atonal` for pitch class analysis
2. Use `ix_stats` for statistical summary
3. Use `ix_chaos_lyapunov` to measure harmonic unpredictability
