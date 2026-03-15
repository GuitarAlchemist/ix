---
name: music-ml-demo
description: Concrete, runnable examples combining GA music theory tools with ix ML algorithms for chord clustering, harmonic analysis, and scale recommendation
---

# Music ML Analysis — Runnable Examples

Detailed walkthroughs that combine GA's music theory MCP tools with ix's ML algorithms. Each example includes exact tool invocations, expected data formats, and interpretation guidance.

## When to Use

- When you want to **demonstrate** the GA + ix federation with real music data
- When a user asks for music analysis that goes beyond basic theory (clustering, spectral analysis, classification)
- When exploring timbral similarity, harmonic complexity, or scale recommendation
- As a teaching tool for how federation workflows connect multiple MCP servers

## Prerequisites

- GA MCP server running (provides `ga_chord`, `ga_chord_atonal`, `ga_scale`, `ga_dsl`)
- ix MCP server running (provides `ix_kmeans`, `ix_stats`, `ix_fft`, `ix_chaos_lyapunov`, `ix_supervised`)

---

## Example 1: Chord Voicing Clustering

**Goal**: Cluster common chord voicings by timbral similarity to discover which voicings sound alike regardless of root note.

### Step 1.1: Gather Voicing Data from GA

Call `ga_chord` for each chord in multiple positions to build a dataset.

**Tool**: `ga_chord`
**Calls** (run these in parallel):
```json
{ "name": "C", "position": 0 }    // open C major
{ "name": "C", "position": 5 }    // barre C at fret 5
{ "name": "C", "position": 8 }    // C at fret 8
{ "name": "Am", "position": 0 }   // open A minor
{ "name": "Am", "position": 5 }   // barre Am at fret 5
{ "name": "Am", "position": 8 }   // Am at fret 8
{ "name": "G", "position": 0 }    // open G major
{ "name": "G", "position": 3 }    // barre G at fret 3
{ "name": "G", "position": 7 }    // G at fret 7
{ "name": "F", "position": 0 }    // barre F at fret 1
{ "name": "F", "position": 5 }    // F at fret 5
{ "name": "F", "position": 8 }    // F at fret 8
```

**Expected output per call**:
```json
{
  "name": "C",
  "position": 0,
  "midi_notes": [48, 52, 55, 60, 64],
  "intervals": [4, 3, 5, 4],
  "register_mean": 55.8,
  "span_semitones": 16
}
```

### Step 1.2: Extract Feature Vectors

From each voicing response, construct a feature vector with these dimensions:

| Feature | Description | How to Compute |
|---------|-------------|----------------|
| `interval_content` (6 values) | Interval class vector (IC1–IC6) | Count interval classes between all note pairs |
| `register_mean` | Average MIDI note value | Mean of `midi_notes` |
| `register_spread` | Range of MIDI values | Max - min of `midi_notes` |
| `voice_leading_density` | How close adjacent voices are | Mean of `intervals` array |

This gives a **9-dimensional feature vector** per voicing. Example for open C:
```json
[0, 1, 1, 1, 2, 0, 55.8, 16, 4.0]
```

Assemble all 12 voicings into a feature matrix (12 rows x 9 columns):
```json
{
  "data": [
    [0, 1, 1, 1, 2, 0, 55.8, 16, 4.0],
    [0, 1, 1, 1, 2, 0, 67.2, 16, 4.0],
    [0, 1, 1, 1, 2, 0, 74.4, 16, 4.0],
    [1, 1, 0, 1, 2, 0, 53.6, 15, 3.75],
    [1, 1, 0, 1, 2, 0, 65.0, 15, 3.75],
    [1, 1, 0, 1, 2, 0, 72.8, 15, 3.75],
    [0, 1, 1, 1, 2, 0, 54.2, 19, 4.75],
    [0, 1, 1, 1, 2, 0, 57.6, 17, 4.25],
    [0, 1, 1, 1, 2, 0, 68.0, 17, 4.25],
    [1, 1, 1, 0, 2, 0, 56.4, 17, 4.25],
    [1, 1, 1, 0, 2, 0, 67.8, 17, 4.25],
    [1, 1, 1, 0, 2, 0, 74.0, 17, 4.25]
  ],
  "labels": ["C_0", "C_5", "C_8", "Am_0", "Am_5", "Am_8", "G_0", "G_3", "G_7", "F_0", "F_5", "F_8"]
}
```

### Step 1.3: Cluster with K-Means

**Tool**: `ix_kmeans`
**Input**:
```json
{
  "data": [[0,1,1,1,2,0,55.8,16,4.0], "... (all 12 rows)"],
  "k": 3,
  "max_iterations": 100,
  "seed": 42
}
```
**Expected output**:
```json
{
  "assignments": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2],
  "centroids": [
    [0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 65.8, 16.0, 4.0],
    [1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 63.8, 15.0, 3.75],
    [0.33, 1.0, 1.0, 0.33, 2.0, 0.0, 62.3, 17.5, 4.29]
  ],
  "inertia": 234.5,
  "iterations": 8
}
```

**Interpretation**: The clusters tend to group by interval content (major vs. minor quality) and register spread rather than by root note — voicings with similar harmonic structure cluster together regardless of pitch.

### Step 1.4: Characterize Each Cluster

**Tool**: `ix_stats` (call once per cluster)
**Input** (for cluster 0):
```json
{
  "data": [55.8, 67.2, 74.4],
  "compute": ["mean", "std", "min", "max"]
}
```
**Expected output**:
```json
{
  "mean": 65.8,
  "std": 9.3,
  "min": 55.8,
  "max": 74.4
}
```

**Cluster labels** (assign after inspecting centroids):
- Cluster 0: "Bright major voicings" — major interval content, moderate register
- Cluster 1: "Warm minor voicings" — minor interval content, compact voicing
- Cluster 2: "Spread voicings" — wider register span, mixed quality

---

## Example 2: Harmonic Complexity Analysis

**Goal**: Measure the harmonic complexity and predictability of a chord progression using spectral and chaos analysis.

### Step 2.1: Get Pitch Class Sets

Analyze the progression: `Cmaj7 → Am7 → Dm7 → G7` (a ii-V-I turnaround variant).

**Tool**: `ga_chord_atonal` (call for each chord)
**Calls**:
```json
{ "notes": [0, 4, 7, 11] }   // Cmaj7 → pitch classes {0, 4, 7, 11}
{ "notes": [9, 0, 4, 7] }    // Am7 → pitch classes {9, 0, 4, 7}
{ "notes": [2, 5, 9, 0] }    // Dm7 → pitch classes {2, 5, 9, 0}
{ "notes": [7, 11, 2, 5] }   // G7 → pitch classes {7, 11, 2, 5}
```

**Expected output per call**:
```json
{
  "pitch_classes": [0, 4, 7, 11],
  "interval_vector": [2, 1, 1, 1, 1, 0],
  "forte_number": "4-20",
  "complement": [1, 2, 3, 5, 6, 8, 9, 10]
}
```

### Step 2.2: Compute Interval Vectors

Collect the interval vectors from all four chords:
```
Cmaj7: [2, 1, 1, 1, 1, 0]
Am7:   [2, 1, 1, 1, 1, 0]
Dm7:   [2, 1, 1, 1, 1, 0]
G7:    [2, 1, 1, 1, 1, 0]
```

Note: All four chords are of the same set class (4-20), so their interval vectors are identical. This is a hallmark of smooth jazz harmony — same structure, different transposition.

To capture the *movement* between chords, compute the **root motion series** as semitone intervals:
```
C→Am = -3 (or 9)
Am→Dm = 5
Dm→G = 5
G→C = 5
```

Root motion signal: `[9, 5, 5, 5]`

### Step 2.3: Spectral Analysis of Harmonic Rhythm

**Tool**: `ix_fft`
**Input**:
```json
{
  "signal": [9, 5, 5, 5, 9, 5, 5, 5, 9, 5, 5, 5, 9, 5, 5, 5],
  "sample_rate": 4
}
```

We repeat the 4-chord pattern 4 times to give the FFT enough data to detect periodicity.

**Expected output**:
```json
{
  "frequencies": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
  "magnitudes": [96.0, 8.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
  "dominant_frequency": 0.25,
  "dominant_magnitude": 8.0
}
```

**Interpretation**: The dominant frequency at 0.25 (period = 4 beats) confirms the 4-chord cycle. The relatively low magnitude of the dominant peak vs. DC component (8.0 vs 96.0) indicates a progression that is harmonically stable with mild periodic variation — characteristic of pop/jazz turnarounds.

### Step 2.4: Chaos/Predictability Analysis

**Tool**: `ix_chaos_lyapunov`
**Input**:
```json
{
  "time_series": [9, 5, 5, 5, 9, 5, 5, 5, 9, 5, 5, 5, 9, 5, 5, 5],
  "embedding_dim": 2,
  "tau": 1
}
```

**Expected output**:
```json
{
  "lyapunov_exponent": -0.42,
  "is_chaotic": false,
  "interpretation": "negative_stable"
}
```

**Interpretation**:
- **Negative Lyapunov exponent** (-0.42): The progression is highly predictable. Nearby trajectories in phase space converge, meaning the harmonic motion is stable and periodic.
- For comparison, a chromatic or atonal progression would yield a **positive** exponent (> 0), indicating chaotic/unpredictable harmony.
- A value near zero would indicate "edge of chaos" — complex but not random (think: Coltrane changes).

### Complexity Scale Reference

| Lyapunov Exponent | Harmonic Character | Example |
|---|---|---|
| < -0.5 | Highly predictable, repetitive | 12-bar blues, I-IV-V |
| -0.5 to 0 | Structured with mild variation | Jazz standards, pop |
| ~0 | Complex, structured unpredictability | Coltrane changes, late Romantic |
| > 0 | Chaotic, atonal | Free jazz, serial music |

---

## Example 3: Scale Recommendation via Classification

**Goal**: Given a target mood (e.g., "mysterious"), recommend scales by classifying scale interval patterns against mood labels.

### Step 3.1: Gather Scale Data

**Tool**: `ga_scale` (call for each scale)
**Calls**:
```json
{ "name": "major", "root": "C" }
{ "name": "natural_minor", "root": "C" }
{ "name": "harmonic_minor", "root": "C" }
{ "name": "melodic_minor", "root": "C" }
{ "name": "dorian", "root": "C" }
{ "name": "phrygian", "root": "C" }
{ "name": "lydian", "root": "C" }
{ "name": "mixolydian", "root": "C" }
{ "name": "locrian", "root": "C" }
{ "name": "whole_tone", "root": "C" }
{ "name": "diminished", "root": "C" }
{ "name": "pentatonic_major", "root": "C" }
{ "name": "pentatonic_minor", "root": "C" }
{ "name": "blues", "root": "C" }
```

**Expected output per call**:
```json
{
  "name": "harmonic_minor",
  "root": "C",
  "intervals": [2, 1, 2, 2, 1, 3, 1],
  "notes": ["C", "D", "Eb", "F", "G", "Ab", "B"],
  "degree_count": 7
}
```

### Step 3.2: Build Feature Matrix

For each scale, extract a feature vector from its interval pattern:

| Feature | Description |
|---------|-------------|
| `semitone_count` | Number of semitone (1) intervals |
| `whole_tone_count` | Number of whole-tone (2) intervals |
| `aug_second_count` | Number of augmented-second (3) intervals |
| `degree_count` | Number of notes in the scale |
| `interval_variance` | Variance of interval sizes |
| `max_interval` | Largest interval in the scale |
| `symmetry` | 1 if palindromic interval pattern, 0 otherwise |

**Feature matrix** (14 rows x 7 columns):
```json
{
  "data": [
    [0, 5, 0, 7, 0.24, 2, 1],
    [2, 3, 0, 7, 0.24, 2, 1],
    [2, 2, 1, 7, 0.57, 3, 0],
    [1, 4, 0, 7, 0.24, 2, 0],
    [1, 4, 0, 7, 0.24, 2, 0],
    [2, 3, 0, 7, 0.24, 2, 0],
    [0, 5, 0, 7, 0.24, 2, 0],
    [1, 4, 0, 7, 0.24, 2, 0],
    [2, 3, 0, 7, 0.24, 2, 0],
    [0, 6, 0, 6, 0.00, 2, 1],
    [4, 0, 0, 8, 0.25, 2, 1],
    [0, 2, 0, 5, 0.96, 3, 0],
    [0, 1, 1, 5, 1.36, 3, 0],
    [0, 1, 1, 6, 1.14, 3, 0]
  ],
  "labels": [
    "major", "natural_minor", "harmonic_minor", "melodic_minor",
    "dorian", "phrygian", "lydian", "mixolydian", "locrian",
    "whole_tone", "diminished",
    "pentatonic_major", "pentatonic_minor", "blues"
  ]
}
```

### Step 3.3: Train Mood Classifier

Assign mood labels to the training data based on established music theory associations:

```json
{
  "mood_labels": [
    "bright",       "sad",          "mysterious",   "jazzy",
    "groovy",       "dark",         "dreamy",       "relaxed",
    "tense",        "ethereal",     "tense",
    "bright",       "bluesy",       "bluesy"
  ]
}
```

**Tool**: `ix_supervised`
**Input**:
```json
{
  "task": "classify",
  "algorithm": "knn",
  "k": 3,
  "features": [
    [0,5,0,7,0.24,2,1], [2,3,0,7,0.24,2,1], [2,2,1,7,0.57,3,0],
    [1,4,0,7,0.24,2,0], [1,4,0,7,0.24,2,0], [2,3,0,7,0.24,2,0],
    [0,5,0,7,0.24,2,0], [1,4,0,7,0.24,2,0], [2,3,0,7,0.24,2,0],
    [0,6,0,6,0.00,2,1], [4,0,0,8,0.25,2,1],
    [0,2,0,5,0.96,3,0], [0,1,1,5,1.36,3,0], [0,1,1,6,1.14,3,0]
  ],
  "labels": [
    "bright", "sad", "mysterious", "jazzy",
    "groovy", "dark", "dreamy", "relaxed",
    "tense", "ethereal", "tense",
    "bright", "bluesy", "bluesy"
  ],
  "seed": 42
}
```

**Expected output**:
```json
{
  "model_trained": true,
  "accuracy": 0.71,
  "confusion_matrix": { "... (omitted for brevity)" },
  "feature_importance": [0.12, 0.08, 0.25, 0.18, 0.22, 0.10, 0.05]
}
```

**Key insight**: `aug_second_count` (feature 3) and `interval_variance` (feature 5) are the strongest predictors of mood — scales with augmented seconds tend toward "mysterious"/"dark", while low variance correlates with "bright"/"relaxed".

### Step 3.4: Recommend Scales for a Target Mood

To find scales that match "mysterious":

**Tool**: `ix_supervised`
**Input**:
```json
{
  "task": "predict",
  "query_features": [
    [2, 2, 1, 7, 0.57, 3, 0],
    [2, 3, 0, 7, 0.24, 2, 0],
    [0, 6, 0, 6, 0.00, 2, 1]
  ],
  "target_mood": "mysterious"
}
```

**Expected output**:
```json
{
  "predictions": ["mysterious", "dark", "ethereal"],
  "distances_to_target": [0.0, 0.33, 0.67],
  "recommendations": [
    { "scale": "harmonic_minor", "mood": "mysterious", "confidence": 0.95 },
    { "scale": "phrygian", "mood": "dark", "confidence": 0.72, "note": "closest neighbor to mysterious" },
    { "scale": "whole_tone", "mood": "ethereal", "confidence": 0.58, "note": "alternative ethereal option" }
  ]
}
```

**Recommendation summary**: For a "mysterious" mood, use **harmonic minor** (highest confidence). If you want to shade toward darkness, try **phrygian**. For a more floating/ambiguous mystery, try **whole tone**.

---

## Combining Examples

These three examples can be chained in a single session:

1. **Cluster voicings** (Example 1) to find which chord shapes work well together
2. **Analyze complexity** (Example 2) of a progression built from those voicings
3. **Recommend scales** (Example 3) to solo over that progression based on target mood

This gives a complete workflow: voicing selection, progression analysis, and melodic palette — all driven by data rather than convention.
