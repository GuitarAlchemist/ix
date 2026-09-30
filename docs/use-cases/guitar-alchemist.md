# Use Case: Guitar Alchemist — AI-Powered Music Theory

> Combining MCTS, genetic algorithms, wavelets, HMM/Viterbi, and vector search to generate chord progressions, optimize voicings, and analyze harmonic structure.

## The Scenario

You're building [Guitar Alchemist](../../) — an AI music learning platform. Given a melody (MIDI notes), you want to:

1. **Generate a chord progression** that harmonizes the melody (MCTS)
2. **Optimize guitar voicings** so fingers move minimally between chords (genetic algorithms)
3. **Classify the harmonic style** of a progression (wavelets + signal analysis)
4. **Find the optimal fingering path** across a fretboard (Viterbi/HMM)
5. **Search for similar progressions** in a database (cosine similarity + GPU)

This is a cross-cutting use case that demonstrates how ix's algorithms combine for a real creative AI application.

## The Pipeline

```
Melody (MIDI notes)
    │
    ▼
MCTS ──────────────── Generate candidate progressions
    │
    ▼
Genetic Algorithm ──── Optimize voicings (minimize finger movement)
    │
    ▼
Wavelet Analysis ───── Extract harmonic features for style classification
    │
    ▼
HMM/Viterbi ────────── Find optimal fretboard path
    │
    ▼
GPU Similarity ──────── Find similar progressions in database
```

## Step 1: Chord Progression Generation with MCTS

Given a melody, explore the space of possible chord progressions using Monte Carlo Tree Search. Each node represents a partial progression; children are possible next chords.

```rust
use ix_search::mcts::{MctsState, mcts_search};

#[derive(Clone)]
struct ProgressionState {
    chords: Vec<usize>,       // Chord indices chosen so far
    melody: Vec<u8>,          // Target melody (MIDI notes)
    key: usize,               // Musical key (0=C, 1=C#, ...)
    chord_tones: Vec<Vec<u8>>, // Which MIDI notes each chord contains
}

impl MctsState for ProgressionState {
    type Action = usize; // Index of next chord to add

    fn legal_actions(&self) -> Vec<usize> {
        let pos = self.chords.len();
        if pos >= self.melody.len() { return vec![]; }

        let melody_note = self.melody[pos] % 12;
        // Filter to chords whose tones contain the current melody note
        (0..self.chord_tones.len())
            .filter(|&c| self.chord_tones[c].contains(&melody_note))
            .collect()
    }

    fn apply(&self, action: &usize) -> Self {
        let mut next = self.clone();
        next.chords.push(*action);
        next
    }

    fn is_terminal(&self) -> bool {
        self.chords.len() >= self.melody.len()
    }

    fn reward(&self) -> f64 {
        if !self.is_terminal() { return 0.5; }
        // Score: variety + tension/release + voice leading smoothness
        let variety = self.chords.iter().collect::<std::collections::HashSet<_>>().len() as f64
            / self.chords.len() as f64;
        let resolution_bonus = self.count_resolutions() as f64 * 0.1;
        (variety + resolution_bonus).min(1.0)
    }
}

// Generate a progression for a C major melody
let melody = vec![60, 62, 64, 67, 65, 64, 62, 60]; // C D E G F E D C
let initial = ProgressionState::new(melody, key_of_c_major());

let best_action = mcts_search(&initial, 1000, 1.41, 42);
// Returns the best first chord; repeat for full progression
```

> See [`examples/search/astar_qstar.rs`](../../examples/search/astar_qstar.rs) for search algorithm patterns.

## Step 2: Voicing Optimization with Genetic Algorithms

Each chord can be played in many positions on a guitar. A genetic algorithm finds voicings that minimize total finger movement across the progression.

```rust
use ix_evolution::genetic::GeneticAlgorithm;
use ix_evolution::traits::EvolutionResult;
use ndarray::Array1;

// Each gene = fret position for each string (-1 = muted, 0-22 = fret)
// 6 genes per chord × N chords = 6N total genes

let n_chords = 8;
let dim = 6 * n_chords; // 6 strings × 8 chords = 48 dimensions

let fitness = |genes: &Array1<f64>| -> f64 {
    let mut total_cost = 0.0;
    for i in 1..n_chords {
        // Voice leading distance: sum of fret differences between consecutive chords
        for s in 0..6 {
            let prev = genes[6 * (i - 1) + s];
            let curr = genes[6 * i + s];
            if prev >= 0.0 && curr >= 0.0 {
                total_cost += (prev - curr).abs();
            }
        }
    }
    // Bonus for non-barre chords (no full-bar fingerings)
    let barre_penalty = count_barre_chords(genes, n_chords) as f64 * 5.0;
    -(total_cost + barre_penalty) // Negative because GA minimizes
};

let result = GeneticAlgorithm::new()
    .with_population_size(100)
    .with_generations(200)
    .with_mutation_rate(0.15)
    .with_bounds(-1.0, 22.0) // -1 = mute, 0-22 = fret
    .with_seed(42)
    .minimize(&fitness, dim);

println!("Best voice leading cost: {:.2}", -result.best_fitness);
// Extract voicings from result.best_genes (reshape 48 → 8×6)
```

## Step 3: Harmonic Analysis with Wavelets

Convert a chord progression into a signal (root notes over time) and decompose with wavelets to extract harmonic features for style classification.

```rust
use ix_signal::wavelet::{haar_dwt, wavelet_denoise};

// Convert chord roots to a pitch signal (MIDI note numbers)
let progression_signal: Vec<f64> = chord_roots.iter()
    .map(|&root| root as f64)
    .collect();

// Pad to power of 2 if needed
let padded = pad_to_power_of_2(&progression_signal);

// Multi-level wavelet decomposition
let (approx, details) = haar_dwt(&padded, 3);

// Extract features for style classification
let features = WaveletFeatures {
    approx_mean: approx.iter().sum::<f64>() / approx.len() as f64,
    approx_energy: approx.iter().map(|x| x * x).sum::<f64>(),
    detail_energies: details.iter()
        .map(|d| d.iter().map(|x| x * x).sum::<f64>())
        .collect(),
};

// High detail energy = lots of harmonic movement (jazz, progressive)
// Low detail energy = smooth progressions (pop, folk)
println!("Harmonic complexity: {:.2}", features.total_detail_energy());
```

## Step 4: Optimal Fingering Path with Viterbi

Model the fretboard as an HMM: hidden states are hand positions, observations are the desired notes. Viterbi finds the path that minimizes physical cost.

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::{array, Array1, Array2};

// Simplified: 5 fret positions (states), 7 notes (observations)
// Transition probabilities: nearby positions are cheaper to reach
// Emission probabilities: each position can play certain notes well

let initial = Array1::from_vec(vec![0.4, 0.3, 0.15, 0.1, 0.05]);

// Transition: prefer staying put or moving one position
let transition = build_position_transition_matrix(5, /*decay=*/ 0.6);

// Emission: each hand position covers certain frets/notes
let emission = build_note_emission_matrix(5, 7);

let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();

// Target note sequence (mapped to observation indices)
let note_sequence = vec![0, 2, 4, 5, 4, 2, 0]; // C E G A G E C

let (fret_positions, log_prob) = hmm.viterbi(&note_sequence);
println!("Optimal hand positions: {:?}", fret_positions);
println!("Path confidence: {:.4}", log_prob);
```

> See [`examples/sequence/viterbi_hmm.rs`](../../examples/sequence/viterbi_hmm.rs) for the full HMM example.

## Step 5: Similar Progression Search with GPU

Find progressions in a database that are harmonically similar, using GPU-accelerated cosine similarity.

```rust
use ix_gpu::{context::GpuContext, similarity};

// Each progression encoded as a feature vector (from wavelet features)
let query_embedding: Vec<f32> = encode_progression(&my_progression);
let database: Vec<Vec<f32>> = load_progression_embeddings();

// GPU-accelerated top-k search
let ctx = GpuContext::new().ok();
let top_5 = similarity::batch_vector_search(
    ctx.as_ref(),
    &query_embedding,
    &database,
    5, // top-k
);

for (idx, score) in &top_5 {
    println!("Progression #{}: similarity = {:.4}", idx, score);
}
```

> See [`examples/gpu/similarity_search.rs`](../../examples/gpu/similarity_search.rs) for GPU similarity patterns.

## How Guitar Alchemist's C# Maps to ix

| Guitar Alchemist (C#/.NET) | ix (Rust) | Advantage |
|----------------------------|-------------------|-----------|
| Custom MCTS in `GuitarChordProgressionMCTS/` | `ix-search::mcts_search` | Generic trait-based, reusable |
| Embedded GA (50 pop, 100 gens) | `ix-evolution::GeneticAlgorithm` | Configurable, parallel-ready |
| Custom DWT in `WaveletTransformService` | `ix-signal::haar_dwt` | Multi-level, with denoising |
| Manual ILGPU kernels | `ix-gpu::similarity_matrix` | Cross-platform (Vulkan/DX12/Metal) |
| Qdrant vector search | `ix-gpu::batch_vector_search` | Self-contained, no external DB needed |
| Viterbi in `AdvancedTabSolver` | `ix-graph::HiddenMarkovModel::viterbi` | Full HMM with forward-backward + Baum-Welch |

## Integration Architecture

```
Guitar Alchemist (.NET)
    │
    ├─ calls via subprocess ─→ ix-skill CLI (Rust)
    │                            ├── mcts_search
    │                            ├── genetic_algorithm
    │                            ├── wavelet_analysis
    │                            ├── viterbi_decode
    │                            └── gpu_similarity
    │
    └─ or via MCP server ────→ ix MCP tools
```

TARS's `MachinBridge.fs` already supports calling `cargo run -p ix-skill` with JSON I/O, providing a ready bridge between the .NET and Rust worlds.

## Algorithms Used

| Algorithm | Doc | Role |
|-----------|-----|------|
| MCTS | [Search: MCTS](../search-and-graphs/mcts.md) | Progression generation |
| Genetic Algorithms | [Evolutionary: GA](../evolutionary/genetic-algorithms.md) | Voicing optimization |
| Wavelets (Haar DWT) | [Signal: Wavelets](../signal-processing/wavelets.md) | Harmonic feature extraction |
| HMM/Viterbi | [Sequence: Viterbi](../sequence-models/viterbi-algorithm.md) | Optimal fretboard path |
| Cosine Similarity (GPU) | [GPU: Similarity](../gpu-computing/similarity-search.md) | Progression database search |
| PSO | [Optimization: PSO](../optimization/particle-swarm.md) | Hyperparameter tuning |
| Markov Chains | [Sequence: Markov](../sequence-models/markov-chains.md) | Chord transition modeling |
