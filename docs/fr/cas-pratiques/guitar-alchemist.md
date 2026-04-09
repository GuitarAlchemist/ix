# Cas pratique : Guitar Alchemist --- Théorie musicale assistée par IA

> Combiner MCTS, algorithmes génétiques, ondelettes, HMM/Viterbi et recherche vectorielle pour générer des progressions d'accords, optimiser les doigtés et analyser la structure harmonique.

## Le scénario

Vous construisez [Guitar Alchemist](../../) --- une plateforme d'apprentissage musical assistée par IA. Étant donné une mélodie (notes MIDI), vous voulez :

1. **Générer une progression d'accords** qui harmonise la mélodie (MCTS)
2. **Optimiser les voicings de guitare** pour que les doigts bougent le moins possible entre les accords (algorithmes génétiques)
3. **Classifier le style harmonique** d'une progression (ondelettes + analyse de signal)
4. **Trouver le chemin de doigté optimal** sur le manche (Viterbi/HMM)
5. **Rechercher des progressions similaires** dans une base de données (similarité cosinus + GPU)

C'est un cas d'usage transversal qui démontre comment les algorithmes d'ix se combinent pour une véritable application d'IA créative.

## Le pipeline

```
Mélodie (notes MIDI)
    |
    v
MCTS ---------------------- Générer des progressions candidates
    |
    v
Algorithme génétique ------- Optimiser les voicings (minimiser le mouvement des doigts)
    |
    v
Analyse par ondelettes ----- Extraire les caractéristiques harmoniques pour la classification de style
    |
    v
HMM/Viterbi --------------- Trouver le chemin optimal sur le manche
    |
    v
Similarité GPU ------------- Trouver des progressions similaires en base de données
```

## Étape 1 : Génération de progressions d'accords avec MCTS

Étant donné une mélodie, explorer l'espace des progressions d'accords possibles avec Monte Carlo Tree Search. Chaque noeud représente une progression partielle ; les enfants sont les accords suivants possibles.

```rust
use ix_search::mcts::{MctsState, mcts_search};

#[derive(Clone)]
struct ProgressionState {
    chords: Vec<usize>,       // Indices des accords choisis jusqu'ici
    melody: Vec<u8>,          // Mélodie cible (notes MIDI)
    key: usize,               // Tonalité musicale (0=Do, 1=Do#, ...)
    chord_tones: Vec<Vec<u8>>, // Quelles notes MIDI chaque accord contient
}

impl MctsState for ProgressionState {
    type Action = usize; // Indice de l'accord suivant à ajouter

    fn legal_actions(&self) -> Vec<usize> {
        let pos = self.chords.len();
        if pos >= self.melody.len() { return vec![]; }

        let melody_note = self.melody[pos] % 12;
        // Filtrer les accords dont les notes contiennent la note mélodique courante
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
        // Score : variété + tension/résolution + fluidité de la conduite des voix
        let variety = self.chords.iter().collect::<std::collections::HashSet<_>>().len() as f64
            / self.chords.len() as f64;
        let resolution_bonus = self.count_resolutions() as f64 * 0.1;
        (variety + resolution_bonus).min(1.0)
    }
}

// Générer une progression pour une mélodie en Do majeur
let melody = vec![60, 62, 64, 67, 65, 64, 62, 60]; // Do Ré Mi Sol Fa Mi Ré Do
let initial = ProgressionState::new(melody, key_of_c_major());

let best_action = mcts_search(&initial, 1000, 1.41, 42);
// Renvoie le meilleur premier accord ; répéter pour la progression complète
```

> Voir [`examples/search/astar_qstar.rs`](../../examples/search/astar_qstar.rs) pour les schémas d'algorithmes de recherche.

## Étape 2 : Optimisation des voicings avec algorithmes génétiques

Chaque accord peut être joué à de nombreuses positions sur une guitare. Un algorithme génétique trouve les voicings qui minimisent le mouvement total des doigts à travers la progression.

```rust
use ix_evolution::{GeneticAlgorithm, EvolutionResult};
use ndarray::Array1;

// Chaque gène = position de frette pour chaque corde (-1 = étouffée, 0-22 = frette)
// 6 gènes par accord × N accords = 6N gènes au total

let n_chords = 8;
let dim = 6 * n_chords; // 6 cordes × 8 accords = 48 dimensions

let fitness = |genes: &Array1<f64>| -> f64 {
    let mut total_cost = 0.0;
    for i in 1..n_chords {
        // Distance de conduite des voix : somme des différences de frettes entre accords consécutifs
        for s in 0..6 {
            let prev = genes[6 * (i - 1) + s];
            let curr = genes[6 * i + s];
            if prev >= 0.0 && curr >= 0.0 {
                total_cost += (prev - curr).abs();
            }
        }
    }
    // Bonus pour les accords non-barré (pas de doigtés en barré complet)
    let barre_penalty = count_barre_chords(genes, n_chords) as f64 * 5.0;
    -(total_cost + barre_penalty) // Négatif car le GA minimise
};

let result = GeneticAlgorithm::new()
    .with_population_size(100)
    .with_generations(200)
    .with_mutation_rate(0.15)
    .with_bounds(-1.0, 22.0) // -1 = étouffée, 0-22 = frette
    .with_seed(42)
    .minimize(&fitness, dim);

println!("Meilleur coût de conduite des voix : {:.2}", -result.best_fitness);
// Extraire les voicings de result.best_genes (reshape 48 -> 8×6)
```

## Étape 3 : Analyse harmonique avec ondelettes

Convertir une progression d'accords en un signal (notes fondamentales au fil du temps) et décomposer avec des ondelettes pour extraire des caractéristiques harmoniques pour la classification de style.

```rust
use ix_signal::wavelet::{haar_dwt, wavelet_denoise};

// Convertir les fondamentales des accords en signal de hauteur (numéros de notes MIDI)
let progression_signal: Vec<f64> = chord_roots.iter()
    .map(|&root| root as f64)
    .collect();

// Compléter à une puissance de 2 si nécessaire
let padded = pad_to_power_of_2(&progression_signal);

// Décomposition en ondelettes multi-niveaux
let (approx, details) = haar_dwt(&padded, 3);

// Extraire les caractéristiques pour la classification de style
let features = WaveletFeatures {
    approx_mean: approx.iter().sum::<f64>() / approx.len() as f64,
    approx_energy: approx.iter().map(|x| x * x).sum::<f64>(),
    detail_energies: details.iter()
        .map(|d| d.iter().map(|x| x * x).sum::<f64>())
        .collect(),
};

// Haute énergie de détail = beaucoup de mouvement harmonique (jazz, progressif)
// Basse énergie de détail = progressions douces (pop, folk)
println!("Complexité harmonique : {:.2}", features.total_detail_energy());
```

## Étape 4 : Chemin de doigté optimal avec Viterbi

Modéliser le manche comme un HMM : les états cachés sont les positions de la main, les observations sont les notes souhaitées. Viterbi trouve le chemin qui minimise le coût physique.

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::{array, Array1, Array2};

// Simplifié : 5 positions de frettes (états), 7 notes (observations)
// Probabilités de transition : les positions proches sont moins coûteuses à atteindre
// Probabilités d'émission : chaque position peut bien jouer certaines notes

let initial = Array1::from_vec(vec![0.4, 0.3, 0.15, 0.1, 0.05]);

// Transition : préférer rester en place ou bouger d'une position
let transition = build_position_transition_matrix(5, /*decay=*/ 0.6);

// Émission : chaque position de main couvre certaines frettes/notes
let emission = build_note_emission_matrix(5, 7);

let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();

// Séquence de notes cible (mappée en indices d'observations)
let note_sequence = vec![0, 2, 4, 5, 4, 2, 0]; // Do Mi Sol La Sol Mi Do

let (fret_positions, log_prob) = hmm.viterbi(&note_sequence);
println!("Positions de main optimales : {:?}", fret_positions);
println!("Confiance du chemin : {:.4}", log_prob);
```

> Voir [`examples/sequence/viterbi_hmm.rs`](../../examples/sequence/viterbi_hmm.rs) pour l'exemple HMM complet.

## Étape 5 : Recherche de progressions similaires avec GPU

Trouver dans une base de données les progressions harmoniquement similaires, en utilisant la similarité cosinus accélérée GPU.

```rust
use ix_gpu::{GpuContext, similarity};

// Chaque progression encodée comme vecteur de caractéristiques (des caractéristiques d'ondelettes)
let query_embedding: Vec<f32> = encode_progression(&my_progression);
let database: Vec<Vec<f32>> = load_progression_embeddings();

// Recherche top-k accélérée GPU
let ctx = GpuContext::new().ok();
let top_5 = similarity::batch_vector_search(
    ctx.as_ref(),
    &query_embedding,
    &database,
    5, // top-k
);

for (idx, score) in &top_5 {
    println!("Progression #{} : similarité = {:.4}", idx, score);
}
```

> Voir [`examples/gpu/similarity_search.rs`](../../examples/gpu/similarity_search.rs) pour les schémas de similarité GPU.

## Comment le C# de Guitar Alchemist correspond à ix

| Guitar Alchemist (C#/.NET) | ix (Rust) | Avantage |
|----------------------------|-------------------|-----------|
| MCTS personnalisé dans `GuitarChordProgressionMCTS/` | `ix-search::mcts_search` | Basé sur des traits génériques, réutilisable |
| GA intégré (50 pop, 100 gen) | `ix-evolution::GeneticAlgorithm` | Configurable, prêt pour le parallèle |
| DWT personnalisé dans `WaveletTransformService` | `ix-signal::haar_dwt` | Multi-niveaux, avec débruitage |
| Kernels ILGPU manuels | `ix-gpu::similarity_matrix` | Multiplateforme (Vulkan/DX12/Metal) |
| Recherche vectorielle Qdrant | `ix-gpu::batch_vector_search` | Autonome, pas de base externe nécessaire |
| Viterbi dans `AdvancedTabSolver` | `ix-graph::HiddenMarkovModel::viterbi` | HMM complet avec forward-backward + Baum-Welch |

## Architecture d'intégration

```
Guitar Alchemist (.NET)
    |
    +-- appel via sous-processus --> ix-skill CLI (Rust)
    |                                 +-- mcts_search
    |                                 +-- genetic_algorithm
    |                                 +-- wavelet_analysis
    |                                 +-- viterbi_decode
    |                                 +-- gpu_similarity
    |
    +-- ou via serveur MCP ---------> outils MCP ix
```

Le `MachinBridge.fs` de TARS supporte déjà l'appel de `cargo run -p ix-skill` avec des E/S JSON, fournissant un pont prêt à l'emploi entre les mondes .NET et Rust.

## Algorithmes utilisés

| Algorithme | Doc | Rôle |
|-----------|-----|------|
| MCTS | [Recherche : MCTS](../search-and-graphs/mcts.md) | Génération de progressions |
| Algorithmes génétiques | [Évolution : AG](../evolutionary/genetic-algorithms.md) | Optimisation des voicings |
| Ondelettes (Haar DWT) | [Signal : Ondelettes](../signal-processing/wavelets.md) | Extraction de caractéristiques harmoniques |
| HMM/Viterbi | [Séquences : Viterbi](../modeles-sequentiels/algorithme-de-viterbi.md) | Chemin optimal sur le manche |
| Similarité cosinus (GPU) | [GPU : Similarité](../calcul-gpu/recherche-de-similarite.md) | Recherche en base de progressions |
| PSO | [Optimisation : PSO](../optimization/particle-swarm.md) | Réglage d'hyperparamètres |
| Chaînes de Markov | [Séquences : Markov](../modeles-sequentiels/chaines-de-markov.md) | Modélisation des transitions d'accords |
