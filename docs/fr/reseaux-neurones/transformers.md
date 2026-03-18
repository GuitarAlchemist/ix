# Réseaux de neurones Transformer dans ix

## Le problème

Les réseaux de neurones traditionnels traitent les séquences élément par
élément. Les réseaux récurrents (RNN, LSTM) propagent un état caché pas à
pas, ce qui signifie que les tokens en début de séquence peuvent être
« oubliés » lorsque le réseau atteint la fin. Les modèles convolutifs ne
voient qu'une fenêtre locale de taille fixe.

De nombreuses tâches nécessitent un *contexte global*. Un mot en position 2
peut dépendre d'un mot en position 50. Une mesure de capteur à l'instant 10
n'a de sens qu'à la lumière d'un pic à l'instant 200. Il faut un mécanisme
qui permette à chaque position de prêter attention à toutes les autres en
une seule étape.

Les transformers résolvent ce problème grâce à **l'auto-attention**.

## L'intuition

Imaginez la lecture d'une phrase : « Le chat s'est assis sur le tapis parce
qu'**il** était fatigué. » Pour comprendre à quoi « il » fait référence,
vous examinez chaque mot précédent et décidez lesquels comptent le plus.
C'est exactement ce que fait l'auto-attention — chaque token produit une
*requête* (« que cherché-je ? »), une *clé* (« que contiens-je ? ») et une
*valeur* (« quelle information porté-je ? »). Les scores d'attention sont
calculés comme le produit scalaire des requêtes et des clés, mis à l'échelle
et passés par un softmax, puis utilisés pour pondérer les valeurs.

L'attention multi-têtes exécute plusieurs opérations d'attention
indépendantes en parallèle, chacune apprenant à se concentrer sur un aspect
différent de l'entrée — syntaxe, sémantique, position, etc.

## Comment ça fonctionne

Un bloc transformer effectue quatre étapes :

1. **Attention multi-têtes** — chaque tête projette Q, K, V dans un
   sous-espace, calcule l'attention par produit scalaire mis à l'échelle,
   puis concatène les résultats.
2. **Addition et LayerNorm** — une connexion résiduelle ajoute la sortie de
   l'attention à l'entrée, suivie d'une normalisation de couche.
3. **Réseau feed-forward** — deux couches linéaires avec une activation GELU
   entre les deux (`d_model -> d_ff -> d_model`).
4. **Addition et LayerNorm** — une nouvelle connexion résiduelle et
   normalisation.

Avant le premier bloc, un **encodage positionnel** injecte l'information
d'ordre séquentiel à l'aide de fonctions sinusoïdales (Vaswani et al. 2017).
Sans cela, le modèle serait invariant par permutation et incapable de
distinguer l'ordre des mots.

```
Entrée
  |
  +-- Encodage positionnel (sinusoïdal)
  |
  v
[ Attention multi-têtes ]  --+-- Addition & LayerNorm
  |                            |
  v                            |
[ Feed-Forward (GELU) ]     --+-- Addition & LayerNorm
  |
  v
Sortie (même forme que l'entrée)
```

Plusieurs blocs sont empilés pour former un **encodeur transformer**.
L'implémentation ix supporte une profondeur arbitraire via `n_layers`.

## En Rust : les briques de base

### FeedForward et TransformerBlock

L'API de plus bas niveau travaille directement avec des tenseurs
`ndarray::Array3<f64>` de forme `(batch, seq_len, d_model)`.

```rust
use ndarray::Array3;
use ix_nn::transformer::{TransformerBlock, FeedForward};

// A feed-forward network: 16-dim input, 64-dim hidden
let ff = FeedForward::new(16, 64, /*seed=*/42);

// A full transformer block: d_model=16, 2 heads, d_ff=64
let block = TransformerBlock::new(16, 2, 64, /*seed=*/42);

// Forward pass: batch=1, seq_len=4, d_model=16
let x = Array3::ones((1, 4, 16));
let out = block.forward(&x, None); // None = no dropout
assert_eq!(out.shape(), &[1, 4, 16]);
```

### Encodage positionnel

```rust
use ix_nn::positional::sinusoidal_encoding;

// Generate a (seq_len=10, d_model=16) encoding matrix
let pe = sinusoidal_encoding(10, 16);
assert_eq!(pe.shape(), &[10, 16]);
```

### Attention multi-têtes

```rust
use ndarray::Array3;
use ix_nn::attention::{scaled_dot_product_attention, multi_head_attention};

let q = Array3::ones((1, 4, 8));
let k = Array3::ones((1, 4, 8));
let v = Array3::ones((1, 4, 8));

let (output, weights) = scaled_dot_product_attention(&q, &k, &v, None);
assert_eq!(output.shape(), &[1, 4, 8]);
assert_eq!(weights.shape(), &[1, 4, 4]); // seq_len x seq_len
```

## Entraîner un classifieur Transformer

`TransformerConfig` encapsule tous les hyperparamètres dans une seule
structure. `TransformerClassifier` implémente le trait `Classifier` de
`ix-supervised`, ce qui permet d'appeler `fit` et `predict` comme pour
n'importe quel autre modèle.

```rust
use ix_nn::classifier::{TransformerConfig, TransformerClassifier, LrSchedule};
use ix_supervised::traits::Classifier;
use ndarray::Array2;

// Configure the model
let config = TransformerConfig {
    d_model: 32,
    n_heads: 4,
    n_layers: 2,
    d_ff: 128,
    seq_len: None,        // inferred: n_features / d_model
    epochs: 100,
    learning_rate: 0.001,
    seed: 42,
    dropout: 0.1,         // 10% dropout after attention and FFN
    batch_size: Some(16), // mini-batch gradient descent
    lr_schedule: LrSchedule::WarmupCosine {
        warmup_steps: 10, // linear warmup for 10 steps
        min_lr: 1e-5,     // cosine decay floor
    },
    use_gpu: false,
};

let mut model = TransformerClassifier::new(config);

// Training data: 100 samples, 64 features each
let x_train = Array2::ones((100, 64));
let y_train = vec![0usize; 50].into_iter()
    .chain(vec![1usize; 50])
    .collect::<Vec<_>>();

model.fit(&x_train, &y_train);

// Predict on new data
let x_test = Array2::ones((10, 64));
let predictions = model.predict(&x_test);
assert_eq!(predictions.len(), 10);
```

### Comment `seq_len` est inféré

Quand `seq_len` vaut `None`, le modèle le calcule comme
`n_features / d_model`. Dans l'exemple ci-dessus : 64 caractéristiques /
32 d_model = 2 positions de séquence. Chaque ligne de `x_train` est
remodelée en une matrice `(2, 32)`, puis l'encodage positionnel est ajouté.

### Planification du taux d'apprentissage

La planification `WarmupCosine` évite deux problèmes courants :

- **Démarrage à froid** — de forts gradients en début d'entraînement
  déstabilisent les poids d'attention. Le réchauffement linéaire augmente
  progressivement le taux d'apprentissage de 0 à `learning_rate` sur
  `warmup_steps` étapes.
- **Sur-entraînement** — la décroissance cosinus réduit progressivement le
  taux d'apprentissage vers `min_lr`, empêchant le modèle de dépasser le
  minimum une fois qu'il en est proche.

## Quand utiliser les Transformers

**Bons candidats :**

- Données séquentielles avec des dépendances à longue portée (texte, séries
  temporelles, génomique)
- Données tabulaires avec de nombreuses caractéristiques interagissant de
  manière non évidente
- Problèmes où les poids d'attention interprétables sont utiles

**Envisagez d'abord des modèles plus simples quand :**

- Vous disposez de moins de quelques centaines d'échantillons d'entraînement
- Les caractéristiques sont indépendantes ou seulement localement corrélées
- Le temps d'entraînement et la mémoire sont limités (l'attention est en
  O(n^2) par rapport à seq_len)

Une forêt aléatoire ou une régression linéaire surpassera souvent un
transformer sur de petits jeux de données tabulaires. Commencez simplement,
puis montez en puissance si les données le justifient.

## Référence des paramètres clés

| Paramètre       | Type              | Défaut    | Description                                             |
|------------------|-------------------|-----------|---------------------------------------------------------|
| `d_model`        | `usize`           | 32        | Dimension d'embedding. Doit être divisible par `n_heads` |
| `n_heads`        | `usize`           | 4         | Nombre de têtes d'attention parallèles                   |
| `n_layers`       | `usize`           | 2         | Nombre de blocs transformer empilés                      |
| `d_ff`           | `usize`           | 128       | Dimension cachée du réseau feed-forward                  |
| `seq_len`        | `Option<usize>`   | `None`    | Longueur de séquence (inférée si `None`)                 |
| `epochs`         | `usize`           | 50        | Nombre de passes complètes sur les données               |
| `learning_rate`  | `f64`             | 0.001     | Taux d'apprentissage de base                             |
| `seed`           | `u64`             | 42        | Graine aléatoire pour l'initialisation des poids         |
| `dropout`        | `f64`             | 0.0       | Probabilité de dropout (0.0 à 1.0)                      |
| `batch_size`     | `Option<usize>`   | `None`    | Taille de mini-lot (`None` = lot complet)               |
| `lr_schedule`    | `LrSchedule`      | `Constant`| Planification du taux d'apprentissage                    |
| `use_gpu`        | `bool`            | `false`   | Utiliser l'attention accélérée par WGPU si disponible    |

### Règles pratiques

- Fixer `d_ff` à 4 * `d_model` comme point de départ.
- Utiliser un `dropout` entre 0.1 et 0.3 pour la régularisation.
- Un `warmup_steps` de 5 à 10 % du nombre total d'étapes fonctionne bien
  en pratique.
- Un `batch_size` de 16 à 64 équilibre bruit de gradient et calcul.
- Activer `use_gpu` pour les grands modèles ou les longues séquences ; pour
  les petits problèmes, le chemin CPU est souvent plus rapide en raison du
  coût de lancement des noyaux GPU.

## Pour aller plus loin

- Vaswani et al., « Attention Is All You Need » (2017) — l'article original.
- Le code source ix : `crates/ix-nn/src/attention.rs`, `transformer.rs`,
  `classifier.rs`, `positional.rs`.
- Le crate `ix-demo` inclut un onglet de démonstration interactive des
  transformers.
