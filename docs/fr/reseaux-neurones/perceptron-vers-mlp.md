# Du perceptron au MLP

> Un seul neurone ne peut tracer qu'une ligne droite. Empilez-les en couches et ils peuvent tout apprendre.

## Le problème

Vous construisez un système de reconnaissance de chiffres manuscrits. Chaque image fait 28x28 pixels = 784 nombres. À partir de ces 784 intensités de pixels, il faut classer l'image comme un chiffre de 0 à 9.

Les modèles linéaires ne peuvent pas résoudre ce problème — la relation entre les pixels bruts et les chiffres est profondément non linéaire. Un « 7 » et un « 1 » peuvent partager de nombreuses positions de pixels, mais diffèrent par des motifs subtils qu'aucune ligne droite ne peut séparer. Vous avez besoin d'un modèle qui construit des motifs complexes à partir de motifs simples.

## L'intuition

### Le perceptron : un seul neurone

Un perceptron prend des entrées, multiplie chacune par un poids, les additionne, et fait passer le résultat à travers une fonction d'activation.

```
entrées : [x1, x2, x3]
poids :   [w1, w2, w3]
biais :   b

sortie = activation(w1*x1 + w2*x2 + w3*x3 + b)
```

Voyez cela comme un vote. Chaque entrée est un votant, chaque poids représente l'importance de son opinion, et la fonction d'activation prend la décision finale oui/non.

Un seul perceptron ne peut apprendre que des frontières **linéaires** — il trace une ligne droite (ou un hyperplan) pour séparer les classes. Il gère le ET et le OU, mais échoue notoirement sur le XOR.

### Perceptron multicouche (MLP) : empiler des neurones

Empilez des perceptrons en couches et quelque chose de remarquable se produit — le réseau peut apprendre n'importe quelle fonction continue (avec suffisamment de neurones).

```
Couche d'entrée   Couche cachée      Couche de sortie
[784 entrées]  → [128 neurones]  → [10 sorties]
   (pixels)    (caractéristiques    (chiffres 0-9)
                   apprises)
```

**Les couches cachées apprennent des caractéristiques automatiquement.** La première couche cachée pourrait apprendre à détecter des bords. La deuxième pourrait combiner les bords en courbes et en angles. La couche de sortie combine le tout en reconnaissance de chiffres.

Chaque connexion a un poids. L'entraînement ajuste ces poids pour que les prédictions du réseau correspondent aux bonnes réponses.

### Fonctions d'activation : ajouter de la non-linéarité

Sans fonctions d'activation, empiler des couches ne donnerait qu'une grande fonction linéaire (inutile — équivalent à une seule couche). Les activations ajoutent de la non-linéarité, ce qui permet aux réseaux profonds de modéliser des motifs complexes.

- **ReLU** : `max(0, x)` — la plus courante. Simple, rapide, fonctionne bien en pratique. Renvoie 0 pour les entrées négatives, laisse passer les entrées positives sans changement.
- **Sigmoïde** : `1 / (1 + e^(-x))` — comprime la sortie dans l'intervalle (0, 1). Utilisée pour les sorties en probabilité. Peut provoquer des gradients évanescents dans les réseaux profonds.
- **Tanh** : `(e^x - e^(-x)) / (e^x + e^(-x))` — semblable à la sigmoïde mais avec des sorties dans (-1, 1). Centrée sur zéro, ce qui facilite l'entraînement.
- **Softmax** : Convertit un vecteur de scores en probabilités dont la somme vaut 1. Utilisée dans la couche de sortie pour la classification multi-classes.

## Comment ça fonctionne

### Passe avant

Les données circulent de l'entrée vers la sortie, couche par couche :

1. **Entrée** : Caractéristiques brutes (par ex., 784 valeurs de pixels)
2. **Calcul de la couche cachée** : `h = activation(W₁ × x + b₁)`
   - W₁ est une matrice de poids (128×784 pour 784→128)
   - b₁ est un vecteur de biais (128 éléments)
   - L'activation est appliquée élément par élément
3. **Couche de sortie** : `ŷ = softmax(W₂ × h + b₂)`
   - W₂ est de taille 10×128, b₂ a 10 éléments
   - Softmax transforme les scores bruts en probabilités

En clair : multiplier les entrées par les poids, ajouter le biais, appliquer la non-linéarité. Répéter pour chaque couche. La sortie finale est la prédiction.

### Entraînement : ajuster les poids

1. **Passe avant** : Calculer la prédiction
2. **Perte** : Mesurer à quel point la prédiction est fausse (par ex., entropie croisée)
3. **Passe arrière** : Calculer les gradients de la perte par rapport à chaque poids (rétropropagation)
4. **Mise à jour** : Ajuster chaque poids dans la direction qui réduit la perte

C'est la descente de gradient appliquée aux réseaux de neurones. Voir [Rétropropagation](retropropagation.md) pour les détails.

### Initialisation des poids

La manière dont vous définissez les poids initiaux est cruciale :

- **Initialisation de Xavier** : std = √(2 / (fan_in + fan_out)). Adaptée aux réseaux avec sigmoïde/tanh.
- **Initialisation de He** : std = √(2 / fan_in). Conçue pour les réseaux avec ReLU.
- **Zéros** : Ne faites jamais cela — tous les neurones apprennent la même chose (problème de symétrie).

## En Rust

ix fournit les briques de base dans `ix-nn` :

```rust
use ndarray::array;
use ix_nn::layer::{Dense, Layer};
use ix_nn::loss::{mse_loss, mse_gradient};
use ix_math::activation::{relu_array, sigmoid_array, softmax};

// Construire un réseau simple à 2 couches : 3 entrées → 4 cachées → 2 sorties
let mut hidden = Dense::new(3, 4);   // Initialisé avec Xavier
let mut output = Dense::new(4, 2);

// Passe avant
let input = array![[1.0, 0.5, -0.3]];  // batch de 1 échantillon, 3 caractéristiques
let h = hidden.forward(&input);         // forme : (1, 4)
let h_activated = h.mapv(|x| if x > 0.0 { x } else { 0.0 }); // ReLU
let prediction = output.forward(&h_activated);  // forme : (1, 2)

println!("Prédiction : {:?}", prediction);

// Calculer la perte
let target = array![[1.0, 0.0]];  // sortie cible
let loss = mse_loss(&prediction, &target);
println!("Perte : {}", loss);

// Passe arrière (taux d'apprentissage = 0.01)
let grad = mse_gradient(&prediction, &target);
let grad_hidden = output.backward(&grad, 0.01);
hidden.backward(&grad_hidden, 0.01);
```

### Options d'initialisation des poids

```rust
use ix_nn::initializers;

let xavier_weights = initializers::xavier(784, 128);  // Pour sigmoïde/tanh
let he_weights = initializers::he(784, 128);           // Pour ReLU
let zero_weights = initializers::zeros(784, 128);      // Ne pas utiliser
```

### Fonctions d'activation

```rust
use ndarray::array;
use ix_math::activation;

let x = array![1.0, -0.5, 0.0, 2.0];

let relu = activation::relu_array(&x);       // [1.0, 0.0, 0.0, 2.0]
let sigmoid = activation::sigmoid_array(&x);  // [0.73, 0.38, 0.50, 0.88]
let tanh = activation::tanh_array(&x);        // [0.76, -0.46, 0.0, 0.96]

let scores = array![2.0, 1.0, 0.1];
let probs = activation::softmax(&scores);     // [0.66, 0.24, 0.10] — somme = 1.0
```

## Quand l'utiliser

| Modèle | Idéal pour | Limites |
|-------|----------|-------------|
| **Perceptron simple** | Problèmes linéairement séparables | Ne gère pas les motifs non linéaires |
| **MLP (1-2 couches cachées)** | Données tabulaires, complexité modérée | Nécessite plus de données que les modèles linéaires |
| **MLP profond (3+ couches)** | Motifs complexes, grands jeux de données | Lent à entraîner, nécessite un réglage minutieux |
| **Régression linéaire/logistique** | Relations simples, petits jeux de données | Ne capture pas la non-linéarité |

Les MLP sont un bon compromis — plus puissants que les modèles linéaires, plus simples que les architectures de deep learning (CNN, Transformers).

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Conseils |
|-----------|-----------------|----------|
| Taille de la couche cachée | Capacité du modèle | Commencez avec 64-256. Plus de neurones = plus de capacité mais plus lent et plus sujet au surapprentissage |
| Nombre de couches cachées | Profondeur de la hiérarchie de caractéristiques | 1-2 couches pour la plupart des données tabulaires. Rarement besoin de plus de 3. |
| Fonction d'activation | Type de non-linéarité | ReLU pour les couches cachées (par défaut). Sigmoïde pour la sortie binaire. Softmax pour la sortie multi-classes. |
| Taux d'apprentissage | Taille du pas pendant l'entraînement | Commencez à 0.01. Si la perte oscille, diminuez. Si elle stagne, augmentez. |
| Initialisation | Poids de départ | He pour ReLU, Xavier pour sigmoïde/tanh |

## Pièges

- **N'utilisez pas l'initialisation à zéro.** Tous les neurones démarrent identiques et apprennent la même chose indéfiniment (problème de brisure de symétrie). Utilisez toujours Xavier ou He.
- **Les neurones ReLU peuvent « mourir ».** Si la sortie d'un neurone est toujours négative, ReLU le verrouille à 0 et il ne récupère jamais. Utilisez Leaky ReLU (`leaky_relu(x, 0.01)`) si cela se produit.
- **Plus de couches ne veut pas dire mieux.** Pour les données tabulaires, 1-2 couches cachées suffisent généralement. Les réseaux plus profonds nécessitent des techniques comme la normalisation par batch et les connexions résiduelles.
- **Normalisez les entrées.** Les réseaux de neurones s'entraînent beaucoup plus vite quand les entrées sont centrées autour de 0 avec des échelles similaires.
- **Surveillez la courbe de perte.** Si la perte d'entraînement diminue mais que la perte de validation augmente, vous êtes en surapprentissage. Réduisez la taille du modèle ou ajoutez de la régularisation.

## Pour aller plus loin

- **Suite** : [Rétropropagation](retropropagation.md) — comment le réseau apprend en calculant les gradients
- **Suite** : [Fonctions de perte](fonctions-de-perte.md) — choisir le bon objectif
- **Fondement** : [Intuition du calcul](../fondements/intuition-du-calcul.md) — les dérivées alimentent la rétropropagation
- **Fondement** : [Vecteurs et matrices](../fondements/vecteurs-et-matrices.md) — la multiplication matricielle est l'opération centrale
