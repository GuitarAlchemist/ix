# Régression logistique

## Le problème

Votre fournisseur de messagerie traite des millions de messages par heure. Chaque message possède des caractéristiques mesurables : le nombre d'occurrences du mot « gratuit », la présence de l'expéditeur dans la liste de contacts, le ratio liens/texte, la présence de certains motifs d'en-tête. Le système doit décider, pour chaque courriel, s'il s'agit de spam (classe 1) ou non (classe 0).

Il ne s'agit pas d'une prédiction continue — il faut une réponse par oui ou par non. Mais on souhaite aussi savoir *à quel point* le modèle est confiant. Un courriel ayant 51 % de chances d'être du spam devrait peut-être atterrir dans un dossier « suspects », tandis qu'un courriel à 99 % irait directement à la corbeille. Il faut un modèle qui produit une probabilité entre 0 et 1.

La régression logistique fait exactement cela. Malgré son nom, c'est un algorithme de classification. Elle prend la combinaison linéaire des caractéristiques (comme la régression linéaire), puis compresse le résultat à travers une fonction sigmoïde pour produire une probabilité. L'entraînement ajuste les poids afin de maximiser la vraisemblance que les probabilités du modèle correspondent aux étiquettes réelles.

## L'intuition

Imaginez la régression logistique comme une régression linéaire avec un traducteur boulonné à la sortie. La régression linéaire peut produire n'importe quel nombre, de moins l'infini à plus l'infini. La fonction sigmoïde agit comme un traducteur qui convertit ce nombre brut en une valeur entre 0 et 1 — une probabilité.

Imaginez un thermomètre dont le mercure peut monter ou descendre sans limite. Placez maintenant ce thermomètre derrière une lentille spéciale qui compresse la lecture : les nombres très négatifs sont poussés vers 0, les nombres très positifs vers 1, et la plage intermédiaire s'étale harmonieusement entre les deux. Cette compression en forme de S est la sigmoïde.

Pendant l'entraînement, le modèle ajuste ses poids de sorte que les courriels de spam obtiennent un score brut élevé (que la sigmoïde projette près de 1,0) et que les courriels légitimes obtiennent un score brut faible (projeté près de 0,0). Le processus d'apprentissage pousse les poids petit à petit dans la direction qui rend les probabilités plus exactes — c'est la descente de gradient.

## Comment ça fonctionne

### Étape 1 : Calculer le score brut

$$
z = \mathbf{X} \mathbf{w} + b
$$

En termes simples, cela signifie : pour chaque courriel, multiplier ses caractéristiques par les poids appris, les additionner, puis ajouter un terme de biais. On obtient un score brut qui peut être n'importe quel nombre réel.

### Étape 2 : Appliquer la fonction sigmoïde

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

En termes simples, cela signifie : compresser le score brut dans l'intervalle (0, 1). Les scores positifs élevés deviennent des probabilités proches de 1 (probablement du spam). Les scores négatifs élevés deviennent des probabilités proches de 0 (probablement légitime). Un score de exactement 0 correspond à 0,5 — l'incertitude totale.

### Étape 3 : Calculer la perte logarithmique (entropie croisée binaire)

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(a_i) + (1 - y_i) \log(1 - a_i) \right]
$$

En termes simples, cela signifie : mesurer à quel point le modèle se trompe. Si l'étiquette réelle est 1 (spam) et que le modèle indique une probabilité de 0,01, la perte est énorme. S'il indique 0,99, la perte est infime. Cette fonction pénalise sévèrement les erreurs faites avec confiance.

### Étape 4 : Calculer les gradients et mettre à jour les poids

$$
\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} \mathbf{X}^T (\mathbf{a} - \mathbf{y})
$$

$$
\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial L}{\partial \mathbf{w}}
$$

En termes simples, cela signifie : déterminer dans quelle direction ajuster chaque poids pour réduire la perte, puis faire un petit pas dans cette direction. La taille du pas est contrôlée par le taux d'apprentissage alpha. On répète cette opération de nombreuses fois jusqu'à la convergence des poids.

### Étape 5 : Classer

Appliquer un seuil (par défaut 0,5) à la probabilité :

$$
\hat{y} = \begin{cases} 1 & \text{si } \sigma(z) \geq 0.5 \\ 0 & \text{sinon} \end{cases}
$$

En termes simples, cela signifie : si le modèle estime que le courriel est plus probablement du spam que non, l'étiqueter comme spam.

## En Rust

```rust
use ndarray::array;
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Features: [word_free_count, has_known_sender (0/1), link_ratio, suspicious_headers]
    let x = array![
        [0.0, 1.0, 0.05, 0.0],  // legitimate
        [0.0, 1.0, 0.10, 0.0],  // legitimate
        [1.0, 0.0, 0.02, 0.0],  // legitimate (one "free" but known pattern)
        [5.0, 0.0, 0.80, 1.0],  // spam
        [3.0, 0.0, 0.60, 1.0],  // spam
        [7.0, 0.0, 0.90, 1.0],  // spam
    ];
    let y = array![0, 0, 0, 1, 1, 1]; // 0 = not spam, 1 = spam

    // Build and train the model
    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iterations(1000);
    model.fit(&x, &y);

    // Predict hard labels
    let predictions = model.predict(&x);
    println!("Predictions: {}", predictions);

    // Get probability estimates (columns: [P(class=0), P(class=1)])
    let probabilities = model.predict_proba(&x);
    println!("Spam probabilities: {}", probabilities.column(1));

    // Evaluate
    println!("Accuracy:  {:.4}", accuracy(&y, &predictions));
    println!("Precision: {:.4}", precision(&y, &predictions, 1));
    println!("Recall:    {:.4}", recall(&y, &predictions, 1));
    println!("F1:        {:.4}", f1_score(&y, &predictions, 1));
}
```

## Quand l'utiliser

| Situation | Régression logistique | Alternative | Pourquoi |
|---|---|---|---|
| Classification binaire, frontière de décision linéaire | Oui | — | Rapide, interprétable, probabiliste |
| Besoin de probabilités en sortie, pas seulement d'étiquettes | Oui | — | La sigmoïde produit naturellement des probabilités calibrées |
| Frontières de décision non linéaires | Non | Arbre de décision, SVM à noyau, réseau de neurones | La régression logistique ne peut tracer que des droites |
| Multi-classe (3+ classes) | Limité | Bayes naïf, KNN, arbre de décision | L'implémentation d'ix est uniquement binaire |
| Données creuses de très haute dimension (TAL) | Oui | Bayes naïf | La régression logistique gère bien les caractéristiques creuses |
| Besoin de comprendre l'importance des caractéristiques | Oui | — | L'amplitude et le signe des poids sont directement interprétables |

## Paramètres clés

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `learning_rate` | `0.01` | Taux d'apprentissage pour la descente de gradient. Trop élevé : oscillations. Trop faible : convergence lente. |
| `max_iterations` | `1000` | Nombre de passes de descente de gradient sur l'ensemble des données. |

Configuration via le patron de construction :
```rust
LogisticRegression::new()
    .with_learning_rate(0.05)
    .with_max_iterations(2000)
```

## Pièges courants

**Taux d'apprentissage trop élevé.** La perte oscillera ou divergera au lieu de diminuer. Si les prédictions sont aléatoires après l'entraînement, essayez de réduire le taux d'apprentissage d'un facteur 10.

**Caractéristiques non normalisées.** La régression logistique utilise la descente de gradient, qui est sensible à l'échelle des caractéristiques. Si une caractéristique varie de 0 à 1 et une autre de 0 à 10 000, les gradients seront dominés par la caractéristique à grande échelle. Normalisez ou standardisez les caractéristiques avant l'entraînement.

**Données linéairement inséparables.** Si les courriels de spam et les courriels légitimes se chevauchent dans l'espace des caractéristiques sans qu'une droite puisse les séparer, la régression logistique fera de son mieux, mais le plafond de précision sera bas. Envisagez d'ajouter des interactions polynomiales entre les caractéristiques ou de passer à un modèle non linéaire.

**Déséquilibre des classes.** Si 99 % des courriels sont légitimes et 1 % sont du spam, le modèle risque d'apprendre à toujours prédire « non spam » et d'atteindre tout de même 99 % de précision. Utilisez la précision, le rappel et le score F1 (voir [metriques-evaluation.md](./metriques-evaluation.md)) plutôt que l'exactitude, et envisagez d'ajuster le seuil de décision au-delà de 0,5.

**Binaire uniquement.** L'implémentation d'ix ne prend en charge que deux classes (0 et 1). Pour les problèmes multi-classes, utilisez la stratégie « un contre tous » avec plusieurs modèles de régression logistique, ou choisissez un autre classifieur.

## Pour aller plus loin

- **Ingénierie des caractéristiques :** Ajoutez des termes d'interaction (par ex. `word_free_count * link_ratio`) pour capturer des motifs non linéaires dans un cadre linéaire.
- **Régularisation :** Combinez avec `ix_optimize` pour ajouter des termes de pénalité L2 (Ridge) ou L1 (Lasso) afin de prévenir le surapprentissage.
- **Ajustement du seuil :** Au lieu du seuil par défaut de 0,5, choisissez un seuil qui optimise le score F1 ou qui minimise les faux négatifs, selon votre application.
- **Approfondissement de l'évaluation :** Voir [metriques-evaluation.md](./metriques-evaluation.md) pour savoir quand privilégier la précision au rappel.
