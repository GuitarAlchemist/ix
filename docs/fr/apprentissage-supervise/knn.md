# k plus proches voisins (KNN)

## Le problème

Vous gérez un service de streaming musical et souhaitez recommander de nouvelles chansons aux utilisateurs. Chaque chanson peut être décrite par des caractéristiques mesurables : le tempo (BPM), le niveau d'énergie, la dansabilité, l'acousticité et la valence (positivité). Lorsqu'un utilisateur termine une chanson qu'il a aimée, vous voulez trouver les chansons les plus similaires dans votre catalogue et les lui recommander.

L'intuition est très simple : les éléments similaires sont proches les uns des autres. Si un utilisateur adore une chanson pop entraînante à 120 BPM avec une forte dansabilité, il appréciera probablement d'autres chansons entraînantes et dansantes à des tempos similaires. Nul besoin d'apprendre un modèle complexe — il suffit de trouver les voisins les plus proches dans l'espace des caractéristiques.

KNN formalise cette intuition. Il stocke l'intégralité du jeu de données d'entraînement, et lorsqu'on lui demande de classer un nouveau point, il trouve les k points d'entraînement les plus proches, examine leurs étiquettes et choisit par vote majoritaire. Pas de phase d'entraînement, pas de paramètres appris, pas d'hypothèses sur la forme des données. Simplement « dis-moi ce que disent les voisins ».

## L'intuition

Imaginez que vous emménagiez dans un nouveau quartier et que vous vous demandiez si vous aimerez les restaurants du coin. Vous demandez à vos cinq voisins les plus proches (littéralement, les cinq personnes qui habitent le plus près de chez vous) ce qu'ils en pensent. Si quatre sur cinq disent « le restaurant thaï est excellent », vous l'essaieriez probablement. C'est KNN avec k=5.

L'idée clé est que la proximité dans l'espace des caractéristiques implique la similarité. Deux chansons avec un tempo, une énergie et une dansabilité similaires sont « proches » l'une de l'autre, et les utilisateurs qui aiment l'une tendent à aimer l'autre.

Le choix de k est important. Avec k=1, vous ne demandez l'avis qu'à votre voisin immédiat — son opinion pourrait être une valeur aberrante. Avec k=100, vous interrogez tout le quartier, ce qui lisse les particularités individuelles mais risque d'inclure des personnes vivant si loin que leurs goûts ne sont plus pertinents. k=3 à k=10 est généralement la plage idéale.

KNN est un « apprenant paresseux » — il ne fait aucun travail pendant l'entraînement (il se contente de stocker les données) et effectue tout le travail lors de la prédiction (calcul des distances avec chaque point d'entraînement). Cela rend l'entraînement instantané mais la prédiction lente sur de grands jeux de données.

## Comment ça fonctionne

### Étape 1 : Stocker les données d'entraînement

```
fit(X_train, y_train) -> store X_train and y_train
```

En clair, cela signifie : mémoriser toutes les chansons et leurs étiquettes. Il n'y a pas de modèle à apprendre — les données d'entraînement *sont* le modèle.

### Étape 2 : Calculer les distances avec tous les points d'entraînement

Pour un nouveau point de requête x_q, on calcule la distance euclidienne avec chaque point d'entraînement :

$$
d(x_q, x_i) = \sqrt{\sum_{j=1}^{p} (x_{q,j} - x_{i,j})^2}
$$

En clair, cela signifie : mesurer à quel point la nouvelle chanson diffère de chaque chanson du catalogue, caractéristique par caractéristique, puis combiner ces différences en un seul nombre. Plus le nombre est petit, plus les chansons sont similaires.

### Étape 3 : Trouver les k voisins les plus proches

Trier tous les points d'entraînement par distance et prendre les k plus proches.

En clair, cela signifie : parmi l'ensemble de votre catalogue, trouver les k chansons les plus similaires à la chanson requête.

### Étape 4 : Voter

$$
\hat{y} = \text{mode}\{y_{i_1}, y_{i_2}, \ldots, y_{i_k}\}
$$

En clair, cela signifie : examiner les étiquettes des k chansons les plus proches. Si 4 sur 5 appartiennent à la catégorie « aimée », prédire « aimée ». La majorité l'emporte.

### Étape 5 : Estimer les probabilités

$$
P(\text{class} = c) = \frac{\text{count of class } c \text{ among k neighbors}}{k}
$$

En clair, cela signifie : si 3 voisins sur 5 sont « aimés », l'estimation de probabilité est de 60 %. Cela vous donne un niveau de confiance, pas seulement une prédiction catégorique.

## En Rust

```rust
use ndarray::array;
use ix_supervised::knn::KNN;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{accuracy, precision, recall};

fn main() {
    // Features: [tempo_bpm, energy, danceability, acousticness, valence]
    // Labels: 0 = user dislikes, 1 = user likes
    let x_train = array![
        [120.0, 0.8, 0.9, 0.1, 0.7],  // liked (upbeat pop)
        [125.0, 0.9, 0.85, 0.05, 0.8], // liked (dance)
        [115.0, 0.7, 0.8, 0.15, 0.6],  // liked (pop)
        [60.0,  0.2, 0.1, 0.9, 0.3],   // disliked (slow acoustic)
        [70.0,  0.3, 0.2, 0.85, 0.2],  // disliked (ballad)
        [55.0,  0.1, 0.15, 0.95, 0.25],// disliked (ambient)
    ];
    let y_train = array![1, 1, 1, 0, 0, 0];

    // Build KNN classifier with k=3
    let mut knn = KNN::new(3);
    knn.fit(&x_train, &y_train);

    // Classify new songs
    let new_songs = array![
        [118.0, 0.75, 0.82, 0.12, 0.65],  // similar to upbeat cluster
        [65.0,  0.25, 0.18, 0.88, 0.28],   // similar to acoustic cluster
    ];

    let predictions = knn.predict(&new_songs);
    println!("Predictions: {}", predictions);
    // Expected: [1, 0] (liked, disliked)

    // Get probability estimates
    let proba = knn.predict_proba(&new_songs);
    println!("Like probabilities: {}", proba.column(1));

    // Evaluate on training set
    let train_pred = knn.predict(&x_train);
    println!("Training accuracy: {:.2}%", accuracy(&y_train, &train_pred) * 100.0);
}
```

## Quand l'utiliser

| Situation | KNN | Alternative | Pourquoi |
|---|---|---|---|
| Petit jeu de données, peu de caractéristiques | Oui | — | Simple, pas d'entraînement, souvent précis |
| Besoin d'une référence rapide | Oui | — | Aucun réglage d'hyperparamètres (il suffit de choisir k) |
| Frontières de décision non linéaires complexes | Oui | — | KNN peut modéliser n'importe quelle forme de frontière |
| Grand jeu de données (100K+ lignes) | Non | Régression logistique, arbre de décision | Le calcul par force brute de la distance à chaque point est O(n) par requête |
| Données de haute dimension (100+ caractéristiques) | Non | Forêt aléatoire, SVM | Le fléau de la dimensionnalité — les distances perdent leur sens |
| Besoin d'un modèle interprétable | Partiel | Arbre de décision | « Voici vos 5 plus proches voisins » est relativement interprétable |
| Données en flux continu | Non | Régression logistique | KNN doit stocker toutes les données ; ajouter des points est peu coûteux mais la prédiction ralentit |

## Paramètres clés

| Paramètre | Défaut | Description |
|---|---|---|
| `k` | (requis) | Nombre de voisins à considérer. Doit être un entier positif. |

### Choisir k

| k | Comportement |
|---|---|
| 1 | Plus proche voisin. Très sensible au bruit. La frontière de décision est irrégulière. |
| 3-7 | Bonne plage de départ. Équilibre entre lissage et sensibilité locale. |
| sqrt(n) | Une heuristique courante pour les grands jeux de données. |
| n | Prédit la classe majoritaire pour chaque point (inutile). |

**Règle générale :** Utiliser un k impair pour la classification binaire afin d'éviter les égalités.

## Pièges

**Le fléau de la dimensionnalité.** En haute dimension, tous les points deviennent approximativement équidistants. La distance euclidienne perd son sens au-delà d'environ 20 caractéristiques. Réduisez d'abord la dimensionnalité (ACP, sélection de caractéristiques) ou utilisez un algorithme différent.

**La mise à l'échelle des caractéristiques est essentielle.** KNN utilise les distances euclidiennes brutes. Si le tempo varie de 50 à 200 et la valence de 0 à 1, le tempo dominera complètement le calcul de distance. Normalisez toujours les caractéristiques à la même échelle avant d'utiliser KNN.

**Prédiction lente.** L'implémentation ix utilise le calcul de distance par force brute — elle calcule la distance à chaque point d'entraînement pour chaque requête. Cela coûte O(n * p) par prédiction. Pour les grands jeux de données, des méthodes approximatives (KD-trees, ball trees) sont nécessaires.

**Sensible aux caractéristiques non pertinentes.** Si vous incluez des caractéristiques sans rapport avec la tâche (par ex., le numéro de piste dans l'album), elles ajoutent du bruit au calcul de distance et dégradent les performances. N'incluez que les caractéristiques qui comptent.

**Consommation mémoire élevée.** KNN stocke l'intégralité du jeu de données d'entraînement. Contrairement aux modèles paramétriques qui compressent les données en un petit ensemble de poids, le « modèle » de KNN est constitué des données elles-mêmes.

## Pour aller plus loin

- **Métriques de distance :** Le module `ix_math::distance` d'ix fournit les fonctions `euclidean`, `manhattan`, `cosine_distance`, `chebyshev` et `minkowski`. Différentes métriques conviennent à différents types de données.
- **Vote pondéré :** Au lieu de votes égaux, pondérer le vote de chaque voisin par l'inverse de sa distance. Les voisins plus proches ont plus d'influence.
- **Plus proches voisins approximatifs :** Pour la recherche de similarité à grande échelle, le crate `ix-gpu` offre une recherche vectorielle par lots sur GPU, ce qui peut accélérer considérablement l'étape de recherche des voisins.
- **Réduction de dimensionnalité :** Utilisez `ix-unsupervised` pour l'ACP ou d'autres méthodes de réduction de dimensionnalité afin de combattre le fléau de la dimensionnalité avant d'appliquer KNN.
