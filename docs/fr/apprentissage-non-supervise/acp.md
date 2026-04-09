# ACP (Analyse en Composantes Principales)

> Réduire le nombre de caractéristiques tout en conservant l'essentiel de l'information — visualiser des données multidimensionnelles en 2D ou 3D.

## Le problème

Vous analysez un jeu de données comportant 50 caractéristiques par échantillon — revenus, âge, historique d'achats, habitudes de navigation, données de localisation, etc. Vous souhaitez :

1. **Visualiser** les données (impossible de tracer 50 dimensions)
2. **Accélérer** les algorithmes en aval (KNN avec 50 caractéristiques est lent)
3. **Supprimer le bruit** (certaines caractéristiques sont redondantes ou bruitées)

Vous devez réduire 50 caractéristiques à une poignée — disons 2 ou 3 — en perdant le moins d'information possible. L'ACP trouve le meilleur résumé basse dimension de vos données.

## L'intuition

Imaginez que vous photographiez un objet 3D. Une photo de face capture l'essentiel du détail intéressant. Une photo de côté ajoute un peu plus. Une photo du dessus n'apporte presque rien car l'objet est plat. L'ACP trouve les « meilleurs angles de prise de vue » pour vos données.

Plus précisément : vos 50 caractéristiques contiennent de la redondance. Si les revenus et les dépenses sont fortement corrélés, on peut presque capturer les deux avec un seul nombre (un axe « richesse »). L'ACP trouve ces axes naturels — les directions dans lesquelles vos données varient le plus — et les classe par importance.

**Composante principale 1** : La direction de variance maximale (le plus grand étalement). Ce seul axe capture le maximum d'information possible.

**Composante principale 2** : La direction de variance maximale restante, perpendiculaire à CP1.

Et ainsi de suite. En général, les 2-3 premières composantes capturent 80-95 % de la variance totale, et vous pouvez ignorer le reste.

## Comment ça fonctionne

**Entrée** : Jeu de données X (n échantillons, d caractéristiques), nombre cible de composantes k.

**Étapes :**

1. **Centrer les données** : Soustraire la moyenne de chaque caractéristique pour centrer les données à zéro.

   `X_centré = X - moyenne(X)`

   En clair : on décale les données pour que leur centre soit à l'origine. Cela ne perd aucune information — cela simplifie juste les calculs.

2. **Calculer la matrice de covariance** : Une matrice d×d qui capture les relations entre caractéristiques.

   `C = (1/n) × X_centré^T × X_centré`

   En clair : l'entrée C[i][j] indique à quel point la caractéristique i et la caractéristique j varient ensemble. Les entrées diagonales sont la variance de chaque caractéristique.

3. **Trouver les vecteurs propres et les valeurs propres** : Les vecteurs propres de C sont les directions des composantes principales. Les valeurs propres indiquent quelle proportion de variance chaque direction capture.

   En clair : les vecteurs propres sont les « axes naturels » de la dispersion de vos données. La valeur propre est « l'importance » de chaque axe.

4. **Trier par valeur propre** (décroissante) et garder les k premiers vecteurs propres.

5. **Projeter** : Multiplier les données centrées par les k premiers vecteurs propres pour obtenir la représentation réduite.

   `X_réduit = X_centré × W_k`

   où W_k est une matrice d×k des k premiers vecteurs propres.

**Ratio de variance expliquée** : La fraction de la variance totale capturée par chaque composante. Si CP1 a une valeur propre de 50 et le total est de 100, CP1 explique 50 % de la variance.

## En Rust

```rust
use ndarray::array;
use ix_unsupervised::{PCA, DimensionReducer};

// 6 clients avec 4 caractéristiques
let data = array![
    [65000.0, 35.0, 12.0, 500.0],   // revenus, âge, achats, dépense_moy
    [72000.0, 42.0, 15.0, 620.0],
    [45000.0, 28.0, 8.0, 300.0],
    [120000.0, 55.0, 25.0, 950.0],
    [38000.0, 23.0, 5.0, 200.0],
    [95000.0, 48.0, 20.0, 800.0],
];

// Réduire à 2 composantes pour la visualisation
let mut pca = PCA::new(2);
let reduced = pca.fit_transform(&data);

println!("Forme originale : {:?}", data.dim());    // (6, 4)
println!("Forme réduite : {:?}", reduced.dim());   // (6, 2)

// Quelle proportion d'information avons-nous conservée ?
if let Some(ratios) = pca.explained_variance_ratio() {
    println!("Variance expliquée : {:?}", ratios);
    let total: f64 = ratios.sum();
    println!("Total : {:.1}%", total * 100.0);
}

// Les directions des composantes principales
if let Some(components) = pca.components() {
    println!("Direction CP1 : {:?}", components.row(0));
    println!("Direction CP2 : {:?}", components.row(1));
}
```

### Pipeline ACP + Clustering

Un schéma classique — réduire les dimensions d'abord, puis appliquer le clustering :

```rust
use ix_unsupervised::{PCA, DimensionReducer, KMeans, Clusterer};

// Données de haute dimension
let data = /* ... jeu de données à 50 caractéristiques ... */;

// Étape 1 : Réduire à 5 composantes
let mut pca = PCA::new(5);
let reduced = pca.fit_transform(&data);

// Étape 2 : Clustering dans l'espace réduit (bien plus rapide)
let mut kmeans = KMeans::new(4).with_seed(42);
let labels = kmeans.fit_predict(&reduced);
```

## Quand l'utiliser

| Situation | Utiliser l'ACP ? |
|-----------|----------|
| Trop de caractéristiques, besoin de réduire | Oui — c'est son usage principal |
| Visualiser des données multidimensionnelles en 2D/3D | Oui — tracer les 2-3 premières composantes |
| Accélérer les algorithmes basés sur la distance (KNN, K-Means) | Oui — moins de dimensions = plus rapide |
| Supprimer la multicolinéarité (caractéristiques corrélées) | Oui — les composantes de l'ACP sont décorrélées |
| Les caractéristiques ont des relations non linéaires | Non — l'ACP ne capture que la structure linéaire |
| Besoin d'interpréter quelles caractéristiques originales comptent | Délicat — les composantes de l'ACP sont des mélanges de caractéristiques |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `n_components` | Dimensions en sortie | Choisir pour que le ratio de variance expliquée > 0,8-0,95. Pour la visualisation : 2 ou 3. |

**Combien de composantes ?** Tracez le ratio cumulé de variance expliquée en fonction du nombre de composantes. Choisissez le « coude » — là où ajouter des composantes supplémentaires apporte un gain décroissant.

## Pièges courants

- **Standardisez d'abord !** L'ACP cherche les directions de variance maximale. Si une caractéristique varie de 0 à 100 000 et une autre de 0 à 1, la première domine uniquement à cause de l'échelle. Utilisez `linalg::standardize()` avant l'ACP.
- **Les composantes de l'ACP sont difficiles à interpréter.** CP1 pourrait être « 0,5 × revenus + 0,3 × âge + 0,4 × achats » — c'est un mélange, pas une seule caractéristique. Si l'interprétabilité compte, envisagez plutôt la sélection de variables.
- **L'ACP est linéaire.** Elle ne peut pas capturer de structure non linéaire. Si vos données reposent sur une courbe ou une variété, l'ACP donnera une réduction médiocre.
- **N'utilisez pas l'ACP sur des données catégorielles.** L'ACP suppose des caractéristiques continues. Pour les données catégorielles, envisagez l'Analyse des Correspondances Multiples (pas encore dans ix).
- **La perte d'information est inévitable.** Vous échangez un peu de précision contre de la simplicité. Vérifiez toujours le ratio de variance expliquée pour vous assurer de conserver suffisamment d'information.
- **ix utilise l'itération de puissance** pour la décomposition en valeurs propres, pas LAPACK. C'est précis pour les premières composantes mais moins exact pour un grand nombre de composantes sur de grandes matrices.

## Pour aller plus loin

- **Avant l'ACP** : [Vecteurs et matrices](../foundations/vectors-and-matrices.md) — comprendre les opérations matricielles derrière l'ACP
- **Avant l'ACP** : [Probabilités et statistiques](../foundations/probability-and-statistics.md) — matrices de covariance
- **Combiner avec** : [K-Means](kmeans.md) ou [DBSCAN](dbscan.md) — clustering dans l'espace réduit
- **Cas pratique** : [Détection de fraude](../cas-pratiques/detection-fraude.md) — ACP pour la réduction de dimension avant classification
