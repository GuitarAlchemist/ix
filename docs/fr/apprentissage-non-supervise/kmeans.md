# Clustering K-Means

> Partitionner les données en K groupes en assignant itérativement chaque point au centroïde le plus proche, puis en mettant à jour les centroïdes.

## Le problème

Vous gérez une plateforme e-commerce comptant 50 000 clients. Chaque client possède des caractéristiques : panier moyen, fréquence d'achat, ancienneté de la dernière commande et nombre de catégories consultées. Vous souhaitez segmenter ces clients en groupes distincts pour des campagnes marketing ciblées — mais personne ne vous a dit quels étaient ces groupes. C'est à l'algorithme de les découvrir.

C'est de l'apprentissage non supervisé. Il n'y a pas d'étiquettes. Vous disposez uniquement de données et d'une question : « Quels regroupements naturels existent ? »

## L'intuition

Imaginez que vous organisez une fête et devez placer 3 buffets dans une grande salle remplie de monde. Vous voulez que chaque invité soit proche d'au moins un buffet.

1. **Départ** : Placez 3 buffets au hasard.
2. **Affectation** : Chaque invité se dirige vers le buffet le plus proche.
3. **Mise à jour** : Déplacez chaque buffet au centre des invités qui l'entourent.
4. **Répétition** : Les invités se réaffectent au buffet désormais le plus proche, les buffets bougent à nouveau.
5. **Arrêt** : Quand plus personne ne change de buffet.

C'est K-Means. Les « buffets » sont des **centroïdes** (centres de clusters), et l'algorithme alterne entre affectation des points et mise à jour des centroïdes jusqu'à convergence.

### Initialisation K-Means++

Des centroïdes initiaux aléatoires peuvent donner de mauvais résultats. K-Means++ est plus malin : il choisit le premier centroïde au hasard, puis sélectionne chaque centroïde suivant avec une probabilité proportionnelle à sa distance aux centroïdes existants. Cela espace les centroïdes initiaux, ce qui produit des clusters meilleurs et plus stables.

ix utilise K-Means++ par défaut.

## Comment ça fonctionne

**Entrée** : Jeu de données X (n points, d caractéristiques), nombre de clusters K.

**Algorithme** :

1. **Initialiser** K centroïdes avec K-Means++ (ou aléatoirement)
2. **Affecter** chaque point au centroïde le plus proche :

   `cluster(xᵢ) = argmin_k ||xᵢ - μₖ||²`

   En clair : pour chaque point, on trouve le centroïde le plus proche (en distance euclidienne) et on l'affecte à ce cluster.

3. **Mettre à jour** chaque centroïde en calculant la moyenne de ses points assignés :

   `μₖ = (1/|Cₖ|) × Σᵢ∈Cₖ xᵢ`

   En clair : on déplace chaque centroïde à la position moyenne de tous les points actuellement dans son cluster.

4. **Répéter** les étapes 2-3 jusqu'à stabilisation des affectations ou atteinte du nombre maximal d'itérations.

**Inertie** (somme intra-cluster des carrés) :

`inertia = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²`

En clair : la distance totale de tous les points à leurs centroïdes assignés. Plus c'est bas, mieux c'est. Utilisez cette mesure pour comparer différentes valeurs de K (la « méthode du coude » — tracez l'inertie en fonction de K et cherchez le point d'inflexion).

## En Rust

```rust
use ndarray::array;
use ix_unsupervised::{KMeans, Clusterer};

// Données clients : [panier_moyen, fréquence, ancienneté, catégories]
let customers = array![
    [50.0, 12.0, 5.0, 3.0],    // Fréquent, dépensier modéré
    [45.0, 10.0, 7.0, 2.0],    // Similaire au précédent
    [200.0, 2.0, 30.0, 8.0],   // Rare, panier élevé
    [180.0, 3.0, 25.0, 7.0],   // Similaire au précédent
    [15.0, 1.0, 90.0, 1.0],    // Inactif, faible valeur
    [10.0, 1.0, 120.0, 1.0],   // Similaire au précédent
];

// Segmenter en 3 groupes
let mut kmeans = KMeans::new(3).with_seed(42);
let labels = kmeans.fit_predict(&customers);

println!("Affectations : {:?}", labels);
// ex. [0, 0, 1, 1, 2, 2] — trois segments clients distincts

// Vérifier la qualité des clusters
if let Some(centroids) = &kmeans.centroids {
    let score = ix_unsupervised::inertia(&customers, &labels, centroids);
    println!("Inertie : {:.2}", score);
}
```

> Voir [`examples/unsupervised/kmeans_clustering.rs`](../../examples/unsupervised/kmeans_clustering.rs) pour la version complète exécutable.

### Méthode du coude : choisir K

```rust
use ix_unsupervised::{KMeans, Clusterer, inertia};

for k in 2..=8 {
    let mut kmeans = KMeans::new(k).with_seed(42);
    let labels = kmeans.fit_predict(&data);
    let score = inertia(&data, &labels, kmeans.centroids.as_ref().unwrap());
    println!("K={}: inertie={:.2}", k, score);
}
// Cherchez le « coude » — là où ajouter plus de clusters n'apporte plus grand-chose
```

## Quand l'utiliser

| Algorithme | Idéal pour | Limites |
|-----------|----------|-------------|
| **K-Means** | Clusters approximativement sphériques de taille similaire | Il faut spécifier K ; suppose des clusters de taille égale |
| **DBSCAN** | Formes irrégulières, détection d'outliers | Sensible à eps/min_points ; en difficulté avec des densités variables |
| **ACP** | Réduire les dimensions avant le clustering | Ce n'est pas un algorithme de clustering en soi |

Choisissez K-Means quand :
- Vous avez une idée approximative du nombre de clusters souhaité
- Les clusters sont à peu près sphériques (ni allongés ni irréguliers)
- Vous voulez des résultats rapides et scalables

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `k` | Nombre de clusters | Utilisez la méthode du coude ou votre connaissance du domaine |
| `max_iterations` | Quand s'arrêter si pas de convergence | La valeur par défaut suffit généralement (100-300) |
| `seed` | Reproductibilité de l'initialisation K-Means++ | Fixez-la pour des résultats cohérents |

## Pièges courants

- **K-Means trouve toujours K clusters**, même si les données comportent moins de groupes naturels. Testez avec différentes valeurs de K et validez.
- **Sensible à l'échelle.** Standardisez toujours vos caractéristiques au préalable — une caractéristique variant de 0 à 1000 dominera une autre variant de 0 à 1. Utilisez `linalg::standardize()`.
- **Sensible à l'initialisation.** Même avec K-Means++, les résultats peuvent varier. Exécutez plusieurs fois avec différentes graines et gardez la meilleure inertie.
- **Incapable de détecter des clusters non sphériques.** Si vos clusters sont allongés, en anneau ou irréguliers, utilisez plutôt DBSCAN.
- **Les outliers déforment les centroïdes.** Un seul point extrême tire le centroïde de son cluster loin du vrai centre. Envisagez de supprimer les outliers au préalable ou d'utiliser une alternative robuste.

## Pour aller plus loin

- **Alternative** : [DBSCAN](dbscan.md) — clustering par densité qui détecte les formes irrégulières
- **Prétraitement** : [ACP](acp.md) — réduire les dimensions avant le clustering pour de meilleurs résultats
- **Fondements** : [Distance et similarité](../foundations/distance-and-similarity.md) — les métriques de distance utilisées par K-Means
- **Cas pratique** : [Détection de fraude](../cas-pratiques/detection-fraude.md) — combiner clustering et classification
