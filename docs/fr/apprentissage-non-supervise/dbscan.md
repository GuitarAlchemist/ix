# DBSCAN

> Clustering spatial par densité — détecte des clusters de forme quelconque et identifie automatiquement les outliers.

## Le problème

Vous analysez des données GPS d'une flotte de camions de livraison. Vous voulez trouver les zones où les camions passent beaucoup de temps (points chauds de livraison, entrepôts, goulets d'étranglement). Mais ces zones ne forment pas de cercles bien nets — elles suivent les routes, se concentrent autour des carrefours et ont des formes irrégulières. Vous devez aussi identifier les points GPS qui n'appartiennent à aucun cluster (le bruit — erreurs de signal ou arrêts ponctuels).

K-Means ne peut pas faire cela. Il ne trouve que des clusters sphériques et force chaque point dans un cluster. Vous avez besoin d'un algorithme qui comprend la *densité*.

## L'intuition

Pensez à une foule lors d'un concert. Les gens forment naturellement des groupes denses — près de la scène, au bar, vers les sorties. Entre ces groupes, il y a des zones clairsemées avec peu de monde.

DBSCAN fonctionne ainsi :
1. Prenez une personne quelconque. Regardez autour d'elle dans un rayon d'un bras (distance ε).
2. S'il y a au moins `min_points` personnes à portée, cette personne est dans une **zone dense** — un cluster commence.
3. Chacun de ses voisins regarde aussi autour de lui. S'ils ont aussi suffisamment de voisins, le cluster s'étend.
4. On continue l'expansion tant qu'on trouve des connexions denses.
5. Les personnes inaccessibles depuis une zone dense sont du **bruit** (outliers).

L'idée clé : les clusters sont des régions connectées de haute densité, séparées par des régions de basse densité. Pas besoin de spécifier le nombre de clusters — DBSCAN les découvre automatiquement.

## Comment ça fonctionne

**Entrée** : Jeu de données X, rayon de voisinage ε (eps), nombre minimum de voisins `min_points`.

**Trois types de points :**
- **Point noyau** : A au moins `min_points` voisins dans un rayon ε
- **Point frontière** : Situé à distance ε d'un point noyau, mais n'a pas assez de voisins lui-même
- **Point bruit** : Ni noyau ni frontière — un outlier

**Algorithme :**
1. Pour chaque point non visité p :
   - Trouver tous les points à distance ε de p
   - Si moins de `min_points` voisins → marquer comme bruit (provisoirement)
   - Si assez de voisins → p est un point noyau. Démarrer un nouveau cluster :
     - Ajouter p et tous ses ε-voisins au cluster
     - Pour chaque voisin qui est aussi un point noyau, ajouter récursivement *ses* voisins
     - Continuer jusqu'à ce qu'il n'y ait plus de points accessibles par densité

En clair : on part d'un point dense, on étend le cluster vers l'extérieur à travers les points connectés par densité, on s'arrête quand on atteint des régions clairsemées.

**Propriété clé** : Un point bruit à proximité d'aucun point noyau reste en bruit. Un point bruit qui se retrouve dans le rayon ε d'un point noyau d'un autre cluster est reclassé comme point frontière.

## En Rust

```rust
use ndarray::array;
use ix_unsupervised::{DBSCAN, Clusterer};

// Coordonnées GPS des arrêts de camions [latitude, longitude]
let stops = array![
    // Cluster 1 : zone d'entrepôt
    [40.712, -74.006],
    [40.713, -74.005],
    [40.711, -74.007],
    [40.714, -74.004],
    // Cluster 2 : zone de livraison centre-ville
    [40.758, -73.985],
    [40.757, -73.986],
    [40.759, -73.984],
    [40.756, -73.987],
    // Bruit : arrêts ponctuels
    [40.800, -73.950],
    [40.650, -74.100],
];

let mut dbscan = DBSCAN::new(
    0.005,  // eps : ~500m de rayon à la latitude de NYC
    3,      // min_points : au moins 3 voisins nécessaires
);

let labels = dbscan.fit_predict(&stops);
println!("Labels : {:?}", labels);
// Cluster 1 : label 1, Cluster 2 : label 2, Bruit : label 0

// Compter les clusters et le bruit
let n_clusters = *labels.iter().max().unwrap_or(&0);
let n_noise = labels.iter().filter(|&&l| l == 0).count();
println!("{} clusters trouvés, {} points de bruit", n_clusters, n_noise);
```

> Voir [`examples/unsupervised/dbscan_anomaly.rs`](../../examples/unsupervised/dbscan_anomaly.rs) pour la version complète exécutable.

## Quand l'utiliser

| Situation | DBSCAN | K-Means |
|-----------|------------|-------------|
| Nombre de clusters inconnu | Oui — trouve K automatiquement | Non — K doit être spécifié |
| Formes de clusters irrégulières | Oui — suit la densité | Non — suppose des formes sphériques |
| Besoin de détecter des outliers | Oui — étiquette le bruit comme 0 | Non — force tous les points dans des clusters |
| Clusters de tailles différentes | Partiellement — fonctionne si la densité est similaire | Oui — gère les différentes tailles |
| Très grands jeux de données | Plus lent (O(n²) sans index spatial) | Rapide (O(n×k×itérations)) |
| Clusters de densité variable | Médiocre — un seul eps ne capture pas les deux | Également médiocre — problème différent |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Comment le choisir |
|-----------|-----------------|---------------|
| `eps` | Rayon de voisinage | Tracez le graphique k-distance (triez les distances au k-ième plus proche voisin). Le « coude » suggère un bon eps. |
| `min_points` | Nombre minimum de voisins pour être un point noyau | Règle empirique : `min_points ≥ dimensions + 1`. Pour des données bruitées, augmentez-le. Courant : 5-10. |

**Le choix d'eps est la partie difficile.** Trop petit → tout est du bruit. Trop grand → tout forme un seul cluster. Le graphique k-distance aide : calculez la distance de chaque point à son `min_points`-ième plus proche voisin, triez ces distances et cherchez une inflexion nette.

## Pièges courants

- **Sensible à eps et min_points.** Contrairement à K-Means où K est intuitif, le réglage d'eps nécessite de comprendre l'échelle de vos données. Standardisez les caractéristiques au préalable.
- **Ne gère pas la densité variable.** Si un cluster est dense (points espacés de 1 cm) et un autre est clairsemé (points espacés de 1 m), aucun eps unique ne convient aux deux. Envisagez HDBSCAN (pas encore dans ix) ou lancez DBSCAN à plusieurs échelles.
- **Le label bruit est 0.** Dans ix, les labels de clusters commencent à 1. Les points étiquetés 0 sont du bruit/outliers. Ne confondez pas 0 avec « cluster 0 ».
- **O(n²) sans indexation spatiale.** Pour les grands jeux de données, le calcul des distances par paires est coûteux. ix utilise la force brute, ce qui convient jusqu'à environ 10 000 points.
- **Les points frontières sont non-déterministes.** Un point frontière accessible depuis deux clusters est assigné à celui qui le découvre en premier. En pratique, cela a rarement un impact.

## Pour aller plus loin

- **Alternative** : [K-Means](kmeans.md) — quand vous connaissez K et que les clusters sont sphériques
- **Prétraitement** : [ACP](acp.md) — réduire les dimensions avant d'exécuter DBSCAN
- **Fondements** : [Distance et similarité](../foundations/distance-and-similarity.md) — DBSCAN utilise la distance euclidienne
- **Cas pratique** : [Détection d'anomalies](../use-cases/anomaly-detection.md) — utiliser la détection de bruit de DBSCAN pour les anomalies
