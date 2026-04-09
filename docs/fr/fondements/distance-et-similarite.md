# Distance et similarite

> Comment mesurer la "proximite" ou la "similarite" entre deux elements -- le fondement de KNN, du clustering et des systemes de recommandation.

## Le probleme

Vous construisez un moteur de recommandation musicale. Un utilisateur aime la chanson A. Vous disposez de 10 000 autres chansons dans votre catalogue, chacune decrite par des caracteristiques : tempo, energie, dansabilite, volume sonore, acousticite. Quelles chansons sont les plus *similaires* a la chanson A ?

Vous avez besoin d'un moyen de mesurer la "distance" entre deux chansons quelconques dans l'espace des caracteristiques. Differentes metriques de distance donnent des reponses differentes -- et choisir la bonne compte plus qu'on ne le pense.

## L'intuition

Imaginez chaque chanson comme un point dans l'espace, ou chaque caracteristique est une dimension. Deux chansons qui se ressemblent sont "proches" dans cet espace. La question est : comment definir "proche" ?

### Distance euclidienne : a vol d'oiseau

La distance en ligne droite entre deux points. Le theoreme de Pythagore generalise a un nombre quelconque de dimensions.

```
Chanson A : [120 bpm, 0.8 energie, 0.7 danse]
Chanson B : [125 bpm, 0.75 energie, 0.65 danse]

Distance = sqrt((120-125)^2 + (0.8-0.75)^2 + (0.7-0.65)^2)
         = sqrt(25 + 0.0025 + 0.0025)
         = sqrt(25.005) ~ 5.0
```

**Probleme** : le BPM va de 60 a 200 mais l'energie va de 0 a 1. La dimension BPM domine completement. C'est pourquoi il faut standardiser les caracteristiques au prealable.

### Distance de Manhattan : les pates de maisons

La distance que vous parcourriez dans une grille urbaine -- uniquement horizontal ou vertical, pas de diagonales.

```
Manhattan = |120-125| + |0.8-0.75| + |0.7-0.65|
          = 5 + 0.05 + 0.05 = 5.1
```

Plus robuste aux valeurs aberrantes que la distance euclidienne car il n'y a pas d'elevation au carre (les outliers sur une dimension ne dominent pas autant).

### Similarite cosinus : la direction, pas la magnitude

Mesure l'angle entre deux vecteurs, en ignorant leur longueur. Deux vecteurs pointant dans la meme direction ont une similarite cosinus de 1, des vecteurs perpendiculaires ont 0, des vecteurs opposes ont -1.

```
Chanson A : [120, 0.8, 0.7]
Chanson B : [240, 1.6, 1.4]  <- deux fois plus grand, mais meme direction !

Similarite cosinus = 1.0 (direction identique)
Distance euclidienne = grande (magnitudes tres differentes)
```

**Quand utiliser le cosinus** : Quand la *direction* (proportions relatives) compte plus que la *magnitude* (valeurs absolues). C'est courant en analyse de texte (un document long a plus de mots mais le meme sujet) et dans les systemes de recommandation.

### Distance de Tchebychev : la pire dimension

La difference maximale sur une dimension quelconque :

```
Tchebychev = max(|120-125|, |0.8-0.75|, |0.7-0.65|) = 5
```

Utile quand vous vous souciez de la deviation la plus defavorable sur une seule caracteristique.

## Fonctionnement detaille

### Distance euclidienne

`d(a, b) = sqrt(somme((ai - bi)^2))`

En clair : elever les differences au carre, les additionner, prendre la racine carree. Pour comparer des distances (sans avoir besoin de la valeur exacte), vous pouvez eviter la racine carree -- `euclidean_squared` est plus rapide.

### Distance de Manhattan

`d(a, b) = somme(|ai - bi|)`

En clair : prendre les differences en valeur absolue et les additionner. Aussi appelee distance L1 ou distance du taxi.

### Distance de Minkowski (generalisation)

`d(a, b) = (somme(|ai - bi|^p))^(1/p)`

En clair : Minkowski avec p=1 donne Manhattan, p=2 donne Euclidienne, p->infini donne Tchebychev. Le parametre p controle a quel point on penalise les grandes differences sur une seule dimension.

### Similarite cosinus

`cos(a, b) = (a . b) / (||a|| x ||b||)`

En clair : le produit scalaire de a et b, divise par le produit de leurs longueurs. Cela normalise la magnitude, ne laissant que la relation directionnelle.

**Distance cosinus** = 1 - similarite cosinus (donc 0 signifie identique, 2 signifie oppose).

## En Rust

Toutes les fonctions de distance sont dans `ix_math::distance` :

```rust
use ndarray::array;
use ix_math::distance;

let song_a = array![120.0, 0.8, 0.7, -5.0, 0.3];
let song_b = array![125.0, 0.75, 0.65, -6.0, 0.25];

// Euclidienne (ligne droite)
let d = distance::euclidean(&song_a, &song_b).unwrap();

// Euclidienne au carre (plus rapide pour les comparaisons -- pas de sqrt)
let d2 = distance::euclidean_squared(&song_a, &song_b).unwrap();

// Manhattan (pates de maisons)
let m = distance::manhattan(&song_a, &song_b).unwrap();

// Similarite cosinus (correspondance directionnelle, -1 a 1)
let cos = distance::cosine_similarity(&song_a, &song_b).unwrap();

// Distance cosinus (0 = identique, 2 = oppose)
let cos_d = distance::cosine_distance(&song_a, &song_b).unwrap();

// Minkowski (generalisee, p=3)
let mink = distance::minkowski(&song_a, &song_b, 3.0).unwrap();

// Tchebychev (difference maximale sur une dimension)
let cheb = distance::chebyshev(&song_a, &song_b).unwrap();
```

### Trouver les plus proches voisins

Voici un exemple pratique -- trouver les 3 chansons les plus similaires :

```rust
use ndarray::{array, Array1};
use ix_math::distance;

let query = array![120.0, 0.8, 0.7];
let catalog = vec![
    array![125.0, 0.75, 0.65],  // Chanson 1
    array![90.0, 0.3, 0.9],     // Chanson 2
    array![118.0, 0.82, 0.72],  // Chanson 3
    array![140.0, 0.9, 0.5],    // Chanson 4
];

// Calculer les distances et trier
let mut distances: Vec<(usize, f64)> = catalog.iter()
    .enumerate()
    .map(|(i, song)| (i, distance::euclidean(&query, song).unwrap()))
    .collect();

distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

// Les 3 plus proches
for (idx, dist) in distances.iter().take(3) {
    println!("Chanson {}: distance = {:.4}", idx, dist);
}
```

> Voir [`examples/unsupervised/kmeans_clustering.rs`](../../examples/unsupervised/kmeans_clustering.rs) pour le clustering avec des metriques de distance.

## Quand les utiliser

| Metrique de distance | Ideale pour | A eviter quand |
|---------------------|-------------|----------------|
| **Euclidienne** | Usage general ; caracteristiques sur la meme echelle | Les caracteristiques ont des echelles tres differentes (standardisez d'abord !) |
| **Euclidienne au carre** | Comparer des distances (pas besoin de la valeur reelle) | Vous avez besoin de la valeur exacte de la distance |
| **Manhattan** | Donnees en haute dimension ; donnees creuses ; robuste aux outliers | Les caracteristiques sont correlees |
| **Cosinus** | Similarite textuelle ; quand la magnitude est sans importance | La magnitude compte (ex. prix reels) |
| **Minkowski (p)** | Quand vous voulez regler la sensibilite | Incertain sur le p a choisir (commencez avec p=2, c.-a-d. euclidienne) |
| **Tchebychev** | Quand la deviation la plus defavorable sur une dimension compte | La plupart des taches ML (trop sensible a une seule dimension) |

## Parametres cles

| Parametre | Ce qu'il controle |
|-----------|-------------------|
| p (Minkowski) | A quel point les grandes differences sur une seule dimension sont penalisees. p=1 est Manhattan (tolerant), p=2 est Euclidienne (equilibre), p->infini est Tchebychev (intolerant) |

## Pieges courants

- **Toujours standardiser d'abord.** Si une caracteristique va de 0 a 1000 et une autre de 0 a 1, la distance euclidienne est dominee par la premiere. Utilisez `linalg::standardize()` pour normaliser a moyenne nulle et variance unitaire.
- **La haute dimension est contre-intuitive.** En haute dimension (100+ caracteristiques), tous les points tendent a etre approximativement equidistants -- les metriques de distance perdent en pertinence. C'est la "malediction de la dimensionnalite". Envisagez l'ACP ou la selection de caracteristiques pour reduire les dimensions au prealable.
- **La similarite cosinus ne voit pas la magnitude.** Les vecteurs [1, 2, 3] et [100, 200, 300] ont une similarite cosinus de 1.0. Si la magnitude compte (ex. montants d'achat), utilisez plutot la distance euclidienne ou Manhattan.
- **Les valeurs manquantes cassent tout.** Si certaines caracteristiques sont absentes, les distances sont indefinies. Soit vous imputez les valeurs manquantes, soit vous utilisez une metrique qui les gere.

## Pour aller plus loin

- **Utilise ceci** : [KNN](../apprentissage-supervise/knn.md) -- classifie les points par leurs plus proches voisins
- **Utilise ceci** : [K-Means](../apprentissage-non-supervise/kmeans.md) -- regroupe les points par distance aux centroides
- **Utilise ceci** : [DBSCAN](../apprentissage-non-supervise/dbscan.md) -- clustering par densite utilisant des voisinages epsilon
- **Utilise ceci** : [Recherche de similarite GPU](../calcul-gpu/recherche-similarite.md) -- similarite cosinus acceleree sur GPU a grande echelle
- **Retour a** : [INDEX](../INDEX.md)
