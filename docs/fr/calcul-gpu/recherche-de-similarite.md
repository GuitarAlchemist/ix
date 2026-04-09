# Recherche de similarité sur GPU

## Le problème

Vous construisez un moteur de recommandation temps réel. Chaque fois qu'un utilisateur interagit avec un article, vous devez trouver les 10 articles les plus similaires parmi un catalogue de 100 000 vecteurs d'embedding, chacun de 512 dimensions. Une recherche brute sur CPU calcule 100 000 similarités cosinus — soit 51,2 millions de multiplications-additions par requête. À 10 requêtes par seconde, votre CPU est saturé. Vous devez décharger cette charge massivement parallèle sur le GPU, où des milliers de produits scalaires s'exécutent simultanément.

Ce schéma se retrouve dans d'autres contextes :

- **Recherche sémantique.** Étant donné un embedding de requête, trouver les documents les plus proches dans une base vectorielle.
- **Détection de doublons.** Calculer la matrice complète de similarité par paires pour identifier les quasi-doublons.
- **Génération augmentée par récupération (RAG).** Trouver les top-k fragments pertinents avant de les fournir à un LLM.

---

## L'intuition

La similarité cosinus mesure l'angle entre deux vecteurs, en ignorant leur longueur. Deux vecteurs pointant exactement dans la même direction ont une similarité de 1,0. Des vecteurs perpendiculaires ont une similarité de 0,0. Des vecteurs opposés ont une similarité de -1,0.

Sur un GPU, on calcule la similarité cosinus en :
1. Calculant le produit scalaire des deux vecteurs (multiplier chaque paire d'éléments, sommer).
2. Calculant la magnitude (longueur) de chaque vecteur.
3. Divisant : `dot(a, b) / (|a| * |b|)`.

Le shader GPU exécute les trois accumulations en parallèle sur 256 threads via une **réduction de workgroup** : chaque thread somme une tranche du vecteur, puis les threads coopèrent pour fusionner leurs sommes partielles.

Pour la recherche par batch, l'astuce est encore meilleure : aplatir tous les vecteurs en une seule matrice, calculer la matrice entière de produits scalaires requêtes-corpus avec une seule multiplication matricielle GPU, puis normaliser par les normes. Cela calcule les similarités pour **toutes les requêtes contre tous les vecteurs du corpus** en un seul passage GPU.

---

## Comment ça fonctionne

### Similarité cosinus

```
cosinus(a, b) = dot(a, b) / (||a|| * ||b||)

où :
  dot(a, b)  = sum(a_i * b_i)
  ||a||      = sqrt(sum(a_i^2))
```

**En clair :** multiplier les vecteurs élément par élément, sommer les produits et diviser par les longueurs des deux vecteurs. Le résultat indique à quel point les vecteurs sont alignés, indépendamment de l'échelle.

### Schéma de réduction GPU

Le shader WGSL utilise une réduction parallèle :

1. Chacun des 256 threads calcule une somme partielle sur sa tranche du vecteur (boucle à pas de grille).
2. Les sommes partielles sont écrites en mémoire partagée (workgroup).
3. Les threads coopèrent pour diviser le tableau par deux : le thread 0 ajoute la valeur du thread 128, le thread 1 ajoute la valeur du thread 129, et ainsi de suite. Répéter jusqu'à ce que seul le thread 0 ait le total.
4. Le thread 0 écrit le produit scalaire final, norm_a² et norm_b² dans le buffer de sortie.

### Top-k par batch via multiplication matricielle

Pour `nq` requêtes contre `nc` vecteurs de corpus de dimension `d` :

```
matrice_dot = Q (nq x d) * C^T (d x nc) = (nq x nc) produits scalaires
similarité[i][j] = matrice_dot[i][j] / (||requête_i|| * ||corpus_j||)
```

**En clair :** traiter les vecteurs requêtes et corpus comme des matrices, les multiplier (accéléré GPU), puis normaliser chaque cellule par le produit des normes concernées. On obtient chaque similarité requête-corpus d'un seul coup.

---

## En Rust

> Exemple complet exécutable : [`examples/gpu/similarity_search.rs`](../../examples/gpu/similarity_search.rs)

### Similarité cosinus par paires

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::similarity::{cosine_similarity_gpu, cosine_similarity_cpu};

let ctx = GpuContext::new().expect("GPU nécessaire");

let query   = vec![0.1_f32, 0.5, 0.3, 0.8];
let product = vec![0.2_f32, 0.4, 0.3, 0.7];

// Chemin GPU (f32)
let sim_gpu = cosine_similarity_gpu(&ctx, &query, &product);
println!("Similarité GPU : {:.4}", sim_gpu);

// Repli CPU (aussi f32 pour la compatibilité API)
let sim_cpu = cosine_similarity_cpu(&query, &product);
println!("Similarité CPU : {:.4}", sim_cpu);
```

### Produit scalaire

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::similarity::dot_product_gpu;

let ctx = GpuContext::new().unwrap();

let a = vec![1.0_f32, 2.0, 3.0];
let b = vec![4.0_f32, 5.0, 6.0];

let dot = dot_product_gpu(&ctx, &a, &b);
println!("Produit scalaire : {}", dot);  // 1*4 + 2*5 + 3*6 = 32
```

### Matrice de similarité complète

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::batch::similarity_matrix;

let ctx = GpuContext::new().unwrap();

let vectors = vec![
    vec![1.0_f32, 0.0, 0.0],   // "électronique"
    vec![0.0_f32, 1.0, 0.0],   // "vêtements"
    vec![0.7_f32, 0.7, 0.0],   // "mode tech"
];

// Matrice de similarité N x N en un passage GPU
let matrix = similarity_matrix(Some(&ctx), &vectors);

// matrix[0][2] ~ 0.707 (électronique partiellement similaire à mode tech)
// matrix[0][1] ~ 0.0   (électronique orthogonal à vêtements)
for (i, row) in matrix.iter().enumerate() {
    println!("Vecteur {} : {:?}", i, row);
}
```

### Recherche top-k (moteur de recommandation)

```rust
use ix_gpu::batch::top_k_similar;

let query  = vec![0.1_f32, 0.5, 0.3, 0.8];
let corpus = vec![
    vec![0.2, 0.4, 0.3, 0.7],   // Produit A
    vec![0.9, 0.1, 0.0, 0.2],   // Produit B
    vec![0.1, 0.6, 0.2, 0.9],   // Produit C
    vec![0.5, 0.5, 0.5, 0.5],   // Produit D
];

// Trouver les 2 produits les plus similaires (chemin CPU montré ; passer Some(&ctx) pour GPU)
let results = top_k_similar(None, &query, &corpus, 2);

for (index, similarity) in &results {
    println!("Produit {} : similarité {:.4}", index, similarity);
}
// Résultats triés par similarité décroissante
```

### Top-k par batch (plusieurs requêtes à la fois)

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::batch::batch_top_k;

let ctx = GpuContext::new().unwrap();

let queries = vec![
    vec![1.0_f32, 0.0, 0.0],  // Goûts de l'utilisateur A
    vec![0.0_f32, 1.0, 0.0],  // Goûts de l'utilisateur B
];
let corpus = vec![
    vec![0.9, 0.1, 0.0],  // Article 0
    vec![0.1, 0.9, 0.0],  // Article 1
    vec![0.5, 0.5, 0.0],  // Article 2
];

// Chaque utilisateur reçoit ses propres top-2 recommandations
let results = batch_top_k(Some(&ctx), &queries, &corpus, 2);

for (qi, recs) in results.iter().enumerate() {
    println!("Utilisateur {} : {:?}", qi, recs);
}
// Utilisateur 0 : [(0, 0.99..), (2, 0.70..)]  -- préfère l'Article 0
// Utilisateur 1 : [(1, 0.99..), (2, 0.70..)]  -- préfère l'Article 1
```

---

## Quand l'utiliser

| Scénario | Fonction recommandée | Pourquoi |
|----------|---------------------|-----|
| Comparer deux vecteurs | `cosine_similarity_gpu` / `_cpu` | Comparaison directe par paires |
| Trouver les top-k pour une requête | `top_k_similar` | Calcule toutes les similarités du corpus, renvoie trié |
| Trouver les top-k pour plusieurs requêtes | `batch_top_k` | Une seule multiplication matricielle GPU pour toutes les requêtes |
| Similarité par paires pour le clustering | `similarity_matrix` | Matrice N × N complète en un passage |
| Juste le produit scalaire | `dot_product_gpu` | Sans normalisation |

### Décision GPU vs. CPU

| Taille du corpus | Dimensions | Bénéfice GPU |
|:-:|:-:|:-:|
| < 1 000 | < 100 | Négligeable (utiliser CPU) |
| 1 000 - 10 000 | 100 - 512 | Accélération 2-5x |
| 10 000+ | 512+ | Accélération 10-100x |
| 100 000+ | 768+ | Obligatoire pour le temps réel |

---

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `ctx: Option<&GpuContext>` | Chemin GPU vs CPU | Passer `Some(&ctx)` pour GPU, `None` pour repli CPU |
| `k` (fonctions top-k) | Nombre de résultats à renvoyer | Garder petit (10-100) pour la recherche ; matrice complète pour le clustering |
| Dimension des vecteurs | Longueur de chaque embedding | Plus de dimensions = plus de travail par paire = plus de bénéfice GPU |
| Précision `f32` | Tous les chemins GPU utilisent des flottants 32 bits | Suffisant pour la recherche de similarité ; différences avec f64 < 0,001 |

---

## Pièges courants

1. **Le GPU utilise f32, le crate math CPU utilise f64.** Si vous comparez les résultats de similarité GPU à des valeurs calculées par `ix-math` (qui utilise `f64`), attendez-vous à de petits écarts de l'ordre de 1e-4 à 1e-6. C'est acceptable pour le classement mais soyez prudent avec les seuils de correspondance exacte.

2. **Les vecteurs courts ne bénéficient pas du GPU.** Pour des vecteurs de moins de ~64 dimensions, la réduction parallèle a plus de surcoût que la sommation séquentielle. Le repli CPU sera souvent plus rapide.

3. **Le tri top-k se fait sur CPU.** `top_k_similar` et `batch_top_k` calculent les similarités sur GPU mais trient les résultats sur CPU. Pour de très grands corpus, ce tri final peut être un goulot d'étranglement.

4. **Limites mémoire.** Une matrice de similarité pour 100 000 vecteurs fait 100 000² × 4 octets = ~37 Go. Pour les grands jeux de données, utilisez `batch_top_k` (qui calcule une matrice requêtes × corpus, pas corpus × corpus) ou découpez le calcul en tuiles.

5. **Le premier appel est lent.** La compilation du shader se fait à la première invocation. Les appels suivants réutilisent le pipeline compilé et sont bien plus rapides. Envisagez un appel de préchauffage avec des données factices lors de l'initialisation.

---

## Pour aller plus loin

- Les **index de plus proches voisins approximatifs (ANN)** comme HNSW ou IVF-PQ échangent une petite perte de rappel contre des ordres de grandeur en vitesse de recherche. Combinez avec un re-ranking exact accéléré GPU des meilleurs candidats.
- La **distance euclidienne** est disponible via `euclidean_distance_cpu` pour les cas où vous vous intéressez à la distance absolue plutôt qu'à la similarité angulaire. Une version GPU pourrait être ajoutée en adaptant le shader cosinus.
- La **[multiplication matricielle sur GPU](./multiplication-matricielle-gpu.md)** est le composant de base qui rend `batch_top_k` et `similarity_matrix` rapides. Lisez cette doc pour comprendre le compute shader tuilé sous le capot.
- L'**[introduction au calcul GPU](./introduction-calcul-gpu.md)** couvre les fondamentaux WGPU — comment `GpuContext` fonctionne, la gestion des buffers et le modèle de dispatch.
