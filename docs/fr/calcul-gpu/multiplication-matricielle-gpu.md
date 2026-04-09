# Multiplication matricielle sur GPU

## Le problème

Vous effectuez de l'inférence par batch sur un réseau de neurones. L'opération centrale — faire passer un batch de 256 entrées à travers une couche de 1 024 neurones — est une multiplication matricielle : `Sortie = Entrée * Poids`. Sur CPU, une implémentation naïve de 256 × 1 024 × 768 = ~201 millions de multiplications-additions prend des centaines de millisecondes. Vous avez besoin de le faire en quelques millisecondes car vous avez des dizaines de couches et des contraintes de latence temps réel.

Autres situations où la multiplication matricielle domine :

- **Attention des transformers.** Q * K^T et poids-d'attention * V sont des multiplications matricielles.
- **Lookup d'embeddings + projection.** Multiplier une entrée one-hot (ou creuse) par une matrice d'embedding.
- **Calculs ACP / SVD.** Matrices de covariance, projections et reconstructions se réduisent toutes à des matmul.
- **Recherche de similarité par batch.** Calculer `Requêtes * Corpus^T` pour obtenir tous les produits scalaires par paires (voir [recherche de similarité](./recherche-de-similarite.md)).

---

## L'intuition

La multiplication matricielle est le « hello world » du calcul GPU car elle correspond parfaitement aux forces du GPU :

- Chaque élément de la matrice de sortie est **indépendant** — il ne dépend que d'une ligne de A et d'une colonne de B.
- Chaque élément nécessite la même quantité de travail (un produit scalaire de longueur K).
- Il y a M × N éléments de sortie à calculer, offrant au GPU des millions de tâches indépendantes.

Le shader GPU assigne un thread à chaque élément de sortie `C[ligne][col]`. Ce thread parcourt la dimension partagée K, multipliant `A[ligne][k] * B[k][col]` et accumulant. Avec une grille de threads 16×16 par workgroup, 256 threads calculent une tuile 16×16 de la sortie simultanément.

---

## Comment ça fonctionne

### Les mathématiques

```
C = A * B

où :
  A est M x K  (M lignes, K colonnes)
  B est K x N  (K lignes, N colonnes)
  C est M x N  (M lignes, N colonnes)

C[i][j] = sum(A[i][k] * B[k][j] pour k dans 0..K)
```

**En clair :** chaque élément de la sortie est le produit scalaire d'une ligne de A et d'une colonne de B.

### Disposition plate en row-major

Les fonctions CPU et GPU attendent les matrices comme des **`Vec<f32>` plats en row-major** :

```
Matrice :       Tableau plat :
| 1 2 3 |      [1, 2, 3, 4, 5, 6]
| 4 5 6 |

L'élément [i][j] est à l'indice : i * nb_colonnes + j
```

### Le shader WGSL

Le shader reçoit trois buffers (A, B, C) et un buffer uniforme avec les dimensions (M, N, K) :

```wgsl
@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= M || col >= N) { return; }

    var sum: f32 = 0.0;
    for (var k: u32 = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}
```

**En clair :** chaque thread GPU calcule exactement un élément de la matrice de sortie. Le dispatcher lance `ceil(M/16) * ceil(N/16)` workgroups, chacun avec 256 threads organisés en grille 16×16.

### Géométrie du dispatch

```
workgroups_x = ceil(M / 16)
workgroups_y = ceil(N / 16)
total_threads = workgroups_x * 16 * workgroups_y * 16
```

Les threads dont les coordonnées `(ligne, col)` dépassent les bornes de la matrice font un retour anticipé sans écrire.

---

## En Rust

### Multiplication matricielle de base

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::matmul::{matmul_gpu, matmul_cpu};

let ctx = GpuContext::new().unwrap();

// A est 2x3, B est 3x2 -> C est 2x2
//
//  A = | 1 2 3 |    B = | 7  8 |
//      | 4 5 6 |        | 9 10 |
//                        |11 12 |
let a = vec![1.0_f32, 2.0, 3.0,
             4.0,     5.0, 6.0];
let b = vec![ 7.0_f32,  8.0,
              9.0,     10.0,
             11.0,     12.0];

let (m, k, n) = (2, 3, 2);

// Chemin GPU
let c_gpu = matmul_gpu(&ctx, &a, &b, m, k, n);
// c_gpu = [58.0, 64.0, 139.0, 154.0]
//
// C = | 1*7+2*9+3*11  1*8+2*10+3*12 | = | 58   64 |
//     | 4*7+5*9+6*11  4*8+5*10+6*12 |   |139  154 |

// Repli CPU (même API, même résultat)
let c_cpu = matmul_cpu(&a, &b, m, k, n);

// Vérifier la correspondance
for (g, c) in c_gpu.iter().zip(c_cpu.iter()) {
    assert!((g - c).abs() < 1e-3);
}
```

### Inférence par batch d'un réseau de neurones

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::matmul::matmul_gpu;

let ctx = GpuContext::new().unwrap();

let batch_size = 256;
let input_dim = 768;
let output_dim = 1024;

// Batch d'entrée aléatoire (256 x 768) et matrice de poids (768 x 1024)
let inputs: Vec<f32> = (0..batch_size * input_dim)
    .map(|i| (i as f32 * 0.001).sin())
    .collect();
let weights: Vec<f32> = (0..input_dim * output_dim)
    .map(|i| (i as f32 * 0.0007).cos() * 0.01)
    .collect();

// Passe forward : Sortie = Entrée * Poids
let output = matmul_gpu(&ctx, &inputs, &weights, batch_size, input_dim, output_dim);

assert_eq!(output.len(), batch_size * output_dim);  // 256 * 1024 = 262 144
println!("Inférence par batch terminée : {} sorties", output.len());
```

### Repli CPU quand aucun GPU n'est disponible

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::matmul::{matmul_gpu, matmul_cpu};

let a = vec![1.0_f32, 0.0, 0.0, 1.0]; // identité 2x2
let b = vec![5.0_f32, 6.0, 7.0, 8.0]; // matrice 2x2

let result = match GpuContext::new() {
    Ok(ctx) => matmul_gpu(&ctx, &a, &b, 2, 2, 2),
    Err(_)  => matmul_cpu(&a, &b, 2, 2, 2),
};

println!("Résultat : {:?}", result);  // [5.0, 6.0, 7.0, 8.0]
```

---

## Quand l'utiliser

| Taille de matrice (M x K x N) | Bénéfice GPU | Notes |
|:--|:-:|:--|
| 10 x 10 x 10 | Aucun | Le surcoût de transfert domine |
| 64 x 64 x 64 | Marginal | Le SIMD CPU est compétitif |
| 256 x 768 x 1024 | Significatif (5-20x) | Taille typique d'une couche de NN |
| 1024 x 1024 x 1024 | Important (20-100x) | Clairement le territoire GPU |
| 4096 x 4096 x 4096 | Massif (50-200x) | Le GPU est la seule option pratique |

### `matmul_gpu` vs `matmul_cpu`

| | `matmul_gpu` | `matmul_cpu` |
|--|:-:|:-:|
| Précision | f32 | f32 |
| Parallélisme | Milliers de threads GPU | Un seul thread CPU (pas de SIMD) |
| Coût d'initialisation | ~1ms (premier appel : compilation shader) | Aucun |
| Coût de transfert | Proportionnel à la taille des données | Aucun |
| Idéal pour | Grandes matrices (>= 256 x 256) | Petites matrices ou pas de GPU |

---

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `m` | Nombre de lignes de A / lignes de C | Taille de batch en inférence de réseau de neurones |
| `k` | Dimension partagée (colonnes de A, lignes de B) | Dimension d'entrée / taille cachée |
| `n` | Nombre de colonnes de B / colonnes de C | Dimension de sortie |
| Taille de workgroup | 16 x 16 = 256 threads par groupe | Codé en dur dans le shader ; optimal pour la plupart des GPU |
| Disposition des données | Row-major, `Vec<f32>` plat | L'élément `[i][j]` d'une matrice `M x N` est à l'indice `i * N + j` |

---

## Pièges courants

1. **Tableaux plats, pas de tableaux 2D.** Les deux fonctions prennent des `&[f32]` avec des dimensions explicites `m`, `k`, `n`. Passer les mauvaises dimensions produira des résultats incohérents (ou un panic sur les assertions de longueur). Vérifiez bien que `a.len() == m * k` et `b.len() == k * n`.

2. **La disposition row-major est supposée.** Si vous avez des données column-major (ex. de Fortran ou certaines bibliothèques d'algèbre linéaire), vous devez transposer avant de les passer. Alternativement, inversez les arguments : `matmul(B^T, A^T)` calcule `(A * B)^T` en column-major.

3. **Le repli CPU est un O(M*K*N) naïf.** Il n'utilise ni SIMD, ni blocking de cache, ni multithreading. Pour des charges CPU en production sur de grandes matrices, envisagez de lier contre un BLAS optimisé. Le `matmul_cpu` fourni est conçu comme référence de correction et repli, pas comme implémentation CPU haute performance.

4. **Limites de mémoire GPU.** Trois matrices doivent tenir en VRAM simultanément : A (M*K*4 octets), B (K*N*4 octets), C (M*N*4 octets). Une matrice 10 000 × 10 000 fait 400 Mo. Vérifiez la VRAM de votre GPU avant de tenter de très grandes multiplications.

5. **Pas d'accumulation en place.** Chaque appel alloue de nouveaux buffers GPU, dispatche et lit les résultats. Pour des multiplications itérées (ex. chaîner les couches d'un réseau de neurones), le surcoût des uploads et readbacks répétés est significatif. Une optimisation future garderait les résultats intermédiaires en mémoire GPU entre les opérations.

---

## Pour aller plus loin

- La **multiplication matricielle tuilée avec mémoire partagée** divise le calcul en tuiles qui tiennent dans la mémoire locale rapide du workgroup GPU, améliorant considérablement la localité de cache. Le shader actuel utilise une approche simple par élément ; le tuilage est une optimisation naturelle suivante.
- Les **opérations fusionnées** (ex. matmul + biais + ReLU) réduisent le nombre de dispatches GPU et les allers-retours mémoire. Courant dans les moteurs d'inférence de réseaux de neurones.
- Le support de la **demi-précision (f16)** doublerait le débit sur les GPU avec tensor cores (NVIDIA Ampere et ultérieurs). Le support f16 de WGPU est en cours d'évolution.
- Le crate `ix-nn` utilise des opérations matricielles pour les passes forward et backward des réseaux de neurones. Intégrer `matmul_gpu` comme backend accélérerait l'entraînement et l'inférence.
- La **[recherche de similarité](./recherche-de-similarite.md)** s'appuie sur `matmul_gpu` pour calculer des matrices de similarité par batch en un seul passage GPU.
