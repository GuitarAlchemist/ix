# Introduction au calcul GPU

## Le problème

Vous avez un lot de 10 000 vecteurs d'embedding, chacun de 768 dimensions, et vous devez calculer la similarité cosinus entre chaque paire. Sur un CPU, cela représente environ 50 millions de produits scalaires, chacun nécessitant 768 multiplications-additions. Même sur une machine rapide, cela prend des secondes. Un GPU possède des milliers de coeurs capables de calculer chacun un produit scalaire simultanément — transformant des secondes en millisecondes. Mais la programmation GPU a la réputation d'être pénible : API propriétaires, gestion manuelle de la mémoire, compilation de shaders, synchronisation. Vous avez besoin d'un moyen d'obtenir l'accélération GPU sans être noyé dans le boilerplate.

---

## L'intuition

Pensez au CPU comme un mathématicien brillant et unique qui traite les problèmes un par un, très vite. Un GPU est un stade rempli de 5 000 mathématiciens moyens qui peuvent chacun résoudre un problème simple simultanément. Si votre charge de travail est une grande équation avec de profondes dépendances entre les étapes, le mathématicien brillant unique l'emporte. Mais si votre charge est constituée de milliers de problèmes *indépendants* — comme calculer 10 000 produits scalaires — le stade plein de travailleurs termine en le temps qu'il faut pour en faire un seul.

Le coût d'utilisation du stade : vous devez écrire votre problème sur un tableau dans un langage que les travailleurs comprennent (un **shader**), transporter les données au stade (upload vers la mémoire GPU) et ramener les résultats (readback). Pour les petits problèmes, le transport prend plus longtemps que le calcul. Pour les gros problèmes, c'est massivement rentable.

---

## Comment ça fonctionne

### La pile WGPU

ix utilise [WGPU](https://wgpu.rs/), une couche d'abstraction GPU multiplateforme :

| Plateforme | Backend |
|----------|---------|
| Windows | Vulkan ou DX12 (sélection automatique) |
| macOS | Metal |
| Linux | Vulkan |

Vous écrivez les compute shaders en **WGSL** (WebGPU Shading Language), un langage simple de type C. WGPU les compile à l'exécution pour le backend disponible.

### Le pipeline de calcul

Chaque calcul GPU suit ce schéma :

```
1. Initialiser   -> GpuContext::new()
2. Uploader      -> create_buffer_init(label, &data)
3. Compiler      -> create_compute_pipeline(label, shader_source, entry_point)
4. Lier          -> lier les buffers aux variables du shader
5. Dispatcher    -> lancer N workgroups de M threads chacun
6. Readback      -> read_buffer(&output_buffer, size)
```

**En clair :**

- **Initialiser** trouve un GPU et ouvre une connexion (device + queue).
- **Uploader** copie vos données f32 de la RAM CPU vers la VRAM GPU.
- **Compiler** transforme votre code source WGSL en instructions GPU natives.
- **Lier** indique au shader « le buffer 0 est votre entrée A, le buffer 1 est l'entrée B, le buffer 2 est la sortie ».
- **Dispatcher** dit « lancer autant de groupes de threads ; chaque groupe exécute le shader sur sa tranche de données ».
- **Readback** copie les résultats vers la RAM CPU pour les utiliser dans du code Rust normal.

### Pourquoi f32 et non f64 ?

Les GPU sont optimisés pour la virgule flottante 32 bits. La plupart des GPU grand public ont un débit f32 **32 fois** supérieur au f64. Les algorithmes CPU d'ix utilisent `f64` pour la précision, mais tous les chemins GPU utilisent `f32`. Pour la grande majorité des charges ML (recherche de similarité, multiplication matricielle, inférence de réseaux de neurones), le f32 est largement suffisant.

---

## En Rust

> Exemple complet exécutable : [`examples/gpu/similarity_search.rs`](../../examples/gpu/similarity_search.rs)

### Initialiser le contexte GPU

```rust
use ix_gpu::context::GpuContext;

// Initialisation synchrone (bloque jusqu'à ce que le GPU soit prêt)
let ctx = GpuContext::new().expect("Aucun GPU compatible trouvé");

println!("GPU : {}", ctx.gpu_name());     // ex. "NVIDIA GeForce RTX 4090"
println!("Backend : {:?}", ctx.backend()); // ex. Vulkan
```

Pour du code asynchrone (dans un runtime Tokio) :

```rust
use ix_gpu::context::GpuContext;

let ctx = GpuContext::new_async().await.expect("Aucun GPU compatible trouvé");
```

### Créer des buffers

```rust
use ix_gpu::context::GpuContext;

let ctx = GpuContext::new().unwrap();

// Uploader des données f32 vers la mémoire GPU
let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
let gpu_buffer = ctx.create_buffer_init("my_data", &data);

// Créer un buffer de sortie vide (4 floats = 16 octets)
let output = ctx.create_output_buffer("result", 16);

// Lire les résultats après un calcul
let results: Vec<f32> = ctx.read_buffer(&output, 16);
```

### Comprendre les champs de GpuContext

```rust
use ix_gpu::context::GpuContext;

let ctx = GpuContext::new().unwrap();

// Accès direct au device et à la queue wgpu sous-jacents pour un usage avancé
let _device: &wgpu::Device = &ctx.device;
let _queue: &wgpu::Queue = &ctx.queue;

// Informations sur l'adaptateur pour le diagnostic
let name = ctx.gpu_name();      // Nom lisible du GPU
let backend = ctx.backend();     // Vulkan, DX12, Metal, etc.
```

### Gérer « aucun GPU disponible »

```rust
use ix_gpu::context::GpuContext;

match GpuContext::new() {
    Ok(ctx) => {
        println!("Exécution sur GPU : {}", ctx.gpu_name());
        // ... chemin accéléré GPU ...
    }
    Err(e) => {
        eprintln!("GPU indisponible ({}), repli sur CPU", e);
        // ... chemin de repli CPU ...
    }
}
```

Tous les modules d'`ix-gpu` fournissent des fonctions de repli CPU à côté de leurs versions GPU, pour que votre code dégrade gracieusement.

---

## Quand l'utiliser

| Charge de travail | GPU rentable ? | Pourquoi |
|----------|:------------:|-----|
| Recherche de similarité sur 10 000+ vecteurs | Oui | Des milliers de produits scalaires indépendants |
| Multiplication matricielle (grandes matrices) | Oui | Parallélisme massif de multiplications-additions |
| Un seul produit scalaire de 2 vecteurs | Non | Le surcoût de transfert domine |
| Algorithme séquentiel avec dépendances de données | Non | Le GPU ne peut pas paralléliser du travail séquentiel |
| Petit jeu de données (< 1 000 éléments) | Rarement | Le CPU est assez rapide ; le setup GPU est un surcoût |
| Inférence par batch sur beaucoup d'entrées | Oui | Chaque entrée est indépendante |

**Règle pratique :** si vous pouvez exprimer votre problème comme « faire la même chose sur des milliers de points de données indépendants », l'accélération GPU sera bénéfique. Le point de croisement se situe typiquement autour de 1 000 à 10 000 éléments, selon la dimensionnalité des vecteurs.

---

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| Taille de workgroup | Threads par workgroup (défini en WGSL : `@workgroup_size(256)`) | 256 est un défaut sûr pour la plupart des GPU. Doit être une puissance de 2. |
| Nombre de workgroups | Parallélisme total : `ceil(taille_données / taille_workgroup)` | Calculé automatiquement par les fonctions ix |
| Drapeaux d'usage des buffers | `STORAGE`, `COPY_SRC`, `MAP_READ`, etc. | Gérés par `create_buffer_init`, `create_output_buffer`, `create_readback_buffer` |
| Préférence d'alimentation | `HighPerformance` vs `LowPower` | ix utilise `HighPerformance` par défaut (préfère le GPU discret) |

---

## Pièges courants

1. **L'initialisation du GPU est lente.** `GpuContext::new()` prend 10 à 100 ms pour énumérer les adaptateurs, demander un device et compiler les shaders. Créez-le une fois et réutilisez-le pour tous les calculs.

2. **Le surcoût de transfert est réel.** Uploader des données vers la VRAM et lire les résultats prend un temps proportionnel à la taille des données. Pour de petites entrées (quelques centaines de floats), le transfert seul peut dépasser le temps de calcul CPU.

3. **Perte de précision en f32.** Si votre algorithme est numériquement sensible (ex. calculer de minuscules différences entre de grands nombres), le passage de f64 à f32 peut introduire une erreur significative. Les fonctions de repli CPU utilisent aussi f32 (pour la compatibilité API), donc comparez avec les routines f64 d'`ix-math` comme référence.

4. **Tous les GPU ne se valent pas.** Les GPU intégrés (Intel UHD, AMD APU) ont un débit de calcul bien inférieur aux GPU discrets (NVIDIA RTX, AMD RX). Testez sur votre matériel cible.

5. **La compilation des shaders se fait à l'exécution.** Les shaders WGSL sont compilés lors du premier appel à une fonction GPU. Les appels suivants réutilisent le pipeline compilé. Le premier appel est donc plus lent que prévu.

---

## Pour aller plus loin

- **[Recherche de similarité](./recherche-de-similarite.md) :** Similarité cosinus accélérée GPU, produit scalaire et recherche vectorielle top-k par batch.
- **[Multiplication matricielle](./multiplication-matricielle-gpu.md) :** Multiplication matricielle GPU pour l'inférence par batch et les passes forward de réseaux de neurones.
- Spécification WGSL : [https://www.w3.org/TR/WGSL/](https://www.w3.org/TR/WGSL/)
- Documentation WGPU : [https://docs.rs/wgpu](https://docs.rs/wgpu)
- Pour comprendre l'architecture GPU : « A Trip Through the Graphics Pipeline » offre une excellente intuition sur la façon dont les GPU traitent des milliers de threads simultanément.
