# Cache et mémoïsation

## Le problème

Vous avez un pipeline de données en cinq étapes qui prend 30 secondes de bout en bout. Vous modifiez la logique de normalisation à l'étape 4 et relancez. Les étapes 1 à 3 produisent exactement la même sortie qu'avant — mais le pipeline les recalcule quand même, gaspillant 25 secondes. Vous avez besoin d'un moyen de sauter les noeuds dont les entrées n'ont pas changé, en ne ré-exécutant que les parties du DAG réellement affectées.

C'est le problème du **recalcul incrémental**. Les outils de build (Make, Bazel), les environnements de notebooks (cache de cellules Jupyter) et les frameworks ETL (dbt) en résolvent tous des versions. L'exécuteur de pipeline d'ix le résout au niveau du noeud avec une interface de cache enfichable.

---

## L'intuition

Imaginez une cuisine où chaque cuisinier note sa recette et ses ingrédients sur une fiche. Avant de commencer à cuisiner, il vérifie une étagère de plats préparés. Si quelqu'un a déjà préparé exactement ce plat avec exactement ces ingrédients, il le prend sur l'étagère au lieu de cuisiner à nouveau. Sinon, il cuisine et met une copie sur l'étagère pour la prochaine fois.

L'« étagère » est le cache. La « recette + ingrédients » est la clé de cache. Le drapeau `cacheable` sur chaque noeud contrôle si ce cuisinier prend la peine de vérifier l'étagère — certains plats (comme le « plat du jour ») doivent toujours être préparés frais.

---

## Comment ça fonctionne

### Le trait PipelineCache

```rust
pub trait PipelineCache: Send + Sync {
    /// Tenter de récupérer un résultat en cache pour un noeud.
    fn get(&self, cache_key: &str) -> Option<Value>;

    /// Stocker un résultat dans le cache.
    fn set(&self, cache_key: &str, value: &Value);
}
```

**En clair :** tout type capable de chercher et stocker des valeurs JSON par clé chaîne peut servir de cache de pipeline. Cela peut être un HashMap en mémoire, une connexion Redis, le store embarqué `ix-cache` ou un cache sur fichier.

### Génération de la clé de cache

L'exécuteur génère une clé de cache déterministe pour chaque noeud à partir de :

1. L'**ID du noeud** (ex. `"normalize"`).
2. Les **valeurs d'entrée** (triées par nom d'entrée, puis hachées avec FNV-1a).

```
clé_cache = "pipeline:{node_id}:{hash(entrées_triées)}"
```

**En clair :** si le même noeud reçoit les mêmes entrées, il produit la même clé de cache, et un résultat précédent peut être réutilisé. Si une entrée change (parce qu'un noeud en amont a produit une sortie différente), le hash change et le noeud se ré-exécute.

### Mise en cache par noeud

Chaque noeud possède un drapeau `cacheable` (défaut : `true`). Utilisez la méthode `.no_cache()` du constructeur pour désactiver le cache pour les noeuds qui :

- Ont des **effets de bord** (écrire dans un fichier, envoyer une requête HTTP).
- Sont **non-déterministes** (lire l'heure courante, échantillonner des nombres aléatoires).
- Doivent toujours refléter des **données fraîches** (lire depuis une base de données en direct).

```rust
.node("fetch_live_data", |b| b
    .compute(|_| { /* interroger la base de données */ })
    .no_cache()  // toujours ré-exécuter
)
```

### L'implémentation NoCache

Pour les pipelines qui n'ont pas besoin de cache, passez `&NoCache` :

```rust
pub struct NoCache;

impl PipelineCache for NoCache {
    fn get(&self, _key: &str) -> Option<Value> { None }
    fn set(&self, _key: &str, _value: &Value) {}
}
```

C'est un moyen sans coût de désactiver complètement le cache.

---

## En Rust

### Observer les hits de cache en action

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, PipelineCache, NoCache};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;

// Un cache en mémoire simple
struct MemoryCache {
    store: Mutex<HashMap<String, Value>>,
}

impl MemoryCache {
    fn new() -> Self {
        Self { store: Mutex::new(HashMap::new()) }
    }
}

impl PipelineCache for MemoryCache {
    fn get(&self, key: &str) -> Option<Value> {
        self.store.lock().unwrap().get(key).cloned()
    }
    fn set(&self, key: &str, value: &Value) {
        self.store.lock().unwrap().insert(key.to_string(), value.clone());
    }
}

let pipeline = PipelineBuilder::new()
    .source("data", || Ok(json!({"values": [1, 2, 3]})))
    .node("expensive_stats", |b| b
        .input("x", "data")
        .cost(10.0)  // coût élevé -- on veut mettre en cache
        .compute(|inputs| {
            println!("  Calcul de expensive_stats...");
            let vals = inputs["x"]["values"].as_array().unwrap();
            let sum: f64 = vals.iter().map(|v| v.as_f64().unwrap()).sum();
            Ok(json!({"sum": sum}))
        })
    )
    .build()
    .unwrap();

let cache = MemoryCache::new();
let inputs = HashMap::new();

// Première exécution : tout est calculé
println!("Exécution 1 :");
let r1 = execute(&pipeline, &inputs, &cache).unwrap();
println!("  Hits de cache : {}", r1.cache_hits);  // 0

// Deuxième exécution : mêmes entrées, le cache intervient
println!("Exécution 2 :");
let r2 = execute(&pipeline, &inputs, &cache).unwrap();
println!("  Hits de cache : {}", r2.cache_hits);  // 1 (expensive_stats était en cache)

// Les deux exécutions produisent la même sortie
assert_eq!(r1.output("expensive_stats"), r2.output("expensive_stats"));
```

### Mélanger noeuds cachables et non cachables

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, PipelineCache};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;

struct MemoryCache {
    store: Mutex<HashMap<String, Value>>,
}
impl MemoryCache {
    fn new() -> Self { Self { store: Mutex::new(HashMap::new()) } }
}
impl PipelineCache for MemoryCache {
    fn get(&self, key: &str) -> Option<Value> {
        self.store.lock().unwrap().get(key).cloned()
    }
    fn set(&self, key: &str, value: &Value) {
        self.store.lock().unwrap().insert(key.to_string(), value.clone());
    }
}

let pipeline = PipelineBuilder::new()
    .source("config", || Ok(json!({"threshold": 0.5})))
    .node("load_data", |b| b
        .compute(|_| {
            // Simule une lecture depuis une source en direct
            Ok(json!({"records": 42}))
        })
        .no_cache()  // toujours chercher les données fraîches
    )
    .node("process", |b| b
        .input("config", "config")
        .input("data", "load_data")
        .compute(|inputs| {
            let threshold = inputs["config"]["threshold"].as_f64().unwrap();
            let records = inputs["data"]["records"].as_f64().unwrap();
            Ok(json!(records * threshold))
        })
        // cachable par défaut -- mais se ré-exécutera si load_data change
    )
    .build()
    .unwrap();

let cache = MemoryCache::new();
let r = execute(&pipeline, &HashMap::new(), &cache).unwrap();
println!("Sortie process : {}", r.output("process").unwrap());
```

### Inspecter les détails d'exécution du PipelineResult

```rust
use ix_pipeline::executor::PipelineResult;

fn print_execution_report(result: &PipelineResult) {
    println!("Durée totale : {:?}", result.total_duration);
    println!("Hits de cache : {}", result.cache_hits);

    println!("\nOrdre d'exécution :");
    for (level, nodes) in result.execution_order.iter().enumerate() {
        println!("  Niveau {} : {:?}", level, nodes);
    }

    println!("\nDétails par noeud :");
    for (id, node_result) in &result.node_results {
        println!("  {} -- {:?} (hit cache : {})",
            id,
            node_result.duration,
            node_result.cache_hit,
        );
    }
}
```

Les champs du `PipelineResult` pertinents pour le cache :

| Champ | Type | Signification |
|-------|------|---------|
| `cache_hits` | `usize` | Nombre total de noeuds ayant renvoyé un résultat en cache |
| `execution_order` | `Vec<Vec<NodeId>>` | Quels noeuds ont tourné à chaque niveau (les hits de cache y apparaissent aussi) |
| `node_results[id].cache_hit` | `bool` | Si ce noeud spécifique a été servi depuis le cache |
| `node_results[id].duration` | `Duration` | `Duration::ZERO` pour les hits de cache ; durée réelle pour les noeuds calculés |

---

## Quand l'utiliser

| Situation | Stratégie de cache |
|-----------|-----------------|
| Le pipeline tourne de façon répétée avec les mêmes entrées | Utiliser un cache persistant (fichier ou `ix-cache`) |
| Le pipeline tourne une seule fois | Utiliser `NoCache` pour éviter le surcoût |
| Certains noeuds sont non-déterministes | Les marquer `.no_cache()` |
| Débogage : voir tous les calculs | Utiliser `NoCache` temporairement |
| Pipeline distribué entre machines | Utiliser un cache partagé (compatible Redis, ou `ix-cache` avec TCP) |

---

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `cacheable` (par noeud) | Si l'exécuteur vérifie/stocke le cache pour ce noeud | Défaut : `true`. Désactiver avec `.no_cache()` pour les effets de bord ou le non-déterminisme |
| Clé de cache | `"pipeline:{node_id}:{hash_entrées}"` | Déterministe à partir de l'ID du noeud + valeurs d'entrée triées. Change quand une entrée change |
| Implémentation `PipelineCache` | Où les valeurs en cache sont stockées | HashMap en mémoire pour les tests ; `ix-cache` pour la production |
| `initial_inputs` | Valeurs externes fournies au pipeline | Les changer modifie les clés de cache de tous les noeuds qui les lisent |

---

## Pièges courants

1. **Les noeuds non-déterministes empoisonnent les caches en aval.** Si un noeud non cachable produit une sortie différente à chaque exécution, tous les noeuds en aval manqueront le cache aussi (car leurs entrées ont changé). C'est le comportement correct, mais cela signifie que le cache n'aide que les branches purement déterministes.

2. **Le cache n'expire pas automatiquement.** L'exécuteur de pipeline n'implémente ni TTL ni éviction. Si vous utilisez un simple cache HashMap, il croît sans limite. Connectez-vous à `ix-cache` (qui supporte TTL et éviction LRU) pour les charges de production.

3. **Les grandes valeurs JSON sont coûteuses à mettre en cache.** Le cache stocke des objets `serde_json::Value`. Si un noeud produit un tableau JSON de plusieurs mégaoctets, le cloner dans le cache et le récupérer ajoute du surcoût. Pour les données intermédiaires volumineuses, envisagez de stocker une référence (chemin de fichier, ID de buffer) plutôt que les données elles-mêmes.

4. **Les collisions de clés de cache sont théoriquement possibles.** Le hash FNV-1a utilisé pour les clés de cache est rapide mais pas cryptographique. Deux ensembles d'entrées différents pourraient (extrêmement rarement) produire le même hash, causant des hits de cache incorrects. Pour les pipelines critiques, implémentez un `PipelineCache` qui stocke et vérifie aussi les entrées complètes à côté du résultat.

5. **Pas d'invalidation automatique du cache lors des changements de code.** Si vous changez la fonction de calcul d'un noeud mais gardez les mêmes entrées, le cache renverra des résultats obsolètes de l'ancienne fonction. Videz le cache quand vous changez la logique du pipeline, ou incluez un numéro de version dans vos ID de noeuds (ex. `"normalize_v2"`).

---

## Pour aller plus loin

- **Connectez-vous à `ix-cache`** pour un cache de grade production avec TTL, éviction LRU, notifications pub/sub et un serveur RESP compatible Redis. Implémentez `PipelineCache` pour appeler l'API du crate cache.
- Le **cache adressé par contenu** hache le *code* du noeud (ou un tag de version) à côté de ses entrées, invalidant automatiquement quand la logique change. C'est ainsi que Bazel et Nix réalisent des builds reproductibles.
- **Ré-exécution partielle.** Utilisez `dag.has_path(noeud_modifié, noeud_cible)` pour déterminer quels noeuds en aval doivent être ré-exécutés après un changement, sans ré-exécuter l'ensemble du pipeline.
- L'**[exécution DAG](./execution-dag.md)** couvre le constructeur de pipeline, l'exécuteur, les types d'erreurs et l'ordonnancement par niveaux parallèles en détail.
