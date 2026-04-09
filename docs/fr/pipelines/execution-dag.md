# Exécution de pipelines DAG

## Le problème

Vous construisez un workflow d'entraînement de modèle en cinq étapes : charger les données, calculer les statistiques, normaliser les caractéristiques, entraîner le modèle, évaluer. Certaines étapes dépendent d'autres (impossible de normaliser sans les statistiques), mais certaines sont indépendantes (les statistiques et une branche séparée d'ingénierie de caractéristiques peuvent s'exécuter en parallèle). Vous avez besoin d'un système qui :

1. Comprend les dépendances et refuse de créer des dépendances circulaires.
2. Détermine automatiquement quelles étapes peuvent s'exécuter en parallèle.
3. Exécute tout dans le bon ordre.
4. Passe les sorties d'une étape aux entrées de la suivante.

C'est le problème du pipeline ETL, du pipeline CI/CD, de l'orchestration de workflows basée sur un DAG. Le crate `ix-pipeline` d'ix le résout avec un DAG typé, vérifié contre les cycles, et un exécuteur parallèle par niveaux.

---

## L'intuition

Pensez à une recette avec plusieurs cuisiniers dans une cuisine :

- **Cuisinier A** coupe les légumes (pas de dépendance).
- **Cuisinier B** fait bouillir l'eau (pas de dépendance).
- **Cuisinier C** prépare la sauce (besoin des légumes coupés du cuisinier A).
- **Cuisinier D** cuit les pâtes (besoin de l'eau bouillante du cuisinier B).
- **Cuisinier E** dresse l'assiette (besoin de la sauce du cuisinier C *et* des pâtes du cuisinier D).

Les cuisiniers A et B peuvent travailler simultanément (niveau 0). Les cuisiniers C et D peuvent commencer dès que leurs entrées sont prêtes (niveau 1). Le cuisinier E attend C et D (niveau 2). C'est un DAG — un graphe orienté acyclique. La partie « acyclique » signifie que personne n'attend après soi-même, ce qui serait un deadlock.

Le constructeur de pipeline vous permet de déclarer les noeuds (cuisiniers) et les arêtes (qui a besoin de quoi), puis l'exécuteur détermine le parallélisme et lance tout.

---

## Comment ça fonctionne

### La structure de données DAG

Un `Dag<N>` est un graphe orienté qui **rejette les cycles au moment de l'insertion**. Chaque appel à `add_edge(from, to)` lance un BFS de `to` vers `from` — si un chemin existe, l'arête créerait un cycle et est rejetée.

| Opération | Complexité | Ce qu'elle fait |
|-----------|:----------:|--------------|
| `add_node(id, data)` | O(1) | Enregistrer un noeud |
| `add_edge(from, to)` | O(V + E) | Ajouter une arête de dépendance (avec vérification de cycle) |
| `topological_sort()` | O(V + E) | Algorithme de Kahn : ordonner les noeuds pour que les dépendances passent en premier |
| `parallel_levels()` | O(V + E) | Grouper les noeuds en niveaux parallélisables |
| `has_path(a, b)` | O(V + E) | Vérification d'accessibilité par BFS |
| `critical_path(cost_fn)` | O(V + E) | Plus long chemin pondéré à travers le DAG |

### Types d'erreurs

L'enum `DagError` détecte les problèmes structurels tôt :

| Variante | Quand elle se déclenche |
|---------|--------------|
| `DuplicateNode(id)` | Appel de `add_node` avec un ID qui existe déjà |
| `NodeNotFound(id)` | Référence à un noeud dans `add_edge` qui n'a jamais été ajouté |
| `CycleDetected(from, to)` | L'arête proposée créerait un cycle |
| `SelfLoop(id)` | Une arête d'un noeud vers lui-même |

### Le PipelineBuilder

L'API du constructeur permet de déclarer des noeuds avec des fonctions de calcul et de les connecter :

```
PipelineBuilder::new()
    .node(id, |builder| builder
        .compute(|inputs| -> Result<Value>)
        .input(name, source_node)
        .cost(estimated_time)
        .no_cache()
    )
    .edge(from, to)          // arête explicite
    .build()                  // renvoie Dag<PipelineNode>
```

Les arêtes déclarées via `.input(name, source_node)` sont **auto-détectées** : si l'entrée d'un noeud référence un autre noeud par son ID, le constructeur crée automatiquement l'arête. Vous n'avez pas besoin d'appeler `.edge()` séparément sauf pour des contraintes d'ordonnancement supplémentaires.

### L'exécuteur

`execute(dag, initial_inputs, cache)` exécute le pipeline :

1. Calculer `parallel_levels()` pour grouper les noeuds par profondeur de dépendance.
2. Pour chaque niveau, exécuter tous les noeuds de ce niveau. (Les niveaux à un seul noeud s'exécutent directement ; les niveaux à plusieurs noeuds s'exécutent en parallèle.)
3. La fonction de calcul de chaque noeud reçoit une `HashMap<String, Value>` de ses entrées déclarées, résolues à partir des sorties des prédécesseurs ou des `initial_inputs`.
4. Les résultats sont collectés dans un `PipelineResult` avec les sorties par noeud, les durées, les hits de cache et l'ordre d'exécution.

---

## En Rust

> Exemple complet exécutable : [`examples/pipeline/dag_pipeline.rs`](../../examples/pipeline/dag_pipeline.rs)

### Pipeline ETL avec l'API constructeur

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, NoCache};
use serde_json::{json, Value};
use std::collections::HashMap;

let pipeline = PipelineBuilder::new()
    // Noeud source : produit les données brutes
    .source("raw_data", || {
        Ok(json!({"values": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    })
    // Calculer les statistiques (dépend de raw_data)
    .node("stats", |b| {
        b.input("data", "raw_data")
         .compute(|inputs| {
             let vals = inputs["data"]["values"].as_array().unwrap();
             let sum: f64 = vals.iter().map(|v| v.as_f64().unwrap()).sum();
             let mean = sum / vals.len() as f64;
             Ok(json!({"mean": mean, "count": vals.len()}))
         })
    })
    // Normaliser (dépend À LA FOIS de raw_data et stats)
    .node("normalize", |b| {
        b.input("data", "raw_data")
         .input("stats", "stats")
         .compute(|inputs| {
             let vals = inputs["data"]["values"].as_array().unwrap();
             let mean = inputs["stats"]["mean"].as_f64().unwrap();
             let normalized: Vec<f64> = vals.iter()
                 .map(|v| v.as_f64().unwrap() - mean)
                 .collect();
             Ok(json!(normalized))
         })
    })
    .build()
    .unwrap();

let result = execute(&pipeline, &HashMap::new(), &NoCache).unwrap();
println!("Normalisé : {}", result.output("normalize").unwrap());
println!("Exécuté en {} niveaux", result.execution_order.len());
```

### Motif en losange (branches parallèles)

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, NoCache};
use serde_json::{json, Value};
use std::collections::HashMap;

let pipeline = PipelineBuilder::new()
    .source("data", || Ok(json!(100.0)))
    .node("branch_a", |b| b
        .input("x", "data")
        .compute(|inputs| {
            let x = inputs["x"].as_f64().unwrap();
            Ok(json!(x + 50.0))  // 150
        })
    )
    .node("branch_b", |b| b
        .input("x", "data")
        .compute(|inputs| {
            let x = inputs["x"].as_f64().unwrap();
            Ok(json!(x * 0.5))  // 50
        })
    )
    .node("merge", |b| b
        .input("a", "branch_a")
        .input("b", "branch_b")
        .compute(|inputs| {
            let a = inputs["a"].as_f64().unwrap();
            let b = inputs["b"].as_f64().unwrap();
            Ok(json!(a - b))  // 100
        })
    )
    .build()
    .unwrap();

// Niveaux d'exécution :
//   Niveau 0 : [data]
//   Niveau 1 : [branch_a, branch_b]  <-- parallèle
//   Niveau 2 : [merge]
let result = execute(&pipeline, &HashMap::new(), &NoCache).unwrap();
assert_eq!(result.execution_order[1].len(), 2);  // branch_a et branch_b en parallèle
println!("Résultat : {}", result.output("merge").unwrap());  // 100
```

### Utilisation de l'API DAG brute

```rust
use ix_pipeline::dag::{Dag, DagError};

let mut dag = Dag::new();
dag.add_node("load",    "Charger CSV").unwrap();
dag.add_node("clean",   "Supprimer les nulls").unwrap();
dag.add_node("feature", "Ingénierie de caractéristiques").unwrap();
dag.add_node("train",   "Entraîner le modèle").unwrap();
dag.add_node("eval",    "Évaluer").unwrap();

dag.add_edge("load", "clean").unwrap();
dag.add_edge("clean", "feature").unwrap();
dag.add_edge("clean", "train").unwrap();   // train peut commencer après clean
dag.add_edge("feature", "train").unwrap(); // mais a aussi besoin des features
dag.add_edge("train", "eval").unwrap();

// Tri topologique
let sorted = dag.topological_sort();
println!("Ordre d'exécution : {:?}", sorted);

// Niveaux parallèles
let levels = dag.parallel_levels();
for (i, level) in levels.iter().enumerate() {
    let ids: Vec<&str> = level.iter().map(|s| s.as_str()).collect();
    println!("Niveau {} : {:?}", i, ids);
}

// Détection de cycle
let err = dag.add_edge("eval", "load");
assert!(matches!(err, Err(DagError::CycleDetected(_, _))));

// Requêtes de chemin
assert!(dag.has_path("load", "eval"));
assert!(!dag.has_path("eval", "load"));

// Chemin critique
let (path, cost) = dag.critical_path(|_, _| 1.0);
println!("Chemin critique : {:?} (coût : {})", path, cost);
```

### Entrées externes

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, NoCache};
use serde_json::Value;
use std::collections::HashMap;

let pipeline = PipelineBuilder::new()
    .node("greet", |b| b
        .input("name", "name")  // "name" est une entrée externe, pas un noeud
        .compute(|inputs| {
            let name = inputs["name"].as_str().unwrap_or("world");
            Ok(Value::from(format!("Bonjour, {} !", name)))
        })
        .no_cache()  // les noeuds non-déterministes ou à effets de bord doivent ignorer le cache
    )
    .build()
    .unwrap();

let mut inputs = HashMap::new();
inputs.insert("name".to_string(), Value::from("ix"));

let result = execute(&pipeline, &inputs, &NoCache).unwrap();
println!("{}", result.output("greet").unwrap());  // "Bonjour, ix !"
```

---

## Quand l'utiliser

| Situation | Pipeline DAG | Boucle séquentielle | Pool de threads |
|-----------|:-----------:|:---------------:|:-----------:|
| Tâches avec dépendances complexes | Optimal | Ordonnancement manuel | Sync manuelle |
| Parallélisme automatique des tâches indépendantes | Oui | Non | Manuel |
| Détection de cycles au moment de la construction | Oui | N/A | N/A |
| Flux de données entre tâches | Intégré (maps entrée/sortie) | Variables manuelles | Canaux manuels |
| Cache / mémoïsation | Intégré | Manuel | Manuel |
| Analyse du chemin critique | Intégré | Manuel | Manuel |

**Règle pratique :** utilisez un pipeline DAG quand vous avez plus de 3 étapes avec des relations de dépendance non triviales, surtout si certaines branches peuvent s'exécuter en parallèle.

---

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Recommandation |
|-----------|-----------------|----------|
| `compute` du noeud | La fonction exécutée pour cette étape | Prend `&HashMap<String, Value>`, renvoie `Result<Value, PipelineError>` |
| `input(name, source)` du noeud | D'où ce noeud obtient ses données | `source` est l'ID d'un autre noeud ou le nom d'une entrée externe |
| `input_field(name, source, field)` du noeud | Lire un champ JSON spécifique d'une source | Utilisez quand une source produit un objet et que vous ne voulez qu'une clé |
| `cost` du noeud | Temps d'exécution estimé (pour le chemin critique) | Défaut : 1,0. Augmenter pour les noeuds coûteux afin d'obtenir une analyse de chemin critique pertinente |
| `no_cache()` du noeud | Désactiver le cache pour ce noeud | Pour les noeuds non-déterministes ou à effets de bord |
| `initial_inputs` | Valeurs externes passées au pipeline | Disponibles pour tout noeud via `input(name, clé_externe)` |

---

## Pièges courants

1. **La détection de cycle s'exécute à chaque `add_edge`.** Pour les DAG avec des milliers de noeuds et d'arêtes, cette vérification BFS peut être coûteuse. Construisez les grands DAG dans l'ordre des dépendances (ajoutez les arêtes des sources vers les puits) pour garder la vérification d'accessibilité courte.

2. **Les auto-arêtes ne fonctionnent qu'avec le constructeur.** Si vous construisez un `Dag` manuellement et définissez les entrées `input_map` sans appeler `add_edge`, l'exécuteur échouera avec `MissingInput`. La méthode `build()` du constructeur appelle `auto_edges()` pour vous ; le DAG brut ne le fait pas.

3. **Des valeurs JSON partout.** Le pipeline passe les données sous forme de `serde_json::Value`. C'est flexible mais signifie que vous perdez la sécurité des types dans les fonctions de calcul. Parsez et validez les entrées tôt dans chaque closure de calcul.

4. **Le parallélisme est par niveaux, pas total.** L'exécuteur lance tous les noeuds d'un niveau de dépendance donné avant de passer au suivant. C'est plus simple qu'un ordonnancement asynchrone complet mais signifie qu'un noeud lent au niveau 1 bloque tout le niveau 2, même si certains noeuds du niveau 2 ne dépendent que d'un noeud rapide du niveau 1.

5. **`PipelineError::MissingInput` à l'exécution.** Si un noeud déclare `.input("x", "noeud_inexistant")` et que ce noeud n'est ni dans le pipeline ni dans `initial_inputs`, vous obtenez une erreur à l'exécution, pas à la construction. Validez la structure de votre pipeline avant l'exécution.

---

## Pour aller plus loin

- Le **[cache et la mémoïsation](./cache-et-memoisation.md)** couvrent le trait `PipelineCache`, la mise en cache par noeud et comment connecter `ix-cache` pour le recalcul incrémental.
- L'**analyse du chemin critique** (`dag.critical_path(cost_fn)`) identifie la chaîne goulot d'étranglement dans votre pipeline. Utilisez-la pour décider quels noeuds optimiser ou déplacer sur GPU.
- **`parallel_levels()`** renvoie le planning d'exécution. Vous pouvez l'utiliser pour la visualisation, les barres de progression ou un ordonnancement personnalisé sans exécuter l'exécuteur complet.
- Le `Dag` est générique sur les données de noeud (`Dag<N>`). Le pipeline utilise `Dag<PipelineNode>`, mais vous pouvez utiliser `Dag<String>`, `Dag<MyTask>` ou tout autre type pour des charges DAG non-pipeline (graphes de dépendances, systèmes de build, ordonnanceurs de tâches).
