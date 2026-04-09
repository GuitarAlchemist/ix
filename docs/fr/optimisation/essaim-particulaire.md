# Optimisation par essaim particulaire

> Une nuée d'oiseaux en quête de nourriture. Aucun oiseau ne sait où elle se trouve, mais chacun se souvient du meilleur endroit qu'il a trouvé et peut voir où se situe le meilleur endroit du groupe. Ensemble, ils convergent vers le champ le plus riche.

**Prérequis :** [Probabilités et statistiques](../foundations/probability-and-statistics.md), [Vecteurs et matrices](../foundations/vectors-and-matrices.md)

---

## Le problème

Vous entraînez un modèle de gradient boosting pour détecter les transactions frauduleuses. Le modèle possède des hyperparamètres qui ne peuvent pas être appris par descente de gradient : le nombre d'arbres, la profondeur maximale de chaque arbre, le taux d'apprentissage et la force de régularisation. Vous devez trouver la combinaison qui minimise l'erreur de validation croisée sur votre jeu de données de test.

La recherche par grille est exhaustive — si chacun des 4 hyperparamètres a 10 valeurs candidates, cela fait 10 000 entraînements de modèle. La recherche aléatoire est meilleure mais sans direction — elle n'apprend pas des essais précédents. Vous voulez un algorithme qui explore l'espace des hyperparamètres intelligemment, en partageant l'information entre les essais pour que la recherche converge vers les bonnes régions.

L'optimisation par essaim particulaire (PSO) fait exactement cela. Elle lance un essaim de solutions candidates (les « particules »), chacune explorant l'espace de manière indépendante tout en communiquant ses découvertes. L'essaim s'auto-organise vers l'optimum.

---

## L'intuition

Imaginez 30 drones qui survolent une chaîne de montagnes à la recherche de la vallée la plus basse. Chaque drone :

1. **Se souvient de son meilleur personnel.** « L'altitude la plus basse que j'aie jamais mesurée était aux coordonnées (x, y). »
2. **Connaît le meilleur global.** Un canal radio diffuse l'altitude la plus basse trouvée par n'importe quel drone.
3. **Possède une inertie.** Il ne se téléporte pas ; il vole avec une vitesse, conservant sa direction précédente.

À chaque pas de temps, chaque drone ajuste sa trajectoire en fonction de trois forces :

- **Inertie :** Continuer dans la même direction. Cela empêche les zigzags erratiques.
- **Attraction cognitive (meilleur personnel) :** Se diriger vers le meilleur emplacement que vous avez personnellement trouvé. « J'ai trouvé une bonne vallée là-bas — laissez-moi y retourner pour explorer les alentours. »
- **Attraction sociale (meilleur global) :** Se diriger vers le meilleur emplacement trouvé par quiconque. « L'essaim a trouvé quelque chose d'encore mieux — allons dans cette direction. »

L'équilibre entre ces trois forces détermine le comportement de l'essaim :
- Forte inertie et attractions faibles : l'essaim se disperse et explore largement.
- Faible inertie et forte attraction sociale : l'essaim se replie rapidement sur le meilleur global (risque de convergence prématurée).
- Équilibré : l'essaim explore efficacement et converge progressivement.

Contrairement à la descente de gradient, PSO ne calcule jamais de gradient. Il évalue seulement la fonction objectif (« cette position est-elle bonne ? ») et utilise la mémoire collective de l'essaim pour décider où chercher ensuite.

---

## Comment ça fonctionne

### Mise à jour de la vitesse

```
v_i(t+1) = w * v_i(t)
          + c1 * r1 * (pbest_i - x_i(t))
          + c2 * r2 * (gbest   - x_i(t))
```

**En clair :** La nouvelle vitesse de chaque particule est une somme pondérée de trois composantes :

- `w * v_i(t)` : **Inertie.** La particule conserve une partie de sa vitesse actuelle. C'est comme un drone qui maintient son cap. Un `w` plus élevé signifie que la particule est plus difficile à rediriger.
- `c1 * r1 * (pbest_i - x_i(t))` : **Composante cognitive.** La particule est attirée vers sa propre meilleure position. `c1` contrôle la force de cette attraction. `r1` est un nombre aléatoire entre 0 et 1 qui ajoute de la variation stochastique — sans lui, toutes les particules suivraient des trajectoires déterministes et pourraient toutes converger vers le même minimum local.
- `c2 * r2 * (gbest - x_i(t))` : **Composante sociale.** La particule est attirée vers la meilleure position trouvée par l'ensemble de l'essaim. `c2` contrôle cette attraction. `r2` est un autre nombre aléatoire.

### Mise à jour de la position

```
x_i(t+1) = x_i(t) + v_i(t+1)
```

**En clair :** On déplace la particule en ajoutant la vitesse. On la clamp ensuite aux bornes de recherche pour que les particules ne sortent pas de la région autorisée.

### Mise à jour des meilleurs personnel et global

Après le déplacement, chaque particule évalue la fonction objectif à sa nouvelle position. Si la nouvelle position est meilleure que son meilleur personnel, on met à jour `pbest_i`. Si c'est aussi mieux que le meilleur global, on met à jour `gbest`.

**En clair :** Chaque particule tient un registre du meilleur endroit qu'elle a visité. L'essaim garde aussi la trace du meilleur endroit que n'importe quelle particule a jamais trouvé. Ces registres sont ce qui pilote la convergence — ils agissent comme des « balises » attirant l'essaim vers les régions prometteuses.

---

## En Rust

### Définir la recherche d'hyperparamètres

```rust
use ix_optimize::pso::ParticleSwarm;
use ix_optimize::traits::{ClosureObjective, ObjectiveFunction};
use ndarray::Array1;

// Simulate cross-validation error as a function of 4 hyperparameters:
//   x[0] = learning rate     (optimal: 0.1)
//   x[1] = max depth         (optimal: 6.0)
//   x[2] = num trees         (optimal: 200.0)
//   x[3] = regularization    (optimal: 1.5)
//
// In production, each evaluation would train a model and compute CV error.
// Here we use a synthetic function with multiple local minima.
let cv_error = ClosureObjective {
    f: |x: &Array1<f64>| {
        // Shifted Ackley-like surface: global minimum at (0.1, 6.0, 200.0, 1.5)
        let targets = [0.1, 6.0, 200.0, 1.5];
        let shifted: Vec<f64> = x.iter().zip(&targets).map(|(xi, ti)| xi - ti).collect();
        let sum_sq: f64 = shifted.iter().map(|s| s * s).sum();
        let sum_cos: f64 = shifted.iter().map(|s| (2.0 * std::f64::consts::PI * s).cos()).sum();
        let n = shifted.len() as f64;
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp()
            - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E
    },
    dimensions: 4,
};
```

### Exécuter PSO

```rust
let pso = ParticleSwarm::new()
    .with_particles(40)                // 40 candidate solutions exploring in parallel
    .with_max_iterations(500)          // 500 generations
    .with_bounds(-10.0, 300.0)         // search space (covers all hyperparameter ranges)
    .with_seed(42);                    // reproducible results

let result = pso.minimize(&cv_error);

println!("Best hyperparameters: {:?}", result.best_params);
println!("CV error: {:.6}", result.best_value);
println!("Iterations: {}", result.iterations);
println!("Converged: {}", result.converged);
// Expected: best_params close to [0.1, 6.0, 200.0, 1.5]
```

### Régler l'essaim

Les paramètres par défaut (`inertia=0.7`, `cognitive=1.5`, `social=1.5`) fonctionnent bien pour la plupart des problèmes. Pour un contrôle plus fin, réglez-les directement :

```rust
let mut pso = ParticleSwarm::new()
    .with_particles(60)
    .with_max_iterations(1000)
    .with_bounds(-10.0, 300.0)
    .with_seed(123);

// Access fields directly to tune behavior:
pso.inertia = 0.5;    // Less momentum -> faster convergence, less exploration
pso.cognitive = 2.0;   // Stronger personal memory -> more individual exploration
pso.social = 1.0;      // Weaker social pull -> less tendency to cluster prematurely
```

### Comprendre la valeur de retour

`minimize` renvoie un `OptimizeResult` :

| Champ         | Type          | Signification                                          |
|---------------|---------------|--------------------------------------------------------|
| `best_params` | `Array1<f64>` | Le meilleur vecteur de paramètres trouvé par une particule |
| `best_value`  | `f64`         | La valeur de la fonction objectif à `best_params`       |
| `iterations`  | `usize`       | Le nombre de générations complétées                     |
| `converged`   | `bool`        | `true` si `best_value` est passé sous `1e-12`          |

---

## Quand l'utiliser

| Situation | Utiliser PSO ? |
|-----------|----------------|
| Réglage d'hyperparamètres (pas de gradient disponible) | Oui — PSO est conçu pour l'optimisation boîte noire |
| Paysage multimodal (beaucoup de minima locaux) | Oui — l'essaim explore plusieurs régions simultanément |
| Dimensionnalité modérée (2 à 50 paramètres) | Oui — le terrain de prédilection de PSO |
| Vous pouvez évaluer la fonction objectif mais pas la dériver | Oui — PSO n'appelle que `evaluate`, jamais `gradient` |
| Problème lisse et convexe (régression linéaire) | Non — la [descente de gradient](descente-de-gradient.md) est plus rapide et exacte |
| Très haute dimensionnalité (>100 paramètres) | Avec prudence — PSO nécessite exponentiellement plus de particules quand la dimension augmente |
| Vous avez besoin d'une réponse unique et déterministe | Non — PSO est stochastique ; utilisez-le pour l'exploration, puis affinez avec des méthodes à gradient |
| Budget très serré (<100 évaluations) | Non — PSO a besoin de suffisamment d'itérations pour que l'essaim communique et converge |

---

## Paramètres clés

### Nombre de particules (`with_particles`)

- Plus de particules signifie une meilleure couverture de l'espace de recherche mais plus d'évaluations par itération.
- Règle empirique : 20 à 50 particules pour les problèmes de moins de 10 dimensions. 50 à 100 pour les dimensions supérieures.
- Défaut : `30`.

### Nombre maximal d'itérations (`with_max_iterations`)

- Le nombre de générations de l'essaim. Chaque génération évalue chaque particule une fois.
- Total des évaluations = `nombre_particules * max_iterations`. Budgétisez en conséquence.
- Défaut : `1000`.

### Bornes (`with_bounds(lo, hi)`)

- La région de recherche pour toutes les dimensions. Les particules sont initialisées uniformément dans les bornes et ramenées dans les bornes après chaque déplacement.
- Fixez les bornes en fonction de la connaissance du domaine. Trop serrées et vous risquez de rater l'optimum. Trop larges et les particules gaspillent du temps dans des régions sans intérêt.

### Poids d'inertie (`inertia`, défaut `0.7`)

C'est le paramètre de réglage PSO le plus important.

- **Forte inertie (0.9+) :** Les particules gardent plus d'élan et explorent largement. Bien pour la phase initiale de recherche.
- **Faible inertie (0.4-0.5) :** Les particules réagissent rapidement aux meilleurs personnel et global. Bien pour la convergence.
- **Stratégie adaptative (non intégrée) :** Commencer haut et diminuer linéairement vers le bas au cours de l'exécution. Vous pouvez implémenter cela en exécutant PSO par étapes avec une inertie décroissante.

### Poids cognitif (`cognitive` / c1, défaut `1.5`)

- Contrôle à quel point chaque particule est attirée vers son propre meilleur personnel.
- Un c1 plus élevé signifie plus d'exploration individuelle. Chaque particule affine indépendamment sa propre meilleure région.
- Si c1 est beaucoup plus grand que c2, les particules agissent presque indépendamment (moins de comportement d'essaim).

### Poids social (`social` / c2, défaut `1.5`)

- Contrôle à quel point chaque particule est attirée vers le meilleur global.
- Un c2 plus élevé signifie une convergence plus rapide — tout l'essaim se précipite vers le meilleur point connu.
- Si c2 est beaucoup plus grand que c1, l'essaim peut converger prématurément vers un minimum local car toutes les particules se regroupent autour d'un même point.

### L'équilibre c1/c2

| c1 vs c2 | Comportement |
|-----------|--------------|
| c1 = c2 = 1.5 | Exploration et convergence équilibrées (valeur par défaut recommandée) |
| c1 > c2 | Plus d'exploration individuelle, convergence plus lente, meilleur pour les problèmes multimodaux |
| c1 < c2 | Convergence rapide, risque de regroupement prématuré, bien pour les problèmes unimodaux |
| c1 + c2 > 4.0 | Les particules risquent d'osciller violemment ; réduisez les deux ou augmentez l'inertie |

### Graine aléatoire (`with_seed`)

- PSO est stochastique (initialisation aléatoire, r1/r2 aléatoires à chaque étape). Des graines différentes donnent des trajectoires différentes.
- Pour les applications critiques, lancez PSO avec 5 à 10 graines différentes et gardez le meilleur résultat.

---

## Pièges courants

**Convergence prématurée.** L'essaim se replie sur un minimum local parce que l'attraction sociale est trop forte. Toutes les particules se regroupent et cessent d'explorer. Solution : augmenter l'`inertia`, augmenter `c1` par rapport à `c2`, ou ajouter plus de particules.

**Trop de particules, trop peu d'itérations.** Si vous avez 200 particules mais seulement 50 itérations, chaque particule bouge à peine. L'essaim ne converge jamais. PSO a besoin de suffisamment d'itérations pour que l'information se propage. Solution : équilibrer le budget (particules x itérations = total d'évaluations).

**Bornes trop serrées.** L'optimum se trouve en dehors de vos bornes spécifiées, donc aucune particule ne peut l'atteindre. Ajoutez toujours une marge à vos bornes. Si les particules butent sans cesse contre la frontière (leur meilleure position est sur le bord), élargissez les bornes.

**Bornes trop larges.** Avec des bornes de -1 000 à 1 000 et seulement 30 particules, les positions initiales aléatoires sont trop dispersées. Les particules interagissent à peine car elles sont trop éloignées. Solution : resserrer les bornes en fonction de la connaissance du domaine, ou augmenter le nombre de particules.

**Ignorer le coût de l'évaluation.** Chaque itération PSO appelle `evaluate` une fois par particule. Si votre fonction objectif est coûteuse (par exemple entraîner un modèle ML complet), 40 particules x 500 itérations = 20 000 entraînements de modèle. Budgétisez soigneusement. Pour les évaluations coûteuses, envisagez l'optimisation bayésienne ou utilisez PSO avec très peu de particules (10-15) et plus d'itérations.

**Utiliser PSO pour des problèmes lisses et convexes.** Sur une simple quadratique, la descente de gradient trouvera le minimum exact en quelques centaines d'itérations. PSO l'approximera après des milliers d'évaluations. Utilisez le bon outil pour le bon problème.

---

## Pour aller plus loin

- **Voir en action :** [`examples/optimization/pso_rosenbrock.rs`](../../examples/optimization/pso_rosenbrock.rs) minimise la fonction de Rosenbrock en 10 dimensions avec PSO — un benchmark classique avec une vallée étroite et incurvée.
- **Alternative basée sur le gradient :** Quand les gradients sont disponibles, la [descente de gradient (SGD, Momentum, Adam)](descente-de-gradient.md) sera plus rapide et plus précise.
- **Autre méthode sans gradient :** Le [recuit simulé](recuit-simule.md) utilise un seul agent avec un caractère aléatoire contrôlé par la température. Il peut être plus efficace pour les problèmes de basse dimension mais n'a pas l'exploration parallèle de PSO.
- **Alternative évolutionnaire :** Les [algorithmes génétiques](../evolutionnaire/algorithmes-genetiques.md) maintiennent une population et utilisent le croisement et la mutation. Ils sont plus flexibles pour les problèmes combinatoires (discrets), tandis que PSO gère naturellement les espaces continus.
- **Approches hybrides :** Une stratégie puissante consiste à exécuter PSO d'abord pour trouver une bonne région, puis affiner avec Adam. PSO fournit l'exploration globale ; Adam apporte la précision locale.
- **Métriques de distance :** PSO utilise implicitement la distance euclidienne (les particules se déplacent dans un espace euclidien). Pour des espaces de paramètres non euclidiens, voir [Distance et similarité](../foundations/distance-and-similarity.md).
