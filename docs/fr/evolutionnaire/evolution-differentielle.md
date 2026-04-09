# Évolution différentielle

## Le problème

Vous calibrez un modèle hydrologique qui prédit le débit d'une rivière à partir des précipitations. Le modèle possède 8 paramètres (perméabilité du sol, coefficient de ruissellement, taux d'évaporation, etc.). Vous pouvez lancer la simulation et comparer le débit prédit au débit observé, mais la surface d'erreur est accidentée — de petites variations de paramètres provoquent parfois de grands sauts de précision. La descente de gradient échoue car la fonction n'est pas lisse. Vous avez besoin d'un optimiseur robuste, sans gradient, qui fonctionne de manière fiable avec un minimum de réglage.

Scénarios réels :
- **Calibration de paramètres :** Ajuster les paramètres de simulation aux données observées (modèles physiques, financiers, biologiques).
- **Réglage de contrôleur PID :** Trouver les gains proportionnel, intégral et dérivé d'un système de commande.
- **Conception de filtres optiques :** Optimiser les épaisseurs des couches d'un revêtement multicouche pour atteindre une transmittance cible.
- **Optimisation d'hyperparamètres de machine learning :** Régler le taux d'apprentissage, la régularisation, les choix d'architecture.
- **Optimisation de procédés chimiques :** Ajuster température, pression et débits pour maximiser le rendement.

## L'intuition

L'évolution différentielle (DE) ressemble à un groupe de randonneurs qui cherchent la vallée la plus basse d'une chaîne de montagnes sans carte.

Chaque randonneur se tient à un point aléatoire. Pour décider où aller ensuite, un randonneur observe trois *autres* randonneurs et calcule une direction :

1. Choisir trois autres randonneurs : A, B et C.
2. Calculer une direction : « Aller de A vers la position de B par rapport à C » — c'est-à-dire A + F * (B - C).
3. Essayer cette nouvelle position. Si elle est plus basse que la position actuelle, s'y déplacer. Sinon, rester sur place.

Cette méthode est remarquablement efficace car :
- La **taille du pas s'adapte automatiquement.** Quand la population est dispersée, (B - C) est grand, donc les pas sont grands (exploration). À mesure que la population converge, (B - C) se réduit, et les pas deviennent fins (exploitation).
- **Aucun gradient nécessaire.** Uniquement des comparaisons d'aptitude (le nouveau point est-il meilleur que l'ancien ?).
- **Peu de paramètres à régler.** Seulement F (facteur de mutation) et CR (probabilité de croisement), et DE est robuste aux valeurs exactes de ces paramètres.

## Comment ça fonctionne

DE maintient une population de N vecteurs (solutions candidates) dans D dimensions. La variante implémentée dans ix est **DE/rand/1/bin** (la plus courante).

### Pour chaque individu x_i de la population

**Étape 1 : Mutation** — Créer un vecteur mutant :

```
v = x_r1 + F * (x_r2 - x_r3)
```

Où r1, r2, r3 sont trois indices aléatoires distincts (tous différents de i), et F est le facteur de mutation.

**En clair :** On part d'un membre aléatoire de la population et on fait un pas dans la direction définie par la différence de deux autres membres. F contrôle la taille du pas.

**Étape 2 : Croisement** — Créer un vecteur d'essai en mélangeant le mutant avec l'original :

```
For each dimension j:
    if (random() < CR) or (j == j_rand):
        trial[j] = v[j]    (take from mutant)
    else:
        trial[j] = x_i[j]  (keep original)
```

Où CR est la probabilité de croisement et j_rand garantit qu'au moins une dimension provient du mutant.

**En clair :** On mélange aléatoirement les vecteurs original et mutant. CR contrôle quelle proportion du mutant passe. CR = 0.9 signifie que 90 % des dimensions viennent du mutant.

**Étape 3 : Sélection** — Garder le meilleur des deux :

```
if fitness(trial) <= fitness(x_i):
    x_i = trial
```

**En clair :** On ne se déplace que si la nouvelle position est au moins aussi bonne. Cette sélection gloutonne garantit que la population ne se dégrade jamais.

## En Rust

```rust
use ix_evolution::differential::DifferentialEvolution;
use ndarray::Array1;

// Minimize the Sphere function: f(x) = sum(x_i^2)
let result = DifferentialEvolution::new()
    .with_population_size(50)      // 50 candidate solutions
    .with_generations(1000)        // 1000 iterations
    .with_mutation_factor(0.8)     // F: step size scaling (0.5-1.0 typical)
    .with_crossover_prob(0.9)      // CR: how much of mutant to use
    .with_bounds(-5.0, 5.0)       // search range per dimension
    .with_seed(42)                 // reproducible
    .minimize(
        &|x: &Array1<f64>| x.mapv(|v| v * v).sum(),
        3,  // 3 dimensions
    );

println!("Best solution: {:.6}", result.best_genes);
println!("Best fitness:  {:.8}", result.best_fitness);
// DE typically finds fitness < 0.01 on the sphere function
```

### Problème plus difficile : la fonction de Rosenbrock

La fonction de Rosenbrock possède une vallée étroite et incurvée qui met à l'épreuve de nombreux optimiseurs :

```rust
use ix_evolution::differential::DifferentialEvolution;
use ndarray::Array1;

// Rosenbrock: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
// Global minimum at (1, 1, ..., 1) with f = 0
let rosenbrock = |x: &Array1<f64>| -> f64 {
    (0..x.len() - 1)
        .map(|i| {
            100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2)
        })
        .sum::<f64>()
};

let result = DifferentialEvolution::new()
    .with_population_size(60)
    .with_generations(2000)
    .with_mutation_factor(0.8)
    .with_crossover_prob(0.9)
    .with_bounds(-5.0, 5.0)
    .with_seed(42)
    .minimize(&rosenbrock, 5);  // 5 dimensions

println!("Best fitness: {:.6} (optimal is 0.0)", result.best_fitness);
println!("Best genes: {:.4}", result.best_genes);
// Expected: all genes near 1.0
```

### Suivre la convergence

```rust
// The fitness_history tracks the best fitness at each generation
let result = DifferentialEvolution::new()
    .with_population_size(30)
    .with_generations(500)
    .with_seed(42)
    .minimize(&|x: &Array1<f64>| x.mapv(|v| v * v).sum(), 3);

// Plot or analyze the convergence curve
let first_10: Vec<f64> = result.fitness_history[..10].to_vec();
let last_10: Vec<f64> = result.fitness_history[result.fitness_history.len()-10..].to_vec();
println!("First 10 gen fitness: {:?}", first_10);
println!("Last 10 gen fitness:  {:?}", last_10);
// Early: large improvements. Late: diminishing returns.
```

## Quand l'utiliser

| Méthode | Effort de réglage | Vitesse de convergence | Robustesse | Meilleur pour |
|---|---|---|---|---|
| **Évolution différentielle** | Faible (F et CR) | Rapide | Très élevée | Calibration de paramètres continus |
| **Algorithme génétique** | Moyen (mutation, croisement, sélection) | Modérée | Élevée | Problèmes combinatoires ou à types mixtes |
| **Essaim particulaire (PSO)** | Faible | Rapide en unimodal | Modérée | Paysages lisses, basse dimension |
| **Recuit simulé** | Moyen (programme de refroidissement) | Lente | Modérée | Solution unique, toute représentation |
| **CMA-ES** | Très faible | Très rapide | Très élevée | Continu en dimension petite à moyenne |

**Utilisez DE quand :**
- Le problème est une optimisation continue (paramètres à valeurs réelles).
- Vous avez besoin d'une méthode robuste qui fonctionne d'emblée.
- Le paysage d'aptitude est bruité, discontinu ou multimodal.
- La dimension est modérée (2 à 100 paramètres).

**Utilisez un AG plutôt quand :**
- L'espace de recherche n'est pas continu (permutations, choix discrets, structures arborescentes).
- Vous voulez utiliser des opérateurs de croisement personnalisés.
- Vous avez besoin d'une population diversifiée pour l'optimisation multi-objectifs.

**Utilisez les méthodes à gradient plutôt quand :**
- L'objectif est lisse et dérivable — la descente de gradient sera 100 à 1000 fois plus rapide.

## Paramètres clés

| Paramètre | Méthode | Défaut | Plage typique | Description |
|---|---|---|---|---|
| `population_size` | `with_population_size(n)` | 50 | 5D à 10D | Nombre de solutions candidates. Règle empirique : 5 à 10 fois le nombre de dimensions |
| `generations` | `with_generations(n)` | 1000 | 500-5000 | Nombre d'itérations. Plus = meilleure convergence |
| `mutation_factor` (F) | `with_mutation_factor(f)` | 0.8 | 0.4-1.0 | Facteur d'échelle du vecteur différence. Plus élevé = pas plus grands, plus d'exploration |
| `crossover_prob` (CR) | `with_crossover_prob(cr)` | 0.9 | 0.1-1.0 | Fraction de dimensions provenant du mutant. Plus élevé = changements plus perturbateurs |
| `bounds` | `with_bounds(lo, hi)` | (-10, 10) | Spécifique au problème | Plage de recherche. Les vecteurs d'essai sont ramenés dans les bornes |
| `seed` | `with_seed(s)` | 42 | Tout u64 | Graine du générateur aléatoire pour la reproductibilité |

### Guide d'interaction des paramètres

| Scénario | F recommandé | CR recommandé |
|---|---|---|
| Fonction séparable (dimensions indépendantes) | 0.5-0.8 | 0.1-0.3 |
| Non séparable (dimensions couplées) | 0.8-1.0 | 0.9-1.0 |
| Aptitude bruitée | 0.4-0.6 | 0.5-0.7 |
| Nombreux optima locaux | 0.8-1.0 | 0.9-1.0 |
| Haute dimension (D > 30) | 0.5-0.7 | 0.9-1.0 |

### Champs d'EvolutionResult

| Champ | Type | Description |
|---|---|---|
| `best_genes` | `Array1<f64>` | Le meilleur vecteur solution trouvé |
| `best_fitness` | `f64` | Aptitude de la meilleure solution (plus bas est mieux) |
| `generations` | `usize` | Nombre de générations exécutées |
| `fitness_history` | `Vec<f64>` | Meilleure aptitude à chaque génération |

## Pièges courants

1. **Population trop petite = convergence prématurée.** Avec moins d'individus que de dimensions, DE ne peut pas générer assez de vecteurs différences diversifiés. Utilisez au moins 5 fois le nombre de dimensions.

2. **F trop bas = stagnation.** Avec F < 0.4, les vecteurs différences sont tellement réduits que la population bouge à peine. L'algorithme converge trop tôt vers une solution sous-optimale.

3. **F trop élevé = instabilité.** Avec F > 1.2, les pas dépassent la cible, et les vecteurs d'essai atterrissent fréquemment en dehors de la zone utile (ils sont ramenés aux bornes, ce qui entraîne une perte d'information).

4. **CR trop bas sur les fonctions non séparables.** Quand CR est petit, seules 1 à 2 dimensions changent par essai. Si la fonction nécessite des changements coordonnés entre les dimensions (par exemple la vallée incurvée de Rosenbrock), un CR faible empêche l'algorithme de se déplacer en diagonale dans l'espace de recherche.

5. **Mêmes bornes pour toutes les dimensions.** Si le paramètre 1 varie de 0 à 1 et le paramètre 2 de 0 à 10 000, utiliser `with_bounds(0.0, 10000.0)` gaspille de l'effort de recherche sur le paramètre 1. ix utilise actuellement des bornes uniformes ; pour des échelles hétérogènes, normalisez d'abord vos paramètres.

6. **Pas d'arrêt anticipé.** L'algorithme s'exécute toujours pour le nombre total de générations. Si l'historique d'aptitude stagne tôt, vous gaspillez du calcul. Surveillez `fitness_history` et ajoutez votre propre logique d'arrêt anticipé.

## Pour aller plus loin

- **Algorithmes génétiques :** Pour l'optimisation combinatoire ou à types mixtes, voir [algorithmes-genetiques.md](./algorithmes-genetiques.md).
- **Variantes de DE :** DE/best/1/bin utilise le meilleur individu au lieu d'un individu aléatoire pour la mutation (`v = x_best + F * (x_r1 - x_r2)`). Converge plus vite mais risque de rester piégé dans des optima locaux. Pas encore implémenté mais facile à ajouter.
- **DE auto-adaptatif (jDE, SHADE) :** Adapte automatiquement F et CR pendant l'exécution. Élimine le besoin de choisir ces paramètres.
- **Gestion des contraintes :** Pénaliser les solutions infaisables dans la fonction d'aptitude : `fitness(x) + penalty * max(0, g(x))` où g(x) > 0 signifie contrainte violée.
- **DE multi-objectifs :** Maintenir un front de Pareto de solutions non dominées. À combiner avec les opérateurs de sélection de `ix-evolution`.
- **Méthodes hybrides :** Utiliser DE pour l'exploration globale, puis passer à un optimiseur local (par exemple les méthodes à gradient de `ix-optimize`) pour l'affinage autour de la meilleure solution.
