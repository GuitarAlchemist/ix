# Algorithmes génétiques

## Le problème

Vous concevez une antenne pour un satellite. La forme de l'antenne est définie par 12 paramètres (longueurs, angles, courbures). La qualité du signal pour une forme donnée peut être simulée, mais la fonction qui associe les paramètres à la qualité est sauvagement non linéaire, pleine d'optima locaux et ne présente aucun gradient exploitable. L'optimisation par le calcul différentiel (descente de gradient) est inutile ici.

Scénarios réels :
- **Conception de circuits :** Faire évoluer les valeurs des composants (résistances, condensateurs) pour atteindre une réponse en fréquence cible.
- **Ordonnancement d'atelier :** Faire évoluer les séquences de tâches sur les machines pour minimiser le temps total d'exécution.
- **Recherche d'architecture neuronale :** Faire évoluer les topologies de réseau (tailles des couches, connexions) pour maximiser la précision.
- **Équilibrage de personnages de jeu :** Faire évoluer les paramètres de comportement des PNJ (agressivité, vitesse, prudence) pour un gameplay équilibré.
- **Ingénierie structurelle :** Faire évoluer les configurations de treillis pour minimiser le poids tout en respectant les contraintes de charge.

## L'intuition

Les algorithmes génétiques s'inspirent de la sélection naturelle. Imaginez que vous élevez des chiens pour la vitesse :

1. **Commencez avec une population aléatoire** de chiens aux longueurs de pattes, masses musculaires et morphologies variées.
2. **Testez l'aptitude** en faisant courir tous les chiens. Les plus rapides sont « aptes ».
3. **Sélectionnez les parents** — les chiens rapides ont plus de chances de se reproduire (mais les lents ont encore une petite chance).
4. **Croisement** — les chiots héritent de certains traits de chaque parent. Un chiot pourrait avoir la longueur de pattes du parent A et la masse musculaire du parent B.
5. **Mutation** — occasionnellement, un trait aléatoire change légèrement. Peut-être qu'un chiot est un peu plus grand que ses deux parents.
6. **Répétez** pendant de nombreuses générations. Au fil du temps, la population converge vers des chiens rapides.

L'idée clé : vous n'avez jamais besoin de comprendre *pourquoi* certains traits rendent les chiens rapides. Il suffit de mesurer la vitesse et de laisser la sélection faire le reste. Cela rend les AG idéaux pour les problèmes où la fonction objectif est une boîte noire.

## Comment ça fonctionne

### L'algorithme

```
1. Initialize population of N random individuals
2. Evaluate fitness of each individual
3. Repeat for G generations:
   a. Select parents (tournament, roulette, or rank selection)
   b. Create offspring via crossover
   c. Apply mutation to offspring
   d. Evaluate fitness of offspring
   e. Replace population (keep elite individuals)
4. Return best individual found
```

### Sélection

Trois méthodes contrôlent quels individus deviennent parents :

**Sélection par tournoi :** Choisir k individus au hasard, garder le meilleur.
```
parent = best of k randomly chosen individuals
```
**En clair :** On organise une mini-compétition. Des tournois plus grands (k plus élevé) exercent plus de pression de sélection — moins d'individus faibles survivent.

**Sélection par roulette :** Probabilité proportionnelle à l'aptitude.
```
P(select i) = (max_fitness - fitness_i) / sum(max_fitness - fitness_j)
```
**En clair :** On fait tourner une roue pondérée. Les meilleurs individus ont une part plus grande. (Inversé pour la minimisation — aptitude plus basse = part plus grande.)

**Sélection par rang :** Probabilité proportionnelle au rang, pas à l'aptitude brute.
```
P(select i) = rank(i) / sum(all ranks)
```
**En clair :** Le meilleur obtient le rang N, le pire le rang 1. Cela évite le problème où un individu super-apte domine la sélection par roulette.

### Croisement (BLX-alpha)

Pour l'optimisation continue, ix utilise le croisement par mélange (BLX-alpha) :

```
For each gene dimension d:
    lo = min(parent1[d], parent2[d]) - alpha * |parent1[d] - parent2[d]|
    hi = max(parent1[d], parent2[d]) + alpha * |parent1[d] - parent2[d]|
    child[d] = random uniform in [lo, hi]
```

Avec alpha = 0.5 (la valeur par défaut).

**En clair :** La valeur du gène de l'enfant se situe quelque part entre les parents, avec un peu de marge de chaque côté. Cela permet aux descendants d'explorer légèrement au-delà de la plage de leurs parents.

### Mutation (gaussienne)

```
For each gene with probability 0.3:
    gene += Normal(0, mutation_rate)
```

**En clair :** Occasionnellement, on perturbe légèrement la valeur d'un gène par un petit montant aléatoire. Le taux de mutation contrôle l'amplitude de la perturbation.

### Aptitude (plus bas est mieux)

ix utilise la minimisation. La fonction d'aptitude prend une solution candidate (un vecteur de f64) et renvoie un score. Les scores les plus bas sont les meilleurs.

```
fitness(x) = your_objective_function(x)  // e.g., sum of squared errors
```

## En Rust

```rust
use ix_evolution::genetic::GeneticAlgorithm;
use ndarray::Array1;

// Minimize the Sphere function: f(x) = sum(x_i^2)
// Global minimum is at the origin with f(0) = 0
let result = GeneticAlgorithm::new()
    .with_population_size(100)    // 100 candidate solutions per generation
    .with_generations(500)        // run for 500 generations
    .with_mutation_rate(0.15)     // Gaussian std dev for mutation
    .with_bounds(-5.0, 5.0)      // search space: each dimension in [-5, 5]
    .with_seed(42)                // reproducible results
    .minimize(
        &|x: &Array1<f64>| x.mapv(|v| v * v).sum(),  // fitness function
        3,  // 3 dimensions
    );

println!("Best solution: {:.4}", result.best_genes);
println!("Best fitness:  {:.6}", result.best_fitness);
println!("Generations:   {}", result.generations);
println!("Fitness curve: first={:.4}, last={:.4}",
    result.fitness_history.first().unwrap(),
    result.fitness_history.last().unwrap(),
);
```

### Problème plus difficile : la fonction de Rastrigin

La fonction de Rastrigin possède de nombreux optima locaux, ce qui en fait un test standard pour l'optimisation globale :

```rust
use ix_evolution::genetic::GeneticAlgorithm;
use ndarray::Array1;
use std::f64::consts::PI;

let rastrigin = |x: &Array1<f64>| -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter()
        .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
        .sum::<f64>()
};

let result = GeneticAlgorithm::new()
    .with_population_size(200)
    .with_generations(1000)
    .with_mutation_rate(0.2)      // higher mutation to escape local optima
    .with_bounds(-5.12, 5.12)    // standard Rastrigin bounds
    .with_seed(123)
    .minimize(&rastrigin, 5);     // 5 dimensions

println!("Best fitness: {:.4} (global optimum is 0.0)", result.best_fitness);
```

### Sélection personnalisée et le trait Individual

L'AG utilise en interne `RealIndividual` qui implémente le trait `Individual` :

```rust
use ix_evolution::traits::{Individual, RealIndividual};
use ix_evolution::selection;
use ndarray::array;
use rand::SeedableRng;

// Create individuals manually
let pop = vec![
    RealIndividual::new(array![1.0, 2.0]).with_fitness(10.0),
    RealIndividual::new(array![0.5, 0.5]).with_fitness(3.0),
    RealIndividual::new(array![0.1, 0.1]).with_fitness(0.5),  // best
];

let mut rng = rand::rngs::StdRng::seed_from_u64(42);

// Tournament selection: pick best of 3 random candidates
let parent = selection::tournament(&pop, 3, &mut rng);
println!("Selected fitness: {}", parent.fitness());

// Roulette selection: fitness-proportional
let parent = selection::roulette(&pop, &mut rng);

// Rank selection: rank-proportional
let parent = selection::rank(&pop, &mut rng);

// Crossover and mutation
let child = parent.crossover(&pop[0], &mut rng);
let mut mutated = child.clone();
mutated.mutate(0.1, &mut rng);
```

## Quand l'utiliser

| Méthode | Meilleur quand | Gradient nécessaire | Gère les optima locaux |
|---|---|---|---|
| **Descente de gradient** | Objectif lisse et dérivable | Oui | Non (reste bloqué) |
| **Algorithme génétique** | Paysage accidenté, fonction boîte noire | Non | Oui (diversité de la population) |
| **Évolution différentielle** | Optimisation continue, moins de paramètres à régler | Non | Oui (souvent meilleur que l'AG) |
| **Recuit simulé** | Recherche à solution unique, simple à implémenter | Non | Partiellement (les redémarrages aléatoires aident) |
| **Essaim particulaire** | Continu, unimodal ou légèrement multimodal | Non | Partiellement |

**Utilisez un AG quand :**
- La fonction d'aptitude est une boîte noire (pas de gradient disponible).
- L'espace de recherche possède de nombreux optima locaux.
- Vous pouvez vous permettre de nombreuses évaluations d'aptitude (les AG ne sont pas économes en échantillons).
- Vous voulez une population de solutions diversifiées, pas seulement la meilleure.

**N'utilisez pas un AG quand :**
- L'objectif est lisse et dérivable (utilisez les méthodes à gradient — elles sont des ordres de grandeur plus rapides).
- L'évaluation de l'aptitude est extrêmement coûteuse (chaque génération évalue toute la population).
- La dimension du problème est très élevée (>100) — les AG peinent en haute dimension.

## Paramètres clés

| Paramètre | Méthode | Défaut | Description |
|---|---|---|---|
| `population_size` | `with_population_size(n)` | 100 | Nombre de solutions candidates. Plus grand = plus de diversité mais plus lent |
| `generations` | `with_generations(n)` | 500 | Nombre de cycles évolutionnaires. Plus = meilleure convergence |
| `mutation_rate` | `with_mutation_rate(r)` | 0.1 | Écart-type de la mutation gaussienne. Plus élevé = plus d'exploration |
| `crossover_rate` | (interne) | 0.8 | Probabilité de croisement vs. clonage. Fixé à 0.8 |
| `tournament_size` | (interne) | 3 | Nombre de candidats dans la sélection par tournoi |
| `elitism` | (interne) | 2 | Les N meilleurs individus survivent inchangés à la génération suivante |
| `bounds` | `with_bounds(lo, hi)` | (-10, 10) | Bornes de l'espace de recherche par dimension. Les gènes sont ramenés dans cette plage |
| `seed` | `with_seed(s)` | 42 | Graine du générateur aléatoire pour la reproductibilité |

### Champs d'EvolutionResult

| Champ | Type | Description |
|---|---|---|
| `best_genes` | `Array1<f64>` | Le meilleur vecteur solution trouvé |
| `best_fitness` | `f64` | Aptitude de la meilleure solution (plus bas est mieux) |
| `generations` | `usize` | Nombre de générations exécutées |
| `fitness_history` | `Vec<f64>` | Meilleure aptitude à chaque génération (devrait décroître de manière monotone grâce à l'élitisme) |

## Pièges courants

1. **Convergence prématurée.** Si la population devient trop homogène trop vite, l'AG reste bloqué dans un optimum local. Solution : augmenter le taux de mutation, augmenter la taille de la population, ou réduire la taille du tournoi (moins de pression de sélection).

2. **Taux de mutation trop élevé = recherche aléatoire.** Si le taux de mutation est supérieur à l'échelle de l'espace de recherche, les descendants sont essentiellement aléatoires. Maintenez-le à 5-20 % de la plage des bornes.

3. **L'élitisme est crucial.** Sans élitisme (conserver les meilleurs individus inchangés), la meilleure solution peut être perdue à cause du croisement et de la mutation. ix conserve les 2 meilleurs par défaut.

4. **Les bornes doivent correspondre au problème.** Si l'optimum global est à x = 100 mais que vos bornes sont [-10, 10], l'AG ne le trouvera jamais. Fixez toujours les bornes pour couvrir la région faisable.

5. **L'évaluation de l'aptitude domine le temps d'exécution.** L'AG lui-même est rapide. Si votre fonction d'aptitude prend 1 seconde, chaque génération de 100 individus prend 100 secondes. Envisagez de paralléliser l'évaluation de l'aptitude (non intégré dans l'AG d'ix, mais vous pouvez pré-évaluer et utiliser `with_fitness()`).

6. **Pas adapté à l'optimisation sous contraintes.** L'AG ne possède pas de gestion intégrée des contraintes. Si votre problème comporte des contraintes comme « x1 + x2 <= 10 », vous devez les encoder comme des pénalités dans la fonction d'aptitude.

## Pour aller plus loin

- **Évolution différentielle :** Un algorithme évolutionnaire plus simple qui surpasse souvent les AG en optimisation continue. Voir [evolution-differentielle.md](./evolution-differentielle.md).
- **Recuit simulé et PSO :** Disponibles dans `ix-optimize` pour des stratégies alternatives d'optimisation globale.
- **Types d'Individual personnalisés :** Implémentez le trait `Individual` pour des représentations non continues (permutations, chaînes de bits, arbres).
- **Mutation adaptative :** Diminuer le taux de mutation au fil des générations : `ga.with_mutation_rate(0.3 / (gen as f64 + 1.0).sqrt())`. Nécessite d'exécuter l'AG manuellement dans une boucle.
- **Modèle en îlots :** Exécuter plusieurs populations AG en parallèle et migrer occasionnellement des individus entre elles pour une meilleure diversité.
