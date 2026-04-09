# Recuit simulé

> Accepter une solution moins bonne maintenant pour en trouver une meilleure plus tard — puis cesser progressivement d'accepter les mauvais mouvements en « refroidissant ».

**Prérequis :** [Intuition du calcul différentiel](../foundations/calculus-intuition.md), [Probabilités et statistiques](../foundations/probability-and-statistics.md)

---

## Le problème

Vous gérez un entrepôt de 50 postes de travail, 200 zones de rayonnage et 12 quais d'expédition. Chaque jour, les opérateurs marchent entre les zones pour préparer les commandes. L'agencement actuel a évolué de manière organique pendant une décennie et manque d'efficacité — les opérateurs parcourent en moyenne 2,3 kilomètres par poste rien qu'en se déplaçant entre les zones. Vous souhaitez réorganiser les zones pour minimiser la distance totale de marche.

C'est un problème d'optimisation combinatoire. Il existe plus d'agencements possibles que d'atomes dans l'univers (200 factorielle). On ne peut pas tous les essayer. Pire encore, la fonction de coût (distance totale de marche quotidienne) n'est pas lisse — déplacer une zone peut améliorer les choses, en déplacer deux peut les dégrader, et le paysage est truffé de minima locaux (des agencements qui semblent bons mais qui sont loin de l'optimal).

La descente de gradient ne peut pas aider ici : il n'y a pas de gradient dans un problème discret d'agencement. Vous avez besoin d'un algorithme qui explore aléatoirement, tolère des reculs temporaires et se concentre progressivement sur les bonnes solutions.

---

## L'intuition

Le recuit simulé est emprunté à la métallurgie. Quand on chauffe un métal puis qu'on le refroidit lentement, les atomes se disposent dans une structure cristalline à basse énergie. Si on le refroidit trop vite, on obtient une structure fragile et désorganisée.

L'algorithme fonctionne de la même manière :

1. **Démarrer à chaud.** À haute température, l'algorithme accepte presque n'importe quel mouvement — même ceux qui empirent la solution. Cela lui permet d'explorer largement et de s'échapper des mauvais voisinages.

2. **Évaluer les voisins.** À chaque étape, apporter un petit changement aléatoire à la solution actuelle (échanger deux zones, décaler un poste de travail). Calculer le nouveau coût.

3. **Accepter ou rejeter.**
   - Si la nouvelle solution est *meilleure*, l'accepter systématiquement.
   - Si la nouvelle solution est *pire*, l'accepter avec une probabilité qui dépend de (a) à quel point elle est pire et (b) la température actuelle. Haute température signifie forte probabilité d'acceptation. Basse température signifie probabilité quasi nulle.

4. **Refroidir.** Réduire progressivement la température selon un programme de refroidissement. Au début, l'algorithme est un explorateur aventureux. En fin de course, c'est un grimpeur prudent qui n'accepte que les améliorations.

L'idée clé est qu'accepter des mouvements défavorables en début de parcours évite de rester bloqué dans le premier minimum local rencontré. À mesure que la température baisse, l'algorithme passe naturellement de l'exploration globale à l'affinage local.

---

## Comment ça fonctionne

### Probabilité d'acceptation

```
If cost_new < cost_current:
    accept (always)

If cost_new >= cost_current:
    accept with probability P = exp(-delta / T)
    where delta = cost_new - cost_current
```

**En clair :** Quand un voisin est moins bon, on calcule à quel point il est pire (`delta`) et quelle est la température actuelle (`T`). La probabilité d'accepter le mouvement défavorable est `e^(-delta/T)`. Quand `T` est grand, cette probabilité est proche de 1 (on accepte presque tout). Quand `T` est minuscule, cette probabilité est proche de 0 (on rejette presque tout ce qui est pire). Quand `delta` est énorme (bien pire), la probabilité chute même à haute température — on évite quand même les mouvements catastrophiques.

### Programmes de refroidissement

Le programme de refroidissement détermine la vitesse à laquelle la température baisse. C'est le principal levier de contrôle au-delà de la température initiale.

**Refroidissement exponentiel :** `T(k) = T0 * alpha^k`

**En clair :** On multiplie la température par un facteur constant (par exemple 0.995) à chaque étape. C'est le programme le plus courant. Il refroidit vite au début, puis ralentit — passant plus de temps aux basses températures où l'algorithme affine la solution.

**Refroidissement linéaire :** `T(k) = T0 / (1 + alpha * k)`

**En clair :** La température diminue à un taux fixe proportionnel au numéro de l'étape. Plus simple que l'exponentiel mais peut refroidir trop vite dans les premières phases.

**Refroidissement logarithmique :** `T(k) = T0 / ln(1 + k)`

**En clair :** Le programme le plus lent. La température baisse très graduellement. Théoriquement, il garantit de trouver l'optimum global (en temps infini), mais en pratique on dispose rarement d'un temps infini. À utiliser pour les problèmes où rester bloqué dans un minimum local est une préoccupation sérieuse et où l'on peut se permettre beaucoup d'itérations.

---

## En Rust

### Définir la fonction objectif

Comme tous les optimiseurs dans ix, le recuit simulé fonctionne avec tout ce qui implémente `ObjectiveFunction`. Pour le problème d'agencement d'entrepôt, chaque paramètre pourrait représenter un décalage de coordonnées pour une zone :

```rust
use ix_optimize::traits::{ClosureObjective, ObjectiveFunction};
use ix_optimize::annealing::{SimulatedAnnealing, CoolingSchedule};
use ndarray::{array, Array1};

// Simplified warehouse cost: total distance between zone pairs weighted by traffic.
// In reality, you'd compute Euclidean distances between zone positions
// multiplied by order frequency between each pair.
let warehouse_cost = ClosureObjective {
    f: |positions: &Array1<f64>| {
        // Rastrigin-like function: many local minima, one global minimum at origin.
        // This mimics a warehouse layout with many "pretty good" arrangements
        // but only one optimal one.
        let n = positions.len() as f64;
        10.0 * n
            + positions
                .iter()
                .map(|&x| x * x - 10.0 * (2.0 * std::f64::consts::PI * x).cos())
                .sum::<f64>()
    },
    dimensions: 6, // 3 zones, each with an (x, y) coordinate
};
```

### Exécuter le recuit simulé

```rust
let sa = SimulatedAnnealing::new()
    .with_temp(100.0, 1e-10)                              // start hot, cool to near-zero
    .with_cooling(CoolingSchedule::Exponential { alpha: 0.995 }) // multiply temp by 0.995 each step
    .with_max_iterations(50_000)                           // budget
    .with_step_size(0.5)                                   // radius of random perturbation
    .with_seed(42);                                        // reproducibility

let initial_layout = array![5.0, -3.0, 7.0, -1.0, 4.0, 2.0]; // random starting positions

let result = sa.minimize(&warehouse_cost, initial_layout);

println!("Best layout: {:?}", result.best_params);
println!("Cost: {:.4}", result.best_value);
println!("Iterations: {}", result.iterations);
println!("Converged (cooled fully): {}", result.converged);
```

### Essayer différents programmes de refroidissement

```rust
let schedules = vec![
    ("Exponential(0.995)", CoolingSchedule::Exponential { alpha: 0.995 }),
    ("Exponential(0.999)", CoolingSchedule::Exponential { alpha: 0.999 }),
    ("Linear(0.01)",       CoolingSchedule::Linear { alpha: 0.01 }),
    ("Logarithmic",        CoolingSchedule::Logarithmic),
];

let initial = array![5.0, -3.0, 7.0, -1.0, 4.0, 2.0];

for (name, schedule) in schedules {
    let sa = SimulatedAnnealing::new()
        .with_temp(100.0, 1e-10)
        .with_cooling(schedule)
        .with_max_iterations(50_000)
        .with_step_size(0.5)
        .with_seed(42);

    let result = sa.minimize(&warehouse_cost, initial.clone());
    println!("{:25} | cost: {:.4} | iters: {}", name, result.best_value, result.iterations);
}
```

### Comprendre la valeur de retour

`minimize` renvoie un `OptimizeResult` :

| Champ         | Type          | Signification                                              |
|---------------|---------------|-------------------------------------------------------------|
| `best_params` | `Array1<f64>` | Le meilleur vecteur de paramètres trouvé pendant toute l'exécution |
| `best_value`  | `f64`         | La valeur de la fonction objectif à `best_params`            |
| `iterations`  | `usize`       | Le nombre d'étapes exécutées                                 |
| `converged`   | `bool`        | `true` si la température est passée sous `min_temp`          |

Notez que `best_params` suit le meilleur résultat global, pas seulement la position courante. Même si l'algorithme s'éloigne d'une bonne solution (en acceptant un mouvement défavorable), il mémorise la meilleure solution jamais trouvée.

---

## Quand l'utiliser

| Situation | Utiliser le recuit simulé ? |
|-----------|----------------------------|
| Problèmes combinatoires (ordonnancement, agencement, routage) | Oui — le recuit simulé excelle ici car il ne nécessite pas de gradient |
| Fonction de coût avec de nombreux minima locaux | Oui — l'acceptation aléatoire permet au recuit de s'échapper des pièges |
| Vous pouvez calculer le coût mais pas son gradient | Oui — le recuit n'appelle que `evaluate`, jamais `gradient` |
| Fonction de perte lisse et convexe (régression linéaire) | Non — la [descente de gradient](descente-de-gradient.md) sera plus rapide et plus précise |
| Vous avez besoin de la solution provablement optimale | Peut-être — le recuit donne de bonnes solutions approchées mais pas de garantie d'optimalité en temps fini |
| Problèmes de très haute dimension (>1000 paramètres) | Avec prudence — le recuit ralentit car le voisinage est vaste ; envisagez plutôt l'[essaim particulaire](essaim-particulaire.md) |

---

## Paramètres clés

### Température initiale (`with_temp(initial, min)`)

- La température initiale doit être suffisamment élevée pour que l'algorithme accepte la plupart des mouvements au début. Une heuristique courante : la fixer pour que la probabilité d'acceptation du « mauvais mouvement moyen » soit d'environ 0.8.
- Si vous observez que l'algorithme explore à peine (le coût diminue de manière monotone dès le début), la température est trop basse.
- Si l'algorithme erre aléatoirement pendant la majorité de l'exécution et ne s'améliore que vers la fin, la température est trop haute ou le refroidissement est trop lent.
- Plage typique : `10.0` à `10 000.0` selon l'ordre de grandeur de votre fonction de coût.

### Température minimale

- Quand `T < min_temp`, l'algorithme s'arrête et renvoie le résultat. Fixez cette valeur très bas (`1e-8` à `1e-10`) sauf si vous souhaitez un arrêt anticipé.

### Programme de refroidissement

- **Exponentiel avec `alpha = 0.995` :** Bon choix par défaut. Termine en environ 1 000 étapes (la température tombe à ~0.7 % de l'initiale).
- **Exponentiel avec `alpha = 0.999` :** Refroidissement beaucoup plus lent. À utiliser avec un `max_iterations` plus élevé pour les problèmes difficiles.
- **Linéaire :** Plus simple. Utile quand vous savez approximativement combien d'itérations vous pouvez vous permettre.
- **Logarithmique :** Théoriquement optimal mais extrêmement lent. À utiliser quand la qualité de la solution compte plus que le temps d'exécution.

### Taille du pas (`with_step_size`)

- Contrôle l'écart-type de la perturbation gaussienne ajoutée à chaque paramètre. Des valeurs plus grandes explorent plus agressivement.
- Si l'algorithme trouve une bonne région mais n'arrive pas à l'affiner, réduisez `step_size`.
- Si l'algorithme produit sans cesse des voisins avec des coûts radicalement différents, réduisez `step_size`.
- Un bon point de départ : environ 5 à 10 % de la plage des paramètres.

### Graine aléatoire (`with_seed`)

- Le recuit simulé est stochastique. Des graines différentes donnent des exécutions différentes. Pour les problèmes importants, lancez le recuit plusieurs fois avec des graines différentes et gardez le meilleur résultat.

---

## Pièges courants

**Refroidissement trop rapide.** L'erreur la plus courante. Si `alpha` est trop petit (par exemple 0.9), la température chute en quelques dizaines d'étapes et l'algorithme explore à peine. Il se comporte comme une descente gloutonne et reste bloqué dans le premier minimum local qu'il trouve. Solution : augmenter `alpha` vers 1.0 (par exemple 0.999) et augmenter `max_iterations`.

**Refroidissement trop lent.** L'algorithme gaspille tout son budget à errer aléatoirement parce que la température ne baisse jamais assez pour se concentrer. Solution : diminuer `alpha` (par exemple de 0.999 à 0.995) ou augmenter `max_iterations`.

**Inadéquation de la taille du pas.** Si `step_size` est beaucoup plus grand que l'échelle de vos paramètres, chaque voisin est essentiellement aléatoire. S'il est beaucoup plus petit, l'algorithme rampe. Adaptez la taille du pas à la plage des paramètres.

**Oublier l'échelle.** La probabilité d'acceptation dépend de `delta / T`. Si les valeurs de votre fonction de coût sont en millions, une température initiale de 100.0 n'acceptera presque rien. Calibrez la température en fonction de l'ordre de grandeur de votre fonction de coût.

**N'exécuter qu'une seule fois.** Le recuit est stochastique. Une seule exécution peut atterrir dans un minimum local médiocre. Bonne pratique : lancer 5 à 10 exécutions indépendantes avec des graines différentes et garder le meilleur résultat.

**Ne pas suivre le meilleur résultat global.** L'implémentation d'ix le fait déjà — `best_params` est le meilleur jamais observé, pas le dernier visité. Mais si vous implémentez votre propre boucle de recuit, veillez à conserver une variable `best_ever` distincte.

---

## Pour aller plus loin

- **Alternative basée sur le gradient :** Si votre fonction objectif est lisse et dérivable, la [descente de gradient](descente-de-gradient.md) convergera plus vite et plus précisément.
- **Alternative par population :** L'[optimisation par essaim particulaire](essaim-particulaire.md) explore avec de nombreux agents simultanément, ce qui peut être plus efficace sur les paysages multimodaux.
- **Approche évolutionnaire :** Les [algorithmes génétiques](../evolutionnaire/algorithmes-genetiques.md) maintiennent une population et utilisent le croisement et la mutation — une autre façon d'équilibrer exploration et exploitation.
- **Exemple concret :** [`examples/optimization/pso_rosenbrock.rs`](../../examples/optimization/pso_rosenbrock.rs) démontre l'optimisation sur la fonction de Rosenbrock ; essayez de remplacer PSO par le recuit pour comparer.
- **Extensions combinatoires :** Pour les problèmes discrets (voyageur de commerce, ordonnancement de tâches), l'étape de perturbation devient un échange discret au lieu d'un décalage gaussien. La logique d'acceptation reste identique.
