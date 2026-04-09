# Dynamiques évolutionnaires

## Le problème

Vous étudiez un écosystème où faucons et colombes se disputent des ressources. Les faucons combattent agressivement ; les colombes partagent pacifiquement. Quand deux faucons se rencontrent, ils se battent et les deux sont blessés. Quand un faucon rencontre une colombe, le faucon prend tout. Quand deux colombes se rencontrent, elles partagent la ressource.

Au fil de nombreuses générations, que se passe-t-il pour la composition de la population ? Les faucons vont-ils prendre le dessus ? Les colombes vont-elles survivre ? Existe-t-il un équilibre stable ?

Ou bien : vous modélisez un marché où des entreprises peuvent choisir une tarification agressive (faucon) ou coopérative (colombe). Avec le temps, les entreprises qui réalisent plus de profits grandissent ; celles qui en perdent rétrécissent ou disparaissent. Quel est l'équilibre à long terme ?

Ce sont des questions sur les **dynamiques évolutionnaires** -- comment les populations de stratégies changent au fil du temps lorsque les individus interagissent et que les stratégies qui réussissent se reproduisent davantage.

## L'intuition

Imaginez une grande population où chacun joue un jeu simple contre des adversaires aléatoires. Chaque individu utilise une stratégie fixe. Après chaque tour :

- Les stratégies qui ont obtenu des gains supérieurs à la moyenne **croissent** (plus d'individus les adoptent).
- Les stratégies qui ont obtenu des gains inférieurs à la moyenne **déclinent**.

C'est la **dynamique du réplicateur** -- l'analogue évolutionnaire de la sélection naturelle. Elle ne nécessite pas d'évolution biologique ; tout système où le succès engendre l'imitation suit les mêmes mathématiques (diffusion des mèmes en ligne, stratégies d'entreprise sur les marchés, langages de programmation gagnant en popularité).

Une **Stratégie Evolutivement Stable (SES)** est une stratégie qui, une fois dominante dans la population, ne peut être envahie par un petit groupe de mutants utilisant une stratégie différente. Pensez-y comme un équilibre auto-renforçant.

La différence clé avec l'équilibre de Nash : Nash demande « un *individu* peut-il s'améliorer en changeant ? » La SES demande « un *petit groupe d'envahisseurs* peut-il conquérir la population ? » La SES est strictement plus forte -- toute SES correspond à un équilibre de Nash, mais tout EN n'est pas une SES.

## Comment ça fonctionne

### Dynamique du réplicateur

Soit `x_i` la fraction de la population utilisant la stratégie `i`. La **matrice de gains** `A` définit le gain de la stratégie `i` contre la stratégie `j` comme `A[i,j]`.

La fitness de la stratégie `i` dans la population actuelle :

```
f_i = sum_j A[i,j] * x_j
```

**En clair :** votre fitness est le gain moyen que vous obtenez en jouant contre un membre aléatoire de la population.

La fitness moyenne de toute la population :

```
f_avg = sum_i x_i * f_i
```

L'équation du réplicateur :

```
dx_i/dt = x_i * (f_i - f_avg)
```

**En clair :** une stratégie croît proportionnellement à sa supériorité par rapport à la moyenne. Si les faucons gagnent plus que la moyenne de la population, la fraction de faucons augmente. S'ils gagnent moins, elle diminue. Le facteur `x_i` signifie qu'une stratégie à 0% reste à 0% (les stratégies éteintes ne peuvent pas apparaître spontanément).

### Stratégie Evolutivement Stable (SES)

La stratégie `i` est une SES si pour toute autre stratégie `j` :

**Condition 1 (Nash) :** `A[i,i] >= A[j,i]`

L'occupant fait au moins aussi bien contre lui-même que tout envahisseur contre l'occupant.

**Condition 2 (Stabilité) :** Si `A[i,i] = A[j,i]`, alors `A[i,j] > A[j,j]`

Si l'envahisseur fait match nul contre l'occupant, l'occupant doit faire strictement mieux contre l'envahisseur que l'envahisseur contre lui-même.

**En clair :** (1) la stratégie occupante est une meilleure réponse à elle-même, et (2) en cas d'égalité, l'occupant bat l'envahisseur dans le duel miroir.

### Jeu faucon-colombe

L'exemple classique avec une valeur de ressource `V` et un coût de combat `C` :

|        | Faucon       | Colombe |
|--------|--------------|---------|
| **Faucon** | `(V-C)/2` | `V`     |
| **Colombe** | `0`        | `V/2`   |

- Si `V > C` : Faucon est une SES (le combat en vaut la peine).
- Si `V < C` : Aucune stratégie pure n'est une SES. La population converge vers un mélange de `V/C` faucons et `(1 - V/C)` colombes.

## En Rust

Le crate `ix-game` fournit les dynamiques évolutionnaires avec `ndarray` :

```rust
use ix_game::evolutionary::{
    replicator_dynamics, is_ess, find_ess,
    hawk_dove_matrix, rps_matrix,
    two_population_replicator,
};
use ndarray::{array, Array1, Array2};

fn main() {
    // --- Jeu faucon-colombe ---
    let hd: Array2<f64> = hawk_dove_matrix(2.0, 4.0); // V=2, C=4 (V < C)

    // Vérifier la SES
    let ess_strategies: Vec<usize> = find_ess(&hd);
    println!("Stratégies SES : {:?}", ess_strategies);
    // Vide -- ni Faucon pur ni Colombe pure n'est SES quand V < C.

    println!("Faucon est SES ? {}", is_ess(&hd, 0));  // false
    println!("Colombe est SES ? {}", is_ess(&hd, 1));  // false

    // Simuler la dynamique du réplicateur en partant de 80% faucons, 20% colombes.
    let initial = Array1::from_vec(vec![0.8, 0.2]);
    let trajectory: Vec<Array1<f64>> = replicator_dynamics(&hd, &initial, 0.01, 10_000);

    let final_state = trajectory.last().unwrap();
    println!("Population finale : Faucon={:.2}%, Colombe={:.2}%",
             final_state[0] * 100.0, final_state[1] * 100.0);
    // Converge vers V/C = 50% Faucons, 50% Colombes.

    // --- Dilemme du prisonnier ---
    // La trahison domine : les traîtres envahissent la population.
    let pd = array![[3.0, 0.0], [5.0, 1.0]]; // Coopérer, Trahir
    let pd_initial = Array1::from_vec(vec![0.5, 0.5]);
    let pd_traj = replicator_dynamics(&pd, &pd_initial, 0.01, 5000);
    let pd_final = pd_traj.last().unwrap();
    println!("DP final : Coopérer={:.2}%, Trahir={:.2}%",
             pd_final[0] * 100.0, pd_final[1] * 100.0);
    // Fraction Trahir -> ~100%

    // --- Pierre-feuille-ciseaux ---
    // Les populations oscillent sans converger (aucune SES n'existe).
    let rps: Array2<f64> = rps_matrix(1.0, -1.0, 0.0);
    let rps_initial = Array1::from_vec(vec![0.4, 0.3, 0.3]);
    let rps_traj = replicator_dynamics(&rps, &rps_initial, 0.01, 10_000);
    let rps_final = rps_traj.last().unwrap();
    println!("PFC final : P={:.2}, F={:.2}, C={:.2}",
             rps_final[0], rps_final[1], rps_final[2]);
    // Les trois stratégies persistent (oscillent autour de 1/3 chacune).

    // --- Réplicateur à deux populations ---
    // Jeu asymétrique : prédateurs vs proies avec des ensembles de stratégies différents.
    let pred_payoff = array![[3.0, 1.0], [2.0, 4.0]]; // Stratégies prédateurs vs proies
    let prey_payoff = array![[1.0, 4.0], [3.0, 2.0]]; // Stratégies proies vs prédateurs
    let pred_init = Array1::from_vec(vec![0.5, 0.5]);
    let prey_init = Array1::from_vec(vec![0.5, 0.5]);

    let (pred_traj, prey_traj) = two_population_replicator(
        &pred_payoff, &prey_payoff,
        &pred_init, &prey_init,
        0.01, 5000,
    );
    let pred_final = pred_traj.last().unwrap();
    let prey_final = prey_traj.last().unwrap();
    println!("Prédateurs : {:?}", pred_final);
    println!("Proies : {:?}", prey_final);
}
```

### Résumé de l'API

| Fonction | Signature | Ce qu'elle fait |
|----------|-----------|--------------|
| `replicator_dynamics(payoff, initial, dt, steps)` | `-> Vec<Array1<f64>>` | Simuler l'évolution de la population, retourner la trajectoire complète |
| `is_ess(payoff, strategy)` | `-> bool` | Vérifier si la stratégie pure `i` est évolutivement stable |
| `find_ess(payoff)` | `-> Vec<usize>` | Trouver toutes les SES parmi les stratégies pures |
| `hawk_dove_matrix(V, C)` | `-> Array2<f64>` | Matrice de gains classique faucon-colombe |
| `rps_matrix(win, lose, draw)` | `-> Array2<f64>` | Matrice de gains pierre-feuille-ciseaux |
| `two_population_replicator(A, B, init_a, init_b, dt, steps)` | `-> (Vec<Array1>, Vec<Array1>)` | Dynamique asymétrique à deux populations |

### Lecture de la trajectoire

La trajectoire est un `Vec<Array1<f64>>` où chaque élément est un instantané des proportions de la population à un pas de temps. Les proportions s'additionnent toujours pour donner 1.0 (elles vivent sur le simplexe des probabilités).

```rust
let traj = replicator_dynamics(&payoff, &initial, 0.01, 1000);

// Population au pas de temps 0 (initiale) :
println!("{:?}", traj[0]);

// Population au pas de temps 500 :
println!("{:?}", traj[500]);

// Population finale :
println!("{:?}", traj.last().unwrap());
```

## Quand l'utiliser

| Situation | Outil | Pourquoi |
|-----------|------|-----|
| La stratégie X va-t-elle dominer une population ? | `replicator_dynamics` | Simule les dynamiques de sélection naturelle |
| Une stratégie est-elle résistante à l'invasion ? | `is_ess` | Vérifie les conditions de Nash et de stabilité |
| Equilibre à long terme d'un jeu | `replicator_dynamics` | Converge vers les points fixes stables |
| Populations asymétriques (prédateur-proie) | `two_population_replicator` | Dynamiques séparées pour chaque population |
| Analyse stratégique ponctuelle | Equilibres de Nash plutôt | La SES concerne les populations, pas les individus |

## Paramètres clés

| Paramètre | Type | Ce qu'il contrôle |
|-----------|------|-----------------|
| `payoff_matrix` | `Array2<f64>` | `A[i,j]` = gain de la stratégie `i` contre la stratégie `j`. Doit être carrée pour la dynamique à une population. |
| `initial` | `Array1<f64>` | Fractions initiales de la population. Doivent sommer à 1.0. Doivent être > 0 pour les stratégies à suivre (zéro = éteinte pour toujours). |
| `dt` | `f64` | Pas de temps pour l'intégration d'Euler. Plus petit = plus précis, plus de pas nécessaires. 0.01 est un bon défaut. |
| `steps` | `usize` | Nombre de pas de temps. Plus = simulation plus longue. 5 000-10 000 est typique pour la convergence. |
| `V, C` (faucon-colombe) | `f64` | Valeur de la ressource et coût du combat. `V > C` = les faucons gagnent ; `V < C` = équilibre mixte à `V/C` faucons. |

### Choisir dt

L'équation du réplicateur est intégrée par la méthode d'Euler explicite. Si `dt` est trop grand, la simulation peut dépasser et produire des populations négatives (qui sont ramenées à zéro). Règles pratiques :

- `dt = 0.01` avec `steps = 10 000` (simule 100 unités de temps) fonctionne pour la plupart des jeux.
- Si les populations oscillent violemment ou tombent à zéro de manière inattendue, réduisez `dt` à 0.001.
- Pour les jeux avec de grandes valeurs de gains, réduisez `dt` proportionnellement.

## Pièges

1. **Une population initiale nulle reste nulle.** L'équation du réplicateur a `dx/dt proportionnel à x`. Si `x_i = 0` initialement, cette stratégie ne peut jamais apparaître. C'est par conception (pas de mutation spontanée), mais cela signifie que vous devez inclure toutes les stratégies qui vous intéressent dans la population initiale, même à une fraction minuscule comme 0.01.

2. **La vérification SES ne porte que sur les stratégies pures.** `is_ess` vérifie si une seule stratégie pure est évolutivement stable. Une SES *mixte* (comme l'équilibre 50/50 faucon-colombe quand V < C) n'est pas détectée par `find_ess`. Utilisez `replicator_dynamics` pour trouver la composition à long terme de la population.

3. **Pierre-feuille-ciseaux oscille sans converger.** La dynamique du réplicateur pour PFC orbite autour du point fixe intérieur (1/3, 1/3, 1/3) sans converger. En temps continu, les orbites sont neutrement stables. En temps discret (Euler), elles peuvent lentement spiraler vers l'extérieur. Réduisez `dt` si cela pose problème.

4. **La matrice de gains doit être cohérente.** Pour la dynamique à une population, la matrice de gains doit être carrée (n stratégies jouant contre n stratégies). Pour la dynamique à deux populations, les matrices peuvent être rectangulaires.

5. **La dynamique du réplicateur ignore les mutations.** Contrairement à l'évolution biologique, l'équation du réplicateur n'a pas de terme de mutation. Une fois qu'une stratégie s'éteint, elle ne peut pas revenir. Si vous avez besoin de mutations, ajoutez un petit terme de mélange uniforme à la mise à jour de la population.

## Pour aller plus loin

- **Faucon-colombe-bourgeois :** Ajoutez une troisième stratégie (« si je suis arrivé le premier, combattre comme un faucon ; sinon, se retirer comme une colombe »). C'est souvent l'unique SES.
- **Dynamique du réplicateur stochastique :** Ajouter du bruit pour modéliser les populations finies où la dérive aléatoire compte. Important pour les petites populations.
- **Equilibres de Nash :** La théorie non évolutionnaire de l'interaction stratégique. Voir [Equilibres de Nash](./equilibres-de-nash.md). Toute SES est un équilibre de Nash, mais pas l'inverse.
- **Jeux à champ moyen :** Le crate `ix-game` inclut également la théorie des jeux à champ moyen pour les grandes populations avec des espaces de stratégies continus. Voir `ix_game::mean_field`.
- Lecture : Hofbauer et Sigmund, *Evolutionary Games and Population Dynamics* (1998) -- le manuel de référence.
