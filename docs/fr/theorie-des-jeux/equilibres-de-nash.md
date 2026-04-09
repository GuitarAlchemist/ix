# Equilibres de Nash

## Le problème

Deux entreprises de VTC concurrentes fixent leurs prix pour la même ville. Si les deux pratiquent des tarifs élevés, elles se partagent le marché confortablement. Si l'une casse ses prix, elle capte la majorité des clients. Si les deux cassent leurs prix, c'est une guerre tarifaire où personne ne gagne.

Chaque entreprise choisit une stratégie. Le résultat dépend des *deux* stratégies combinées. Aucune entreprise ne peut modifier sa stratégie pour faire mieux, *étant donné ce que fait l'autre*. Ce point de blocage est un **équilibre de Nash** -- l'issue stable où aucun joueur n'a d'incitation à dévier unilatéralement.

Les équilibres de Nash apparaissent partout : stratégie de prix, routage réseau (où chaque paquet choisit le « meilleur » chemin), biologie évolutive (stratégies de population stables) et courses aux armements.

## L'intuition

Imaginez deux joueurs assis face à face. Chacun inscrit secrètement une stratégie, puis les deux révèlent simultanément. Les gains dépendent de la combinaison.

Un équilibre de Nash est une paire de stratégies où les deux joueurs regardent le résultat et disent : « Vu ce que l'autre a fait, je n'aurais pas pu faire mieux. » Aucun des deux n'a de regret.

Point clé : un équilibre de Nash n'est pas nécessairement le *meilleur* résultat pour quiconque. Dans le dilemme du prisonnier, la double défection est un équilibre de Nash -- mais les deux joueurs seraient mieux lotis s'ils coopéraient. L'équilibre est *stable*, pas *optimal*.

**Stratégie pure :** chaque joueur choisit une action déterminée. « Toujours trahir. »

**Stratégie mixte :** chaque joueur randomise. « Jouer Pile avec probabilité 0.5, Face avec probabilité 0.5. » Nash a prouvé que tout jeu fini possède au moins un équilibre (éventuellement mixte).

## Comment ça fonctionne

### Jeu bimatriciel

Un jeu à deux joueurs est défini par deux matrices de gains :

- `payoff_a[i][j]` = gain du joueur A lorsque A joue la stratégie `i` et B joue la stratégie `j`.
- `payoff_b[i][j]` = gain du joueur B dans le même scénario.

### Meilleure réponse

La **meilleure réponse** du joueur A à la stratégie de B est la stratégie (ou le mélange) qui maximise le gain espéré de A :

```
BR_A(strategy_b) = argmax_i sum_j (strategy_b[j] * payoff_a[i][j])
```

**En clair :** si je sais ce que vous allez faire (ou votre distribution de probabilité), je choisis ce qui me donne le gain espéré le plus élevé.

### Equilibre de Nash

Un profil de stratégie (strategy_a, strategy_b) est un équilibre de Nash si :

```
strategy_a is a best response to strategy_b
AND
strategy_b is a best response to strategy_a
```

**En clair :** personne ne peut s'améliorer en changeant de stratégie, étant donné ce que fait l'autre.

### Trouver les équilibres

Pour les jeux 2x2, les équilibres mixtes peuvent être trouvés analytiquement en utilisant des **conditions d'indifférence** : le joueur B mélange de sorte que le joueur A soit indifférent entre ses stratégies, et vice versa.

Pour la probabilité de mélange `q` du joueur B :

```
payoff_a[0,0] * q + payoff_a[0,1] * (1-q) = payoff_a[1,0] * q + payoff_a[1,1] * (1-q)
```

**En clair :** B randomise de sorte que A obtienne le même gain espéré quel que soit son choix. Si A n'était pas indifférent, A aurait une préférence stricte, et le mélange de B ne serait pas stable.

## En Rust

Le crate `ix-game` fournit l'analyse de jeux bimatriciels avec `ndarray` :

```rust
use ix_game::nash::{
    BimatrixGame, StrategyProfile, fictitious_play, dominant_strategy_equilibrium,
};
use ndarray::{array, Array1};

fn main() {
    // --- Dilemme du prisonnier ---
    //          Coopérer     Trahir
    // Coop     (3,3)        (0,5)
    // Trahir   (5,0)        (1,1)
    let pd = BimatrixGame::new(
        array![[3.0, 0.0], [5.0, 1.0]],  // Gains du joueur A
        array![[3.0, 5.0], [0.0, 1.0]],  // Gains du joueur B
    );

    // Trouver l'équilibre en stratégie dominante (les deux trahissent).
    if let Some(dom) = dominant_strategy_equilibrium(&pd) {
        println!("Stratégie dominante : A={:?}, B={:?}", dom.player_a, dom.player_b);
        // A=[0, 1], B=[0, 1]  -- les deux jouent Trahir (indice 1)
    }

    // --- Bataille des sexes ---
    //          Opéra     Football
    // Opéra    (3,2)     (0,0)
    // Football (0,0)     (2,3)
    let bos = BimatrixGame::new(
        array![[3.0, 0.0], [0.0, 2.0]],
        array![[2.0, 0.0], [0.0, 3.0]],
    );

    // L'énumération des supports trouve TOUS les équilibres de Nash (purs + mixtes).
    let equilibria = bos.support_enumeration();
    for (i, eq) in equilibria.iter().enumerate() {
        let pay_a = eq.expected_payoff_a(&bos);
        let pay_b = eq.expected_payoff_b(&bos);
        println!("EN {}: A={:?}, B={:?}, gains=({:.2}, {:.2})",
                 i, eq.player_a, eq.player_b, pay_a, pay_b);
    }
    // Trouve : (Opéra,Opéra), (Football,Football), et un équilibre mixte.

    // --- Meilleure réponse ---
    // Si B joue 60% Opéra, 40% Football :
    let b_strategy = Array1::from_vec(vec![0.6, 0.4]);
    let a_best = bos.best_response_a(&b_strategy);
    println!("Meilleure réponse de A à {:?}: {:?}", b_strategy, a_best);

    // --- Fictitious play (apprentissage itératif) ---
    let learned = fictitious_play(&pd, 1000);
    println!("Fictitious play a convergé vers : A={:?}, B={:?}",
             learned.player_a, learned.player_b);

    // --- Jeu à somme nulle (jeu de la pièce) ---
    let mp = BimatrixGame::zero_sum(array![[1.0, -1.0], [-1.0, 1.0]]);
    let eq = mp.support_enumeration();
    // Trouve l'unique EN mixte : les deux jouent 50/50.

    // --- Vérifier si un profil est un équilibre de Nash ---
    let profile = StrategyProfile {
        player_a: Array1::from_vec(vec![0.0, 1.0]),  // Trahir
        player_b: Array1::from_vec(vec![0.0, 1.0]),  // Trahir
    };
    println!("Est un EN ? {}", pd.is_nash_equilibrium(&profile, 1e-8));
}
```

### Résumé de l'API

| Fonction/Méthode | Ce qu'elle fait |
|----------------|--------------|
| `BimatrixGame::new(payoff_a, payoff_b)` | Créer un jeu à deux joueurs général |
| `BimatrixGame::zero_sum(payoff_a)` | Créer un jeu à somme nulle (gain de B = -gain de A) |
| `game.best_response_a(&strategy_b)` | Réponse optimale (possiblement mixte) de A face à B |
| `game.best_response_b(&strategy_a)` | Réponse optimale de B face à A |
| `game.support_enumeration()` | Trouver tous les équilibres de Nash (exact, pour les petits jeux) |
| `game.is_nash_equilibrium(&profile, tol)` | Vérifier si un profil de stratégie est un EN |
| `dominant_strategy_equilibrium(&game)` | Trouver l'EN en stratégie strictement dominante (s'il existe) |
| `fictitious_play(&game, iterations)` | Apprendre un EN approximatif par meilleure réponse itérée |

Voir l'exemple complet : [examples/game-theory/nash_equilibrium.rs](../../examples/game-theory/nash_equilibrium.rs)

## Quand l'utiliser

| Situation | Méthode | Pourquoi |
|-----------|--------|-----|
| Jeu 2x2, besoin de tous les équilibres | `support_enumeration()` | Exact, rapide pour les petits jeux |
| Petit jeu (jusqu'à ~5x5) | `support_enumeration()` | Enumère toutes les paires de supports |
| Grand jeu, EN approximatif | `fictitious_play()` | Passage à l'échelle, converge pour de nombreux jeux |
| Vérifier une stratégie connue | `is_nash_equilibrium()` | Vérification rapide |
| Jeu à somme nulle | `BimatrixGame::zero_sum()` + tout solveur | Structure plus simple, valeur unique garantie |
| Existence d'une stratégie dominante | `dominant_strategy_equilibrium()` | Vérification en O(n*m), pas d'énumération |

## Paramètres clés

| Paramètre | Type | Ce qu'il contrôle |
|-----------|------|-----------------|
| `payoff_a` | `Array2<f64>` | Matrice de gains du joueur ligne (lignes = stratégies de A, colonnes = stratégies de B) |
| `payoff_b` | `Array2<f64>` | Matrice de gains du joueur colonne (même dimensions) |
| `tolerance` | `f64` | Tolérance numérique pour les vérifications d'équilibre (1e-8 est typique) |
| `iterations` (fictitious play) | `usize` | Nombre de tours de meilleure réponse. Plus = plus proche de l'équilibre. |

## Pièges

1. **L'énumération des supports est exponentielle.** Elle énumère tous les sous-ensembles de stratégies, soit O(2^n * 2^m) pour un jeu n-par-m. Correct pour un jeu 5x5, impraticable pour 20x20. Utilisez fictitious play pour les grands jeux.

2. **La résolution d'EN mixtes est limitée au 2x2.** L'implémentation actuelle de `solve_support` gère les EN purs pour toute taille mais les EN mixtes uniquement pour les jeux 2x2. Les équilibres mixtes de plus grande taille nécessiteraient de la programmation linéaire (pas encore implémentée).

3. **Fictitious play peut ne pas converger.** La convergence est garantie pour les jeux à somme nulle et les jeux à EN unique. Pour des jeux comme celui de Shapley (3x3 avec meilleures réponses cycliques), l'algorithme peut boucler indéfiniment. Vérifiez la convergence en comparant les itérations successives.

4. **Les équilibres multiples sont fréquents.** La bataille des sexes a 3 équilibres ; les jeux de coordination peuvent en avoir beaucoup plus. L'algorithme les trouve tous (via l'énumération des supports), mais votre application doit décider *lequel* utiliser. C'est le **problème de la sélection d'équilibre** -- la théorie des jeux ne le résout pas pour vous.

5. **L'équilibre de Nash suppose la rationalité.** Les humains réels (et de nombreux agents IA) ne jouent pas les stratégies de Nash. Si vous modélisez des agents à rationalité limitée, envisagez plutôt les dynamiques évolutionnaires ou le raisonnement de niveau k.

## Pour aller plus loin

- **Dynamiques évolutionnaires :** Modéliser comment une *population* converge vers l'équilibre au fil du temps. Voir [Dynamiques évolutionnaires](./dynamiques-evolutionnaires.md).
- **Jeux coopératifs (valeur de Shapley) :** Lorsque les joueurs peuvent former des coalitions et partager les gains. Voir [Valeur de Shapley](./valeur-de-shapley.md).
- **Conception de mécanismes (enchères) :** Concevoir les *règles du jeu* pour que l'équilibre de Nash produise le résultat souhaité. Voir [Mécanismes d'enchères](./mecanismes-encheres.md).
- **Equilibre corrélé :** Une généralisation de Nash où un médiateur suggère des stratégies. Plus efficace à calculer (programmation linéaire).
- Lecture : Nisan et al., *Algorithmic Game Theory* (2007) -- la référence standard.
