# Minimax et élagage Alpha-Bêta

## Le problème

Vous développez une IA pour le morpion, les dames ou les échecs. Deux joueurs jouent à
tour de rôle. L'un (le *maximiseur*) veut atteindre les états avec le score le plus
élevé ; l'autre (le *minimiseur*) veut le score le plus bas. Chaque joueur joue
parfaitement — il suppose que l'adversaire jouera lui aussi de manière optimale.

Vous avez besoin d'un algorithme qui répond à : « Quel est le meilleur coup que je puisse
jouer, en supposant que mon adversaire répondra toujours par *son* meilleur coup ? »

## L'intuition

Voyez cela comme une négociation. Vous proposez un accord (votre coup). Votre adversaire
choisira toujours la contre-offre la pire pour vous. Vous devez donc choisir la
proposition où, même après que l'adversaire a choisi sa meilleure réponse, vous êtes
encore dans la meilleure position possible.

**Minimax** formalise ceci en construisant l'arbre de jeu complet :

- À **vos** tours (maximiseur), vous choisissez l'enfant avec la valeur la **plus haute**.
- Aux tours de l'**adversaire** (minimiseur), il choisit l'enfant avec la valeur la **plus
  basse**.
- Aux états terminaux (fin de partie ou limite de profondeur), vous évaluez la position
  avec un score statique.

Le problème : l'arbre de jeu est immense. Aux échecs, il y a environ 35 coups légaux
par position et les parties durent 40+ coups, soit 35^40 noeuds. Impossible de tous les
explorer.

**L'élagage Alpha-Bêta** est l'idée clé : vous n'en avez pas besoin. Si vous savez déjà
que le maximiseur a un coup garantissant un score de 5, et que vous découvrez une branche
où le minimiseur peut forcer un score de 3, vous pouvez sauter le reste de cette branche.
Le maximiseur n'irait jamais là.

Alpha-bêta évalue les mêmes coups que minimax mais **élague** (saute) les branches qui ne
peuvent pas affecter la décision finale. Dans le meilleur cas (coups parfaitement
ordonnés), il n'explore que la racine carrée du nombre de noeuds que minimax aurait
parcourus.

**Expectiminimax** étend ceci aux jeux avec des éléments de hasard (lancers de dés,
tirages de cartes). Les noeuds de hasard calculent la moyenne pondérée sur tous les
résultats aléatoires.

## Fonctionnement

### Minimax

```
minimax(state, depth):
    if depth == 0 or state is terminal:
        return evaluate(state)
    if maximizer's turn:
        return max over children of minimax(child, depth - 1)
    else:
        return min over children of minimax(child, depth - 1)
```

**En clair :** regardez `depth` coups à l'avance, supposez que les deux joueurs jouent de
façon optimale, et faites remonter le meilleur score atteignable.

### Alpha-Bêta

Ajoutez deux bornes qui suivent ce que chaque joueur peut déjà garantir :

- **alpha :** le meilleur score que le maximiseur peut garantir jusqu'ici (commence à
  moins l'infini).
- **bêta :** le meilleur score que le minimiseur peut garantir jusqu'ici (commence à
  plus l'infini).

```
alpha_beta(state, depth, alpha, beta):
    if depth == 0 or state is terminal:
        return evaluate(state)
    if maximizer's turn:
        for each child:
            value = alpha_beta(child, depth-1, alpha, beta)
            alpha = max(alpha, value)
            if alpha >= beta: break   # Coupure bêta — le minimiseur éviterait ceci
        return alpha
    else:
        for each child:
            value = alpha_beta(child, depth-1, alpha, beta)
            beta = min(beta, value)
            if alpha >= beta: break   # Coupure alpha — le maximiseur éviterait ceci
        return beta
```

**En clair :** « J'ai déjà un accord valant 5. Cette branche montre que l'adversaire
peut forcer 3 ici. Inutile de chercher plus loin — je ne choisirais jamais cette
branche. »

### Expectiminimax

Aux noeuds de hasard, calculez :

```
expected_value = sum over outcomes of: probability * minimax(outcome_state, depth - 1)
```

**En clair :** s'il y a un lancer de dé, pondérez chaque résultat par sa probabilité et
prenez la moyenne.

## En Rust

Le crate `ix-search` modélise les jeux adversariaux via le trait `GameState` :

```rust
use ix_search::adversarial::{
    GameState, AdversarialResult, minimax, alpha_beta, expectiminimax,
    StochasticGameState,
};

// Un jeu de nombres : deux joueurs ajoutent +1 ou -1.
// Le maximiseur veut une valeur élevée.
#[derive(Clone, Debug)]
struct NumberGame {
    value: i32,
    max_turn: bool,
    turns_left: usize,
}

impl GameState for NumberGame {
    type Move = i32;

    fn legal_moves(&self) -> Vec<i32> {
        if self.is_terminal() { vec![] } else { vec![1, -1] }
    }

    fn apply_move(&self, m: &i32) -> Self {
        NumberGame {
            value: self.value + m,
            max_turn: !self.max_turn,
            turns_left: self.turns_left - 1,
        }
    }

    fn is_terminal(&self) -> bool {
        self.turns_left == 0 || self.value <= 0 || self.value >= 10
    }

    fn is_maximizer_turn(&self) -> bool {
        self.max_turn
    }

    fn evaluate(&self) -> f64 {
        self.value as f64
    }
}

fn main() {
    let state = NumberGame { value: 5, max_turn: true, turns_left: 6 };

    // --- Minimax complet (exhaustif) ---
    let mm: AdversarialResult<i32> = minimax(&state, 6);
    println!("Minimax : meilleur_coup={:?}, valeur={}, noeuds={}",
             mm.best_move, mm.value, mm.nodes_evaluated);

    // --- Alpha-Bêta (même résultat, moins de noeuds) ---
    let ab: AdversarialResult<i32> = alpha_beta(&state, 6);
    println!("Alpha-Bêta : meilleur_coup={:?}, valeur={}, noeuds={}",
             ab.best_move, ab.value, ab.nodes_evaluated);
    // ab.value == mm.value (toujours)
    // ab.nodes_evaluated <= mm.nodes_evaluated (généralement beaucoup moins)
}
```

### Champs de AdversarialResult

| Champ | Type | Description |
|-------|------|-------------|
| `best_move` | `Option<M>` | Le coup recommandé (None si terminal ou aucun coup légal) |
| `value` | `f64` | La valeur minimax de la position |
| `nodes_evaluated` | `usize` | Total de noeuds visités pendant la recherche |

### Trait GameState

| Méthode | Signature | Rôle |
|---------|-----------|------|
| `legal_moves` | `&self -> Vec<Move>` | Tous les coups valides depuis cette position |
| `apply_move` | `&self, &Move -> Self` | État après un coup (immuable) |
| `is_terminal` | `&self -> bool` | Partie terminée ? |
| `is_maximizer_turn` | `&self -> bool` | À qui le tour ? |
| `evaluate` | `&self -> f64` | Évaluation statique. Positif = favorable au maximiseur. |

### Expectiminimax

Pour les jeux avec des noeuds de hasard (dés, cartes), implémentez
`StochasticGameState` :

```rust
impl StochasticGameState for MyGameState {
    fn is_chance_node(&self) -> bool {
        self.next_is_dice_roll
    }

    fn chance_outcomes(&self) -> Vec<(f64, Self)> {
        // Renvoie des paires (probabilité, état_résultant)
        vec![
            (1.0 / 6.0, self.with_roll(1)),
            (1.0 / 6.0, self.with_roll(2)),
            // ...
        ]
    }
}

let result = expectiminimax(&state, 8);
```

### Également disponible : Negamax

`negamax(&state, depth)` est une implémentation simplifiée d'alpha-bêta qui utilise la
négation du score au lieu de branches max/min séparées. Produit les mêmes résultats, avec
un code interne plus propre.

## Quand l'utiliser

| Situation | Algorithme | Pourquoi |
|-----------|-----------|----------|
| Petit arbre de jeu (morpion) | `minimax` | Résout le jeu complètement |
| Arbre moyen (dames, Puissance 4) | `alpha_beta` | Élague assez pour atteindre une profondeur utile |
| Jeu déterministe, jeu parfait requis | `alpha_beta` | Valeur minimax exacte avec élagage |
| Jeu avec dés/cartes (Backgammon) | `expectiminimax` | Gère correctement les noeuds de hasard |
| Immense arbre de jeu (Go, stratégie complexe) | MCTS à la place | Alpha-bêta ne peut pas atteindre une profondeur utile |
| Temps réel (< 1ms) | `alpha_beta` à faible profondeur | Déterministe, temps prévisible |

## Paramètres clés

| Paramètre | Type | Ce qu'il contrôle |
|-----------|------|-------------------|
| `state` | `&S: GameState` | Position de jeu actuelle (passée par référence, non consommée) |
| `depth` | `usize` | Combien de coups chercher à l'avance. Plus profond = plus fort mais exponentiellement plus lent. |
| `alpha`, `beta` (alpha_beta uniquement) | `f64` | Bornes initiales. Utilisez `f64::NEG_INFINITY` et `f64::INFINITY` à la racine. La fonction `alpha_beta` s'en charge pour vous. |

### Choisir la profondeur

La relation entre profondeur et noeuds est exponentielle : `noeuds ~ b^d` où `b` est le
facteur de branchement.

| Jeu | Facteur de branchement | Profondeur 4 | Profondeur 6 | Profondeur 8 |
|-----|------------------------|--------------|--------------|--------------|
| Morpion | ~4 | 256 | 4K | 65K |
| Dames | ~8 | 4K | 260K | 17M |
| Échecs | ~35 | 1.5M | 1.8G | trop |

L'élagage alpha-bêta prend approximativement la racine carrée : la profondeur 8 avec
élagage coûte environ autant que la profondeur 4 sans.

## Pièges courants

1. **evaluate n'est pas evaluate(terminal).** La méthode `evaluate()` est appelée aux
   *feuilles* de la recherche — à la fois les états terminaux et les états à la limite de
   profondeur. Pour les états terminaux, renvoyez le résultat réel. Pour les feuilles non
   terminales, renvoyez votre meilleure estimation statique. Une mauvaise fonction
   d'évaluation fera mal jouer même une recherche profonde.

2. **L'ordre des coups compte pour alpha-bêta.** Alpha-bêta élague au mieux quand les
   bons coups sont essayés en premier. Si vous examinez le pire coup en premier, aucun
   élagage ne se produit et alpha-bêta se dégrade en minimax pur. Triez `legal_moves()`
   par une heuristique rapide (par ex. captures d'abord aux échecs) pour un gain massif.

3. **La profondeur est primordiale.** Passer de la profondeur 4 à 6 peut transformer un
   joueur faible en joueur fort. Si votre recherche est lente, profilez `successors` et
   `evaluate` — ce sont les chemins chauds.

4. **Expectiminimax ne peut pas élaguer aussi agressivement.** Les noeuds de hasard
   empêchent les bornes nettes dont alpha-bêta dépend. Attendez-vous à ce
   qu'expectiminimax soit significativement plus lent qu'alpha-bêta à la même profondeur.

5. **Effet d'horizon.** À la limite de profondeur, l'IA ne voit pas ce qui suit. Un coup
   qui semble bon à profondeur 6 peut être terrible à profondeur 7 (par ex. retarder une
   perte inévitable). La recherche de quiescence (chercher plus profondément dans les
   positions « instables ») est la solution standard, mais n'est pas encore intégrée à
   ix-search.

## Pour aller plus loin

- **Approfondissement itératif :** Cherchez à profondeur 1, puis 2, puis 3, etc., en
  utilisant le temps comme budget au lieu de la profondeur. Chaque itération réutilise
  l'ordre des coups de la précédente.
- **Table de transposition :** Mettez en cache les positions évaluées (par hachage) pour
  éviter de ré-explorer la même position atteinte par des ordres de coups différents.
- **MCTS :** Pour les jeux où alpha-bêta ne peut pas atteindre une profondeur utile (Go,
  jeux de stratégie complexes). Voir [MCTS](./mcts.md).
- **Negamax :** La fonction `negamax(&state, depth)` d'ix-search implémente alpha-bêta
  via la négation du score, ce qui est plus propre pour les jeux à somme nulle à deux
  joueurs.
- Lire : Knuth et Moore, "An Analysis of Alpha-Beta Pruning" (1975).
