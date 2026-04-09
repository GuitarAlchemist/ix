# Q-Learning

## Le problème

Vous construisez un agent IA pour un jeu sur grille. L'agent démarre dans un coin et doit atteindre l'objectif dans le coin opposé, en évitant les obstacles. Contrairement aux bandits (où l'on choisit simplement entre des bras), ici chaque action change l'*état* du monde -- se déplacer à gauche vous met dans une cellule différente, où un ensemble d'actions différent est disponible et une récompense différente vous attend.

Scénarios concrets :
- **Navigation robotique :** Un robot d'entrepôt apprend à naviguer du quai de chargement à une étagère sans heurter les autres robots.
- **IA de jeu :** Un PNJ apprend à jouer au morpion, aux ouvertures d'échecs ou à des jeux vidéo simples.
- **Gestion des ressources :** Un thermostat apprend quand chauffer, climatiser ou rester en veille pour minimiser le coût énergétique tout en maintenant le confort.
- **Routage réseau :** Un routeur apprend quel chemin emprunter pour acheminer les paquets afin de minimiser la latence.

## L'intuition

Imaginez que vous déménagez dans une nouvelle ville et que vous cherchez le trajet le plus rapide vers le travail. Le premier jour, vous choisissez au hasard. Certains trajets sont terribles, d'autres corrects. Chaque jour, vous notez la durée de chaque trajet. Avec le temps, vous cessez d'essayer les trajets qui prennent systématiquement 45 minutes et vous privilégiez celui qui prend 20 minutes.

Mais voici la subtilité : chaque trajet n'est pas qu'une *seule* décision -- c'est une *séquence* de virages. La valeur de tourner à gauche au premier carrefour dépend des options disponibles après ce virage. Le Q-learning capture cela : il attribue une valeur non pas simplement à « tourner à gauche » en général, mais à « tourner à gauche *quand on est au carrefour n.5* ». C'est cette paire état-action qui le rend plus puissant que les bandits.

## Comment ça fonctionne

Le Q-learning maintient une **table Q** : une matrice où les lignes sont les états, les colonnes les actions, et chaque cellule Q(s, a) estime la récompense future totale de prendre l'action a dans l'état s.

### La règle de mise à jour de la table Q

Après avoir pris l'action a dans l'état s, reçu la récompense r et atterri dans l'état s' :

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

**En clair :** Comparer ce que vous *attendiez* (Q(s, a)) avec ce que vous avez *réellement obtenu* (r plus le mieux que vous puissiez faire depuis le prochain état). Si la réalité était meilleure que prévu, ajuster l'estimation vers le haut. Si c'était pire, l'ajuster vers le bas.

Décomposition des termes :
- **alpha** (taux d'apprentissage, 0 < alpha <= 1) : Combien faire confiance aux nouvelles informations vs. les anciennes estimations. alpha = 0.1 signifie « mettre à jour lentement », alpha = 1.0 signifie « écraser complètement ».
- **gamma** (facteur d'actualisation, 0 <= gamma < 1) : Combien valoriser les récompenses futures. gamma = 0.99 signifie « les récompenses futures valent presque autant que les immédiates ». gamma = 0 signifie « ne se soucier que de la prochaine étape ».
- **max_a' Q(s', a')** : La meilleure valeur atteignable depuis le prochain état -- c'est l'« anticipation » qui rend le Q-learning puissant.

### La boucle d'apprentissage

```
Initialize Q(s, a) = 0 for all states and actions
Repeat for each episode:
    s = starting state
    While s is not terminal:
        a = choose action (epsilon-greedy on Q)
        Take action a, observe reward r and next state s'
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        s = s'
```

**En clair :** Jouer la partie de manière répétée. Chaque fois que vous faites un pas, regarder ce qui s'est passé et mettre à jour vos notes. Au fil de nombreuses parties, la table Q converge vers les vraies valeurs, et la politique gloutonne (toujours choisir argmax Q) devient optimale.

## En Rust

> **Note :** Le Q-learning dans `ix-rl` est actuellement une ébauche/TODO. Les traits `Environment` et `Agent` sont définis mais l'implémentation tabulaire du Q-learning n'est pas encore construite. Ci-dessous, l'API prévue basée sur les définitions de traits, ainsi qu'une implémentation manuelle utilisable dès maintenant.

### Les définitions de traits (disponibles maintenant)

```rust
use ix_rl::traits::{Environment, Agent};

// Le trait Environment définit le monde avec lequel l'agent interagit
// trait Environment {
//     type State: Clone;
//     type Action: Clone;
//     fn reset(&mut self) -> Self::State;
//     fn step(&mut self, action: &Self::Action) -> (Self::State, f64, bool);
//     fn actions(&self) -> Vec<Self::Action>;
// }

// Le trait Agent définit l'apprenant
// trait Agent<E: Environment> {
//     fn select_action(&self, state: &E::State) -> E::Action;
//     fn update(&mut self, state: &E::State, action: &E::Action,
//               reward: f64, next_state: &E::State, done: bool);
// }
```

### Q-Learning manuel (en Rust standard)

```rust
use std::collections::HashMap;

struct QLearner {
    q_table: HashMap<(usize, usize), f64>,  // (état, action) -> valeur
    alpha: f64,    // taux d'apprentissage
    gamma: f64,    // facteur d'actualisation
    epsilon: f64,  // taux d'exploration
    n_actions: usize,
}

impl QLearner {
    fn new(n_actions: usize, alpha: f64, gamma: f64, epsilon: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            alpha, gamma, epsilon, n_actions,
        }
    }

    fn q(&self, state: usize, action: usize) -> f64 {
        *self.q_table.get(&(state, action)).unwrap_or(&0.0)
    }

    fn best_action(&self, state: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| self.q(state, a)
                .partial_cmp(&self.q(state, b)).unwrap())
            .unwrap()
    }

    fn update(&mut self, s: usize, a: usize, r: f64, s_next: usize, done: bool) {
        let max_next = if done {
            0.0
        } else {
            (0..self.n_actions)
                .map(|a| self.q(s_next, a))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let old = self.q(s, a);
        let new_val = old + self.alpha * (r + self.gamma * max_next - old);
        self.q_table.insert((s, a), new_val);
    }
}
```

### En rapport : la recherche Q* dans ix-search

Bien que le Q-learning tabulaire ne soit pas encore implémenté, le crate `ix-search` offre la **recherche Q\*** -- un algorithme de recherche de chemin qui utilise des Q-valeurs apprises comme heuristiques pour A\*. C'est un concept différent (recherche, pas apprentissage) mais qui partage l'idée d'utiliser des Q-valeurs pour guider les décisions :

```rust
use ix_search::qstar::{qstar_search, TabularQ};

// Q* utilise un trait QFunction pour estimer le coût restant
let mut q = TabularQ::new(10.0);  // valeur heuristique par défaut
let result = qstar_search(start_state, &q);
```

## Quand l'utiliser

| Approche | Idéale quand | Gère les décisions séquentielles | Espace d'états |
|---|---|---|---|
| **Bandits multi-bras** | Décision unique, pas de changement d'état | Non | N/A |
| **Q-learning tabulaire** | Espaces état/action petits et discrets | Oui | < ~10 000 états |
| **Deep Q-learning (DQN)** | Espaces d'états grands ou continus | Oui | Illimité (nécessite un réseau de neurones) |
| **Recherche Q\*** | Objectif connu, besoin du chemin optimal | Oui (recherche, pas apprentissage) | Structuré en graphe |

**Utilisez le Q-learning quand :**
- L'espace d'états est assez petit pour être énuméré (mondes en grille, jeux de plateau, contrôle simple).
- Vous pouvez simuler l'environnement à faible coût (de nombreux épisodes sont nécessaires).
- Vous voulez un algorithme off-policy (apprend la politique optimale même en explorant).

**N'utilisez pas le Q-learning quand :**
- L'espace d'états est continu ou très grand (utilisez l'approximation de fonction / DQN à la place).
- Vous avez besoin de décisions en temps réel sans phase d'entraînement.
- Les actions ont des valeurs continues (utilisez les méthodes de gradient de politique à la place).

## Paramètres clés

| Paramètre | Plage typique | Effet |
|---|---|---|
| `alpha` (taux d'apprentissage) | 0.01 -- 0.5 | Plus élevé = apprentissage plus rapide, mais plus de bruit. Plus bas = stable mais lent |
| `gamma` (facteur d'actualisation) | 0.9 -- 0.99 | Plus élevé = valorise les récompenses à long terme. Plus bas = myope (glouton) |
| `epsilon` (taux d'exploration) | 0.05 -- 0.3 | Plus élevé = plus d'exploration. Souvent décroissant au fil du temps |
| `episodes` | 100 -- 100 000 | Plus d'épisodes = meilleure convergence, mais plus de calcul |

## Pièges

1. **La table Q explose avec la taille de l'espace d'états.** Une grille 100x100 avec 4 actions nécessite 40 000 entrées. Un échiquier a ~10^43 états -- le Q-learning tabulaire ne peut pas le gérer. Utilisez l'approximation de fonction pour les grands problèmes.

2. **Convergence lente.** Le Q-learning doit visiter chaque paire (état, action) de nombreuses fois. Si certains états sont rarement atteints, leurs Q-valeurs restent à zéro. Envisagez une initialisation optimiste (commencer les Q-valeurs hautes pour encourager l'exploration).

3. **Environnements non stationnaires.** Si l'environnement change au fil du temps, les anciennes Q-valeurs deviennent obsolètes. Utilisez un taux d'apprentissage plus élevé ou réinitialisez périodiquement la table Q.

4. **Biais de surestimation.** Le `max` dans la règle de mise à jour provoque un biais systématique vers le haut. Le **Double Q-learning** corrige cela en maintenant deux tables Q et en utilisant l'une pour sélectionner et l'autre pour évaluer.

5. **Off-policy vs. on-policy.** Le Q-learning est off-policy (apprend la politique gloutonne tout en suivant une politique exploratoire). **SARSA** est la variante on-policy : il met à jour en utilisant l'action réellement prise, pas la meilleure action. SARSA est plus sûr dans les environnements où l'exploration est dangereuse.

## Pour aller plus loin

- **Les bandits comme cas particulier :** Le Q-learning avec 1 état et sans actualisation se réduit à la règle de mise à jour des bandits. Voir [bandits-multi-bras.md](./bandits-multi-bras.md).
- **Stratégies d'exploration :** L'exploration epsilon-greedy du Q-learning est la même que pour les bandits. Voir [exploration-vs-exploitation.md](./exploration-vs-exploitation.md) pour les alternatives.
- **Recherche Q\* :** Si vous avez une Q-fonction apprise et que vous voulez trouver des chemins optimaux, voir le module `qstar` du crate `ix-search`, qui utilise les Q-valeurs comme heuristiques A\*.
- **Deep Q-Networks (DQN) :** Remplacer la table Q par un réseau de neurones d'`ix-nn`. Pas encore intégré, mais l'architecture est : état -> `ix-nn::Layer` -> Q-valeurs pour chaque action.
- **Méthodes de gradient de politique :** Au lieu d'apprendre une fonction de valeur, apprendre directement une politique. Une approche fondamentalement différente, pas encore dans ix.
