# Recherche arborescente Monte Carlo (MCTS)

## Le problème

Vous développez une IA pour un jeu de plateau — Go, échecs ou un jeu de stratégie
personnalisé. L'arbre de jeu est immense : le Go compte environ 10^170 positions légales,
bien trop pour une exploration exhaustive. Vous ne pouvez pas non plus écrire une
fonction d'évaluation fiable, car le jeu est trop complexe pour des heuristiques simples.

Vous avez besoin d'un algorithme capable de jouer à bon niveau même sans voir l'arbre
entier, qui apprend quels coups sont prometteurs par échantillonnage, et qui s'améliore
avec davantage de temps de réflexion.

## L'intuition

Imaginez que vous déménagez dans une nouvelle ville et cherchez le meilleur restaurant.
Vous pourriez :

1. **Essayer un restaurant au hasard** chaque soir (échantillonnage purement aléatoire).
2. **Toujours retourner au meilleur** que vous avez trouvé (exploitation pure).
3. **Équilibrer les deux :** fréquenter surtout les restaurants que vous aimez déjà, mais
   en essayer un nouveau de temps en temps au cas où vous passeriez à côté de quelque
   chose.

MCTS fait l'option 3 pour les arbres de jeu. Il répète :

- **Sélection** d'une branche prometteuse de l'arbre (exploitation), avec un bonus pour
  les branches peu explorées (exploration).
- **Simulation** d'une partie aléatoire à partir de ce point jusqu'à la fin (rollout).
- **Mise à jour** des statistiques victoire/défaite en remontant l'arbre.

Après des milliers d'itérations, les statistiques convergent : les coups menant aux
victoires sont davantage visités, ceux menant aux défaites sont évités. L'algorithme n'a
pas besoin d'une fonction d'évaluation artisanale — les rollouts aléatoires *sont*
l'évaluation.

Le compromis exploration-exploitation est contrôlé par la **formule UCB1**, empruntée au
problème du bandit manchot. C'est la même mathématique qui aide les annonceurs en ligne
à décider quelle publicité afficher.

## Fonctionnement

Chaque itération comporte quatre phases :

### 1. Sélection

En partant de la racine, descendez dans l'arbre en choisissant l'enfant avec le score
UCB1 le plus élevé, jusqu'à atteindre un noeud avec des actions non essayées ou un état
terminal.

```
UCB1(enfant) = (victoires / visites) + c * sqrt(ln(visites_parent) / visites)
```

| Terme | Signification |
|-------|---------------|
| `victoires / visites` | **Exploitation :** récompense moyenne de cet enfant (entre 0 et 1) |
| `c * sqrt(ln(visites_parent) / visites)` | **Exploration :** bonus pour les enfants rarement visités |
| `c` | Constante d'exploration. Plus élevée = plus d'exploration. sqrt(2) ~ 1.41 est l'optimum théorique pour des récompenses dans [0,1]. |

**En clair :** choisissez l'enfant qui soit (a) a beaucoup gagné, soit (b) n'a pas été
souvent essayé. Le paramètre `c` contrôle le degré d'aventure de l'algorithme.

### 2. Expansion

Au noeud sélectionné, choisissez une action non essayée au hasard. Créez un nouveau noeud
enfant pour celle-ci.

### 3. Simulation (Rollout)

À partir du nouvel enfant, jouez une partie entièrement aléatoire jusqu'à un état
terminal. Enregistrez la récompense (1.0 = victoire, 0.0 = défaite, 0.5 = nulle).

### 4. Rétropropagation

Remontez du nouvel enfant jusqu'à la racine, en ajoutant 1 aux `visites` et la
récompense au `total_reward` à chaque ancêtre.

Après toutes les itérations, renvoyez l'**enfant le plus visité** de la racine comme
meilleur coup (le plus visité, et non celui à la meilleure moyenne, car le nombre de
visites est plus robuste).

## En Rust

Le crate `ix-search` fournit MCTS via le trait `MctsState` :

```rust
use ix_search::mcts::{MctsState, mcts_search};

// Un jeu simple de type Nim : les joueurs ajoutent 1 à 3 à tour de rôle,
// l'objectif est d'atteindre 21.
#[derive(Clone, Debug)]
struct NimState {
    count: i32,
    my_turn: bool,
}

impl MctsState for NimState {
    type Action = i32; // Combien ajouter (1, 2 ou 3)

    fn legal_actions(&self) -> Vec<i32> {
        if self.is_terminal() { return vec![]; }
        let max = 3.min(21 - self.count);
        (1..=max).collect()
    }

    fn apply(&self, action: &i32) -> Self {
        NimState {
            count: self.count + action,
            my_turn: !self.my_turn,
        }
    }

    fn is_terminal(&self) -> bool {
        self.count >= 21
    }

    fn reward(&self) -> f64 {
        // Le joueur qui vient de jouer pour atteindre 21 gagne.
        if self.count >= 21 {
            if self.my_turn { 0.0 } else { 1.0 }
        } else {
            0.5 // Nulle si non terminal
        }
    }
}

fn main() {
    let state = NimState { count: 0, my_turn: true };

    // Exécuter 5000 itérations avec constante d'exploration 1.41, graine 42.
    let best_action: Option<i32> = mcts_search(&state, 5000, 1.41, 42);

    match best_action {
        Some(a) => println!("MCTS recommande d'ajouter : {}", a),
        None => println!("Aucun coup légal"),
    }
}
```

### Référence de l'API

```rust
pub fn mcts_search<S: MctsState>(
    root: &S,           // État de jeu actuel (non consommé)
    iterations: usize,  // Nombre de cycles sélection-expansion-rollout
    exploration: f64,    // Constante d'exploration UCB1 c
    seed: u64,           // Graine RNG pour la reproductibilité
) -> Option<S::Action>  // Le meilleur coup, ou None si aucun coup légal
```

### Trait MctsState

| Méthode | Signature | Rôle |
|---------|-----------|------|
| `legal_actions` | `&self -> Vec<Action>` | Tous les coups valides depuis cet état |
| `apply` | `&self, &Action -> Self` | Renvoie l'état après une action (immuable) |
| `is_terminal` | `&self -> bool` | Vrai si la partie est terminée |
| `reward` | `&self -> f64` | Évaluation terminale : 1.0 = victoire, 0.0 = défaite, 0.5 = nulle |

## Quand l'utiliser

| Situation | Utiliser MCTS ? | Alternative |
|-----------|-----------------|-------------|
| Grand facteur de branchement (Go, stratégie complexe) | Oui | -- |
| Petit arbre de jeu (morpion) | Excessif | Minimax avec alpha-bêta |
| Besoin d'un jeu parfait déterministe | Non | Alpha-bêta en profondeur totale |
| Pas de bonne fonction d'évaluation | Oui | -- |
| Contrainte temps réel (< 1ms par coup) | Pas forcément | Alpha-bêta à faible profondeur |
| Jeu stochastique (dés, tirage de cartes) | Oui, gère naturellement l'aléa | Expectiminimax |
| Recherche mono-agent (cheminement) | Non | A* ou Q* |

## Paramètres clés

| Paramètre | Valeur typique | Effet |
|-----------|----------------|-------|
| `iterations` | 1 000 -- 100 000 | Plus = jeu plus fort, temps de réflexion plus long. La force croît environ comme sqrt(itérations). |
| `exploration` (c) | 1.41 (sqrt(2)) | Plus bas = plus d'exploitation (agressif, peut rater de bons coups). Plus haut = plus d'exploration (prudent, convergence plus lente). |
| `seed` | N'importe quel u64 | Détermine l'aléa des rollouts. Même graine = même résultat pour la reproductibilité. |

### Réglage de l'exploration

- **c = 0.5 :** Très exploitatif. Bien quand le jeu a peu de pièges (la plupart des coups
  se valent).
- **c = 1.41 :** Valeur par défaut équilibrée. Commencez par là.
- **c = 2.0+ :** Très exploratoire. À utiliser quand le jeu comporte des coups rares mais
  critiques faciles à rater.

## Pièges courants

1. **La récompense doit être vue du bon côté.** `reward()` est appelée sur un état
   terminal et rétropropagée à tous les ancêtres. Si votre jeu alterne les tours,
   assurez-vous que la récompense reflète la perspective du joueur *racine*, pas celle du
   joueur qui vient de jouer. L'implémentation d'ix-search rétropropage la même
   récompense à tous les ancêtres, alors concevez `reward()` en conséquence.

2. **Les rollouts aléatoires peuvent être faibles.** Dans les jeux où le jeu aléatoire
   diffère radicalement du bon jeu (les échecs par ex. — les coups aléatoires perdent du
   matériel instantanément), le MCTS brut avec rollouts aléatoires nécessitera beaucoup
   plus d'itérations. La solution est une meilleure politique de rollout (pas encore
   intégrée à ix-search, mais l'algorithme est modifiable).

3. **Limite de profondeur des rollouts.** L'implémentation plafonne les rollouts à
   500 coups pour éviter les parties infinies. Si votre jeu peut durer plus longtemps,
   le rollout renverra le `reward()` d'un état non terminal.

4. **La mémoire croît avec les itérations.** Chaque itération ajoute au plus un noeud
   à l'arbre. 100 000 itérations = jusqu'à 100 000 noeuds en mémoire. Pour des
   recherches très longues, cela peut devenir significatif.

5. **Non adapté à l'optimisation mono-agent.** MCTS est conçu pour les jeux adversariaux
   ou stochastiques. Pour le cheminement, utilisez A*. Pour l'optimisation, utilisez le
   recuit simulé ou les algorithmes évolutionnaires.

## Pour aller plus loin

- **RAVE (Rapid Action Value Estimation) :** Partage les statistiques à travers l'arbre
  pour les coups apparaissant à différentes positions. Accélère considérablement la
  convergence au Go.
- **Élargissement progressif :** Limite le facteur de branchement dans les espaces
  d'actions continus en n'étendant de nouveaux enfants que lorsque le nombre de visites
  dépasse un seuil.
- **Rollouts par réseau de neurones :** Remplacez les rollouts aléatoires par un réseau
  de politique entraîné (l'approche AlphaGo). Le trait `MctsState` rend cela simple —
  implémentez un rollout personnalisé qui interroge votre modèle.
- **Minimax + Alpha-Bêta :** Pour les jeux plus petits, entièrement observables et
  déterministes, alpha-bêta est plus fort par noeud. Voir
  [Minimax et Alpha-Bêta](./minimax-alpha-beta.md).
- Lire la synthèse : Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012).
