# Recherche Q* : heuristiques apprises

## Le problème

La recherche A* est puissante, mais elle nécessite une heuristique artisanale — une fonction qui estime la distance jusqu'à l'objectif. Pour une grille, la distance de Manhattan fonctionne. Pour un réseau routier, la distance à vol d'oiseau fonctionne. Mais qu'en est-il de :

- Un robot naviguant dans un entrepôt où certaines allées sont périodiquement bloquées ?
- Un optimiseur logistique routant des colis à travers un réseau à tarification dynamique ?
- Une IA de jeu où la « distance à la victoire » dépend d'une combinaison complexe de facteurs ?

Dans ces domaines, écrire une bonne heuristique admissible à la main est quelque part entre difficile et impossible. Vous avez des données (recherches passées, simulations, logs de jeu) mais pas de formule.

**La recherche Q*** remplace l'heuristique artisanale par une **fonction Q apprise** — un modèle entraîné pour estimer le coût restant depuis n'importe quel état. Le nom vient de la combinaison du Q-learning (apprentissage par renforcement) avec la recherche A*.

## L'intuition

L'A* standard demande à un expert humain : « À quelle distance cet état est-il de l'objectif ? » et utilise la réponse pour prioriser la recherche.

Q* pose la même question à un modèle entraîné. Le modèle a vu des milliers d'instances résolues et appris des motifs qu'un humain ne pourrait pas facilement formuler. Il peut se tromper parfois, mais en moyenne il guide la recherche bien mieux qu'une heuristique naïve.

L'astuce clé : l'A* standard évalue l'heuristique pour **chaque successeur**. Si un état a 100 successeurs, cela fait 100 appels à l'heuristique par expansion. Q* évalue l'heuristique **une seule fois par noeud expandé** et ajuste les estimations des successeurs en soustrayant le coût de l'étape. Dans les domaines avec de grands facteurs de branchement (beaucoup d'actions par état), cela réduit les évaluations heuristiques de plusieurs ordres de grandeur.

Pensez-y ainsi : si vous avez une estimation approximative de votre distance à l'objectif, et que vous faites un pas qui coûte 3, votre successeur est approximativement à « votre estimation moins 3 » de l'objectif. Pas besoin de réévaluer depuis zéro.

## Comment ça fonctionne

Q* utilise le même cadre `f(n) = g(n) + h(n)` que l'A*, avec deux changements :

1. **h(n) vient d'une fonction Q apprise** au lieu d'une formule artisanale.
2. **h(successeur) est approximé** comme `max(0, h(parent) - coût_étape)` au lieu de rappeler la fonction Q.

```
qstar(départ, Q) :
    h_départ = Q.estimate_cost_to_go(départ)
    open = file_priorité avec (départ, g=0, f=h_départ)

    tant que open n'est pas vide :
        courant = extraire le noeud de plus petit f
        si courant est objectif : retourner chemin

        h_courant = Q.estimate_cost_to_go(courant)   # Un appel Q par expansion

        pour (action, successeur, coût_étape) dans courant.successors() :
            new_g = g(courant) + coût_étape
            h_succ = max(0, h_courant - coût_étape)    # Pas d'appel Q pour les successeurs !
            f_succ = new_g + h_succ
            ajouter successeur à open avec f_succ
```

**En clair :** Demander au modèle « à quelle distance suis-je ? » une seule fois à chaque expansion de noeud. Pour chaque successeur, estimer « un peu plus près du coût que je viens de payer » sans redemander au modèle.

### Q* pondéré

Comme l'A* pondéré, on peut gonfler la fonction Q : `f(n) = g(n) + w * Q(n)`. Cela échange l'optimalité contre la vitesse :

- `w = 1.0` : Optimal (si Q est admissible).
- `w > 1.0` : Recherche plus rapide, coût du chemin au plus `w * optimal`.

### Q* à deux têtes

Pour les coûts d'actions non uniformes, une seule valeur Q par état ne suffit pas. La variante **à deux têtes** utilise :

- **Tête 1 :** Estime le coût de transition `c(état, action)`.
- **Tête 2 :** Estime le coût restant `h(successeur)`.

C'est plus précis mais nécessite d'appeler la fonction Q pour chaque successeur (perdant l'avantage d'un seul appel par expansion).

## En Rust

Le crate `ix-search` fournit Q* via le trait `QFunction` :

```rust
use ix_search::astar::SearchState;
use ix_search::qstar::{
    QFunction, TabularQ, qstar_search, qstar_weighted,
    qstar_two_head, compare_qstar_vs_astar, QStarResult,
};

// --- Utilisation de TabularQ pour les petits espaces d'états discrets ---

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GridPos {
    x: i32, y: i32, goal_x: i32, goal_y: i32, width: i32, height: i32,
}

impl SearchState for GridPos {
    type Action = (i32, i32);
    fn successors(&self) -> Vec<(Self::Action, Self, f64)> {
        let dirs = [(0,1),(0,-1),(1,0),(-1,0)];
        dirs.iter().filter_map(|&(dx,dy)| {
            let (nx, ny) = (self.x + dx, self.y + dy);
            if nx >= 0 && nx < self.width && ny >= 0 && ny < self.height {
                Some(((dx,dy), GridPos { x: nx, y: ny, ..*self }, 1.0))
            } else { None }
        }).collect()
    }
    fn is_goal(&self) -> bool { self.x == self.goal_x && self.y == self.goal_y }
}

fn main() {
    let start = GridPos { x: 0, y: 0, goal_x: 9, goal_y: 9, width: 20, height: 20 };

    // TabularQ : une fonction Q simple basée sur une table de hachage.
    // La valeur par défaut 10.0 signifie "les états inconnus sont estimés à 10 pas."
    let q = TabularQ::new(10.0);
    // En pratique, vous l'entraîneriez : q.set(état, valeur_apprise);

    // --- Recherche Q* de base ---
    let result: QStarResult<GridPos> = qstar_search(start.clone(), &q).unwrap();
    println!("Coût Q* : {}, noeuds expandés : {}, appels heuristiques : {}",
             result.cost, result.nodes_expanded, result.heuristic_calls);

    // --- Q* pondéré (plus rapide, sous-optimal borné) ---
    let fast = qstar_weighted(start.clone(), &q, 2.0).unwrap();
    println!("Coût Q* pondéré : {} (au plus 2x l'optimal)", fast.cost);

    // --- Comparer Q* vs A* sur le même problème ---
    let manhattan = |s: &GridPos| ((s.x - s.goal_x).abs() + (s.y - s.goal_y).abs()) as f64;
    let (qr, ar) = compare_qstar_vs_astar(start, &q, manhattan);
    println!("Q* expandés : {}, A* expandés : {}",
             qr.unwrap().nodes_expanded, ar.unwrap().nodes_expanded);
}
```

### Implémenter une fonction Q personnalisée

Pour les applications réelles, remplacez `TabularQ` par un réseau de neurones ou tout modèle appris :

```rust
use ix_search::qstar::QFunction;

struct MyNeuralQ {
    // Vos poids de modèle, runtime ONNX, etc.
}

impl QFunction<MyState> for MyNeuralQ {
    fn estimate_cost_to_go(&self, state: &MyState) -> f64 {
        // Exécuter l'inférence sur votre modèle
        self.model.predict(state.to_features())
    }

    // Optionnel : pour Q* à deux têtes
    fn estimate_transition_cost(&self, state: &MyState, action_idx: usize) -> Option<f64> {
        Some(self.model.predict_transition(state.to_features(), action_idx))
    }
}
```

### Champs de QStarResult

| Champ | Type | Description |
|-------|------|-------------|
| `path` | `Vec<S>` | États du départ à l'objectif |
| `actions` | `Vec<S::Action>` | Actions effectuées |
| `cost` | `f64` | Coût total du chemin |
| `nodes_expanded` | `usize` | Noeuds extraits de la liste ouverte |
| `nodes_generated` | `usize` | Noeuds successeurs créés |
| `heuristic_calls` | `usize` | Nombre d'évaluations de la fonction Q (la métrique d'efficacité clé) |

### Résumé de l'API

| Fonction | Optimal ? | Appels Q par expansion | Cas d'usage |
|----------|----------|----------------------|----------|
| `qstar_search(start, &q)` | Oui (si Q admissible) | 1 | Recherche standard avec heuristique apprise |
| `qstar_weighted(start, &q, w)` | Borné (coût <= w * optimal) | 1 | Recherche plus rapide avec sous-optimalité bornée |
| `qstar_two_head(start, &q)` | Dépend du modèle | 1 + par successeur | Coûts d'actions non uniformes |
| `qstar_bounded(start, &q, eps)` | coût <= (1+eps) * optimal | 1 | Recherche explicitement epsilon-bornée |
| `compare_qstar_vs_astar(start, &q, h)` | Les deux | -- | Comparaison de performance |

Voir l'exemple complet fonctionnel : [examples/search/astar_qstar.rs](../../examples/search/astar_qstar.rs)

## Quand l'utiliser

| Situation | Q* ou A* ? | Pourquoi |
|-----------|-----------|-----|
| Bonne heuristique artisanale disponible | A* | Plus simple, pas d'entraînement nécessaire |
| Pas de bonne heuristique, mais des données d'entraînement | Q* | La fonction Q apprise comble le vide |
| Grand facteur de branchement (beaucoup d'actions) | Q* | Un appel Q par expansion vs un par successeur |
| L'inférence du réseau de neurones est coûteuse | Q* (avec cache) | Minimise les appels d'inférence |
| Garantie d'optimalité nécessaire | Les deux (si heuristique/Q admissible) | Même garantie |
| Environnement dynamique (les coûts changent) | Q* | Ré-entraîner la fonction Q ; les heuristiques artisanales cassent |

## Paramètres clés

| Paramètre | Type | Ce qu'il contrôle |
|-----------|------|-----------------|
| `start` | `S: SearchState` | État initial |
| `q_function` | `&Q: QFunction<S>` | Estimateur de coût restant appris |
| `weight` | `f64` | Gonflement de l'heuristique (1.0 = optimal, >1 = plus rapide) |
| `epsilon` (borné) | `f64` | Borne de sous-optimalité (coût <= (1+eps) * optimal) |

### Paramètres de TabularQ

| Paramètre | Type | Ce qu'il contrôle |
|-----------|------|-----------------|
| `default` | `f64` | Estimation de coût pour les états non vus. Plus haut = plus conservateur (explore plus). Plus bas = plus agressif (peut manquer des chemins). |

## Pièges courants

1. **Fonction Q inadmissible.** Si la fonction Q surestime le coût restant, Q* peut renvoyer des chemins sous-optimaux. Contrairement aux heuristiques artisanales où l'admissibilité est prouvable, les modèles appris peuvent surestimer de façon imprévisible. Utilisez `qstar_weighted` avec une borne connue si l'optimalité compte.

2. **L'astuce de soustraction peut perdre de l'information.** L'approximation `h(successeur) = max(0, h(parent) - coût_étape)` fonctionne bien quand les coûts sont uniformes, mais peut être imprécise pour les coûts non uniformes. Utilisez `qstar_two_head` dans ces cas.

3. **TabularQ ne généralise pas.** `TabularQ` est une table de hachage — elle ne connaît que les états que vous avez explicitement insérés. Pour les grands espaces d'états ou les espaces continus, vous avez besoin d'un approximateur de fonction (réseau de neurones, forêt aléatoire, etc.) qui implémente `QFunction`.

4. **Entraîner la fonction Q est un problème séparé.** Q* suppose que vous avez déjà un modèle entraîné. L'entraîner correctement (à partir de logs de recherche, de RL ou d'apprentissage supervisé sur des instances résolues) est la partie difficile. Une fonction Q mal entraînée peut être pire que la distance de Manhattan.

5. **La valeur par défaut compte.** Dans `TabularQ::new(default)`, la valeur par défaut est l'estimation de coût pour tous les états inconnus. Trop basse (optimiste) = la recherche saute des états prometteurs. Trop haute (pessimiste) = la recherche explore trop largement. Commencez avec une valeur proche du coût restant moyen dans votre domaine.

## Pour aller plus loin

- **Entraîner une fonction Q :** Utilisez le Q-learning ou les retours Monte Carlo à partir d'instances résolues. Stockez des paires `(état, coût_restant_réel)` et entraînez un régresseur.
- **Intégration de réseau de neurones :** Implémentez `QFunction` avec un runtime ONNX ou une bibliothèque ML légère en Rust. L'inférence par batch sur plusieurs états pour l'efficacité GPU.
- **Comparaison avec A* :** Utilisez `compare_qstar_vs_astar` pour benchmarker votre heuristique apprise contre une heuristique artisanale. Suivez `heuristic_calls` comme métrique d'efficacité clé.
- **Recherche A* :** Pour les domaines où les heuristiques artisanales fonctionnent bien. Voir [recherche A*](./astar-search.md).
- Lecture : Agostinelli et al., « Solving the Rubik's Cube with Deep Reinforcement Learning and Search » (2019) — une application concrète d'heuristiques apprises avec A*.
