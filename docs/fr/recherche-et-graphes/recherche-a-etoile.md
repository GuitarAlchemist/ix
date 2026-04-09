# Recherche A* et variantes

## Le problème

Vous développez un jeu vidéo dans lequel les ennemis doivent naviguer dans un donjon en
tuiles pour poursuivre le joueur. Le donjon comporte des murs, des couloirs et des salles
ouvertes. Vous avez besoin du chemin le plus court entre l'ennemi et le joueur — et il
doit être calculé en millisecondes, pas en secondes, car il y a des dizaines d'ennemis
à l'écran simultanément.

Ou bien : vous planifiez les itinéraires de camions de livraison à travers une ville.
Chaque segment de route a un temps de parcours. Vous voulez l'itinéraire le plus rapide
de l'entrepôt à chaque client, et vous voulez que l'algorithme écarte rapidement les
directions manifestement mauvaises (s'éloigner de la destination) plutôt que d'explorer
chaque route de la ville.

Ce sont des instances du **problème du plus court chemin** sur un graphe où l'on dispose
d'une bonne estimation (heuristique) de la distance de chaque noeud au but.

## L'intuition

Imaginez que vous êtes perdu dans un labyrinthe. À chaque intersection, vous disposez de
deux informations :

1. **La distance déjà parcourue** depuis l'entrée (le *coût g*).
2. **Une estimation approximative** de la distance restante jusqu'à la sortie — par
   exemple, la distance à vol d'oiseau à travers les murs (l'*heuristique h*).

Une approche naïve (algorithme de Dijkstra) n'utilise que (1) : elle s'étend uniformément
dans toutes les directions comme une onde circulaire. A* ajoute (2) : il dit « développe
le noeud qui semble le moins coûteux au total — celui où *distance parcourue + estimation
restante* est le plus petit ». Cela concentre la recherche vers l'objectif, comme un
faisceau de lampe torche au lieu d'un projecteur à 360 degrés.

La garantie : tant que votre estimation **ne surestime jamais** la distance réelle
(propriété d'*admissibilité*), A* trouvera le vrai plus court chemin. Si l'estimation
est aussi *cohérente* (l'inégalité triangulaire est respectée), A* n'aura jamais besoin
de ré-expanser un noeud.

**A* pondéré** relâche l'optimalité au profit de la vitesse : il gonfle l'heuristique
d'un facteur `w`. Le chemin trouvé coûte au plus `w` fois le coût optimal, mais la
recherche peut être considérablement plus rapide.

**Recherche gloutonne (Best-First)** abandonne complètement le coût parcouru et ne suit
que l'heuristique. C'est rapide, mais le chemin trouvé peut être arbitrairement mauvais.

## Fonctionnement

A* maintient deux structures de données :

- Une **liste ouverte** (file de priorité) de noeuds à explorer, triée par `f(n)`.
- Un **ensemble fermé** de noeuds déjà développés.

Pour chaque noeud `n` :

```
f(n) = g(n) + h(n)
```

| Symbole | Signification |
|---------|---------------|
| `g(n)` | Coût réel du chemin le moins cher connu du **départ** à `n` |
| `h(n)` | Estimation heuristique du coût de `n` au **but** |
| `f(n)` | Coût total estimé du chemin le moins cher passant par `n` |

**En clair :** g est ce que vous avez dépensé, h est ce que vous prévoyez de dépenser,
et f est le budget total. A* choisit toujours le noeud avec le budget total le plus bas.

L'algorithme :

1. Placer le noeud de départ dans la liste ouverte avec `g = 0`, `f = h(départ)`.
2. Extraire le noeud avec le `f` le plus bas de la liste ouverte.
3. Si c'est le but, reconstruire le chemin et le renvoyer.
4. Sinon, le développer : pour chaque successeur, calculer `new_g = g(courant) + coût_pas`.
   Si `new_g` est meilleur que tout chemin précédemment connu vers ce successeur, le
   mettre à jour et l'ajouter à la liste ouverte.
5. Répéter jusqu'à trouver le but ou que la liste ouverte soit vide (pas de chemin).

**A* pondéré** utilise `f(n) = g(n) + w * h(n)`. Un `w` plus grand rend l'heuristique
plus dominante, poussant la recherche en ligne droite vers le but.

**Recherche gloutonne** utilise `f(n) = h(n)` — elle ignore complètement le coût du
chemin.

## En Rust

Le crate `ix-search` modélise les problèmes de recherche via le trait `SearchState` :

```rust
use ix_search::astar::{SearchState, astar, weighted_astar, greedy_best_first, SearchResult};

// Définissez votre état en implémentant SearchState.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GridPos {
    x: i32,
    y: i32,
    goal_x: i32,
    goal_y: i32,
    width: i32,
    height: i32,
}

impl SearchState for GridPos {
    // Les actions sont des tuples de direction (dx, dy).
    type Action = (i32, i32);

    // Renvoie des triplets (action, état_successeur, coût_pas).
    fn successors(&self) -> Vec<(Self::Action, Self, f64)> {
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        dirs.iter()
            .filter_map(|&(dx, dy)| {
                let nx = self.x + dx;
                let ny = self.y + dy;
                if nx >= 0 && nx < self.width && ny >= 0 && ny < self.height {
                    Some(((dx, dy), GridPos { x: nx, y: ny, ..*self }, 1.0))
                } else {
                    None
                }
            })
            .collect()
    }

    fn is_goal(&self) -> bool {
        self.x == self.goal_x && self.y == self.goal_y
    }
}

fn main() {
    let start = GridPos { x: 0, y: 0, goal_x: 9, goal_y: 9, width: 20, height: 20 };

    // Heuristique de distance de Manhattan (admissible pour les grilles 4-connexes).
    let h = |s: &GridPos| ((s.x - s.goal_x).abs() + (s.y - s.goal_y).abs()) as f64;

    // --- A* standard (optimal) ---
    let result: SearchResult<GridPos> = astar(start.clone(), h).unwrap();
    println!("Coût A* : {}, noeuds développés : {}", result.cost, result.nodes_expanded);
    // result.path    -- Vec<GridPos> du départ au but
    // result.actions -- Vec<(i32,i32)> directions prises
    // result.cost    -- coût total du chemin (18.0 pour une grille 9+9)

    // --- A* pondéré (plus rapide, sous-optimal borné) ---
    let fast = weighted_astar(start.clone(), h, 1.5).unwrap();
    println!("Coût WA* : {}, noeuds développés : {}", fast.cost, fast.nodes_expanded);
    // fast.cost <= 1.5 * result.cost  (garanti)

    // --- Recherche gloutonne (rapide, pas de garantie d'optimalité) ---
    let greedy = greedy_best_first(start, h).unwrap();
    println!("Coût glouton : {}, noeuds développés : {}", greedy.cost, greedy.nodes_expanded);
}
```

### Champs de SearchResult

| Champ | Type | Description |
|-------|------|-------------|
| `path` | `Vec<S>` | Séquence d'états du départ au but |
| `actions` | `Vec<S::Action>` | Actions effectuées à chaque étape |
| `cost` | `f64` | Coût total du chemin |
| `nodes_expanded` | `usize` | Noeuds extraits de la liste ouverte |
| `nodes_generated` | `usize` | Total de noeuds successeurs créés |

Voir l'exemple complet : [examples/search/astar_qstar.rs](../../examples/search/astar_qstar.rs)

## Quand l'utiliser

| Algorithme | Optimal ? | Vitesse | Mémoire | Idéal pour |
|------------|-----------|---------|---------|-----------|
| `astar` | Oui (avec h admissible) | Rapide avec une bonne h | O(noeuds) | La plupart des problèmes de cheminement |
| `weighted_astar` (w > 1) | Non, mais borné (coût <= w * optimal) | Plus rapide | O(noeuds) | Jeux temps réel, chemins « suffisamment bons » |
| `greedy_best_first` | Non | Le plus rapide (souvent) | O(noeuds) | Approximation rapide, quand l'optimalité n'importe pas |
| `uniform_cost_search` | Oui | Le plus lent (A* avec h=0) | O(noeuds) | Quand on n'a pas d'heuristique |
| `bidirectional_astar` | Oui | Plus rapide sur grands graphes symétriques | 2x mémoire | Grandes cartes avec départ et but connus |

## Paramètres clés

| Paramètre | Type | Ce qu'il contrôle |
|-----------|------|-------------------|
| `start` | `S: SearchState` | L'état initial. Doit implémenter `Clone + Eq + Hash`. |
| `heuristic` | `Fn(&S) -> f64` | Coût estimé jusqu'au but. Doit être >= 0. **Admissible** signifie qu'elle ne surestime jamais. |
| `weight` (A* pondéré) | `f64` | Inflation de l'heuristique. 1.0 = A* standard. 2.0 = chemins au plus 2x l'optimal mais recherche beaucoup plus rapide. |

### Choisir une heuristique

- **Grille (4-connexe) :** Distance de Manhattan `|dx| + |dy|`
- **Grille (8-connexe, diagonales) :** Distance de Tchebychev `max(|dx|, |dy|)`
- **Espace euclidien :** Distance en ligne droite `sqrt(dx^2 + dy^2)`
- **Réseaux routiers :** Distance de Haversine (orthodromie)
- **Sans connaissance du domaine :** Utiliser `|_| 0.0` (revient à Dijkstra)

## Pièges courants

1. **Une heuristique inadmissible casse l'optimalité.** Si votre heuristique surestime ne
   serait-ce qu'une fois, A* peut renvoyer un chemin sous-optimal. A* pondéré est la
   manière rigoureuse d'échanger l'optimalité contre la vitesse — utilisez-le au lieu
   d'une heuristique approximative.

2. **Mémoire.** A* stocke chaque noeud généré. Sur une grille 10 000 x 10 000 sans
   obstacles, cela peut représenter des millions de noeuds. Envisagez IDA* (A* par
   approfondissement itératif) pour les environnements à mémoire limitée, ou A* pondéré
   pour réduire la taille de la frontière.

3. **Collisions de hachage.** Le trait `SearchState` exige `Hash + Eq`. Si le hachage de
   vos états est lent ou sujet aux collisions, les recherches dans l'ensemble fermé
   deviennent un goulet d'étranglement. Gardez les états petits et facilement hachables.

4. **Successeurs coûteux.** `successors()` est appelé une fois par expansion. Si la
   génération des successeurs est coûteuse (simulation physique, par ex.), minimisez le
   facteur de branchement ou mettez les résultats en cache.

5. **Oublier le but dans l'état.** La méthode `is_goal()` est appelée sur chaque état.
   Si l'information de but est externe à la structure d'état, vous devez l'intégrer
   (ou utiliser une fermeture) pour que la méthode du trait puisse la vérifier.

## Pour aller plus loin

- **A* bidirectionnel :** `bidirectional_astar(start, goal, fwd_h, rev_h)` cherche depuis
  les deux extrémités simultanément, se rejoignant au milieu. Peut réduire le temps de
  recherche de moitié environ sur les graphes symétriques.
- **Recherche Q* :** Remplacez l'heuristique manuelle par une fonction Q apprise pour les
  domaines où concevoir de bonnes heuristiques est difficile. Voir
  [Heuristiques apprises Q*](./qstar-learned-heuristics.md).
- **MCTS :** Pour les arbres de jeu dont le facteur de branchement est trop grand pour A*,
  utilisez la recherche arborescente Monte Carlo. Voir [MCTS](./mcts.md).
- **IDA* :** Pas encore dans ix-search, mais facile à construire par-dessus `SearchState`
  en utilisant l'approfondissement itératif avec un seuil sur le coût f.
- Lire l'article original : Hart, Nilsson, Raphael, "A Formal Basis for the Heuristic
  Determination of Minimum Cost Paths" (1968).
