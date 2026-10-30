# Optimisation par colonie de fourmis

> Les fourmis trouvent le plus court chemin vers la nourriture sans carte. Chaque fourmi dépose des phéromones sur le trajet qu'elle a parcouru. Les chemins courts sont parcourus plus souvent, donc ils accumulent plus de phéromones et attirent plus de fourmis. La piste des longs détours s'évapore.

**Prérequis :** [Probabilités et statistiques](../fondements/probabilites-et-statistiques.md), [Essaim particulaire](essaim-particulaire.md)

---

## Le problème

Une camionnette de livraison quitte l'entrepôt, passe une fois par chaque magasin et revient. Dans quel ordre visiter les magasins pour que la boucle soit la plus courte possible ? C'est le **problème du voyageur de commerce** (TSP). Avec 20 magasins, il existe plus de 10^16 itinéraires possibles : impossible de tous les essayer. La méthode gloutonne « aller au magasin le plus proche » est rapide, mais sur les deux instances TSPLIB mesurées plus bas, elle donne en moyenne une boucle 24 à 27 % plus longue que la meilleure, tous points de départ confondus.

L'optimisation par colonie de fourmis (ACO) construit de nombreux itinéraires en parallèle. Elle apprend, à partir des bons, quelles routes font partie d'une boucle courte, et y concentre les itinéraires suivants.

---

## L'intuition

Imaginez une colonie de fourmis où chacune construit un itinéraire de livraison complet :

1. **Chaque fourmi parcourt une boucle entière.** À chaque magasin, elle choisit le suivant au hasard, mais pas uniformément. Les magasins proches sont favorisés (l'*heuristique*), de même que les routes marquées d'une forte piste de phéromones (la *mémoire* de la colonie).
2. **Les phéromones s'évaporent.** Après chaque tour, toutes les pistes s'affaiblissent un peu. Une route que personne n'emprunte disparaît.
3. **Les bonnes boucles laissent des pistes plus fortes.** Une fourmi dépose des phéromones sur chaque route de sa boucle, en quantité inversement proportionnelle à la longueur de la boucle. Les boucles courtes marquent donc leurs routes plus fortement.

Tour après tour, les routes des boucles courtes accumulent les phéromones, et les fourmis concentrent leur recherche autour d'elles.

---

## Comment ça marche

### Choisir la ville suivante

Depuis la ville `i`, une fourmi va vers une ville non visitée `j` avec la probabilité

```
p(i -> j) = tau[i][j]^alpha * eta[i][j]^beta  /  somme du même terme sur les villes k non visitées
```

**En clair :** `tau` est la quantité de phéromones sur la route `i-j`, et `eta = 1 / distance` favorise les villes proches. `alpha` pondère la mémoire de la colonie et `beta` pondère la carte. Avec `alpha = 0`, les fourmis s'ignorent entre elles et se comportent comme des recherches gloutonnes aléatoires.

### Évaporation et dépôt

```
tau <- (1 - rho) * tau              (chaque route, à chaque tour)
tau[a][b] += 1 / L                  (chaque route a-b d'une boucle de longueur L)
```

**En clair :** l'évaporation (`rho`) fait oublier à la colonie ses anciennes décisions, et les dépôts récompensent les boucles courtes. Les deux variantes diffèrent par **qui** dépose :

- **Ant System** (`AntColony::new()`) : toutes les fourmis déposent. C'est simple, mais sur les instances plus grandes, la colonie continue de récompenser des boucles médiocres et stagne.
- **MAX-MIN Ant System** (`AntColony::max_min()`) : seule la meilleure fourmi du tour dépose. Chaque valeur de phéromone est ensuite bornée à `[tau_min, tau_max]`. Ces bornes gardent chaque route possible (aucune piste ne tombe à zéro) et empêchent un itinéraire de tout accaparer (aucune piste ne grandit sans limite).

### Recherche locale 2-opt

Une boucle qui se croise n'est jamais optimale : décroiser les deux routes qui se croisent la raccourcit toujours. **2-opt** essaie chaque paire de routes, échange la paire chaque fois que cela raccourcit la boucle, et recommence jusqu'à ce qu'aucun échange n'aide plus. Avec `with_local_search(true)`, la boucle de chaque fourmi est nettoyée ainsi avant d'être évaluée. Les fourmis explorent alors les bonnes régions et 2-opt peaufine chaque résultat, ce qui est bien plus rapide que de laisser les phéromones faire tout le travail.

---

## En Rust

### Planifier une tournée de livraison

```rust
use ix_optimize::aco::AntColony;
use ndarray::Array2;

let shops: [(f64, f64); 6] = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (0.0, 1.0)];
let km = Array2::from_shape_fn((shops.len(), shops.len()), |(i, j)| {
    let (dx, dy) = (shops[i].0 - shops[j].0, shops[i].1 - shops[j].1);
    (dx * dx + dy * dy).sqrt()
});

let route = AntColony::max_min_2opt().with_seed(42).solve_tsp(&km);

assert_eq!(route.tour[0], 0); // tours start at the depot
assert_eq!(route.length, 6.0); // around the grid's edge: the shortest loop
```

Cet exemple est le doctest de `crates/ix-optimize/src/aco.rs`, donc `cargo test -p ix-optimize --doc` l'exécute. Les commentaires du code restent en anglais pour que le texte soit identique à celui du doctest.

La matrice des distances doit être carrée, symétrique, finie et positive ou nulle ; sinon `solve_tsp` panique. N'importe quelle distance convient : kilomètres, minutes ou euros.

### Préréglages

| Préréglage | Variante | Recherche locale | `rho` | Fourmis | Itérations |
|------------|----------|------------------|-------|---------|------------|
| `AntColony::new()` | Ant System | non | 0,5 | une par ville | 200 |
| `AntColony::max_min()` | MAX-MIN | non | 0,02 | une par ville | 1000 |
| `AntColony::max_min_2opt()` | MAX-MIN | 2-opt | 0,2 | 25 | 200 |

**Commencez par `max_min_2opt()`.** C'est le meilleur des trois dans toutes les mesures ci-dessous. Les réglages sont ceux que Dorigo & Stützle (2004) recommandent pour chaque cas.

### Comprendre la valeur de retour

`solve_tsp` renvoie un `TourResult` :

| Champ     | Type         | Signification                                                        |
|-----------|--------------|----------------------------------------------------------------------|
| `tour`    | `Vec<usize>` | Chaque ville exactement une fois, en commençant par la ville 0       |
| `length`  | `f64`        | La longueur de la boucle fermée, retour à la ville 0 compris         |
| `history` | `Vec<f64>`   | La meilleure longueur trouvée après chaque itération (jamais en hausse) |

`ix_optimize::aco::tour_length(&distances, &tour)` mesure n'importe quelle tournée de la même façon.

---

## Mesures sur TSPLIB

`crates/ix-optimize/examples/aco_tsplib.rs` exécute chaque configuration avec les graines 1 à 10, sur deux instances TSPLIB dont l'optimum est prouvé. L'écart vaut `longueur / optimum - 1`. Les temps sont ceux d'une compilation release sur une machine de bureau et ne donnent qu'un ordre de grandeur.

```text
cargo run -p ix-optimize --release --example aco_tsplib
```

**kroA100** (100 villes, optimum 21282) :

| Configuration | Itérations | Écart moyen | Optimum atteint | Temps par run |
|---------------|-----------:|------------:|----------------:|--------------:|
| Ant System | 1000 | 7,01 % | 0/10 | ~1,9 s |
| MAX-MIN | 1000 | 0,43 % | 1/10 | ~1,8 s |
| Ant System + 2-opt | 50 | 0,07 % | 3/10 | ~0,4 s |
| `max_min_2opt()` | 25 | 0,33 % | 3/10 | ~0,08 s |
| `max_min_2opt()` | 50 | 0,00 % | 10/10 | ~0,1 s |

**berlin52** (52 villes, optimum 7542) :

| Configuration | Itérations | Écart moyen | Optimum atteint |
|---------------|-----------:|------------:|----------------:|
| Ant System | 200 | 1,68 % | 0/10 |
| MAX-MIN | 200 | 9,62 % | 0/10 |
| MAX-MIN | 1000 | 0,00 % | 10/10 |
| `max_min_2opt()` | 25 | 0,00 % | 10/10 |

Deux leçons de ces mesures :
- **Sans recherche locale, MAX-MIN a besoin d'un gros budget.** À 200 itérations, il fait moins bien qu'Ant System, car avec `rho = 0,02` il n'a pas encore convergé. C'est pourquoi `max_min()` utilise 1000 itérations par défaut.
- **Avec recherche locale, MAX-MIN a besoin d'autres réglages.** Avec le `rho = 0,02` prévu sans recherche locale, il reste derrière Ant System + 2-opt. Avec `rho = 0,2` et 25 fourmis, il atteint l'optimum pour chaque graine.

---

## Quand l'utiliser

| Situation | ACO ? |
|-----------|-------|
| Plus courte boucle passant par un ensemble de lieux (tournées, perçage, préparation de commandes) | Oui -- c'est ce que fait `solve_tsp` |
| Jusqu'à quelques centaines de villes | Oui -- `max_min_2opt()` y est rapide et quasi optimal |
| Vous avez besoin d'un optimum *prouvé* | Non -- ACO ne fournit aucun certificat ; utilisez un solveur exact (par ex. Concorde) |
| Des milliers de villes | Avec prudence -- chaque itération coûte O(fourmis * n^2) ; comptez des secondes, voire des minutes |
| Distances asymétriques (sens uniques) | Non -- `solve_tsp` exige une matrice symétrique |
| Paramètres continus (réglage d'hyperparamètres) | Non -- utilisez l'[essaim particulaire](essaim-particulaire.md) |

---

## Paramètres clés

### Variante et recherche locale

- Utilisez `max_min_2opt()`, sauf si vous étudiez l'algorithme lui-même.
- `with_local_search(true)` fonctionne aussi avec `AntColony::new()`. C'est aussi, pour Ant System, l'amélioration la plus importante.

### Itérations (`with_max_iterations`)

- Chaque itération construit une tournée par fourmi. Son coût est O(fourmis * n^2), plus celui de 2-opt quand la recherche locale est activée.
- `history` montre quand la recherche a cessé de progresser. Si ses dernières valeurs sont toutes égales, moins d'itérations auraient suffi.

### Fourmis (`with_ants`)

- Par défaut, une fourmi par ville. Avec la recherche locale, 25 fourmis suffisent, car 2-opt fait le réglage fin.

### `alpha`, `beta` (`with_alpha`, `with_beta`, 1 et 2 par défaut)

- Un `beta` plus élevé fait davantage confiance à la carte : la recherche est plus gloutonne et converge plus vite, mais parfois vers la mauvaise boucle.
- Un `alpha` plus élevé fait davantage confiance à la colonie : la recherche converge plus vite, avec un risque de stagnation accru.

### Évaporation (`with_evaporation`, `rho`)

- Un `rho` élevé donne une mémoire courte : la colonie oublie vite et suit les dernières bonnes boucles.
- Un `rho` faible donne une mémoire longue : la recherche est large mais lente. Sans recherche locale, MAX-MIN a besoin d'un `rho` faible *et* de nombreuses itérations.

### Graine (`with_seed`)

- La même graine avec la même matrice donne la même tournée. Pour les tournées importantes, lancez 5 à 10 graines et gardez la plus courte.

---

## Pièges

**Juger MAX-MIN sur un petit budget.** Avec `rho = 0,02`, 200 itérations ne suffisent pas : la colonie explore encore. Comparez les variantes au budget pour lequel elles sont conçues, ou activez la recherche locale.

**Garder les réglages sans recherche locale après avoir activé la recherche locale.** Le meilleur `rho` change quand on ajoute 2-opt (0,02 devient 0,2 pour MAX-MIN). Le préréglage `max_min_2opt()` porte les bonnes valeurs.

**Matrices asymétriques ou invalides.** `solve_tsp` vérifie la matrice et panique si elle n'est pas carrée, si elle est asymétrique, ou si une valeur est négative, NaN ou infinie. Nettoyez les données d'abord.

**Des distances arrondies autrement qu'une référence.** TSPLIB arrondit chaque distance euclidienne à l'entier le plus proche. Ne comparez vos longueurs aux optimums publiés que si votre matrice suit la même règle.

---

## Pour aller plus loin

- **Voir les mesures :** [`crates/ix-optimize/examples/aco_tsplib.rs`](../../../crates/ix-optimize/examples/aco_tsplib.rs) affiche les tableaux ci-dessus.
- **Un essaim dans un espace continu :** l'[essaim particulaire](essaim-particulaire.md) applique une idée proche de « mémoire partagée » aux paramètres réels.
- **Une alternative à un seul agent :** le [recuit simulé](recuit-simule.md) peut aussi chercher des tournées, avec un seul agent et une température qui décroît.
- **Une alternative par population :** les [algorithmes génétiques](../evolutionnaire/algorithmes-genetiques.md) font évoluer des solutions par croisement et mutation.
- **GPU :** aucune version GPU n'existe encore. [ix#362](https://github.com/GuitarAlchemist/ix/issues/362) explique pourquoi cette version CPU vient d'abord : tout kernel `wgpu` dans `ix-gpu` serait vérifié par rapport à elle.
- **Références :** M. Dorigo & T. Stützle, *Ant Colony Optimization*, MIT Press, 2004. T. Stützle & H. H. Hoos, « MAX-MIN Ant System », *Future Generation Computer Systems* 16(8), 2000.
