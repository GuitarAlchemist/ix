# Chaînes de Markov

## Le problème

Vous tenez un camion de glaces dont l'activité dépend de la météo. Le temps de demain (Ensoleillé, Nuageux, Pluvieux) dépend uniquement du temps d'aujourd'hui — pas de celui de la semaine passée. S'il fait beau aujourd'hui, il y a 70 % de chances qu'il fasse beau demain et 30 % de chances qu'il soit nuageux. Vous voulez répondre à des questions comme :
- Quelle est la probabilité de pluie dans 5 jours, sachant qu'il fait beau aujourd'hui ?
- À long terme, quelle fraction des jours sont pluvieux ?
- S'il se met à pleuvoir, combien de jours en moyenne avant le prochain jour ensoleillé ?

Scénarios concrets :
- **Génération de texte :** Chaque mot dépend du mot précédent. « Je » est suivi de « suis » bien plus souvent que de « banane ».
- **Comportement client :** Un utilisateur est soit Actif, Dormant ou Perdu. Les probabilités de transition alimentent les modèles de valeur vie client.
- **Théorie des files d'attente :** Un serveur est Inactif, Occupé ou Surchargé. La prédiction d'utilisation nécessite de connaître les taux de transition.
- **Modélisation financière :** Une note de crédit migre entre AAA, AA, A, BBB, ... Défaut. Les matrices de transition servent à valoriser les dérivés de crédit.
- **Jeux de plateau :** La probabilité d'atterrir sur chaque case au Monopoly est une chaîne de Markov sur les 40 positions du plateau.

## L'intuition

Une chaîne de Markov est un système qui saute entre des états, où l'état suivant dépend *uniquement de l'état courant* — pas de la façon dont on y est arrivé. C'est la **propriété de Markov** (absence de mémoire).

Pensez-y comme un jeu de plateau avec des dés truqués. Vous êtes sur une case. Vous lancez les dés et, selon le résultat, vous passez à la case suivante. Les dés sont *différents* pour chaque case (les probabilités de transition), mais ils ne tiennent pas compte des cases visitées auparavant.

La matrice de transition P encode tous les dés. La ligne i est la distribution de probabilité pour « où vais-je depuis l'état i ? ». Chaque ligne somme à 1.

Au fil de nombreuses étapes, quelque chose de remarquable se produit : quel que soit le point de départ, on finit par visiter chaque état avec une fréquence fixe. Cette fréquence est la **distribution stationnaire** — l'équilibre à long terme du système.

## Comment ça fonctionne

### Matrice de transition

Pour N états, la matrice de transition P est N × N :

```
P[i][j] = probabilité de passer de l'état i à l'état j
```

Chaque ligne somme à 1 (stochastique par ligne). Toutes les entrées sont non négatives.

### Distribution d'état après k étapes

En partant d'une distribution initiale pi_0 (un vecteur de probabilités sur les états) :

```
pi_k = pi_0 * P^k
```

**En clair :** Multipliez la distribution initiale par la matrice de transition k fois. Chaque multiplication avance d'un pas de temps.

### Distribution stationnaire

La distribution stationnaire pi* satisfait :

```
pi* = pi* * P
```

**En clair :** Si vous êtes dans la distribution stationnaire, une étape supplémentaire vous laisse dans la même distribution. C'est l'équilibre. ix la trouve par itération de puissance : partir d'une distribution uniforme et multiplier par P jusqu'à stabilisation.

### Temps moyen de premier passage

Le nombre moyen d'étapes pour atteindre l'état j en partant de l'état i :

```
M(i, j) = E[min{t > 0 : X_t = j | X_0 = i}]
```

**En clair :** « En moyenne, combien d'étapes pour aller de i à j pour la première fois ? » ix l'estime par simulation Monte Carlo (en exécutant de nombreuses marches aléatoires et en faisant la moyenne).

### Ergodicité

Une chaîne est **ergodique** si elle est à la fois irréductible (chaque état est accessible depuis tout autre état) et apériodique (elle ne cycle pas avec une période fixe). Les chaînes ergodiques ont une distribution stationnaire unique.

```
Ergodique = toutes les entrées de P^k sont positives (pour un certain k)
```

**En clair :** Après suffisamment d'étapes, il y a une chance non nulle d'être dans *n'importe quel* état, quel que soit le point de départ.

## En Rust

```rust
use ix_graph::markov::MarkovChain;
use ndarray::{array, Array1};

// Modèle météo : Ensoleillé=0, Nuageux=1, Pluvieux=2
let transition = array![
    [0.7, 0.2, 0.1],   // Ensoleillé -> 70% Ensoleillé, 20% Nuageux, 10% Pluvieux
    [0.3, 0.4, 0.3],   // Nuageux    -> 30% Ensoleillé, 40% Nuageux, 30% Pluvieux
    [0.2, 0.3, 0.5],   // Pluvieux   -> 20% Ensoleillé, 30% Nuageux, 50% Pluvieux
];

let mc = MarkovChain::new(transition)
    .unwrap()
    .with_names(vec![
        "Sunny".into(), "Cloudy".into(), "Rainy".into()
    ]);

// Q : S'il fait beau aujourd'hui, quelle est la distribution météo dans 5 jours ?
let today = array![1.0, 0.0, 0.0];  // 100% Ensoleillé
let in_5_days = mc.state_distribution(&today, 5);
println!("Dans 5 jours : {:.4}", in_5_days);
// ex. [0.4861, 0.2714, 0.2425]

// Q : Quelle est la distribution météo à long terme ?
let stationary = mc.stationary_distribution(1000, 1e-10);
println!("Stationnaire : {:.4}", stationary);
// Converge quel que soit l'état initial

// Q : Simuler une séquence météo de 30 jours à partir de Pluvieux
let path = mc.simulate(2, 30, 42);  // départ=Pluvieux(2), 30 étapes, graine=42
let names: Vec<&str> = path.iter()
    .map(|&s| mc.state_names[s].as_str())
    .collect();
println!("Prévision 30 jours : {:?}", &names[..10]);  // 10 premiers jours

// Q : En moyenne, combien de jours de Pluvieux à Ensoleillé ?
let mfpt = mc.mean_first_passage(
    2,       // depuis : Pluvieux
    0,       // vers : Ensoleillé
    10_000,  // nombre de simulations
    1000,    // étapes max par simulation
    42,      // graine
);
println!("Jours moyens Pluvieux -> Ensoleillé : {:.1}", mfpt);

// Q : Cette chaîne est-elle ergodique ?
let ergodic = mc.is_ergodic(100);
println!("Ergodique : {}", ergodic);  // true (tous les états accessibles, apériodique)
```

### Chaînes absorbantes

```rust
use ix_graph::markov::{MarkovChain, AbsorbingChain};
use ndarray::array;

// Cycle de vie client : Actif=0, Dormant=1, Perdu=2 (absorbant)
let transition = array![
    [0.7, 0.2, 0.1],   // Actif   -> 70% reste actif, 20% dormant, 10% perdu
    [0.3, 0.4, 0.3],   // Dormant -> 30% réactivé, 40% reste dormant, 30% perdu
    [0.0, 0.0, 1.0],   // Perdu   -> 100% reste perdu (état absorbant)
];

let mc = MarkovChain::new(transition).unwrap();
let absorbing = AbsorbingChain::new(mc);

println!("États absorbants : {:?}", absorbing.absorbing_states);  // [2]
println!("L'état 2 est-il absorbant ? {}", absorbing.is_absorbing_state(2));  // true
```

## Quand l'utiliser

| Modèle | Idéal quand | Mémoire | Espace d'états |
|---|---|---|---|
| **Chaîne de Markov** | L'état suivant ne dépend que de l'état courant | Sans mémoire | Discret, petit à moyen |
| **Modèle de Markov caché** | Les états sont cachés, on observe des émissions | Sans mémoire (caché) | Discret caché + observé |
| **ARIMA** | Séries temporelles continues avec tendances | Ordre fixe | Continu |
| **Réseau de neurones récurrent** | Dépendances à longue portée, grandes données | Apprise (illimitée) | Continu |

**Utilisez les chaînes de Markov quand :**
- Le système a un ensemble clair d'états discrets.
- L'hypothèse d'absence de mémoire est raisonnable (ou une bonne approximation).
- Vous voulez des résultats analytiques (distribution stationnaire, temps moyens de passage).
- Les probabilités de transition sont connues ou estimables à partir de données.

**N'utilisez pas quand :**
- L'historique compte (ex. « l'action a monté 3 jours de suite » change le comportement).
- L'espace d'états est continu ou extrêmement grand.
- Vous devez modéliser des états cachés (utilisez un HMM — voir [modeles-de-markov-caches.md](./modeles-de-markov-caches.md)).

## Paramètres clés

| Méthode | Paramètres | Description |
|---|---|---|
| `MarkovChain::new(transition)` | `Array2<f64>` | Matrice de transition stochastique par ligne. Renvoie `Result<Self, String>` |
| `.with_names(names)` | `Vec<String>` | Noms d'états optionnels lisibles par l'humain |
| `.state_distribution(initial, steps)` | `Array1<f64>`, `usize` | Faire évoluer un vecteur de probabilités dans le temps |
| `.stationary_distribution(max_iter, tol)` | `usize`, `f64` | Trouver l'équilibre par itération de puissance. tol = seuil de convergence |
| `.simulate(start, steps, seed)` | `usize`, `usize`, `u64` | Marche aléatoire renvoyant `Vec<usize>` des états visités |
| `.mean_first_passage(from, to, n_sim, max_steps, seed)` | `usize`, `usize`, `usize`, `usize`, `u64` | Estimation Monte Carlo du temps moyen de premier passage |
| `.is_ergodic(steps)` | `usize` | Vérifier si P^steps a toutes ses entrées positives |

## Pièges courants

1. **Les lignes doivent sommer à 1.** `MarkovChain::new()` valide cela avec une tolérance de 1e-6. Si vous construisez la matrice à partir de données (en comptant les transitions), normalisez chaque ligne : `row /= row.sum()`.

2. **Les chaînes non ergodiques n'ont pas de distribution stationnaire unique.** Si la chaîne a des états absorbants ou des cycles périodiques, `stationary_distribution()` peut converger vers une distribution dépendant du point de départ. Vérifiez `is_ergodic()` d'abord.

3. **Le temps moyen de premier passage est estimé, pas exact.** La méthode `mean_first_passage()` utilise une simulation Monte Carlo, donc les résultats varient selon la graine et le nombre de simulations. Utilisez au moins 10 000 simulations pour des estimations stables. Pour un calcul exact, résolvez le système d'équations linéaires M = 1 + P * M (pas encore implémenté).

4. **Grands espaces d'états.** La matrice de transition est dense (N × N). Une chaîne avec 10 000 états utilise 800 Mo. Pour les chaînes creuses, envisagez une représentation en matrice creuse (pas encore supporté).

5. **La propriété de Markov est une hypothèse.** Les systèmes réels ont souvent de la mémoire. Un client dormant depuis 6 mois est moins susceptible de se réactiver qu'un client dormant depuis 1 semaine. Les chaînes de Markov d'ordre supérieur (conditionnement sur les k derniers états) ou les HMM peuvent aider.

6. **Chaînes périodiques.** Une chaîne qui alterne de manière déterministe (0 -> 1 -> 0 -> 1) a une période de 2. Elle possède une distribution stationnaire ([0,5, 0,5]) mais n'y converge jamais réellement — elle oscille. `is_ergodic()` renverra false.

## Pour aller plus loin

- **Modèles de Markov cachés :** Quand on ne peut pas observer l'état directement. Voir [modeles-de-markov-caches.md](./modeles-de-markov-caches.md).
- **Algorithme de Viterbi :** Décoder la séquence d'états la plus probable à partir d'observations. Voir [algorithme-de-viterbi.md](./algorithme-de-viterbi.md).
- **Chaînes de Markov à temps continu (CTMC) :** Les transitions se produisent à des instants aléatoires (distribution exponentielle) plutôt qu'à des pas fixes. Modélisation par matrices de taux au lieu de matrices de transition. Pas encore dans ix.
- **PageRank :** L'algorithme original de Google est une chaîne de Markov sur le graphe du web. La distribution stationnaire donne l'importance des pages.
- **Monte Carlo par chaînes de Markov (MCMC) :** Concevoir une chaîne de Markov dont la distribution stationnaire est la distribution cible à échantillonner. Fondamental pour l'inférence bayésienne.
- **Génération de texte :** Construire une matrice de transition à partir des fréquences de paires de mots dans un corpus. `simulate()` génère du texte.
