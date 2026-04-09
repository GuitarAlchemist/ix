# Valeur de Shapley

## Le problème

Trois départements d'une entreprise partagent un cluster de calcul cloud. L'équipe IT seule aurait besoin d'un serveur à 100 000 $. Le marketing seul aurait besoin de 80 000 $. Les ventes seules auraient besoin de 90 000 $. Mais le partage d'un cluster unique à 200 000 $ sert les trois (au lieu de 270 000 $ séparément). L'entreprise économise 70 000 $ -- mais comment répartir le coût ?

Si on divise à parts égales (66 700 $ chacun), le marketing se plaint -- son coût autonome n'était que de 80 000 $, donc il n'économise presque rien. Si on divise au prorata du coût autonome, l'IT paie le plus mais peut arguer que ses charges de travail sont les plus importantes.

La **valeur de Shapley** est une réponse fondée : chaque département paie sa *contribution marginale* moyenne à travers tous les ordres possibles d'arrivée dans la coalition. C'est l'unique répartition des coûts satisfaisant les axiomes d'équité (symétrie, efficacité, linéarité et joueur nul).

Les mêmes mathématiques servent pour :
- **Importance des caractéristiques en ML :** quelle est la contribution de chaque caractéristique à la prédiction d'un modèle ? (Les valeurs SHAP sont des valeurs de Shapley.)
- **Pouvoir de vote :** quelle influence a chaque parti dans un gouvernement de coalition ?
- **Fiabilité réseau :** quel noeud causerait le plus de dégâts en tombant en panne ?

## L'intuition

Imaginez que les trois départements arrivent un par un, dans un ordre aléatoire, pour mettre en place le cluster partagé. Le premier arrivé paie le coût total de ce dont il a besoin. Chaque arrivant suivant paie uniquement le *coût supplémentaire* que sa présence ajoute.

Si l'IT arrive en premier, il paie 100 000 $. Si le marketing arrive en deuxième, le cluster doit passer de 100 000 $ à 150 000 $, donc le marketing paie 50 000 $. Si les ventes arrivent en dernier, il passe de 150 000 $ à 200 000 $, donc les ventes paient 50 000 $.

Mais l'ordre compte. Dans un ordre différent, les coûts changent. La valeur de Shapley est la **moyenne de la contribution marginale de chaque joueur à travers tous les ordres d'arrivée possibles**. Avec 3 joueurs, il y a 3! = 6 ordres.

Cette moyenne est ce qui la rend équitable :
- **Symétrie :** les joueurs qui contribuent de manière égale paient de manière égale.
- **Efficacité :** les valeurs de Shapley totales s'additionnent pour donner la valeur de la grande coalition (pas d'argent restant ni manquant).
- **Joueur nul :** un joueur qui n'apporte rien à aucune coalition ne paie rien.

## Comment ça fonctionne

Pour un jeu à `n` joueurs et une fonction caractéristique `v(S)` (la valeur de la coalition S) :

```
phi_i = sum over S not containing i:
    |S|! * (n - |S| - 1)! / n! * [v(S union {i}) - v(S)]
```

| Symbole | Signification |
|--------|---------|
| `phi_i` | Valeur de Shapley du joueur `i` |
| `S` | Une coalition (sous-ensemble de joueurs) ne contenant pas `i` |
| `v(S)` | Valeur que la coalition `S` peut atteindre seule |
| `v(S union {i}) - v(S)` | **Contribution marginale** du joueur `i` à la coalition `S` |
| `\|S\|! * (n - \|S\| - 1)! / n!` | Poids = probabilité que `S` soit « déjà là » dans un ordre aléatoire |

**En clair :** pour chaque groupe possible qui aurait pu se former avant l'arrivée du joueur `i`, on calcule combien `i` apporte à ce groupe. On fait la moyenne de ces contributions sur tous les groupes possibles, pondérée par la probabilité de chaque groupe dans un ordre d'arrivée aléatoire.

### Représentation des coalitions

Les coalitions sont stockées sous forme de **masques de bits** : le joueur `i` fait partie de la coalition si le bit `i` est actif. Avec un masque `u64`, cela supporte jusqu'à **63 joueurs**. La coalition vide est `0`, et la grande coalition (tous les joueurs) est `(1 << n) - 1`.

Exemple avec 3 joueurs :
- `0b000 = 0` -- coalition vide
- `0b001 = 1` -- joueur 0 seul
- `0b011 = 3` -- joueurs 0 et 1
- `0b111 = 7` -- grande coalition (les trois)

## En Rust

Le crate `ix-game` fournit la théorie des jeux coopératifs avec des coalitions basées sur les masques de bits :

```rust
use ix_game::cooperative::{CooperativeGame, weighted_voting_game};

fn main() {
    // --- Exemple de répartition des coûts ---
    // Trois départements partageant une infrastructure.
    let mut game = CooperativeGame::new(3);

    // Définir les valeurs à l'aide d'indices de joueurs (convertis en masques de bits en interne).
    game.set_value_for(&[0], 100.0);         // IT seul
    game.set_value_for(&[1], 80.0);          // Marketing seul
    game.set_value_for(&[2], 90.0);          // Ventes seules
    game.set_value_for(&[0, 1], 150.0);      // IT + Marketing
    game.set_value_for(&[0, 2], 160.0);      // IT + Ventes
    game.set_value_for(&[1, 2], 140.0);      // Marketing + Ventes
    game.set_value_for(&[0, 1, 2], 200.0);   // Grande coalition

    // Ou de manière équivalente, avec des masques de bits bruts :
    // game.set_value(0b001, 100.0);  // Joueur 0
    // game.set_value(0b011, 150.0);  // Joueurs 0 et 1
    // game.set_value(0b111, 200.0);  // Tous les joueurs

    // --- Valeur de Shapley ---
    let shapley: Vec<f64> = game.shapley_value();
    println!("Valeurs de Shapley : IT={:.1}, Marketing={:.1}, Ventes={:.1}",
             shapley[0], shapley[1], shapley[2]);
    // Les valeurs s'additionnent pour donner 200.0 (valeur de la grande coalition).

    // --- Appartenance au coeur ---
    // Vérifier si une allocation proposée est stable (aucune coalition ne veut se séparer).
    let proposal = vec![75.0, 55.0, 70.0]; // somme = 200
    println!("Dans le coeur ? {}", game.is_in_core(&proposal));

    // La valeur de Shapley elle-même peut ou non être dans le coeur.
    println!("Shapley dans le coeur ? {}", game.is_in_core(&shapley));

    // --- Vérification de superadditivité ---
    println!("Superadditif ? {}", game.is_superadditive());

    // --- Indice de pouvoir de Banzhaf ---
    // Mesure le pouvoir de vote : à quelle fréquence chaque joueur est-il un votant pivot ?
    let banzhaf: Vec<f64> = game.banzhaf_index();
    println!("Indice de Banzhaf : {:?}", banzhaf);

    // --- Jeu de vote pondéré ---
    // Modéliser un parlement : les partis ont des sièges, il faut la majorité (> 50%) pour voter les lois.
    let voting_game = weighted_voting_game(
        &[45.0, 30.0, 15.0, 10.0],  // Nombre de sièges par parti
        51.0,                         // Quota de majorité
    );
    let power = voting_game.shapley_value();
    println!("Pouvoir de vote : {:?}", power);
    // Souvent surprenant : un parti avec 15% des sièges peut avoir
    // bien plus ou bien moins de 15% du pouvoir.

    let banzhaf_power = voting_game.banzhaf_index();
    println!("Pouvoir de Banzhaf : {:?}", banzhaf_power);
}
```

### Résumé de l'API

| Méthode | Signature | Ce qu'elle fait |
|--------|-----------|--------------|
| `CooperativeGame::new(n)` | `usize -> Self` | Créer un jeu à `n` joueurs (max 63) |
| `set_value(coalition, value)` | `u64, f64` | Définir `v(S)` avec un masque de bits |
| `set_value_for(&[usize], value)` | `&[usize], f64` | Définir `v(S)` avec des indices de joueurs |
| `value(coalition)` | `u64 -> f64` | Obtenir `v(S)` (retourne 0.0 pour les coalitions non définies) |
| `grand_coalition()` | `-> u64` | Masque de bits pour tous les joueurs |
| `shapley_value()` | `-> Vec<f64>` | Calculer la valeur de Shapley de chaque joueur |
| `is_in_core(&allocation)` | `&[f64] -> bool` | Vérifier si l'allocation est dans le coeur |
| `is_superadditive()` | `-> bool` | Vérifier si la fusion des coalitions est toujours bénéfique |
| `banzhaf_index()` | `-> Vec<f64>` | Calculer l'indice de pouvoir de Banzhaf normalisé |
| `weighted_voting_game(&weights, quota)` | `-> CooperativeGame` | Créer un jeu de vote |

## Quand l'utiliser

| Situation | Méthode | Pourquoi |
|-----------|--------|-----|
| Répartition équitable des coûts/bénéfices | `shapley_value()` | Unique allocation satisfaisant les axiomes d'équité |
| Importance des caractéristiques ML (SHAP) | `shapley_value()` | Standard de l'industrie pour l'explicabilité |
| Vérifier si une allocation est stable | `is_in_core()` | Aucune coalition ne veut dévier |
| Analyse du pouvoir de vote | `weighted_voting_game()` + `shapley_value()` | Révèle le vrai pouvoir vs. le nombre brut de sièges |
| Comparaison d'indices de pouvoir | `banzhaf_index()` | Mesure de pouvoir alternative, pondérations plus simples |

## Paramètres clés

| Paramètre | Type | Ce qu'il contrôle |
|-----------|------|-----------------|
| `num_players` | `usize` | Nombre de joueurs (max 63 en raison du masque u64) |
| `coalition` | `u64` | Masque de bits représentant un ensemble de joueurs |
| `allocation` | `&[f64]` | Vecteur de gains proposé (une entrée par joueur) |
| `weights` (jeu de vote) | `&[f64]` | Poids de vote de chaque joueur |
| `quota` (jeu de vote) | `f64` | Seuil pour gagner (ex. 51 pour une majorité simple) |

## Pièges

1. **Complexité exponentielle.** Le calcul exact de la valeur de Shapley nécessite d'itérer sur les `2^n` coalitions pour chacun des `n` joueurs. Avec 20 joueurs, cela fait 20 millions de coalitions. Avec 30, 30 milliards. Pour les grands jeux, utilisez des approximations par échantillonnage (pas encore dans ix-game).

2. **Limite à 63 joueurs.** La représentation par masque de bits utilise `u64`, donc le nombre maximum de joueurs est 63. C'est suffisant pour la plupart des applications en théorie des jeux, mais trop petit pour les valeurs SHAP sur des modèles ML de haute dimension.

3. **Les coalitions non définies valent 0 par défaut.** Si vous oubliez de définir `v(S)` pour une coalition, elle vaut 0.0 par défaut. C'est correct pour de nombreux jeux (où les petites coalitions ne peuvent rien accomplir) mais peut silencieusement produire des résultats erronés si vous vouliez attribuer une valeur non nulle.

4. **La valeur de Shapley peut ne pas être dans le coeur.** La valeur de Shapley est toujours efficace (somme = `v(N)`) mais peut ne pas satisfaire la rationalité de coalition. Certaines coalitions pourraient préférer se séparer et faire mieux seules. Utilisez `is_in_core()` pour vérifier.

5. **Le pouvoir de vote est contre-intuitif.** Dans un parlement avec des sièges [50, 49, 1], le troisième parti avec 1 siège a un pouvoir de Shapley nul (il n'est jamais un votant pivot si le quota est 51). Mais avec des sièges [49, 49, 2] et un quota de 51, le petit parti a un pouvoir égal aux grands. Calculez toujours, ne devinez jamais.

## Pour aller plus loin

- **Valeurs SHAP pour le ML :** Le lien entre les valeurs de Shapley et l'importance des caractéristiques en ML. Chaque caractéristique est un « joueur », et la « valeur de coalition » est la prédiction du modèle utilisant ce sous-ensemble de caractéristiques. Les bibliothèques comme SHAP utilisent ce cadre.
- **Shapley par échantillonnage :** Pour un grand `n`, approximer la valeur de Shapley en échantillonnant des ordres aléatoires au lieu d'énumérer toutes les coalitions.
- **Nucléolus :** L'unique allocation qui minimise le mécontentement maximal de toute coalition. Pas encore implémenté mais repose sur la même structure `CooperativeGame`.
- **Equilibres de Nash :** La théorie des jeux non coopératifs où les joueurs agissent indépendamment. Voir [Equilibres de Nash](./equilibres-de-nash.md).
- Lecture : Shapley, "A Value for n-Person Games" (1953) -- l'article original, remarquablement lisible.
