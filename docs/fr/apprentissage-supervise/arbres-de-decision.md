# Arbres de décision

## Le problème

Une banque reçoit des milliers de demandes de prêt par jour. Chaque demande contient des données structurées : le revenu annuel du demandeur, son score de crédit, son ancienneté d'emploi, ses dettes existantes et le montant du prêt demandé. Un analyste crédit humain examine ces facteurs et décide d'approuver ou de refuser. La banque souhaite automatiser ce processus, mais il y a une contrainte — les régulateurs exigent que chaque refus puisse être expliqué. « L'algorithme a dit non » n'est pas une réponse acceptable. Ils ont besoin d'un modèle dont le raisonnement peut être retracé étape par étape.

Les arbres de décision résolvent ce problème parce qu'ils prennent des décisions exactement comme un humain les décrirait : « Si le revenu est supérieur à 60 000 € *et* le score de crédit est supérieur à 700 *et* le ratio dette/revenu est inférieur à 0,4, approuver le prêt. Sinon, refuser. » Le modèle est un organigramme. On peut l'imprimer, le remettre à un régulateur, et celui-ci peut suivre la logique de la racine à la feuille.

Contrairement aux modèles linéaires qui produisent une unique somme pondérée, un arbre de décision partitionne l'espace des caractéristiques en régions rectangulaires, chacune associée à une étiquette de classe. Cela lui permet de capturer des motifs non linéaires et des interactions entre caractéristiques sans aucune ingénierie de caractéristiques.

## L'intuition

Imaginez que vous organisez un tiroir en désordre rempli de documents dans des dossiers étiquetés. Vous prenez toute la pile et posez la question : « Quelle question unique sépare cette pile en tas les plus homogènes possible ? » Peut-être est-ce « Le revenu est-il supérieur à 50 000 € ? » Vous séparez la pile. Puis vous prenez le tas de gauche et posez une autre question. Et encore une autre. Vous continuez à séparer jusqu'à ce que chaque tas ne contienne que des approbations ou que des refus.

Voilà ce qu'est un arbre de décision. Chaque question est une « division » sur une caractéristique à un seuil donné. Chaque tas final est une « feuille » avec une classe prédite. L'art consiste à choisir les *bonnes* questions — celles qui créent les tas les plus purs le plus rapidement.

L'algorithme mesure la pureté des tas à l'aide de l'impureté de Gini : un tas composé à 100 % d'approbations a un Gini = 0 (pur). Un tas réparti 50/50 a un Gini = 0,5 (impureté maximale). À chaque étape, l'arbre choisit la question qui réduit le plus l'impureté de Gini moyenne.

## Fonctionnement

### Étape 1 : Mesurer l'impureté avec l'indice de Gini

$$
\text{Gini}(S) = 1 - \sum_{c=1}^{C} p_c^2
$$

où p_c est la proportion d'échantillons dans l'ensemble S qui appartiennent à la classe c.

En termes simples, cela signifie : si l'on choisit deux échantillons au hasard dans le tas et qu'ils ont de grandes chances d'être de la même classe, le tas est pur (Gini faible). Si c'est un tirage à pile ou face, le tas est impur (Gini élevé).

### Étape 2 : Évaluer chaque division possible

Pour chaque caractéristique et chaque seuil possible (point médian entre les valeurs distinctes consécutives) :

$$
\text{Gain} = \text{Gini}(\text{parent}) - \frac{n_L}{n} \text{Gini}(L) - \frac{n_R}{n} \text{Gini}(R)
$$

En termes simples, cela signifie : essayer chaque façon de diviser les données, mesurer à quel point les deux tas résultants sont plus purs par rapport à l'original, et choisir la division offrant la plus grande amélioration.

### Étape 3 : Récursion

Appliquer la meilleure division pour créer deux nœuds enfants. Répéter le processus sur chaque enfant jusqu'à ce qu'une des conditions d'arrêt soit remplie :
- La profondeur maximale de l'arbre est atteinte
- Le nœud contient moins de `min_samples_split` échantillons
- Le nœud est déjà pur (Gini = 0)

En termes simples, cela signifie : continuer à poser des questions jusqu'à ce que les tas soient suffisamment purs, ou jusqu'à ce qu'on ait posé assez de questions (limite de profondeur). La limite de profondeur empêche l'arbre de mémoriser le bruit.

### Étape 4 : Prédiction

Pour un nouvel échantillon, on descend dans l'arbre en suivant les divisions. Le nœud feuille auquel on arrive donne la classe prédite (la classe majoritaire dans cette feuille) et les probabilités de classe (la distribution des échantillons d'entraînement dans cette feuille).

En termes simples, cela signifie : répondre à chaque question de l'organigramme, suivre la flèche et lire la réponse en bas.

## En Rust

```rust
use ndarray::array;
use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{accuracy, precision, recall};

fn main() {
    // Features: [annual_income_k, credit_score, employment_years, debt_to_income]
    let x = array![
        [85.0, 750.0, 10.0, 0.20],   // approved
        [45.0, 620.0,  2.0, 0.55],   // denied
        [70.0, 710.0,  5.0, 0.30],   // approved
        [30.0, 580.0,  1.0, 0.70],   // denied
        [95.0, 780.0, 15.0, 0.15],   // approved
        [50.0, 640.0,  3.0, 0.50],   // denied
        [60.0, 690.0,  4.0, 0.35],   // approved
        [35.0, 590.0,  1.0, 0.65],   // denied
    ];
    let y = array![1, 0, 1, 0, 1, 0, 1, 0]; // 1 = approved, 0 = denied

    // Build a tree with max depth 3, requiring at least 2 samples to split
    let mut tree = DecisionTree::new(3).with_min_samples_split(2);
    tree.fit(&x, &y);

    // Predict on training data
    let predictions = tree.predict(&x);
    println!("Predictions: {}", predictions);
    println!("Accuracy: {:.2}%", accuracy(&y, &predictions) * 100.0);

    // Get probability estimates for a new applicant
    let new_applicant = array![[65.0, 700.0, 6.0, 0.32]];
    let proba = tree.predict_proba(&new_applicant);
    println!(
        "Denial probability: {:.1}%, Approval probability: {:.1}%",
        proba[[0, 0]] * 100.0,
        proba[[0, 1]] * 100.0
    );

    // Evaluate
    println!("Precision (approved): {:.4}", precision(&y, &predictions, 1));
    println!("Recall (approved):    {:.4}", recall(&y, &predictions, 1));
}
```

## Quand l'utiliser

| Situation | Arbre de décision | Alternative | Pourquoi |
|---|---|---|---|
| Explicabilité requise | Oui | — | Les décisions peuvent être retracées comme des règles si/sinon |
| Types de caractéristiques mixtes, motifs non linéaires | Oui | — | Gère les interactions sans ingénierie de caractéristiques |
| Haute précision sur des données complexes | Non | Forêt aléatoire, gradient boosting | Un arbre seul tend au surapprentissage ou au sous-apprentissage |
| Frontières de décision lisses nécessaires | Non | Régression logistique, SVM | Les arbres produisent des frontières rectangulaires alignées sur les axes |
| Gestion des données manquantes (travail futur) | Peut-être | — | CART peut être étendu pour gérer les valeurs manquantes |
| Petit jeu de données | Oui | — | Les arbres sont rapides à entraîner et faciles à valider |

## Paramètres clés

| Paramètre | Défaut | Description |
|---|---|---|
| `max_depth` | (requis) | Profondeur maximale de l'arbre. Contrôle le surapprentissage. |
| `min_samples_split` | `2` | Nombre minimum d'échantillons requis pour tenter une division. Des valeurs plus élevées créent des arbres plus simples. |

Configuration via le patron de construction :
```rust
DecisionTree::new(5)              // max_depth = 5
    .with_min_samples_split(10)   // need at least 10 samples to split
```

### Profondeur et surapprentissage

| max_depth | Comportement |
|---|---|
| 1 | « Souche de décision » — une seule question, très simple, sous-apprentissage probable |
| 3-5 | Bonne plage de départ pour la plupart des problèmes |
| 10-20 | Très détaillé, risque de mémoriser le bruit d'entraînement |
| Illimité (grande valeur) | S'ajustera parfaitement aux données d'entraînement, généralement catastrophique sur de nouvelles données |

## Pièges

**Surapprentissage.** Un arbre profond mémorisera les données d'entraînement, y compris le bruit. Définissez toujours `max_depth` à une valeur raisonnable et validez sur des données de test séparées. Un arbre qui obtient 100 % de précision à l'entraînement et 60 % en test est en surapprentissage.

**Instabilité.** De petits changements dans les données d'entraînement peuvent produire des arbres complètement différents. Un seul point de données supprimé peut changer la division à la racine, se propageant en cascade dans toute la structure. C'est une propriété fondamentale des arbres — les forêts aléatoires y remédient en moyennant de nombreux arbres.

**Divisions alignées sur les axes uniquement.** L'arbre ne peut diviser que sur une caractéristique à la fois avec un seuil (par ex., « revenu > 50 000 »). Il ne peut pas représenter nativement des frontières de décision diagonales comme « revenu + 0,5 × score_crédit > 400 ». Il approxime les diagonales par un escalier de divisions alignées sur les axes, ce qui peut nécessiter un arbre très profond.

**Biais en faveur des caractéristiques à nombreuses valeurs distinctes.** Une caractéristique continue avec 1 000 valeurs uniques offre davantage de points de division possibles qu'une caractéristique binaire, de sorte que l'arbre peut la favoriser même lorsque la caractéristique binaire est plus prédictive. C'est une propriété connue de CART.

**Pas d'extrapolation.** Comme toutes les méthodes arborescentes, les prédictions sur des données en dehors de la plage d'entraînement renverront simplement la valeur de la feuille la plus proche. Un arbre entraîné sur des revenus allant de 30 000 € à 100 000 € prédira la même probabilité d'approbation pour 200 000 € que pour 100 000 €.

## Pour aller plus loin

- **Forêts aléatoires :** Combiner de nombreux arbres de décision pour réduire la variance et améliorer la précision. Voir [random-forest.md](./random-forest.md).
- **Importance des caractéristiques :** Les caractéristiques utilisées dans les divisions proches de la racine sont les plus importantes. Suivre quelles caractéristiques apparaissent à quelles profondeurs pour comprendre ce qui guide les décisions.
- **Élagage :** Entraîner un arbre profond, puis supprimer les branches qui n'améliorent pas la précision de validation. C'est une alternative à la définition de `max_depth` en amont.
- **Multi-classe :** L'implémentation ix gère n'importe quel nombre de classes — les étiquettes sont des `Array1<usize>` avec les classes 0, 1, 2, etc.
