# Forêt aléatoire

## Le problème

Une société de cartes de crédit surveille des millions de transactions chaque jour. Chaque transaction possède des caractéristiques : le montant, la catégorie du commerçant, l'heure de la journée, la distance par rapport au domicile du titulaire, la présence physique de la carte, et bien d'autres encore. Les transactions frauduleuses sont rares — environ 1 sur 1 000 — mais chacune coûte de l'argent réel. L'entreprise a besoin d'un modèle qui détecte la fraude de manière fiable sans bloquer les achats légitimes.

Un seul arbre de décision peut apprendre des schémas comme « les transactions de plus de 5 000 $ dans les magasins d'électronique à 3 heures du matin sont suspectes », mais il est fragile. Modifiez quelques exemples d'entraînement et l'arbre change complètement. Il a aussi tendance au surapprentissage — mémorisant les cas de fraude spécifiques sur lesquels il a été entraîné plutôt que d'apprendre des schémas généraux.

Et si vous construisiez 100 arbres de décision différents, chacun examinant un échantillon aléatoire légèrement différent des données et un sous-ensemble aléatoire de caractéristiques, puis les laissiez voter ? Les erreurs des arbres individuels s'annuleraient, l'ensemble serait bien plus robuste, et vous obtiendriez une meilleure détection de la fraude. C'est une forêt aléatoire.

## L'intuition

Imaginez que vous demandiez à 100 agents de crédit différents d'examiner la même demande, mais que chaque agent ne voit qu'un sous-ensemble aléatoire des documents et un sous-ensemble aléatoire des dossiers passés. Certains feront des erreurs, mais leurs erreurs seront des erreurs *différentes*. Lorsque vous comptez les votes, la réponse majoritaire est presque toujours correcte, car il est peu probable que la plupart des agents se trompent de la même manière.

C'est le principe de la « sagesse des foules » appliqué aux arbres de décision. Chaque arbre est délibérément rendu légèrement imparfait (en l'entraînant sur un échantillon bootstrap aléatoire avec un sous-ensemble aléatoire de caractéristiques), et la moyenne collective lisse les erreurs individuelles.

Les deux sources d'aléa sont :
1. **Bagging (agrégation bootstrap) :** Chaque arbre s'entraîne sur un échantillon aléatoire *avec remise* à partir des données d'entraînement. Certains exemples apparaissent plusieurs fois ; d'autres sont entièrement exclus.
2. **Sous-échantillonnage des caractéristiques :** À chaque division, l'arbre ne considère qu'un sous-ensemble aléatoire de caractéristiques (typiquement la racine carrée du nombre total de caractéristiques). Cela décorrèle les arbres — si une caractéristique est très forte, tous les arbres ne l'utiliseront pas à la racine.

## Fonctionnement

### Étape 1 : Créer les échantillons bootstrap

Pour chaque arbre, tirer n échantillons *avec remise* à partir des n exemples d'entraînement originaux.

En termes simples, cela signifie : imaginez mettre tous vos exemples d'entraînement dans un sac, en tirer un (et le remettre), puis répéter n fois. Certains exemples sont tirés plusieurs fois, et environ 37 % ne sont jamais tirés. Chaque arbre voit une « version » différente des données d'entraînement.

### Étape 2 : Sélectionner un sous-ensemble aléatoire de caractéristiques

À chaque division de chaque arbre, choisir aléatoirement `max_features` caractéristiques (par défaut : sqrt(nombre total de caractéristiques)) et ne considérer que les divisions sur ces caractéristiques.

$$
m = \lceil \sqrt{p} \rceil
$$

où p est le nombre total de caractéristiques.

En termes simples, cela signifie : si vous avez 16 caractéristiques, chaque division n'examine que 4 choisies au hasard. Cela empêche chaque arbre d'utiliser la même caractéristique dominante à la racine, ce qui rendrait tous les arbres corrélés et annulerait l'intérêt de l'ensemble.

### Étape 3 : Faire croître chaque arbre

En utilisant l'échantillon bootstrap et la règle de sous-échantillonnage des caractéristiques, faire croître un arbre de décision en utilisant l'algorithme CART avec l'impureté de Gini (voir [decision-trees.md](./decision-trees.md)).

En termes simples, cela signifie : chaque arbre pose ses questions en utilisant uniquement ses caractéristiques assignées et ses données bootstrappées. Les arbres sont développés jusqu'à la `max_depth` spécifiée.

### Étape 4 : Agréger les prédictions (vote)

Pour la classification, chaque arbre vote pour une classe. La prédiction de la forêt est le vote majoritaire. Pour les estimations de probabilité, la forêt fait la moyenne des distributions de probabilité des classes sur tous les arbres.

$$
P(\text{class} = c \mid x) = \frac{1}{T} \sum_{t=1}^{T} P_t(\text{class} = c \mid x)
$$

En termes simples, cela signifie : demandez à chaque arbre « cette transaction est-elle une fraude ? » et prenez la réponse sur laquelle la majorité des arbres s'accordent. Pour les probabilités, faites la moyenne de la confiance de chaque arbre.

## En Rust

```rust
use ndarray::array;
use ix_ensemble::random_forest::RandomForest;
use ix_ensemble::traits::EnsembleClassifier;
use ix_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Features: [amount, merchant_cat, hour, distance_km, card_present]
    let x = array![
        [25.0,  1.0,  12.0,  2.0,  1.0],   // legitimate
        [15.0,  2.0,  14.0,  1.0,  1.0],   // legitimate
        [120.0, 3.0,  10.0,  5.0,  1.0],   // legitimate
        [45.0,  1.0,  18.0,  0.5,  1.0],   // legitimate
        [4500.0, 5.0,  3.0, 800.0, 0.0],   // fraud
        [2200.0, 5.0,  2.0, 500.0, 0.0],   // fraud
        [3100.0, 4.0,  4.0, 650.0, 0.0],   // fraud
        [5000.0, 5.0,  1.0, 900.0, 0.0],   // fraud
    ];
    let y = array![0, 0, 0, 0, 1, 1, 1, 1]; // 0 = legitimate, 1 = fraud

    // Build a random forest: 50 trees, max depth 5, seeded for reproducibility
    let mut forest = RandomForest::new(50, 5)
        .with_seed(42)
        .with_max_features(3);  // consider 3 of 5 features per split
    forest.fit(&x, &y);

    println!("Number of trees: {}", forest.n_estimators());

    // Predict
    let predictions = forest.predict(&x);
    println!("Predictions: {}", predictions);

    // Probability estimates
    let proba = forest.predict_proba(&x);
    println!("Fraud probabilities:");
    for i in 0..proba.nrows() {
        println!("  Transaction {}: {:.1}% fraud", i, proba[[i, 1]] * 100.0);
    }

    // Evaluate -- for fraud detection, precision and recall matter more than accuracy
    println!("Accuracy:  {:.4}", accuracy(&y, &predictions));
    println!("Precision (fraud): {:.4}", precision(&y, &predictions, 1));
    println!("Recall (fraud):    {:.4}", recall(&y, &predictions, 1));
    println!("F1 (fraud):        {:.4}", f1_score(&y, &predictions, 1));
}
```

## Quand l'utiliser

| Situation | Forêt aléatoire | Alternative | Pourquoi |
|---|---|---|---|
| Données tabulaires, besoin de haute précision | Oui | — | Parmi les meilleurs modèles sur les données structurées |
| Besoin d'un modèle nécessitant peu d'entretien | Oui | — | Fonctionne bien « clé en main » avec un réglage minimal |
| Besoin de calibration des probabilités | Correct | Régression logistique | Les probabilités de la forêt sont raisonnables mais pas parfaitement calibrées |
| Besoin d'un modèle unique et interprétable | Non | Arbre de décision | Une forêt de 50 arbres est une boîte noire |
| Importance des caractéristiques requise | Oui | — | Suivi des caractéristiques les plus utilisées dans tous les arbres |
| Données de très haute dimension (1000+ caractéristiques) | Oui | — | Le sous-échantillonnage des caractéristiques gère cela naturellement |
| Prédiction en temps réel avec contrainte de latence | Peut-être | Régression logistique | Prédire à travers 50 arbres est plus lent qu'un seul produit scalaire |

## Paramètres clés

| Paramètre | Défaut | Description |
|---|---|---|
| `n_trees` | (requis) | Nombre d'arbres dans la forêt. Plus d'arbres = meilleure précision, entraînement plus lent. |
| `max_depth` | (requis) | Profondeur maximale de chaque arbre individuel. |
| `max_features` | `sqrt(n_features)` | Nombre de caractéristiques considérées à chaque division. |
| `seed` | `42` | Graine aléatoire pour la reproductibilité (échantillonnage bootstrap et sélection des caractéristiques). |

Configuration via le patron constructeur :
```rust
RandomForest::new(100, 10)      // 100 trees, max depth 10
    .with_seed(123)
    .with_max_features(4)
```

### Combien d'arbres ?

| n_trees | Comportement |
|---|---|
| 1 | Un seul arbre — aucun bénéfice d'ensemble |
| 10-50 | Bon pour des expérimentations rapides |
| 100-500 | Plage standard en production |
| 1000+ | Rendements décroissants, entraînement beaucoup plus lent |

L'erreur diminue généralement fortement avec les 20 à 30 premiers arbres, puis se stabilise. Vous pouvez tracer l'erreur « out-of-bag » en fonction de n_trees pour trouver le point optimal.

## Pièges

**Non interprétable.** Contrairement à un seul arbre de décision, vous ne pouvez pas retracer le raisonnement d'une forêt aléatoire sous forme d'organigramme simple. Si l'explicabilité réglementaire est obligatoire, utilisez plutôt un seul arbre de décision ou une régression logistique.

**Lent sur les grands jeux de données.** Entraîner 100 arbres sur un million de lignes avec 50 caractéristiques prend du temps. Chaque arbre est indépendant, ce qui rend le problème « embarrassingly parallel » — mais l'implémentation actuelle d'ix est mono-thread.

**Surapprentissage avec des arbres profonds.** Bien que les forêts soient plus résistantes au surapprentissage que les arbres uniques, définir `max_depth` trop élevé sur de petits jeux de données peut encore poser des problèmes. Commencez avec `max_depth = 10` et ajustez à partir de là.

**Déséquilibre des classes.** En détection de fraude, les transactions frauduleuses peuvent représenter 0,1 % des données. La forêt peut apprendre à toujours prédire « légitime » parce que c'est correct 99,9 % du temps. Utilisez un échantillonnage stratifié, des poids de classes, ou des métriques d'évaluation qui tiennent compte du déséquilibre (voir [evaluation-metrics.md](./evaluation-metrics.md)).

**Arbres corrélés.** Si une caractéristique est massivement dominante, même avec le sous-échantillonnage des caractéristiques, de nombreux arbres peuvent finir par l'utiliser à la racine. Réduire davantage `max_features` aide à décorréler les arbres.

## Pour aller plus loin

- **Estimation « out-of-bag » (OOB) :** Les ~37 % d'échantillons non utilisés pour entraîner chaque arbre peuvent servir de jeu de validation intégré. Cela permet d'estimer l'erreur de généralisation sans division séparée en jeu de test.
- **Importance des caractéristiques :** Compter la fréquence d'utilisation de chaque caractéristique dans toutes les divisions, pondérée par la diminution de l'impureté. Les caractéristiques qui apparaissent près de la racine de nombreux arbres sont les plus importantes.
- **Gradient boosting :** Au lieu d'entraîner les arbres en parallèle sur des données bootstrappées, les entraîner séquentiellement où chaque arbre corrige les erreurs du précédent. Le crate `ix-ensemble` contient une ébauche de boosting pour un développement futur.
- **Analyse des arbres individuels :** Puisque chaque arbre est un `DecisionTree` d'`ix-supervised`, vous pouvez inspecter les arbres individuels pour le débogage.
