# Régression linéaire

## Le problème

Vous travaillez dans une agence immobilière et devez estimer le prix des maisons avant leur mise sur le marché. Chaque maison possède des attributs mesurables — superficie, nombre de chambres, distance à l'école la plus proche, taille du terrain — et un prix de vente réellement payé par les acheteurs précédents. Votre mission est d'examiner ces chiffres et de prédire à quel prix une nouvelle maison se vendra.

Un expert immobilier peut le faire, mais il est lent et subjectif. Il risque de s'ancrer sur la dernière maison qu'il a visitée ou de se laisser influencer par une belle cuisine. Vous voulez une méthode qui prenne en compte chaque caractéristique simultanément, les pondère objectivement à partir des données historiques, et produise une estimation en dollars en quelques millisecondes.

La régression linéaire est cette méthode. Elle trouve la relation en ligne droite (ou en plan plat, en dimensions multiples) entre vos caractéristiques d'entrée et le prix, puis utilise cette relation pour prédire les prix de maisons qu'elle n'a jamais vues.

## L'intuition

Imaginez que vous traciez les prix des maisons sur l'axe des ordonnées et la superficie sur l'axe des abscisses. Les points forment un nuage grossièrement ascendant. La régression linéaire trace la droite unique à travers ce nuage qui minimise la distance totale entre chaque point et la droite. « Distance » signifie ici la distance verticale — l'écart entre votre prédiction et le prix réel pour chaque maison historique.

Lorsque vous avez plus d'une caractéristique, la « droite » devient une surface plane (un hyperplan) tendue à travers un espace de dimension supérieure. Vous ne pouvez plus le visualiser, mais les mathématiques restent les mêmes : trouver la surface plane qui se situe aussi près que possible de chaque point de données.

Pensez-y comme un équilibrage d'une règle rigide sur un ensemble de clous dépassant d'une planche. La règle se stabilise dans la position où la tension totale des ressorts (les erreurs au carré) est minimisée. Cette position de repos *est* la droite de régression.

## Comment ça fonctionne

### Étape 1 : Augmenter la matrice de caractéristiques

On ajoute une colonne de uns à la matrice de caractéristiques **X** afin que le terme de biais (l'ordonnée à l'origine) apparaisse naturellement comme un poids supplémentaire.

```
X_aug = [X | 1]
```

En clair, cela signifie : on ajoute une « caractéristique fictive » de valeur 1 à chaque point de données pour que le modèle puisse apprendre un prix de base même lorsque toutes les vraies caractéristiques sont nulles.

### Étape 2 : Résoudre l'équation normale

$$
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

En clair, cela signifie : on trouve le vecteur de poids **w** qui minimise la somme des erreurs de prédiction au carré en une seule opération, sans itérer. La formule examine comment les caractéristiques sont corrélées entre elles (X^T X) et comment elles sont corrélées avec la cible (X^T y), puis résout l'équilibre parfait.

### Étape 3 : Séparer les poids et le biais

Le dernier élément de **w** devient le biais (l'ordonnée à l'origine). Tout ce qui précède constitue le vecteur de poids par caractéristique.

En clair, cela signifie : on sait maintenant que « chaque mètre carré supplémentaire ajoute 215 $ au prix » (un poids) et que « le prix de base d'un terrain vide est de 45 000 $ » (le biais).

### Étape 4 : Prédire

$$
\hat{y} = \mathbf{X} \mathbf{w} + b
$$

En clair, cela signifie : pour une nouvelle maison, on multiplie chaque caractéristique par son poids, on additionne le tout, on ajoute le biais, et c'est le prix prédit.

## En Rust

```rust
use ndarray::array;
use ix_supervised::linear_regression::LinearRegression;
use ix_supervised::traits::Regressor;
use ix_supervised::metrics::{mse, rmse, r_squared};

fn main() {
    // Features: [sqft, bedrooms, distance_to_school_miles]
    let x = array![
        [1400.0, 3.0, 0.5],
        [1600.0, 3.0, 1.2],
        [1700.0, 4.0, 0.8],
        [1875.0, 4.0, 0.3],
        [1100.0, 2.0, 2.0],
        [2200.0, 5.0, 0.6],
    ];

    // Sale prices in thousands of dollars
    let y = array![245.0, 312.0, 279.0, 308.0, 199.0, 395.0];

    // Train the model
    let mut model = LinearRegression::new();
    model.fit(&x, &y);

    // Inspect learned parameters
    let weights = model.weights.as_ref().unwrap();
    println!("Weights: {}", weights);   // per-feature coefficients
    println!("Bias: {:.2}", model.bias); // intercept

    // Predict on new houses
    let new_houses = array![
        [1500.0, 3.0, 1.0],
        [2000.0, 4.0, 0.4],
    ];
    let predictions = model.predict(&new_houses);
    println!("Predicted prices: {}", predictions);

    // Evaluate on training data
    let y_pred = model.predict(&x);
    println!("MSE:  {:.4}", mse(&y, &y_pred));
    println!("RMSE: {:.4}", rmse(&y, &y_pred));
    println!("R^2:  {:.4}", r_squared(&y, &y_pred));
}
```

## Quand l'utiliser

| Situation | Régression linéaire | Alternative | Pourquoi |
|---|---|---|---|
| Cible continue, relation linéaire | Oui | — | C'est son point fort |
| Besoin de coefficients interprétables | Oui | — | Chaque poids a une signification claire |
| Motifs non linéaires dans les données | Non | Arbre de décision, réseau de neurones | La régression linéaire sous-ajustera les courbes |
| Beaucoup de caractéristiques, beaucoup sans pertinence | Attention | Lasso/Ridge (pas encore dans ix) | Les MCO peuvent surajuster avec de nombreuses caractéristiques colinéaires |
| Tâche de classification | Non | Régression logistique, SVM | La régression linéaire prédit des valeurs continues, pas des classes |
| Très grand jeu de données (millions de lignes) | Lent | Régression par descente de gradient stochastique | L'équation normale inverse une matrice N×N |

## Paramètres clés

| Paramètre | Type | Description |
|---|---|---|
| `weights` | `Option<Array1<f64>>` | Coefficients appris par caractéristique. `None` avant l'appel à `fit()`. |
| `bias` | `f64` | Ordonnée à l'origine apprise (valeur de y lorsque toutes les caractéristiques sont nulles). Initialisé à `0.0`. |

`LinearRegression::new()` ne prend aucune configuration — l'équation normale n'a pas d'hyperparamètres à régler. C'est l'un de ses atouts : il n'y a rien à mal paramétrer.

## Pièges

**Matrice singulière.** Si deux caractéristiques sont parfaitement corrélées (par exemple, la superficie en mètres carrés et en pieds carrés), X^T X est singulière et ne peut pas être inversée. L'appel à `fit()` provoquera une panique. Supprimez les caractéristiques dupliquées ou parfaitement colinéaires avant l'entraînement.

**Mise à l'échelle des caractéristiques.** Bien que la régression linéaire produise des résultats corrects quelle que soit l'échelle des caractéristiques, les *poids* seront sur des échelles très différentes si les caractéristiques le sont (par exemple, la superficie en milliers contre le nombre de chambres en unités). Cela rend l'interprétation plus difficile mais n'affecte pas les prédictions.

**Les valeurs aberrantes dominent.** Comme le modèle minimise l'erreur *au carré*, une seule villa à 10 M$ dans un quartier de maisons à 300 K$ tirera la droite de régression vers elle. Envisagez de supprimer les valeurs aberrantes extrêmes ou d'utiliser des techniques de régression robuste.

**L'extrapolation est dangereuse.** Le modèle ne sait rien en dehors de la plage des données d'entraînement. Prédire le prix d'un entrepôt de 5 000 m² alors que votre plus grand exemple d'entraînement faisait 300 m² produira des résultats absurdes.

**Surajustement avec beaucoup de caractéristiques.** Si vous avez presque autant de caractéristiques que de points de données, le modèle peut ajuster le bruit. L'équation normale produira toujours une solution, mais elle ne généralisera pas. Collectez plus de données ou réduisez le nombre de caractéristiques.

## Pour aller plus loin

- **Caractéristiques polynomiales :** Créez des colonnes comme `sqft^2` ou `sqft * bedrooms` et intégrez-les comme caractéristiques supplémentaires. La régression linéaire sur des caractéristiques polynomiales peut modéliser des courbes.
- **Régularisation :** Les régressions Ridge et Lasso ajoutent un terme de pénalité pour empêcher les poids trop élevés. Le crate `ix-optimize` d'ix fournit des optimiseurs SGD et Adam qui pourraient être utilisés pour les implémenter.
- **Alternative par descente de gradient :** Pour les jeux de données trop volumineux pour l'équation normale, utilisez `ix_optimize::sgd` pour minimiser itérativement l'erreur quadratique moyenne au lieu d'inverser une matrice.
- **Évaluation :** Voir [metriques-evaluation.md](./metriques-evaluation.md) pour une analyse approfondie de l'EQM, la REQM et le R².
