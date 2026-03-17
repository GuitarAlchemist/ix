# Machine à vecteurs de support (SVM linéaire)

## Le problème

Vous construisez un système qui classe des images de chiffres manuscrits comme étant soit un « 1 » soit un « 7 » en se basant sur des caractéristiques extraites : l'angle du trait, le nombre d'extrémités, le rapport hauteur/largeur, la présence d'une barre horizontale et la densité d'encre. Ces deux chiffres se ressemblent — un « 1 écrit à la hâte peut ressembler à un « 7 » sans sa barre transversale, et inversement.

Vous avez besoin d'un classifieur qui ne trouve pas simplement *une* frontière entre les deux classes, mais qui trouve la *meilleure* — la frontière qui se situe aussi loin que possible des deux classes. Si de futurs échantillons d'écriture diffèrent légèrement des données d'entraînement (et ce sera le cas), une frontière avec une large marge de sécurité classifiera toujours correctement, tandis qu'une frontière qui passe tout juste entre les deux groupes classifiera mal la moindre variation.

Les machines à vecteurs de support trouvent cette frontière à marge maximale. Elles posent la question : « Quelle est la rue la plus large que je puisse tracer entre les deux classes ? » Les points de données les plus proches de la rue — ceux qui définissent ses bords — sont appelés *vecteurs de support*. Tout le reste est sans importance pour la frontière.

## L'intuition

Imaginez que vous avez des billes rouges et bleues sur une table et que vous devez placer une règle entre elles. Il existe de nombreuses positions possibles pour la règle qui séparent les deux couleurs. Mais certaines positions sont meilleures que d'autres : la meilleure position maximise l'écart entre la règle et la bille la plus proche de chaque côté.

Maintenant, imaginez que vous collez la règle en place et que quelqu'un cogne la table, déplaçant légèrement toutes les billes. Si la règle était placée avec un écart étroit, les billes débordent. Si l'écart était large, la classification survit au choc. La SVM trouve la position de la règle avec l'écart le plus large.

Les « vecteurs de support » sont les billes les plus proches de la règle — ce sont les seules qui comptent pour déterminer sa position. Vous pourriez retirer toutes les autres billes et la frontière resterait la même. C'est ce qui rend la SVM élégante : la décision est guidée par les cas les plus difficiles, pas les cas faciles.

Que se passe-t-il lorsque les billes sont mélangées sans séparation nette ? Le paramètre C contrôle le compromis : un C élevé dit « classe tout correctement, même si la marge est étroite », tandis qu'un C faible dit « autorise quelques erreurs de classification pour obtenir une marge plus large ». Cette marge souple permet à la SVM de traiter des données réelles où les classes se chevauchent.

## Comment ça fonctionne

### Étape 1 : Définir la frontière de décision

La SVM trouve un hyperplan défini par les poids **w** et le biais b :

$$
f(x) = \mathbf{w} \cdot \mathbf{x} + b
$$

La prédiction de classe est basée sur le signe : si f(x) >= 0, on prédit la classe 1 ; sinon on prédit la classe 0.

En termes simples, cela signifie : le modèle apprend un poids pour chaque caractéristique, calcule une somme pondérée plus un biais, et vérifie de quel côté du zéro le résultat tombe. Côté positif = classe 1, côté négatif = classe 0.

### Étape 2 : Transformer les étiquettes en {-1, +1}

En interne, la classe 0 devient -1 et la classe 1 devient +1. Cela simplifie les mathématiques.

En termes simples, cela signifie : au lieu de 0 et 1, l'algorithme raisonne en termes de « côté négatif » et « côté positif » de la frontière.

### Étape 3 : Définir la perte charnière

$$
L_i = \max(0, 1 - y_i \cdot f(x_i))
$$

En termes simples, cela signifie : si un point est du bon côté de la marge (la « rue » est assez large), la perte est nulle — tout va bien. Si le point est à l'intérieur de la marge ou du mauvais côté, la perte croît linéairement en fonction de l'écart. Cette perte ne « s'active » que pour les points problématiques.

### Étape 4 : Minimiser l'objectif de la SVM

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w} \cdot x_i + b))
$$

En termes simples, cela signifie : trouver des poids qui équilibrent deux objectifs. Le premier terme (||w||^2) veut une marge large — des poids petits signifient une rue large. Le second terme (la somme des pertes charnières) veut une classification correcte. C contrôle quel objectif l'emporte : un C élevé privilégie la justesse, un C faible privilégie une marge large.

### Étape 5 : Optimiser par descente de sous-gradient

Pour chaque échantillon d'entraînement, on détermine s'il viole la marge. Si c'est le cas, le gradient pousse la frontière pour l'inclure. Sinon, seul le gradient de régularisation s'applique.

$$
\nabla_w = \begin{cases}
\mathbf{w} - C \cdot y_i \cdot x_i & \text{si } y_i(\mathbf{w} \cdot x_i + b) < 1 \\
\mathbf{w} & \text{sinon}
\end{cases}
$$

En termes simples, cela signifie : l'algorithme parcourt les données de manière répétée, en ajustant les poids. Pour les points correctement classifiés avec une marge suffisante, il réduit simplement les poids légèrement (régularisation). Pour les points mal classifiés ou trop proches de la frontière, il déplace également la frontière vers eux. Le taux d'apprentissage décroît au fil du temps pour assurer la stabilité.

### Étape 6 : Prédire les probabilités (calibration de Platt)

Puisque les SVM ne produisent pas naturellement de probabilités, l'implémentation applique une sigmoïde à la sortie brute de la fonction de décision :

$$
P(\text{classe} = 1 \mid x) = \frac{1}{1 + e^{-f(x)}}
$$

En termes simples, cela signifie : plus un point est éloigné de la frontière de décision, plus la prédiction est confiante. Les points situés exactement sur la frontière obtiennent une probabilité d'environ 50 %.

## En Rust

```rust
use ndarray::array;
use ix_supervised::svm::LinearSVM;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Features: [stroke_angle, endpoints, height_width_ratio, has_crossbar, ink_density]
    // Labels: 0 = digit "1", 1 = digit "7"
    let x = array![
        [85.0,  2.0, 3.5, 0.0, 0.15],   // "1"
        [88.0,  2.0, 3.8, 0.0, 0.12],   // "1"
        [82.0,  2.0, 3.2, 0.0, 0.18],   // "1"
        [90.0,  2.0, 4.0, 0.0, 0.10],   // "1"
        [45.0,  3.0, 1.8, 1.0, 0.35],   // "7"
        [50.0,  3.0, 2.0, 1.0, 0.30],   // "7"
        [40.0,  3.0, 1.5, 1.0, 0.40],   // "7"
        [48.0,  3.0, 1.9, 1.0, 0.33],   // "7"
    ];
    let y = array![0, 0, 0, 0, 1, 1, 1, 1];

    // Build SVM with C=1.0 regularization
    let mut svm = LinearSVM::new(1.0)
        .with_learning_rate(0.01)
        .with_max_iterations(500);
    svm.fit(&x, &y);

    // Predict
    let predictions = svm.predict(&x);
    println!("Predictions: {}", predictions);
    println!("Accuracy: {:.2}%", accuracy(&y, &predictions) * 100.0);

    // Probability estimates (via Platt scaling)
    let proba = svm.predict_proba(&x);
    for i in 0..proba.nrows() {
        println!(
            "Sample {}: P('1')={:.3}, P('7')={:.3}",
            i, proba[[i, 0]], proba[[i, 1]]
        );
    }

    // Test on a new sample
    let new_digit = array![[86.0, 2.0, 3.6, 0.0, 0.14]];
    let pred = svm.predict(&new_digit);
    println!("New digit classified as: {}", if pred[0] == 0 { "'1'" } else { "'7'" });

    // Evaluation
    println!("Precision ('7'): {:.4}", precision(&y, &predictions, 1));
    println!("Recall ('7'):    {:.4}", recall(&y, &predictions, 1));
    println!("F1 ('7'):        {:.4}", f1_score(&y, &predictions, 1));
}
```

## Quand l'utiliser

| Situation | SVM linéaire | Alternative | Pourquoi |
|---|---|---|---|
| Classification binaire, marges nettes | Oui | — | La marge maximale offre une excellente généralisation |
| Données à haute dimension (beaucoup de caractéristiques, peu d'échantillons) | Oui | — | La SVM excelle quand le nombre de caractéristiques >> nombre d'échantillons |
| Besoin d'une généralisation robuste | Oui | — | La maximisation de la marge prévient le surapprentissage |
| Frontière de décision non linéaire | Non | Arbre de décision, réseau de neurones, SVM à noyau | La SVM linéaire ne peut tracer que des hyperplans droits |
| Classification multi-classes | Nécessite un contournement | Arbre de décision, forêt aléatoire | Construire une SVM par paire de classes (un-contre-un ou un-contre-tous) |
| Besoin de probabilités calibrées | Non | Régression logistique | La calibration de Platt est approximative ; la régression logistique est nativement calibrée |
| Très grand jeu de données | Dépend | Régression logistique | La descente de sous-gradient est itérative ; la régression logistique peut converger plus vite |

## Paramètres clés

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `c` | (obligatoire) | Paramètre de régularisation. Plus élevé = moins de régularisation, ajustement plus serré aux données d'entraînement. |
| `learning_rate` | `0.001` | Pas initial pour la descente de sous-gradient. Décroît automatiquement au fil des itérations. |
| `max_iterations` | `1000` | Nombre de passes sur les données. |

Configuration via le patron de construction :
```rust
LinearSVM::new(1.0)               // C = 1.0
    .with_learning_rate(0.01)
    .with_max_iterations(500)
```

### Le paramètre C

| C | Comportement |
|---|---|
| 0.001 | Marge très large, autorise de nombreuses erreurs de classification (biais élevé, variance faible) |
| 0.01 - 0.1 | Modérément régularisé, bonne plage de départ |
| 1.0 | Équilibré (la valeur par défaut la plus courante) |
| 10 - 100 | Marge étroite, essaie fortement de classifier correctement chaque point d'entraînement (biais faible, variance élevée) |
| 1000+ | Effectivement une marge dure, surapprentissage probable si les données sont bruitées |

## Pièges

**La mise à l'échelle des caractéristiques est essentielle.** La SVM calcule des produits scalaires et des normes. Si une caractéristique varie de 0 à 1000 et une autre de 0 à 1, la première domine le calcul de la marge. Standardisez toujours les caractéristiques (moyenne nulle, variance unitaire) avant l'entraînement.

**Binaire uniquement.** Le `LinearSVM` d'ix ne prend en charge que deux classes (0 et 1). Pour les problèmes multi-classes, entraînez plusieurs SVM en configuration un-contre-tous et choisissez la classe ayant la valeur de fonction de décision la plus élevée.

**Linéaire uniquement.** Cette implémentation trouve un hyperplan linéaire. Si les classes ne sont pas linéairement séparables (par exemple, une classe entoure l'autre en forme d'anneau), la SVM fonctionnera mal. Les SVM à noyau (RBF, polynomial) peuvent gérer les frontières non linéaires mais ne sont pas implémentées dans ix.

**Convergence.** La descente de sous-gradient n'est pas aussi régulière que la descente de gradient standard. La perte peut osciller. Augmenter `max_iterations` et utiliser un `learning_rate` plus petit aide, mais la convergence est lente. Surveillez la perte d'entraînement si possible.

**Les estimations de probabilité sont approximatives.** La méthode `predict_proba` applique une sigmoïde à la sortie brute de la fonction de décision (calibration de Platt sans ajustement de calibration). Les probabilités sont directionnellement correctes mais mal calibrées. Ne les utilisez pas pour des tâches nécessitant des estimations de probabilité précises.

**Sensible aux valeurs aberrantes quand C est élevé.** Un seul point de données mal placé près de la marge peut déplacer la frontière de manière substantielle lorsque C est grand. Utilisez une valeur modérée de C ou nettoyez les valeurs aberrantes des données d'entraînement.

## Pour aller plus loin

- **Astuce du noyau :** Projeter les caractéristiques dans un espace de dimension supérieure où les motifs non linéaires deviennent linéairement séparables. Les noyaux courants incluent le RBF (gaussien) et le polynomial. C'est l'étape logique suivante après `LinearSVM`.
- **Multi-classes un-contre-tous :** Entraîner un `LinearSVM` par classe (classe c contre toutes les autres) et prédire la classe dont la SVM a la sortie de fonction de décision la plus élevée.
- **Importance des caractéristiques :** Le vecteur de poids **w** indique directement quelles caractéristiques comptent le plus. Les poids à forte valeur absolue correspondent aux caractéristiques importantes.
- **Comparaison avec la régression logistique :** Les deux trouvent une frontière linéaire. La régression logistique minimise la perte logarithmique ; la SVM minimise la perte charnière. La SVM se concentre sur la frontière (vecteurs de support), tandis que la régression logistique utilise tous les points. Voir [regression-logistique.md](./regression-logistique.md).
- **Évaluation :** Voir [metriques-evaluation.md](./metriques-evaluation.md) pour choisir entre exactitude, précision, rappel et score F1 pour les tâches de classification binaire.
