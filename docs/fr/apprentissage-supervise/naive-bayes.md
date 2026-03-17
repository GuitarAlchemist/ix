# Bayes naïf

## Le problème

Vous construisez un système d'avis produits pour une plateforme de commerce en ligne. Les clients laissent des avis textuels, et vous devez automatiquement classer chaque avis comme positif (classe 1) ou négatif (classe 0). Après avoir converti chaque avis en caractéristiques numériques — par exemple les valeurs moyennes des plongements de mots, la longueur de l'avis, le nombre de points d'exclamation et un score de lexique de sentiment — vous avez besoin d'un classifieur qui s'entraîne rapidement, gère bien les données même avec peu d'exemples, et produit des estimations de probabilité.

L'analyse de sentiment est un domaine où les classes se chevauchent souvent considérablement dans l'espace des caractéristiques. Un avis mentionnant « pas mal » contient le mot « mal » mais exprime quelque chose de positif. Vous avez besoin d'un modèle qui pondère toutes les caractéristiques ensemble en utilisant la théorie des probabilités plutôt que de tracer des frontières de décision rigides.

Le Bayes naïf est l'un des classifieurs les plus anciens et les plus fiables pour ce type de tâche. Il applique le théorème de Bayes pour calculer la probabilité de chaque classe étant donné les caractéristiques observées, en faisant l'hypothèse simplificatrice que les caractéristiques sont indépendantes conditionnellement à la classe. Cette hypothèse « naïve » n'est presque jamais vraie en pratique, et pourtant le classifieur fonctionne remarquablement bien malgré tout.

## L'intuition

Imaginez que vous êtes un médecin diagnostiquant si un patient a un rhume ou la grippe. Vous observez des symptômes : la température, la sévérité de la toux et le niveau de fatigue. Pour chaque maladie, vous connaissez la plage typique de chaque symptôme grâce aux patients précédents. Le théorème de Bayes vous permet de combiner ces observations : « Étant donné cette température ET cette toux ET cette fatigue, quelle maladie est la plus probable ? »

La partie « naïve » est l'hypothèse d'indépendance. En réalité, une température élevée et une toux sévère sont corrélées — mais le Bayes naïf fait comme si elles ne l'étaient pas. Il traite chaque symptôme comme fournissant une preuve indépendante. Cette simplification rend les mathématiques calculables et, étonnamment, nuit rarement beaucoup à la précision de la classification.

La variante gaussienne utilisée dans ix suppose que chaque caractéristique suit une courbe en cloche (distribution normale) au sein de chaque classe. Ainsi, pour les « avis positifs », le score de sentiment pourrait avoir une moyenne de 0,7 avec une certaine dispersion, tandis que pour les « avis négatifs », il pourrait être centré autour de 0,3. Lorsqu'un nouvel avis arrive, le modèle demande : « Ce score de sentiment est-il plus vraisemblable sous la courbe en cloche positive ou négative ? » — et fait cela pour chaque caractéristique, puis multiplie les réponses ensemble.

## Comment ça fonctionne

### Étape 1 : Estimer les probabilités a priori des classes

$$
P(c) = \frac{\text{nombre d'échantillons d'entraînement dans la classe } c}{n}
$$

En clair, cela signifie : comptez combien d'avis positifs et négatifs vous avez. Si 60 % des avis sont positifs, la probabilité a priori de « positif » est 0,6. Avant même d'examiner les caractéristiques, le modèle penche déjà vers la prédiction de la classe la plus fréquente.

### Étape 2 : Estimer les paramètres gaussiens par classe

Pour chaque classe c et chaque caractéristique j, calculez la moyenne et la variance :

$$
\mu_{c,j} = \frac{1}{n_c} \sum_{i \in c} x_{i,j}
$$

$$
\sigma^2_{c,j} = \frac{1}{n_c} \sum_{i \in c} (x_{i,j} - \mu_{c,j})^2
$$

En clair, cela signifie : pour chaque classe, apprendre ce qui est « typique » pour chaque caractéristique. Les avis positifs pourraient avoir un score de sentiment moyen de 0,7 avec une variance de 0,04. Cela définit une courbe en cloche qui décrit le comportement de cette caractéristique dans cette classe.

### Étape 3 : Calculer les vraisemblances conditionnelles aux classes à l'aide de la densité gaussienne

$$
P(x_j \mid c) = \frac{1}{\sqrt{2\pi \sigma^2_{c,j}}} \exp\left(-\frac{(x_j - \mu_{c,j})^2}{2\sigma^2_{c,j}}\right)
$$

En clair, cela signifie : pour la valeur d'une caractéristique d'un nouvel avis, demandez « quelle est la vraisemblance de cette valeur sous la courbe en cloche de la classe c ? » Un score de sentiment de 0,8 est très vraisemblable sous la courbe en cloche « positive » mais peu vraisemblable sous la courbe « négative ».

### Étape 4 : Appliquer le théorème de Bayes avec l'hypothèse naïve d'indépendance

$$
P(c \mid x) \propto P(c) \prod_{j=1}^{p} P(x_j \mid c)
$$

En clair, cela signifie : multipliez la probabilité a priori (à quel point cette classe est-elle fréquente ?) par la vraisemblance de chaque caractéristique (à quel point chaque valeur de caractéristique est-elle typique pour cette classe ?), en supposant que les caractéristiques contribuent indépendamment. La classe avec le produit le plus élevé l'emporte.

### Étape 5 : Normaliser pour obtenir des probabilités

En pratique, on travaille dans l'espace logarithmique pour éviter le soupassement numérique (multiplication de nombreuses petites probabilités), puis on reconvertit via l'astuce log-somme-exp.

En clair, cela signifie : les produits bruts ne sont pas des probabilités correctes (ils ne somment pas à 1). On les redimensionne pour qu'ils le fassent, ce qui nous donne une distribution de probabilité propre sur les classes.

## En Rust

```rust
use ndarray::array;
use ix_supervised::naive_bayes::GaussianNaiveBayes;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Features: [sentiment_score, review_length_norm, exclamation_count, avg_word_embedding]
    // Labels: 0 = negative, 1 = positive
    let x = array![
        [0.8,  0.6, 2.0, 0.55],   // positive
        [0.7,  0.4, 1.0, 0.50],   // positive
        [0.9,  0.7, 3.0, 0.60],   // positive
        [0.65, 0.5, 1.0, 0.48],   // positive
        [0.2,  0.3, 0.0, 0.30],   // negative
        [0.1,  0.8, 0.0, 0.25],   // negative
        [0.3,  0.4, 0.0, 0.35],   // negative
        [0.15, 0.6, 0.0, 0.28],   // negative
    ];
    let y = array![1, 1, 1, 1, 0, 0, 0, 0];

    // Train the classifier
    let mut model = GaussianNaiveBayes::new();
    model.fit(&x, &y);

    // Predict on new reviews
    let new_reviews = array![
        [0.75, 0.5, 2.0, 0.52],   // probably positive
        [0.25, 0.4, 0.0, 0.32],   // probably negative
    ];

    let predictions = model.predict(&new_reviews);
    println!("Predictions: {}", predictions);

    // Get probability estimates
    let proba = model.predict_proba(&new_reviews);
    for i in 0..proba.nrows() {
        println!(
            "Review {}: P(negative)={:.3}, P(positive)={:.3}",
            i, proba[[i, 0]], proba[[i, 1]]
        );
    }

    // Evaluate on training data
    let train_pred = model.predict(&x);
    println!("Accuracy:  {:.4}", accuracy(&y, &train_pred));
    println!("F1 (positive): {:.4}", f1_score(&y, &train_pred, 1));
}
```

## Quand l'utiliser

| Situation | Bayes naïf | Alternative | Pourquoi |
|---|---|---|---|
| Petit jeu d'entraînement | Oui | — | Nécessite très peu d'exemples pour estimer les moyennes et variances |
| Entraînement et prédiction rapides requis | Oui | — | L'entraînement est en O(n*p), la prédiction en O(C*p). Pas d'itération. |
| Classification de texte / TAL | Oui | Régression logistique | Choix classique pour la classification de documents |
| Les caractéristiques sont véritablement indépendantes | Oui | — | L'hypothèse naïve est en fait correcte |
| Les caractéristiques sont fortement corrélées | Peut-être | Régression logistique, SVM | Fonctionne étonnamment bien malgré tout, mais les probabilités peuvent être mal calibrées |
| Besoin de probabilités bien calibrées | Non | Régression logistique | L'hypothèse naïve d'indépendance fausse les estimations de probabilité |
| Frontières non linéaires complexes | Non | Arbre de décision, forêt aléatoire | Le Bayes naïf suppose des distributions de classes gaussiennes simples |

## Paramètres clés

`GaussianNaiveBayes::new()` ne prend aucun paramètre de configuration. Le modèle apprend tout à partir des données :

| Paramètre appris | Description |
|---|---|
| `means` | Moyenne par classe pour chaque caractéristique |
| `variances` | Variance par classe pour chaque caractéristique (avec un plancher epsilon de 1e-9 pour éviter la division par zéro) |
| `priors` | Probabilité a priori de chaque classe |

C'est l'un des modèles les plus simples à utiliser — il n'y a rien à régler.

## Pièges

**L'hypothèse gaussienne.** Le modèle suppose que chaque caractéristique suit une courbe en cloche au sein de chaque classe. Si une caractéristique est bimodale, fortement asymétrique ou binaire, l'hypothèse gaussienne est inadaptée. Les caractéristiques binaires sont mieux servies par le Bayes naïf de Bernoulli (pas encore disponible dans ix).

**Les caractéristiques corrélées dégradent les estimations de probabilité.** Bien que la précision de classification soit souvent robuste face à la violation de l'indépendance, les *probabilités* peuvent être excessivement confiantes ou insuffisamment confiantes. Si vous avez besoin de probabilités calibrées, utilisez la régression logistique ou appliquez une calibration a posteriori.

**Caractéristiques à variance nulle.** Si une caractéristique a exactement la même valeur pour tous les échantillons d'une classe, la variance est nulle et la densité gaussienne devient une fonction delta. ix se protège contre cela avec une variance minimale de 1e-9, mais de telles caractéristiques devraient être supprimées.

**Caractéristiques continues uniquement.** La variante gaussienne attend des caractéristiques numériques continues. Les caractéristiques catégorielles doivent être encodées numériquement (par exemple, encodage one-hot) avant utilisation, et l'hypothèse gaussienne sur des caractéristiques binaires est une approximation grossière.

**Dominé par les caractéristiques informatives.** Contrairement aux arbres de décision qui peuvent ignorer les caractéristiques non pertinentes, le Bayes naïf multiplie les probabilités de *toutes* les caractéristiques. Les caractéristiques bruitées et non pertinentes diluent le signal des caractéristiques informatives. La sélection de caractéristiques aide.

## Pour aller plus loin

- **Fondements probabilistes :** Le module `ix_math::stats` fournit les fonctions de moyenne, variance et autres statistiques utilisées en interne par ce classifieur.
- **Pipeline de classification de texte :** Convertissez le texte en caractéristiques numériques à l'aide de TF-IDF ou de plongements de mots, puis alimentez `GaussianNaiveBayes`. Pour les caractéristiques binaires de présence de mots, une variante de Bernoulli serait plus appropriée.
- **Multi-classe :** L'implémentation gère automatiquement n'importe quel nombre de classes — les étiquettes 0, 1, 2, ... sont toutes prises en charge.
- **Combinaison par ensemble :** Utilisez les prédictions du Bayes naïf comme caractéristique d'entrée d'une `RandomForest` pour une approche d'ensemble par empilement.
- **Évaluation :** Voir [metriques-evaluation.md](./metriques-evaluation.md) pour comprendre quand la précision, le rappel et le score F1 sont appropriés pour les tâches d'analyse de sentiment.
