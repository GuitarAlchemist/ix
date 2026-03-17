# Gradient Boosting

## Le problème

Les forêts aléatoires moyennent de nombreux arbres indépendants pour réduire la variance. Mais que se passerait-il si, au lieu d'entraîner les arbres en parallèle sur des sous-ensembles aléatoires, vous les entraîniez l'un après l'autre, chaque nouvel arbre ciblant spécifiquement les erreurs de tous les arbres précédents ?

C'est le gradient boosting. Là où les forêts aléatoires combattent la *variance* (prédictions bruitées, surajustement), le gradient boosting combat le *biais* (sous-ajustement, prédictions systématiquement fausses). Chaque arbre est petit et faible seul, mais la séquence d'arbres se compose en un apprenant puissant.

Le gradient boosting domine régulièrement les classements sur les données tabulaires — des compétitions Kaggle à la détection de fraude et aux systèmes de prédiction de clics en production. XGBoost, LightGBM et CatBoost sont tous des implémentations de cette idée fondamentale.

## L'intuition

Imaginez que vous apprenez à lancer des fléchettes. Au premier lancer, vous manquez la cible de beaucoup — vous touchez le coin supérieur gauche. Au deuxième lancer, vous *corrigez* cette erreur en visant le coin inférieur droit. Au troisième lancer, vous corrigez l'erreur restante. Chaque lancer ne vise pas la cible directement — il vise à corriger l'erreur accumulée de tous les lancers précédents.

Le gradient boosting fonctionne de la même manière :
1. Commencer avec une prédiction simple (par ex., prédire la classe la plus fréquente pour tout le monde)
2. Calculer les erreurs (résidus) — où le modèle se trompe-t-il ?
3. Entraîner un petit arbre pour prédire ces erreurs
4. Ajouter les prédictions de l'arbre (multipliées par un taux d'apprentissage) au total courant
5. Répéter : calculer les nouveaux résidus, entraîner un autre arbre, l'ajouter

Après 50 à 100 itérations, vous avez un modèle qui a commencé simplement et a progressivement corrigé toutes ses erreurs.

## Comment ça marche

### L'algorithme (pour la classification)

1. **Initialiser** avec les probabilités a priori des classes (log-cotes) :
$$
F_0(x) = \log\left(\frac{\text{nombre}(c) + 1}{n + K}\right) \quad \text{pour chaque classe } c
$$

2. **Pour chaque itération** t = 1, ..., T :
   - Calculer les probabilités via softmax :
   $$
   p_c(x_i) = \frac{e^{F_c(x_i)}}{\sum_j e^{F_j(x_i)}}
   $$
   - Calculer les pseudo-résidus (gradient négatif de la log-perte) :
   $$
   r_{ic} = y_{ic} - p_c(x_i)
   $$
   où $y_{ic}$ vaut 1 si l'échantillon i appartient à la classe c, sinon 0.
   - Ajuster une souche de régression $h_{tc}$ aux résidus $r_{ic}$ pour chaque classe
   - Mettre à jour :
   $$
   F_c(x) \leftarrow F_c(x) + \eta \cdot h_{tc}(x)
   $$
   où $\eta$ est le taux d'apprentissage

3. **Prédire** en prenant l'argmax de softmax(F(x))

### Pourquoi « Gradient » Boosting ?

Les pseudo-résidus sont le *gradient négatif* de la fonction de perte. Chaque arbre effectue un pas de descente de gradient dans l'espace des fonctions. Au lieu de mettre à jour les paramètres du modèle (comme dans les réseaux de neurones), on ajoute une nouvelle fonction (arbre) qui pointe dans la direction de la descente la plus raide.

### Le taux d'apprentissage

Le taux d'apprentissage $\eta$ (typiquement 0,01 à 0,3) contrôle la contribution de chaque arbre. Des taux plus bas nécessitent plus d'arbres mais produisent des modèles plus lisses et plus généralisables. On parle parfois de « rétrécissement » (shrinkage).

$$
\text{Modèle} = F_0 + \eta \cdot h_1 + \eta \cdot h_2 + \cdots + \eta \cdot h_T
$$

## En Rust

### Classification binaire

```rust
use ndarray::{array, Array2};
use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
use ix_ensemble::traits::EnsembleClassifier;
use ix_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Détection de fraude : [montant, heure, distance_km, carte_présente]
    let x = Array2::from_shape_vec((8, 4), vec![
        25.0,  12.0,  2.0,  1.0,    // légitime
        15.0,  14.0,  1.0,  1.0,    // légitime
        120.0, 10.0,  5.0,  1.0,    // légitime
        45.0,  18.0,  0.5,  1.0,    // légitime
        4500.0, 3.0, 800.0, 0.0,    // fraude
        2200.0, 2.0, 500.0, 0.0,    // fraude
        3100.0, 4.0, 650.0, 0.0,    // fraude
        5000.0, 1.0, 900.0, 0.0,    // fraude
    ]).unwrap();
    let y = array![0, 0, 0, 0, 1, 1, 1, 1];

    let mut gbc = GradientBoostedClassifier::new(50, 0.1);
    gbc.fit(&x, &y);

    let preds = gbc.predict(&x);
    let proba = gbc.predict_proba(&x);

    println!("Prédictions : {}", preds);
    println!("Exactitude : {:.4}", accuracy(&y, &preds));
    println!("Précision (fraude) : {:.4}", precision(&y, &preds, 1));
    println!("Rappel (fraude) : {:.4}", recall(&y, &preds, 1));

    // Estimations de probabilité
    for i in 0..x.nrows() {
        println!("  Échantillon {} : {:.1} % fraude", i, proba[[i, 1]] * 100.0);
    }
}
```

## Quand l'utiliser

| Situation | Gradient Boosting | Alternative | Pourquoi |
|-----------|-------------------|-------------|----------|
| Précision maximale sur données tabulaires | Oui | — | Meilleur performeur de manière constante |
| Baseline rapide, réglage minimal | Non | Forêt aléatoire | Le GBM nécessite un réglage du taux d'apprentissage |
| Très petit jeu de données (< 50 échantillons) | Non | KNN, Rég. logistique | Risque de surajustement |
| Calibration de probabilités | Correct | Régression logistique | Les sorties softmax du GBM sont raisonnablement calibrées |
| Prédiction temps réel, faible latence | Peut-être | Modèle linéaire | La prédiction est séquentielle à travers les arbres |
| Interprétabilité requise | Non | Arbre de décision | Un ensemble d'arbres est une boîte noire |

## Paramètres clés

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `n_estimators` | (requis) | Nombre d'itérations de boosting. 50 à 200 typique. |
| `learning_rate` | (requis) | Rétrécissement du pas. Plus bas = plus stable mais nécessite plus d'itérations. |
| `max_depth` | (requis) | Profondeur de chaque apprenant faible. 1 à 3 typique pour le boosting. |
| `min_samples_leaf` | `1` | Nombre minimum d'échantillons dans chaque feuille. |

### Guide de réglage

| learning_rate | n_estimators | Comportement |
|---------------|-------------|--------------|
| 0,3–0,5 | 20–50 | Entraînement rapide, risque de surajustement |
| 0,1 | 50–200 | Bon défaut |
| 0,01–0,05 | 200–1000 | Meilleure généralisation, entraînement lent |

**Règle empirique :** Diminuez le taux d'apprentissage et augmentez le nombre d'estimateurs ensemble. `lr=0.1, n=100` et `lr=0.01, n=1000` donnent souvent des résultats similaires, mais le second généralise mieux.

## Pièges

**Surajustement avec trop d'itérations.** Contrairement aux forêts aléatoires où plus d'arbres n'est jamais nuisible, le gradient boosting peut surajuster si vous ajoutez trop d'itérations. Utilisez la validation croisée pour trouver le nombre optimal.

**Taux d'apprentissage trop élevé.** Un taux élevé rend chaque arbre trop influent. Le modèle oscille autour de l'optimum au lieu de converger en douceur. Commencez à 0,1 et diminuez si nécessaire.

**Les arbres profonds annulent l'effet.** La puissance du boosting vient de la combinaison de nombreux *apprenants faibles*. Utiliser des arbres profonds (max_depth > 5) rend chaque apprenant trop fort, annulant l'effet d'ensemble et risquant le surajustement. Une profondeur de 1 à 3 est typique.

**Sensible aux valeurs aberrantes.** La perte d'erreur quadratique amplifie les valeurs aberrantes. Un seul résidu extrême peut dominer le découpage d'un arbre.

**Séquentiel, pas parallèle.** Chaque arbre dépend des précédents, donc l'entraînement est inhéremment séquentiel. Les forêts aléatoires peuvent entraîner tous les arbres en parallèle.

## Pour aller plus loin

- **Arrêt prématuré :** Surveillez la perte de validation et arrêtez quand elle commence à augmenter. Cela trouve automatiquement le nombre optimal d'itérations.
- **Importance des variables :** Comme les forêts aléatoires, comptez la fréquence d'utilisation de chaque variable pour les découpages, pondérée par la diminution de l'impureté.
- **Régularisation :** Outre le taux d'apprentissage, vous pouvez régulariser en limitant max_depth, min_samples_leaf, ou en ajoutant une pénalité L2 sur les valeurs des feuilles.
- **Support de la régression :** L'implémentation actuelle est uniquement pour la classification. Le gradient boosting pour la régression utilise la perte d'erreur quadratique au lieu de la log-perte.
