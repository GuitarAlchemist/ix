# Cas pratique : Détection de fraude par carte bancaire avec SMOTE

> Combattre le déséquilibre extrême des classes grâce au suréchantillonnage synthétique et au gradient boosting.

## Le problème

Vous recevez 100 000 transactions par carte bancaire chaque jour. Seulement 1 %
sont frauduleuses -- soit 1 000 transactions frauduleuses cachées parmi 99 000
transactions légitimes. Un modèle naïf qui prédit systématiquement
« légitime » atteint 99 % de précision globale et ne détecte aucune fraude.
**L'exactitude est un mensonge lorsque les classes sont déséquilibrées.**

L'idée clé : il faut enseigner au modèle à quoi ressemble la fraude en lui
fournissant suffisamment d'exemples. SMOTE (Synthetic Minority Over-sampling
Technique) génère des échantillons de fraude synthétiques et réalistes afin
que le classifieur puisse apprendre des frontières de décision pertinentes.

## Les données

Chaque transaction possède quatre caractéristiques :

| Caractéristique | Description                            | Plage       |
|-----------------|----------------------------------------|-------------|
| `amount`        | Montant de la transaction en dollars   | 0,5 -- 5000 |
| `hour`          | Heure de la journée (0--23)            | 0 -- 23     |
| `distance_km`   | Distance par rapport au domicile       | 0 -- 500    |
| `card_present`  | Carte physique utilisée (1) ou non (0) | 0 ou 1      |

Étiquettes : `0` = légitime, `1` = fraude.

## Étape 1 : Diagnostiquer le déséquilibre

Avant toute chose, mesurons la distribution des classes :

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::resampling::class_distribution;

// Jeu de données simulé : 20 légitimes, 2 frauduleuses (ratio 10:1)
let y = array![
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1,
];

let dist = class_distribution(&y);
for (class, count, pct) in &dist {
    let label = if *class == 0 { "Legitimate" } else { "Fraud" };
    println!("{}: {} samples ({:.1}%)", label, count, pct);
}
// Legitimate: 20 samples (90.9%)
// Fraud:       2 samples  (9.1%)
```

Le classifieur verra 10 transactions légitimes pour chaque fraude -- il
apprendra à ignorer complètement la classe minoritaire.

## Étape 2 : Appliquer SMOTE pour équilibrer les données d'entraînement

SMOTE génère des échantillons de fraude synthétiques en interpolant entre les
exemples de fraude existants et leurs k plus proches voisins. Point essentiel :
**appliquer SMOTE uniquement sur l'ensemble d'entraînement**, jamais sur
l'ensemble de test.

```rust
use ix_supervised::resampling::{Smote, class_distribution};

// Caractéristiques : [amount, hour, distance_km, card_present]
// 20 légitimes + 2 frauduleuses
let x = Array2::from_shape_vec((22, 4), vec![
    // -- Transactions légitimes (classe 0) --
    45.0, 10.0,  2.0, 1.0,   120.0, 14.0,  5.0, 1.0,
    30.0, 9.0,   1.0, 1.0,    85.0, 16.0,  3.0, 1.0,
    15.0, 11.0,  0.5, 1.0,   200.0, 13.0,  8.0, 1.0,
    55.0, 15.0,  4.0, 1.0,    70.0, 17.0,  2.5, 1.0,
    25.0, 12.0,  1.5, 1.0,    90.0, 10.0,  3.5, 1.0,
    40.0, 8.0,   2.0, 1.0,   110.0, 14.0,  6.0, 1.0,
    35.0, 9.0,   1.0, 1.0,    60.0, 16.0,  2.0, 1.0,
    20.0, 11.0,  0.8, 1.0,   150.0, 13.0,  7.0, 1.0,
    50.0, 15.0,  3.0, 1.0,    75.0, 17.0,  4.0, 1.0,
    28.0, 12.0,  1.2, 1.0,    95.0, 10.0,  5.0, 1.0,
    // -- Transactions frauduleuses (classe 1) --
    980.0, 3.0, 350.0, 0.0,  1500.0, 2.0, 420.0, 0.0,
]).unwrap();

let y = array![
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1,
];

// Appliquer SMOTE : k=1 (seulement 2 échantillons minoritaires), graine=42
let smote = Smote::new(1, 42);
let (x_bal, y_bal) = smote.fit_resample(&x, &y);

// Comparer avant/après
println!("Before SMOTE:");
for (cls, count, pct) in class_distribution(&y) {
    println!("  Class {}: {} ({:.1}%)", cls, count, pct);
}

println!("After SMOTE:");
for (cls, count, pct) in class_distribution(&y_bal) {
    println!("  Class {}: {} ({:.1}%)", cls, count, pct);
}
// Before: Class 0: 20 (90.9%), Class 1: 2 (9.1%)
// After:  Class 0: 20 (50.0%), Class 1: 20 (50.0%)
```

SMOTE a créé 18 échantillons de fraude synthétiques en interpolant entre les
deux transactions frauduleuses originales. Le classifieur voit désormais un
nombre égal des deux classes.

## Étape 3 : Entraîner un classifieur à gradient boosting

Le gradient boosting construit un ensemble d'apprenants faibles (des souches de
décision) qui corrigent séquentiellement les erreurs des précédents -- idéal
pour les schémas de fraude complexes.

```rust
use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
use ix_ensemble::traits::EnsembleClassifier;

// Entraîner sur les données équilibrées par SMOTE
let mut gbc = GradientBoostedClassifier::new(
    100,  // 100 itérations de boosting
    0.1,  // taux d'apprentissage
    3,    // profondeur maximale
);
gbc.fit(&x_bal, &y_bal);

// Prédire sur les données originales (non équilibrées) comme vérification
let predictions = gbc.predict(&x);
let probabilities = gbc.predict_proba(&x);

// Vérifier les probabilités de fraude pour les échantillons connus
println!("Fraud probability for sample 20: {:.3}", probabilities[[20, 1]]);
println!("Fraud probability for sample 21: {:.3}", probabilities[[21, 1]]);
```

## Étape 4 : Évaluer avec la matrice de confusion, précision, rappel, F1 et AUC

Ne faites jamais confiance à l'exactitude seule. Utilisez des métriques qui
révèlent la capacité du modèle à détecter la classe minoritaire.

```rust
use ix_supervised::metrics::{
    ConfusionMatrix, precision, recall, f1_score, auc_score,
};

let y_pred = gbc.predict(&x);

// Matrice de confusion
let cm = ConfusionMatrix::from_labels(&y, &y_pred, 2);
println!("{}", cm.display());
// pred ->  0    1
// true 0: 20    0   (toutes les légitimes correctement classées)
// true 1:  0    2   (les deux fraudes détectées)

let (prec_vec, rec_vec, f1_vec, support) = cm.classification_report();
println!("Fraud precision: {:.3}", prec_vec[1]);
println!("Fraud recall:    {:.3}", rec_vec[1]);
println!("Fraud F1:        {:.3}", f1_vec[1]);

// AUC à partir des probabilités prédites
let y_scores = Array1::from_iter(
    (0..x.nrows()).map(|i| probabilities[[i, 1]])
);
let auc = auc_score(&y, &y_scores);
println!("AUC: {:.3}", auc);
```

## Étape 5 : Comparer avec et sans SMOTE

Le véritable gain -- montrer que SMOTE améliore considérablement le rappel
sur la fraude.

```rust
// --- SANS SMOTE : entraînement sur données déséquilibrées ---
let mut gbc_no_smote = GradientBoostedClassifier::new(100, 0.1);
gbc_no_smote.fit(&x, &y);  // données originales déséquilibrées
let pred_no_smote = gbc_no_smote.predict(&x);
let recall_no_smote = recall(&y, &pred_no_smote, 1);

// --- AVEC SMOTE : entraînement sur données équilibrées ---
let mut gbc_smote = GradientBoostedClassifier::new(100, 0.1);
gbc_smote.fit(&x_bal, &y_bal);  // données équilibrées par SMOTE
let pred_smote = gbc_smote.predict(&x);
let recall_smote = recall(&y, &pred_smote, 1);

println!("Recall WITHOUT SMOTE: {:.3}", recall_no_smote);
println!("Recall WITH SMOTE:    {:.3}", recall_smote);
// Sans SMOTE le modèle manque souvent la fraude (rappel ~ 0,0-0,5)
// Avec SMOTE le modèle détecte la plupart des fraudes (rappel ~ 0,5-1,0)
```

## Points clés à retenir

1. **L'exactitude ment avec des données déséquilibrées.** Un score de 99 %
   ne signifie rien lorsque 99 % des transactions sont légitimes. Utilisez
   toujours la précision, le rappel, le F1 et l'AUC pour les problèmes
   déséquilibrés.

2. **SMOTE + gradient boosting est une combinaison puissante.** SMOTE fournit
   au classifieur suffisamment d'exemples minoritaires pour apprendre. Le
   gradient boosting se concentre séquentiellement sur les échantillons
   difficiles à classifier -- exactement ce que sont les transactions
   frauduleuses.

3. **Appliquer SMOTE uniquement aux données d'entraînement.** Ne jamais
   rééchantillonner l'ensemble de test. Celui-ci doit refléter les proportions
   réelles des classes pour fournir des métriques honnêtes.

4. **Le rappel est roi pour la détection de fraude.** Manquer une fraude coûte
   bien plus cher qu'une fausse alerte. Optimisez d'abord le rappel, puis
   ajustez la précision pour réduire les faux positifs à un niveau acceptable.

5. **Surveillez la distribution des classes avant et après le
   rééchantillonnage.** La fonction `class_distribution` permet de vérifier
   facilement que SMOTE a atteint l'équilibre souhaité.

## Algorithmes utilisés

| Algorithme | Crate | Rôle |
|------------|-------|------|
| SMOTE | `ix_supervised::resampling` | Équilibrer les données d'entraînement |
| Gradient Boosted Classifier | `ix_ensemble::gradient_boosting` | Classification |
| Matrice de confusion | `ix_supervised::metrics` | Analyse des erreurs |
| Précision / Rappel / F1 | `ix_supervised::metrics` | Évaluation par classe |
| AUC | `ix_supervised::metrics` | Évaluation indépendante du seuil |
| `class_distribution` | `ix_supervised::resampling` | Diagnostic du déséquilibre |

## Documentation associée

- [SMOTE et rééchantillonnage](../apprentissage-supervise/reechantillonnage-smote.md)
- [Gradient Boosting](../apprentissage-supervise/gradient-boosting.md)
- [Métriques d'évaluation](../apprentissage-supervise/metriques-evaluation.md)
- [Validation croisée](../apprentissage-supervise/validation-croisee.md)
