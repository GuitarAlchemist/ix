# Prédiction de l'attrition client avec ix_supervised

Un flux de travail complet de classification binaire : de l'inspection
des données brutes jusqu'à l'évaluation ROC/AUC.

## Problématique

Une entreprise de télécommunications souhaite prédire quels abonnés
résilieront leur forfait au prochain trimestre. Le coût d'un churner
non détecté (faux négatif) dépasse celui d'une offre de fidélisation
envoyée à un client loyal (faux positif) -- le **rappel sur la classe
« attrition » prime donc sur la précision globale**.

## Les données

Chaque abonné est décrit par quatre caractéristiques :

| Caractéristique   | Type  | Description                              |
|-------------------|-------|------------------------------------------|
| tenure_months     | f64   | Ancienneté en mois                       |
| monthly_charges   | f64   | Facture mensuelle moyenne (USD)           |
| support_calls     | f64   | Tickets d'assistance (90 derniers jours)  |
| contract_type     | f64   | 0 = sans engagement, 1 = contrat annuel   |

Étiquette : `0` = resté, `1` = résilié. Les jeux de données d'attrition
sont presque toujours déséquilibrés -- généralement 20 à 30 % de classe
positive.

---

## Étape 1 -- Inspecter la distribution des classes

Avant tout entraînement, vérifier l'équilibre des étiquettes.

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::resampling::class_distribution;

// 20 abonnés : 14 restés (0), 6 résiliés (1)
let y = array![0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1];
let dist = class_distribution(&y);

for (class, count, pct) in &dist {
    println!("Class {}: {} samples ({:.1}%)", class, count, pct);
}
// Class 0: 14 samples (70.0%)
// Class 1:  6 samples (30.0%)
```

Un rapport 70/30 constitue un déséquilibre modéré. Un classifieur naïf
prédisant systématiquement « resté » atteindrait 70 % de précision
globale sans détecter le moindre churner.

---

## Étape 2 -- Validation croisée avec StratifiedKFold

La découpe stratifiée garantit que chaque pli conserve le ratio 70/30,
évitant les plis où la classe minoritaire serait absente.

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::validation::{StratifiedKFold, cross_val_score};

// Features: tenure, charges, support_calls, contract_type
let x = Array2::from_shape_vec((20, 4), vec![
    24.0, 65.0, 1.0, 1.0,
    48.0, 55.0, 0.0, 1.0,
    12.0, 70.0, 2.0, 0.0,
    36.0, 50.0, 0.0, 1.0,
    60.0, 45.0, 1.0, 1.0,
    30.0, 60.0, 0.0, 1.0,
    42.0, 52.0, 1.0, 1.0,
    18.0, 58.0, 0.0, 0.0,
    54.0, 48.0, 0.0, 1.0,
    36.0, 55.0, 1.0, 1.0,
    15.0, 62.0, 0.0, 0.0,
    45.0, 50.0, 0.0, 1.0,
    28.0, 57.0, 1.0, 1.0,
    33.0, 53.0, 0.0, 1.0,
     3.0, 80.0, 4.0, 0.0,
     6.0, 75.0, 3.0, 0.0,
     2.0, 85.0, 5.0, 0.0,
     5.0, 78.0, 3.0, 0.0,
     4.0, 82.0, 4.0, 0.0,
     7.0, 73.0, 2.0, 0.0,
]).unwrap();

let y = array![0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1];

let scores = cross_val_score(
    &x, &y,
    || LogisticRegression::new()
        .with_learning_rate(0.001)
        .with_max_iterations(2000),
    5,   // 5-fold
    42,  // seed
);

let mean = scores.iter().sum::<f64>() / scores.len() as f64;
println!("Per-fold accuracy: {:?}", scores);
println!("Mean CV accuracy:  {:.3}", mean);
```

`cross_val_score` utilise en interne `StratifiedKFold`, de sorte que
chaque pli reflète les proportions de classes d'origine.

---

## Étape 3 -- Entraîner le modèle final et évaluer avec ConfusionMatrix

Une fois la validation croisée confirmant la capacité de généralisation
du modèle, on ré-entraîne sur l'ensemble complet et on examine la
matrice de confusion.

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::ConfusionMatrix;

let x = Array2::from_shape_vec((20, 4), vec![
    24.0, 65.0, 1.0, 1.0,
    48.0, 55.0, 0.0, 1.0,
    12.0, 70.0, 2.0, 0.0,
    36.0, 50.0, 0.0, 1.0,
    60.0, 45.0, 1.0, 1.0,
    30.0, 60.0, 0.0, 1.0,
    42.0, 52.0, 1.0, 1.0,
    18.0, 58.0, 0.0, 0.0,
    54.0, 48.0, 0.0, 1.0,
    36.0, 55.0, 1.0, 1.0,
    15.0, 62.0, 0.0, 0.0,
    45.0, 50.0, 0.0, 1.0,
    28.0, 57.0, 1.0, 1.0,
    33.0, 53.0, 0.0, 1.0,
     3.0, 80.0, 4.0, 0.0,
     6.0, 75.0, 3.0, 0.0,
     2.0, 85.0, 5.0, 0.0,
     5.0, 78.0, 3.0, 0.0,
     4.0, 82.0, 4.0, 0.0,
     7.0, 73.0, 2.0, 0.0,
]).unwrap();

let y = array![0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1];

let mut model = LogisticRegression::new()
    .with_learning_rate(0.001)
    .with_max_iterations(2000);
model.fit(&x, &y);

let preds = model.predict(&x);
let cm = ConfusionMatrix::from_labels(&y, &preds, 2);

println!("{}", cm.display());

let (prec, rec, f1, support) = cm.classification_report();
println!("Class | Precision | Recall | F1   | Support");
for c in 0..2 {
    println!("  {}   |   {:.3}   | {:.3} | {:.3} |   {}",
        c, prec[c], rec[c], f1[c], support[c]);
}
```

Portez une attention particulière au rappel de la classe 1 (attrition).
Un modèle affichant 95 % de précision globale mais seulement 40 % de
rappel sur l'attrition laisse échapper plus de la moitié des abonnés
à risque.

---

## Étape 4 -- Analyse ROC/AUC

Les courbes ROC évaluent le classifieur sur l'ensemble des seuils de
décision possibles, ce qui les rend insensibles au déséquilibre des
classes.

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{roc_curve, roc_auc};

let x = Array2::from_shape_vec((20, 4), vec![
    24.0, 65.0, 1.0, 1.0,
    48.0, 55.0, 0.0, 1.0,
    12.0, 70.0, 2.0, 0.0,
    36.0, 50.0, 0.0, 1.0,
    60.0, 45.0, 1.0, 1.0,
    30.0, 60.0, 0.0, 1.0,
    42.0, 52.0, 1.0, 1.0,
    18.0, 58.0, 0.0, 0.0,
    54.0, 48.0, 0.0, 1.0,
    36.0, 55.0, 1.0, 1.0,
    15.0, 62.0, 0.0, 0.0,
    45.0, 50.0, 0.0, 1.0,
    28.0, 57.0, 1.0, 1.0,
    33.0, 53.0, 0.0, 1.0,
     3.0, 80.0, 4.0, 0.0,
     6.0, 75.0, 3.0, 0.0,
     2.0, 85.0, 5.0, 0.0,
     5.0, 78.0, 3.0, 0.0,
     4.0, 82.0, 4.0, 0.0,
     7.0, 73.0, 2.0, 0.0,
]).unwrap();

let y = array![0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1];

let mut model = LogisticRegression::new()
    .with_learning_rate(0.001)
    .with_max_iterations(2000);
model.fit(&x, &y);

let probas = model.predict_proba(&x);
let scores = probas.column(1).to_owned();

let (fpr, tpr, thresholds) = roc_curve(&y, &scores);
let auc = roc_auc(&fpr, &tpr);

println!("AUC = {:.3}", auc);
println!("\nSample ROC points (FPR, TPR, Threshold):");
for i in (0..fpr.len()).step_by(fpr.len() / 5) {
    println!("  ({:.3}, {:.3}, {:.3})", fpr[i], tpr[i], thresholds[i]);
}
```

Une AUC de 0,5 signifie un classement aléatoire ; 1,0 une séparation
parfaite. Une AUC supérieure à 0,80 est considérée comme suffisante
pour une mise en production.

---

## Points clés à retenir

1. **Toujours inspecter la distribution des classes en premier.**
   `class_distribution` révèle si la précision globale est fiable ou
   trompeuse.
2. **Utiliser la validation croisée stratifiée.** `StratifiedKFold`
   empêche les plis où la classe minoritaire est sous-représentée.
3. **Lire la matrice de confusion, pas seulement la précision globale.**
   Le rappel sur la classe 1 répond à la question : « Parmi tous les
   churners réels, combien en avons-nous détecté ? »
4. **La courbe ROC/AUC offre une vue indépendante du seuil.** Les
   courbes ROC permettent de choisir le bon compromis après
   l'entraînement.
5. **Envisager le rééchantillonnage en cas de déséquilibre sévère.**
   Sous les 10-15 % de classe minoritaire, `Smote` peut synthétiser des
   échantillons avant l'entraînement.
*Crates utilisés : `ix_supervised` (régression logistique, validation, métriques, rééchantillonnage).*
