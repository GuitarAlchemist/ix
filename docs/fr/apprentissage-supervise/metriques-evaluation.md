# Métriques d'évaluation

## Le problème

Vous avez entraîné un modèle de détection de fraude qui affiche 99 % de précision. Votre manager est impressionné. Mais vous réalisez alors que seulement 0,5 % des transactions sont frauduleuses — un modèle qui prédit bêtement « pas de fraude » pour chaque transaction obtiendrait 99,5 % de précision. Votre « bon » modèle est en réalité *pire* que de ne rien faire. L'exactitude vous a menti.

Cela arrive constamment en apprentissage automatique. La métrique que vous choisissez pour évaluer votre modèle détermine ce que vous optimisez, et la mauvaise métrique peut vous amener à déployer un modèle qui échoue sur ce qui compte vraiment.

## L'intuition

Pensez à une alarme incendie. Elle peut faire deux types d'erreurs :
- **Faux positif :** L'alarme se déclenche mais il n'y a pas de feu (agaçant, mais tout le monde est en sécurité).
- **Faux négatif :** Il y a un feu mais l'alarme reste silencieuse (catastrophique).

Vous accepteriez volontiers quelques fausses alarmes pour vous assurer que l'alarme ne rate jamais un vrai feu. Cela signifie que vous vous souciez davantage du **rappel** (détecter tous les feux) que de la **précision** (chaque alarme est un vrai feu).

Maintenant pensez à un tribunal. Le système peut faire deux types d'erreurs :
- **Faux positif :** Une personne innocente est condamnée (dévastateur).
- **Faux négatif :** Une personne coupable est libérée (mauvais, mais moins irréversible).

« Présumé innocent jusqu'à preuve du contraire » signifie que le système judiciaire privilégie la **précision** (chaque condamnation doit être correcte) au **rappel** (attraper chaque criminel).

## Comment ça marche

### La matrice de confusion

Chaque prédiction de classification tombe dans l'une des quatre catégories :

|  | Prédit positif | Prédit négatif |
|---|---|---|
| **Réellement positif** | Vrai Positif (VP) | Faux Négatif (FN) |
| **Réellement négatif** | Faux Positif (FP) | Vrai Négatif (VN) |

VP = fraude correctement détectée. VN = transaction légitime correctement acceptée. FP = transaction légitime faussement bloquée. FN = fraude qui est passée à travers.

### Exactitude (Accuracy)

$$
\text{Exactitude} = \frac{VP + VN}{VP + VN + FP + FN}
$$

Sur toutes les prédictions, quelle fraction était correcte ? Traite toutes les erreurs de manière égale. Correct quand les classes sont équilibrées (environ 50/50), mais trompeur quand une classe domine.

### Précision

$$
\text{Précision} = \frac{VP}{VP + FP}
$$

De tout ce que le modèle a *désigné comme positif*, quelle fraction l'était réellement ? Une haute précision signifie peu de fausses alarmes. Quand le modèle dit « fraude », vous pouvez lui faire confiance.

### Rappel (Sensibilité)

$$
\text{Rappel} = \frac{VP}{VP + FN}
$$

De tout ce qui *était réellement positif*, quelle fraction le modèle a-t-il détecté ? Un haut rappel signifie peu de cas ratés. Le modèle trouve la plupart des vraies fraudes.

### Score F1

$$
F_1 = \frac{2 \cdot \text{Précision} \cdot \text{Rappel}}{\text{Précision} + \text{Rappel}}
$$

La moyenne harmonique de la précision et du rappel. Un seul chiffre qui équilibre les deux. La moyenne harmonique pénalise les déséquilibres extrêmes — si la précision est 1,0 mais le rappel est 0,1, le F1 n'est que 0,18, pas 0,55 comme le suggérerait la moyenne arithmétique.

### Courbe ROC et AUC

La courbe ROC (Receiver Operating Characteristic) trace le taux de vrais positifs (rappel) contre le taux de faux positifs à différents seuils de classification.

- **Coin supérieur gauche (0,0 ; 1,0)** = classificateur parfait
- **Diagonale** = classificateur aléatoire
- **AUC = 1,0** = séparation parfaite entre les classes
- **AUC = 0,5** = pas mieux que le hasard

L'AUC (Area Under Curve) résume la courbe en un seul chiffre : la probabilité que le modèle classe un positif aléatoire devant un négatif aléatoire.

### Log Loss (Entropie croisée binaire)

$$
\text{Log Loss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]
$$

Pénalise non seulement les erreurs mais aussi la *confiance* dans les erreurs. Prédire 0,99 pour un échantillon qui est en fait 0 est catastrophique pour le log loss.

## En Rust

### Matrice de confusion

```rust
use ndarray::array;
use ix_supervised::metrics::ConfusionMatrix;

fn main() {
    let y_vrai = array![0, 0, 1, 1, 2, 2];
    let y_pred = array![0, 1, 1, 1, 2, 0];

    let cm = ConfusionMatrix::from_labels(&y_vrai, &y_pred, 3);
    println!("{}", cm.display());

    // Rapport de classification par classe
    let (prec, rappel, f1, support) = cm.classification_report();
    for c in 0..3 {
        println!("Classe {} : précision={:.2}, rappel={:.2}, f1={:.2}, support={}",
            c, prec[c], rappel[c], f1[c], support[c]);
    }

    println!("Exactitude globale : {:.4}", cm.accuracy());
}
```

### Courbe ROC et AUC

```rust
use ndarray::array;
use ix_supervised::metrics::{roc_curve, roc_auc, auc_score};

fn main() {
    // Détection de fraude : étiquettes binaires et scores de probabilité
    let y_vrai = array![0, 0, 0, 1, 1, 1];
    let y_scores = array![0.1, 0.3, 0.4, 0.6, 0.8, 0.95];

    // Méthode détaillée
    let (taux_fp, taux_vp, seuils) = roc_curve(&y_vrai, &y_scores);
    let auc = roc_auc(&taux_fp, &taux_vp);
    println!("AUC (détaillé) : {:.4}", auc);

    // Méthode raccourcie
    let auc_rapide = auc_score(&y_vrai, &y_scores);
    println!("AUC (raccourci) : {:.4}", auc_rapide);

    // Afficher les points de la courbe ROC
    println!("\nCourbe ROC :");
    for i in 0..taux_fp.len() {
        println!("  TFP={:.2}, TVP={:.2}, seuil={:.2}", taux_fp[i], taux_vp[i], seuils[i]);
    }
}
```

### Moyennes macro et pondérée

```rust
use ndarray::array;
use ix_supervised::metrics::{precision_avg, recall_avg, f1_avg, Average};

fn main() {
    let y_vrai = array![0, 0, 0, 1, 1, 1, 2, 2];
    let y_pred = array![0, 0, 1, 1, 1, 2, 2, 2];

    // Macro : chaque classe a le même poids
    println!("F1 macro : {:.4}", f1_avg(&y_vrai, &y_pred, Average::Macro));

    // Pondérée : poids proportionnel au nombre d'échantillons par classe
    println!("F1 pondérée : {:.4}", f1_avg(&y_vrai, &y_pred, Average::Weighted));
}
```

### Log Loss

```rust
use ndarray::array;
use ix_supervised::metrics::log_loss;

fn main() {
    let y_vrai = array![0, 0, 1, 1];

    // Bonnes prédictions → faible log loss
    let y_bonnes = array![0.1, 0.2, 0.8, 0.9];
    println!("Bonnes prédictions — log loss : {:.4}", log_loss(&y_vrai, &y_bonnes));

    // Mauvaises prédictions → fort log loss
    let y_mauvaises = array![0.9, 0.8, 0.2, 0.1];
    println!("Mauvaises prédictions — log loss : {:.4}", log_loss(&y_vrai, &y_mauvaises));
}
```

## Quand utiliser quelle métrique

| Métrique | Utiliser quand | NE PAS utiliser quand |
|----------|---------------|----------------------|
| **Exactitude** | Classes équilibrées (50/50) | Classes déséquilibrées (99/1) |
| **Précision** | Les faux positifs coûtent cher (filtre anti-spam) | Les faux négatifs sont le plus grand risque |
| **Rappel** | Les faux négatifs coûtent cher (dépistage du cancer) | Les fausses alarmes sont le plus grand risque |
| **F1** | Les deux types d'erreurs comptent | Un type d'erreur est clairement plus important |
| **AUC** | Comparer des modèles indépendamment du seuil | Vous avez besoin d'un seuil spécifique |
| **Log Loss** | Vous vous souciez de la calibration des probabilités | Vous ne vous souciez que des prédictions discrètes |
| **MSE/RMSE** | Régression, les grosses erreurs sont disproportionnées | Vous voulez la robustesse aux valeurs aberrantes |
| **MAE** | Régression, robustesse aux valeurs aberrantes | Les grosses erreurs doivent être pénalisées |
| **R²** | « Quelle part de la variance mon modèle explique-t-il ? » | Comparer des modèles entre différents jeux de données |

### Guide de décision pour la classification déséquilibrée

| Scénario | Métrique principale | Métrique secondaire |
|----------|--------------------|--------------------|
| Détection de fraude (fraude rare, manques coûteux) | Rappel | F1 |
| Filtre anti-spam (faux positifs agaçants) | Précision | F1 |
| Dépistage médical (ne pas rater une maladie) | Rappel | Précision |
| Résultats de moteur de recherche | Précision | Rappel |
| Classification binaire équilibrée | F1 ou Exactitude | Précision, Rappel |

## Pièges

**L'exactitude sur des données déséquilibrées est trompeuse.** Si 99,5 % des transactions sont légitimes, un modèle qui prédit toujours « légitime » obtient 99,5 % d'exactitude mais ne détecte aucune fraude. Vérifiez toujours précision et rappel quand les classes sont déséquilibrées.

**Précision et rappel sont inversement liés.** Abaisser le seuil de classification (par ex., de 0,5 à 0,3) capture plus de positifs (rappel plus élevé) mais signale aussi plus de négatifs (précision plus basse). Vous ne pouvez pas maximiser les deux simultanément.

**Les métriques par classe sont essentielles.** L'exactitude globale peut masquer une performance médiocre sur la classe minoritaire. Calculez toujours précision, rappel et F1 *pour chaque classe individuellement*.

**R² peut être négatif.** Ce n'est pas un bug — cela signifie que les prédictions du modèle sont pires que la simple prédiction de la moyenne. Si vous voyez un R² négatif, votre modèle a un problème fondamental.

## Pour aller plus loin

- **Réglage du seuil :** La plupart des classifieurs produisent des probabilités. Par défaut, le seuil est 0,5, mais vous pouvez l'ajuster. L'abaisser à 0,3 augmente le rappel au détriment de la précision. Tracez la courbe précision-rappel à différents seuils.
- **Moyennes macro vs. micro :** Pour les problèmes multi-classes, vous pouvez calculer le F1 par classe et moyenner (macro), ou regrouper tous les VP/FP/FN (micro). Macro traite chaque classe de manière égale ; micro pondère par la fréquence des classes.
- **Validation croisée :** Au lieu d'un seul découpage train/test, utilisez la validation croisée k-fold pour obtenir des estimations de métriques plus fiables. Voir [validation-croisée.md](./validation-croisee.md).
