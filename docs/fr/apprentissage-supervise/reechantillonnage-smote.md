# Rééchantillonnage & SMOTE

## Le problème

Vous construisez un système de détection de fraude par carte de crédit. Sur 100 000 transactions dans vos données d'entraînement, seules 500 sont frauduleuses — un taux de positifs de 0,5 %. Vous entraînez un classifieur et il affiche 99,5 % d'exactitude. Impressionnant ? Non. Un modèle qui prédit bêtement « pas de fraude » pour chaque transaction obtient exactement la même exactitude. Votre modèle n'a rien appris sur la fraude — il a appris à l'ignorer.

C'est le problème du déséquilibre des classes. Quand une classe dépasse largement l'autre, les classifieurs prennent le chemin de moindre résistance : prédire la classe majoritaire pour tout. La fonction de perte ne voit aucune raison de bien classifier les 0,5 % minoritaires quand obtenir les 99,5 % majoritaires est tellement plus facile.

Il y a deux stratégies : augmenter la classe minoritaire (suréchantillonnage) ou réduire la classe majoritaire (sous-échantillonnage). SMOTE — Synthetic Minority Over-sampling Technique — est la méthode de suréchantillonnage la plus utilisée. Au lieu de dupliquer les échantillons minoritaires existants (ce qui cause le surajustement à des copies exactes), elle génère de *nouveaux* échantillons synthétiques en interpolant entre les échantillons existants.

## L'intuition

Imaginez une carte avec 500 épingles rouges (fraude) regroupées dans quelques quartiers, et 99 500 épingles bleues (légitime) réparties partout. Un classifieur regardant cette carte tracera des frontières qui ignorent les groupes rouges parce qu'ils sont si petits comparés à l'océan bleu.

La solution de SMOTE : pour chaque épingle rouge, trouver ses voisines rouges les plus proches et placer de nouvelles épingles rouges le long des lignes qui les relient. Si deux cas de fraude sont aux coordonnées (3, 5) et (5, 7), SMOTE pourrait placer un échantillon synthétique à (4, 6) — à mi-chemin entre eux. Cela remplit le « quartier de la fraude » avec des cas synthétiques plausibles, le rendant assez grand pour que le classifieur ne puisse plus l'ignorer.

L'idée clé est qu'interpoler entre des échantillons minoritaires réels produit de *nouveaux échantillons plausibles*. Un point entre deux vrais cas de fraude ressemble probablement aussi à un cas de fraude. C'est bien mieux que le suréchantillonnage aléatoire (duplication), qui force le classifieur à mémoriser des exemples spécifiques.

## Comment ça marche

### Algorithme SMOTE

Pour chaque échantillon minoritaire x_i :
1. Trouver ses k plus proches voisins au sein de la même classe
2. Sélectionner aléatoirement un voisin x_nn
3. Générer un échantillon synthétique le long du segment :

$$
x_{new} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim \text{Uniforme}(0, 1)
$$

4. Répéter jusqu'à ce que la classe minoritaire atteigne le nombre cible

### Sous-échantillonnage aléatoire

L'alternative plus simple : supprimer aléatoirement des échantillons majoritaires jusqu'à équilibrer les classes. Rapide et facile, mais gaspille des données potentiellement utiles.

### Choisir une stratégie

| Stratégie | Avantages | Inconvénients |
|-----------|-----------|---------------|
| SMOTE (suréchantillonner la minorité) | Aucune donnée perdue, crée des échantillons plausibles | Peut créer des échantillons bruités près des frontières |
| Sous-échantillonnage aléatoire | Rapide, simple, réduit le temps d'entraînement | Jette des données potentiellement utiles |
| Combiner les deux | Le meilleur des deux mondes | Plus complexe à régler |

## En Rust

### Diagnostiquer le déséquilibre

```rust
use ndarray::array;
use ix_supervised::resampling::class_distribution;

fn main() {
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
    let dist = class_distribution(&y);

    for (class, count, pct) in &dist {
        println!("Classe {} : {} échantillons ({:.1} %)", class, count, pct);
    }
    // Classe 0 : 8 échantillons (80,0 %)
    // Classe 1 : 2 échantillons (20,0 %)
}
```

### Suréchantillonnage SMOTE

```rust
use ndarray::{array, Array2};
use ix_supervised::resampling::{Smote, class_distribution};

fn main() {
    // Déséquilibré : 8 légitimes, 2 fraudes
    let x = Array2::from_shape_vec((10, 2), vec![
        0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
        0.8, 0.1,  0.2, 0.7,  0.6, 0.3,  0.4, 0.8,
        5.0, 5.0,  5.5, 5.5,
    ]).unwrap();
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

    println!("Avant SMOTE :");
    for (c, n, pct) in class_distribution(&y) {
        println!("  Classe {} : {} ({:.0} %)", c, n, pct);
    }

    let smote = Smote::new(5, 42);
    let (x_balanced, y_balanced) = smote.fit_resample(&x, &y);

    println!("Après SMOTE :");
    for (c, n, pct) in class_distribution(&y_balanced) {
        println!("  Classe {} : {} ({:.0} %)", c, n, pct);
    }
    // Classe 0 : 8 (50 %), Classe 1 : 8 (50 %)
}
```

### Sous-échantillonnage aléatoire

```rust
use ndarray::{array, Array2};
use ix_supervised::resampling::random_undersample;

fn main() {
    let x = Array2::from_shape_vec((10, 2), vec![
        0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
        0.8, 0.1,  0.2, 0.7,  0.6, 0.3,  0.4, 0.8,
        5.0, 5.0,  5.5, 5.5,
    ]).unwrap();
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

    let (x_under, y_under) = random_undersample(&x, &y, 42);
    // 2 classe-0, 2 classe-1 — équilibré en supprimant des échantillons majoritaires
    println!("Sous-échantillonné : {} échantillons", y_under.len()); // 4
}
```

### Pipeline complet : SMOTE + évaluation

```rust
use ndarray::{array, Array2};
use ix_supervised::resampling::Smote;
use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{accuracy, recall, ConfusionMatrix};

fn main() {
    // Jeu de données déséquilibré
    let x = Array2::from_shape_vec((12, 2), vec![
        0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
        0.8, 0.1,  0.2, 0.7,  0.6, 0.3,  0.4, 0.8,
        0.9, 0.4,  0.1, 0.6,
        5.0, 5.0,  5.5, 5.5,
    ]).unwrap();
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

    // Étape 1 : Équilibrer avec SMOTE
    let smote = Smote::new(1, 42);
    let (x_bal, y_bal) = smote.fit_resample(&x, &y);

    // Étape 2 : Entraîner sur les données équilibrées
    let mut tree = DecisionTree::new(5);
    tree.fit(&x_bal, &y_bal);

    // Étape 3 : Évaluer sur les données originales
    let preds = tree.predict(&x);
    let cm = ConfusionMatrix::from_labels(&y, &preds, 2);

    println!("Rappel (fraude) : {:.4}", recall(&y, &preds, 1));
    println!("{}", cm.display());
}
```

## Quand l'utiliser

| Situation | Stratégie | Pourquoi |
|-----------|-----------|----------|
| Minorité < 10 % des données | SMOTE | Le classifieur ignorera la minorité sans aide |
| Minorité 10–30 % des données | Peut-être SMOTE | Essayer sans d'abord ; ajouter si le rappel est bas |
| Données équilibrées (40–60 %) | Aucune | Pas de déséquilibre à corriger |
| Très petite minorité (< 10 échantillons) | Sous-échantillonner ou collecter plus de données | SMOTE ne peut pas bien interpoler avec trop peu de graines |
| Grand jeu de données (100K+ majorité) | Sous-éch. + SMOTE | Réduire la majorité d'abord, puis SMOTE la minorité |
| Séries temporelles / données ordonnées | Attention | SMOTE suppose i.i.d. ; les échantillons synthétiques peuvent violer l'ordre temporel |

## Paramètres clés

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `k` | (requis) | Nombre de plus proches voisins pour l'interpolation. 5 est standard. |
| `seed` | (requis) | Graine aléatoire pour la reproductibilité. |
| `target_ratio` | `1.0` | Ratio minorité/majorité. 1,0 = équilibre complet. 0,5 = suréchantillonner à la moitié. |

## Pièges

**Appliquer SMOTE uniquement aux données d'entraînement.** Ne jamais rééchantillonner le jeu de test. Des échantillons synthétiques dans le test rendent l'évaluation absurde — vous testeriez sur des données fabriquées. Découper d'abord, puis appliquer SMOTE au pli d'entraînement.

**SMOTE près des frontières de classes crée du bruit.** Si un échantillon minoritaire se trouve juste à côté d'échantillons majoritaires, SMOTE interpole vers la frontière, créant des échantillons synthétiques ambigus. Des variantes comme Borderline-SMOTE et SMOTE-Tomek traitent ce problème.

**Les classes à échantillon unique ne peuvent pas être SMOTÉes.** Il faut au moins 2 échantillons dans une classe pour interpoler entre eux. Les classes avec un seul échantillon sont ignorées.

**SMOTE n'aide pas si les classes ne sont pas géométriquement regroupées.** L'hypothèse d'interpolation est que la ligne entre deux échantillons minoritaires reste en territoire minoritaire. Si les échantillons minoritaires sont dispersés parmi les majoritaires, les points synthétiques peuvent atterrir en région majoritaire.

**Combiner avec les bonnes métriques.** SMOTE corrige les données d'entraînement, mais il faut toujours évaluer avec le rappel, la précision, le F1 et l'AUC — pas l'exactitude. Voir [métriques-évaluation.md](./metriques-evaluation.md).

## Pour aller plus loin

- **Borderline-SMOTE :** Ne suréchantillonner que les échantillons minoritaires proches de la frontière de décision (les cas les plus difficiles), pas ceux au cœur du groupe minoritaire.
- **SMOTE + liens de Tomek :** Après SMOTE, supprimer les liens de Tomek (paires de plus proches voisins de classes différentes) pour nettoyer la zone frontière.
- **ADASYN :** Adaptive Synthetic Sampling — génère plus d'échantillons synthétiques pour les exemples minoritaires les plus difficiles à apprendre (plus de voisins majoritaires).
- **Apprentissage sensible au coût :** Au lieu de rééchantillonner, assigner des coûts de mauvaise classification plus élevés à la classe minoritaire. Le modèle apprend à prêter plus d'attention aux cas rares sans modifier les données.
