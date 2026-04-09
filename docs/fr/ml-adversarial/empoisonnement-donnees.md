# Détection d'empoisonnement de données

## Le problème

Vous entraînez un classificateur de spam à partir d'exemples signalés par les utilisateurs. Un adversaire qui contrôle une fraction des signalements peut injecter des échantillons mal étiquetés — des emails légitimes étiquetés comme spam, ou du spam étiqueté comme légitime — pour corrompre le modèle. Vous avez besoin de méthodes automatisées pour détecter et supprimer ces échantillons empoisonnés avant qu'ils ne dégradent votre classificateur.

L'empoisonnement de données menace aussi l'apprentissage automatique en imagerie médicale (labels de radiologie falsifiés), la conduite autonome (annotations lidar corrompues) et tout système entraîné sur des données crowdsourcées ou scrapées.

## L'intuition

Les échantillons empoisonnés sont des imposteurs : ils ont les caractéristiques d'une classe mais le label d'une autre. Les méthodes de détection cherchent les échantillons qui « n'ont pas leur place » parmi leurs voisins étiquetés :

- **Cohérence de labels KNN :** Si les K plus proches voisins d'un échantillon sont majoritairement en désaccord avec son label, il est suspect. Une photo de chien étiquetée « chat » sera entourée d'autres photos de chiens.
- **Fonctions d'influence :** Estimer combien la suppression d'un seul échantillon d'entraînement changerait la prédiction du modèle sur un point de test. Les échantillons empoisonnés ont une influence disproportionnée.
- **Signatures spectrales :** Les attaques par porte dérobée intègrent un motif cohérent dans les échantillons empoisonnés. Ce motif apparaît comme une direction aberrante dans la covariance des caractéristiques. L'itération de puissance trouve cette direction, et les échantillons avec de grandes projections sur celle-ci sont signalés.

## Comment ça fonctionne

### Détection de retournement de labels KNN

Pour chaque échantillon, trouver ses K plus proches voisins par distance euclidienne. Si le label majoritaire parmi les voisins diffère du label de l'échantillon, le signaler.

### Fonction d'influence (simplifiée)

Approximer combien un point d'entraînement affecte une prédiction de test en utilisant le gradient et le hessien de la perte. Les points à haute influence peuvent être empoisonnés.

### Défense par signature spectrale

Par classe, centrer les caractéristiques, trouver le vecteur singulier dominant par itération de puissance, et signaler les échantillons dont la projection sur ce vecteur dépasse un seuil en percentile.

## En Rust

```rust
use ix_adversarial::poisoning::{
    detect_label_flips,
    influence_function,
    spectral_signature_defense,
};
use ndarray::{array, Array2};

// Détection de retournement de labels par KNN
let features = Array2::from_shape_vec((6, 2), vec![
    0.0, 0.0,  0.1, 0.1,  0.05, 0.05,   // cluster 0
    1.0, 1.0,  1.1, 1.1,  1.05, 1.05,   // cluster 1
]).unwrap();
let labels = array![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // l'indice 2 est retourné !
let suspicious = detect_label_flips(&features, &labels, 3);
assert!(suspicious.contains(&2));

// Fonction d'influence
let test_point = array![0.5, 0.5];
let influences = influence_function(&features, &labels, &test_point, 1.0, 0.1);

// Défense par signature spectrale
let flagged = spectral_signature_defense(&features, &labels, 2, 50.0);
```

## Quand l'utiliser

| Méthode | Détecte | Coût |
|--------|---------|------|
| **Retournement de labels KNN** | Échantillons mal étiquetés proches du mauvais cluster | O(N²) distances par paires |
| **Fonctions d'influence** | Points avec un impact disproportionné sur les prédictions | Nécessite une approximation du hessien |
| **Signatures spectrales** | Motifs de porte dérobée (déclencheur cohérent) | Itération de puissance par classe |

## Pièges courants

1. **KNN en difficulté aux frontières de décision.** Les échantillons proches de la vraie frontière entre classes peuvent être faussement signalés car leurs voisins sont mixtes.
2. **Les fonctions d'influence sont approximatives.** L'approximation linéaire n'est valide que près de l'optimum ; en début d'entraînement elle peut être trompeuse.
3. **La défense spectrale suppose que le poison est minoritaire.** Si plus de ~30 % d'une classe est empoisonné, la direction « aberrante » peut en fait être la majorité.

## Pour aller plus loin

- Exécutez `detect_label_flips` comme étape de prétraitement avant l'entraînement, en supprimant les échantillons signalés ou en les envoyant en revue humaine.
- Combinez avec l'entraînement adversarial de `ix_adversarial::defense` pour construire des modèles robustes à la fois contre l'évasion et l'empoisonnement.
- Utilisez le clustering de `ix_unsupervised` (K-Means) pour vérifier indépendamment la structure des classes avant l'entraînement.
- Voir [confidentialite-differentielle.md](confidentialite-differentielle.md) pour limiter l'influence de tout échantillon d'entraînement individuel via DP-SGD.
