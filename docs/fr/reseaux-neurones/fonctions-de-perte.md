# Fonctions de perte

> Le nombre qui indique à votre modèle à quel point il se trompe — et choisir la bonne fonction change tout.

## Le problème

Votre réseau de neurones fait une prédiction. La prédiction est fausse. Mais *à quel point* ? Et comment quantifier cette erreur pour que le modèle puisse s'améliorer ?

La fonction de perte est le bulletin de notes de votre modèle. Elle prend la prédiction et la vraie réponse, et produit un seul nombre : bas = bon, élevé = mauvais. La descente de gradient minimise ce nombre. **Différentes fonctions de perte définissent différentes notions d'erreur** — et un mauvais choix peut saboter silencieusement votre modèle.

## L'intuition

### MSE : erreur quadratique moyenne

**Mean Squared Error** — le choix par défaut pour la régression (prédire des nombres).

Imaginez une cible de fléchettes. La MSE mesure la distance moyenne entre vos fléchettes et le centre. Mais élever les distances au carré signifie qu'une fléchette à 10 cm du centre est pénalisée **100 fois plus** qu'une fléchette à 1 cm. Cela rend la MSE très sensible aux valeurs aberrantes — une seule prédiction catastrophique domine la perte.

### Entropie croisée : mesurer la surprise

**Binary Cross-Entropy** — le choix par défaut pour la classification (prédire des catégories).

Imaginez un prévisionniste météo. S'il annonce « 90 % de chances de pluie » et qu'il pleut, ce n'est pas surprenant — perte faible. S'il annonce « 10 % de chances de pluie » et qu'il pleut, c'est très surprenant — perte élevée. L'entropie croisée mesure à quel point votre modèle est « surpris » par la vraie réponse.

La différence clé avec la MSE : l'entropie croisée tient compte de la *confiance*. Prédire 0.51 pour un vrai positif et 0.99 pour un vrai positif ont le même signe mais des pertes très différentes. L'entropie croisée récompense les prédictions correctes avec confiance et pénalise fortement les prédictions fausses avec confiance.

## Comment ça fonctionne

### Erreur quadratique moyenne (MSE)

`MSE = (1/n) × Σᵢ (yᵢ - ŷᵢ)²`

En clair : pour chaque prédiction, calculer l'erreur (vraie valeur - prédiction), l'élever au carré, et faire la moyenne sur tous les échantillons.

**Gradient** : `dMSE/dŷ = (2/n) × (ŷ - y)`

En clair : le gradient est proportionnel à l'erreur. Les erreurs plus importantes produisent des gradients plus grands, poussant le modèle à les corriger plus fortement.

**Quand l'utiliser** : Régression — prédiction de valeurs continues (prix, température, score).

### Entropie croisée binaire (BCE)

`BCE = -(1/n) × Σᵢ [yᵢ × log(ŷᵢ) + (1-yᵢ) × log(1-ŷᵢ)]`

En clair : pour chaque échantillon, si l'étiquette vraie est 1, pénaliser selon -log(probabilité prédite de 1). Si l'étiquette vraie est 0, pénaliser selon -log(probabilité prédite de 0). Faire la moyenne sur tous les échantillons.

Pourquoi le logarithme ? Parce que -log(0.99) ≈ 0.01 (confiant et correct → perte minuscule) tandis que -log(0.01) ≈ 4.6 (confiant et faux → perte énorme). Cela crée un gradient abrupt quand le modèle se trompe avec confiance, accélérant la correction.

**Gradient** : `dBCE/dŷ = (1/n) × (ŷ - y) / (ŷ × (1 - ŷ))`

**Quand l'utiliser** : Classification binaire — prédire oui/non, spam/pas spam, fraude/légitime.

### Entropie croisée catégorielle

`CCE = -(1/n) × Σᵢ Σ_c y_ic × log(ŷ_ic)`

En clair : pour les problèmes multi-classes. y est encodé en one-hot (par ex., [0, 0, 1, 0] pour la classe 2). Seule la probabilité prédite de la vraie classe contribue à la perte.

**Quand l'utiliser** : Classification multi-classes — reconnaissance de chiffres (0-9), classification d'espèces, etc.

## En Rust

```rust
use ndarray::array;
use ix_nn::loss;

// --- Régression : MSE ---
let predicted = array![[2.5], [0.0], [2.1], [7.8]];
let actual    = array![[3.0], [0.0], [2.0], [8.0]];

let mse = loss::mse_loss(&predicted, &actual);
println!("MSE: {:.4}", mse);  // Faible — les prédictions sont proches

let mse_grad = loss::mse_gradient(&predicted, &actual);
println!("Gradient MSE: {:?}", mse_grad);
// Le gradient pointe de la prédiction vers la valeur réelle

// --- Classification : entropie croisée binaire ---
let pred_probs = array![[0.9], [0.2], [0.8], [0.1]];  // Probabilités du modèle
let true_labels = array![[1.0], [0.0], [1.0], [0.0]];  // Classes réelles

let bce = loss::binary_cross_entropy(&pred_probs, &true_labels);
println!("BCE: {:.4}", bce);  // Faible — le modèle est confiant et correct

let bce_grad = loss::binary_cross_entropy_gradient(&pred_probs, &true_labels);
println!("Gradient BCE: {:?}", bce_grad);

// --- Que se passe-t-il avec une prédiction FAUSSE mais confiante ? ---
let bad_pred = array![[0.01]];   // Le modèle dit 1% de chance pour la classe 1
let true_val = array![[1.0]];    // Mais c'EST la classe 1 !

let bad_loss = loss::binary_cross_entropy(&bad_pred, &true_val);
println!("Confiant mais faux : {:.4}", bad_loss);  // Très élevé !
```

### Utiliser les fonctions de perte dans l'entraînement

```rust
use ndarray::array;
use ix_nn::layer::{Dense, Layer};
use ix_nn::loss;

let mut layer = Dense::new(3, 1);
let x = array![[1.0, 2.0, 3.0]];
let target = array![[5.0]];

// Passe avant
let pred = layer.forward(&x);
let loss_val = loss::mse_loss(&pred, &target);

// Passe arrière — le gradient remonte de la perte à travers les couches
let grad = loss::mse_gradient(&pred, &target);
layer.backward(&grad, 0.01);

println!("Perte avant mise à jour : {:.4}", loss_val);

// Après la mise à jour, la perte devrait être plus faible
let new_pred = layer.forward(&x);
let new_loss = loss::mse_loss(&new_pred, &target);
println!("Perte après mise à jour : {:.4}", new_loss);
```

## Quand l'utiliser

| Fonction de perte | Tâche | Activation de sortie | Comportement du gradient |
|--------------|------|-------------------|-------------------|
| **MSE** | Régression (prédire un nombre) | Linéaire (aucune) | Proportionnel à l'erreur |
| **Entropie croisée binaire** | Classification binaire | Sigmoïde | Gradient fort quand le modèle est confiant et faux |
| **Entropie croisée catégorielle** | Classification multi-classes | Softmax | Identique à la BCE mais pour plusieurs classes |

**Guide de décision rapide :**
- Vous prédisez une valeur continue ? → **MSE**
- Vous prédisez oui/non ? → **Entropie croisée binaire**
- Vous prédisez une classe parmi N ? → **Entropie croisée catégorielle**

## Paramètres clés

Les fonctions de perte n'ont pas d'hyperparamètres en soi — ce sont des formules fixes. Mais soyez attentif à :

| Préoccupation | Ce qu'il faut surveiller |
|---------|---------------|
| Stabilité numérique | L'entropie croisée avec log(0) = -infini. ix bride les prédictions pour éviter cela. |
| Échelle | La MSE est en unités au carré. La RMSE (√MSE) est dans les unités d'origine et souvent plus interprétable. |
| Sensibilité aux valeurs aberrantes | La MSE est très sensible aux valeurs aberrantes. Envisagez la perte de Huber (pas encore dans ix) pour une régression robuste. |

## Pièges

- **N'utilisez pas la MSE pour la classification.** La MSE traite la différence entre 0.4 et 0.6 de la même manière que celle entre 0.0 et 0.2. L'entropie croisée pénalise correctement les prédictions fausses avec confiance bien davantage, produisant de meilleurs gradients.
- **N'utilisez pas l'entropie croisée pour la régression.** L'entropie croisée attend des probabilités (0 à 1). Les sorties brutes de régression peuvent être n'importe quel nombre.
- **Attention à log(0).** Si votre modèle prédit exactement 0 ou 1, log(0) = -infini. ix bride les prédictions dans un petit intervalle epsilon pour éviter cela, mais si vous implémentez une perte personnalisée, soyez prudent.
- **Vérifiez l'activation de sortie.** L'entropie croisée binaire suppose que les prédictions sont des probabilités (sortie sigmoïde). Si votre couche de sortie n'a pas d'activation, les prédictions pourraient être négatives ou > 1, rendant la perte dénuée de sens.
- **La perte devient NaN ?** Cela signifie généralement : taux d'apprentissage trop élevé, log(0) quelque part, ou gradients explosifs. Réduisez d'abord le taux d'apprentissage.

## Pour aller plus loin

- **Avant** : [Du perceptron au MLP](perceptron-vers-mlp.md) — les réseaux que ces fonctions de perte entraînent
- **Avant** : [Rétropropagation](retropropagation.md) — comment les gradients de la perte se propagent en arrière
- **Lié** : [Métriques d'évaluation](../apprentissage-supervise/metriques-evaluation.md) — exactitude, précision, rappel (différent de la perte !)
- **Lié** : [Descente de gradient](../optimisation/descente-de-gradient.md) — l'optimiseur qui minimise la perte
