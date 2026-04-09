# Confidentialité différentielle

## Le problème

Vous construisez une plateforme d'analytique pour un réseau hospitalier. Chaque hôpital contribue des données patients pour entraîner un modèle diagnostique partagé, mais la réglementation interdit de révéler des informations individuelles sur les patients. Même les probabilités de sortie du modèle peuvent fuiter des données privées : un attaquant qui interroge le modèle avec de légères variations des caractéristiques d'un patient peut inférer si ce patient faisait partie des données d'entraînement (attaque par inférence d'appartenance). Vous avez besoin de garanties mathématiques qu'aucune donnée individuelle n'affecte significativement les sorties du modèle.

La confidentialité différentielle fournit exactement cette garantie : le modèle se comporte presque identiquement qu'un patient donné soit inclus ou exclu des données d'entraînement.

## L'intuition

Imaginez un sondage où les gens répondent « oui » ou « non » à une question sensible. Avant d'enregistrer, chaque personne lance une pièce en privé. Si c'est face, elle répond honnêtement ; si c'est pile, elle relance et répond « oui » pour face, « non » pour pile. Les résultats agrégés reflètent toujours la vraie distribution, mais la réponse enregistrée de chaque individu est plausiblement aléatoire. C'est l'essence de la confidentialité différentielle : ajouter du bruit calibré pour que les contributions individuelles soient noyées dans l'agrégat.

En apprentissage automatique, le bruit est ajouté aux gradients pendant l'entraînement (DP-SGD) ou aux sorties du modèle au moment de l'inférence.

## Comment ça fonctionne

### Mécanisme gaussien

```
gradient_bruité = gradient + N(0, sigma^2)
sigma = sensibilité * sqrt(2 * ln(1.25 / delta)) / epsilon
```

**En clair :** La sensibilité est le changement maximal qu'un seul échantillon d'entraînement peut induire sur le gradient. On ajoute du bruit gaussien calibré pour que la probabilité de toute sortie particulière change d'au plus un facteur e^epsilon quand un échantillon est ajouté ou retiré. Delta est la probabilité que la garantie échoue.

### Budget de confidentialité (epsilon, delta)

- **epsilon :** Plus bas = plus confidentiel. epsilon=1 est fort ; epsilon=10 est faible.
- **delta :** Probabilité d'une violation de confidentialité. Typiquement 1/N² où N est la taille du jeu de données.

### Protections supplémentaires

- **Mise à l'échelle de température :** Diviser les logits par une température > 1 avant le softmax. Cela produit des distributions de probabilité plus plates qui fuient moins d'information sur les échantillons d'entraînement individuels.
- **Purification de prédiction :** Mettre à zéro toutes les classes de sortie sauf les top-K, empêchant un attaquant d'extraire de l'information des classes à faible probabilité.
- **Score d'inférence d'appartenance :** Mesurer avec quelle confiance le modèle prédit le vrai label d'un point. Une confiance élevée suggère que le point était dans l'entraînement.

## En Rust

```rust
use ix_adversarial::privacy::{
    differential_privacy_noise,
    model_confidence_masking,
    prediction_purification,
    membership_inference_score,
};
use ndarray::array;

// Ajouter du bruit DP à un gradient
let gradient = array![1.0, 2.0, 3.0];
let noisy = differential_privacy_noise(
    &gradient,
    1.0,    // epsilon (budget de confidentialité)
    1e-5,   // delta
    1.0,    // sensibilité (norme max du gradient par échantillon)
    42,     // graine pour la reproductibilité
);

// Mise à l'échelle de température pour réduire les fuites d'information
let logits = array![2.0, 1.0, 0.0];
let sharp = model_confidence_masking(&logits, 1.0);    // softmax standard
let smooth = model_confidence_masking(&logits, 10.0);  // distribution plus plate

// Purification de prédiction : n'exposer que la classe top-1
let output = array![0.1, 0.7, 0.15, 0.05];
let purified = prediction_purification(&output, 1);
// Seule la valeur la plus haute (0.7) survit ; le reste devient 0.0

// Évaluation du risque d'inférence d'appartenance
let risk = membership_inference_score(
    |_x| array![0.1, 0.9, 0.0],  // fonction modèle
    &array![1.0, 2.0],            // point de test
    1,                              // indice du vrai label
);
// risk = 0.9 (haute confiance -> probablement un membre de l'entraînement)
```

## Quand l'utiliser

| Technique | Protège contre | Compromis |
|-----------|-----------------|-----------|
| **Bruit DP sur les gradients** | Inférence d'appartenance, inversion de modèle | Réduit la précision proportionnellement au bruit |
| **Mise à l'échelle de température** | Extraction d'information par les sorties | Réduit la netteté des prédictions |
| **Purification de prédiction** | Extraction des probabilités de classes | Perd le classement multi-classes |
| **Score d'inférence d'appartenance** | Audit du risque de confidentialité de votre modèle | Diagnostic uniquement ; n'ajoute pas de protection |

## Pièges courants

1. **Compromis confidentialité-précision.** Une confidentialité plus forte (epsilon plus petit) nécessite plus de bruit, ce qui dégrade la précision du modèle. Il n'y a pas de solution miracle.
2. **Estimation de la sensibilité.** Le mécanisme gaussien nécessite de connaître la norme maximale du gradient par échantillon. Sous-estimer la sensibilité brise la garantie ; la surestimer ajoute du bruit inutile. Le clipping du gradient à une norme fixe est la pratique standard.
3. **Composition.** Chaque fois que vous interrogez le modèle ou entraînez une époque supplémentaire, vous dépensez du budget de confidentialité. Suivez l'epsilon cumulé sur toutes les opérations.
4. **Gestion des graines.** La graine déterministe est utile pour la reproductibilité mais ne doit pas être réutilisée entre différentes requêtes en production (elle produirait un bruit identique).

## Pour aller plus loin

- Implémentez DP-SGD en clippant les gradients par échantillon à une norme fixe, puis en ajoutant `differential_privacy_noise` avant l'étape de l'optimiseur. Utilisez `ix_optimize` pour l'optimiseur SGD/Adam sous-jacent.
- Combinez la mise à l'échelle de température avec la purification de prédiction pour une défense en couches.
- Utilisez `membership_inference_score` comme audit pré-déploiement : si de nombreux échantillons d'entraînement ont des scores au-dessus d'un seuil, le modèle fuite trop d'information.
- Voir [defenses-adversariales.md](defenses-adversariales.md) pour des protections complémentaires au moment de l'inférence.
