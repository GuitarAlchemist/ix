# Défenses adversariales

## Le problème

Vous déployez un classificateur d'imagerie médicale qui détecte les tumeurs dans les radiographies. Après avoir exécuté des attaques par évasion (FGSM, PGD), vous découvrez que le modèle peut être trompé par des perturbations imperceptibles. Vous avez besoin de défenses qui soit détectent les entrées adversariales avant qu'elles n'atteignent le modèle, soit rendent le modèle intrinsèquement robuste aux petites perturbations.

## L'intuition

Les perturbations adversariales exploitent deux propriétés : (1) la frontière de décision du modèle est proche des entrées normales, et (2) les perturbations sont précisément calibrées sur les poids exacts du modèle. Les défenses s'attaquent à l'une ou aux deux propriétés :

- **L'entraînement adversarial** repousse la frontière de décision loin des entrées normales en entraînant sur des exemples perturbés.
- **La randomisation d'entrée** brise le calibrage précis en ajoutant du bruit aléatoire avant la classification. Si la sortie du modèle change dramatiquement sous un petit bruit aléatoire, l'entrée est probablement adversariale.
- **La compression de caractéristiques** (feature squeezing) quantifie l'entrée, détruisant la perturbation fine sur laquelle l'attaque repose.
- **La régularisation du gradient** pénalise les grands gradients d'entrée pendant l'entraînement, rendant la sortie du modèle plus lisse et plus difficile à attaquer.

## Comment ça fonctionne

### Augmentation par entraînement adversarial

Générer des exemples FGSM à partir de chaque batch d'entraînement et les mélanger aux données d'entraînement. Le modèle apprend à classifier correctement les entrées propres et perturbées.

### Détection par randomisation d'entrée

Ajouter du bruit gaussien à l'entrée N fois. Si la variance de la sortie du modèle dépasse un seuil, signaler l'entrée comme adversariale. Les entrées légitimes produisent des sorties stables ; les entrées adversariales se trouvent sur un fil de couteau de la frontière de décision et sont sensibles au bruit.

### Compression de caractéristiques

Quantifier les valeurs d'entrée sur `bit_depth` bits. Une perturbation de 0,003 sur une entrée [0,1] disparaît quand on arrondit à une précision de 4 bits (résolution de 1/15 ~ 0,067).

### Clipping et lissage

Borner les entrées à la plage valide, puis appliquer un filtre à moyenne glissante. Le bruit adversarial haute fréquence est moyenné.

## En Rust

```rust
use ix_adversarial::defense::{
    adversarial_training_augment,
    input_gradient_regularization,
    detect_adversarial,
    feature_squeezing,
    clip_and_smooth,
};
use ndarray::array;

// Augmenter les données d'entraînement avec des exemples adversariaux
let inputs = vec![array![0.5, 0.5], array![0.3, 0.7]];
let gradients = vec![array![1.0, -1.0], array![-0.5, 0.2]];
let augmented = adversarial_training_augment(&inputs, &gradients, 0.1);

// Pénalité de régularisation du gradient (ajouter à la perte pendant l'entraînement)
let grad = array![3.0, 4.0];
let penalty = input_gradient_regularization(&grad); // 25.0

// Détecter une entrée adversariale par randomisation
let suspicious_input = array![0.5, 0.5];
let is_adversarial = detect_adversarial(
    &suspicious_input,
    |x| x * 100.0,  // modèle qui amplifie les entrées
    0.1,             // écart-type du bruit
    100,             // nombre d'échantillons aléatoires
    0.01,            // seuil de variance
    42,              // graine RNG
);

// Compression de caractéristiques : réduire à une précision de 4 bits
let squeezed = feature_squeezing(&array![0.503, 0.297, 0.801], 4);

// Clipping et lissage : borner puis moyenne glissante
let smoothed = clip_and_smooth(&array![0.0, 1.0, 0.0], 0.0, 1.0, 3);
```

## Quand l'utiliser

| Défense | Force | Coût |
|---------|----------|------|
| **Entraînement adversarial** | Fort contre les attaques connues | Double le temps d'entraînement ; peut réduire la précision sur données propres |
| **Randomisation d'entrée** | Détecte de nombreux types d'attaques | Ajoute de la latence à l'inférence ; faux positifs possibles |
| **Compression de caractéristiques** | Simple ; pas de ré-entraînement | Réduit la résolution d'entrée ; peut nuire aux tâches fines |
| **Régularisation du gradient** | Rend le modèle intrinsèquement plus lisse | Ajoute un hyperparamètre ; augmente le coût d'entraînement |
| **Clipping et lissage** | Supprime le bruit haute fréquence | Estompe les détails légitimes |

## Pièges courants

1. **Attaques adaptatives.** Un attaquant qui sait que vous utilisez la compression de caractéristiques peut concevoir des perturbations qui survivent à la quantification. Aucune défense unique n'est inviolable.
2. **Réglage du seuil de détection.** Trop sensible = faux positifs sur les entrées bruitées mais légitimes. Trop laxiste = les entrées adversariales passent.
3. **Compromis avec la précision propre.** L'entraînement adversarial et la régularisation du gradient réduisent souvent la précision sur les entrées non perturbées de 1 à 3 %.

## Pour aller plus loin

- Combinez plusieurs défenses : comprimer d'abord, puis randomiser, puis classifier.
- Utilisez `ix_adversarial::evasion::pgd` pour générer des exemples adversariaux puissants pour l'augmentation d'entraînement au lieu de FGSM.
- Voir [empoisonnement-donnees.md](empoisonnement-donnees.md) pour les défenses contre les attaques au moment de l'entraînement.
- Voir [confidentialite-differentielle.md](confidentialite-differentielle.md) pour protéger les sorties du modèle contre les fuites d'information.
