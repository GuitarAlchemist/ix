# Attaques par évasion : FGSM, PGD, C&W et JSMA

## Le problème

Vous testez le système de perception d'une voiture autonome avant son déploiement. Le classificateur basé sur la caméra identifie les panneaux stop avec une précision de 99,7 % sur le jeu de test. Mais un adversaire pourrait placer quelques autocollants sur un panneau stop, invisibles pour les humains mais amenant le classificateur à lire « limitation 45 km/h ». Avant de déployer, vous devez sonder systématiquement la quantité de perturbation nécessaire pour tromper le modèle — et quelles entrées sont les plus vulnérables.

Les attaques par évasion génèrent des exemples adversariaux : des entrées qui paraissent normales pour les humains mais trompent les modèles d'apprentissage automatique. Elles sont essentielles pour les tests de robustesse en imagerie médicale, détection de malwares, modération de contenu et tout système de classification critique pour la sécurité.

## L'intuition

La frontière de décision d'un réseau de neurones est une surface en haute dimension. La plupart du temps, vos entrées se situent confortablement loin de cette surface, et le modèle est confiant. Mais la surface comporte des crêtes aiguës et des péninsules étroites qui s'étendent très près des entrées normales.

**FGSM** est un seul pas audacieux vers la crête la plus proche. Imaginez braquer une lampe torche sur la frontière de décision : le gradient vous indique la direction, et vous faites un grand pas dans cette direction. C'est rapide mais grossier.

**PGD** fait de nombreux petits pas vers la frontière, en reprojetant dans le budget de perturbation autorisé après chaque pas. C'est comme marcher vers une falaise en s'assurant de ne jamais s'éloigner de plus d'epsilon mètres du point de départ.

**C&W** est un optimiseur précis qui trouve la *plus petite* perturbation nécessaire pour franchir la frontière, équilibrant taille de perturbation et succès de l'attaque. Pensez-y comme un instrument de précision comparé au marteau de PGD.

**JSMA** travaille caractéristique par caractéristique, ne perturbant que le pixel (ou la caractéristique) le plus impactant à chaque étape. Il produit des perturbations creuses — seuls quelques pixels changent.

## Comment ça fonctionne

### FGSM (Fast Gradient Sign Method)

```
x_adv = x + epsilon * sign(gradient_de_la_perte)
```

**En clair :** Calculer le gradient de la perte par rapport à l'entrée. Prendre le signe de chaque composante (+1 ou -1). Faire un pas de taille epsilon dans cette direction. Un coup, une perturbation.

### PGD (Projected Gradient Descent)

```
pour chaque étape :
    x_adv = x_adv + alpha * sign(gradient)
    x_adv = projeter(x_adv, x, epsilon)   # clipper dans la boule L-infini
```

**En clair :** FGSM itéré avec des pas plus petits (alpha < epsilon). Après chaque pas, clipper la perturbation pour qu'elle reste dans un rayon epsilon de l'entrée originale.

### C&W (Carlini & Wagner)

```
minimiser :  ||delta||_2 + c * perte(x + delta, cible)
```

**En clair :** Trouver la plus petite perturbation L2 qui minimise aussi la perte de classification vers une classe cible. La constante de compromis c contrôle l'effort mis sur la mauvaise classification vs. la réduction de la perturbation.

### JSMA (Jacobian-based Saliency Map)

```
pour chaque étape :
    calculer saillance = impact de chaque caractéristique sur la classe cible
    perturber la caractéristique non modifiée la plus saillante de theta
```

**En clair :** Choisir gloutonnement la caractéristique unique qui augmente le plus la probabilité de la classe cible. La perturber. Répéter jusqu'à un budget de max_perturbations caractéristiques.

## En Rust

```rust
use ix_adversarial::evasion::{fgsm, pgd, cw_attack, jsma};
use ndarray::array;

let input = array![0.5, 0.3, 0.8, 0.1];

// --- FGSM : attaque en un pas ---
let gradient = array![0.1, -0.2, 0.05, 0.3];
let adversarial = fgsm(&input, &gradient, 0.1);
// Chaque dimension bouge de +/- 0.1 dans la direction du signe du gradient
println!("FGSM : {:?}", adversarial);

// --- PGD : attaque itérative (plus puissante) ---
let adversarial_pgd = pgd(
    &input,
    |_x| array![0.1, -0.2, 0.05, 0.3],  // fonction gradient
    0.1,   // epsilon (budget de perturbation)
    0.01,  // alpha (taille de pas par itération)
    40,    // nombre d'itérations
);
// La perturbation est garantie de rester dans la boule L-inf epsilon
let diff = &adversarial_pgd - &input;
for &d in diff.iter() {
    assert!(d.abs() <= 0.1 + 1e-10);
}

// --- C&W : attaque à perturbation minimale ---
let adversarial_cw = cw_attack(
    &input,
    1,                           // classe cible
    |_x, _t| 1.0,               // fonction de perte
    |x| x.clone(),              // fonction gradient
    1.0,                         // constante de compromis c
    50,                          // étapes d'optimisation
    0.01,                        // taux d'apprentissage
);

// --- JSMA : perturbation creuse ---
let adversarial_jsma = jsma(
    &input,
    1,                                      // classe cible
    |_x| array![0.1, 0.9, 0.3, 0.2],     // fonction de saillance
    2,                                      // max caractéristiques à perturber
    0.5,                                    // magnitude de perturbation theta
);
// Seulement 2 caractéristiques sont modifiées
```

> Exemple complet exécutable : [examples/adversarial/robustness_test.rs](../../examples/adversarial/robustness_test.rs)

## Quand l'utiliser

| Attaque | Vitesse | Qualité de la perturbation | Idéal pour |
|--------|-------|---------------------|----------|
| **FGSM** | Très rapide (1 gradient) | Grossière ; souvent détectable | Vérifications rapides ; augmentation pour l'entraînement adversarial |
| **PGD** | Modérée (N gradients) | Attaque L-inf forte et bornée | Benchmark de robustesse standard ; tests de certification |
| **C&W** | Lente (boucle d'optimisation) | Perturbation L2 minimale | Mesurer la vraie vulnérabilité du modèle ; audits de sécurité |
| **JSMA** | Modérée | Creuse (peu de caractéristiques changées) | Analyse d'importance des caractéristiques ; interprétabilité |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Règle pratique |
|-----------|-----------------|---------------|
| `epsilon` | Magnitude maximale de perturbation | 0,3 pour MNIST (plage 0-1) ; 8/255 pour CIFAR ; spécifique au domaine |
| `alpha` (PGD) | Taille de pas par itération | alpha = epsilon / steps est un défaut sûr |
| `steps` (PGD) | Nombre d'itérations | 20-40 pour la plupart des applications |
| `c` (C&W) | Compromis : taille de perturbation vs. succès de l'attaque | La recherche binaire sur c est standard ; commencer à 1,0 |
| `lr` (C&W) | Taux d'apprentissage de l'optimisation | 0,01 est typique ; réduire si la perte oscille |
| `max_perturbations` (JSMA) | Budget : combien de caractéristiques à changer | Plus bas = plus creux ; 10-20 % des dimensions d'entrée |
| `theta` (JSMA) | Amplitude du changement sur chaque caractéristique sélectionnée | Dépend de l'échelle des caractéristiques |

## Pièges courants

1. **Qualité du gradient.** Toutes ces attaques nécessitent des gradients précis de la perte par rapport à l'entrée. Si votre modèle utilise des opérations non différentiables (argmax, échantillonnage discret), vous aurez besoin d'approximations du gradient.

2. **Epsilon trop grand.** Une perturbation visible à l'oeil humain n'est pas un exemple adversarial significatif. Gardez epsilon en dessous du seuil perceptuel pour votre domaine.

3. **Minima locaux de PGD.** PGD peut rester bloqué. Utilisez des redémarrages aléatoires (exécuter PGD plusieurs fois depuis différentes perturbations aléatoires de l'entrée) pour des attaques plus fiables.

4. **C&W est lente.** Pour de grandes entrées (images), la boucle d'optimisation est coûteuse. C'est une attaque de référence pour l'évaluation, pas pour la génération de données d'entraînement adversarial.

5. **JSMA suppose l'indépendance des caractéristiques.** La stratégie gloutonne une-caractéristique-à-la-fois peut manquer des attaques nécessitant des changements coordonnés sur plusieurs caractéristiques.

## Pour aller plus loin

- Utilisez `ix_adversarial::evasion::universal_perturbation` pour trouver une perturbation unique qui trompe le modèle sur de nombreuses entrées simultanément.
- Après avoir généré des exemples adversariaux, testez les défenses de `ix_adversarial::defense` — voir [defenses-adversariales.md](defenses-adversariales.md).
- Combinez PGD avec l'entraînement adversarial : générez des exemples PGD à chaque étape d'entraînement et incluez-les dans le batch pour renforcer le modèle.
- Soumettez les exemples adversariaux à `ix_adversarial::poisoning::detect_label_flips` pour vérifier si une attaque sur les données d'entraînement pourrait passer inaperçue.
