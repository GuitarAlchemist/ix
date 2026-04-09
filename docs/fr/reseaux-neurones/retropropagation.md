# Rétropropagation

> Comment les réseaux de neurones apprennent — la règle de la chaîne appliquée à rebours à travers les couches pour calculer les gradients efficacement.

## Le problème

Vous avez construit un MLP avec des milliers de poids. Après une passe avant, vous savez que la prédiction est fausse — la perte est élevée. Mais quels poids ont causé l'erreur ? Et dans quelle direction faut-il ajuster chacun d'eux ?

Calculer le gradient de la perte par rapport à chaque poids individuellement (par perturbation numérique) nécessiterait des millions de passes avant — d'une lenteur insoutenable. La rétropropagation le fait en une seule passe arrière.

## L'intuition

Imaginez une chaîne de montage fabriquant un produit. L'inspecteur final détecte un défaut. Pour le corriger, il faut remonter la chaîne à rebours : « Est-ce le dernier poste ? Celui d'avant ? Celui encore avant ? »

Chaque poste (couche) reçoit une part de responsabilité proportionnelle à sa contribution à l'erreur. L'idée clé : **il n'est pas nécessaire d'examiner chaque poste indépendamment** — on peut faire remonter le « signal de responsabilité » à travers la chaîne, et chaque poste calcule sa propre part.

Ce « signal de responsabilité » est le **gradient**. Il circule en sens inverse depuis la perte, à travers chaque couche, en s'accumulant selon la règle de la chaîne :

```
Perte → Couche de sortie → Couche cachée 2 → Couche cachée 1 → Entrée
  ←le gradient remonte←
```

### La règle de la chaîne : le cœur mathématique

Si f(g(x)) est une composition de fonctions, alors :

`df/dx = df/dg × dg/dx`

En clair : le taux de variation à travers une chaîne d'opérations est le produit des taux de variation de chaque étape.

Pour trois couches : `dPerte/dW₁ = dPerte/dSortie × dSortie/dCachée × dCachée/dW₁`

Chaque couche n'a besoin de connaître que deux choses :
1. Quel gradient arrive d'en haut (dPerte/d(sortie de cette couche))
2. Son propre gradient local (comment sa sortie varie en fonction de son entrée et de ses poids)

## Comment ça fonctionne

### Passe avant (sauvegarde des valeurs intermédiaires)

Pendant la passe avant, chaque couche sauvegarde son entrée — c'est nécessaire pour la passe arrière.

Pour une couche dense : `sortie = W × entrée + b`

On sauvegarde `entrée` dans un cache.

### Passe arrière

Étant donné `grad_output` (gradient de la perte par rapport à la sortie de cette couche), on calcule :

1. **Gradient par rapport aux poids** : `dL/dW = grad_output^T × entrée_en_cache`

   En clair : de combien faut-il modifier chaque poids ? On multiplie le signal d'erreur par l'entrée que ce poids a vue.

2. **Gradient par rapport au biais** : `dL/db = sum(grad_output)` selon la dimension du batch

   En clair : le gradient du biais est simplement la somme des signaux d'erreur.

3. **Gradient par rapport à l'entrée** (à transmettre en arrière) : `dL/d(entrée) = grad_output × W`

   En clair : quelle part d'erreur transmettre à la couche précédente. Chaque entrée a contribué proportionnellement aux poids par lesquels elle a été multipliée.

4. **Mise à jour des poids** : `W -= taux_apprentissage × dL/dW`

### Exemple complet : réseau à 2 couches

```
Passe avant :
  h = W₁ × x + b₁       (couche cachée, sauvegarder x)
  h_relu = relu(h)        (activation, sauvegarder h)
  y = W₂ × h_relu + b₂   (couche de sortie, sauvegarder h_relu)
  perte = MSE(y, cible)

Passe arrière :
  dL/dy = 2(y - cible)/n           (gradient MSE)
  dL/dW₂ = dL/dy^T × h_relu        (poids de sortie)
  dL/db₂ = sum(dL/dy)               (biais de sortie)
  dL/dh_relu = dL/dy × W₂           (propagation arrière)
  dL/dh = dL/dh_relu ⊙ relu'(h)    (à travers l'activation : ⊙ est élément par élément)
  dL/dW₁ = dL/dh^T × x             (poids cachés)
  dL/db₁ = sum(dL/dh)               (biais cachés)
```

Le symbole ⊙ désigne la multiplication élément par élément. `relu'(h)` vaut 1 là où h > 0, 0 ailleurs.

## En Rust

La couche `Dense` d'ix gère les passes avant et arrière :

```rust
use ndarray::array;
use ix_nn::layer::{Dense, Layer};
use ix_nn::loss::{mse_loss, mse_gradient, binary_cross_entropy, binary_cross_entropy_gradient};

// Réseau : 2 entrées → 4 cachées (ReLU) → 1 sortie (sigmoïde)
let mut hidden = Dense::new(2, 4);
let mut output = Dense::new(4, 1);

let x = array![[0.5, -0.3]];       // Entrée (batch de 1)
let target = array![[1.0]];         // Cible

let learning_rate = 0.01;

// Boucle d'entraînement
for epoch in 0..1000 {
    // Passe avant
    let h = hidden.forward(&x);
    let h_relu = h.mapv(|v| if v > 0.0 { v } else { 0.0 });
    let pred = output.forward(&h_relu);

    // Perte
    let loss = mse_loss(&pred, &target);
    if epoch % 100 == 0 {
        println!("Époque {}: perte = {:.6}", epoch, loss);
    }

    // Passe arrière
    let grad = mse_gradient(&pred, &target);
    let grad_hidden = output.backward(&grad, learning_rate);

    // Appliquer la dérivée de ReLU avant de propager le gradient
    let relu_grad = h.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
    let grad_through_relu = &grad_hidden * &relu_grad;
    hidden.backward(&grad_through_relu, learning_rate);
}
```

### Ce que Dense::backward fait en interne

La méthode `backward` de `Dense` gère les étapes 1 à 4 décrites plus haut :

```rust
// Vue simplifiée de ce que Dense::backward fait :
fn backward(&mut self, grad_output: &Array2<f64>, lr: f64) -> Array2<f64> {
    let input = self.input_cache.as_ref().unwrap();  // Sauvegardé pendant la passe avant

    // Gradient par rapport aux poids : dL/dW = entrée^T × grad_output
    let grad_weights = input.t().dot(grad_output);

    // Gradient par rapport au biais : somme selon la dimension du batch
    let grad_bias = grad_output.sum_axis(ndarray::Axis(0));

    // Gradient à propager en arrière : dL/d(entrée) = grad_output × W^T
    let grad_input = grad_output.dot(&self.weights.t());

    // Mise à jour
    self.weights = &self.weights - &(&grad_weights * lr);
    self.bias = &self.bias - &(&grad_bias * lr);

    grad_input
}
```

## Quand l'utiliser

La rétropropagation est utilisée chaque fois que vous entraînez un réseau de neurones. Il n'existe pas d'alternative aussi efficace :

| Méthode | Coût par gradient | Praticable ? |
|--------|------------------|------------|
| **Rétropropagation** | 1 passe avant + 1 passe arrière | Oui — standard |
| **Gradient numérique** | 2 passes avant *par poids* | Tests uniquement (bien trop lent pour l'entraînement) |
| **Différences finies** | Identique au numérique | Idem |

Utilisez les gradients numériques pour *vérifier* que votre implémentation de rétropropagation est correcte (vérification de gradient), pas pour l'entraînement réel.

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Conseils |
|-----------|-----------------|----------|
| Taux d'apprentissage | Amplitude de chaque mise à jour des poids | Trop élevé → la perte oscille. Trop bas → l'entraînement est lent. Commencez à 0.01 |
| Taille du batch | Nombre d'échantillons par calcul de gradient | Plus grand = gradients plus stables mais plus lent par étape. 32-256 est courant. |

## Pièges

- **Gradients évanescents** : Dans les réseaux profonds avec sigmoïde/tanh, les gradients sont multipliés à travers de nombreuses couches. Chaque multiplication par un nombre < 1 réduit le signal. Résultat : les premières couches n'apprennent presque rien. Solution : utiliser l'activation ReLU, l'initialisation de He, ou les connexions résiduelles.
- **Gradients explosifs** : L'inverse — les gradients croissent exponentiellement à travers les couches. Résultat : les poids deviennent NaN. Solution : écrêtage de gradient, initialisation soignée, taux d'apprentissage plus faible.
- **ReLU mort** : Si un neurone ReLU produit toujours 0, son gradient est toujours 0 et il ne récupère jamais. Solution : utiliser Leaky ReLU ou un taux d'apprentissage plus faible.
- **Oubli de la mise en cache des entrées** : La passe arrière a besoin des valeurs intermédiaires de la passe avant. `Dense` met en cache automatiquement — n'exécutez pas `backward` sans un `forward` préalable.
- **Vérification de gradient** : Lors de l'implémentation de couches personnalisées, vérifiez toujours avec les gradients numériques. Si les gradients analytiques et numériques diffèrent de plus de 1e-5, il y a un bug.

## Pour aller plus loin

- **Suite** : [Fonctions de perte](fonctions-de-perte.md) — ce que le gradient cherche à minimiser
- **Utilise** : [Intuition du calcul](../fondements/intuition-du-calcul.md) — la règle de la chaîne expliquée
- **Utilise** : [Descente de gradient](../optimisation/descente-de-gradient.md) — l'optimiseur qui applique les gradients
- **Avant** : [Du perceptron au MLP](perceptron-vers-mlp.md) — l'architecture que la rétropropagation entraîne
