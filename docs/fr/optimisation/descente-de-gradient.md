# Descente de gradient : SGD, Momentum et Adam

> Descendez la colline pas à pas jusqu'au fond de la vallée. La seule question est *comment* vous marchez.

**Prérequis :** [Intuition du calcul différentiel](../foundations/calculus-intuition.md), [Vecteurs et matrices](../foundations/vectors-and-matrices.md)

---

## Le problème

Vous construisez un prédicteur de prix immobilier. Votre modèle prend en entrée la superficie, le nombre de chambres et la distance au centre-ville, puis produit une estimation de prix. En interne, il possède un poids pour chaque caractéristique (plus un biais). Au départ, ces poids sont aléatoires — le modèle prédit n'importe quoi. Il vous faut un moyen systématique de les ajuster pour que les prédictions se rapprochent des prix réels.

Vous disposez de 10 000 ventes labellisées. Pour un jeu de poids donné, vous pouvez calculer un nombre unique — l'erreur quadratique moyenne — qui indique à quel point le modèle se trompe globalement. Votre objectif : trouver les poids qui rendent ce nombre aussi petit que possible. C'est un problème d'optimisation, et la descente de gradient est le cheval de bataille qui le résout.

---

## L'intuition

### SGD : un pas prudent à la fois

Imaginez que vous êtes perdu dans le brouillard sur un terrain vallonné et que vous cherchez la vallée la plus basse. Vous ne voyez pas plus loin que vos pieds. À chaque pas, vous sentez la pente du sol sous vous et faites un pas dans la direction de descente la plus raide. C'est la descente de gradient.

Le **gradient** est un vecteur qui pointe vers le haut. Vous vous déplacez dans la direction opposée. Le **taux d'apprentissage** (learning rate) contrôle la taille de vos pas. Trop grand et vous dépassez la vallée, rebondissant d'un côté à l'autre. Trop petit et vous avancez à pas de fourmi, prenant une éternité.

La descente de gradient **stochastique** (SGD) signifie que vous estimez la pente à partir d'un sous-ensemble aléatoire (un « batch ») de vos données plutôt que des 10 000 maisons. C'est plus bruité mais beaucoup plus rapide, et le bruit aide en réalité à s'échapper des creux locaux peu profonds qui ne sont pas le vrai minimum.

### Momentum : une balle qui roule

Le SGD simple est nerveux — chaque pas peut zigzaguer parce que le gradient change de direction. Le momentum corrige cela en donnant à l'optimiseur de l'« inertie », comme une lourde balle qui dévale une pente. La balle accumule de la vitesse dans les directions où elle avance régulièrement et atténue les oscillations dans les directions où le gradient ne cesse de changer de signe.

Si le gradient pointe toujours dans la même direction, le momentum vous accélère vers le minimum. Si le gradient zigzague (fréquent dans les vallées allongées), le momentum lisse la trajectoire.

### Adam : le momentum avec un contrôle de vitesse par paramètre

Adam combine le momentum avec une idée supplémentaire : il suit non seulement la direction moyenne du gradient (comme le momentum) mais aussi l'*amplitude* des gradients pour chaque paramètre individuellement. Les paramètres qui reçoivent régulièrement des gradients importants voient leur taux d'apprentissage réduit ; ceux qui reçoivent des gradients faibles et rares voient le leur augmenté.

Pensez-y comme une équipe de randonneurs où chacun ajuste la longueur de ses pas indépendamment en fonction du terrain qu'il a personnellement traversé. Un randonneur sur terrain plat fait de grands pas ; un autre sur une pente rocheuse et raide fait des pas plus petits et plus prudents.

---

## Comment ça fonctionne

### Règle de mise à jour SGD

```
params_new = params - lr * gradient
```

**En clair :** Prenez les paramètres actuels. Calculez le gradient (dans quelle direction l'erreur augmente). Faites un petit pas dans la direction opposée. Le taux d'apprentissage `lr` contrôle la taille du pas.

### Règle de mise à jour avec momentum

```
v(t) = mu * v(t-1) - lr * gradient
params_new = params + v(t)
```

**En clair :** La vitesse `v` est une moyenne glissante des gradients passés. À chaque pas, vous mélangez l'ancienne vitesse (multipliée par le facteur de momentum `mu`, typiquement 0.9) avec le nouveau gradient. Vous vous déplacez dans la direction de la vitesse accumulée plutôt que du gradient brut. C'est pourquoi une balle qui dévale une pente accélère — elle se souvient de la direction dans laquelle elle allait.

- `mu = 0` : Pas de mémoire, identique au SGD.
- `mu = 0.9` : Momentum fort, trajectoire lisse.
- `mu = 0.99` : Balle très lourde, lente à changer de direction.

### Règles de mise à jour Adam

```
m(t) = beta1 * m(t-1) + (1 - beta1) * gradient         # premier moment (moyenne)
v(t) = beta2 * v(t-1) + (1 - beta2) * gradient^2        # second moment (variance)
m_hat = m(t) / (1 - beta1^t)                             # correction du biais
v_hat = v(t) / (1 - beta2^t)                             # correction du biais
params_new = params - lr * m_hat / (sqrt(v_hat) + epsilon)
```

**En clair :**

- `m(t)` est la moyenne mobile exponentielle du gradient — elle suit la direction comme le fait le momentum.
- `v(t)` est la moyenne mobile exponentielle du gradient au carré — elle suit l'amplitude des gradients.
- La correction du biais (`m_hat`, `v_hat`) résout un problème de démarrage : puisque `m` et `v` sont initialisés à zéro, ils sont biaisés vers zéro au début. Diviser par `(1 - beta^t)` compense ce biais.
- La mise à jour effective divise le terme de momentum par la racine carrée de la variance. Si un paramètre a reçu des gradients importants, `sqrt(v_hat)` est grand, et le pas effectif rétrécit. Si les gradients étaient faibles, le pas grandit. Ce taux d'apprentissage adaptatif par paramètre est ce qui rend Adam si robuste.

---

## En Rust

### Définir la fonction objectif

Toute optimisation dans ix commence par une `ObjectiveFunction`. Ce trait possède trois méthodes :

- `evaluate(&self, x) -> f64` — calculer l'erreur pour un jeu de paramètres donné
- `gradient(&self, x) -> Array1<f64>` — calculer le gradient (par défaut, utilise la différentiation numérique si vous ne la redéfinissez pas)
- `dim(&self) -> usize` — nombre de paramètres

Pour des expériences rapides, utilisez `ClosureObjective` pour encapsuler une closure :

```rust
use ix_optimize::traits::{ClosureObjective, ObjectiveFunction, OptimizeResult};
use ix_optimize::convergence::ConvergenceCriteria;
use ix_optimize::gradient::{SGD, Momentum, Adam, minimize};
use ndarray::{array, Array1};

// Mean squared error for a simple linear model: price = w0 * sqft + w1 * beds + w2
// Given training data, the loss surface is a bowl -- gradient descent will find the bottom.
let objective = ClosureObjective {
    f: |w: &Array1<f64>| {
        // Simulated loss: (w0 - 0.15)^2 + (w1 - 50.0)^2 + (w2 - 100_000.0)^2
        // True optimum is [0.15, 50.0, 100_000.0]
        (w[0] - 0.15).powi(2)
            + (w[1] - 50.0).powi(2)
            + (w[2] - 100_000.0).powi(2)
    },
    dimensions: 3,
};
```

### Minimiser avec SGD

```rust
let mut sgd = SGD::new(0.01); // learning rate = 0.01
let criteria = ConvergenceCriteria {
    max_iterations: 5000,
    tolerance: 1e-8, // stop when gradient norm falls below this
};

let result = minimize(&objective, &mut sgd, array![0.0, 0.0, 0.0], &criteria);

println!("SGD found: {:?}", result.best_params);
println!("Loss: {:.6}", result.best_value);
println!("Converged: {} in {} iterations", result.converged, result.iterations);
```

### Ajouter le momentum

```rust
let mut momentum = Momentum::new(0.01, 0.9); // lr=0.01, momentum=0.9
let criteria = ConvergenceCriteria {
    max_iterations: 5000,
    tolerance: 1e-8,
};

let result = minimize(&objective, &mut momentum, array![0.0, 0.0, 0.0], &criteria);
println!("Momentum found: {:?}", result.best_params);
// Expect faster convergence than plain SGD on elongated loss surfaces
```

### Utiliser Adam

```rust
let mut adam = Adam::new(0.001)       // lower lr is typical for Adam
    .with_betas(0.9, 0.999);          // defaults -- usually no need to change

let criteria = ConvergenceCriteria {
    max_iterations: 10000,
    tolerance: 1e-10,
};

let result = minimize(&objective, &mut adam, array![0.0, 0.0, 0.0], &criteria);
println!("Adam found: {:?}", result.best_params);
println!("Loss: {:.10}", result.best_value);
```

### Comparer les trois méthodes

```rust
let initial = array![0.0, 0.0, 0.0];
let criteria = ConvergenceCriteria { max_iterations: 5000, tolerance: 1e-8 };

let optimizers: Vec<(&str, Box<dyn FnMut() -> OptimizeResult>)> = vec![
    ("SGD",      Box::new(|| minimize(&objective, &mut SGD::new(0.01), initial.clone(), &criteria))),
    ("Momentum", Box::new(|| minimize(&objective, &mut Momentum::new(0.01, 0.9), initial.clone(), &criteria))),
    ("Adam",     Box::new(|| minimize(&objective, &mut Adam::new(0.001), initial.clone(), &criteria))),
];

for (name, mut run) in optimizers {
    let r = run();
    println!("{:10} | iters: {:5} | loss: {:.6e} | converged: {}",
             name, r.iterations, r.best_value, r.converged);
}
```

### Comprendre la valeur de retour

`minimize` renvoie un `OptimizeResult` :

| Champ         | Type          | Signification                                      |
|---------------|---------------|-----------------------------------------------------|
| `best_params` | `Array1<f64>` | Le vecteur de paramètres avec la perte la plus faible |
| `best_value`  | `f64`         | La perte à `best_params`                             |
| `iterations`  | `usize`       | Le nombre de pas effectués par l'optimiseur           |
| `converged`   | `bool`        | `true` si la norme du gradient est passée sous `tolerance` |

### Le trait Optimizer

Tous les optimiseurs basés sur le gradient implémentent le même trait :

```rust
pub trait Optimizer {
    fn step(&mut self, params: &Array1<f64>, gradient: &Array1<f64>) -> Array1<f64>;
    fn name(&self) -> &str;
}
```

Cela signifie que vous pouvez écrire du code générique qui fonctionne avec n'importe quel optimiseur, remplacer SGD par Adam en une ligne, ou construire votre propre optimiseur et le brancher directement.

---

## Quand l'utiliser

| Situation | Optimiseur recommandé |
|-----------|-----------------------|
| Premier essai sur un nouveau problème | **Adam** — fonctionne bien d'emblée avec un minimum de réglage |
| Deep learning à grande échelle (millions de paramètres) | **SGD + Momentum** — généralise souvent mieux qu'Adam à la convergence |
| Vous voulez la ligne de base la plus simple | **SGD** — le plus facile à comprendre et à déboguer |
| La perte est bruitée (petits batchs, apprentissage par renforcement) | **Adam** — le taux d'apprentissage adaptatif gère bien le bruit |
| Surface de perte lisse et convexe (régression linéaire) | **SGD** ou **Momentum** — convergent de manière fiable, Adam peut dépasser la cible |
| Vous soupçonnez l'optimiseur d'être trop agressif | Réduisez le taux d'apprentissage, ou passez du SGD à Adam qui s'adapte automatiquement |

---

## Paramètres clés

### Taux d'apprentissage (`lr`)

L'hyperparamètre le plus important. Points de départ typiques :

- SGD : `0.01` à `0.1`
- Momentum : `0.01` à `0.1`
- Adam : `0.001` à `0.0001`

Si la perte oscille violemment, divisez le taux d'apprentissage par 10. Si la perte diminue désespérément lentement, augmentez-le.

### Facteur de momentum (`momentum` dans `Momentum::new`)

- `0.9` est le point de départ standard. Presque personne ne le modifie.
- Des valeurs plus élevées (0.95, 0.99) donnent plus d'inertie, utile pour des gradients très bruités.
- Des valeurs plus basses (0.5) réagissent plus vite aux changements de gradient mais perdent l'effet de lissage.

### Betas d'Adam (`beta1`, `beta2`)

- `beta1 = 0.9` contrôle le premier moment (mémoire de la direction du gradient). Plus élevé signifie une mémoire plus longue.
- `beta2 = 0.999` contrôle le second moment (mémoire de l'amplitude du gradient). Plus élevé signifie des taux par paramètre plus stables.
- Les valeurs par défaut conviennent presque toujours. Ne modifiez `beta2` que si vous observez une instabilité de l'entraînement en fin d'optimisation.

### Critères de convergence

- `max_iterations` : Filet de sécurité. Fixez généreusement (5 000 à 100 000 selon la complexité du problème).
- `tolerance` : Seuil de norme du gradient. Quand le gradient est aussi petit, on se trouve effectivement à un point plat. `1e-6` à `1e-8` pour la plupart des problèmes.

---

## Pièges courants

**Taux d'apprentissage trop élevé.** La perte saute dans tous les sens ou diverge. La solution est toujours la même : réduire `lr` d'un facteur 10. C'est le problème le plus fréquent chez les débutants.

**Taux d'apprentissage trop faible.** La perte diminue mais prend une éternité. Vous risquez d'atteindre `max_iterations` avant de converger. Si `converged` est `false` et que `best_value` continue à s'améliorer, augmentez `max_iterations` ou augmentez `lr`.

**Points selle et minima locaux.** En haute dimension, le SGD peut rester bloqué aux points selle (plats dans certaines directions). Le momentum et Adam aident tous deux à s'en échapper car leur vitesse accumulée les porte à travers les régions plates.

**Le déficit de généralisation d'Adam.** La recherche a montré qu'Adam converge parfois vers des minima plus abrupts que le SGD avec momentum. En deep learning, cela peut se traduire par une précision de test légèrement inférieure. La solution consiste souvent à commencer avec Adam (pour un progrès rapide au départ) puis à passer au SGD avec momentum pour l'affinage.

**Le gradient numérique est lent.** L'implémentation par défaut de `gradient()` utilise la différentiation numérique (différences finies). Cela évalue la fonction objectif `2 * dim` fois par pas. Pour les problèmes de haute dimension, implémentez le gradient analytiquement en redéfinissant la méthode `gradient` sur `ObjectiveFunction`.

**Paramètres avec des échelles très différentes.** Si un paramètre est typiquement ~0.01 et un autre ~100 000, le SGD et le momentum auront du mal car le même taux d'apprentissage est appliqué partout. Adam gère cela naturellement (taux par paramètre). Sinon, normalisez d'abord vos caractéristiques.

---

## Pour aller plus loin

- **Voir en action :** [`examples/optimization/pso_rosenbrock.rs`](../../examples/optimization/pso_rosenbrock.rs) montre une optimisation sur la fonction de Rosenbrock, un problème test classique avec une vallée incurvée qui met à l'épreuve les méthodes à gradient.
- **Alternatives sans gradient :** Quand vous n'avez pas de gradients (problèmes discrets, fonctions boîte noire), essayez le [Recuit simulé](recuit-simule.md) ou l'[Optimisation par essaim particulaire](essaim-particulaire.md).
- **D'où viennent les gradients :** [Intuition du calcul différentiel](../foundations/calculus-intuition.md) explique les dérivées et la différentiation numérique.
- **Entraînement de réseaux de neurones :** [Rétropropagation](../neural-networks/backpropagation.md) montre comment les gradients se propagent à travers les couches d'un réseau — la règle de la chaîne en action.
- **Planification du taux d'apprentissage :** Un taux d'apprentissage fixe est souvent sous-optimal. Des stratégies avancées réduisent le taux au fil de l'entraînement (cosine annealing, warmup). Elles peuvent être implémentées en modifiant `lr` entre les appels à `minimize`.
