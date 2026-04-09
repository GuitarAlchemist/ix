# Intuition du calcul differentiel

> Les derivees vous donnent la pente. Les gradients vous donnent la direction du changement le plus rapide. C'est tout ce qu'il vous faut pour le ML.

## Le probleme

Vous entrainez un modele pour predire les prix immobiliers. Votre modele a des parametres (poids) qui determinent ses predictions. Certaines valeurs de parametres donnent des predictions catastrophiques, d'autres d'excellentes predictions. Vous devez trouver les meilleurs parametres.

Imaginez-vous debout sur un paysage vallonne dans un brouillard epais. Vous ne voyez pas la vallee la plus basse, mais vous *sentez* dans quelle direction le sol s'incline sous vos pieds. Le calcul differentiel vous donne cette "sensation de pente" -- il vous indique dans quelle direction marcher pour descendre. Les algorithmes de ML utilisent cela pour minimiser les erreurs de prediction.

## L'intuition

### Derivees : la pente d'une courbe

Une **derivee** vous dit a quelle vitesse quelque chose change en un point precis.

Si vous conduisez et que votre position au fil du temps est une courbe, la derivee est votre compteur de vitesse -- elle vous donne votre vitesse *a cet instant*, pas votre vitesse moyenne.

Pour une fonction f(x) :
- Si la derivee est **positive**, f monte (croissante)
- Si la derivee est **negative**, f descend (decroissante)
- Si la derivee est **nulle**, vous etes sur un point plat (peut etre un minimum, un maximum ou un point-selle)

Imaginez une bille roulant dans un bol. Au fond du bol, la surface est plate (derivee = 0) -- c'est le minimum.

### Gradients : les derivees en plusieurs dimensions

La plupart des modeles de ML ont de nombreux parametres, pas un seul. Au lieu d'une derivee unique, vous obtenez un **gradient** -- un vecteur de derivees, une par parametre.

Le gradient pointe dans la direction de la plus forte *augmentation*. Pour minimiser (descendre), on marche dans la direction *opposee* au gradient. C'est litteralement ce que fait la descente de gradient.

```
Parametres : [w1, w2]
Gradient :   [df/dw1, df/dw2]

Le gradient dit : "Si vous augmentez w1, l'erreur augmente de df/dw1.
                   Si vous augmentez w2, l'erreur augmente de df/dw2."

Pour reduire l'erreur : avancer dans la direction opposee au gradient.
```

### La regle de la chaine : pourquoi le deep learning fonctionne

Les modeles complexes sont construits a partir de pieces simples enchainees : entree -> couche 1 -> couche 2 -> sortie. La regle de la chaine dit :

"La derivee d'une chaine de fonctions est le produit des derivees de chaque fonction."

En clair : pour savoir comment l'entree affecte la sortie, on multiplie l'effet de chaque etape sur la suivante. C'est le fondement mathematique de la retropropagation dans les reseaux de neurones.

## Fonctionnement detaille

### Derivee numerique

On n'a pas toujours une formule pour la derivee. L'approche la plus simple est de l'approximer :

`f'(x) ~ (f(x + epsilon) - f(x - epsilon)) / (2*epsilon)`

En clair : on deplace legerement x dans les deux directions, on observe de combien f change, et on divise par l'amplitude du deplacement. Plus epsilon est petit, plus l'approximation est precise (mais trop petit cause des problemes de precision en virgule flottante).

C'est la methode des **differences centrees**. C'est ce qu'ix utilise quand vous ne fournissez pas de gradient analytique.

### Gradient numerique (multi-dimensionnel)

Pour une fonction de plusieurs variables, on calcule la derivee partielle pour chaque variable separement :

```
df/dx1 ~ (f(x1+epsilon, x2, ...) - f(x1-epsilon, x2, ...)) / (2*epsilon)
df/dx2 ~ (f(x1, x2+epsilon, ...) - f(x1, x2-epsilon, ...)) / (2*epsilon)
...
```

Le vecteur gradient est `[df/dx1, df/dx2, ...]`.

### Matrice hessienne (derivees secondes)

La hessienne est une matrice de derivees secondes. Elle renseigne sur la *courbure* -- non seulement dans quelle direction descendre, mais aussi a quel point le terrain est pentu ou plat dans chaque direction.

```
H[i][j] = d^2f / (dxi dxj)
```

En clair : la hessienne vous dit si le minimum est une vallee etroite (forte courbure, on peut faire de grands pas) ou une plaine peu profonde (faible courbure, il faut etre prudent). Certains optimiseurs avances utilisent cette information.

## En Rust

ix fournit la differentiation numerique dans `ix-math` :

```rust
use ndarray::array;
use ix_math::calculus;

// Derivee scalaire
// f(x) = x^2 -> f'(x) = 2x
let f = |x: f64| x * x;
let derivative_at_3 = calculus::derivative(&f, 3.0, 1e-7);
// derivative_at_3 ~ 6.0

// Gradient d'une fonction multi-variable
// f(x, y) = x^2 + y^2 -> gradient = [2x, 2y]
let g = |x: &ndarray::Array1<f64>| x[0] * x[0] + x[1] * x[1];
let point = array![3.0, 4.0];
let grad = calculus::numerical_gradient(&g, &point, 1e-7);
// grad ~ [6.0, 8.0]

// Matrice hessienne (derivees secondes)
let hessian = calculus::numerical_hessian(&g, &point, 1e-5);
// Pour f = x^2 + y^2, hessienne ~ [[2, 0], [0, 2]] (courbure constante)
```

### Gradients et optimisation

Voici comment les gradients se connectent a l'optimisation -- la boucle centrale de l'entrainement de tout modele :

```rust
use ndarray::array;
use ix_optimize::{SGD, Optimizer, ClosureObjective, ObjectiveFunction};
use ix_optimize::gradient::minimize;
use ix_optimize::ConvergenceCriteria;

// Minimiser f(x, y) = (x-3)^2 + (y-7)^2
// Le minimum est en (3, 7)
let objective = ClosureObjective {
    f: |x: &ndarray::Array1<f64>| {
        (x[0] - 3.0).powi(2) + (x[1] - 7.0).powi(2)
    },
    dimensions: 2,
};

let mut optimizer = SGD::new(0.1);  // taux d'apprentissage = 0.1
let initial = array![0.0, 0.0];    // depart loin de la reponse
let criteria = ConvergenceCriteria {
    max_iterations: 1000,
    tolerance: 1e-8,
};

let result = minimize(&objective, &mut optimizer, initial, &criteria);
// result.best_params ~ [3.0, 7.0]
```

A chaque iteration : calculer le gradient -> faire un pas dans la direction opposee -> recommencer.

## Quand l'utiliser

| Situation | Ce dont vous avez besoin |
|-----------|--------------------------|
| Entrainer tout modele supervise | La descente de gradient utilise les gradients pour minimiser la perte |
| Deboguer la descente de gradient | Gradient numerique pour verifier votre gradient analytique |
| Comprendre le paysage d'optimisation | La hessienne revele la courbure, aide a choisir le taux d'apprentissage |
| Implementer la retropropagation | La regle de la chaine compose les derivees a travers les couches |

## Parametres cles

| Parametre | Ce qu'il controle | Trop petit | Trop grand |
|-----------|-------------------|------------|------------|
| epsilon dans le gradient numerique | Precision de l'approximation | Le bruit en virgule flottante domine | Mauvaise approximation de la vraie derivee |
| Taux d'apprentissage (descente de gradient) | Taille du pas en descente | Converge trop lentement | Depasse le minimum, diverge |

Le bon compromis pour epsilon numerique est autour de `1e-7` pour les derivees et `1e-5` pour les hessiennes (les derivees secondes amplifient le bruit).

## Pieges courants

- **Les gradients numeriques sont lents.** Calculer le gradient de n parametres necessite 2n evaluations de la fonction (un decalage vers l'avant et un vers l'arriere par parametre). Pour les reseaux de neurones avec des millions de parametres, les gradients analytiques via la retropropagation sont indispensables.
- **Minima locaux.** La descente de gradient trouve *un* minimum, pas necessairement *le* minimum. Pour les fonctions convexes (comme la perte de la regression lineaire), tout minimum est le minimum global. Pour les reseaux de neurones, il existe de nombreux minima locaux -- mais en pratique, ils sont generalement satisfaisants.
- **Gradients evanescents/explosifs.** Dans les reseaux profonds, la regle de la chaine multiplie beaucoup de petits nombres (evanescents) ou beaucoup de grands nombres (explosifs). C'est pourquoi le choix de la fonction d'activation et l'initialisation des poids sont cruciaux.
- **Ne confondez pas derivee = 0 avec "reponse trouvee".** Une derivee nulle peut indiquer un maximum, un minimum ou un point-selle. En pratique, la descente de gradient evite naturellement les maxima et les points-selle grace au bruit.

## Pour aller plus loin

- **Suivant** : [Distance et similarite](distance-et-similarite.md) -- mesurer la proximite entre les points de donnees
- **Utilise ceci** : [Descente de gradient](../optimisation/descente-de-gradient.md) -- l'algorithme d'optimisation propulse par les gradients
- **Utilise ceci** : [Retropropagation](../reseaux-de-neurones/retropropagation.md) -- la regle de la chaine appliquee aux reseaux de neurones
