# Exposants de Lyapunov

## Le problème

Vous gérez un fonds de trading quantitatif. Vos modèles fonctionnent bien en période de
marché calme mais explosent pendant les périodes turbulentes. Vous avez besoin d'un
indicateur numérique qui vous dit, en temps réel, si le régime de marché est stable,
périodique ou chaotique — avant que les pertes ne s'accumulent.

Les exposants de Lyapunov répondent exactement à cette question pour tout système
dynamique : « Si je perturbe l'état actuel d'une quantité infinitésimale, la perturbation
croît-elle ou décroît-elle au fil du temps ? » Ils constituent la référence pour détecter
le chaos en physique, en climatologie, en épidémiologie et en ingénierie.

## L'intuition

Placez deux feuilles côte à côte sur une rivière :

- **Dans un bassin calme** (exposant de Lyapunov négatif), elles dérivent l'une vers
  l'autre. Les petites différences se réduisent. Le système est stable.
- **Sur un courant régulier** (exposant nul), elles restent à la même distance. Le système
  est périodique ou quasi-périodique.
- **Dans des rapides** (exposant positif), elles s'éloignent exponentiellement vite. Le
  système est chaotique — de minuscules différences de position initiale conduisent à des
  résultats radicalement différents.

L'**exposant maximal de Lyapunov (MLE)** mesure le taux de cette divergence exponentielle.
Un MLE positif est la signature mathématique du chaos.

## Fonctionnement

### MLE pour une application 1D

Pour une application discrète x_{n+1} = f(x_n), le MLE vaut :

```
lambda = (1/N) * sum_{n=0}^{N-1} ln|f'(x_n)|
```

**En clair :** À chaque pas, calculez la dérivée (combien la fonction étire ou comprime
les points voisins). Faites la moyenne du logarithme de la valeur absolue de la dérivée
le long de toute l'orbite. Si cette moyenne est positive, les orbites voisines divergent
exponentiellement — le système est chaotique.

### Spectre de Lyapunov pour les systèmes continus

Pour une EDO n-dimensionnelle dx/dt = F(x), le spectre complet de n exposants se calcule
ainsi :

1. Intégrez l'état x(t) en avant par RK4.
2. Faites évoluer simultanément n vecteurs de perturbation à l'aide de la matrice
   jacobienne J(x).
3. Ré-orthogonalisez périodiquement les vecteurs de perturbation (Gram-Schmidt) pour
   les empêcher de se rabattre sur la direction la plus instable.
4. Le taux de croissance de chaque vecteur orthogonalisé donne un exposant de Lyapunov.

**En clair :** Suivez l'évolution de n petites flèches indépendantes emportées par le
flux. La flèche à la croissance la plus rapide donne le MLE ; les autres complètent le
spectre, révélant la géométrie complète du système.

### Classification

| Valeur du MLE | `DynamicsType` | Signification |
|---------------|---------------|---------------|
| MLE < -seuil | `FixedPoint` | Le système converge vers un équilibre |
| -seuil <= MLE <= +seuil | `Periodic` | Cycle limite ou orbite quasi-périodique |
| MLE > +seuil | `Chaotic` | Sensibilité aux conditions initiales |
| MLE > 10 | `Divergent` | Le système diverge (instable) |

## En Rust

```rust
use ix_chaos::lyapunov::{mle_1d, lyapunov_spectrum, classify_dynamics, DynamicsType};

// --- Application 1D : suite logistique x_{n+1} = r*x*(1-x) ---
let r = 4.0;  // régime pleinement chaotique
let f  = |x: f64| r * x * (1.0 - x);
let df = |x: f64| r * (1.0 - 2.0 * x);  // dérivée de f

let mle = mle_1d(f, df, 0.1, 10_000, 1000);
// mle ~ ln(2) ~ 0.693 pour r=4.0
println!("MLE = {:.4}", mle);

let dynamics = classify_dynamics(mle, 0.01);
assert_eq!(dynamics, DynamicsType::Chaotic);

// --- Comparaison de différentes valeurs de r ---
for &r in &[2.5, 3.2, 3.5, 3.9] {
    let f  = |x: f64| r * x * (1.0 - x);
    let df = |x: f64| r * (1.0 - 2.0 * x);
    let le = mle_1d(f, df, 0.1, 10_000, 1000);
    println!("r={:.1}: MLE={:.4} -> {:?}", r, le, classify_dynamics(le, 0.01));
}
// r=2.5: FixedPoint, r=3.2: FixedPoint (2-cycle stable), r=3.5: Periodic, r=3.9: Chaotic

// --- Système continu : spectre de Lyapunov pour Lorenz ---
let lorenz_dynamics = |x: &[f64], _t: f64| -> Vec<f64> {
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]
};

let lorenz_jacobian = |x: &[f64], _t: f64| -> Vec<f64> {
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![
        -sigma,  sigma,  0.0,        // ligne 0
        rho - x[2], -1.0, -x[0],    // ligne 1
        x[1],    x[0],  -beta,       // ligne 2
    ]
};

let spectrum = lyapunov_spectrum(
    &lorenz_dynamics,
    &lorenz_jacobian,
    &[1.0, 1.0, 1.0],  // état initial
    0.01,                // dt
    10_000,              // pas
    1_000,               // transitoire à ignorer
);
println!("Spectre de Lorenz : {:?}", spectrum);
// Attendu : [~+0.9, ~0.0, ~-14.6] (un positif, un nul, un négatif)
```

> Exemple complet exécutable : [examples/chaos/logistic_map.rs](../../examples/chaos/logistic_map.rs)

## Quand l'utiliser

| Technique | Idéal pour | Limites |
|-----------|-----------|---------|
| **MLE (application 1D)** | Détection rapide du chaos dans les applications discrètes | Nécessite la dérivée f'(x) sous forme analytique |
| **Spectre de Lyapunov** | Caractérisation complète de systèmes dynamiques continus | Nécessite le jacobien ; coûteux pour les systèmes de haute dimension |
| Diagramme de bifurcation | Visualiser comment la dynamique change avec un paramètre | Qualitatif ; ne fournit pas un nombre unique |
| Dimension par comptage de boîtes | Mesurer la structure fractale d'un attracteur | Capture la géométrie, pas la dynamique |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Règle empirique |
|-----------|-------------------|-----------------|
| `iterations` / `steps` | Durée de la moyenne | 10 000+ pour des estimations fiables ; davantage pour les systèmes bruités |
| `transient` | Pas à ignorer avant l'accumulation | 10-20 % du total ; garantit que l'orbite est sur l'attracteur |
| `dt` | Pas d'intégration pour les systèmes continus | Assez petit pour un RK4 stable (0.01 pour Lorenz) |
| `threshold` | Frontière entre périodique et chaotique dans `classify_dynamics` | 0.01 est une valeur par défaut courante |
| `x0` / état initial | Point de départ | Doit être proche de l'attracteur ; le transitoire aide dans tous les cas |

## Pièges courants

1. **Le transitoire compte.** Si l'orbite ne s'est pas encore installée sur l'attracteur,
   les échantillons transitoires contaminent l'estimation de l'exposant. Ignorez toujours
   au moins 1000 itérations initiales.

2. **Orbites superstables.** À certaines valeurs de paramètre (par ex. suite logistique
   r=2), la dérivée passe par zéro. Le MLE renvoie correctement moins l'infini, mais le
   logarithme de zéro nécessite un traitement spécial (l'implémentation gère ce cas).

3. **Temps fini vs asymptotique.** Le véritable exposant de Lyapunov est une limite quand
   le nombre d'itérations tend vers l'infini. Un nombre fini d'échantillons donne une
   approximation. Vérifiez toujours la convergence en augmentant `iterations` et en
   vérifiant que le résultat se stabilise.

4. **Précision du jacobien.** Pour le spectre, un jacobien incorrect produira des
   exposants erronés. Vérifiez deux fois chaque dérivée partielle.

5. **Systèmes de haute dimension.** Le calcul du spectre est en O(n^2) par pas
   (Gram-Schmidt sur n vecteurs de dimension n). Pour n > ~20, envisagez de calculer
   uniquement le MLE par la méthode de perturbation directe.

## Pour aller plus loin

- Tracez le MLE en fonction d'un paramètre pour construire un **diagramme de Lyapunov**
  — un compagnon quantitatif du diagramme de bifurcation.
- Utilisez `ix_chaos::attractors::lorenz` pour générer des trajectoires, puis analysez
  leur spectre de Lyapunov pour différentes valeurs de paramètres.
- Combinez avec `ix_chaos::fractal::correlation_dimension` pour relier le nombre
  d'exposants positifs à la dimension fractale de l'attracteur (conjecture de
  Kaplan-Yorke).
- Injectez des rendements financiers dans `mle_1d` avec une application appropriée
  pour détecter les changements de régime en temps réel.
