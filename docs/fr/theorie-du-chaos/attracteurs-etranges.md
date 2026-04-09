# Attracteurs étranges

## Le problème

Vous simulez des schémas météorologiques pour un institut de recherche climatique. Même
avec des équations du mouvement parfaites, les prévisions à long terme divergent
fortement de la réalité. Edward Lorenz a découvert ce phénomène en 1963 : des systèmes
déterministes peuvent produire des trajectoires qui ne se répètent jamais, ne divergent
jamais vers l'infini, et sont extrêmement sensibles aux conditions initiales. Les formes
géométriques que ces trajectoires dessinent — les attracteurs étranges — sont les
empreintes visuelles du chaos.

Comprendre les attracteurs est essentiel en modélisation de la turbulence, en analyse du
rythme cardiaque, en conception de circuits électroniques, et dans tout domaine où
l'imprévisibilité déterministe se manifeste.

## L'intuition

Un **point fixe** est une bille posée au fond d'un bol — elle se stabilise et reste.
Un **cycle limite** est une bille roulant autour du bord d'un bol en boucle répétitive.
Un **attracteur étrange** n'est ni l'un ni l'autre : la bille ne se stabilise jamais, ne
se répète jamais, mais ne s'échappe jamais non plus. Elle trace un motif infiniment
complexe dans une région bornée, comme un papillon tissant sans fin à travers le même
volume d'air sans jamais retracer son chemin.

L'attracteur de Lorenz (le célèbre « papillon ») est l'archétype. Ses deux lobes
représentent deux équilibres instables que la trajectoire visite dans un ordre
imprévisible — comme un flipper rebondissant entre deux bumpers à l'infini.

## Fonctionnement

### Système de Lorenz

```
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z
```

**En clair :** Trois variables (différence de température, vitesse du fluide, transport
de chaleur) sont couplées de sorte que chacune entraîne les autres. Avec les paramètres
classiques (sigma=10, rho=28, beta=8/3), le système est chaotique : les trajectoires ne
se répètent jamais mais restent confinées sur l'attracteur en forme de papillon.

### Intégration

Tous les attracteurs en temps continu sont intégrés par la méthode de Runge-Kutta
d'ordre 4 (RK4) :

```
k1 = f(state)
k2 = f(state + 0.5*dt*k1)
k3 = f(state + 0.5*dt*k2)
k4 = f(state + dt*k3)
new_state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
```

**En clair :** Évaluez la dérivée en quatre points stratégiquement choisis, puis prenez
une moyenne pondérée. Cela donne une précision en O(dt^4) par pas — bien meilleure que
la simple intégration d'Euler.

### Attracteurs implémentés

| Attracteur | Type | Dimension | Comportement clé |
|------------|------|-----------|-------------------|
| **Lorenz** | Continu 3D | ~2.06 fractale | Papillon à deux lobes ; sensible à sigma, rho, beta |
| **Rössler** | Continu 3D | ~2.01 fractale | Spirale simple à une bande ; modélise les oscillations chimiques |
| **Chen** | Continu 3D | Similaire à Lorenz | Variante avec une structure de couplage différente |
| **Hénon** | Discret 2D | ~1.26 fractale | Application 2D classique ; attracteur en forme de banane |
| **Suite logistique** | Discret 1D | Intervalle [0,1] | Route vers le chaos par doublement de période |

## En Rust

```rust
use ix_chaos::attractors::{
    State3D, LorenzParams, RosslerParams, ChenParams, HenonParams,
    lorenz, rossler, chen, henon, logistic_map,
    integrate, rk4_step,
};

// --- Attracteur de Lorenz ---
let params = LorenzParams::default(); // sigma=10, rho=28, beta=8/3
let initial = State3D::new(1.0, 1.0, 1.0);
let trajectory = lorenz(initial, &params, 0.01, 10_000);

// La trajectoire reste bornée (attracteur étrange)
for s in &trajectory {
    assert!(s.x.abs() < 100.0 && s.y.abs() < 100.0 && s.z.abs() < 100.0);
}
println!("Lorenz : {} points, final = ({:.2}, {:.2}, {:.2})",
    trajectory.len(), trajectory.last().unwrap().x,
    trajectory.last().unwrap().y, trajectory.last().unwrap().z);

// --- Attracteur de Rössler ---
let rossler_params = RosslerParams::default(); // a=0.2, b=0.2, c=5.7
let ross_traj = rossler(State3D::new(1.0, 1.0, 1.0), &rossler_params, 0.01, 10_000);

// --- Attracteur de Chen ---
let chen_params = ChenParams::default(); // a=35, b=3, c=28
let chen_traj = chen(State3D::new(1.0, 1.0, 1.0), &chen_params, 0.001, 50_000);

// --- Application de Hénon (discrète) ---
let henon_params = HenonParams::default(); // a=1.4, b=0.3
let henon_traj = henon(0.1, 0.1, &henon_params, 10_000);
for &(x, y) in &henon_traj[100..] {
    assert!(x.abs() < 3.0 && y.abs() < 3.0);  // borné sur l'attracteur
}

// --- Suite logistique ---
let orbit = logistic_map(0.5, 3.9, 1000);  // x0=0.5, r=3.9
let last = *orbit.last().unwrap();
assert!(last >= 0.0 && last <= 1.0);  // toujours dans [0, 1]

// --- EDO personnalisée avec integrate() et rk4_step() ---
let custom_deriv = |s: State3D| -> State3D {
    State3D::new(-s.y, s.x, -0.1 * s.z)  // harmonique simple + décroissance
};
let custom_traj = integrate(
    State3D::new(1.0, 0.0, 1.0),
    0.01,
    5000,
    &custom_deriv,
);

// Un seul pas RK4 pour un contrôle manuel
let next = rk4_step(State3D::new(1.0, 0.0, 1.0), 0.01, &custom_deriv);
```

## Quand l'utiliser

| Attracteur | Idéal pour | Complexité |
|------------|-----------|------------|
| **Lorenz** | Exemple canonique de chaos ; analogies météo/convection | 3 EDO, 3 paramètres |
| **Rössler** | Chaos spiral plus simple ; modèles d'oscillations chimiques | 3 EDO, 3 paramètres |
| **Chen** | Variante de Lorenz pour études comparatives | 3 EDO, 3 paramètres |
| **Hénon** | Chaos discret 2D ; rapide à calculer | Application 2D, 2 paramètres |
| **Suite logistique** | Modèle de chaos le plus simple ; usage pédagogique ; analyse 1D | Application 1D, 1 paramètre |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Règle empirique |
|-----------|-------------------|-----------------|
| `dt` | Pas d'intégration | Lorenz : 0.01 ; Chen : 0.001 (plus raide). Trop grand = la trajectoire diverge |
| `steps` | Longueur de la trajectoire | 10 000+ pour une forme d'attracteur bien définie |
| `sigma, rho, beta` | Lorenz : intensité du couplage, force motrice, dissipation | Par défaut (10, 28, 8/3) : régime chaotique classique |
| `a, b, c` (Rössler) | Serrage de la spirale et repliement | Par défaut (0.2, 0.2, 5.7) : entonnoir standard |
| `a, b` (Hénon) | Étirement et repliement | Par défaut (1.4, 0.3) : attracteur classique |
| `r` (Logistique) | Paramètre de bifurcation | r < 3 : point fixe ; 3 < r < 3.57 : doublement de période ; r > 3.57 : chaos |

## Pièges courants

1. **dt trop grand.** RK4 est stable pour des dt modérés, mais les systèmes chaotiques
   amplifient les erreurs. Si la trajectoire diverge vers l'infini, réduisez dt d'un
   facteur 10.

2. **Orbites transitoires.** Les premières centaines de points peuvent être loin de
   l'attracteur. Ignorez un transitoire initial (par ex. 1000 pas) avant l'analyse ou
   la visualisation.

3. **Sensibilité aux conditions initiales.** Deux trajectoires démarrant à 1e-10 d'écart
   divergeront exponentiellement sur un attracteur chaotique. C'est une propriété, pas un
   défaut — mais cela signifie que comparer des trajectoires nécessite un alignement
   soigneux.

4. **Divergence de Hénon.** L'application de Hénon avec des paramètres non standard
   (notamment a > 1.4) peut diverger vers l'infini. Vérifiez toujours que la trajectoire
   reste bornée.

5. **Domaine de la suite logistique.** La suite logistique n'a de sens que pour x dans
   [0, 1] et r dans [0, 4]. En dehors de cette plage, les orbites s'échappent vers
   moins l'infini.

## Pour aller plus loin

- Calculez l'exposant de Lyapunov de chaque attracteur avec
  `ix_chaos::lyapunov::lyapunov_spectrum` pour quantifier son degré de chaos.
- Mesurez la dimension fractale de l'attracteur avec
  `ix_chaos::fractal::box_counting_dimension_2d` ou `correlation_dimension`.
- Utilisez `ix_chaos::bifurcation::bifurcation_diagram` pour visualiser comment la
  suite logistique passe du point fixe au doublement de période puis au chaos.
- Injectez les trajectoires d'attracteurs dans `ix_chaos::control::ogy_control` pour
  stabiliser les orbites périodiques instables enfouies dans l'attracteur chaotique.
