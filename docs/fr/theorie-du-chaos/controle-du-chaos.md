# Contrôle du chaos

## Le problème

Le coeur d'un patient fibrille — le muscle cardiaque se contracte selon un schéma
chaotique et inefficace au lieu du rythme régulier nécessaire pour pomper le sang. Plutôt
que de choquer le coeur avec une forte impulsion de défibrillation, des chercheurs ont
démontré que de minuscules impulsions électriques précisément synchronisées peuvent
stabiliser l'une des nombreuses orbites périodiques instables cachées dans l'attracteur
chaotique, rétablissant ainsi un rythme normal.

Le contrôle du chaos est l'art de dompter les systèmes chaotiques avec une intervention
minimale. Il s'applique à la stabilisation de lasers, au contrôle d'écoulements
turbulents, à la gestion des oscillations de réacteurs chimiques et au maintien de
systèmes de communication verrouillés sur une porteuse.

## L'intuition

Un attracteur chaotique est densément peuplé d'orbites périodiques instables — comme une
pelote de laine emmêlée contenant des boucles cachées de toutes les longueurs. Chaque
boucle est instable : le système la visite brièvement avant d'être projeté vers une
autre partie de l'attracteur.

**Méthode OGY (Ott-Grebogi-Yorke) :** Attendez que le système dérive naturellement
près de l'orbite instable désirée, puis appliquez un minuscule ajustement de paramètre
qui le pousse *sur* l'orbite. Parce que vous exploitez la dynamique propre du système,
la perturbation nécessaire est infinitésimale. C'est comme équilibrer une balle sur la
pointe d'une colline en effectuant des ajustements microscopiques de la pente.

**Méthode de Pyragas (rétroaction retardée) :** Ajoutez un signal de contrôle continu
proportionnel à la différence entre l'état actuel et l'état une période plus tôt :
`controle = K * (x(t - T) - x(t))`. Quand le système est sur une orbite de période T,
le signal de contrôle est nul (aucune énergie gaspillée). Quand il s'en écarte, la
rétroaction le corrige. C'est comme un thermostat qui compare la température actuelle à
ce qu'elle était il y a exactement un cycle.

**Synchronisation du chaos :** Deux systèmes chaotiques identiques, démarrés avec des
conditions initiales différentes, suivront des trajectoires complètement différentes.
Mais en couplant une variable du système « réponse » au système « pilote », on peut
forcer leur synchronisation. C'est la base de la communication sécurisée par chaos.

## Fonctionnement

### Contrôle OGY (applications discrètes)

Pour une application x_{n+1} = f(x_n, r) avec un point fixe instable x* au paramètre
r_0 :

```
delta_r = -(df/dx) / (df/dr) * (x_n - x*)
r_n = r_0 + clamp(delta_r, -max_perturbation, +max_perturbation)
```

**En clair :** Quand l'orbite passe près du point fixe cible, décalez le paramètre d'une
quantité proportionnelle à l'écart. Le rapport des dérivées partielles détermine la
direction optimale. La perturbation est limitée pour éviter les grands sauts.

### Rétroaction retardée de Pyragas (systèmes continus)

```
dx/dt = F(x) + K * (x(t - tau) - x(t))
```

**En clair :** Le terme de contrôle est nul quand x(t) = x(t - tau), c'est-à-dire
quand le système est sur une orbite de période tau. Tout écart par rapport à la
périodicité génère une force de rappel proportionnelle à K.

### Synchronisation pilote-réponse

```
dx_response/dt = F(x_response) + coupling * (x_driver - x_response)
```

**En clair :** Le système réponse suit sa propre dynamique chaotique, plus une correction
qui le tire vers le pilote chaque fois qu'ils divergent. Avec une force de couplage
suffisante, la réponse se verrouille sur la trajectoire du pilote.

## En Rust

```rust
use ix_chaos::control::{
    ogy_control, pyragas_control, drive_response_sync,
};

// --- OGY : stabiliser la suite logistique ---
let r = 3.8;  // régime chaotique
let target = 1.0 - 1.0 / r;           // point fixe instable
let df_dx = r * (1.0 - 2.0 * target); // dérivée partielle df/dx en (x*, r)
let df_dr = target * (1.0 - target);   // dérivée partielle df/dr en (x*, r)

let trajectory = ogy_control(
    |x, r| r * x * (1.0 - x),  // la suite logistique
    target,                      // point fixe désiré
    r,                           // paramètre nominal
    df_dx, df_dr,                // linéarisation au point cible
    0.1,                         // perturbation max du paramètre
    0.5,                         // état initial
    200,                         // nombre total de pas
    50,                          // activer le contrôle au pas 50
);

// Après activation du contrôle, x converge vers la cible
let (last_x, last_r) = trajectory.last().unwrap();
println!("Stabilisé à x={:.4} (cible={:.4}), r={:.4}", last_x, target, last_r);

// --- Pyragas : stabiliser un système continu ---
let oscillator = |x: &[f64]| -> Vec<f64> {
    vec![x[1], -x[0] + 0.3 * x[1] * (1.0 - x[0] * x[0])]  // van der Pol
};

let controlled = pyragas_control(
    &oscillator,
    &[1.0, 0.0],   // état initial
    0.01,           // dt
    5000,           // pas
    628,            // retard ~ une période (2*pi / 0.01)
    0.5,            // gain de rétroaction K
    1000,           // activer le contrôle après 1000 pas
);
println!("Pyragas : {} pas, état final = {:?}",
    controlled.len(), controlled.last().unwrap());

// --- Synchroniser deux systèmes de Lorenz ---
let lorenz = |x: &[f64]| -> Vec<f64> {
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]
};

let (driver, response, errors) = drive_response_sync(
    &lorenz,
    &[1.0, 1.0, 1.0],    // état initial du pilote
    &[5.0, 5.0, 5.0],    // état initial de la réponse (différent !)
    0.01,                  // dt
    5000,                  // pas
    5.0,                   // force de couplage
    &[0],                  // coupler uniquement la variable x
);
println!("Erreur de synchronisation initiale : {:.4}", errors[0]);
println!("Erreur de synchronisation finale :   {:.4}", errors.last().unwrap());
```

## Quand l'utiliser

| Méthode | Idéal pour | Prérequis |
|---------|-----------|-----------|
| **OGY** | Applications discrètes ; stabilisation de points fixes | Connaître l'application, sa dérivée et le point fixe cible |
| **Pyragas** | Systèmes continus ; stabilisation d'orbites périodiques | Connaître la période approximative ; pas besoin des équations explicites |
| **Pilote-réponse** | Synchroniser deux systèmes chaotiques identiques | Dynamiques identiques ; force de couplage suffisante |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Règle empirique |
|-----------|-------------------|-----------------|
| `max_perturbation` (OGY) | Limite l'ajustement du paramètre par pas | Plus petit = moins invasif mais convergence plus lente |
| `delay_steps` (Pyragas) | Doit correspondre à la période de l'orbite cible | Estimer la période à partir de la trajectoire non contrôlée |
| `gain` (Pyragas) | Force de la rétroaction | Trop faible = pas de contrôle ; trop fort = dépassement et instabilité |
| `coupling_strength` (sync) | Force d'attraction de la réponse vers le pilote | Doit dépasser un seuil dépendant du système pour la synchronisation |
| `control_start` | Quand le contrôle commence | Laisser la dynamique transitoire se stabiliser avant d'activer le contrôle |

## Pièges courants

1. **OGY ne fonctionne que près de la cible.** La méthode repose sur la linéarisation,
   qui n'est valide que dans un petit voisinage du point fixe. Le système doit
   naturellement s'en approcher avant que le contrôle puisse s'engager. Cela peut prendre
   de nombreuses itérations.

2. **Mauvaise estimation de période pour Pyragas.** Si le retard ne correspond pas à la
   période réelle de l'orbite, la rétroaction combat la dynamique naturelle au lieu de
   renforcer l'orbite.

3. **Seuil de couplage.** En dessous d'une force de couplage critique, la synchronisation
   pilote-réponse échoue complètement. Le seuil dépend des exposants de Lyapunov du
   système.

4. **Intégration d'Euler.** L'implémentation de Pyragas utilise l'intégration d'Euler
   par simplicité. Pour les systèmes raides, cela peut nécessiter un dt très petit.
   Envisagez d'envelopper la dynamique dans un intégrateur RK4 via
   `ix_chaos::attractors::rk4_step` pour une meilleure précision.

## Pour aller plus loin

- Combinez OGY avec `ix_chaos::lyapunov::mle_1d` pour vérifier que l'orbite contrôlée
  a un exposant de Lyapunov négatif (confirmant la stabilisation).
- Utilisez `ix_chaos::bifurcation::bifurcation_diagram` pour identifier toutes les
  orbites périodiques instables disponibles avant de choisir une cible pour le contrôle
  OGY.
- Explorez la communication basée sur le chaos : encodez l'information dans les
  paramètres du système pilote, puis décodez en mesurant l'erreur de synchronisation
  dans la réponse.
- Combinez le contrôle de Pyragas avec `ix_signal::kalman::KalmanFilter` pour estimer
  l'état du système à partir d'observations bruitées avant d'appliquer la rétroaction.
