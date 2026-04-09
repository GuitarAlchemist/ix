# Filtre de Kalman

## Le problème

Vous construisez un système de suivi de drone par GPS. Le récepteur GPS fournit une
position chaque seconde, mais chaque mesure comporte 3 à 5 mètres d'erreur aléatoire.
Parallèlement, vous connaissez la vitesse du drone grâce à son contrôleur de vol. Vous
devez combiner ces deux sources d'information imparfaites pour produire une estimation
de position lisse et précise — idéalement capable de prédire la position du drone
*entre* les mises à jour GPS.

Les filtres de Kalman apparaissent partout où des capteurs bruités rencontrent un modèle
physique connu : localisation de véhicules autonomes, navigation spatiale, lissage de
séries temporelles financières et contrôle de bras robotiques.

## L'intuition

Imaginez que vous essayez de déterminer où un ami se promène dans un parc brumeux. Vous
disposez de deux sources d'information :

1. **Un modèle physique :** « Mon ami marche vers le nord à 1 mètre par seconde. » Cela
   vous permet de *prédire* sa position, mais la prédiction dérive avec le temps car
   vous ne connaissez pas la vitesse exacte.
2. **Un capteur bruité :** Toutes les quelques secondes, vous l'apercevez à travers le
   brouillard. L'observation est approximativement correcte mais imprécise.

Le filtre de Kalman est une manière optimale de fusionner ces deux sources. Lorsque la
mesure du capteur arrive, le filtre se demande : « Quelle confiance dois-je accorder à
cette mesure par rapport à ma prédiction ? » Il répond par le **gain de Kalman** — un
nombre entre 0 (ignorer complètement la mesure) et 1 (faire totalement confiance à la
mesure). Au fil du temps, le filtre converge vers une estimation meilleure que chaque
source prise isolément.

## Fonctionnement

### Modèle d'état

```
x_k = F * x_{k-1} + B * u_k + w_k      (transition d'état)
z_k = H * x_k + v_k                      (observation)
```

Où w ~ N(0, Q) est le bruit de processus et v ~ N(0, R) est le bruit de mesure.

### Étape de prédiction

```
x_predicted = F * x + B * u
P_predicted = F * P * F^T + Q
```

**En clair :** Utilisez le modèle physique pour extrapoler l'état vers l'avant.
L'incertitude (P) augmente car le modèle n'est pas parfait (Q ajoute de l'incertitude).

### Étape de mise à jour

```
innovation     = z - H * x_predicted
S              = H * P * H^T + R
K              = P * H^T * S^{-1}         (gain de Kalman)
x_updated      = x_predicted + K * innovation
P_updated      = (I - K * H) * P_predicted
```

**En clair :** Comparez la mesure réelle à ce que le modèle avait prédit (innovation).
Calculez le degré de confiance à accorder à la mesure par rapport à la prédiction (K).
Fusionnez les deux. L'incertitude diminue car nous avons incorporé une nouvelle
information.

## En Rust

```rust
use ix_signal::kalman::{KalmanFilter, constant_velocity_1d};
use ndarray::{array, Array1};

// --- Démarrage rapide : suivre un objet à vitesse constante ---
let mut kf = constant_velocity_1d(
    0.1,   // bruit de processus (mouvement imprévisible ?)
    1.0,   // bruit de mesure (GPS bruité ?)
    1.0,   // dt (secondes entre mises à jour)
);
// État = [position, vitesse], observation = [position]

// Simuler des lectures GPS bruitées d'un objet à position = 10 + 2*t
let measurements: Vec<Array1<f64>> = (0..20)
    .map(|t| {
        let true_pos = 10.0 + 2.0 * t as f64;
        array![true_pos + 0.5 * (t as f64 % 3.0 - 1.0)]  // ajout de bruit
    })
    .collect();

let states = kf.filter(&measurements);
let last = states.last().unwrap();
println!("Position estimée : {:.2}, vitesse : {:.2}", last[0], last[1]);
// la vitesse converge vers ~2.0

// --- Configuration manuelle pour des modèles d'état personnalisés ---
let mut kf = KalmanFilter::new(2, 1);  // state_dim=2, obs_dim=1
kf.transition = array![[1.0, 1.0], [0.0, 1.0]];   // F : vitesse constante
kf.observation = array![[1.0, 0.0]];                // H : observer la position seule
kf.process_noise = array![[0.01, 0.0], [0.0, 0.01]]; // Q
kf.measurement_noise = array![[1.0]];                  // R

// Un cycle prédiction-mise à jour
kf.predict(None);                                     // pas d'entrée de commande
kf.update(&array![5.0]);                              // mesure
println!("État : {:?}", kf.state);
println!("Diagonale de la covariance : {:?}", kf.covariance.diag());

// Ou les deux en un seul appel
let estimated = kf.step(&array![5.1], None);
```

## Quand l'utiliser

| Technique | Idéal pour | Limites |
|-----------|-----------|---------|
| **Filtre de Kalman** | Systèmes linéaires avec bruit gaussien, fusion de capteurs | Suppose la linéarité et le bruit gaussien |
| Filtre de Kalman étendu | Systèmes modérément non linéaires | La linéarisation peut diverger en cas de forte non-linéarité |
| Filtre particulaire | Systèmes fortement non linéaires, non gaussiens | Coûteux en calcul (O(N) particules) |
| Moyenne mobile | Lissage simple sans modèle physique | Pas de capacité de prédiction ; introduit du retard |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Règle empirique |
|-----------|-------------------|-----------------|
| `transition` (F) | Comment l'état évolue entre les pas de temps | Dérivé de la physique : vitesse constante, accélération constante, etc. |
| `observation` (H) | Quelles parties de l'état sont mesurables | 1 pour les états directement observés, 0 pour les états cachés |
| `process_noise` (Q) | Erreur du modèle par pas de temps | Q plus grand = moins de confiance dans le modèle, adaptation plus rapide |
| `measurement_noise` (R) | Bruit du capteur | R plus grand = moins de confiance dans les mesures, sortie plus lisse |
| `state` | Estimation initiale de l'état | Meilleure estimation ; le filtre la corrigera en quelques pas |
| `covariance` (P) | Incertitude initiale | Commencer grand (matrice identité) si l'état initial est incertain |

## Pièges courants

1. **Réglage de Q et R.** Le filtre n'est « optimal » que lorsque Q et R reflètent
   fidèlement le bruit réel. Si Q est trop petit, le filtre fait trop confiance au
   modèle et réagit lentement aux changements réels. Si R est trop petit, il suit
   chaque mesure bruitée.

2. **Hypothèse de linéarité.** Le filtre de Kalman standard suppose que F et H sont des
   matrices constantes et que le bruit est gaussien. Pour les modèles non linéaires, le
   filtre peut diverger.

3. **Observabilité.** Si H ne fournit pas assez d'information pour reconstruire l'état
   complet, la covariance des états non observés ne diminuera pas. Par exemple, mesurer
   uniquement la position ne permet pas d'estimer l'accélération sans un modèle les
   reliant.

4. **Stabilité numérique.** La covariance P doit rester symétrique et définie positive.
   Les erreurs accumulées en virgule flottante peuvent briser cette propriété. Forcez
   périodiquement la symétrie : `P = (P + P^T) / 2`.

5. **Inversion de matrice.** L'étape de mise à jour inverse S. Si S est singulière
   (mesures dégénérées), le filtre échouera. La fonction utilitaire
   `constant_velocity_1d` évite ce problème pour le cas courant du suivi 1D.

## Pour aller plus loin

- Combinez plusieurs capteurs (GPS + IMU + boussole) en empilant leurs observations
  dans une seule matrice H et un R bloc-diagonal.
- Pour des modèles variant dans le temps, mettez à jour `kf.transition` à chaque pas
  avant d'appeler `predict`.
- Utilisez `ix_signal::filter::IirFilter::first_order_lowpass()` comme alternative
  plus simple quand vous n'avez pas besoin d'un modèle physique — juste du lissage.
- Injectez les estimations d'état filtrées par Kalman dans
  `ix_chaos::lyapunov::lyapunov_spectrum` pour détecter des dynamiques chaotiques
  dans des données capteur nettoyées.
