# Filtres numériques (FIR et IIR)

## Le problème

Vous concevez un égaliseur audio pour une application de streaming musical. L'audio brut
contient un bourdonnement basse fréquence dû à l'alimentation (60 Hz), un souffle haute
fréquence provenant d'un préamplificateur de microphone bruyant, et la musique que vous
souhaitez conserver. Vous devez supprimer chirurgicalement les bandes de fréquence
indésirables sans déformer le reste.

Les filtres numériques résolvent ce problème dans de nombreux domaines : suppression de la
dérive de ligne de base dans les signaux ECG, anti-repliement avant sous-échantillonnage,
isolation d'un canal radio par rapport aux canaux adjacents, et lissage en temps réel de
lectures de capteurs bruités.

## L'intuition

Un filtre numérique est une recette pour calculer chaque échantillon de sortie comme une
combinaison pondérée des échantillons d'entrée récents (et parfois des échantillons de
sortie récents).

- **Filtre FIR (réponse impulsionnelle finie) :** Semblable à une moyenne mobile pondérée.
  La sortie ne dépend que des valeurs d'*entrée* actuelles et passées. Elle revient
  toujours à zéro après l'arrêt de l'entrée — la « réponse impulsionnelle » est finie.

- **Filtre IIR (réponse impulsionnelle infinie) :** La sortie dépend aussi des valeurs
  de *sortie* passées (rétroaction). Cela permet de construire des filtres plus sélectifs
  avec moins de coefficients, mais la rétroaction peut résonner indéfiniment — la réponse
  impulsionnelle est théoriquement infinie.

Pensez au FIR comme un filtre à café (un seul passage, pas de rétroaction) et à l'IIR
comme un thermostat (la sortie influence la sortie suivante par rétroaction).

## Fonctionnement

### Filtre FIR

```
y[n] = sum_{k=0}^{M} h[k] * x[n-k]
```

**En clair :** Chaque échantillon de sortie est une somme pondérée des M+1 échantillons
d'entrée les plus récents. Les poids h[] sont les coefficients du filtre, conçus pour
laisser passer certaines fréquences et en bloquer d'autres.

### Filtre IIR

```
y[n] = sum_{k=0}^{P} b[k] * x[n-k]  -  sum_{k=1}^{Q} a[k] * y[n-k]
```

**En clair :** Même somme directe que le FIR (coefficients b[]), plus une somme de
rétroaction utilisant les valeurs de sortie passées (coefficients a[]). C'est la
rétroaction qui donne aux filtres IIR une « mémoire » plus longue que leur nombre de
coefficients.

### Méthodes de conception

| Type de filtre | Méthode de conception | API ix |
|----------------|----------------------|--------|
| FIR passe-bas | Sinus cardinal fenêtré (fenêtre de Hamming) | `FirFilter::lowpass(cutoff, order)` |
| FIR passe-haut | Inversion spectrale du passe-bas | `FirFilter::highpass(cutoff, order)` |
| FIR passe-bande | Différence de deux passe-bas | `FirFilter::bandpass(low, high, order)` |
| IIR 1er ordre passe-bas | Moyenne mobile exponentielle | `IirFilter::first_order_lowpass(alpha)` |
| IIR 2e ordre passe-bas | Butterworth (maximalement plat) | `butterworth_lowpass_2nd(cutoff)` |

## En Rust

```rust
use ix_signal::filter::{
    FirFilter, IirFilter, butterworth_lowpass_2nd,
};

// --- FIR passe-bas : garder les fréquences en dessous de 0.1 * Nyquist ---
let fir = FirFilter::lowpass(0.1, 32);   // cutoff=0.1, order=32
let noisy_signal: Vec<f64> = (0..256)
    .map(|i| {
        let t = i as f64 / 256.0;
        // Signal basse fréquence + bruit haute fréquence
        (2.0 * std::f64::consts::PI * 5.0 * t).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 100.0 * t).sin()
    })
    .collect();
let filtered = fir.apply(&noisy_signal);
// La composante haute fréquence est atténuée après le transitoire initial

// --- FIR passe-haut : supprimer la dérive basse fréquence ---
let hp = FirFilter::highpass(0.05, 64);
let stable = hp.apply(&noisy_signal);

// --- FIR passe-bande : isoler une plage de fréquences spécifique ---
let bp = FirFilter::bandpass(0.05, 0.15, 64);
let band_signal = bp.apply(&noisy_signal);

// --- IIR 1er ordre passe-bas (lissage exponentiel) ---
let iir = IirFilter::first_order_lowpass(0.1);  // alpha=0.1 = lissage fort
let smoothed = iir.apply(&noisy_signal);

// --- Butterworth 2e ordre passe-bas (bande passante maximalement plate) ---
let bw = butterworth_lowpass_2nd(0.1);  // fréquence de coupure normalisée 0.1
let butter_filtered = bw.apply(&noisy_signal);
```

## Quand l'utiliser

| Filtre | Idéal pour | Compromis |
|--------|-----------|-----------|
| **FIR passe-bas** | Coupure fréquentielle nette, phase linéaire | Nécessite beaucoup de coefficients pour une coupure abrupte ; latence plus élevée |
| **FIR passe-haut** | Suppression de la composante DC / dérive de ligne de base | Mêmes exigences d'ordre que le passe-bas |
| **FIR passe-bande** | Isolation d'une bande de fréquence | L'ordre doit être suffisant pour les deux bords |
| **IIR 1er ordre** | Lissage temps réel simple | Atténuation douce ; inadapté au filtrage sélectif |
| **Butterworth** | Bande passante maximalement plate, atténuation plus raide | Distorsion de phase (phase non linéaire) ; peut résonner |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Règle empirique |
|-----------|-------------------|-----------------|
| `cutoff` | Fréquence normalisée (0 à 0.5, où 0.5 = Nyquist) | cutoff = fréquence_désirée_hz / sample_rate |
| `order` | Nombre de coefficients moins 1 (FIR) | Ordre plus élevé = transition plus nette mais plus de latence ; commencer avec 32-64 |
| `alpha` | Facteur de lissage IIR (0 à 1) | Plus alpha est petit, plus le lissage est fort ; alpha = dt / (RC + dt) |

## Pièges courants

1. **Transitoire du filtre.** Les `order` premiers échantillons de sortie sont affectés par
   le « remplissage » du filtre. Ignorez ou écartez ces échantillons pour l'analyse.

2. **Fréquence normalisée vs absolue.** Toutes les valeurs de coupure sont normalisées
   dans [0, 0.5], où 0.5 est la fréquence de Nyquist (la moitié de la fréquence
   d'échantillonnage). Pour filtrer à 100 Hz avec un taux d'échantillonnage de 1000 Hz :
   `cutoff = 100 / 1000 = 0.1`.

3. **Instabilité IIR.** Les filtres à rétroaction peuvent devenir instables si les
   coefficients du dénominateur a[] sont mal choisis. La conception Butterworth garantit
   la stabilité, mais les coefficients IIR ajustés manuellement nécessitent une analyse
   attentive des pôles.

4. **Distorsion de phase.** Les filtres IIR introduisent un retard dépendant de la
   fréquence (phase non linéaire). Les filtres FIR à coefficients symétriques ont une
   phase parfaitement linéaire. Utilisez un FIR quand la phase est importante (audio,
   communications).

5. **Ordre vs sélectivité.** Un filtre FIR de faible ordre a une atténuation progressive.
   Si vous avez besoin d'une coupure nette, augmentez l'ordre — mais cela augmente à la
   fois le calcul et la latence.

## Pour aller plus loin

- Appliquez `ix_signal::window::hamming()` à votre signal avant l'analyse FFT, ou
  utilisez ces filtres comme pré-traitement pour isoler une bande avant l'analyse
  spectrale.
- Chaînez les filtres : `FirFilter::highpass(0.02, 64)` suivi de
  `FirFilter::lowpass(0.2, 64)` pour un passe-bande propre avec un contrôle indépendant
  de chaque bord.
- Utilisez `ix_signal::fft::rfft` + `ix_signal::fft::irfft` pour le filtrage dans le
  domaine fréquentiel, comme alternative à la convolution temporelle pour les signaux
  très longs.
- Injectez les signaux filtrés dans `ix_signal::kalman::KalmanFilter` pour l'estimation
  d'état sur des données pré-nettoyées.
