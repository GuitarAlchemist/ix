# Transformée de Fourier rapide (FFT)

## Le problème

Vous construisez un système de surveillance vibratoire pour des machines industrielles.
Les capteurs enregistrent des milliers d'échantillons d'accélération par seconde, mais
une forme d'onde brute dans le domaine temporel ne vous dit presque rien sur *quel
roulement est défaillant*. Chaque défaut mécanique vibre à une fréquence caractéristique.
Vous devez décomposer le signal brut en ses fréquences constitutives afin de faire
correspondre les pics aux signatures de pannes connues.

Le même défi se retrouve dans les analyseurs de spectre audio, les récepteurs radio,
l'échographie médicale et les pipelines de reconnaissance vocale.

## L'intuition

Imaginez que vous êtes dans une pièce où un pianiste, un guitariste et un batteur jouent
simultanément. Vos oreilles perçoivent une seule forme d'onde combinée, et pourtant votre
cerveau distingue sans effort chaque instrument. La FFT fait la même chose
mathématiquement : elle prend un signal mixte et le décompose en une liste d'ingrédients
sinusoïdaux purs, chacun avec sa propre fréquence et son amplitude.

Pensez au signal comme à un smoothie. La FFT est une machine qui « dé-mixe » le smoothie
pour retrouver ses fruits d'origine, en vous disant « il y a 40 % de fraise (10 Hz),
30 % de banane (50 Hz), et 30 % de mangue (120 Hz) ».

## Fonctionnement

### La transformée de Fourier discrète (DFT)

Étant donné N échantillons x[0], x[1], ..., x[N-1], la DFT calcule N coefficients
complexes dans le domaine fréquentiel :

```
X[k] = sum_{n=0}^{N-1} x[n] * e^{-j*2*pi*k*n/N}
```

**En clair :** Pour chaque bin fréquentiel k, parcourez chaque échantillon en le
multipliant par un nombre complexe tournant à cette fréquence. Le résultat indique
la quantité de fréquence k présente (le module) et où elle commence dans le cycle
(la phase).

L'évaluation directe coûte O(N^2). La FFT exploite la symétrie des facteurs de rotation
(twiddle factors) pour scinder une DFT de taille N en deux DFT de taille N/2
récursivement, réduisant le coût à O(N log N).

### Relations clés

| Formule | En clair |
|---------|----------|
| `magnitude = sqrt(re^2 + im^2)` | Quelle est l'intensité de cette fréquence ? |
| `phase = atan2(im, re)` | Où dans son cycle cette fréquence commence-t-elle ? |
| `power = re^2 + im^2` | Énergie concentrée à cette fréquence |
| `frequency_bins[k] = k * sample_rate / N` | À quelle fréquence réelle correspond le bin k ? |
| `ifft(fft(x)) = x` | On peut reconstruire parfaitement le signal original |

### Théorème de Parseval

L'énergie totale dans le domaine temporel est égale à l'énergie totale dans le domaine
fréquentiel divisée par N. C'est un test de cohérence utile : si la sortie de votre FFT
a une énergie totale très différente, quelque chose ne va pas.

## En Rust

```rust
use ix_signal::fft::{
    Complex, fft, ifft, rfft, irfft,
    magnitude_spectrum, power_spectrum, frequency_bins,
};

// 1. Construire un signal test : 10 Hz + 50 Hz à 256 Hz de fréquence d'échantillonnage
let sample_rate = 256.0;
let n = 256;
let signal: Vec<f64> = (0..n)
    .map(|i| {
        let t = i as f64 / sample_rate;
        (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 50.0 * t).sin()
    })
    .collect();

// 2. Calculer la FFT du signal réel
let spectrum = rfft(&signal);

// 3. Inspecter le module et la puissance
let mags = magnitude_spectrum(&spectrum);
let power = power_spectrum(&spectrum);

// 4. Associer les indices de bin aux Hz
let freqs = frequency_bins(n, sample_rate);
for (bin, &mag) in mags.iter().enumerate().take(n / 2) {
    if mag > 10.0 {
        println!("Pic à {:.1} Hz, module = {:.2}", freqs[bin], mag);
    }
}

// 5. Aller-retour : reconstruire le signal à partir du spectre
let recovered = irfft(&spectrum);
for (a, b) in signal.iter().zip(recovered.iter()) {
    assert!((a - b).abs() < 1e-10);
}

// 6. Travailler directement avec les nombres complexes
let c = Complex::new(3.0, 4.0);
assert!((c.magnitude() - 5.0).abs() < 1e-10);
assert!((c.phase() - (4.0_f64).atan2(3.0)).abs() < 1e-10);
```

> Exemple complet exécutable : [examples/signal/fft_analysis.rs](../../examples/signal/fft_analysis.rs)

## Quand l'utiliser

| Technique | Idéal pour | Limites |
|-----------|-----------|---------|
| **FFT** | Signaux stationnaires, analyse spectrale temps réel, identification de fréquences | Suppose que le signal est périodique sur la fenêtre ; faible résolution temporelle |
| Ondelettes | Signaux non stationnaires, détection de transitoires | Coût de calcul plus élevé ; plus de paramètres à régler |
| FFT à court terme (STFT) | Spectres variant dans le temps, spectrogrammes | La fenêtre fixe impose un compromis résolution temps/fréquence |
| Filtres FIR/IIR | Suppression de bandes de fréquence connues | Il faut déjà savoir quelles fréquences garder/supprimer |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Règle empirique |
|-----------|-------------------|-----------------|
| `fft_size` (N) | Résolution fréquentielle = sample_rate / N | Plus N est grand, meilleure est la résolution fréquentielle mais moins bonne est la résolution temporelle |
| `sample_rate` | Fréquence maximale détectable = sample_rate / 2 (Nyquist) | Doit être au moins 2x la fréquence la plus haute d'intérêt |
| Fonction de fenêtrage | Réduit les fuites spectrales dues aux bords non périodiques | Appliquer une fenêtre de Hamming ou Hanning avant d'appeler `fft` |

## Pièges courants

1. **Longueurs non puissance de 2.** L'implémentation complète automatiquement avec des
   zéros jusqu'à la puissance de 2 suivante, mais cela modifie l'espacement de vos bins
   fréquentiels. Complétez explicitement si vous avez besoin d'un alignement exact.

2. **Fuites spectrales.** Si votre signal ne contient pas un nombre entier exact de
   cycles dans la fenêtre, l'énergie « fuit » d'un pic réel vers les bins voisins.
   Appliquez toujours une fonction de fenêtrage (Hamming, Hanning, Blackman) avant la FFT.

3. **Repliement spectral (aliasing).** Les fréquences au-dessus de sample_rate/2 se
   replient dans les bins inférieurs, produisant des pics fantômes. Assurez-vous que
   votre fréquence d'échantillonnage respecte le critère de Nyquist.

4. **Interprétation de la seconde moitié.** Pour les entrées réelles, les bins N/2+1 à
   N-1 sont le miroir conjugué des bins 1 à N/2-1. Seuls les N/2+1 premiers bins
   contiennent de l'information unique.

5. **Composante continue (DC).** Une moyenne non nulle dans votre signal crée un pic
   important au bin 0 (DC). Soustrayez la moyenne avant la FFT si seules les composantes
   oscillantes vous intéressent.

## Pour aller plus loin

- Appliquez une **fenêtre de Hamming** avant la FFT pour réduire les fuites :
  `ix_signal::window::hamming(n)`.
- Utilisez `ix_signal::spectral` pour la **FFT à court terme** (spectrogrammes) lorsque
  le contenu fréquentiel de votre signal évolue dans le temps.
- Combinez avec `ix_signal::filter::FirFilter::lowpass()` pour pré-filtrer avant
  l'analyse, en isolant une bande de fréquence d'intérêt.
- Pour les signaux non stationnaires (données sismiques, parole), voir
  [ondelettes.md](ondelettes.md).
