# Transformée en ondelettes (Haar DWT)

## Le problème

Vous concevez un système d'analyse sismique. Les signaux de tremblement de terre
contiennent des salves transitoires abruptes (ondes P, ondes S) noyées dans un bruit
de fond lentement variable. Une FFT vous indique *quelles fréquences* sont présentes
mais pas *quand* elles sont apparues. Vous avez besoin d'une transformée qui fournit
simultanément l'information en fréquence et en temps, afin de localiser précisément
l'heure d'arrivée de chaque phase sismique.

Le même besoin apparaît en compression d'image (JPEG 2000 utilise les ondelettes), en
détection d'anomalies ECG et en débruitage de séries temporelles financières.

## L'intuition

La FFT décompose un signal en sinusoïdes infinies. Les ondelettes utilisent à la place
de courtes « petites ondes » localisées. La différence peut se résumer ainsi :

- **FFT :** « Quelles notes figurent dans cette chanson entière ? »
- **Ondelettes :** « Quelles notes sont jouées *en ce moment*, seconde par seconde ? »

L'ondelette de Haar est la plus simple possible : une fonction en escalier qui vaut +1
puis -1. Malgré sa simplicité, elle capture l'idée essentielle de l'analyse
multi-résolution.

À chaque niveau de décomposition, le signal est scindé en :
- **Coefficients d'approximation :** la partie lisse, lentement variable (basses fréquences).
- **Coefficients de détail :** la partie abrupte, rapidement variable (hautes fréquences).

C'est comme zoomer progressivement en arrière sur une photographie : à chaque niveau de
zoom, on perd les détails fins mais on conserve la structure globale.

## Fonctionnement

### Transformée de Haar directe à un niveau

Étant donné un signal de longueur N, on produit N/2 coefficients d'approximation et N/2
coefficients de détail :

```
approx[i] = (1/sqrt(2)) * (signal[2i] + signal[2i+1])
detail[i] = (1/sqrt(2)) * (signal[2i] - signal[2i+1])
```

**En clair :** Prenez chaque paire d'échantillons adjacents. Leur moyenne (mise à
l'échelle) devient un coefficient d'approximation. Leur différence (mise à l'échelle)
devient un coefficient de détail. L'approximation capture la tendance ; le détail capture
la variation.

### DWT multi-niveaux

Appliquez la transformée directe récursivement aux coefficients d'approximation. Après L
niveaux, vous obtenez un vecteur d'approximation final et L vecteurs de détail, chacun à
une échelle différente (bande de fréquence).

### Débruitage par seuillage doux

```
coeff_denoised = sign(coeff) * max(|coeff| - threshold, 0)
```

**En clair :** Les petits coefficients de détail sont probablement du bruit, les grands
sont probablement du signal. Le seuillage doux réduit tout vers zéro du montant du seuil,
et tout ce qui est en dessous du seuil devient exactement zéro.

## En Rust

```rust
use ix_signal::wavelet::{
    haar_forward, haar_inverse,
    haar_dwt, haar_idwt,
    wavelet_denoise,
};

// --- Transformée à un niveau ---
let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let (approx, detail) = haar_forward(&signal);
// approx capture la tendance lisse, detail capture les changements brusques

// Reconstruction parfaite
let recovered = haar_inverse(&approx, &detail);
for (a, b) in signal.iter().zip(recovered.iter()) {
    assert!((a - b).abs() < 1e-10);
}

// --- Décomposition multi-niveaux ---
let (final_approx, details) = haar_dwt(&signal, 3);
// details[0] = échelle la plus fine (fréquences les plus hautes)
// details[2] = échelle la plus grossière (fréquences les plus basses au-dessus du DC)
// final_approx = le résidu très lisse

let reconstructed = haar_idwt(&final_approx, &details);
assert_eq!(reconstructed.len(), signal.len());

// --- Débruitage d'un signal bruité ---
let noisy: Vec<f64> = (0..64)
    .map(|i| (i as f64 * 0.1).sin() + 0.3 * ((i * 7) as f64 % 1.3 - 0.65))
    .collect();

let clean = wavelet_denoise(&noisy, 3, 0.4);
// clean conserve la sinusoïde mais supprime le bruit pseudo-aléatoire
assert_eq!(clean.len(), noisy.len());
```

## Quand l'utiliser

| Technique | Idéal pour | Limites |
|-----------|-----------|---------|
| **Haar DWT** | Analyse multi-résolution rapide, débruitage, compression | La base en escalier produit des artefacts en blocs |
| FFT | Analyse fréquentielle pure de signaux stationnaires | Pas de localisation temporelle |
| STFT | Analyse temps-fréquence à résolution fixe | Le compromis de résolution est fixé par la taille de fenêtre |
| Ondelettes de Daubechies | Analyse en ondelettes plus lisse (moins d'artefacts) | Conception de filtre plus complexe |

## Paramètres clés

| Paramètre | Ce qu'il contrôle | Règle empirique |
|-----------|-------------------|-----------------|
| `levels` | Profondeur de la décomposition multi-résolution | Chaque niveau divise la longueur du signal par 2 ; max = log2(N) |
| `threshold` | Agressivité du débruitage | Commencer avec sigma * sqrt(2 * ln(N)) où sigma est l'écart-type du bruit |
| Longueur du signal | Doit être divisible par 2^levels | Compléter avec des zéros si nécessaire |

## Pièges courants

1. **La longueur du signal doit être paire** (et divisible par 2^levels pour la DWT
   multi-niveaux). Les signaux de longueur impaire perdront silencieusement le dernier
   échantillon.

2. **Artefacts de Haar.** L'ondelette de Haar est une fonction en escalier, elle introduit
   donc des discontinuités en blocs dans les signaux débruités ou compressés. Pour des
   résultats plus lisses, envisagez des ondelettes d'ordre supérieur.

3. **Choix du seuil.** Trop bas = le bruit survit ; trop haut = le signal est déformé.
   Le seuil universel sigma * sqrt(2*ln(N)) est un point de départ raisonnable, mais un
   réglage spécifique à l'application est généralement nécessaire.

4. **Redistribution de l'énergie.** Contrairement à la FFT, les coefficients d'ondelettes
   à différents niveaux représentent des bandes de fréquence différentes à des résolutions
   temporelles différentes. Ne comparez pas directement les amplitudes entre niveaux.

## Pour aller plus loin

- Combinez les ondelettes avec `ix_signal::fft::rfft` pour une analyse en deux étapes :
  les ondelettes pour la localisation temporelle, la FFT pour l'identification précise
  des fréquences dans chaque fenêtre.
- Utilisez `ix_signal::filter::FirFilter::bandpass()` pour pré-filtrer avant l'analyse
  en ondelettes quand vous connaissez la bande de fréquence d'intérêt.
- Pour la compression d'image, appliquez `haar_forward` le long des lignes, puis le long
  des colonnes (DWT 2D).
- Explorez le module `ix_signal::spectral` pour l'analyse temps-fréquence par STFT
  comme alternative aux ondelettes.
