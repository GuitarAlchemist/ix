# Analyse de séries temporelles avec ix-signal

## Le problème

Les données ordonnées dans le temps sont partout — cours boursiers à chaque seconde,
relevés de capteurs IoT en continu, temps de réponse enregistrés dans un centre
d'appels d'urgence. Contrairement aux données tabulaires où les lignes sont
indépendantes, les séries temporelles ont une structure temporelle stricte. On ne
peut pas les mélanger, on ne peut pas séparer aléatoirement les jeux
d'entraînement et de test, et le passé contient de l'information sur l'avenir.

Le ML classique traite chaque observation comme interchangeable. Les séries
temporelles brisent cette hypothèse. Si vous répartissez aléatoirement des cours
boursiers entre entraînement et test, votre modèle « voit l'avenir » pendant
l'entraînement — une fuite de données fatale qui produit des métriques
artificiellement bonnes et des performances catastrophiques en production.

## L'intuition

L'idée fondamentale est simple : **le passé prédit l'avenir**. Mais combien de
passé ? Une moyenne glissante sur 5 minutes lisse le bruit mais réagit en retard
aux changements brusques. Une fenêtre de 60 minutes capture les tendances mais
rate les pics. La bonne taille de fenêtre dépend du signal recherché.

Les **fenêtres glissantes** parcourent vos données en calculant des statistiques
(moyenne, écart-type, min, max) sur un voisinage de taille fixe. Elles
transforment un signal bruité en tendance lisse.

Les **caractéristiques de décalage** (lag features) reformulent la série temporelle
en problème d'apprentissage supervisé : « étant donné les N dernières valeurs,
prédire la suivante ». Cela permet de brancher n'importe quel modèle de
régression — régression linéaire, forêt aléatoire, réseau de neurones — sur une
tâche de série temporelle.

L'**EWMA** (moyenne mobile à pondération exponentielle) accorde plus de poids aux
observations récentes, réagissant plus vite aux changements qu'une simple moyenne
glissante.

## Statistiques glissantes

### rolling_mean / rolling_std

Les fonctions de base du lissage de séries temporelles. La moyenne glissante révèle
les tendances ; l'écart-type glissant révèle la volatilité.

```rust
use ix_signal::timeseries::{rolling_mean, rolling_std};

let prices = vec![100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 110.0];

// 3-period moving average — smooths noise, reveals uptrend
let ma = rolling_mean(&prices, 3);
// [NaN, NaN, 101.0, 102.67, 103.0, 105.0, 106.67]

// 3-period volatility — spikes signal regime changes
let vol = rolling_std(&prices, 3);
// Low vol = steady market, high vol = turbulence
```

Les `window - 1` premières valeurs sont `NaN` — il n'y a pas encore assez de
données pour remplir la fenêtre. C'est intentionnel : pas de regard vers l'avenir,
pas de triche.

### rolling_min / rolling_max

Suivent les extrêmes sur une fenêtre glissante. Utiles pour les niveaux de
support/résistance en finance ou la détection de violations de plage de capteurs.

```rust
use ix_signal::timeseries::{rolling_min, rolling_max};

let temps = vec![72.0, 68.0, 75.0, 71.0, 80.0, 77.0];

let lows  = rolling_min(&temps, 3);  // Trailing 3-period low
let highs = rolling_max(&temps, 3);  // Trailing 3-period high
// Bandwidth (highs - lows) measures recent variability
```

### EWMA — Moyenne mobile à pondération exponentielle

L'EWMA réagit plus vite que la moyenne glissante car elle pondère les données
récentes de manière exponentielle. Le paramètre `alpha` contrôle la réactivité :
un alpha élevé (0.9) suit de près, un alpha faible (0.1) lisse fortement.

```rust
use ix_signal::timeseries::ewma;

let response_times = vec![120.0, 125.0, 190.0, 185.0, 130.0, 128.0];

// alpha=0.3: smooth, good for long-term trend
let smooth = ewma(&response_times, 0.3);

// alpha=0.8: reactive, good for anomaly detection
let reactive = ewma(&response_times, 0.8);
```

## Ingénierie des caractéristiques

### lag_features — Rendre les séries temporelles compatibles avec le ML

La transformation clé : convertir une séquence en paires (X, y) où X contient les
N valeurs précédentes et y est la valeur suivante. Cela transforme la prévision de
séries temporelles en régression classique.

```rust
use ix_signal::timeseries::lag_features;

// Daily stock closes
let closes = vec![100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 110.0];

// Use 3 days of history to predict the next day
let (x, y) = lag_features(&closes, 3);
// x[0] = [100, 102, 101] -> y[0] = 105
// x[1] = [102, 101, 105] -> y[1] = 103
// x[2] = [101, 105, 103] -> y[2] = 107
// x[3] = [105, 103, 107] -> y[3] = 110
assert_eq!(x.nrows(), 4);
assert_eq!(x.ncols(), 3);
```

### lag_features_with_stats — Des caractéristiques plus riches

Ajoute la moyenne glissante et l'écart-type glissant calculés sur la fenêtre de
décalage. Ces caractéristiques construites améliorent souvent la précision du modèle
car elles encodent directement la tendance locale et la volatilité.

```rust
use ix_signal::timeseries::lag_features_with_stats;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
let (x, y) = lag_features_with_stats(&data, 3);
// Each row: [lag1, lag2, lag3, rolling_mean, rolling_std]
assert_eq!(x.ncols(), 5);  // 3 lags + 2 stats
assert_eq!(x.nrows(), 4);  // 7 - 3 = 4 samples
```

### difference et pct_change — Transformations vers la stationnarité

De nombreux modèles supposent la stationnarité (moyenne et variance constantes). Les
prix bruts sont non stationnaires — ils ont une tendance haussière. La différenciation
et le pourcentage de variation corrigent cela.

```rust
use ix_signal::timeseries::{difference, pct_change};

let prices = vec![100.0, 110.0, 108.0, 115.0];

// First-order difference: absolute changes
let diff = difference(&prices, 1);
// [10.0, -2.0, 7.0]

// Percent change: relative changes (better for comparing assets)
let pct = pct_change(&prices);
// [0.1, -0.0182, 0.0648]

// Second-order difference: acceleration of price changes
let diff2 = difference(&prices, 2);
// [-12.0, 9.0]
```

## Séparation temporelle entraînement/test

**N'utilisez jamais de séparation aléatoire sur des séries temporelles.** Si vous
assignez aléatoirement les données de mars à l'entraînement et celles de janvier au
test, votre modèle a déjà vu l'avenir. Utilisez `temporal_split` pour garantir que
les données d'entraînement précèdent toujours les données de test.

```rust
use ix_signal::timeseries::temporal_split;

// 100 observations, 80% train / 20% test
let (train_idx, test_idx) = temporal_split(100, 0.8);
assert_eq!(train_idx.len(), 80);   // indices 0..79
assert_eq!(test_idx.len(), 20);    // indices 80..99
// Training ends BEFORE testing begins — no leakage
```

## Pipeline complet — Prédiction de cours boursier

Le tout assemblé : caractéristiques de décalage, séparation temporelle et évaluation.

```rust
use ix_signal::timeseries::{lag_features_with_stats, temporal_split};

// Simulated daily closes (30 days)
let closes: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64) * 0.5
    + (i as f64 * 0.7).sin() * 3.0).collect();

// Step 1: Create features with 5-day lookback
let (x, y) = lag_features_with_stats(&closes, 5);
// x has 7 columns: 5 lags + rolling_mean + rolling_std

// Step 2: Temporal split (80/20)
let (train_idx, test_idx) = temporal_split(x.nrows(), 0.8);

// Step 3: Extract train/test sets (preserving temporal order)
// train_x = x[train_idx], train_y = y[train_idx]
// test_x  = x[test_idx],  test_y  = y[test_idx]

// Step 4: Train a model (e.g., LinearRegression from ix-supervised)
// Step 5: Predict on test_x, compute RMSE against test_y
```

## Application PSAP — Surveillance des temps de réponse

Les centres de traitement des appels d'urgence (PSAP) doivent respecter les normes
NFPA 1221 : 90 % des appels d'urgence décrochés en moins de 15 secondes, 95 % en
moins de 20 secondes. La surveillance des tendances des temps de réponse est
essentielle pour la conformité.

```rust
use ix_signal::timeseries::{rolling_mean, expanding_mean, ewma, rolling_std};

// Hourly average response times (seconds) over 24 hours
let response_times = vec![
    12.0, 11.5, 13.0, 14.5, 16.0, 18.0,  // midnight-5am (low volume)
    14.0, 12.0, 11.0, 10.5, 11.0, 12.5,  // 6am-11am (morning)
    13.0, 14.0, 13.5, 12.0, 11.5, 15.0,  // noon-5pm (afternoon)
    17.0, 19.0, 16.0, 14.0, 13.0, 12.0,  // 6pm-11pm (evening rush)
];

// 4-hour rolling average — spot short-term trends
let trend = rolling_mean(&response_times, 4);

// EWMA with alpha=0.3 — smooth long-term baseline
let baseline = ewma(&response_times, 0.3);

// Expanding mean — cumulative daily performance
let daily_avg = expanding_mean(&response_times);

// Rolling volatility — flag unstable periods
let volatility = rolling_std(&response_times, 4);

// Alert logic: if rolling mean exceeds 15s threshold for compliance
for (hour, &avg) in trend.iter().enumerate() {
    if !avg.is_nan() && avg > 15.0 {
        // Flag: NFPA compliance risk at this hour
        println!("ALERT hour {}: avg response {:.1}s exceeds 15s", hour, avg);
    }
}
```

## Quand utiliser chaque fonction

| Fonction | Cas d'utilisation |
|---|---|
| `rolling_mean` | Détection de tendance, lissage du bruit, moyennes mobiles |
| `rolling_std` | Mesure de volatilité, détection d'anomalies |
| `rolling_min` / `rolling_max` | Suivi de plage, support/résistance, limites de capteurs |
| `ewma` | Lissage adaptatif, surveillance en temps réel, alertes |
| `expanding_mean` | Métriques de performance cumulatives, moyennes courantes |
| `lag_features` | Convertir une série temporelle en problème de ML supervisé |
| `lag_features_with_stats` | Caractéristiques ML enrichies avec tendance/volatilité |
| `difference` | Supprimer les tendances, atteindre la stationnarité |
| `pct_change` | Rendements relatifs, comparaison inter-actifs |
| `temporal_split` | Évaluation honnête entraînement/test pour séries temporelles |

## Pièges à éviter

- **Fuite de données par séparation aléatoire** — L'erreur la plus courante.
  Utilisez toujours `temporal_split`. Un modèle qui « voit l'avenir » pendant
  l'entraînement paraîtra brillant en validation et échouera en production.

- **Biais d'anticipation** (look-ahead bias) — Calculer des caractéristiques avec
  des données futures. Les statistiques glissantes d'ix-signal ne regardent que vers
  le passé (les `window - 1` premières valeurs sont NaN). Respectez ces NaN ; ne
  les imputez pas avec des valeurs calculées à partir du futur.

- **Stationnarité** — Les modèles linéaires supposent une moyenne et une variance
  constantes. Appliquez `difference` ou `pct_change` avant de modéliser des niveaux
  de prix bruts. Vérifiez que votre série différenciée semble stationnaire (moyenne
  constante, dispersion constante).

- **Choix de la taille de fenêtre** — Trop petite : bruitée, sur-ajustement aux
  fluctuations récentes. Trop grande : retard par rapport aux vrais changements.
  Commencez avec la connaissance du domaine (par exemple, semaine de 5 jours de
  bourse, cycle journalier de 24 heures) et itérez.

- **Réglage de l'alpha EWMA** — Un alpha élevé (proche de 1.0) suit chaque
  oscillation. Un alpha faible (proche de 0.0) réagit à peine. Pour la détection
  d'anomalies, utilisez un alpha élevé ; pour l'estimation de tendance, utilisez
  un alpha faible.
