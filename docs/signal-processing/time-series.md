# Time Series Analysis with ix-signal

## The Problem

Time-ordered data is everywhere — stock prices ticking every second, sensor readings
streaming from IoT devices, response times logged at a 911 dispatch center. Unlike
tabular data where rows are independent, time series data has a strict temporal
structure. You cannot shuffle it, you cannot randomly sample train/test splits, and
the past carries information about the future.

Standard ML treats each observation as exchangeable. Time series breaks that
assumption. If you randomly split stock prices into train and test sets, your model
will "see the future" during training — a fatal form of data leakage that produces
unrealistically good metrics and catastrophic real-world performance.

## The Intuition

The core idea is simple: **past predicts future**. But how much past? A 5-minute
rolling average smooths noise but lags behind sudden shifts. A 60-minute window
captures trends but misses spikes. The right window size depends on the signal you
are hunting.

**Rolling windows** slide across your data, computing statistics (mean, std, min,
max) over a fixed-size neighborhood. They turn a noisy signal into a smooth trend.

**Lag features** reframe time series as a supervised learning problem: "given the
last N values, predict the next one." This lets you plug any regression model —
linear regression, random forest, neural network — into a time series task.

**EWMA** (Exponentially Weighted Moving Average) gives more weight to recent
observations, reacting faster to changes than a simple rolling mean.

**Drift detectors** go one step further: instead of smoothing the signal, they
track whether the data-generating process itself has shifted. This is useful for
monitoring model error rates, service latency, and sensor baselines.

## Rolling Statistics

### rolling_mean / rolling_std

The workhorses of time series smoothing. Rolling mean reveals trends; rolling
standard deviation reveals volatility.

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

The first `window - 1` values are `NaN` — there is not enough data to fill the
window yet. This is intentional: no look-ahead, no cheating.

### rolling_min / rolling_max

Track extremes over a sliding window. Useful for support/resistance levels in
finance or detecting sensor range violations.

```rust
use ix_signal::timeseries::{rolling_min, rolling_max};

let temps = vec![72.0, 68.0, 75.0, 71.0, 80.0, 77.0];

let lows  = rolling_min(&temps, 3);  // Trailing 3-period low
let highs = rolling_max(&temps, 3);  // Trailing 3-period high
// Bandwidth (highs - lows) measures recent variability
```

### EWMA — Exponentially Weighted Moving Average

EWMA reacts faster than rolling mean because it weights recent data exponentially
more. The `alpha` parameter controls responsiveness: high alpha (0.9) tracks closely,
low alpha (0.1) smooths aggressively.

```rust
use ix_signal::timeseries::ewma;

let response_times = vec![120.0, 125.0, 190.0, 185.0, 130.0, 128.0];

// alpha=0.3: smooth, good for long-term trend
let smooth = ewma(&response_times, 0.3);

// alpha=0.8: reactive, good for anomaly detection
let reactive = ewma(&response_times, 0.8);
```

## Drift Detection

### DDM — Drift Detection Method

DDM works on a stream of binary outcomes, usually "correct vs error" for a model
or rule system. It tracks the running error rate and raises a warning or drift
state when performance degrades beyond its historical baseline.

```rust
use ix_signal::timeseries::{ddm_detect, DdmConfig, DriftState};

let mut errors = vec![false; 60];
errors.extend(vec![true; 40]); // sudden degradation

let snapshots = ddm_detect(&errors, DdmConfig::default());
assert!(snapshots.iter().any(|s| s.state == DriftState::Warning));
assert!(snapshots.iter().any(|s| s.state == DriftState::Drift));
```

Use DDM when you already have a supervised signal such as classification error,
SLA breach/not-breach, or reviewer reject/accept.

### Page-Hinkley — Mean Shift Detection

Page-Hinkley watches a numeric stream directly. It is a good fit for latency,
queue depth, throughput, temperature, and any metric where you care about a
persistent mean shift rather than isolated spikes.

```rust
use ix_signal::timeseries::{page_hinkley_detect, DriftState, PageHinkleyConfig};

let mut response_times = vec![120.0; 80];
response_times.extend(vec![180.0; 40]); // new slower regime

let snapshots = page_hinkley_detect(
    &response_times,
    PageHinkleyConfig {
        min_samples: 20,
        delta: 0.05,
        lambda: 10.0,
        alpha: 1.0,
    },
);
assert!(snapshots.iter().any(|s| s.state == DriftState::Drift));
```

Use Page-Hinkley when you want low-state online monitoring of numeric metrics.

## Feature Engineering

### lag_features — Making Time Series ML-Ready

The key transformation: convert a sequence into (X, y) pairs where X contains the
previous N values and y is the next value. This turns time series forecasting into
standard regression.

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

### lag_features_with_stats — Richer Features

Appends rolling mean and rolling standard deviation computed from the lag window.
These engineered features often improve model accuracy because they encode the
local trend and volatility directly.

```rust
use ix_signal::timeseries::lag_features_with_stats;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
let (x, y) = lag_features_with_stats(&data, 3);
// Each row: [lag1, lag2, lag3, rolling_mean, rolling_std]
assert_eq!(x.ncols(), 5);  // 3 lags + 2 stats
assert_eq!(x.nrows(), 4);  // 7 - 3 = 4 samples
```

### difference and pct_change — Stationarity Transforms

Many models assume stationarity (constant mean and variance). Raw prices are
non-stationary — they trend upward. Differencing and percent change fix this.

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

## Temporal Train/Test Split

**Never use random splits on time series.** If you randomly assign March data to
training and January data to testing, your model has already seen the future. Use
`temporal_split` to ensure training data always comes before test data.

```rust
use ix_signal::timeseries::temporal_split;

// 100 observations, 80% train / 20% test
let (train_idx, test_idx) = temporal_split(100, 0.8);
assert_eq!(train_idx.len(), 80);   // indices 0..79
assert_eq!(test_idx.len(), 20);    // indices 80..99
// Training ends BEFORE testing begins — no leakage
```

## Full Pipeline Example — Stock Price Prediction

Putting it all together: lag features, temporal split, and evaluation.

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

## PSAP Application — Response Time Monitoring

Public Safety Answering Points (PSAPs) must meet NFPA 1221 standards: 90% of
emergency calls answered within 15 seconds, 95% within 20 seconds. Monitoring
response time trends is critical for compliance.

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

## When to Use Each Function

| Function | Use Case |
|---|---|
| `rolling_mean` | Trend detection, noise smoothing, moving averages |
| `rolling_std` | Volatility measurement, anomaly detection |
| `rolling_min` / `rolling_max` | Range tracking, support/resistance, sensor bounds |
| `ewma` | Adaptive smoothing, real-time monitoring, alerting |
| `ddm_detect` | Drift on error-rate streams, classifier/service quality monitoring |
| `page_hinkley_detect` | Drift on numeric streams, latency/load/regime shift detection |
| `expanding_mean` | Cumulative performance metrics, running averages |
| `lag_features` | Convert time series to supervised ML problem |
| `lag_features_with_stats` | Richer ML features with built-in trend/volatility |
| `difference` | Remove trends, achieve stationarity |
| `pct_change` | Relative returns, cross-asset comparison |
| `temporal_split` | Honest train/test evaluation for time series |

## Pitfalls

- **Data leakage from random splits** — The most common mistake. Always use
  `temporal_split`. A model that "sees the future" during training will look
  brilliant in validation and fail in production.

- **Look-ahead bias** — Computing features using future data. Rolling statistics
  in ix-signal only look backward (the first `window - 1` values are NaN). Respect
  those NaNs; do not impute them with future-aware values.

- **Stationarity** — Linear models assume constant mean/variance. Apply `difference`
  or `pct_change` before modeling raw price levels. Check if your differenced series
  looks stationary (constant mean, constant spread).

- **Window size selection** — Too small: noisy, overfits to recent fluctuations.
  Too large: lags behind real changes. Start with domain knowledge (e.g., 5-day
  trading week, 24-hour daily cycle) and iterate.

- **EWMA alpha tuning** — High alpha (close to 1.0) tracks every wiggle. Low alpha
  (close to 0.0) barely reacts. For anomaly detection, use high alpha; for trend
  estimation, use low alpha.
