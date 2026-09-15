# Use Case: GIS & Spatial Analysis

> Combining Kalman filters, DBSCAN, A*, FFT, and HMMs for GPS tracking, spatial clustering, route optimization, and terrain analysis.

## The Scenario

You're building a logistics platform that manages a fleet of 500 delivery vehicles. You need to:

1. **Smooth GPS tracks** — raw GPS jumps around; you need clean trajectories (Kalman filter)
2. **Find delivery hotspots** — cluster stop locations to identify warehouses and frequent destinations (DBSCAN)
3. **Optimize routes** — find shortest paths through the road network (A*)
4. **Analyze terrain** — detect periodic patterns in elevation data for road quality assessment (FFT)
5. **Snap GPS to roads** — match noisy GPS points to the most likely road segments (HMM/Viterbi)
6. **Detect anomalies** — identify unusual vehicle behavior patterns (Lyapunov exponents + Bloom filters)

## Step 1: GPS Track Smoothing with Kalman Filter

Raw GPS readings scatter ±10 meters due to atmospheric interference, multipath reflection, and sensor noise. The Kalman filter fuses noisy position readings with a motion model to produce smooth, accurate tracks.

```rust
use ix_signal::kalman::KalmanFilter;
use ndarray::array;

// Constant-velocity model: state = [x, vx, y, vy]
// GPS measures position only: observation = [x, y]
let dt = 1.0; // 1-second GPS updates
let mut kf = KalmanFilter::new(4, 2); // 4 state dims, 2 observation dims

// State transition: x_new = x + vx*dt, vx_new = vx (constant velocity)
kf.transition = array![
    [1.0, dt,  0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, dt ],
    [0.0, 0.0, 0.0, 1.0],
];

// Observation: we see x and y, not velocities
kf.observation = array![
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

// Noise: GPS has ~10m accuracy, vehicle acceleration varies
kf.measurement_noise = array![[100.0, 0.0], [0.0, 100.0]]; // 10m std dev
kf.process_noise = array![
    [0.25, 0.5, 0.0, 0.0],
    [0.5,  1.0, 0.0, 0.0],
    [0.0,  0.0, 0.25, 0.5],
    [0.0,  0.0, 0.5,  1.0],
];

// Process noisy GPS readings
let gps_readings = vec![
    array![40.7128, -74.0060],  // NYC coordinates (simplified)
    array![40.7131, -74.0055],
    array![40.7150, -74.0040],  // Jump — probably noise
    array![40.7135, -74.0048],
    // ... hundreds more readings
];

let smoothed: Vec<_> = gps_readings.iter()
    .map(|reading| kf.step(reading, None))
    .collect();

// smoothed[i] = [x, vx, y, vy] — position AND estimated velocity
for (i, state) in smoothed.iter().enumerate() {
    println!("t={}: pos=({:.4}, {:.4}), speed=({:.2}, {:.2})",
        i, state[0], state[2], state[1], state[3]);
}
```

> See [Signal: Kalman Filter](../signal-processing/kalman-filter.md) for the full Kalman filter doc.

## Step 2: Spatial Clustering with DBSCAN

Find delivery hotspots — areas where vehicles frequently stop. DBSCAN is perfect because hotspots have irregular shapes (they follow buildings, loading docks, intersections) and you need to identify noise (one-off stops).

```rust
use ix_unsupervised::dbscan::DBSCAN;
use ix_unsupervised::traits::Clusterer;
use ndarray::array;

// Stop locations from fleet: [latitude, longitude]
let stops = array![
    // Warehouse cluster (Brooklyn)
    [40.6782, -73.9442], [40.6785, -73.9440], [40.6780, -73.9445],
    [40.6783, -73.9441], [40.6781, -73.9443],
    // Office district cluster (Midtown)
    [40.7549, -73.9840], [40.7551, -73.9838], [40.7548, -73.9842],
    [40.7550, -73.9839],
    // Noise: random one-off stops
    [40.7000, -73.9500], [40.8000, -73.9000],
];

// eps ≈ 0.001 degrees ≈ 100m at NYC latitude
let mut dbscan = DBSCAN::new(0.001, 3); // min 3 stops to form a cluster
let labels = dbscan.fit_predict(&stops);

// Label 0 = noise, 1+ = cluster ID
let n_clusters = *labels.iter().max().unwrap_or(&0);
let n_noise = labels.iter().filter(|&&l| l == 0).count();
println!("{} hotspots found, {} noise points", n_clusters, n_noise);

// Compute cluster centroids for each hotspot
for cluster_id in 1..=n_clusters {
    let points: Vec<_> = stops.outer_iter()
        .zip(labels.iter())
        .filter(|(_, &l)| l == cluster_id)
        .map(|(row, _)| row.to_owned())
        .collect();
    let centroid_lat = points.iter().map(|p| p[0]).sum::<f64>() / points.len() as f64;
    let centroid_lon = points.iter().map(|p| p[1]).sum::<f64>() / points.len() as f64;
    println!("Hotspot {}: ({:.4}, {:.4}) — {} stops",
        cluster_id, centroid_lat, centroid_lon, points.len());
}
```

> See [Unsupervised: DBSCAN](../unsupervised-learning/dbscan.md) for the full DBSCAN doc.

## Step 3: Route Optimization with A*

Find the shortest path between delivery stops on a road network. A* uses a heuristic (straight-line distance) to focus the search toward the goal.

```rust
use ix_search::astar::{SearchState, astar, SearchResult};

#[derive(Clone, Hash, Eq, PartialEq)]
struct RoadNode {
    id: usize,
    lat: i64,  // Fixed-point (lat * 10000) for Hash/Eq
    lon: i64,
}

impl SearchState for RoadNode {
    type Action = usize; // Edge ID

    fn successors(&self) -> Vec<(usize, Self, f64)> {
        // Return connected roads with travel time as cost
        get_road_neighbors(self.id)
            .iter()
            .map(|(edge_id, neighbor, distance_km)| {
                let travel_time = distance_km / 50.0; // Assume 50 km/h average
                (*edge_id, neighbor.clone(), travel_time)
            })
            .collect()
    }

    fn is_goal(&self) -> bool {
        self.id == DESTINATION_ID
    }
}

// Heuristic: straight-line distance / max_speed
let heuristic = |node: &RoadNode| -> f64 {
    let dx = (node.lat - dest_lat) as f64 / 10000.0;
    let dy = (node.lon - dest_lon) as f64 / 10000.0;
    let straight_line_km = (dx * dx + dy * dy).sqrt() * 111.0; // ~111km per degree
    straight_line_km / 80.0 // Optimistic: max 80 km/h
};

let start = RoadNode { id: 0, lat: 407128, lon: -740060 };
if let Some(result) = astar(&start, &heuristic) {
    println!("Route found: {} segments, {:.1} min", result.path.len(), result.cost * 60.0);
    println!("Nodes explored: {}", result.nodes_expanded);
}
```

> See [Search: A*](../search-and-graphs/astar-search.md) for the full A* doc.

## Step 4: Terrain Analysis with FFT

Analyze road elevation profiles to detect periodic patterns — potholes at regular intervals, speed bumps, or road surface quality.

```rust
use ix_signal::fft::{rfft, magnitude_spectrum, frequency_bins};

// Elevation readings every 1 meter along a road (sampled via lidar/GPS)
let elevation: Vec<f64> = load_elevation_profile("route_42.csv");
let sample_rate = 1.0; // 1 sample per meter

let spectrum = rfft(&elevation);
let magnitudes = magnitude_spectrum(&spectrum);
let freqs = frequency_bins(spectrum.len() * 2, sample_rate);

// Find dominant spatial frequencies
let mut peaks: Vec<(f64, f64)> = freqs.iter()
    .zip(magnitudes.iter())
    .filter(|(&f, _)| f > 0.001) // Skip DC component
    .map(|(&f, &m)| (f, m))
    .collect();
peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

println!("Top spatial frequencies:");
for (freq, mag) in peaks.iter().take(5) {
    let wavelength = 1.0 / freq;
    println!("  Wavelength: {:.1}m, Magnitude: {:.2}", wavelength, mag);
}
// Wavelength ~5m → speed bumps; ~0.5m → rough surface; ~50m → gentle hills
```

> See [Signal: FFT](../signal-processing/fft-intuition.md) for the full FFT doc.

## Step 5: Map Matching with HMM/Viterbi

Snap noisy GPS points to the most likely road segments. Hidden states = road segments, observations = GPS zones, transitions = road connectivity.

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::{Array1, Array2};

// Build HMM from road network
// States: road segments, Observations: discretized GPS regions
let n_segments = 20;
let n_gps_zones = 10;

let initial = Array1::from_vec(vec![1.0 / n_segments as f64; n_segments]);
let transition = build_road_transition_matrix(n_segments); // From road connectivity
let emission = build_gps_emission_matrix(n_segments, n_gps_zones); // GPS accuracy model

let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();

// Noisy GPS readings → discretized zone IDs
let gps_zones = vec![3, 3, 4, 5, 5, 6, 7, 7, 8, 8];

let (road_segments, log_prob) = hmm.viterbi(&gps_zones);
println!("GPS zones:     {:?}", gps_zones);
println!("Road segments: {:?}", road_segments);
println!("Confidence: {:.2}", log_prob);
```

> See [Sequence: Viterbi](../sequence-models/viterbi-algorithm.md) for the full Viterbi doc.

## Step 6: Anomaly Detection with Bloom Filters

Track which route patterns are "normal" using a Bloom filter. When a vehicle's route hash isn't in the filter, flag it for review.

```rust
use ix_probabilistic::bloom::BloomFilter;

// Train: insert all normal route patterns
let mut normal_routes = BloomFilter::new(10_000, 0.01); // 1% false positive rate

for route in historical_normal_routes {
    let route_hash = format!("{:?}", route.segment_sequence);
    normal_routes.insert(&route_hash);
}

// Monitor: check if current route is in the normal set
let current_route_hash = format!("{:?}", current_vehicle.segment_sequence);
if !normal_routes.contains(&current_route_hash) {
    println!("ALERT: Vehicle {} on unusual route!", current_vehicle.id);
}
```

> See [Probabilistic: Bloom Filters](../probabilistic-structures/bloom-filters.md) for the full doc.

---

## Life & Safety: PSAP / First Responder Applications

The same spatial algorithms apply to public safety — where latency and accuracy are life-or-death.

### Scenario: Next-Generation 911 (NG911) Dispatch

A Public Safety Answering Point (PSAP) receives thousands of emergency calls daily. Each call carries location data (GPS from cell phones, ALI from landlines), but the data is noisy, incomplete, and sometimes wrong. Dispatchers need to:

1. **Locate the caller accurately** — cell GPS can be off by 50-300m indoors
2. **Find the nearest available unit** — ambulance, fire, police
3. **Route the unit optimally** — shortest *time*, not shortest distance
4. **Predict incident density** — pre-position units in high-risk zones
5. **Detect call patterns** — distinguish prank calls, detect mass casualty events

### Emergency Caller Location (Kalman + Map Matching)

Cell phones report GPS coordinates, but indoors or in urban canyons, accuracy degrades to 100m+. Fuse multiple location sources (GPS, cell tower triangulation, Wi-Fi) with a Kalman filter, then snap to the nearest building with HMM/Viterbi.

```rust
use ix_signal::kalman::KalmanFilter;
use ndarray::array;

// Fuse GPS + cell tower readings for a 911 caller
// State: [lat, lon, accuracy_estimate]
let mut kf = KalmanFilter::new(4, 2);

// Cell phone reports GPS every ~1 second during call
let cell_readings = vec![
    array![40.7589, -73.9851],  // Initial fix (outdoors, good)
    array![40.7585, -73.9860],  // Caller moves indoors, accuracy degrades
    array![40.7600, -73.9830],  // Big jump — multipath reflection off building
    array![40.7587, -73.9855],  // Returns closer to true position
];

// High measurement noise: indoor GPS is unreliable
kf.measurement_noise = array![[400.0, 0.0], [0.0, 400.0]]; // ~20m std dev
kf.transition = array![
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
kf.observation = array![
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

// Low process noise: caller is likely stationary or walking
kf.process_noise = array![
    [0.01, 0.0, 0.0, 0.0],
    [0.0, 0.01, 0.0, 0.0],
    [0.0, 0.0, 0.01, 0.0],
    [0.0, 0.0, 0.0, 0.01],
];

let best_location = cell_readings.iter()
    .map(|r| kf.step(r, None))
    .last()
    .unwrap();

println!("Best estimate: ({:.4}, {:.4})", best_location[0], best_location[2]);
// Fused location is more accurate than any single reading
```

### Nearest Unit Dispatch (A* with Time-Based Cost)

Find the closest available ambulance — not by straight-line distance, but by *estimated travel time* considering road speeds, traffic, and one-way streets.

```rust
use ix_search::astar::{SearchState, astar};

#[derive(Clone, Hash, Eq, PartialEq)]
struct Intersection {
    id: usize,
    lat_fp: i64,  // fixed-point for Hash
    lon_fp: i64,
}

impl SearchState for Intersection {
    type Action = usize;

    fn successors(&self) -> Vec<(usize, Self, f64)> {
        // Cost = travel time in minutes, factoring in:
        // - Road speed limits
        // - Current traffic conditions
        // - Emergency vehicle preemption (traffic light override)
        get_road_segments(self.id)
            .iter()
            .map(|(edge, neighbor, distance_km, speed_limit)| {
                // Emergency vehicles travel ~75% of speed limit in urban areas
                let effective_speed = speed_limit * 0.75;
                let travel_minutes = (distance_km / effective_speed) * 60.0;
                (*edge, neighbor.clone(), travel_minutes)
            })
            .collect()
    }

    fn is_goal(&self) -> bool {
        self.id == incident_location_id()
    }
}

// For each available unit, compute time-to-scene
// Dispatch the one with shortest ETA
let heuristic = |node: &Intersection| -> f64 {
    let dx = (node.lat_fp - incident_lat) as f64 / 10000.0;
    let dy = (node.lon_fp - incident_lon) as f64 / 10000.0;
    let km = (dx * dx + dy * dy).sqrt() * 111.0;
    km / 120.0 * 60.0 // Optimistic: 120 km/h with lights and sirens, in minutes
};
```

### Incident Hotspot Prediction (DBSCAN + Gradient Boosting)

Pre-position ambulances by predicting where incidents will cluster. Use DBSCAN on historical incidents to find hotspots, then train a classifier to predict which zones will be active at a given time.

```rust
use ix_unsupervised::dbscan::DBSCAN;
use ix_unsupervised::traits::Clusterer;
use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
use ix_ensemble::traits::EnsembleClassifier;
use ndarray::{array, Array2};

// Historical 911 incidents: [lat, lon]
let incidents = array![
    // Downtown bar district (Friday/Saturday night cluster)
    [40.7580, -73.9855], [40.7582, -73.9852], [40.7578, -73.9858],
    [40.7581, -73.9854], [40.7579, -73.9856],
    // Highway interchange (rush hour accident cluster)
    [40.7230, -73.9950], [40.7232, -73.9948], [40.7228, -73.9952],
    [40.7231, -73.9949],
    // Random scattered incidents (noise)
    [40.7100, -73.9700], [40.7400, -73.9600],
];

let mut dbscan = DBSCAN::new(0.0005, 3); // ~50m radius, min 3 incidents
let labels = dbscan.fit_predict(&incidents);

let n_hotspots = *labels.iter().max().unwrap_or(&0);
println!("{} hotspots identified", n_hotspots);

// Now predict WHEN hotspots are active
// Features: [hour, day_of_week, is_holiday, temperature, rain]
let x = Array2::from_shape_vec((8, 5), vec![
    22.0, 5.0, 0.0, 72.0, 0.0,  // Fri 10pm, warm, dry → bar district active
    23.0, 6.0, 0.0, 70.0, 0.0,  // Sat 11pm → bar district active
     8.0, 1.0, 0.0, 45.0, 1.0,  // Mon 8am, cold rain → highway active
    17.0, 3.0, 0.0, 50.0, 0.0,  // Wed 5pm → highway active
    14.0, 2.0, 0.0, 65.0, 0.0,  // Tue 2pm → quiet
    10.0, 7.0, 0.0, 60.0, 0.0,  // Sun 10am → quiet
     3.0, 4.0, 0.0, 55.0, 0.0,  // Thu 3am → quiet
    12.0, 1.0, 1.0, 40.0, 0.0,  // Holiday noon → quiet
]).unwrap();
let y = array![1, 1, 1, 1, 0, 0, 0, 0]; // 1 = high-risk period

let mut gbc = GradientBoostedClassifier::new(50, 0.1);
gbc.fit(&x, &y);

// Predict risk for upcoming shift
let upcoming = array![[21.0, 5.0, 0.0, 68.0, 0.0]]; // Friday 9pm
let risk = gbc.predict_proba(&upcoming);
println!("Incident risk: {:.0}%", risk[[0, 1]] * 100.0);
// → High risk → pre-position an ambulance near the bar district
```

### Mass Casualty Event Detection (Anomaly via Call Clustering)

When multiple 911 calls arrive from the same area within minutes, detect the spatial-temporal cluster as a potential mass casualty incident (MCI) requiring multi-unit response.

```rust
use ix_unsupervised::dbscan::DBSCAN;
use ix_unsupervised::traits::Clusterer;
use ndarray::Array2;

// Sliding window: 911 calls in the last 5 minutes
// Features: [lat, lon, minutes_ago]
let recent_calls = Array2::from_shape_vec((7, 3), vec![
    // Cluster: 4 calls near Times Square in 2 minutes
    40.7580, -73.9855, 0.5,
    40.7582, -73.9852, 1.0,
    40.7579, -73.9856, 1.5,
    40.7581, -73.9854, 2.0,
    // Unrelated calls elsewhere
    40.7100, -73.9700, 0.2,
    40.7400, -73.9600, 3.0,
    40.8000, -73.9300, 4.5,
]).unwrap();

// Small radius in space+time: eps ~0.001° (~100m) with time scaled
let mut dbscan = DBSCAN::new(0.002, 3); // min 3 calls to flag MCI
let labels = dbscan.fit_predict(&recent_calls);

let n_clusters = *labels.iter().max().unwrap_or(&0);
if n_clusters > 0 {
    println!("⚠ ALERT: {} potential MCI cluster(s) detected!", n_clusters);
    println!("  Triggering multi-unit dispatch protocol");
    // Auto-dispatch: 2 ambulances, 1 fire, 1 police supervisor
}
```

### Response Time Analysis (Cross-Validation + Metrics)

Evaluate dispatch model performance: are we meeting the NFPA 1710 standard (first unit on scene within 4 minutes for 90% of calls)?

```rust
use ndarray::{array, Array2};
use ix_supervised::validation::cross_val_score;
use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::metrics::{ConfusionMatrix, recall};

// Historical responses: features that predict whether we meet the 4-min target
// [distance_km, time_of_day, units_available, road_type (0=local,1=arterial,2=highway)]
let x = Array2::from_shape_vec((12, 4), vec![
    1.0, 14.0, 5.0, 1.0,   // short, daytime, units available → met
    0.5,  2.0, 3.0, 0.0,   // very close, night → met
    2.0, 10.0, 4.0, 1.0,   // moderate distance → met
    1.5, 16.0, 2.0, 0.0,   // local road, few units → met
    0.8,  8.0, 6.0, 2.0,   // highway access → met
    3.0, 12.0, 3.0, 1.0,   // farther → met
    5.0, 17.0, 1.0, 0.0,   // far, rush hour, 1 unit → missed
    4.0,  8.0, 2.0, 0.0,   // far, local roads → missed
    6.0, 12.0, 2.0, 1.0,   // very far → missed
    3.5, 17.0, 1.0, 0.0,   // rush hour, 1 unit → missed
    4.5,  7.0, 1.0, 0.0,   // far, morning rush → missed
    5.5, 18.0, 1.0, 1.0,   // far, evening → missed
]).unwrap();
let y = array![1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]; // 1=met 4-min, 0=missed

// Cross-validate: can we predict which calls will miss the target?
let scores = cross_val_score(&x, &y, || DecisionTree::new(3), 4, 42);
let mean = scores.iter().sum::<f64>() / scores.len() as f64;
println!("CV accuracy: {:.1}%", mean * 100.0);

// Train final model and evaluate
let mut tree = DecisionTree::new(3);
use ix_supervised::traits::Classifier;
tree.fit(&x, &y);
let preds = tree.predict(&x);

let cm = ConfusionMatrix::from_labels(&y, &preds, 2);
println!("{}", cm.display());
// Key metric: recall for class 0 (missed responses)
// → identifies which factors cause missed targets
println!("Recall (missed): {:.2} — % of misses we can predict",
    recall(&y, &preds, 0));
```

### PSAP Use Cases Summary

| Use Case | Algorithms | Standard |
|----------|-----------|----------|
| Caller location fusion | Kalman filter + HMM/Viterbi | FCC E911 Z-axis accuracy |
| Nearest unit dispatch | A* with time-cost | NFPA 1710 (4 min first unit) |
| Incident hotspot prediction | DBSCAN + Gradient Boosting | Proactive deployment |
| Mass casualty detection | DBSCAN (space-time) | NIMS/ICS MCI protocols |
| Response time analysis | Decision Tree + Cross-validation | NFPA 1710 compliance |
| Prank/repeat caller filtering | Bloom filter (seen callers) | PSAP workload reduction |
| Ambulance pre-positioning | K-Means zone partitioning | Covering location models |

---

## Algorithms Used

| Algorithm | Doc | Role in GIS |
|-----------|-----|-------------|
| Kalman Filter | [Signal: Kalman](../signal-processing/kalman-filter.md) | GPS track smoothing |
| DBSCAN | [Unsupervised: DBSCAN](../unsupervised-learning/dbscan.md) | Spatial clustering / hotspots |
| A* Search | [Search: A*](../search-and-graphs/astar-search.md) | Route optimization |
| FFT | [Signal: FFT](../signal-processing/fft-intuition.md) | Terrain frequency analysis |
| HMM/Viterbi | [Sequence: Viterbi](../sequence-models/viterbi-algorithm.md) | GPS-to-road map matching |
| Bloom Filter | [Probabilistic: Bloom](../probabilistic-structures/bloom-filters.md) | Route anomaly detection |
| K-Means | [Unsupervised: K-Means](../unsupervised-learning/kmeans.md) | Zone partitioning |
| Markov Chains | [Sequence: Markov](../sequence-models/markov-chains.md) | Traffic flow modeling |
| Lyapunov Exponents | [Chaos: Lyapunov](../chaos-theory/lyapunov-exponents.md) | Traffic chaos detection |
| Gradient Boosting | [Supervised: Gradient Boosting](../supervised-learning/gradient-boosting.md) | Incident risk prediction |
| Decision Tree + CV | [Supervised: Cross-Validation](../supervised-learning/cross-validation.md) | Response time compliance |
| Confusion Matrix | [Supervised: Metrics](../supervised-learning/evaluation-metrics.md) | Dispatch model evaluation |
