# Cas pratique : SIG et analyse spatiale

> Combinaison de filtres de Kalman, DBSCAN, A*, FFT et HMM pour le suivi GPS, le regroupement spatial, l'optimisation d'itinéraires et l'analyse de terrain.

## Le scénario

Vous développez une plateforme logistique qui gère une flotte de 500 véhicules de livraison. Vous devez :

1. **Lisser les traces GPS** — le GPS brut « saute » ; il faut des trajectoires propres (filtre de Kalman)
2. **Trouver les points chauds de livraison** — regrouper les lieux d'arrêt pour identifier les entrepôts et les destinations fréquentes (DBSCAN)
3. **Optimiser les itinéraires** — trouver les chemins les plus courts dans le réseau routier (A*)
4. **Analyser le terrain** — détecter les motifs périodiques dans les données d'altitude pour évaluer la qualité de la chaussée (FFT)
5. **Accrocher le GPS aux routes** — faire correspondre les points GPS bruités aux segments de route les plus probables (HMM/Viterbi)
6. **Détecter les anomalies** — identifier les comportements inhabituels des véhicules (exposants de Lyapunov + filtres de Bloom)

## Étape 1 : Lissage des traces GPS avec le filtre de Kalman

Les mesures GPS brutes sont dispersées à ±10 mètres en raison des interférences atmosphériques, des réflexions multitrajets et du bruit des capteurs. Le filtre de Kalman fusionne les mesures de position bruitées avec un modèle de mouvement pour produire des trajectoires lisses et précises.

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

> Voir [Signal : Filtre de Kalman](../signal-processing/kalman-filter.md) pour la documentation complète du filtre de Kalman.

## Étape 2 : Regroupement spatial avec DBSCAN

Trouver les points chauds de livraison — les zones où les véhicules s'arrêtent fréquemment. DBSCAN est idéal car les points chauds ont des formes irrégulières (ils suivent les bâtiments, les quais de chargement, les intersections) et il faut identifier le bruit (arrêts ponctuels).

```rust
use ix_unsupervised::{DBSCAN, Clusterer};
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

> Voir [Apprentissage non supervisé : DBSCAN](../unsupervised-learning/dbscan.md) pour la documentation complète de DBSCAN.

## Étape 3 : Optimisation d'itinéraire avec A*

Trouver le chemin le plus court entre les arrêts de livraison sur un réseau routier. A* utilise une heuristique (distance à vol d'oiseau) pour concentrer la recherche en direction de l'objectif.

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

> Voir [Recherche : A*](../search-and-graphs/astar-search.md) pour la documentation complète de A*.

## Étape 4 : Analyse de terrain avec FFT

Analyser les profils d'altitude des routes pour détecter des motifs périodiques — nids-de-poule à intervalles réguliers, dos-d'âne ou qualité du revêtement routier.

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

> Voir [Signal : FFT](../signal-processing/fft-intuition.md) pour la documentation complète de la FFT.

## Étape 5 : Calage cartographique avec HMM/Viterbi

Accrocher les points GPS bruités aux segments de route les plus probables. Les états cachés sont les segments de route, les observations sont les zones GPS, et les transitions représentent la connectivité du réseau routier.

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

> Voir [Séquences : Viterbi](../sequence-models/viterbi-algorithm.md) pour la documentation complète de l'algorithme de Viterbi.

## Étape 6 : Détection d'anomalies avec les filtres de Bloom

Suivre les schémas d'itinéraires « normaux » à l'aide d'un filtre de Bloom. Lorsque le hachage de l'itinéraire d'un véhicule n'est pas dans le filtre, le signaler pour examen.

```rust
use ix_probabilistic::BloomFilter;

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

> Voir [Structures probabilistes : Filtres de Bloom](../probabilistic-structures/bloom-filters.md) pour la documentation complète.

---

## Sécurité des personnes : Applications PSAP / Premiers intervenants

Les mêmes algorithmes spatiaux s'appliquent à la sécurité publique — où la latence et la précision sont une question de vie ou de mort.

### Scénario : Système d'appels d'urgence de nouvelle génération (NG911)

Un PSAP (Public Safety Answering Point — centre de réception des appels d'urgence) reçoit des milliers d'appels d'urgence chaque jour. Chaque appel contient des données de localisation (GPS des téléphones portables, ALI pour les lignes fixes), mais ces données sont bruitées, incomplètes, et parfois erronées. Les opérateurs doivent :

1. **Localiser l'appelant avec précision** — le GPS des téléphones peut être décalé de 50 à 300 m en intérieur
2. **Trouver l'unité disponible la plus proche** — ambulance, pompiers, police
3. **Acheminer l'unité de manière optimale** — en temps de trajet le plus court, et non en distance la plus courte
4. **Prédire la densité des incidents** — prépositonner les unités dans les zones à haut risque
5. **Détecter les schémas d'appels** — distinguer les canulars, détecter les incidents de masse

### Localisation de l'appelant d'urgence (Kalman + Calage cartographique)

Les téléphones portables transmettent des coordonnées GPS, mais en intérieur ou dans les « canyons urbains », la précision se dégrade à plus de 100 m. On fusionne plusieurs sources de localisation (GPS, triangulation par antennes-relais, Wi-Fi) avec un filtre de Kalman, puis on accroche le résultat au bâtiment le plus proche avec HMM/Viterbi.

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

### Envoi de l'unité la plus proche (A* avec coût temporel)

Trouver l'ambulance disponible la plus proche — non pas en distance à vol d'oiseau, mais en *temps de trajet estimé* tenant compte de la vitesse des routes, du trafic et des sens uniques.

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

### Prédiction des points chauds d'incidents (DBSCAN + Gradient Boosting)

Prépositonner les ambulances en prédisant où les incidents vont se concentrer. Utiliser DBSCAN sur les incidents historiques pour trouver les points chauds, puis entraîner un classifieur pour prédire quelles zones seront actives à un moment donné.

```rust
use ix_unsupervised::{DBSCAN, Clusterer};
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

### Détection d'incidents de masse (Anomalie par regroupement d'appels)

Lorsque plusieurs appels d'urgence arrivent de la même zone en quelques minutes, détecter le regroupement spatio-temporel comme un potentiel incident de masse nécessitant une réponse multi-unités.

```rust
use ix_unsupervised::{DBSCAN, Clusterer};
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

### Analyse des temps de réponse (Validation croisée + Métriques)

Évaluer la performance du modèle de régulation : respectons-nous la norme NFPA 1710 (première unité sur les lieux en 4 minutes pour 90 % des appels) ?

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

### Résumé des cas d'usage PSAP

| Cas d'usage | Algorithmes | Norme / Référence |
|-------------|------------|-------------------|
| Fusion de la localisation de l'appelant | Filtre de Kalman + HMM/Viterbi | FCC E911 — précision sur l'axe Z |
| Envoi de l'unité la plus proche | A* avec coût temporel | NFPA 1710 (première unité en 4 min) |
| Prédiction des points chauds d'incidents | DBSCAN + Gradient Boosting | Déploiement proactif |
| Détection d'incidents de masse | DBSCAN (spatio-temporel) | Protocoles NIMS/ICS pour incidents de masse |
| Analyse des temps de réponse | Arbre de décision + Validation croisée | Conformité NFPA 1710 |
| Filtrage des canulars / appelants récurrents | Filtre de Bloom (appelants déjà vus) | Réduction de la charge du PSAP |
| Prépositionnement des ambulances | Partitionnement par K-Means | Modèles de couverture territoriale |

---

## Algorithmes utilisés

| Algorithme | Documentation | Rôle en SIG |
|------------|--------------|-------------|
| Filtre de Kalman | [Signal : Kalman](../signal-processing/kalman-filter.md) | Lissage des traces GPS |
| DBSCAN | [Apprentissage non supervisé : DBSCAN](../unsupervised-learning/dbscan.md) | Regroupement spatial / points chauds |
| Recherche A* | [Recherche : A*](../search-and-graphs/astar-search.md) | Optimisation d'itinéraires |
| FFT | [Signal : FFT](../signal-processing/fft-intuition.md) | Analyse fréquentielle du terrain |
| HMM/Viterbi | [Séquences : Viterbi](../sequence-models/viterbi-algorithm.md) | Calage cartographique GPS-route |
| Filtre de Bloom | [Structures probabilistes : Bloom](../probabilistic-structures/bloom-filters.md) | Détection d'anomalies d'itinéraires |
| K-Means | [Apprentissage non supervisé : K-Means](../unsupervised-learning/kmeans.md) | Partitionnement de zones |
| Chaînes de Markov | [Séquences : Markov](../sequence-models/markov-chains.md) | Modélisation du flux de trafic |
| Exposants de Lyapunov | [Chaos : Lyapunov](../chaos-theory/lyapunov-exponents.md) | Détection du chaos dans le trafic |
| Gradient Boosting | [Apprentissage supervisé : Gradient Boosting](../supervised-learning/gradient-boosting.md) | Prédiction du risque d'incidents |
| Arbre de décision + VC | [Apprentissage supervisé : Validation croisée](../supervised-learning/cross-validation.md) | Conformité des temps de réponse |
| Matrice de confusion | [Apprentissage supervisé : Métriques](../supervised-learning/evaluation-metrics.md) | Évaluation du modèle de régulation |
