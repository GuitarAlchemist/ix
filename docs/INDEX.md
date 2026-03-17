# ix Learning Path

> From zero to ML practitioner — using real Rust code you can run today.

This index is your roadmap. Follow it top to bottom as a curriculum, or jump to any topic you need. Every doc links to runnable code in [`examples/`](../examples/).

---

## Level 0: Rust from Scratch

Brand new to Rust? Start here before anything else.

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 0 | [Zero to Hero: Rust](foundations/zero-to-hero-rust.md) | Variables, ownership, traits, iterators, error handling — everything you need |

---

## Level 1: Foundations

Start here if you know basic Rust but are new to math or ML. These five docs give you everything you need before touching any algorithm.

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 1 | [Rust for ML](foundations/rust-for-ml.md) | ndarray, iterators, traits — the Rust patterns every ML crate uses |
| 2 | [Vectors & Matrices](foundations/vectors-and-matrices.md) | What vectors and matrices actually are, and why ML is mostly matrix math |
| 3 | [Probability & Statistics](foundations/probability-and-statistics.md) | Mean, variance, distributions, Bayes' theorem — the language of uncertainty |
| 4 | [Calculus Intuition](foundations/calculus-intuition.md) | Derivatives = slope, gradients = direction of steepest change |
| 5 | [Distance & Similarity](foundations/distance-and-similarity.md) | Euclidean, cosine, Manhattan — measuring how "close" things are |

---

## Level 2: Core Algorithms

With the foundations in place, learn the workhorse algorithms.

### Optimization — Finding the Best Answer

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 6 | [Gradient Descent](optimization/gradient-descent.md) | SGD, Momentum, Adam — how models learn from data |
| 7 | [Simulated Annealing](optimization/simulated-annealing.md) | Escaping local optima by "cooling down" random exploration |
| 8 | [Particle Swarm](optimization/particle-swarm.md) | Swarm intelligence for hard optimization problems |

### Supervised Learning — Prediction from Labeled Data

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 9 | [Linear Regression](supervised-learning/linear-regression.md) | Fitting a line through data — house price prediction |
| 10 | [Logistic Regression](supervised-learning/logistic-regression.md) | Binary classification — spam vs. not spam |
| 11 | [Decision Trees](supervised-learning/decision-trees.md) | If-then rules learned from data — loan approval |
| 12 | [Random Forest](supervised-learning/random-forest.md) | Many trees vote together — fraud detection |
| 13 | [Gradient Boosting](supervised-learning/gradient-boosting.md) | Sequential error correction — the tabular data champion |
| 14 | [KNN](supervised-learning/knn.md) | Classify by nearest neighbors — recommendation systems |
| 15 | [Naive Bayes](supervised-learning/naive-bayes.md) | Fast probabilistic classification — sentiment analysis |
| 16 | [SVM](supervised-learning/svm.md) | Maximum margin classification — image boundaries |
| 17 | [Evaluation Metrics](supervised-learning/evaluation-metrics.md) | Confusion matrix, precision, recall, F1, ROC/AUC, log loss |
| 18 | [Cross-Validation](supervised-learning/cross-validation.md) | K-Fold, Stratified K-Fold — reliable model evaluation |
| 19 | [Resampling & SMOTE](supervised-learning/resampling-smote.md) | Handling imbalanced data — synthetic oversampling, undersampling |

### Unsupervised Learning — Discovering Structure

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 17 | [K-Means](unsupervised-learning/kmeans.md) | Customer segmentation via clustering |
| 18 | [DBSCAN](unsupervised-learning/dbscan.md) | Density-based clustering — anomaly detection in GPS data |
| 19 | [PCA](unsupervised-learning/pca.md) | Dimensionality reduction — seeing high-dimensional data |

---

## Level 3: Advanced Topics

These require comfort with the Level 2 material.

### Neural Networks

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 20 | [Perceptron to MLP](neural-networks/perceptron-to-mlp.md) | From a single neuron to deep networks — digit recognition |
| 21 | [Backpropagation](neural-networks/backpropagation.md) | How networks learn — the chain rule made intuitive |
| 22 | [Loss Functions](neural-networks/loss-functions.md) | MSE vs. cross-entropy — choosing the right objective |

### Reinforcement Learning

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 23 | [Multi-Armed Bandits](reinforcement-learning/multi-armed-bandits.md) | Explore vs. exploit — A/B testing, ad placement |
| 24 | [Q-Learning](reinforcement-learning/q-learning.md) | Learning optimal actions — game AI, navigation |
| 25 | [Exploration vs. Exploitation](reinforcement-learning/exploration-vs-exploitation.md) | The fundamental trade-off in sequential decisions |

### Evolutionary Algorithms

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 26 | [Genetic Algorithms](evolutionary/genetic-algorithms.md) | Evolution-inspired optimization — scheduling, circuit design |
| 27 | [Differential Evolution](evolutionary/differential-evolution.md) | Parameter calibration with mutation vectors |

### Sequence Models

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 28 | [Markov Chains](sequence-models/markov-chains.md) | State transitions — text generation, weather modeling |
| 29 | [Hidden Markov Models](sequence-models/hidden-markov-models.md) | Inferring hidden states — speech recognition, gene finding |
| 30 | [Viterbi Algorithm](sequence-models/viterbi-algorithm.md) | Finding the most likely sequence — GPS path correction |

### Search & Graphs

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 31 | [A* Search](search-and-graphs/astar-search.md) | Optimal pathfinding — game maps, route planning |
| 32 | [Monte Carlo Tree Search](search-and-graphs/mcts.md) | Exploring game trees — Go, Chess AI |
| 33 | [Minimax & Alpha-Beta](search-and-graphs/minimax-alpha-beta.md) | Adversarial game strategy — tic-tac-toe, checkers |
| 34 | [Q* Learned Heuristics](search-and-graphs/qstar-learned-heuristics.md) | Adaptive search with learned cost estimates |

### Game Theory

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 35 | [Nash Equilibria](game-theory/nash-equilibria.md) | Strategic balance — pricing, network routing |
| 36 | [Auction Mechanisms](game-theory/auction-mechanisms.md) | First-price, Vickrey, English — ad auctions, spectrum allocation |
| 37 | [Shapley Value](game-theory/shapley-value.md) | Fair contribution measurement — feature importance, cost allocation |
| 38 | [Evolutionary Dynamics](game-theory/evolutionary-dynamics.md) | Replicator dynamics — population modeling, ecosystem stability |

### Signal Processing

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 39 | [FFT Intuition](signal-processing/fft-intuition.md) | Decomposing signals into frequencies — audio analysis |
| 40 | [Wavelets](signal-processing/wavelets.md) | Time-frequency analysis — image compression, seismic data |
| 41 | [Kalman Filter](signal-processing/kalman-filter.md) | Optimal state estimation — GPS smoothing, drone tracking |
| 42 | [Digital Filters](signal-processing/digital-filters.md) | FIR/IIR filters — noise removal, EQ design |

### Probabilistic Data Structures

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 43 | [Bloom Filters](probabilistic-structures/bloom-filters.md) | Space-efficient set membership — URL blocklists, caching |
| 44 | [Count-Min Sketch](probabilistic-structures/count-min-sketch.md) | Frequency estimation — network traffic heavy hitters |
| 45 | [HyperLogLog](probabilistic-structures/hyperloglog.md) | Cardinality estimation — counting unique visitors at scale |
| 46 | [Cuckoo Filters](probabilistic-structures/cuckoo-filters.md) | Deletable set membership — dynamic blocklists |

---

## Level 4: Specialist Topics

### Chaos Theory

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 47 | [Lyapunov Exponents](chaos-theory/lyapunov-exponents.md) | Measuring chaos — financial market stability |
| 48 | [Strange Attractors](chaos-theory/strange-attractors.md) | Order in chaos — weather modeling, turbulence |
| 49 | [Fractal Dimensions](chaos-theory/fractal-dimensions.md) | Measuring complexity — coastlines, texture analysis |
| 50 | [Chaos Control](chaos-theory/chaos-control.md) | Taming chaos — cardiac rhythm stabilization |

### Adversarial ML

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 51 | [FGSM & PGD](adversarial-ml/fgsm-and-pgd.md) | Attacking models — testing self-driving car robustness |
| 52 | [Adversarial Defenses](adversarial-ml/adversarial-defenses.md) | Hardening classifiers — medical imaging safety |
| 53 | [Data Poisoning](adversarial-ml/data-poisoning.md) | Detecting tampered training data |
| 54 | [Differential Privacy](adversarial-ml/differential-privacy.md) | Privacy-preserving analytics |

### GPU Computing

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 55 | [Intro to GPU Compute](gpu-computing/intro-to-gpu-compute.md) | Why GPUs for ML — WGPU basics |
| 56 | [Similarity Search](gpu-computing/similarity-search.md) | Real-time recommendation via GPU-accelerated cosine similarity |
| 57 | [Matrix Multiply on GPU](gpu-computing/matrix-multiply-gpu.md) | Batch inference acceleration with WGSL shaders |

### Pipelines

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 58 | [DAG Execution](pipelines/dag-execution.md) | Orchestrating ML workflows — ETL, training pipelines |
| 59 | [Caching & Memoization](pipelines/caching-and-memoization.md) | Incremental recomputation — avoiding redundant work |

---

## Cross-Cutting Use Cases

These guides combine multiple algorithms to solve real problems.

| Use Case | Algorithms Combined |
|----------|-------------------|
| [Credit Card Fraud (SMOTE + Boosting)](use-cases/credit-card-fraud.md) | SMOTE + Gradient Boosting + Confusion Matrix + ROC/AUC |
| [Customer Churn Prediction](use-cases/customer-churn.md) | Logistic Regression + Cross-Validation + ROC/AUC |
| [Spam Classifier (TF-IDF)](use-cases/spam-classifier.md) | TF-IDF + Naive Bayes + Cross-Validation + Confusion Matrix |
| [Fraud Detection](use-cases/fraud-detection.md) | Random Forest + PCA + Evaluation Metrics |
| [Recommendation Engine](use-cases/recommendation-engine.md) | KNN + Cosine Similarity + GPU Search |
| [Anomaly Detection](use-cases/anomaly-detection.md) | DBSCAN + Bloom Filter + Kalman |
| [Time Series Analysis](use-cases/time-series-analysis.md) | FFT + Lyapunov + HMM |
| [Autonomous Agent](use-cases/autonomous-agent.md) | Q-Learning + A* + Bandits + MCTS |
| [Robust ML Pipeline](use-cases/robust-ml-pipeline.md) | Pipeline + Adversarial + Differential Privacy |
| [Guitar Alchemist](use-cases/guitar-alchemist.md) | MCTS + GA + Wavelets + Viterbi + GPU Similarity |
| [GIS & Spatial Analysis](use-cases/gis-spatial-analysis.md) | Kalman + DBSCAN + A* + FFT + HMM/Viterbi |

---

## Version française

Tutoriels disponibles en français : [docs/fr/INDEX.md](fr/INDEX.md)

---

## How to Use These Docs

**Following the curriculum?** Start at #1 and work down. Each doc builds on previous ones.

**Looking for a specific topic?** Use the table above or browse the folders directly.

**Want to run code?** Every doc links to runnable examples in [`examples/`](../examples/). Run them with:
```bash
cargo run --example <name>
```

**Want to experiment interactively?** Check the [notebooks/](notebooks/) folder for Jupyter notebooks (requires [Evcxr](https://github.com/evcxr/evcxr)).
