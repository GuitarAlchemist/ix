#!/usr/bin/env python3
"""Offline ROC sweep for the IX coverage gate (embeddings front, D4).

Embeds the 52 pipeline-callable skill docs (documents) and the 184-probe labeled
corpus (queries) with an embedding model, then evaluates candidate OOD signals
against the corpus and finds the operating point that best refuses out-of-domain
requests while holding in-domain recall.

NOT a CI test — needs `transformers`+`torch` and a one-time HuggingFace model
download. Run manually:
    python scripts/embedding_coverage_sweep.py
Compares against the lexical TF-IDF baseline (recall 0.990 / overall TNR 0.163).

Model: intfloat/e5-small-v2 — a self-hostable, asymmetric, query/passage-prefixed
retrieval embedder (33M params, 384-dim), the pattern the deep-research report
(reference_embedding_coverage_gate_research) recommended. Skill docs are embedded
as "passage: <doc>", the NL intent as "query: <intent>". (OpenAI text-embedding-3
was the first choice but the available key has insufficient_quota; open models
merely *match* it per the research, so this is the preferred path anyway.)
"""
import json
import os
import sys

import numpy as np

MODEL = os.environ.get("IX_EMBED_MODEL", "intfloat/e5-small-v2")
CORPUS = "state/thinking-machine/coverage-probes.jsonl"
# 52 pipeline-callable skills as "name: doc" documents (must match the catalog).
CATALOG = """adversarial.fgsm: Fast Gradient Sign Method adversarial perturbation
assumption.belief_at: Reconstruct the belief state at a point in belief-time
assumption.claims: Return the @ai: claims anchored under a file or directory prefix
assumption.drift: Compare current @ai: claims against a committed baseline
assumption.query: Build the unified assumption graph for a workspace
bandit: Multi-armed bandit simulation (epsilon-greedy / UCB1 / Thompson)
bloom_filter: Bloom filter: probabilistic set membership
cache: In-memory cache: set/get/delete/list operations
category: Category theory: verify monad laws; free-forgetful adjunction
chaos.lyapunov: Maximal Lyapunov exponent of the logistic map
code_analyze: Code complexity analysis: cyclomatic, cognitive, Halstead, SLOC, MI
context.walk: Deterministic structural context DAG walker over a Rust workspace
distance: Distance between two vectors (euclidean, cosine, manhattan)
evolution: Evolutionary optimization: genetic algorithm / differential evolution
fft: Fast Fourier Transform of a real-valued signal
fractal: Fractals: Takagi curve, Hilbert/Peano curves, Morton encoding
fuzzy.eval: Evaluate hexavalent fuzzy distribution ops: info/not/and/or
ga_bridge: Convert GA music theory data into ML-ready feature matrices
game.nash: Nash equilibria of a 2-player bimatrix game
governance.belief: Query the Demerzel belief engine with a belief state
governance.check: Check a proposed action against the Demerzel constitution
governance.graph: Scan the Demerzel governance submodule, emit artifact graph
governance.graph.rescan: Poll-based governance graph refresh
governance.persona: Load a Demerzel persona by name
governance.policy: Query Demerzel governance policies
gradient_boosting: Gradient boosted trees classifier (binary + multiclass)
grammar.evolve: Grammar rule competition via replicator dynamics
grammar.search: Grammar-guided MCTS derivation search
grammar.weights: Bayesian (Beta-Binomial) update + softmax of grammar rule weights
graph: Graph algorithms: Dijkstra, PageRank, BFS/DFS, topological sort
hyperloglog: HyperLogLog cardinality estimation
kmeans: Cluster points using K-Means
linear_regression: Fit an ordinary least-squares linear regression model
markov: Markov chain stationary distribution via power iteration
ml_pipeline: End-to-end ML pipeline: load -> preprocess -> train -> evaluate
ml_predict: Predict using a previously-persisted ML model
nn.forward: Neural network forward pass: dense/loss/attention/positional enc
number_theory: Number theory: primes, modular arithmetic, gcd/lcm
optimize: Minimize a benchmark function via SGD/Adam/PSO/simulated annealing
pipeline: DAG pipeline analysis: toposort, parallel levels, critical path
random_forest: Random forest classifier: train and predict
rotation: 3D rotation ops: quaternions, SLERP, Euler conversions
search: Search algorithm catalog (A*, BFS, DFS): descriptions and complexity
sedenion: Sedenion/octonion algebra: multiplication, conjugate, norm
session.flywheel_export: Convert a persisted SessionLog to a GA trace file
stats: Statistics (mean, std, min, max, median) on a list of numbers
supervised: Supervised learning dispatcher: train/predict/metrics/cross-validate
tars_bridge: Prepare ix results for TARS ingestion (traces/patterns/grammar)
topo: Topological data analysis: persistent homology + Betti numbers
trace.ingest: Ingest GA trace files and compute summary statistics
viterbi: HMM Viterbi decoding — most-likely hidden state sequence"""


_TOK = _MODEL = None


def _load():
    global _TOK, _MODEL
    if _MODEL is None:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(MODEL)
        _MODEL = AutoModel.from_pretrained(MODEL)
        _MODEL.eval()
    return _TOK, _MODEL


def embed(texts, prefix):
    """Embed `prefix: text` (E5 asymmetric) via mean-pooling; L2-normalized."""
    import torch

    tok, model = _load()
    inputs = [f"{prefix}: {t}" for t in texts]
    out = []
    for i in range(0, len(inputs), 32):
        enc = tok(inputs[i : i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(mean.numpy())
    v = np.concatenate(out).astype(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def metrics(scores, labels, kinds, t):
    """At threshold t: predict in_domain iff score >= t. Higher score = more in-domain."""
    pred_in = scores >= t
    is_in = labels == "in_domain"
    recall = (pred_in & is_in).sum() / max(is_in.sum(), 1)
    is_out = ~is_in
    tnr = (~pred_in & is_out).sum() / max(is_out.sum(), 1)
    nm = kinds == "near_miss_ood"
    far = kinds == "far_ood"
    nm_tnr = (~pred_in & nm).sum() / max(nm.sum(), 1)
    far_tnr = (~pred_in & far).sum() / max(far.sum(), 1)
    return recall, tnr, nm_tnr, far_tnr


def auc(scores, is_in):
    """ROC-AUC for score separating in-domain (positive) from OOD."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = is_in.sum()
    neg = len(scores) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return (ranks[is_in].sum() - pos * (pos + 1) / 2) / (pos * neg)


def main():
    docs = [line for line in CATALOG.splitlines() if line.strip()]
    probes = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
    sentences = [p["sentence"] for p in probes]
    labels = np.array([p["label"] for p in probes])
    kinds = np.array([p["kind"] for p in probes])

    print(f"Embedding {len(docs)} skill docs + {len(sentences)} probes via {MODEL} ...")
    dvec = embed(docs, "passage")     # (n_docs, d)
    qvec = embed(sentences, "query")  # (184, d)
    sims = qvec @ dvec.T              # (184, n_docs) cosine (normalized)

    sims_sorted = np.sort(sims, axis=1)[:, ::-1]
    top1 = sims_sorted[:, 0]
    mean_top3 = sims_sorted[:, :3].mean(axis=1)
    margin = sims_sorted[:, 0] - sims_sorted[:, 1]
    is_in = labels == "in_domain"

    signals = {"top1_cosine": top1, "mean_top3_cosine": mean_top3, "top1_minus_top2_margin": margin}

    print("\n=== Signal AUCs (separating in-domain from OOD) ===")
    for name, s in signals.items():
        print(f"  {name:26s} AUC={auc(s, is_in):.4f}")

    print("\n=== Operating points: max TNR s.t. recall >= target ===")
    print("baseline (TF-IDF @0.08): recall=0.990 overall_TNR=0.163 near_miss_TNR=0.036 far_TNR=0.458\n")
    for name, s in signals.items():
        ts = np.unique(s)
        print(f"-- signal: {name} (AUC {auc(s, is_in):.4f}) --")
        for target in (0.99, 0.97, 0.95):
            best = None
            for t in ts:
                recall, tnr, nm_tnr, far_tnr = metrics(s, labels, kinds, t)
                if recall >= target and (best is None or tnr > best[1]):
                    best = (t, tnr, nm_tnr, far_tnr, recall)
            if best:
                t, tnr, nm_tnr, far_tnr, recall = best
                print(f"   recall>={target}: t={t:.4f} -> recall={recall:.3f} TNR={tnr:.3f} near_miss_TNR={nm_tnr:.3f} far_TNR={far_tnr:.3f}")
            else:
                print(f"   recall>={target}: (unreachable)")
        print()


if __name__ == "__main__":
    sys.exit(main())
