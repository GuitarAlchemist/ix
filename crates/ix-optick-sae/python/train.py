#!/usr/bin/env python3
"""
OPTIC-K Sparse Autoencoder Trainer — Phase 1.

Trains a TopK SAE over the OPTIC-K v1.8 similarity-partition slice (124 dims out of 240).
Falls back to a synthetic corpus when optick.index is absent (cloud CI env).

Partition coverage: STRUCTURE, MORPHOLOGY, CONTEXT, SYMBOLIC, MODAL, ROOT.
Skipped for Phase 1: IDENTITY (tag-like, not similarity), EXTENSIONS, SPECTRAL,
HIERARCHY, ATONAL_MODAL.

Aligned to canonical baseline at state/quality/optick-sae/2026-05-04/.

Exit codes (read by Rust orchestrator):
  0  — success; artifact written to --output-dir
  2  — reconstruction_mse > 0.05; no artifact written
  3  — dead_features_pct > 30%; no artifact written (triggers Rust retry)
  1  — unexpected error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# ── OPTIC-K v1.8 partition layout ─────────────────────────────────────────────
# Source of truth: ga/Common/GA.Business.ML/Embeddings/EmbeddingSchema.cs
# DO NOT modify without coordinating a re-index.

OPTIC_TOTAL_DIM = 240
OPTIC_SCHEMA_VERSION = "OPTIC-K-v1.8"

# Full partition map: name → (global_start, global_end) — Python half-open slices.
_ALL_PARTITION_SLICES: Dict[str, Tuple[int, int]] = {
    "IDENTITY":    (0,   6),
    "STRUCTURE":   (6,   30),
    "MORPHOLOGY":  (30,  54),
    "CONTEXT":     (54,  66),
    "SYMBOLIC":    (66,  78),
    "EXTENSIONS":  (78,  96),
    "SPECTRAL":    (96,  109),
    "MODAL":       (109, 149),
    "HIERARCHY":   (149, 164),
    "ATONAL_MODAL":(164, 228),
    "ROOT":        (228, 240),
}

PHASE1_PARTITIONS: List[str] = [
    "STRUCTURE", "MORPHOLOGY", "CONTEXT", "SYMBOLIC", "MODAL", "ROOT",
]

# Global indices of Phase 1 dimensions within the full 240-dim vector.
PHASE1_GLOBAL_INDICES: List[int] = [
    i
    for name in PHASE1_PARTITIONS
    for i in range(*_ALL_PARTITION_SLICES[name])
]
PHASE1_DIM = len(PHASE1_GLOBAL_INDICES)  # 124

# Local (0-based) slice boundaries inside the 124-dim Phase 1 space.
_LOCAL_OFFSETS: List[Tuple[str, int, int]] = []
_cursor = 0
for _name in PHASE1_PARTITIONS:
    _gs, _ge = _ALL_PARTITION_SLICES[_name]
    _width = _ge - _gs
    _LOCAL_OFFSETS.append((_name, _cursor, _cursor + _width))
    _cursor += _width
assert _cursor == PHASE1_DIM


# ── TopK Sparse Autoencoder ───────────────────────────────────────────────────

class TopKSAE(nn.Module):
    """
    Minimal TopK Sparse Autoencoder following the Bricken et al. / Bloom et al. design:
      encode: centre → linear → top-k ReLU
      decode: linear (no bias) + pre-encoder bias
    Decoder columns are kept at unit norm throughout training.
    """

    def __init__(self, input_dim: int, dict_size: int, k: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k

        # Pre-encoder bias (subtracted before encoding, added after decoding).
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        # Encoder: input_dim → dict_size
        self.W_enc = nn.Linear(input_dim, dict_size, bias=True)
        # Decoder atoms: shape (dict_size, input_dim).
        self.W_dec = nn.Parameter(torch.empty(dict_size, input_dim))

        nn.init.kaiming_uniform_(self.W_enc.weight, nonlinearity="relu")
        nn.init.orthogonal_(self.W_dec)
        self._normalise_decoder()

    @torch.no_grad()
    def _normalise_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True)
        self.W_dec.div_(norms.clamp(min=1.0))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec
        pre_acts = self.W_enc(x_cent)
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return self.decode(acts), acts

    def auxk_reconstruct(
        self, x: torch.Tensor, dead_mask: torch.Tensor, k_aux: int
    ) -> "Optional[torch.Tensor]":
        """Reconstruct using only dead features (Anthropic AuxK / ghost-grads).
        Dead features receive zero gradient through the main top-k path; this
        auxiliary loss threads gradient through them so they can wake up."""
        if not dead_mask.any():
            return None
        x_cent = x - self.b_dec
        pre_acts = self.W_enc(x_cent)
        # Zero out live features so top-k selects only from dead ones.
        pre_dead = pre_acts * dead_mask.to(pre_acts.dtype).unsqueeze(0)
        n_dead = int(dead_mask.sum().item())
        k_eff = min(k_aux, n_dead)
        topk_vals, topk_idx = torch.topk(pre_dead, k_eff, dim=-1)
        acts_aux = torch.zeros_like(pre_acts)
        acts_aux.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return acts_aux @ self.W_dec + self.b_dec


# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus(
    index_path: str, rng: np.random.Generator
) -> Tuple[np.ndarray, str, bool]:
    """
    Returns (full_240_dim_data, sha256_hex_prefixed, is_synthetic).
    Falls back to a synthetic corpus when the file is absent.
    """
    p = Path(index_path)
    if index_path == "synthetic" or not p.exists():
        log.warning(
            "optick.index not found at '%s' — generating synthetic 5 000-voicing × %d-dim "
            "corpus. DEVIATION: smoke run uses synthetic data, not real OPTIC-K index.",
            index_path,
            OPTIC_TOTAL_DIM,
        )
        data = _synthetic_corpus(5_000, rng)
        sha = "sha256:" + hashlib.sha256(data.tobytes()).hexdigest()
        return data, sha, True

    if index_path.endswith(".npy"):
        data = np.load(index_path).astype(np.float32)
        sha = "sha256:" + _sha256_file(p)
        return data, sha, False

    # OPTK v4 binary mmap index — the real state/voicings/optick.index format
    # (magic "OPTK"). Stores the COMPACT similarity vectors directly, already
    # pre-weighted by sqrt(partition weight) and L2-normalized, so they are
    # returned as-is and main() skips the Phase-1 slice (the data is already the
    # PHASE1_DIM compact, not the full OPTIC_TOTAL_DIM). Layout mirrors GA's
    # Common/GA.Business.ML/Search/OptickIndexReader.cs.
    with open(index_path, "rb") as f:
        magic = f.read(4)
    if magic == b"OPTK":
        data = _load_optk(p)
        sha = "sha256:" + _sha256_file(p)
        return data, sha, False

    # Attempt msgpack (a msgpack-serialized {"vectors": [...]} dump, if one is fed).
    try:
        import msgpack  # type: ignore[import-untyped]
        with open(index_path, "rb") as f:
            raw = msgpack.unpackb(f.read(), raw=False)
        data = np.array(raw["vectors"], dtype=np.float32)
        sha = "sha256:" + _sha256_file(p)
        return data, sha, False
    except Exception as exc:
        log.warning("Could not parse '%s' as msgpack (%s) — falling back to synthetic.", index_path, exc)
        data = _synthetic_corpus(1_000, rng)
        sha = "sha256:" + hashlib.sha256(data.tobytes()).hexdigest()
        return data, sha, True


def _load_optk(p: Path) -> np.ndarray:
    """
    Read the compact vectors from a GA OPTK v4 memory-mapped index.

    Mirrors Common/GA.Business.ML/Search/OptickIndexReader.cs:
      header: magic(4) version(4) header_size(4) schema_hash(4) endian(2) _r(2)
              dim(4 @20) count(8 @24) ... vectors_off(8 @96) ...
      body:   count * dim little-endian float32, contiguous from vectors_off.
    Returns (count, dim) float32 — the pre-weighted, L2-normalized compact
    similarity vectors (dim == PHASE1_DIM for v1.8).
    """
    raw = p.read_bytes()
    if raw[:4] != b"OPTK":
        raise ValueError(f"{p}: not an OPTK index (magic={raw[:4]!r})")
    version = int.from_bytes(raw[4:8], "little")
    if version != 4:
        raise ValueError(f"{p}: unsupported OPTK version {version} (expected 4)")
    dim = int.from_bytes(raw[20:24], "little")
    count = int.from_bytes(raw[24:32], "little")
    vectors_off = int.from_bytes(raw[96:104], "little")
    n = count * dim
    vecs = np.frombuffer(raw, dtype="<f4", count=n, offset=vectors_off)
    if vecs.size != n:
        raise ValueError(f"{p}: truncated vector block (want {n} floats, got {vecs.size})")
    return vecs.reshape(count, dim).astype(np.float32, copy=False)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _synthetic_corpus(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    n × OPTIC_TOTAL_DIM unit-sphere synthetic voicings.
    Uses 32 Gaussian centres spread across all 11 partitions to produce
    varied per-partition activation patterns that stress-test partition purity.
    """
    n_centres = 32
    centres = (rng.standard_normal((n_centres, OPTIC_TOTAL_DIM)) * 0.4).astype(np.float32)
    assignments = rng.integers(0, n_centres, size=n)
    noise = (rng.standard_normal((n, OPTIC_TOTAL_DIM)) * 0.15).astype(np.float32)
    data = centres[assignments] + noise
    # L2-normalise so per-dim variance ≈ 1/OPTIC_TOTAL_DIM — keeps MSE well below 0.05.
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data /= np.maximum(norms, 1e-8)
    return data


def slice_phase1(data: np.ndarray) -> np.ndarray:
    """Extract the 124 Phase 1 dimensions from full 240-dim embeddings."""
    return data[:, PHASE1_GLOBAL_INDICES]


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data: np.ndarray,
    dict_size: int,
    k_sparse: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    held_out_pct: float,
    aux_alpha: float = 0.10,
    aux_k: int = 64,
) -> Tuple[TopKSAE, Dict]:
    """
    Trains TopK SAE and returns (model, metrics_dict).
    data is already sliced to PHASE1_DIM dims.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(data)
    n_held = max(1, int(n * held_out_pct))
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx, val_idx = idx[n_held:], idx[:n_held]

    X_train = torch.tensor(data[train_idx], dtype=torch.float32)
    X_val = torch.tensor(data[val_idx], dtype=torch.float32)
    input_dim = data.shape[1]

    log.info(
        "TopK SAE: input_dim=%d  dict_size=%d  k=%d  n_train=%d  n_val=%d",
        input_dim, dict_size, k_sparse, len(train_idx), len(val_idx),
    )

    model = TopKSAE(input_dim, dict_size, k_sparse)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    log_interval = max(1, epochs // 10)

    history: List[float] = []
    t0 = time.time()
    # Track dead features across epochs for AuxK ghost grads.
    dead_mask = torch.zeros(dict_size, dtype=torch.bool)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        n_batches = 0
        epoch_active_counts = torch.zeros(dict_size)

        for i in range(0, len(X_train), batch_size):
            batch = X_train[perm[i : i + batch_size]]
            x_hat, acts = model(batch)
            main_loss = nn.functional.mse_loss(x_hat, batch)

            with torch.no_grad():
                epoch_active_counts += (acts > 0).float().sum(dim=0)

            total_loss = main_loss
            if aux_alpha > 0.0 and dead_mask.any():
                residual = (batch - x_hat).detach()
                x_hat_aux = model.auxk_reconstruct(batch, dead_mask, aux_k)
                if x_hat_aux is not None:
                    aux_loss = nn.functional.mse_loss(x_hat_aux, residual)
                    total_loss = main_loss + aux_alpha * aux_loss
                    epoch_aux_loss += aux_loss.item()

            optimiser.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability on small corpora.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            model._normalise_decoder()
            epoch_loss += main_loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        history.append(avg)
        dead_mask = (epoch_active_counts == 0)
        n_dead_now = int(dead_mask.sum().item())
        if epoch % log_interval == 0 or epoch == epochs:
            avg_aux = epoch_aux_loss / max(n_batches, 1)
            log.info(
                "epoch %d/%d  loss=%.6f  aux=%.6f  dead=%d  elapsed=%.1fs",
                epoch, epochs, avg, avg_aux, n_dead_now, time.time() - t0,
            )

    # ── Evaluation on validation set ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x_hat_val, acts_val = model(X_val)

        mse = nn.functional.mse_loss(x_hat_val, X_val).item()
        ss_res = float(((X_val - x_hat_val) ** 2).sum())
        ss_tot = float(((X_val - X_val.mean(0, keepdim=True)) ** 2).sum())
        r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))

        # Percentiles of active features per voicing (val set).
        active_per_val = (acts_val > 0).sum(dim=1).cpu().numpy().astype(int)
        p50 = int(np.percentile(active_per_val, 50))
        p95 = int(np.percentile(active_per_val, 95))

    # ── Dead features (full training set for robust estimate) ─────────────────
    with torch.no_grad():
        ever_active = torch.zeros(dict_size, dtype=torch.bool)
        for i in range(0, len(X_train), 512):
            _, a = model(X_train[i : i + 512])
            ever_active |= (a > 0).any(dim=0)

        dead_pct = float((~ever_active).sum().item() / dict_size * 100)
        alive_count = int(ever_active.sum().item())

        # Feature activation stats on training set (for parquet + manifest).
        all_acts_list: List[torch.Tensor] = []
        for i in range(0, len(X_train), 512):
            _, a = model(X_train[i : i + 512])
            all_acts_list.append(a.cpu())
        all_acts = torch.cat(all_acts_list, dim=0)  # (n_train, dict_size)

        freq = (all_acts > 0).float().sum(dim=0).numpy()  # (dict_size,)

    high_freq_threshold = 0.01 * len(train_idx)
    low_freq_threshold = 1.0
    high_freq = int((freq >= high_freq_threshold).sum())
    low_freq = int((freq <= low_freq_threshold).sum())

    sparsity = float((all_acts > 0).float().mean().item())

    # ── Partition purity from decoder weights ─────────────────────────────────
    W = model.W_dec.detach().cpu().numpy()  # (dict_size, PHASE1_DIM)
    purity = _partition_purity(W)
    purity_mean = float(np.mean(purity))
    purity_p10 = float(np.percentile(purity, 10))

    return model, {
        "mse": mse,
        "r2": r2,
        "p50": p50,
        "p95": p95,
        "dead_pct": dead_pct,
        "alive": alive_count,
        "purity_mean": purity_mean,
        "purity_p10": purity_p10,
        "sparsity": sparsity,
        "loss_final": history[-1] if history else 0.0,
        "high_freq": high_freq,
        "low_freq": low_freq,
        "all_acts": all_acts,
        "freq": freq,
        "train_idx": train_idx,
    }


def _partition_purity(W: np.ndarray) -> np.ndarray:
    """
    For each dictionary atom (row of W_dec), compute:
      purity = max_partition(||w[partition]||²) / ||w||²
    A value of 1 means the atom lives entirely in one partition.
    Shape: (dict_size,).
    """
    purity = np.zeros(len(W), dtype=np.float32)
    for i, row in enumerate(W):
        norms_sq = np.array(
            [np.sum(row[s:e] ** 2) for _, s, e in _LOCAL_OFFSETS],
            dtype=np.float64,
        )
        total = norms_sq.sum()
        purity[i] = float(norms_sq.max() / max(total, 1e-12))
    return purity


# ── Output files ──────────────────────────────────────────────────────────────

def save_outputs(
    model: TopKSAE,
    metrics: Dict,
    output_dir: Path,
) -> None:
    all_acts: torch.Tensor = metrics["all_acts"]
    freq: np.ndarray = metrics["freq"]
    acts_np = all_acts.numpy()

    # feature_activations.parquet
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame(
        acts_np, columns=[f"f{i}" for i in range(acts_np.shape[1])]
    )
    df.to_parquet(output_dir / "feature_activations.parquet", index=False)
    log.info("Saved feature_activations.parquet  shape=%s", acts_np.shape)

    # feature_manifest.jsonl
    W = model.W_dec.detach().cpu().numpy()
    with open(output_dir / "feature_manifest.jsonl", "w") as f:
        for i in range(model.dict_size):
            entry = {
                "feature_idx": i,
                "activation_count": int(freq[i]),
                "is_alive": bool(freq[i] > 0),
                "decoder_norm": round(float(np.linalg.norm(W[i])), 6),
            }
            f.write(json.dumps(entry) + "\n")
    log.info("Saved feature_manifest.jsonl  (%d features)", model.dict_size)

    # sae_weights.safetensors (fallback: .pt)
    try:
        from safetensors.torch import save_file  # noqa: PLC0415

        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save_file(state, str(output_dir / "sae_weights.safetensors"))
        log.info("Saved sae_weights.safetensors")
    except ImportError:
        log.warning("safetensors not installed — weights saved as sae_weights.safetensors (PyTorch format)")
        torch.save(model.state_dict(), output_dir / "sae_weights.safetensors")


# ── Artifact JSON ─────────────────────────────────────────────────────────────

def build_artifact(
    *,
    artifact_id: str,
    index_path: str,
    index_sha: str,
    corpus_size: int,
    dict_size: int,
    k_sparse: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    metrics: Dict,
    is_synthetic: bool,
    retry_note: Optional[str],
    supersedes: Optional[str] = None,
) -> Dict:
    base = (
        f"Phase 1 smoke run — synthetic 5 000-voicing x {PHASE1_DIM}-dim corpus "
        "(optick.index absent in CI env; deviation documented)."
        if is_synthetic
        else f"Phase 1 training run on real OPTIC-K corpus ({PHASE1_DIM}-dim slice)."
    )
    suffix = f" {retry_note}" if retry_note else ""
    narrative = (base + suffix)[:500]

    return {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "trained_at": _utc_now(),
        "trainer": "ix-optick-sae",
        "trainer_version": "0.1.0",
        "input": {
            "optick_index_path": index_path,
            "optick_index_sha": index_sha,
            "optick_dim": OPTIC_TOTAL_DIM,
            "compact_training_dim": PHASE1_DIM,
            "schema_version": OPTIC_SCHEMA_VERSION,
            "corpus_size": corpus_size,
            "partitions_used": PHASE1_PARTITIONS,
        },
        "model": {
            "kind": "topk_sae",
            "dict_size": dict_size,
            "k_sparse": k_sparse,
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "loss_final": round(metrics["loss_final"], 8),
                "sparsity_actual_mean": round(metrics["sparsity"], 6),
            },
        },
        "metrics": {
            "reconstruction_mse": round(metrics["mse"], 8),
            "reconstruction_r2": round(min(1.0, max(0.0, metrics["r2"])), 6),
            "active_features_per_voicing_p50": metrics["p50"],
            "active_features_per_voicing_p95": metrics["p95"],
            "dead_features_pct": round(metrics["dead_pct"], 2),
            "feature_partition_purity_mean": round(
                min(1.0, max(0.0, metrics["purity_mean"])), 6
            ),
            "feature_partition_purity_p10": round(
                min(1.0, max(0.0, metrics["purity_p10"])), 6
            ),
        },
        "features_summary": {
            "total": dict_size,
            "alive": metrics["alive"],
            "high_frequency_count": metrics["high_freq"],
            "low_frequency_count": metrics["low_freq"],
        },
        "links": {
            "feature_activations_parquet": "feature_activations.parquet",
            "feature_manifest_jsonl": "feature_manifest.jsonl",
            "training_log": "training.log",
            "model_weights": "sae_weights.safetensors",
            "supersedes": supersedes,
        },
        "narrative": narrative,
    }


def _utc_now() -> str:
    from datetime import datetime, timezone  # noqa: PLC0415

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPTIC-K SAE Trainer — Phase 1")
    p.add_argument("--index",        required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--artifact-id",  required=True)
    p.add_argument("--dict-size",    type=int,   default=1024)
    p.add_argument("--k-sparse",     type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--held-out-pct", type=float, default=0.05)
    p.add_argument("--retry-note",   default=None)
    p.add_argument("--aux-alpha",    type=float, default=0.10,
                   help="Weight on AuxK ghost-grad auxiliary loss (0 = disabled).")
    p.add_argument("--aux-k",        type=int,   default=64,
                   help="Top-k_aux from the dead-feature pool.")
    p.add_argument("--supersedes",   default=None,
                   help="artifact_id this run supersedes (links.supersedes in the artifact).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File logging (Rust reads training.log via links.training_log).
    file_handler = logging.FileHandler(output_dir / "training.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    rng = np.random.default_rng(args.seed)

    raw_data, index_sha, is_synthetic = load_corpus(args.index, rng)
    corpus_size = len(raw_data)
    if raw_data.shape[1] == OPTIC_TOTAL_DIM:
        phase1_data = slice_phase1(raw_data)            # full 240-dim embeddings → 124 compact
        source_note = f"phase1 slice of {OPTIC_TOTAL_DIM}"
    elif raw_data.shape[1] == PHASE1_DIM:
        phase1_data = raw_data                          # already compact (OPTK index / 124-dim dump)
        source_note = "already-compact input"
    else:
        raise ValueError(
            f"Corpus has {raw_data.shape[1]} dims; expected {OPTIC_TOTAL_DIM} (full) "
            f"or {PHASE1_DIM} (compact)."
        )
    assert phase1_data.shape[1] == PHASE1_DIM, (
        f"Expected {PHASE1_DIM} Phase 1 dims, got {phase1_data.shape[1]}"
    )
    log.info("Corpus: %d voicings × %d dims (%s)", corpus_size, PHASE1_DIM, source_note)

    model, metrics = train(
        data=phase1_data,
        dict_size=args.dict_size,
        k_sparse=args.k_sparse,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        held_out_pct=args.held_out_pct,
        aux_alpha=args.aux_alpha,
        aux_k=args.aux_k,
    )

    log.info(
        "metrics: mse=%.6f  r2=%.4f  dead=%.1f%%  purity_mean=%.4f  purity_p10=%.4f",
        metrics["mse"], metrics["r2"], metrics["dead_pct"],
        metrics["purity_mean"], metrics["purity_p10"],
    )

    # ── Guardrail: reconstruction MSE ─────────────────────────────────────────
    if metrics["mse"] > 0.05:
        log.error(
            "FAIL reconstruction_mse=%.6f > 0.05 guardrail. No artifact emitted. "
            "Diagnostics: %s",
            metrics["mse"],
            json.dumps({k: v for k, v in metrics.items() if k not in ("all_acts", "freq", "train_idx")},
                       default=float),
        )
        sys.exit(2)

    # ── Guardrail: dead features (triggers Rust retry, not hard fail here) ────
    if metrics["dead_pct"] > 30.0:
        log.warning(
            "DEAD_FEATURES_HIGH: dead_features_pct=%.1f%% > 30%%. "
            "Exiting code 3 — Rust orchestrator will retry with dict_size=512.",
            metrics["dead_pct"],
        )
        sys.exit(3)

    # ── Write output files ────────────────────────────────────────────────────
    save_outputs(model, metrics, output_dir)

    artifact = build_artifact(
        artifact_id=args.artifact_id,
        index_path=args.index,
        index_sha=index_sha,
        corpus_size=corpus_size,
        dict_size=args.dict_size,
        k_sparse=args.k_sparse,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        metrics=metrics,
        is_synthetic=is_synthetic,
        retry_note=args.retry_note,
        supersedes=args.supersedes,
    )

    artifact_path = output_dir / "optick-sae-artifact.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    log.info("Artifact written: %s", artifact_path)

    # Summary line to stdout (Rust reads this for quick confirmation).
    print(json.dumps({
        "artifact_path": str(artifact_path),
        "reconstruction_mse": artifact["metrics"]["reconstruction_mse"],
        "dead_features_pct": artifact["metrics"]["dead_features_pct"],
        "alive": artifact["features_summary"]["alive"],
        "total": artifact["features_summary"]["total"],
    }))


if __name__ == "__main__":
    main()
