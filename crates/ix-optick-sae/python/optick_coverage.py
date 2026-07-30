"""
Activation-coverage declaration for the OPTIC-K SAE artifact.

Stdlib-only by design (no torch/numpy/pandas), so the trainer's split invariant
can be unit-tested without an ML environment — same discipline as
``test_partition_contract.py``.

Motivation (ix #248): ``feature_activations.parquet`` holds ONLY the training
rows (each keyed by ``optick_row`` into the full corpus). The held-out val rows
are legitimately absent, but that fact was undocumented and unasserted — a
consumer joining ``optick_row`` against the whole corpus silently drops the
~5% val split with no error and no contract field. This module turns the split
into an explicit, consumer-assertable block in ``optick-sae-artifact.json`` and
enforces the additive identity ``n_train + n_val == corpus_n`` at produce time.
"""
from __future__ import annotations

from typing import Dict


def activations_coverage(n_train: int, n_val: int, corpus_n: int) -> Dict[str, object]:
    """
    Build the ``activations_coverage`` block for the SAE artifact.

    ``feature_activations.parquet`` contains exactly ``n_train`` rows (the
    seeded train split); ``n_val`` rows are the held-out remainder. A consumer
    can assert both ``parquet_rows == n_train`` and, against the OPTIC-K index,
    ``n_train + n_val == corpus_n`` before joining.

    Raises ``ValueError`` if the counts are negative or if the split does not
    partition the corpus — a non-additive split is a bug, not a warning
    (that is the exact ix #248 failure mode).
    """
    if min(n_train, n_val, corpus_n) < 0:
        raise ValueError(
            f"coverage counts must be non-negative: "
            f"n_train={n_train} n_val={n_val} corpus_n={corpus_n}"
        )
    if n_train + n_val != corpus_n:
        raise ValueError(
            f"split is not additive over the corpus: "
            f"n_train({n_train}) + n_val({n_val}) = {n_train + n_val} "
            f"!= corpus_n({corpus_n}). feature_activations.parquet coverage "
            f"cannot be declared for a non-partitioning split (ix #248)."
        )
    return {
        # The parquet's optick_row values index the train split only.
        "optick_row_split": "train",
        "n_train": n_train,
        "n_val": n_val,
        "corpus_n": corpus_n,
        # Fraction of the corpus present in feature_activations.parquet.
        "coverage_pct": round(100.0 * n_train / max(corpus_n, 1), 2),
    }
