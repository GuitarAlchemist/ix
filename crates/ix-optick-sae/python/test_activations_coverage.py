"""
Contract test for the SAE artifact's ``activations_coverage`` block (ix #248).

Two independent guards, both stdlib-only (no torch/numpy import), matching
``test_partition_contract.py``:

  1. The pure helper ``optick_coverage.activations_coverage`` builds the right
     block for real corpus shapes and enforces the additive split invariant.
  2. Static (AST/source) check that ``train.py`` actually WIRES the helper into
     ``build_artifact`` and feeds ``n_val`` through ``metrics`` — so the block
     can't silently stop being emitted (the wiring is the part that regressed
     in the field: the parquet shipped train-only with no coverage field).

Run::

    python -m unittest crates/ix-optick-sae/python/test_activations_coverage.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parent
_TRAINER_SOURCE = _PYTHON_DIR / "train.py"

# Import the stdlib-only helper directly (no torch), the way the trainer does.
sys.path.insert(0, str(_PYTHON_DIR))
from optick_coverage import activations_coverage  # noqa: E402


class ActivationsCoverageHelperTests(unittest.TestCase):
    # The real 2026-07-20 run shapes (training.log): n_train + n_val == corpus.
    N_TRAIN = 297_395
    N_VAL = 15_652
    CORPUS = 313_047

    def test_block_shape_for_real_corpus(self) -> None:
        block = activations_coverage(self.N_TRAIN, self.N_VAL, self.CORPUS)
        self.assertEqual(block["optick_row_split"], "train")
        self.assertEqual(block["n_train"], self.N_TRAIN)
        self.assertEqual(block["n_val"], self.N_VAL)
        self.assertEqual(block["corpus_n"], self.CORPUS)
        # 297395 / 313047 = 95.0006% -> 95.0
        self.assertEqual(block["coverage_pct"], 95.0)

    def test_additivity_holds_by_construction(self) -> None:
        block = activations_coverage(self.N_TRAIN, self.N_VAL, self.CORPUS)
        self.assertEqual(block["n_train"] + block["n_val"], block["corpus_n"])

    def test_non_partitioning_split_raises(self) -> None:
        # corpus_n one larger than the split total: a silent-drop bug (#248).
        with self.assertRaises(ValueError):
            activations_coverage(self.N_TRAIN, self.N_VAL, self.CORPUS + 1)

    def test_negative_counts_raise(self) -> None:
        with self.assertRaises(ValueError):
            activations_coverage(-1, 10, 9)


class TrainerWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _TRAINER_SOURCE.read_text(encoding="utf-8")

    def test_build_artifact_wires_the_helper(self) -> None:
        self.assertIn(
            "activations_coverage",
            self.source,
            "train.py no longer references activations_coverage — the coverage "
            "block would stop being emitted (the ix #248 regression).",
        )
        self.assertIn(
            '"activations_coverage": coverage',
            self.source,
            "the coverage block is computed but not placed into the artifact dict.",
        )

    def test_metrics_carries_n_val(self) -> None:
        self.assertIn(
            '"n_val": len(val_idx)',
            self.source,
            "metrics must carry n_val so build_artifact can declare the val split.",
        )


if __name__ == "__main__":
    unittest.main()
