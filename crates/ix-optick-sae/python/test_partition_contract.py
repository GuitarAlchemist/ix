"""
Producer/consumer schema-contract test for the Python side of the
OPTIC-K Phase 1 partition list — mirrors ``tests/partition_contract.rs``.

Background: PR #82 (GA) and ix #29 both shipped (or nearly shipped) an
SAE artifact / producer constant missing the ROOT partition. The Rust
test in ``tests/partition_contract.rs`` guards the Rust constant. This
file guards the Python constant in ``train.py`` — they are two
independent declarations of the same fact, and either side can drift.

Stdlib-only by design: parses ``train.py`` with ``ast`` so the test
runs without importing torch/numpy/etc. ``python -m unittest`` is
enough; no pytest dependency.

Run from anywhere via::

    python -m unittest crates/ix-optick-sae/python/test_partition_contract.py
"""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


_PYTHON_DIR = Path(__file__).resolve().parent
_TRAINER_SOURCE = _PYTHON_DIR / "train.py"
_FIXTURE = _PYTHON_DIR.parent / "tests" / "fixtures" / "canonical-partitions.json"


def _extract_phase1_partitions(trainer_path: Path) -> list[str]:
    """
    Statically extract the ``PHASE1_PARTITIONS`` list literal from
    ``train.py`` without importing the module. Returns the partition
    names in declaration order.

    Raises if the constant is missing, not a list literal, or contains
    non-string elements — those are all real drift signals.
    """
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # The trainer declares PHASE1_PARTITIONS with a type annotation
        # (`PHASE1_PARTITIONS: List[str] = [...]`), which the AST models as
        # `AnnAssign` rather than `Assign`. Handle both shapes — bare
        # assignment and annotated assignment — so adding/removing the
        # annotation in the future doesn't silently disable this test.
        target_name: str | None = None
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "PHASE1_PARTITIONS":
                    target_name = tgt.id
                    value_node = node.value
                    break
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "PHASE1_PARTITIONS"
                and node.value is not None
            ):
                target_name = node.target.id
                value_node = node.value

        if target_name is None or value_node is None:
            continue

        if not isinstance(value_node, ast.List):
            raise AssertionError(
                "PHASE1_PARTITIONS is not a list literal — drift detector "
                "needs a static literal it can extract without import."
            )
        names: list[str] = []
        for elt in value_node.elts:
            if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                raise AssertionError(
                    "PHASE1_PARTITIONS contains a non-string element; "
                    "expected list[str] of partition names."
                )
            names.append(elt.value)
        return names

    raise AssertionError(
        f"PHASE1_PARTITIONS not found in {trainer_path}. "
        "If the constant moved, update this contract test."
    )


class PartitionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        with _FIXTURE.open("r", encoding="utf-8") as f:
            self.canonical = json.load(f)

    def test_phase1_partitions_match_canonical_baseline(self) -> None:
        producer = _extract_phase1_partitions(_TRAINER_SOURCE)
        canonical = self.canonical["partitions_used"]
        self.assertEqual(
            producer,
            canonical,
            "\n"
            "PHASE1_PARTITIONS in train.py has drifted from the canonical baseline.\n"
            "\n"
            f"  producer (train.py):  {producer!r}\n"
            f"  canonical (fixture):  {canonical!r}\n"
            "\n"
            "If the producer is wrong, fix the constant.\n"
            "If the canonical baseline legitimately changed (rare — re-indexing\n"
            "cost is high), coordinate the change with GA's optick-sae-artifact.json\n"
            "AND refresh tests/fixtures/canonical-partitions.json in the same PR.\n"
            "\n"
            "This drift is the bug class PR #82 (GA) and ix #29 hit (ROOT missed twice).\n",
        )

    def test_compact_training_dim_matches_canonical_baseline(self) -> None:
        # OPTIC-K v1.8 similarity-partition widths (from
        # ga/Common/GA.Business.ML/Embeddings/EmbeddingSchema.cs):
        #   STRUCTURE 24 + MORPHOLOGY 24 + CONTEXT 12 + SYMBOLIC 12
        #   + MODAL 40 + ROOT 12 = 124
        expected = 124
        actual = self.canonical["compact_training_dim"]
        self.assertEqual(
            actual,
            expected,
            f"Canonical fixture's compact_training_dim ({actual}) drifted from the "
            f"OPTIC-K v1.8 similarity-partition sum ({expected}). If partition widths "
            f"legitimately changed, update this constant AND ga's EmbeddingSchema.cs "
            f"in the same PR.",
        )


if __name__ == "__main__":
    unittest.main()
