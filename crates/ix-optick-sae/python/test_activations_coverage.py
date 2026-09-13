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

import ast
import sys
import unittest
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parent
_TRAINER_SOURCE = _PYTHON_DIR / "train.py"

# Import the stdlib-only helper directly (no torch), the way the trainer does.
sys.path.insert(0, str(_PYTHON_DIR))
from optick_coverage import (  # noqa: E402
    KeyStats,
    Unevaluatable,
    _prefix_split_is_implausible,
    activations_coverage,
    failures,
    reconcile,
    reconcile_snapshot,
)


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


def _call_name(node: ast.Call) -> str:
    """Best-effort callee name for ``f(...)`` and ``mod.f(...)``."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _scope_calling(tree: ast.AST, callee: str):
    """The function that CALLS ``callee``, plus that call's line number.

    Returns ``(FunctionDef, lineno)`` or ``(None, None)``. Scoping to the calling
    function is the whole point: comparing raw line numbers across the module
    would be meaningless here, because ``build_artifact`` is *defined* above
    ``main`` and so its internal calls always have lower line numbers than
    anything in ``main`` — a source-order check would pass while the runtime
    order stayed wrong.
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and _call_name(sub) == callee:
                return node, sub.lineno
    return None, None


class CoverageGuardOrderingTests(unittest.TestCase):
    """The guard must run BEFORE anything is written to disk (ix #248 P0).

    ``activations_coverage`` raises on a non-additive split by design. If it is
    only reached from inside ``build_artifact`` — which runs *after*
    ``save_outputs`` — the raise happens once the parquet, weights and manifest
    are already on disk. That leaves activation files with no declaring artifact
    JSON: exactly the federation-drop #248 added the guard to prevent.

    The pre-existing wiring test cannot catch this. It asserts only that
    ``train.py`` *references* ``activations_coverage`` — true either way — so the
    suite stays green while the ordering is wrong.
    """

    def setUp(self) -> None:
        self.tree = ast.parse(_TRAINER_SOURCE.read_text(encoding="utf-8"))

    def test_coverage_guard_precedes_save_outputs(self) -> None:
        scope, save_line = _scope_calling(self.tree, "save_outputs")
        self.assertIsNotNone(
            scope, "no function in train.py calls save_outputs — test is stale."
        )

        guard_lines = [
            n.lineno
            for n in ast.walk(scope)
            if isinstance(n, ast.Call) and _call_name(n) == "activations_coverage"
        ]
        self.assertTrue(
            guard_lines,
            f"{scope.name}() calls save_outputs (line {save_line}) without ever "
            "calling activations_coverage. The split-additivity guard is only "
            "reachable from build_artifact, which runs after the files are "
            "written; a non-additive split would leave an orphaned parquet.",
        )
        self.assertTrue(
            any(line < save_line for line in guard_lines),
            f"activations_coverage is called in {scope.name}() at line(s) "
            f"{guard_lines}, but save_outputs writes at line {save_line}. The "
            "guard must precede the write, or it cannot prevent partial output.",
        )



class ReconcileTests(unittest.TestCase):
    """Declared-vs-observed reconciliation (ix #248).

    ``activations_coverage`` compares three integers the trainer already holds
    in memory, so it is blind to what actually reached disk. These tests pin the
    other half of the pair: one mutant per assertion, each mutant killing
    exactly the assertion it targets and no other, so a future edit that
    weakens one check cannot hide behind the others.

    Real-corpus shapes (2026-07-20 snapshot, measured with pyarrow):
    297,395 train rows keyed 0..313,046 over a 313,047-row corpus.
    """

    N_TRAIN = 297_395
    N_VAL = 15_652
    CORPUS = 313_047

    def declared(self, **overrides):
        block = activations_coverage(self.N_TRAIN, self.N_VAL, self.CORPUS)
        block.update(overrides)
        return block

    def observed(self, **overrides) -> KeyStats:
        base = dict(
            n_rows=self.N_TRAIN,
            key_present=True,
            n_non_null=self.N_TRAIN,
            n_distinct=self.N_TRAIN,
            key_min=0,
            key_max=self.CORPUS - 1,
        )
        base.update(overrides)
        return KeyStats(**base)

    def red_names(self, declared, observed):
        return {v.name for v in failures(reconcile(declared, observed))}

    # -- positive control ----------------------------------------------------

    def test_healthy_snapshot_is_all_green(self) -> None:
        verdicts = reconcile(self.declared(), self.observed())
        self.assertEqual(failures(verdicts), [])
        # Every assertion must actually run, or "all green" is vacuous.
        self.assertEqual(len(verdicts), 11)

    # -- one mutant per assertion --------------------------------------------

    def test_missing_declaration_is_red(self) -> None:
        # The shipped 2026-07-20 artifact: parquet fine, nothing declares it.
        self.assertEqual(self.red_names(None, self.observed()), {"coverage_declared"})
        self.assertEqual(self.red_names({}, self.observed()), {"coverage_declared"})

    def test_unknown_split_is_red(self) -> None:
        self.assertEqual(
            self.red_names(self.declared(optick_row_split="val"), self.observed()),
            {"split_is_known"},
        )

    def test_non_additive_declaration_is_red(self) -> None:
        # A hand-edited artifact can carry a split that does not partition even
        # though activations_coverage() would have refused to build it.
        red = self.red_names(self.declared(n_val=self.N_VAL - 1), self.observed())
        self.assertIn("split_additivity", red)

    def test_inconsistent_coverage_pct_is_red(self) -> None:
        self.assertEqual(
            self.red_names(self.declared(coverage_pct=100.0), self.observed()),
            {"coverage_pct_consistent"},
        )

    def test_coverage_below_floor_is_red(self) -> None:
        # Half the corpus held out: additive, self-consistent, and unacceptable.
        half = self.CORPUS // 2
        declared = activations_coverage(half, self.CORPUS - half, self.CORPUS)
        observed = self.observed(n_rows=half, n_non_null=half, n_distinct=half)
        self.assertEqual(self.red_names(declared, observed), {"coverage_floor"})

    def test_missing_key_column_is_red_and_stops_early(self) -> None:
        # The pre-#234 shape (2026-06-14 parquet): no optick_row column at all.
        verdicts = reconcile(
            self.declared(), KeyStats(n_rows=self.N_TRAIN, key_present=False)
        )
        self.assertEqual({v.name for v in failures(verdicts)}, {"key_present"})
        # Downstream key assertions are meaningless without a key; they must not
        # be reported as passing.
        self.assertNotIn("key_unique", {v.name for v in verdicts})

    def test_row_count_mismatch_is_red(self) -> None:
        # The writer dropped rows after the counts were computed — the exact
        # blind spot of the in-memory guard. A real short write loses the key
        # values with the rows, so the mutant shrinks all three together; that
        # keeps this a single-assertion kill rather than a shotgun.
        short = self.N_TRAIN - 7
        self.assertEqual(
            self.red_names(
                self.declared(),
                self.observed(n_rows=short, n_non_null=short, n_distinct=short),
            ),
            {"rows_match_declared"},
        )

    def test_null_key_is_red(self) -> None:
        # NULLs are invisible to COUNT(DISTINCT)/bool_and, so this needs its own
        # assertion or an all-NULL key column reads as healthy.
        self.assertEqual(
            self.red_names(self.declared(), self.observed(n_non_null=self.N_TRAIN - 1)),
            {"key_non_null"},
        )

    def test_duplicate_key_is_red(self) -> None:
        self.assertEqual(
            self.red_names(self.declared(), self.observed(n_distinct=self.N_TRAIN - 1)),
            {"key_unique"},
        )

    def test_out_of_range_key_is_red(self) -> None:
        self.assertEqual(
            self.red_names(self.declared(), self.observed(key_max=self.CORPUS)),
            {"key_in_corpus_range"},
        )

    def test_split_positions_key_is_red(self) -> None:
        """The keying bug that passes every other assertion.

        ``optick_row = arange(n_train)`` is unique, non-null, in range, and the
        declared row count — so count/dup/range checks are all green. Only the
        prefix-set discriminator catches it.
        """
        self.assertEqual(
            self.red_names(self.declared(), self.observed(key_max=self.N_TRAIN - 1)),
            {"key_is_corpus_positions"},
        )

    def test_tiny_corpus_skips_the_prefix_check(self) -> None:
        """A toy corpus can legitimately hold out exactly the suffix.

        With corpus_n=10 and n_val=1 there are only 10 possible val sets, so a
        correct producer writes corpus positions {0..8} — indistinguishable from
        split positions — one run in ten. The assertion is a hard produce-time
        failure, so it must stand down where a legitimate prefix is plausible.
        """
        declared = activations_coverage(9, 1, 10)
        observed = KeyStats(
            n_rows=9, key_present=True, n_non_null=9, n_distinct=9,
            key_min=0, key_max=8,  # looks like split positions, is not
        )
        verdicts = reconcile(declared, observed)
        self.assertNotIn("key_is_corpus_positions", {v.name for v in verdicts})
        # Nothing else may go red just because the corpus is small.
        self.assertEqual(failures(verdicts), [])

    def test_production_corpus_still_runs_the_prefix_check(self) -> None:
        # The skip above must not swallow the real detector at real scale:
        # C(313047, 15652) is astronomically larger than the plausibility cap.
        self.assertTrue(_prefix_split_is_implausible(self.CORPUS, self.N_VAL))
        self.assertFalse(_prefix_split_is_implausible(10, 1))
        self.assertEqual(
            self.red_names(self.declared(), self.observed(key_max=self.N_TRAIN - 1)),
            {"key_is_corpus_positions"},
        )

    def test_full_coverage_snapshot_skips_the_prefix_check(self) -> None:
        """With nothing held out, {0..corpus-1} IS the correct key.

        The prefix-set discriminator would misfire on a legitimate 100%-coverage
        artifact, so it is gated on n_val > 0.
        """
        declared = activations_coverage(self.CORPUS, 0, self.CORPUS)
        observed = self.observed(
            n_rows=self.CORPUS,
            n_non_null=self.CORPUS,
            n_distinct=self.CORPUS,
            key_max=self.CORPUS - 1,
        )
        verdicts = reconcile(declared, observed)
        self.assertEqual(failures(verdicts), [])
        self.assertNotIn("key_is_corpus_positions", {v.name for v in verdicts})



class UnevaluatableTests(unittest.TestCase):
    """"Could not check" must never be reported as "checked and contradictory".

    ``feature_activations.parquet`` is gitignored (56 MB), so it is structurally
    absent on any fresh checkout. If that absence returned the same exit code as
    a real contradiction, the distinction the reconciler advertises would be
    fiction and the failure would train people to ignore it.
    """

    def test_missing_snapshot_raises_rather_than_failing(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Unevaluatable):
                reconcile_snapshot(tmp)

    def test_missing_parquet_raises_rather_than_failing(self) -> None:
        import json
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmp:
            (_Path(tmp) / "optick-sae-artifact.json").write_text(
                json.dumps({"activations_coverage": activations_coverage(95, 5, 100)}),
                encoding="utf-8",
            )
            # Declaration is present and fine; only the bytes are unavailable.
            with self.assertRaises(Unevaluatable):
                reconcile_snapshot(tmp)


if __name__ == "__main__":
    unittest.main()
