"""
Activation-coverage declaration + reconciliation for the OPTIC-K SAE artifact.

Stdlib-only by design (no torch/numpy/pandas), so the trainer's split invariant
can be unit-tested without an ML environment — same discipline as
``test_partition_contract.py``. The one place that must touch parquet is
``read_key_stats`` at the bottom, which imports pandas lazily.

Motivation (ix #248): ``feature_activations.parquet`` holds ONLY the training
rows (each keyed by ``optick_row`` into the full corpus). The held-out val rows
are legitimately absent, but that fact was undocumented and unasserted — a
consumer joining ``optick_row`` against the whole corpus silently drops the
~5% val split with no error and no contract field. This module turns the split
into an explicit, consumer-assertable block in ``optick-sae-artifact.json``,
enforces the additive identity ``n_train + n_val == corpus_n`` at produce time,
and — the part ``activations_coverage`` alone cannot do — RECONCILES that
declaration against the bytes that actually landed on disk.

``activations_coverage`` is self-referential: it compares three integers the
trainer is already holding in memory. It cannot see a writer that dropped rows,
a NULL key column, a duplicated id, or a key that indexes split positions
instead of corpus positions. ``reconcile`` is the declared-vs-observed half of
that pair.

Contract: docs/contracts/2026-09-07-optick-sae-activations-coverage.contract.md
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

# Minimum share of the corpus a train-split artifact may declare.
#
# The trainer's held-out fraction is 5% by default and is policy-capped at 10%,
# so any snapshot declaring under 90% coverage means the split changed
# materially and every consumer's "the parquet is ~the corpus" assumption is
# broken. Fail loud rather than let a 60%-coverage artifact federate quietly.
# Revisit trigger: a deliberate held_out_pct > 0.10 — bump this constant in the
# same PR, and never widen the join to hide the gap.
MIN_COVERAGE_PCT = 90.0

# The only split the parquet is allowed to key on today.
OPTICK_ROW_SPLIT = "train"


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
        "optick_row_split": OPTICK_ROW_SPLIT,
        "n_train": n_train,
        "n_val": n_val,
        "corpus_n": corpus_n,
        # Fraction of the corpus present in feature_activations.parquet.
        "coverage_pct": coverage_pct(n_train, corpus_n),
    }


def coverage_pct(n_train: int, corpus_n: int) -> float:
    """The declared coverage percentage, rounded the one canonical way."""
    return round(100.0 * n_train / max(corpus_n, 1), 2)


# -- Declared-vs-observed reconciliation ---------------------------------------


class KeyStats(NamedTuple):
    """Observed facts about the parquet's ``optick_row`` join key.

    Deliberately a plain tuple of integers so the reconciliation logic below is
    stdlib-only and fully unit-testable without pandas/pyarrow/duckdb. The
    parquet read that produces one lives in ``read_key_stats`` and runs only at
    produce time (or from the ``verify`` CLI), where pandas is available.

    ``key_present=False`` means the parquet has no ``optick_row`` column at all
    (the pre-#234 shape, e.g. the 2026-06-14 snapshot); the remaining fields are
    then meaningless and reconciliation stops at that verdict.
    """

    n_rows: int
    key_present: bool
    n_non_null: int = 0
    n_distinct: int = 0
    key_min: int = 0
    key_max: int = 0


class Verdict(NamedTuple):
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


def reconcile(declared: Optional[Dict[str, object]], observed: KeyStats) -> List[Verdict]:
    """
    Audit an artifact's ``activations_coverage`` block against the parquet that
    actually shipped. Returns one :class:`Verdict` per assertion, always in the
    same order, so a caller can report every failure rather than only the first.

    A missing declaration is itself a failure — an artifact that does not say
    what its parquet covers is bug #248 by definition, and no amount of
    inspecting the parquet recovers the intent.
    """
    if not declared:
        return [
            Verdict(
                "coverage_declared",
                False,
                "artifact has no activations_coverage block — the parquet's corpus "
                "coverage is undeclared, so a consumer joining optick_row against "
                "the full index cannot know how many rows it silently drops (ix #248)",
            )
        ]

    verdicts = [Verdict("coverage_declared", True, "activations_coverage block present")]

    split = declared.get("optick_row_split")
    n_train = _as_int(declared.get("n_train"))
    n_val = _as_int(declared.get("n_val"))
    corpus_n = _as_int(declared.get("corpus_n"))
    declared_pct = declared.get("coverage_pct")

    verdicts.append(
        Verdict(
            "split_is_known",
            split == OPTICK_ROW_SPLIT,
            f"optick_row_split={split!r} (expected {OPTICK_ROW_SPLIT!r}; an unknown "
            f"split means these assertions do not describe this artifact)",
        )
    )

    # Additivity: v(train) + v(val) = v(corpus). The declaration must partition.
    verdicts.append(
        Verdict(
            "split_additivity",
            n_train >= 0 and n_val >= 0 and n_train + n_val == corpus_n,
            f"n_train({n_train}) + n_val({n_val}) = {n_train + n_val} "
            f"vs corpus_n({corpus_n})",
        )
    )

    expected_pct = coverage_pct(n_train, corpus_n)
    verdicts.append(
        Verdict(
            "coverage_pct_consistent",
            declared_pct == expected_pct,
            f"declared coverage_pct={declared_pct} vs recomputed {expected_pct}",
        )
    )

    verdicts.append(
        Verdict(
            "coverage_floor",
            expected_pct >= MIN_COVERAGE_PCT,
            f"coverage {expected_pct}% vs floor {MIN_COVERAGE_PCT}% "
            f"({corpus_n - n_train} of {corpus_n} corpus rows absent from the parquet)",
        )
    )

    verdicts.append(
        Verdict(
            "key_present",
            observed.key_present,
            "parquet has an optick_row column"
            if observed.key_present
            else "parquet has NO optick_row column — its rows cannot be joined to "
            "the corpus at all, and a positional join is silently wrong",
        )
    )
    if not observed.key_present:
        return verdicts

    # Declared-vs-observed: the row count the artifact promises must be the row
    # count on disk. This is the assertion activations_coverage cannot make,
    # because it only ever compares the trainer's own in-memory integers.
    verdicts.append(
        Verdict(
            "rows_match_declared",
            observed.n_rows == n_train,
            f"parquet rows={observed.n_rows} vs declared n_train={n_train}",
        )
    )

    # count(DISTINCT) and bool_and both ignore NULLs, so an all-NULL key column
    # would pass uniqueness and range checks. Test for it first and explicitly.
    verdicts.append(
        Verdict(
            "key_non_null",
            observed.n_non_null == observed.n_rows,
            f"non-null optick_row={observed.n_non_null} of {observed.n_rows} rows",
        )
    )

    verdicts.append(
        Verdict(
            "key_unique",
            observed.n_distinct == observed.n_rows,
            f"distinct optick_row={observed.n_distinct} of {observed.n_rows} rows",
        )
    )

    verdicts.append(
        Verdict(
            "key_in_corpus_range",
            observed.key_min >= 0 and observed.key_max < corpus_n,
            f"optick_row range [{observed.key_min}, {observed.key_max}] "
            f"vs corpus [0, {corpus_n - 1}]",
        )
    )

    # The discriminator for the original keying bug class: if optick_row held
    # SPLIT positions rather than CORPUS positions it would be exactly
    # {0..n_train-1} — unique, in range, and the right count, so every other
    # assertion above passes.
    #
    # It is a probabilistic argument, not a proof: a seeded random split *could*
    # legitimately hold out exactly the corpus suffix, leaving correct corpus
    # positions that look like split positions. That has probability
    # 1/C(corpus_n, n_val), which is negligible at production scale but not at
    # toy scale — a 10-row corpus with a 1-row holdout hits it 10% of the time,
    # and this assertion is a hard produce-time failure. Only assert when a
    # legitimate prefix is implausible; below that the check would reject valid
    # artifacts, and a gate that cries wolf gets ignored.
    if n_val > 0 and _prefix_split_is_implausible(corpus_n, n_val):
        verdicts.append(
            Verdict(
                "key_is_corpus_positions",
                observed.key_max >= n_train,
                f"max(optick_row)={observed.key_max}; a {n_train}-row train split "
                f"drawn from {corpus_n} rows must reach past {n_train - 1}, or the "
                f"key holds split positions rather than corpus positions",
            )
        )

    return verdicts


def failures(verdicts: List[Verdict]) -> List[Verdict]:
    """The red verdicts, in report order."""
    return [v for v in verdicts if not v.passed]


def _prefix_split_is_implausible(corpus_n: int, n_val: int, cap: int = 1_000_000) -> bool:
    """Is a legitimate "held out exactly the suffix" split rarer than 1 in ``cap``?

    That happens for exactly one of the ``C(corpus_n, n_val)`` equally likely
    val sets, so the question is whether that binomial exceeds ``cap``. The
    coefficient is astronomically large at production scale, so it is computed
    incrementally and abandoned the moment it passes the cap — never materialised
    in full. At the real corpus (313,047 choose 15,652) two iterations settle it.
    """
    if n_val <= 0 or n_val >= corpus_n:
        return False
    total = 1
    # C(n, k) == C(n, n-k); iterate the cheaper direction.
    for i in range(min(n_val, corpus_n - n_val)):
        total = total * (corpus_n - i) // (i + 1)
        if total >= cap:
            return True
    return total >= cap


def _as_int(value: object) -> int:
    """Coerce a JSON value to int, or -1 for anything that is not one.

    -1 is deliberately out of every legal range, so a malformed declaration
    fails the assertions rather than crashing the reconciler.
    """
    if isinstance(value, bool):
        return -1
    return value if isinstance(value, int) else -1


# -- Produce-time / standalone adapter (the only pandas-touching code) ---------


def read_key_stats(parquet_path) -> KeyStats:
    """Extract :class:`KeyStats` from a feature_activations parquet.

    Reads only the ``optick_row`` column — the 1024 activation columns are the
    56 MB; the key is a few hundred KB.
    """
    import pandas as pd  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    parquet_file = pq.ParquetFile(parquet_path)
    n_rows = parquet_file.metadata.num_rows
    if "optick_row" not in parquet_file.schema_arrow.names:
        return KeyStats(n_rows=n_rows, key_present=False)

    col = pd.read_parquet(parquet_path, columns=["optick_row"])["optick_row"]
    non_null = col.dropna()
    return KeyStats(
        n_rows=n_rows,
        key_present=True,
        n_non_null=int(non_null.size),
        n_distinct=int(non_null.nunique()),
        key_min=int(non_null.min()) if non_null.size else 0,
        key_max=int(non_null.max()) if non_null.size else 0,
    )


class Unevaluatable(Exception):
    """The reconciliation could not be run at all.

    Distinct from a red verdict on purpose. "The parquet contradicts the
    declaration" and "I could not open the parquet" are different facts, and
    collapsing them is how a coverage gap goes quiet — the caller would read a
    fresh checkout (where the gitignored parquet is simply absent) as a
    contradiction, or worse, learn to ignore the failure.
    """


def reconcile_snapshot(snapshot_dir) -> List[Verdict]:
    """Reconcile an on-disk snapshot directory (artifact JSON + parquet).

    Raises :class:`Unevaluatable` when the inputs or the parquet libraries are
    missing. A *missing artifact* is NOT unevaluatable — an absent declaration
    is precisely the ix #248 finding, so it is a red verdict.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    snapshot = Path(snapshot_dir)
    artifact_path = snapshot / "optick-sae-artifact.json"
    if not artifact_path.exists():
        raise Unevaluatable(
            f"{artifact_path} does not exist — there is no declaration to reconcile"
        )
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Unevaluatable(f"cannot parse {artifact_path}: {exc}") from exc
    declared = artifact.get("activations_coverage")

    links = artifact.get("links") or {}
    parquet_path = snapshot / links.get(
        "feature_activations_parquet", "feature_activations.parquet"
    )
    if not parquet_path.exists():
        raise Unevaluatable(
            f"{parquet_path} does not exist — the declaration cannot be reconciled "
            f"against anything. feature_activations.parquet is gitignored, so this "
            f"is expected on a fresh checkout."
        )

    try:
        observed = read_key_stats(parquet_path)
    except ImportError as exc:
        raise Unevaluatable(
            f"the parquet reader is unavailable ({exc}); install pandas + pyarrow"
        ) from exc
    except OSError as exc:
        raise Unevaluatable(f"cannot read {parquet_path}: {exc}") from exc

    return reconcile(declared, observed)


# Exit codes, mirrored by the Rust `verify` subcommand. Kept distinct so a
# caller can tell "this snapshot is wrong" from "I could not check it".
EXIT_OK = 0
EXIT_RECONCILIATION_FAILED = 1
EXIT_NOT_EVALUATABLE = 5


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Reconcile an optick-sae snapshot's declared activation "
        "coverage against the parquet that actually shipped (ix #248)."
    )
    parser.add_argument(
        "snapshot", help="snapshot directory containing optick-sae-artifact.json"
    )
    args = parser.parse_args(argv)

    try:
        verdicts = reconcile_snapshot(args.snapshot)
    except Unevaluatable as exc:
        print(f"NOT VERIFIED: {exc}")
        print("\nThis is not a pass and not a contradiction — nothing was checked.")
        return EXIT_NOT_EVALUATABLE

    for verdict in verdicts:
        print(verdict)

    bad = failures(verdicts)
    if bad:
        print(f"\nRECONCILIATION FAILED: {len(bad)} of {len(verdicts)} assertions red.")
        return EXIT_RECONCILIATION_FAILED
    print(f"\nreconciliation ok: {len(verdicts)} assertions green.")
    return EXIT_OK


if __name__ == "__main__":
    import sys

    sys.exit(main())
