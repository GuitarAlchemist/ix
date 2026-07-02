# Methodology Guard Usefulness Metrics

Date: 2026-07-01

Status: draft / advisory

## Purpose

Define IX-owned metrics for deciding whether Methodology Guard and abstraction lifecycle signals are useful enough to promote beyond pilot/advisory mode.

## Principle

```text
IX measures.
Demerzel governs.
.github executes.
TARS reasons.
Human decides.
```

## Candidate metrics

### false_positive_rate

```text
count(findings marked not useful) / count(total findings reviewed)
```

Purpose:

```text
Prevent strict gates from being enabled while findings are noisy.
```

### review_friction_delta

```text
review_time_after_guard - review_time_before_guard
```

Purpose:

```text
Detect whether the guard adds bureaucracy instead of clarity.
```

### missing_block_resolution_rate

```text
count(missing blocks later fixed) / count(missing block findings)
```

Purpose:

```text
Measure whether findings lead to better issues/PRs.
```

### useful_signal_rate

```text
count(findings used in human/TARS/Demerzel decision) / count(total findings)
```

Purpose:

```text
Measure whether signals are decision-relevant.
```

### abstraction_reuse_count

```text
number of real consumer repos using a promoted abstraction
```

Purpose:

```text
Prevent org-level promotion before real reuse exists.
```

### process_weight_ratio

```text
process_artifacts_changed / product_runtime_artifacts_changed
```

Purpose:

```text
Detect when governance/process work is growing faster than product/runtime value.
```

## Promotion thresholds — initial proposal

```text
false_positive_rate <= 0.20
useful_signal_rate >= 0.50
at least two real consumer repos for org-level promotion
review_friction_delta not strongly positive
```

These thresholds are provisional and should not become policy until calibrated.

## Non-goals

```text
Do not reduce usefulness to one score.
Do not optimize for passing the guard instead of improving work quality.
Do not let metrics override human judgment.
Do not create Goodhart incentives around issue-template compliance.
```

## Related

- GuitarAlchemist/.github#28
- GuitarAlchemist/.github#30
- GuitarAlchemist/Demerzel#588
- GuitarAlchemist/Demerzel#592
