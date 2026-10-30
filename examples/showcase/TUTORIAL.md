# Build your first ix pipeline in 10 steps

A narrative walkthrough that takes you from "cold repo" to "production
pipeline with governance gate" — building a customer-churn predictor
from scratch. Every command is runnable; copy and paste as you go.

Prereq: you've run `cargo build -p ix-skill` so the `ix` binary exists.

---

## Step 1: Find the skill you need

```bash
ix list skills --domain supervised --format table
```

The registry holds 43 skills across 29 domains. Filtering by `supervised`
shows the models that can do classification. For a churn predictor you
want one of these.

```bash
ix list skills --query regression --format json
```

`--query` fuzzy-searches across skill names + doc strings. Use it when
you know what the skill *does* but not its exact name.

---

## Step 2: Introspect the skill you picked

```bash
ix describe skill random_forest --format json
```

You'll see the skill's full JSON schema: required fields (`x_train`,
`y_train`, `x_test`), optional tuning knobs (`n_trees`, `max_depth`),
and its governance tags (`empirical`). You now know exactly what JSON
to hand it.

---

## Step 3: Run it ad-hoc to verify

Before building a pipeline, confirm the skill works with a minimal
payload:

```bash
ix run random_forest --input '{
  "x_train": [[0, 0], [0, 1], [1, 0], [1, 1]],
  "y_train": [0, 1, 1, 0],
  "x_test":  [[0.5, 0.5], [0.1, 0.9]],
  "n_trees": 10,
  "max_depth": 4
}'
```

You'll get back predictions + per-tree probabilities. Sanity-check
passed.

---

## Step 4: Scaffold a pipeline

```bash
mkdir churn-demo && cd churn-demo
ix pipeline new churn
cat ix.yaml
```

`ix pipeline new` writes a minimal `ix.yaml` with one `stats` stage.
You'll replace that with your actual churn pipeline next.

---

## Step 5: Edit `ix.yaml` into a real pipeline

Open `ix.yaml` in your editor and replace the contents with:

```yaml
version: "1"

stages:

  # Historical customer activity — tenure_days, monthly_value, support_tickets
  activity_profile:
    skill: stats
    args:
      data: [120, 45, 890, 230, 15, 567, 89, 340, 12, 720]

  # Train the churn classifier on labeled historical data.
  # Features: [tenure_months, monthly_spend]. Label: 1 = churned.
  churn_model:
    skill: random_forest
    args:
      x_train: [[24, 150], [6, 80], [36, 200], [3, 50], [18, 120], [2, 40], [48, 250], [1, 30]]
      y_train: [0, 1, 0, 1, 0, 1, 0, 1]
      x_test: [[12, 100], [4, 60], [30, 180]]
      n_trees: 20
      max_depth: 5

  # Similar-customer cluster discovery for retention targeting.
  segments:
    skill: kmeans
    args:
      data: [[24, 150], [6, 80], [36, 200], [3, 50], [18, 120], [2, 40], [48, 250], [1, 30]]
      k: 3
      max_iter: 50

  # Governance gate: reviewing the retention-outreach plan against the
  # Demerzel constitution. Article 1 (Truthfulness) + Article 5
  # (Non-Deception) apply — contacting users based on churn prediction
  # needs honest framing.
  retention_outreach_review:
    skill: governance.check
    args:
      action: "send personalized retention offers to customers predicted to churn within 30 days"
    deps: [activity_profile, churn_model, segments]
```

---

## Step 6: Validate before running

```bash
ix pipeline validate
```

This parses the YAML, checks every `skill:` against the registry,
catches cycles, and verifies `deps:` reference existing stages. No
compute yet — just structural validation.

```bash
ix pipeline dag --format table
```

Shows the execution levels:

```
  levels          [2 items]     # roots + governance leaf
  parallel_depth  2
  path            ix.yaml
  total_stages    4
```

---

## Step 7: Run the pipeline

```bash
export IX_ROOT="$PWD/../.."   # point at the workspace root
ix pipeline run --json
```

NDJSON event stream shows each stage starting + completing in real
time. When done, `ix.lock` has been written next to `ix.yaml` with
FNV-1a content hashes of each stage's args + deps + skill — your
reproducibility manifest.

```bash
cat ix.lock
```

Every future run can diff against this lock file to detect drift.

---

## Step 8: Ask the constitutional gate

The `retention_outreach_review` stage returned a verdict based on
Demerzel's constitution. Let's also run a standalone constitutional
check against a higher-stakes variant of the same action:

```bash
ix check action 'delete customer records after 30 days of inactivity'
```

Exit code **3** = verdict **D** (Doubtful) — the keyword "delete"
triggered the dangerous-action detector. This would block merge or
deployment in a CI pipeline.

```bash
ix check action 'send a welcome email to new signups'
```

Exit code **2** = verdict **U** (Unknown) — no rule matched. The check only
recognizes English keywords, so a match-free action is not approved: a human
or a more specific policy has to decide.

---

## Step 9: Record the deployment decision as a belief

```bash
ix beliefs set churn_model_v1 \
    'Churn model v1 ready for A/B rollout' \
    --truth P --confidence 0.72
```

The belief lands in `state/beliefs/churn_model_v1.belief.json` with
hexavalent `P` (Probable — leans true, not yet verified) at confidence
0.72. This is below the 0.9 "autonomous" threshold from the alignment
policy, so it'll need human review before full rollout.

---

## Step 10: Capture a snapshot

```bash
ix beliefs snapshot 'churn-v1-candidate-rollout'
```

Every belief in `state/beliefs/` gets rolled up into a timestamped
snapshot at `state/snapshots/{YYYY-MM-DD}-churn-v1-candidate-rollout.snapshot.json`.
This is the audit-replay artifact — a future analyst can see exactly
what you believed and how confident you were at this moment.

---

## What you built

In 10 commands you've:

1. **Discovered** skills via `list` + `query`
2. **Introspected** one skill's schema via `describe`
3. **Invoked** it directly via `run`
4. **Scaffolded** a pipeline with `pipeline new`
5. **Authored** a 4-stage diamond DAG in `ix.yaml`
6. **Validated** structure without running compute
7. **Visualized** execution levels via `dag`
8. **Executed** the pipeline with `ix.lock` auto-generation
9. **Gated** deployment via constitutional check
10. **Persisted** the decision as a hexavalent belief + snapshot

All 43 registered skills work the same way. Swap `random_forest` for
`gradient_boosting`, `kmeans` for `dbscan`, `stats` for `fft` — the
pipeline machinery doesn't care.

## Next steps

- **Visual author** the same pipeline: `cargo run -p ix-demo`,
  click the Pipeline tab, right-click canvas → add nodes → connect
  pins. Export back to YAML via the Export YAML button.
- **Chain stages** with `{"from": "upstream_stage.field"}` references —
  see `examples/showcase/advanced/chained-spectrum.yaml`.
- **Schedule** this pipeline as a watchdog:
  `bash examples/showcase/advanced/monitoring-loop.sh` adapts easily
  to any pipeline.
- **Federate**: add a stage that calls a TARS or GA skill via the
  `federation.discover` + federation-prefixed skill names.
