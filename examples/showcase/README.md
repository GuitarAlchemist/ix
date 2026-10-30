# ix showcase

A 9-step walkthrough of the ix ecosystem — 43 registry-backed skills, the
7-verb CLI, ix.yaml pipelines with governance gates, hexavalent beliefs, and
the 14-persona Demerzel submodule — all executed end-to-end from one
5-stage pipeline.

## Run it

```bash
# from the workspace root
cargo build -p ix-skill                 # once
bash examples/showcase/demo.sh
```

The script creates a fresh temp workspace, copies `pipeline.yaml` into it,
walks through every core command, and cleans up on exit. It never touches
state outside its temp dir.

## What you get

### 1. Discovery

```
$ ix list domains --format table
  count   29
  domains [29 items]
```

Every `#[ix_skill]`-annotated function across the workspace is visible.

### 2. Introspection

```
$ ix describe skill stats
  crate            ix-agent
  doc              Compute statistics (mean, std, min, max, median) ...
  domain           math
  governance_tags  [2 items]
  inputs           [1 items]
  outputs          [1 items]
  schema           {3 keys}
```

### 3. Ad-hoc invocation

```
$ ix run stats --input '{"data":[1,2,3,4,5,6,7,8,9,10]}'
  count     10
  max       10.0
  mean      5.5
  median    5.5
  min       1.0
  std_dev   2.8722813232690143
  variance  8.25
```

### 4–5. Pipeline: validate → dag → run

The showcase `pipeline.yaml` is a 5-stage diamond DAG:

```
     baseline_stats     signal_spectrum     primality     gcd_test
        (stats)             (fft)        (number_theory) (number_theory)
             \                 |                |              /
              \                |                |             /
               └───────────────┴────────────────┴────────────┘
                                      │
                                      ▼
                             compliance_audit
                            (governance.check)
```

The four root stages run in parallel (parallel_depth=2), then
`compliance_audit` fires once after they all complete:

```
$ ix pipeline run --json
{"event":"start","parallel_depth":2,"path":"ix.yaml","stages":5}
{"cache_hit":false,"duration_ms":0,"event":"stage_complete","stage":"gcd_test"}
{"cache_hit":false,"duration_ms":0,"event":"stage_complete","stage":"compliance_audit"}
{"cache_hit":false,"duration_ms":0,"event":"stage_complete","stage":"signal_spectrum"}
{"cache_hit":false,"duration_ms":0,"event":"stage_complete","stage":"primality"}
{"cache_hit":false,"duration_ms":0,"event":"stage_complete","stage":"baseline_stats"}
{"cache_hits":0,"duration_ms":0,"event":"done"}
```

### 6. Reproducibility via ix.lock

The run writes `ix.lock` next to `ix.yaml` with content-addressed hashes
of every stage's args + dep list + skill:

```yaml
schema: ix-lock/v1
generated: 2026-04-05T20:34:33Z
stages:
  baseline_stats:
    skill: stats
    args_hash: fnv1a64:fb47c8ef0ea3d263
    deps: []
    duration_ms: 0
    cache_hit: false
  compliance_audit:
    skill: governance.check
    args_hash: fnv1a64:dadd7fd83ba8e201
    deps: [baseline_stats, signal_spectrum, primality, gcd_test]
    ...
```

### 7. Governance with hexavalent exit codes

```
$ ix check action 'update the README'
  verdict   U                        # exit 2 — Unknown: no rule matched,
  exit_code 2                        # which is not approval

$ ix check action 'delete production database'
  verdict                     D     # exit 3 — Doubtful, hold action
  exit_code                   3
  dangerous_keywords_matched  true
```

Exit codes map hexavalent values: `0=T · 1=P · 2=U · 3=D · 4=F · 5=C`.
Wire these into CI branches for gate-based deployment decisions.

### 8. Belief state

```
$ ix beliefs set deployment 'Signal pipeline passes all gates' --truth P --confidence 0.8
  action  set
  key     deployment
  path    state/beliefs/deployment.belief.json

$ ix beliefs snapshot 'post-pipeline-run'
  action           snapshot
  captured_beliefs 1
  path             state/snapshots/2026-04-05-post-pipeline-run.snapshot.json
```

Beliefs land in `state/beliefs/*.belief.json`; snapshots roll them up
into `state/snapshots/{date}-{slug}.snapshot.json` per the Demerzel
reconnaissance schema.

### 9. Personas

```
$ ix list personas --format table
  count     14
  personas  [14 items]
  source    .../governance/demerzel/personas
```

14 Demerzel personas: `default`, `demerzel`, `seldon`, `skeptical-auditor`,
`rational-administrator`, `reflective-architect`, `kaizen-optimizer`,
`recovery-agent`, `validator-reflector`, `virtuous-leader`,
`system-integrator`, `communal-steward`, `critical-theorist`,
`convolution-agent`.

Each has a behavioral test in `governance/demerzel/tests/behavioral/`.

## Visual editor

The same 43 skills are reachable from the graphical editor:

```bash
cargo run -p ix-demo
# → click "Pipeline" in the top nav (Pipelines group)
```

In the visual editor you can:
- **Right-click** the canvas to add typed nodes (CSV Read, K-Means, Policy
  Gate, Belief, FFT, …) from the search-filtered palette
- **Drag pins** to connect nodes — incompatible socket types refuse
- **Click** a belief node's hexavalent quadrants (T/P/U/D/F/C) to set its state
- **Run** the graph wire-aware: each node gathers its upstream outputs
  and invokes its registry skill; status dots appear on each node header
- **Validate** to flag ML nodes missing an upstream PolicyGate
- **Export YAML** → produces an ix.yaml compatible with `ix pipeline run`
- **Import YAML** → loads an ix.yaml back as a visual graph of Skill nodes
- **Run ix.yaml** → executes the file via ix-pipeline and shows per-stage
  outputs in a results panel

## Files

- `pipeline.yaml` — the 5-stage signal + governance showcase DAG
- `ml-classification.yaml` — 3 classifiers benchmarked + deployment gate
- `signal-chain.yaml` — FFT + Lyapunov + envelope stats with constitutional review
- `demo.sh` — the walkthrough script
- `README.md` — this file

## Run the other showcases

```bash
# ML classifier bakeoff
cp examples/showcase/ml-classification.yaml /tmp/ix.yaml
(cd /tmp && IX_ROOT=$PWD/../.. \
  C:/Users/spare/source/repos/ix/target/debug/ix.exe pipeline run)

# Signal + chaos chain
cp examples/showcase/signal-chain.yaml /tmp/ix.yaml
(cd /tmp && IX_ROOT=$PWD/../.. \
  C:/Users/spare/source/repos/ix/target/debug/ix.exe pipeline run)
```

Or drop either file into any directory and run `ix pipeline run` from there
(setting `IX_ROOT` to the workspace root).
