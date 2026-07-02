# `ix_dev_rhythm` — a Markov / HMM model of IX's own development

A dogfood demo with a sequential flavour: the dataset is **IX's git history**, and the
tools are IX's own Markov chain + Hidden Markov Model. It reads each commit's
conventional-commit *type* (feat / fix / docs / chore / refactor / test / other) in
chronological order and asks what structure that 539-symbol sequence has.

> _Version française : [`docs/fr/pas-a-pas/rythme-dev.md`](../fr/pas-a-pas/rythme-dev.md)._

```bash
cargo run -p ix-duck --example ix_dev_rhythm --features duck
```

## Three stages

**1. Markov chain.** A first-order transition matrix over commit types feeds
`ix_graph::markov::MarkovChain`; its **stationary distribution** is the long-run mix:

```text
feat 45%   chore 17%   docs 13%   fix 10%   other 6%   refactor 5%   test 4%
P(fix | feat) = 6.8%   vs base P(fix) = 9.8%
```

A fix is *less* likely right after a feat than at baseline — because feats **cluster**
(during a feature push the next commit is usually another feat), so they crowd out
everything else in the immediately-following slot.

**2. Validation — does order carry structure?** If commit types were i.i.d., the next
type would be no more predictable than the marginal. The statistic is the next-type
accuracy of "predict `argmax P(next | current)`"; the null **shuffles the sequence**
(destroying order, keeping the marginal) and re-scores, 2 000 times:

```text
Markov next-type accuracy = 51.3%   (always-predict-feat base = 47.7%)
permutation p = 0.0005  → real sequential structure
```

The lift over base is **modest (~3.6 pts)** but well outside the null — IX's commit stream
genuinely has first-order structure, it isn't a bag of independent types.

**3. Hidden phases — learned, not assumed.** A 2-state HMM is **trained by Baum-Welch**
(`ix_graph::hmm`, no hand-set emissions), then the most-likely hidden-state path is decoded
by the **`ix_viterbi`** UDF on the bench:

```text
phase A (69% of history) emits: feat 64%, docs 12%, fix 7%     ← feature-building
phase B (31% of history) emits: chore 40%, docs 16%, fix 15%   ← maintenance
recent rhythm (last 40 commits): AAAAAAAAAA BBBBBBBBBBBBBBBBBBBB AAAAAAAAAA
```

The model recovers two interpretable development modes from the raw stream — a
**feature-building** phase (feat-dominated) and a **maintenance** phase (chore-dominated:
fleet-status / snapshot bot commits + fixes cluster here) — and the Viterbi timeline shows
them alternating, with a recent maintenance burst bracketed by feature work.

## What's IX-native here

| Stage | IX primitive |
|---|---|
| transition + stationary | `ix_graph::markov::MarkovChain` |
| hidden-phase training | `ix_graph::hmm::HiddenMarkovModel::baum_welch` |
| phase decode (on the bench) | `ix_viterbi` UDF |
| significance | permutation null (same pattern as the mesh / code-health demos) |

## Scope and caveats

- **Advisory, illustrative.** A description of how IX has been developed, not a prescription.
- Commit-type parsing is a **heuristic** on the subject line; merge/non-conventional
  commits fall into `other`.
- The validation is **in-sample** (the permutation null is also in-sample, so the
  comparison is fair) and the effect, while significant, is small.
- **2 states** is a choice, not a discovery; more states would split the phases finer.
  Baum-Welch finds a local optimum from the seeded start.
- Runs anywhere `git` + the repo are present — the data is the history itself.
