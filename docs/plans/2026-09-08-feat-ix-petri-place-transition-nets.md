# Petri nets in IX — `ix-petri`, and the two engines deliberately not built

- **Date**: 2026-09-08
- **Status**: shipped (this PR)
- **Reversibility**: **two-way door.** A new `experimental`-tier crate with no
  stable-tier API, no schema hash, no on-disk artifact and no Galactic Protocol
  contract. One MCP tool (`ix_petri_analyze`) and one dependency edge
  (`ix-agent -> ix-petri`). Deleting the crate is a `git rm` plus removing a
  parity entry.
  **Revisit / delete-by trigger**: see §6.
- **One-way-door check**: the only externally-visible commitment is the *shape*
  of the PNML subset the reader accepts, and that shape is ISO/IEC 15909-2's,
  not one this repository invented. Nothing here freezes an IX format.
- **Maturity tier**: `experimental` in `crate-maturity.toml`. Chosen, not
  copied — see §8.

## 1. The brief was three features; this ships one

The request named **deterministic state machines**, **Petri nets** and **rules
engines**. That is three large features, and the repository has a documented
opinion about shipping surface without an adopter: `ix#299` recorded eight I/O
backends conforming to a trait with **zero implementors**, under a module doc
asserting the opposite. (That specific instance was repaired by PR #312, which
merged into `main` while this branch was open — cited here as the precedent it
set, not as a live defect. The issue is still open at time of writing.) The
gap matrix (`docs/research/tars-v1-advanced-math-ix-gap-matrix.md`) closes on
the rule
"no esoteric algorithms without tests and a consumer".

So this PR delivers **one** feature to usable depth and writes down what the
other two would need. §5 is that write-up. A working Petri net with deadlock
detection is worth more than three half-built engines.

## 2. Survey — verified, not assumed

| Claim | Verified how | Result |
|---|---|---|
| No Petri net or GRAFCET in `crates/` | `grep -rliE "petri\|grafcet" crates/` | Confirmed: no hits. Also none in any `.md` / `.json` / `.jsonl` in the repo. |
| `ix-graph` is probabilistic, not deterministic FSM | read `graph.rs`, `markov.rs`, `hmm.rs`, `state_space.rs`, `routing.rs` | Confirmed. `state_space` is MDP-shaped; there is no deterministic FSM anywhere. |
| `ix_pipeline::dag::Dag` cannot express cycles | `crates/ix-pipeline/src/dag.rs` | Confirmed — `add_edge` rejects cycles. This is the load-bearing gap. |
| `ix-fuzzy` / `ix-grammar` are the rules-engine-adjacent surface | read both crates' module lists | Confirmed, and it is why §5 recommends *not* building a third evaluator. |
| PNML is ISO/IEC 15909-2 with a published RELAX NG grammar | fetched <https://www.pnml.org/> and the grammar files | Confirmed. Part 1 (2019) semantics, Part 2 (2011) syntax, Part 3 (2021) extensibility. Three net types: High-level, Symmetric, Place/Transition. |
| A conforming P/T reader is small | read `ptnet.pntd` (91 lines), `pnmlcoremodel.rng` (657), `conventions.rng` (52) | The P/T *type definition* adds exactly two labels — `initialMarking` on a place, `inscription` on an arc. The semantic core of the document model is eight element names. The bulk of the 657 lines is presentation (graphics, fill, line, font, positions), all semantically ignorable. |

**No workspace XML crate exists.** `quick-xml`, `roxmltree`, `xml-rs`,
`serde-xml`, `minidom` appear in no `Cargo.toml`.

## 3. What ships

`crates/ix-petri`, dependencies `serde` + `serde_json` + `thiserror` only.

- **`net`** — `PetriNet`, weighted arcs, `Marking`, the firing rule, and
  validation that rejects duplicate ids, place/transition id collisions,
  non-bipartite arcs, zero inscriptions and duplicate arcs.
- **`analysis`** — reachability enumeration and five properties: deadlock
  freedom, boundedness, quasi-liveness (dead transitions), L4-liveness and
  reversibility. Liveness and reversibility read off an *iterative* Tarjan SCC
  (recursion would blow the stack on exactly the nets worth analysing).
- **`pnml`** — a reader for the Place/Transition subclass.
- **`xml`** — a ~200-line strict tokenizer, so no XML dependency is added.
- **`models`** — the dining philosophers, both protocols, as a known-answer
  oracle.
- **`examples/analyze_pnml.rs`** — point it at a `.pnml` file.
- **`ix_petri_analyze`** — the MCP tool, without which the crate is unreachable
  from an agent loop (the gap matrix's row-A3 failure mode).

### The three decisions the brief asked for in writing

1. **Read PNML, do not write it.** Reading buys independent verification: a net
   authored elsewhere can be pushed through IX's analyser. Writing buys the
   mirror image — IX's models checked by someone else's tool — but an emitter
   must satisfy *other* tools' readers and nothing in this repository can verify
   that it does. Emitting unvalidated PNML would be a claim, not a capability.
   Deferred with its entry conditions recorded in `crates/ix-petri/src/pnml.rs`.
2. **Place/Transition nets only.** Deadlock and boundedness are meaningful
   there and the type definition is two labels. Symmetric and High-level nets
   carry a sort system and term algebra; a marking becomes a multiset of
   structured tokens and none of the enumeration applies unchanged. A document
   declaring one is **rejected by name**, not silently misread.
3. **No XML dependency.** ~200 lines of tokenizer instead of a new dependency
   family in a workspace that has none. The tokenizer rejects `<!DOCTYPE>`
   outright — no entity declarations means no billion-laughs amplification on
   files received from other tools. Stated in the module docs rather than left
   implicit.

## 4. The consumer, stated without inflation

**What the crate actually answers today**: `crates/ix-petri/tests/worktree_pump.rs`
models the hazard `CLAUDE.md` warns about in every session preamble — the git
stash stack is shared across the main checkout and all worktrees, and a pump
lane holds both a working tree and that one shared stack. The tests establish by
enumeration that opposite acquisition orders can wedge (with the witness
`L0 takes tree -> L1 takes stash`), that one contrarian lane is enough, and that
a canonical acquisition order removes the deadlock for every lane count tested.
That is a concrete, actionable answer to a question the repository already
documents as a hazard.

**What is reach rather than adoption**: `ix_petri_analyze`. The tool exists so
an agent *can* call it. Nothing calls it on a schedule today, and this PR does
not pretend otherwise — hence the delete-by trigger below rather than a claim of
adoption.

### Independent verification

The reader and the analyser were checked against
`https://www.pnml.org/version-2009/examples/philo.pnml` — the dining-philosophers
net published by the standard's own reference site, authored by nobody here:

```
net i943123747 (30 places, 30 transitions)
  states 729  edges 3402
  DEADLOCK x2
    marking: WAIT_LEFT_FORK_1=1 ... WAIT_LEFT_FORK_6=1
    witness: TAKE_RIGHT_1_FORK_4 -> ... -> TAKE_RIGHT_1_FORK_5
    marking: WAIT_RIGHT_FORK_1=1 ... WAIT_RIGHT_FORK_6=1
    witness: TAKE_LEFT_1_FORK_6 -> ... -> TAKE_LEFT_1_FORK_4
  1-bounded / not live / not reversible
```

Six philosophers, each able to reach for either fork first, therefore **two**
symmetric deadlocks — each reached by six same-handed grabs. That is the
textbook answer for this net, produced from a file this repository did not
write. The file is **not vendored** (pnml.org states no licence); the command
that reproduces it is in the guide.

## 5. The two features not built, and what each would need

### Deterministic finite state machines

**Verdict: do not build yet — no consumer named.** The obvious candidates were
checked. `ix-agent-core`'s `MiddlewareVerdict` (`Continue`/`Block`/`Transform`/
`Retry`/`Escalate`) is a verdict enum, not a machine with a state that persists
across calls; `ix-approval`'s tiering is a pure classifier; `ix-session` is an
append-only log whose projections are folds, not transitions.

Entry conditions, in order:
1. Name a component that **persists** a state across calls and whose legal
   transitions are currently enforced by scattered `if` statements.
2. Show at least two such components, so the abstraction has two adapters and is
   a real seam rather than the hypothetical one `ix#299` warns about.
3. Only then: `ix-fsm` with transition-table validation, unreachable-state and
   dead-transition detection, and — the part that would earn it —
   **determinism checking** (no two outgoing transitions on the same input).
   Note the overlap: a 1-safe Petri net with one token *is* a DFA, so a cheap
   first move is a `PetriNet -> Dfa` view rather than a new crate.

### Rules engine

**Verdict: do not build. It would be a shallow module over shipped surface.**
`ix-grammar` already ships `ebnf`, `abnf`, `weighted` and `constrained`
evaluation; `ix-fuzzy` ships membership functions and inference; `ix-types::Hexavalent`
is the canonical truth algebra; `ix-assumption-graph` already does claim
fusion. A third evaluator would duplicate the grammar crate's evaluation, which
is precisely the "shallow module over a shipped one" the brief warned against.

If a rules engine is genuinely wanted later, the honest shape is **not a new
evaluator** but a *thin* Rete-style working-memory index in front of the
existing ones — and its entry condition is a measured one: an existing consumer
whose rule evaluation is demonstrably re-scanning facts it has already matched.
Until someone can point at that profile, this stays unbuilt.

## 6. Delete-by trigger

Per the gap matrix's discipline on speculative surface:

> If, by **2027-03-08** (six months), `ix_petri_analyze` has no caller outside
> `ix-petri`'s own tests **and** no second net has been modelled in the
> repository, delete the MCP tool and demote the crate to a `tests/`-only
> fixture, or delete it outright.

The worktree-pump model is the first net and already earns the analyses; the
trigger is about whether a *second* consumer ever appears.

## 7. Not implemented (each for want of a consumer, not difficulty)

- Inhibitor arcs, place capacities, priorities, timed nets — none is in the PNML
  P/T type definition, and adding them would make IX's nets unreadable by
  conforming tools, forfeiting the reason for using the standard.
- Writing PNML (§3.1).
- Coloured / high-level nets (§3.2).
- Structural analysis without enumeration — P/T-invariants, siphons, traps.
  These decide boundedness and some liveness questions on nets far too large to
  enumerate. Worth adding the day a net in this repository exceeds
  `max_states`; not before.

## 8. Why `experimental` and not another tier

`crate-maturity.toml` is the workspace's answer to "how stable is this surface",
and `ix skill stable-surface` only guards crates at tier `stable`: it hashes
every `pub`-prefixed declaration line and **fails CI** when that hash moves.
Everything below `stable` is unguarded. So the tier is not a label, it is the
choice between a semver commitment and a two-way door.

- **`stable`** would be dishonest and self-contradictory. `Verdict`,
  `Analysis`, `Bounds`, the builder methods and the accepted PNML subset are all
  expected to move as a second consumer appears — that is what §5's entry
  conditions and §6's delete-by trigger *say*. You cannot promise a crate may be
  deleted in six months and simultaneously promise its public API. Concretely,
  adding one `pub fn` (a `PetriNet -> Dfa` view, a PNML writer, a P-invariant
  analysis) would trip the hash guard and fail CI on a crate nobody outside this
  repo consumes yet.
- **`beta`** ("feature-complete, API may change between minor versions") claims
  feature-completeness this does not have: no PNML writer, no structural
  analysis, no reference-node resolution.
- **`internal`** ("tooling/infra, not intended for direct external use") is the
  tier of `ix-approval`, `ix-agent-core`, `ix-fuzzy` — harness plumbing. It is
  wrong here in the one direction that matters: `ix-petri` is *specifically*
  meant to be reached from outside, both through `ix_petri_analyze` and through
  PNML, which exists precisely so external tools can exchange nets with it.
- **`experimental`** ("research-grade, novel math, no stability guarantees",
  downstream-safe: **No**) states exactly the contract this PR claims. It is the
  tier of the sibling algorithm crates — `ix-topo`, `ix-ktheory`, `ix-category`,
  `ix-evolution`, `ix-duck` — and it keeps the door two-way.

**Promotion condition**, so the tier is not permanent by default: promote to
`beta` when a second consumer exists *and* the `Verdict` / `Analysis` shapes
have survived it unchanged. Until then, a caller outside this repo should treat
the JSON as unstable.
