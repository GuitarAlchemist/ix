# Petri nets in IX — when to reach for one, and when not to

**Purpose:** `ix-petri` models systems with **cycles, concurrency and resource
contention at once**, and answers behavioural questions about them: can this
deadlock, can this queue grow without bound, can this transition ever fire.
This page says when that is the right tool and when an existing IX crate is.

**Audience:** humans and Claude Code sessions choosing between `ix-petri`,
`ix-pipeline`, `ix-graph` and `ix-fuzzy`.

**Français :** [docs/fr/guides/reseaux-de-petri-dans-ix.md](../fr/guides/reseaux-de-petri-dans-ix.md)

---

## 1. Choose the right crate first

Most "state machine" tasks in IX are already covered. Reach for `ix-petri` only
when the row below is genuinely yours.

| Your question | Use | Why |
|---|---|---|
| "What order do these tasks run in? What is the critical path?" | `ix_pipeline::dag::Dag` | A DAG, with topological sort, parallel levels and critical path. Rejects cycles by construction. |
| "How likely is the system to be in state X?" | `ix_graph::markov`, `hmm`, `state_space` | Probabilistic state evolution — distributions, Viterbi, stationary behaviour. |
| "How true is this claim?" | `ix_fuzzy`, `ix_types::Hexavalent` | Degrees of truth, not control flow. |
| "Does this string parse / how do I generate one?" | `ix_grammar` (`ebnf`, `abnf`, `weighted`, `constrained`) | Derivation, not concurrency. |
| **"Can these two lanes wedge each other? Can this queue grow forever? Can this step ever run?"** | **`ix-petri`** | Cycles + concurrency + contention, and the answer is a *possibility*, not a probability. |

The dividing line is sharp and worth stating twice: **a DAG cannot express a
resource that is returned to a pool and taken again**, because that is a cycle
and `Dag::add_edge` refuses it. If your model has a lock, a worker pool, a
buffer with a finite number of slots, or a lane that returns to `READY`, a DAG
is the wrong shape and no amount of care will make it the right one.

Equally: if you only need to *represent* places and transitions and never ask a
behavioural question about them, you do not need this crate. A `HashMap` will do.
The properties are the product.

## 2. What it computes

`ix_petri::analyze` enumerates the reachable markings and reads five properties
off the resulting graph. Every one is `holds`, `fails` **with a witness**, or
`unknown` — never a guess.

| Property | `fails` gives you |
|---|---|
| **deadlock-free** | every dead marking, each with the *shortest* firing sequence that reaches it |
| **bounded** | a pair `m < m'` with `m'` reachable from `m`, and the sequence between them — repeat it and the surplus compounds |
| **quasi-live** | the transitions enabled in no reachable marking (steps you wrote that can never run) |
| **live** (L4) | the transitions absent from some terminal strongly-connected component |
| **reversible** | whether the initial marking is reachable from everywhere |

### The honesty boundary

Enumeration is bounded by `Limits::max_states` (default 50 000). That bound is
the point at which the analysis stops claiming things:

- **Exhausted** — results are exact.
- **Truncated with an unboundedness witness** — the witness *proves* the net is
  unbounded, so every property that ranges over the (now infinite) state space
  becomes `unknown`.
- **Truncated with no witness** — nothing is claimed at all.

A deadlock found during a truncated run is still reported: a witness sequence is
a positive existence proof and truncation cannot invalidate it. The *absence* of
a deadlock is never reported from a truncated run.

## 3. Determinism

Firing order is fixed by the net's own type, not chosen at the call site. Places
and transitions are sorted by `id` at `build()` and duplicate ids are rejected,
so ids are unique and byte comparison of two distinct ids never returns `Equal`
— a **total** order with no residual tie for another rule to break. This is the
same discipline, for the same reason, as the tie-breaking rule at the foot of
[`crates/ix-duck/sql/pareto_frontier.sql`](../../crates/ix-duck/sql/pareto_frontier.sql).

Consequence: state numbering, witness sequences and the point at which a
truncated run stops are identical on every run and every machine. Insertion
order into the builder cannot leak into a result.

## 4. Interchange: PNML, read-only

`ix_petri::read_pnml` reads the **Place/Transition** subclass of PNML, the
interchange format standardised as **ISO/IEC 15909-2** (Part 1 gives the
semantics, Part 3 the extensibility framework; the reference site is
<https://www.pnml.org/>, which publishes the RELAX NG grammars).

Three decisions, made deliberately:

- **Read, not write.** Reading lets IX analyse nets authored by tools nobody
  here wrote — which is exactly the independent verification this repository
  keeps needing. Writing buys the mirror-image benefit but is a separate slice:
  an emitter must satisfy *other* tools' readers, and nothing here can verify
  that it does. See `crates/ix-petri/src/pnml.rs` for what an emitter would
  have to clear first.
- **P/T nets only.** `ptnet.pntd` adds precisely two labels to the core model —
  `initialMarking` on a place, `inscription` on an arc. Symmetric and
  High-level nets carry a sort system and term algebra, where a marking is a
  multiset of structured tokens and none of the enumeration here applies
  unchanged. A document declaring one is **rejected by name**, not misread.
- **No XML dependency.** The workspace has no XML crate; adding one to parse a
  grammar whose semantic core is eight element names was not worth the
  dependency family. `crates/ix-petri/src/xml.rs` is a ~200-line strict
  tokenizer that rejects `DOCTYPE` outright — no entity declarations means no
  billion-laughs amplification on files received from other tools.

Point it at a file with:

```bash
cargo run -p ix-petri --example analyze_pnml -- path/to/net.pnml
```

## 5. The worked example that motivated the crate

`crates/ix-petri/tests/worktree_pump.rs` models the hazard `CLAUDE.md` warns
about in every session preamble: the git stash stack is shared across the main
checkout and all worktrees. A pump lane holds two things at once — a working
tree and that one shared stack — and the lanes cycle.

The tests establish, by enumeration rather than argument:

- two lanes acquiring the two resources in **opposite orders can wedge**, with
  the witness `L0 takes tree -> L1 takes stash` and nothing enabled after it;
- **one** contrarian lane is enough, and adding conforming lanes does not repair it;
- a **canonical acquisition order** removes the deadlock for every lane count tested;
- so does **giving every lane its own tree** — one contended resource cannot deadlock.

That is the shape of question this crate is for. If you cannot phrase your
problem that way, you probably want one of the crates in §1.

## 6. Not implemented

Deliberately out of scope for this slice, each because it has no consumer yet
rather than because it is hard:

- **Inhibitor arcs, place capacities, priorities, time.** None is in the PNML
  P/T type definition; adding them would make IX's nets unreadable by
  conforming tools, which is the whole reason for using the standard.
- **Writing PNML** (see §4).
- **Coloured / high-level nets** (see §4).
- **Structural analysis without enumeration** — P- and T-invariants, siphons and
  traps. These decide boundedness and some liveness questions on nets far too
  large to enumerate. Worth adding the day a net in this repository is too big
  for `max_states`; not before.
