# Gaia Consolidated Mission-Room Software Factory Specification v0.2

**Status:** consolidated specification candidate; research and design only
**Date:** 2026-08-13
**Milestone:** M1 established; M2-M5 absent; **NOT INTEGRATED**
**Authority:** advisory. This document authorizes no product read, implementation, install, configuration, spend, push, publication, deployment, or new bus verb.

**Predecessor:** `gaia-consolidated-mission-room-factory-spec-v0.1.md`, 24,177 bytes, SHA-256 `b9819c1abf56db80fd75211767f7db032c32c596601a50e778b78343f7be39aa`.

**Revision.** This is the **R17 repair** of the R16 candidate — the bundle its own manifest declared as **R15**, on the round-versus-bundle numbering `gaia-s1-r2-change-ledger.md` records and which has held since R10. The filename and the `v0.2` version string are unchanged, because renaming the specification and the grammar would invalidate every `spec-v0.2:` and `ebnf-v0.2:` citation in the change ledger and the validator's file constants for no reviewability gain. These are different bytes from every previous candidate and are distinguished by digest, not by name; `gaia-s1-r2-bundle-manifest.md` declares the exact fixed point a reviewer must bind.

**S1 relationship.** The independent S1 R1 review (`gaia-s1-independent-review-r1.md`, 39,060 bytes, SHA-256 `84bc1e0be2cb59837e1e0a78219a7cb4c0f150eae226db0f5a135ec2c4f6f6b2`) returned Standards `REQUEST_CHANGES` and Spec `REQUEST_CHANGES` over the immutable four-file bundle whose aggregate digest was `f44ecac23f0c9d180f1bcdcbacb920163a89dc0d04cebbbe6825dbc556c32832`. **That verdict remains valid over its own exact subject and is NOT superseded by this document.**

The fresh independent review of the R2 candidate (`gaia-s1-r2-independent-review-r1.md`) returned Standards `REQUEST_CHANGES` and Spec `REQUEST_CHANGES` over the fourteen-file bundle whose ordinal aggregate digest was `5b3299f982f0813682fca2f7f20be7c5542ca6482fcab7e20e9fd5ce9f6e51e2`, confirming S1-B1, S1-B2, S1-B3, S1-B4 and S1-B6 independently closed and recording two new blocking findings, `S1R2-B01` and `S1R2-B02`. **That verdict likewise remains valid over its own exact subject and is NOT superseded by this document.**

The fresh independent review of the R3 candidate (`gaia-s1-r3-independent-review-r1.md`) returned Standards `REQUEST_CHANGES` and Spec `REQUEST_CHANGES` over the fourteen-file bundle whose ordinal aggregate digest was `97a3a8552016b9278c29a4e595677be84e4872a67b350d355a5273ecd21feb73`. It independently reproduced the fixed point, confirmed `S1R2-B02` closed and properly guarded, and recorded two new blocking findings, `S1R3-B01` and `S1R3-B02`, plus seven non-blocking observations `S1R3-N01`..`S1R3-N07`. **That verdict likewise remains valid over its own exact subject and is NOT superseded by this document.** **The lineage from R4 onward, in full, and none of it superseded:** every subject from the R4 candidate onward was bound by **two** independent reviews, one per axis, and every one returned `REQUEST_CHANGES` — the R4 candidate at ordinal aggregate digest `cb312cb5e1856fe3594eb42f65689b29f8194c843b275b96a9893107cd16a2ac`, returning `S1R4-B01` and `S1R4-B02`; the R5 candidate at `eb4deb3a1a92ae9061b7c8ece4798ac726b08f41d89ffb1d4dc59153f6b08ff2`, returning `S1R5-B01` and `S1R5-B02`, which the two axes number alike over three distinct mechanisms; the R6 candidate at `0df9e510415b1ffd04e5b312f7f9f89bf650102b217928f7c801cf444f000182`, returning `S1R6-B01` as one identifier on both axes; the R7 candidate at `0b67158b7e2c169bb157d0c59fa736cd8abd23428d5271df6993601dfae39dbb`, returning `S1R7-B01` and `S1R7-B02`; the R8 candidate at `deb16e5ca3a7d4726733d9a2956c9aa1bdb8808f75844fbe21fc7c4ebb913e6c`, returning `S1R8-B01`; the R10 candidate — the bundle its own manifest declared **R9** — at `208d0cacef94182d68f3f21c3e307e035b05540ccaa6b484b8ba54d6da5a041c`, returning `S1R10-B01` on both axes over different defects in different files, plus `S1R10-B02`; the R12 candidate — the bundle its manifest declared **R11** — at `596224b221e9463f49d04cb3909076c26ade0412d916c5f496d2c29e606fdb78`, returning `S1R12-B01` on both axes, again alike-numbered over different defects; the R14 candidate — the bundle its manifest declared **R13** — at ordinal aggregate digest `b219cfff1135575ed1ebd3d46a4049cfe0cff52adbdad76c59a0a0e8de8b00fd`, returning `S1R14-B01` on both axes, alike-numbered for the third time, plus `S1R14-B02` and `S1R14-B03`; and the R16 candidate — the bundle its manifest declared **R15**, the direct entry subject of this revision — at ordinal aggregate digest `0b421ac7679937a13e35dc8f2f43456efa72d7f2611217076569bf3d5f996060` and internal thirteen-file digest `cc903de69840e335581e38c06ba0d9ec8849a59bbb4f32897c0a15352ec9a015`. **The two R16 axes are the first pair in this lineage that did not agree, and that is stated here rather than flattened:** Standards returned `APPROVE`, finding no independently reproducible blocker on its axis; Spec returned `REQUEST_CHANGES` with `S1R16-B01` and `S1R16-B02`. **Twelve subjects have now been independently reviewed and every one of them returned at least one `REQUEST_CHANGES`, so not one is closed** — the R1, R2 and R3 subjects by a single independent review each, every subject from R4 onward by two, of which twenty-three of the twenty-four axis verdicts were `REQUEST_CHANGES`. An `APPROVE` on one axis closes nothing on its own, as its own author stated, and this revision claims nothing from it. Each verdict remains valid over its own exact subject and none is superseded by this document; listing them here is recording them, not producing one. This R17 bundle is a new candidate over new bytes. It inherits no approval. It requires a fresh independent exact Standards + Spec review on both axes, and its author grants no approval to it.

**Repairs in this revision.** S1-B1 through S1-B6, `S1R2-B01`/`S1R2-B02`, `S1R3-B01`/`S1R3-B02` and every repair from R4 through R15 as before, plus the two this round owes, both reported on the Spec axis: `S1R16-B01` — the change ledger carried five citations in a comma-list form no check read, sixteen of whose twenty line numbers were wrong, and an out-of-range citation passed the entire suite at exit 0 — and `S1R16-B02`, a manifest sentence claiming an agreement between the ledger's round-scoped census figures and the shipped captured output that does not hold. Both are traced line by line in `gaia-s1-r2-change-ledger.md`. Beyond those repairs and the candidate inputs declared in §9.6, no behavior, verb, grammar, or authority has been added. The R17 repair adds **no bus verb, no lattice, no refusal code, no authority and no subsystem**: it adds one check bounding the *form* of a ledger citation, normalises five citation spans into the syntax the bundle already had, corrects twenty-one wrong line numbers, and changes no production of the protocol itself — `gaia-mission-room-protocol-v0.2.ebnf` is byte-identical to the R4 candidate and has been since. The six bus verbs `register`, `send`, `inbox`, `ack`, `heartbeat`, `handoff` are byte-identical to R2 and R3, and the twelve ledger and two frame terminals are unchanged.

**Normative engineering inputs:** `gaia-engineering-doctrine-v0.1.md` and `gaia-mission-room-domain-context-v0.1.md`. Gaia is a controlled software-engineering factory, not a conversational code generator. Conflict with either document is an S1 Spec defect.

## 1. Outcome and falsifiable boundary

Gaia SHALL provide a vendor-neutral collaboration substrate in which a temporary, ad-hoc team works on one bounded mission while durable agent identity, validated competency evidence, authority, provenance, and acceptance remain explicit.

The smallest next implementation, if separately authorized, is **T1 / M2: a read-only v0a census tracer over copied immutable evidence**. It MUST NOT route work, score people or agents, mutate product state, acquire a database binary, call a live bus, or make an acceptance decision.

This design is refuted as an M2 candidate if the tracer cannot deterministically reproduce the same typed artifact from the same copied evidence, if any emitted field lacks units/null semantics/provenance, if a stale coordinator can mutate, if a retry can duplicate an intent, or if an advisory output can be interpreted as authority.

## 2. Domain model

| Term | Meaning | Lifetime |
|---|---|---|
| **Agent Profile** | Stable logical identity and capability claims for a human or software agent. | Durable |
| **Agent Incarnation** | One concrete process/session for an Agent Profile. Recyclable surface IDs are not identity. | Session |
| **Actor** | Either an Agent Profile or an Agent Incarnation, always explicitly tagged as one of the two. An untagged Actor reference is a defect. | Reference |
| **Competency Evidence** | Content-addressed evidence that a profile performed a bounded capability under named tests/review. It is not a reputation score. | Durable, append-only |
| **Mission Room** | Ad-hoc collaboration scope for one objective, evidence boundary, budget, and acceptance contract. | Mission |
| **Role Assignment** | Temporary responsibility such as Director, Architect, Developer, Reviewer, Tester, Deployment Operator, Researcher, or Advisory Specialist. Roles do not imply global permissions. | Mission/epoch |
| **Lane** | Ephemeral execution context bound to one profile incarnation, role, work scope, and lease. A pane or tab without a live agent is not a lane. | Lease |
| **Task Capsule** | Portable, typed work interface: immutable base, objective, success criteria, **non-goals**, write scope, **dependencies**, contracts, exact verification, forbidden actions, budget, and evidence refs. | Task |
| **Claim** | Exclusive or shared declaration that an incarnation owns a bounded work item. | Lease |
| **Coordinator Fence** | Capability binding lineage, coordinator, epoch, evidence pre-image, preconditions, and intent. | Mutation |
| **Advisory Artifact** | Deterministic result that may inform a decision but can never grant authority, acceptance, safety, or freshness. | Durable |
| **Evidence Bundle** | Content-addressed inputs, commands, outputs, tests, reviews, and provenance for one claim. | Durable |
| **Acceptance Decision** | Separate decision that evidence satisfies a mission contract. Phase success alone is insufficient. | Mission |
| **Knowledge Package** | Reviewed, compact lesson derived from evidence. Raw conversations are not durable organizational memory. | Durable |

The canonical glossary is `gaia-mission-room-domain-context-v0.1.md`. The candidate work-product and uncertainty glossary is `gaia-artifact-workgraph-context-v0.1.md` (§9.6). Synonym drift that changes a contract, state, authority, freshness, or evidence meaning is a Spec defect.

### 2.1 Deep-module seams

Gaia's public surface SHALL be smaller than its implementation. The six numbered entries below — naming eight components in total, because entry 6 declares three separate per-system adapters rather than one shared seam — are a **provisional surface map**, not a set of selected interfaces. Each of the eight named components requires its own Design It Twice record (`gaia-engineering-doctrine-v0.1.md` §2 principle 4) before its M2 interface is fixed. Two records exist today: the coordinator-in-path versus direct A-to-B choice resolved in §3.2, and the takeover design rejected in §6.3.

1. `MissionRoom`: create/close membership and role assignments without exposing vendor sessions.
2. `TaskCapsule`: validate portable work boundaries independently of the executing model.
3. `CoordinatorFence`: acquire/execute/transfer mutation authority; mutation executors reject unfenced calls.
4. `EvidenceLedger`: append content-addressed observations and receipts; never infer completion from a marker alone.
5. `AcceptanceGate`: evaluate deterministic checks and independent review against the exact subject digest.
6. `IxEvidenceAdapter`, `TarsGrammarAdapter`, `HariReplayAdapter`: narrow, independently versioned adapters, each isolating one external system without granting authority. Each produces an **Advisory Artifact** and nothing else. **No shared advisory seam is declared.** A shared seam may be proposed only after at least two of these adapters demonstrate a common contract with real implementations, per `gaia-engineering-doctrine-v0.1.md` §2 principle 5 and the canonical **Adapter** and **Seam** definitions in `gaia-mission-room-domain-context-v0.1.md`. Until then, generalizing them into one abstraction is the speculative generality that `gaia-engineering-doctrine-v0.1.md` §4 control 6 forbids, and §9 of this document classifies every one of these systems as narrow, deferred, or opt-in.

Gaia does not define a component, role, seam, or artifact named "oracle". The term is ambiguous between "advisory participant" and "deterministic gate", which are opposite sides of this specification's central boundary. Deterministic gates are named `AcceptanceGate`, deterministic checks, or validators; advisory participants are named advisory adapters and hold the `AdvisorySpecialist` role.

### 2.2 Engineering discipline

Every material change SHALL follow the proportional router and completion contract in `gaia-engineering-doctrine-v0.1.md`. In particular:

- canonical domain terms precede interface design;
- consequential interfaces receive two materially different designs before selection;
- the first authorized implementation is a vertical tracer bullet;
- TDD or an equivalent falsifier records a discriminating failure before the minimum fix;
- tests target observable behavior at a declared interface seam;
- tests do not bind private implementation detail while the public interface stays unverified;
- an abstraction, framework, or generalized seam with only one speculative consumer is rejected;
- prototypes remain quarantined and cannot become product evidence;
- specification, implementation, independent review, and acceptance are separate authorities;
- a completion marker, model consensus, or green test alone cannot establish acceptance.

The eighteen testable negative controls in §11 and the fourteen doctrine controls in `gaia-engineering-doctrine-v0.1.md` §4 are **cumulative**. Their union is the M2 test obligation; neither list alone discharges it.

## 3. Architectural split

### 3.1 Governance plane

Gaia owns mission identity, claims, leases, lineage, epochs, fences, budgets, evidence references, and the six-verb orchestration contract. Demerzel may evaluate declared policy. Agent Blackbox independently verifies evidence integrity, negative controls, risk explanations, and fixed points.

The governance plane is authoritative only for explicitly granted effects. It MUST fail closed when accounting, lineage, epoch, evidence, preconditions, or store availability are unknown.

### 3.2 Collaboration/data plane

Agents MAY communicate directly A-to-B for mission content. Direct communication is an optional data path, not optional governance. Every direct message remains attached to a Mission Room and uses the existing `send`/`inbox`/`ack` semantics with exact sender profile, incarnation, recipient, epoch, correlation, idempotency key, deadline, hop limit, and evidence refs.

The coordinator MAY be absent from the healthy content path, but policy, leases, fencing, accounting, observability, and acceptance remain in force. Two coordinators with divergent lineage or epoch MUST NOT both mutate. During a partition, mutation availability is sacrificed rather than risking split brain.

### 3.3 Exactly six non-privileged verbs

The only bus verbs are:

`register`, `send`, `inbox`, `ack`, `heartbeat`, `handoff`.

No verb approves, commits, merges, pushes, deploys, changes configuration, reads credentials, spends money, grants authority, establishes freshness, or executes product mutations. Such effects require a separate fenced executor and explicit policy.

This control is **mechanically witnessable over the whole protocol**, and the scope of that witness is §3.3, the section whose heading stands above this sentence. `gaia-mission-room-protocol-v0.2.ebnf` classifies every production reachable from `protocol` at statement position into exactly one of three disjoint classes.

| Class | Leading terminals | Count | What it may do |
|---|---|---:|---|
| `bus_statement` | `register`, `send`, `inbox`, `ack`, `heartbeat`, `handoff` | 6 | utter on the non-privileged collaboration bus |
| `ledger_statement` | `assign`, `lane`, `claim`, `capsule`, `fence`, `artifact`, `transition`, `receipt`, `evidence`, `advisory`, `verdict`, `refuse` | 12 | record a transcript fact; no transport effect, no channel, no authority |
| `frame_statement` | `mission`, `close` | 2 | open or end the Mission Room frame; no transport effect, no channel, no authority |

`bus_statement` has exactly six alternatives whose leading terminals are exactly the six verbs above, `inbox` included. The twelve ledger terminals are not verbs seven through eighteen, and the two frame terminals are not verbs nineteen and twenty: neither class carries a transport effect, opens a channel, or grants authority.

`close` records an `acceptance_state`, which is authority-bearing. It does not produce that authority. Per §7.1 the acceptance value is produced by a **separate Acceptance Decision**, is attributed to a named `acceptor`, and binds the exact `subject_digest` it accepts. The revocation of role assignments, claims, leases, and fences at `closed` (§6.1) removes authority and grants none.

The **Leading terminals** column and the **Count** column above are bounds over *derivations*, not over member names. For every member of every class, the leading terminal of **every top-level alternative** of that member is collected — descending into groups, stepping past an omittable leading element to the element behind it, and resolving through a bare reference to another production — and the resulting multiset must equal the class row above exactly, in membership and in cardinality. Both halves of that equality are decided **against this table**, not only against the grammar. `SPEC_CLASS_TERMINALS_MATCH_GRAMMAR` reads the **Leading terminals** cell of each row above in full — resolving a Markdown link to its text and stripping inline code and emphasis markup from each name's edges, so that a name spelled bare, bolded or linked is read exactly as a backticked one is — splits the cell on its commas, and requires multiset equality with the measured leading terminals of that class in both directions, for all three classes: membership, the number of names the cell itself lists, and that same number against the adjacent **Count** cell. A name added to a cell in any of those renderings while the grammar and the **Count** cell stay unchanged is therefore a detected defect, and so is a name the cell lists twice. This table is read as exactly one row per class: two rows for one class are themselves a defect here, and both §3.3 checks fail closed on it rather than resolving the ambiguity by position. Which rows the table has is decided by parsing the table itself, not by matching a row pattern: the header above locates the **Class**, **Leading terminals** and **Count** columns by name, and every body row beneath the separator is read whatever Markdown rendering its **Class** cell is given, whatever its **Count** cell states, and at any indentation GFM admits before a table row. A body row whose **Class** cell names no class, whose **Count** cell is not a bare integer, or which carries a cell count the header does not declare is itself a defect and is reported; it is never read as an absence. A row naming a class anywhere in §3.3 outside this table is a defect for the same reason. The R4 candidate asserted this membership equality with no mechanism reading the cell; the R5 candidate read only the last row of each class, only its backticked names, and only as a set, so a shadowed duplicate row, an unbackticked added name and a duplicated name each passed its whole suite; the R6 candidate read every row its pattern matched and called that the table, so a second `bus_statement` row spelled bare, bolded or linked, carrying a **Count** of `seven` or `7<sup>a</sup>`, or indented by one space, declared a seventh bus leading terminal and passed its whole suite. The R4, R5 and R6 independent reviews recorded these as blockers on both axes.

Seven defects are detected mechanically:

| Defect | Detected by |
|---|---|
| a seventh `bus_statement` alternative, a missing verb, or a ledger or frame terminal colliding with a bus terminal | `EXACTLY_SIX_BUS_VERBS`, `LEDGER_TERMINALS_DISJOINT_FROM_BUS` |
| a third frame alternative, or a frame terminal other than `mission` and `close` | `FRAME_STATEMENT_BOUNDED` |
| a production reachable from `protocol` at statement position that belongs to no class or to two classes | `PROTOCOL_PARTITION_EXHAUSTIVE` |
| a leading terminal outside a class row, a declared terminal absent from it, a surplus alternative **inside** a class member, or a respelled terminal that preserves the count — for the bus, ledger and frame classes respectively | `BUS_CLASS_LEADING_TERMINALS_BOUND`, `LEDGER_CLASS_LEADING_TERMINALS_BOUND`, `FRAME_CLASS_LEADING_TERMINALS_BOUND` |
| a leading position that cannot be decided: absent production, cycle, unbalanced group — an opening bracket with no closer **or** a closing bracket with no opener, at any position in the alternative and not only where the element scan reaches it — or an alternative every element of which is omittable | `CLASS_LEADING_TERMINALS_RESOLVE` |
| drift between the terminal names in this prose and the grammar, in **both** directions, for all three classes, in cardinality as well as membership — a name the cell lists twice included — and whatever Markdown rendering the added name is given, in the row's **Class** cell as well as in its **Leading terminals** cell | `SPEC_CLASS_TERMINALS_MATCH_GRAMMAR`; the §3.3 declaration sentence and the dropped-name direction are additionally covered by `BUS_TERMINALS_MATCH_SPEC` and `SPEC_ENUMERATES_ALL_CLASS_TERMINALS`, neither of which sees a name the prose adds, and both of which read backticked names only |
| drift between the **Count** column of the table above and the grammar, and a second row for a class placed so that it would shadow the genuine one — in any **Class** cell rendering, beside any **Count** cell, at any indentation, and inside or outside the table | `SPEC_CLASS_COUNTS_MATCH_GRAMMAR`, which reads every row of the table as the table renders it rather than every row some pattern matches, as `SPEC_CLASS_TERMINALS_MATCH_GRAMMAR` also does |

An undecidable bound fails closed and is never read as satisfied.

**Declared scope of these bounds.** They read the *leading* position only. A verb-like production at field or continuation position inside a statement is outside them by construction. That is the boundary recorded as open in §12, it is unchanged by this revision, and it is exercised mechanically by four disclosed-boundary positive controls in the validator — mutations that must **not** trip a class bound — rather than being asserted in prose alone.

The R2 candidate claimed mechanical witnessability while partitioning only the `statement` set. `protocol = mission , { statement } , close` reaches `mission` and `close`, which no class named, and the R2 independent review demonstrated a seventh verb-like production added at frame level that passed the entire validator suite. The claim was stronger than the evidence. Classifying the frame is what makes the claim true, and `PROTOCOL_PARTITION_EXHAUSTIVE` enumerates reachability from `protocol` from scratch rather than checking only that no stray production leads with an already-known bus verb.

**The R3 candidate then repeated that mistake one level down, and this revision repairs it.** R3 partitioned the classes by member *name* and both of its class-bound checks took only the *first* terminal of a production and stopped, so alternation *inside* a member was invisible to every check and to all forty fixtures. The R3 independent review demonstrated two consequences over aggregate digest `97a3a8552016b9278c29a4e595677be84e4872a67b350d355a5273ecd21feb73`. `S1R3-B01`: adding `| ( "broadcast" , task_id , profile_id , evidence_ref )` as a second top-level alternative of `handoff` makes `broadcast …` derivable at **statement position, inside the bus class** — `statement ⟹ bus_statement ⟹ handoff ⟹ "broadcast" …`, utterable on its own — and it passed 51 of 51 positive checks and 40 of 40 fixtures, including `EXACTLY_SIX_BUS_VERBS`, `NO_SEVENTH_VERB_PRODUCTION` and `BUS_TERMINALS_MATCH_SPEC`, each of which this document had named as detecting exactly that. `S1R3-B02`: `ledger_statement` carried no cardinality or membership bound at all, so a thirteenth ledger statement — one binding the authority-bearing `acceptance_state` — was admitted while this section named `BUS_TERMINALS_MATCH_SPEC` and `SPEC_ENUMERATES_ALL_CLASS_TERMINALS` as catching that drift, and neither did.

Both are the same failure the R2 disclosure above describes: **a bound stated over a set of names while the grammar's real surface is a set of derivations.** The claim was again stronger than the evidence. The four `*_LEADING_TERMINALS_*` checks decide the property over derivations, so the claim is now the one the mechanism makes. Nothing else changed: the six bus verbs `register`, `send`, `inbox`, `ack`, `heartbeat`, `handoff` are byte-identical to R3, and no verb, lattice, refusal code, authority or subsystem was added.

## 4. Vendor-neutral task capsule

Every lane SHALL consume a capsule equivalent to:

```yaml
schema_version: gaia-task-v0.2.0-draft
mission_id: m:<stable id>
task_id: t:<stable id>
lineage_id: lg:<stable lineage>
epoch: <monotonic integer>
actor:
  profile_id: p:<durable logical identity>
  incarnation_id: i:<unique process/session identity>
role_assignment: ra:<temporary role assignment>
subject:
  repo: <qualified repository>
  base_digest: sha256:<immutable revision or artifact digest>
objective: <one falsifiable outcome>
success_criteria: [<mechanically checkable condition>]
non_goals: [<explicit excluded outcome or change>]
write_scope: [<explicit path or none>]
dependencies: [<t:task_id | k:contract_ref that must be satisfied first>]
contract_refs: [k:<versioned contract>]
verification: [<exact command or deterministic validator>]
forbidden_actions: [<effect boundary>]
budget:
  incremental_paid_spend: 0
  token_accounting: <known value or unknown>
claim:
  lease_id: ls:<id>
  expires_at: <timestamp>
evidence_refs: [ev:<content-addressed input>]
```

### 4.1 Cardinality and identity contract

Every field is required to be **present**. An absent field is a binder failure, not an empty value. Unknown accounting or model availability SHALL fail closed.

| Field | Cardinality | Identity type | Empty permitted |
|---|---|---|---|
| `schema_version` | exactly 1 | literal | no |
| `mission_id` | exactly 1 | `m:` Mission Room | no |
| `task_id` | exactly 1 | `t:` Task Capsule | no |
| `lineage_id` | exactly 1 | `lg:` lineage | no |
| `epoch` | exactly 1 | monotonic integer | no |
| `actor.profile_id` | exactly 1 | `p:` Agent Profile | no |
| `actor.incarnation_id` | exactly 1 | `i:` Agent Incarnation | no |
| `role_assignment` | exactly 1 | `ra:` Role Assignment | no |
| `subject.repo` | exactly 1 | qualified name | no |
| `subject.base_digest` | exactly 1 | `sha256:` digest | no |
| `objective` | exactly 1 | text | no |
| `success_criteria` | 1..n | text | no |
| **`non_goals`** | **1..n** | text | **no — an empty list is refused** |
| `write_scope` | 0..n | path | yes, written `[]` |
| **`dependencies`** | **0..n** | `t:` Task Capsule or `k:` contract | **yes, written `[]`; absent is refused** |
| `contract_refs` | 0..n | `k:` contract | yes, written `[]` |
| `verification` | 1..n | exact command or validator | no |
| `forbidden_actions` | 1..n | effect boundary | no |
| `budget.incremental_paid_spend` | exactly 1 | integer, currency-neutral | no |
| `budget.token_accounting` | exactly 1 | known value or `unknown` | no |
| `claim.lease_id` | exactly 1 | `ls:` Lease | no |
| `claim.expires_at` | exactly 1 | timestamp | no |
| `evidence_refs` | 0..n | `ev:` Evidence Item | yes, written `[]` |

`non_goals` is 1..n because `write_scope` bounds *where* a lane may write but not *what it must not attempt*. A lane that rewrites an in-scope file far beyond its objective stays inside `write_scope` and, without `non_goals`, violates no declared field. A capsule that can declare no excluded outcome has not been bounded, so the empty case is refused rather than defaulted.

`dependencies` is 0..n but never absent, because an absent dependency list and an empty dependency list are different claims. `[]` asserts "no declared upstream work"; absence asserts nothing and MUST fail closed, so that a capsule blocked on unbuilt upstream work is distinguishable from a ready one at admission.

Vendor-specific context, memory, skills, and UI state are disposable adapters and MUST NOT be the source of truth.

## 5. Identity, roles, and accumulated competency

1. An Agent Profile is stable; an incarnation is unique and never inferred from pane history.
2. Every Actor reference — sender, recipient, producer, reviewer — is explicitly tagged as a profile or an incarnation. An Actor reference resolvable to both classes is refused as `AMBIGUOUS_RECIPIENT`.
3. Role assignment is scoped to a mission and epoch and expires with its lease.
4. Competency accumulates only through `CompetencyEvidence` naming capability, subject digest, tests, independent reviewer, result, and expiry/revisit trigger.
5. Competency evidence MUST NOT become a scalar trust, productivity, attribution, blame, or routing score.
6. A role MAY require a set of competency evidence, but the acceptance gate evaluates the produced evidence independently.
7. Writer and independent reviewer identities MUST differ for approval-bearing reviews.

This supports ad-hoc teams analogous to a temporary engineering group: Director, Architect, Developers, Testers, Deployment Operators, Researchers, and Advisory Specialists collaborate in one Mission Room, while the team dissolves after verified handoff and acceptance.

## 6. State machines

### 6.1 Mission

`draft -> admitted -> active -> verifying -> accepted | rejected | blocked -> closed`

- `admitted` requires a valid capsule and known authority/budget boundary.
- `active` requires at least one live claimed lane.
- `verifying` freezes the subject digest for acceptance.
- `accepted` requires deterministic gates plus required independent verdicts.
- `blocked` requires a typed reason; `unknown` is not silently converted to pass or fail.
- `closed` revokes role assignments, claims, leases, and fences; durable evidence remains.

### 6.2 Lane

`registered -> claimed -> running -> handoff_ready -> verified -> stopped -> reaped`

A completion marker is evidence only. `verified` requires exact subject, scope, tests/validator where applicable, provenance, and cleanliness. A lane SHALL be stopped and its exact surface reaped only after consumption or explicit discard.

### 6.3 Mutation

`intent_declared -> fence_acquired -> preconditions_rechecked -> executed_once -> receipt_recorded`

The executor SHALL atomically record the idempotency key and receipt.

No time-to-live takeover is in the first tracer. Replacement requires a verified transfer or proof that the previous primary cannot act.

### 6.4 Refusal codes are typed, registered, and open

The **registered refusal codes for v0.2** are the following fifteen. The first ten are the mutation-path codes carried forward unchanged. The last five close the five declared negative controls that previously had no protocol representation.

| Code | Meaning | Control |
|---|---|---|
| `STALE_EPOCH` | acting under a superseded epoch | §11 #2, #5 |
| `FENCE_MISMATCH` | fence does not bind the presented pre-image | §11 #8 |
| `LINEAGE_DIVERGENCE` | two lineages claim the same mutation right | §11 #3, #4 |
| `EVIDENCE_DRIFT` | evidence changed after approval | §11 #7 |
| `PRECONDITION_DRIFT` | a declared precondition no longer holds | §11 #5 |
| `DUPLICATE_INTENT` | a retry would produce a second effect | §11 #6 |
| `PRIMARY_NOT_PROVEN_UNAVAILABLE` | replacement attempted without proof | §6.3 |
| `ACCOUNTING_UNKNOWN` | cost or model availability unknown | §11 #9 |
| `STORE_UNAVAILABLE` | evidence or head store cannot be verified | §3.1 |
| `EXECUTOR_UNFENCED` | mutation attempted without a fence | §11 #8 |
| **`SCOPE_VIOLATION`** | **mutation fell outside the declared `write_scope`; the refusal names the exact violated scope entry** | **§11 #17** |
| `SELF_REVIEW` | writer and approval-bearing reviewer are the same Actor | §11 #12 |
| `UNREGISTERED_ACTOR` | unregistered actor or off-ledger direct message | §11 #16 |
| `AMBIGUOUS_RECIPIENT` | recipient resolves to more than one Actor class | §11 #10 |
| `NON_DISCRIMINATING_GATE` | a quality command cannot fail on a known mutation | §11 #18 |

`SCOPE_VIOLATION` is exact: the refusal carries the declared `write_scope` entry that the attempted mutation fell outside, so a scope violation is never misattributed to `PRECONDITION_DRIFT`. Recording a scope violation under a code that means something materially different is synonym drift, which `gaia-mission-room-domain-context-v0.1.md` working rule 4 defines as a Spec defect.

The set stays **open** without admitting untyped strings. An unregistered refusal uses the typed extension form `X-<UPPER-WORD>{-<UPPER-WORD>}`. The forward-compatibility rule is normative:

1. A consumer that does not recognise a refusal code MUST treat the statement as a refusal and **fail closed**.
2. It MUST NOT convert an unknown code into a pass, an approval, an acceptance, a freshness state, a safety claim, or any authority. An unknown code never widens an effect.
3. It MUST **preserve the code verbatim** in the ledger so that a later reviewed registration can interpret it without evidence loss. Unknown does not mean discardable.
4. It MUST NOT emit an extension code where a registered code applies. Emitting `X-SCOPE-VIOLATION` instead of `SCOPE_VIOLATION` is synonym drift and a Spec defect.

Promoting an extension code to a registered code is a reviewed specification change, not a runtime decision.

## 7. Evidence and acceptance

### 7.1 Evidence rules

- Inputs and outputs use canonical serialization and cryptographic digests.
- Evidence records retain digests and qualified paths, not embedded sensitive pre-images.
- All result fields declare type, unit, nullable states, producer, input digests, method version, and timestamp semantics.
- Message bodies, fragments, and body digests are excluded from the mechanics advisory artifact.
- A reviewer replays claims against the exact subject; it never inherits the writer's verdict.

**Evidence kind and verdict are different lattices, and neither is freshness, execution, or acceptance.** The five lattices below are **pairwise disjoint token sets**. Disjointness is the mechanical meaning of "these states remain distinct": merged into one union they would be type-compatible, and any validator would admit the substitution.

| Lattice | Values | Who may produce it |
|---|---|---|
| **Evidence state** | `PRESENT`, `ABSENT`, `UNKNOWN`, `CONTRADICTORY` | any producer, including an advisory adapter |
| **Verdict state** | `APPROVED`, `REJECTED`, `REQUEST_CHANGES` | only an independent reviewer, over an exact subject digest |
| **Freshness state** | `FRESH`, `SUSPECT`, `STALE`, `FRESHNESS_UNKNOWN` | only a deterministic freshness computation over digests, accepted heads, and edge policy |
| **Execution result** | `RECEIPT_WRITTEN`, `FAILED`, `VERIFIED` | only a Transition executor |
| **Acceptance state** | `accepted`, `rejected`, `blocked` | only a separate Acceptance Decision |

`FRESHNESS_UNKNOWN` is spelled distinctly from `UNKNOWN` for exactly this reason: sharing one token across two lattices would defeat disjointness.

The five token sets are compared **byte-exact and case-sensitively**. `REJECTED` and `rejected` are distinct tokens in different lattices and are never normalized to one another; a consumer that case-folds before comparing conflates a verdict with an acceptance.

### 7.1.1 The "who may produce it" column is attributed, not merely asserted

The third column above states a requirement about **who**. A requirement about who is unwitnessable unless the record names an Actor, so every lattice-bearing production in `gaia-mission-room-protocol-v0.2.ebnf` binds an attribution field, and every attribution field is an `actor_ref` — explicitly tagged as a profile or an incarnation, per §5 item 2.

| Production | Lattice it binds | Attribution field | Binds `subject_digest` |
|---|---|---|---|
| `evidence` | evidence state | `producer` | yes |
| `advisory` | evidence state | `producer` | no — an advisory artifact is bound to its `input_digests` |
| `verdict` | verdict state | `reviewer` | yes |
| `artifact_revision` | freshness state | `producer` | no — a revision is identified by its own `digest` |
| `transition_receipt` | execution result | `executor` | no — a receipt is bound to its `fence_id` and `input_digests` |
| `close` | acceptance state | `acceptor` | **yes** |

The four attribution names — `producer`, `reviewer`, `executor`, `acceptor` — are the complete set. `LATTICE_PRODUCTIONS_NAME_AN_ACTOR` enumerates lattice-bearing productions from the grammar rather than from a fixed list, and fails if any of them names no Actor; `ACTOR_FIELDS_RESOLVE_TO_ACTOR_REF` fails if an attribution field resolves to anything other than `actor_ref`; `ACCEPTANCE_BINDS_SUBJECT_DIGEST` fails if any acceptance-bearing production omits the subject it accepts.

This is what makes the separation of authorities checkable rather than merely stated. `SELF_REVIEW` (§11 #12) compares a writer against a reviewer, and an acceptance against its acceptor; neither comparison is possible against a record that names nobody. In the R2 candidate, `advisory`, `transition_receipt` and `close` named no Actor, `close` bound no subject, and deleting `producer` from `evidence` left the whole validator suite passing — so the attribution pattern was unguarded rather than merely incomplete.

The obligation of §7.1's third column that is **not** discharged at S1 is the *entitlement* half: the grammar witnesses that an Actor is named and typed, not that this particular Actor was entitled to produce that particular lattice value. Entitlement is an M2 binder obligation, and this specification does not claim otherwise.

Consequently:

1. An **Advisory Artifact carries an evidence state only.** It can never carry, imply, or be coerced into a verdict, a freshness state, an execution result, or an acceptance state. `advisory ... APPROVED` does not parse in `gaia-mission-room-protocol-v0.2.ebnf`, which is the mechanical form of §1's refutation condition "an advisory output can be interpreted as authority".
2. Advisory evidence cannot establish **approval, acceptance, safety, authority, or freshness**. Model consensus, an aggregate score, a confidence value, a popularity signal, and a completion marker are likewise barred from all four non-evidence lattices.
3. A `VERIFIED` execution result is not an Acceptance Decision. `RECEIPT_WRITTEN` and process exit zero establish neither.
4. A timestamp, a marker, an agent assertion, or a model claim cannot establish `FRESH`. A missing manifest, rule, or accepted head yields `FRESHNESS_UNKNOWN`.

### 7.2 Acceptance is not phase success

A command exiting zero, a marker, a generated report, a green unit test, a local scratch artifact, or a GitHub distribution alone is not acceptance. Acceptance combines:

1. exact subject and fixed point;
2. declared scope and mutation census;
3. deterministic tests/validator and negative controls;
4. evidence integrity and provenance;
5. independent Standards and Spec verdicts where required;
6. cleanliness and no undeclared effects;
7. explicit authority for the next effect.

The first and fifth elements are bound in the grammar: `close` carries the exact `subject_digest` it accepts and the `acceptor` that exercised the decision, so an acceptance recorded against no subject, or by nobody, does not parse. The remaining five elements are M2 binder obligations and are not witnessed at S1.

## 8. Mechanics advisory tracer doctrine

The mechanics work is **discrete-first** and descriptive. Primary-source families are normative inputs: graph signal processing and Hodge decompositions, queueing/network flow/backpressure, Lyapunov/control stability, Contract Net and SharedPlans, and information geometry only when its assumptions are measured.

Use neutral observable names such as `edge_flow`, `backlog`, `queue_age`, `retry_count`, `component_count`, and `ref_order_anomaly`. A mechanics analogy MAY appear only as an attributed alias with units and an explicit mapping. Torseur/wrench, SE(3)-free torsors, literal constitutive/material laws, and unmeasured continuum claims are rejected. No fitted exponent, consensus value, or advisory metric authorizes action.

### 8.1 T1 / M2 candidate: v0a census only

If separately authorized, T1 SHALL:

- read only a unique copied immutable evidence manifest;
- perform no model fitting and no prediction;
- produce one typed deterministic advisory artifact;
- compare against simple out-of-sample baselines before any richer model;
- emit nothing if its pre-run binder cannot resolve every declared field;
- use an idempotency key for any later `send`, though T1 itself calls no bus;
- remain read-only, local, and zero incremental paid spend.

DuckDB acquisition, live product reads, IXQL, `duck.query`, v0b components/fragmentation, interventional experiments, and automatic IX coupling are explicitly out of T1 scope. Binder incompleteness is a blocker for emitted/locked fields; defects confined to non-authoritative explanatory prose are corrections but must be disclosed.

Feature or surface widening beyond the approved v0a behavior requires a new exact review, regardless of how small the change appears.

## 9. Specialized systems

### 9.1 IX — structural computation adapter (`DEFER` for automatic integration)

IX MAY, on copied immutable evidence:

- validate the EBNF and generate bounded derivation examples;
- project evidence-keyed WorkGraphs and run cycle/topology checks;
- detect contradictions at joins, including validator-present versus validator-absent claims;
- compute census statistics and simple baselines;
- compare candidate models out of sample.

IX MUST NOT route live work, mutate Gaia state, add IXQL grammar, or turn a result into authority. The first useful IX tracer is the narrow evidence-keyed contradiction projection, not a general software factory. Its output is an Advisory Artifact carrying an evidence state.

### 9.2 TARS — grammar adapter (`ADAPTER_ONLY`)

TARS MAY consume the local EBNF, test whether protocol utterances map to canonical intents, identify ambiguous or unparseable clauses, and compare extracted grammar with the committed domain glossary. Grammar extraction is advisory; Gaia's versioned contract remains canonical. TARS MUST NOT change policy, register a global grammar, or grant an effect without a separately reviewed action. Grammar weights are advisory and cannot promote a fact, a freshness state, a safety claim, a quality verdict, or authority.

### 9.3 HARI — epistemic replay adapter (`ADAPTER_ONLY`, opt-in)

HARI MAY ingest a content-addressed event trace and declared pairwise `Supports`, `Implies`, and `Contradicts` relations, preserve six-valued uncertainty/contradiction, and return derivation provenance plus an investigation recommendation. Use deterministic session replay and baseline comparisons.

HARI MUST NOT be the researcher, coordinator, acceptance judge, source of authority, or default trust scorer. Role-weighted consensus remains opt-in and cannot establish truth or permission. Current HARI is an experimental single-tenant research sandbox with an unstable API; real IX integration and cross-session reliability remain unproven. Gaia records HARI output only as an Advisory Artifact with model/config/version and trace digest.

### 9.4 Demerzel and Agent Blackbox

Demerzel evaluates declared governance policy. Agent Blackbox independently verifies exact fixed point, evidence integrity, risk explanation, negative controls, and refusal behavior. Neither may fabricate the other actor's evidence.

### 9.5 JEPA-inspired latent prediction — research only

V-JEPA 2 demonstrates self-supervised prediction in a latent representation, followed by action-conditioned planning in the physical/video domain. Gaia MUST NOT infer that the released visual model optimizes software agents. The transferable research hypothesis is narrower: learn or test whether a compact latent representation of immutable mission traces can predict the next **evidence state** better than simple event-count, recency, and WorkGraph baselines without reconstructing private payloads.

This belongs after M3, never on the critical path to M2. A future experiment must use held-out missions, compare against simple baselines, report calibration and null results, exclude message bodies and agent scoring, and remain advisory. No V-JEPA checkpoint, GPU dependency, or model install is authorized by this specification.

### 9.6 Artifact WorkGraph, fog-of-war, Semantic GPS, reproducibility, geometry, and multi-axis uncertainty — declared candidate inputs

Five research and design artifacts are **explicit candidate inputs** to this bundle. They are named here so an independent reviewer holds the exact bytes they must weigh. Each is copied into the bundle and hashed in §14.

| Candidate input | Contributes | Milestone boundary |
|---|---|---|
| `gaia-artifact-workgraph-context-v0.1.md` | Artifact, Artifact Revision, Artifact Lineage, Artifact WorkGraph, Derivation Edge, Build Recipe, Composite Artifact, Transition Definition, Transition Receipt, Graph Snapshot, Graph Capsule, Maturity Gate, freshness vocabulary, Fog-of-War Envelope, Uncertainty Condition, Probe, Reproducibility Envelope, Determinization Boundary, Semantic Map/Position/Destination/Route, Axis Contract, Product State Space, Semantic Chart, Uncertainty Envelope, Decision Envelope | glossary candidate only |
| `gaia-artifact-workgraph-staleness-design-v0.1.md` | freshness/lifecycle/refresh/quality state separation, edge-specific invalidation, refresh cascade, graph-analysis kernel tiers, blue/green backbone, Wayfinder fog-of-war doctrine (§11.4), Semantic GPS (§11.5), multi-axis geometry (§11.6), thirty-one negative controls | design candidate only; no runtime, no database, no refresh execution |
| `gaia-uncertainty-grammar-v0.1.ebnf` | typed uncertainty conditions, probe grammar, reproducibility axes, navigation and geometry profiles, twenty-two semantic checks | candidate Transition Definition input; **not** an installed or registered TARS grammar |
| `gaia-multiaxis-uncertainty-math-primary-research.md` | pinned primary sources for bilattice knowledge state, credal sets, aleatoric/epistemic separation, proper scoring, conformal limits, projection distortion audit | research evidence only |
| `gaia-multiaxis-uncertainty-analysis-design-v0.1.md` | typed product state space, aggregation recipes, decision envelopes, value-of-information probes, calibration, geometry pipeline, eighteen negative controls | design candidate only |

Their admission into this bundle changes **nothing** about authority or milestone state. Specifically:

1. **They are candidate inputs, not accepted design.** None of them has passed an independent exact review. This bundle does not approve them.
2. They add **no bus verb.** The refresh cascade, the Semantic GPS, the probe machinery, and the projection pipeline all communicate through the existing six verbs or through no bus at all.
3. They add **no authority.** Coordinates, clusters, distances, routes, charts, credal bounds, probe rankings, uncertainty envelopes, and graph analytics are Advisory Artifacts. Per §7.1 they carry an evidence state, and they cannot produce approval, acceptance, safety, authority, or freshness.
4. They authorize **no implementation, database, package, model, or installation.** PostgreSQL, FalkorDB, DuckDB, embedding models, and graph engines remain unacquired. Blue/green backbone replacement is explicitly not required for M2 or M3.
5. Their protocol footprint in `gaia-mission-room-protocol-v0.2.ebnf` is confined to **ledger statements** — `capsule`, `fence`, `artifact`, `transition`, `receipt` — plus the `freshness_state` and `replay_class` terminals. Ledger statements record lineage facts and execute nothing.
6. The typed uncertainty vocabulary does **not** collapse into a scalar. One `UNKNOWN` is insufficient; a confidence number can never upgrade `UNKNOWN` to `FRESH`, `APPROVED`, or `SAFE`; and a contradiction is retained, not resolved by fluency.
7. Learned embeddings, JEPA-style prediction, and graph neural models remain **post-M3 research** that must beat exact graph baselines out of sample and remain advisory.

The v0a T1 boundary in §8.1 is unchanged by these inputs. Nothing in §9.6 enters the critical path to M2.

## 10. Controlled improvement

Gaia distinguishes retry, scaffold improvement, and recursive improvement:

- Retry modifies the work product under the same harness.
- Scaffold improvement creates a quarantined candidate prompt, skill, tool, orchestration rule, or evaluator.
- Controlled recursive improvement exists only after a candidate passes fixed/held-out evaluation, repeated trials, exact provenance, independent promotion review, regression checks, rollback, and then influences a later measured generation.

Agents MUST NOT directly promote their own memory, skills, or control-plane changes. Persistent corrections become reviewed versioned knowledge packages; raw chat and vendor auto-memory remain caches.

## 11. Negative controls

Before M2 or any later mutation, tests SHALL cover at least the following. Controls #1-#18 are carried forward unchanged. Controls #19-#24 were added by the R2 revision and each one discriminates a specific S1 R1 repair. Controls #25-#26 are added by the R3 revision and each one discriminates a specific S1 R2 repair.

1. same logical agent, new incarnation: old lease rejected;
2. old coordinator resumes after replacement: stale epoch rejected;
3. two coordinators claim the same lineage: mutation fails closed;
4. divergent lineage after partition: `LINEAGE_DIVERGENCE`;
5. delayed command crosses transfer: epoch/precondition rejection;
6. duplicate send/intent retry: one receipt, no duplicate effect;
7. changed evidence after approval: `EVIDENCE_DRIFT`;
8. missing fence or bypassed coordinator: executor rejection;
9. unknown accounting/model availability: no launch or spend;
10. ambiguous recipient or recyclable surface identity: rejection, recorded as `AMBIGUOUS_RECIPIENT`;
11. marker exists but tests/provenance/cleanliness fail: not complete;
12. writer reviews itself: approval rejected, recorded as `SELF_REVIEW`;
13. contradictory source claims: retained as contradictory, not collapsed;
14. consensus says pass while deterministic gate fails: reject;
15. evidence path missing or reused: reject;
16. unregistered actor or off-ledger direct message: not admissible evidence, recorded as `UNREGISTERED_ACTOR`;
17. mutation outside `write_scope`: reject, recorded as `SCOPE_VIOLATION` naming the exact violated scope entry;
18. phase exits zero with placeholder quality command: acceptance fails, recorded as `NON_DISCRIMINATING_GATE`;
19. **(S1-B1)** a capsule with an empty `non_goals` list, or with `dependencies` absent rather than `[]`, is refused at admission and is not defaulted;
20. **(S1-B2)** a single speculative consumer cannot justify a shared seam: an adapter set with fewer than two demonstrated implementations of a common contract is refused generalization;
21. **(S1-B3)** an advisory statement carrying a verdict, freshness, execution, or acceptance value is unparseable, and the five lattices share no token;
22. **(S1-B4)** an Actor reference resolvable to more than one identity class is refused, and no two identity type prefixes collide;
23. **(S1-B5)** a seventh bus alternative, a removed verb, or a ledger terminal colliding with a bus terminal fails the verb-set check;
24. **(S1-B6)** an unrecognised refusal code fails closed, is preserved verbatim, and never becomes a pass, approval, acceptance, freshness, safety, or authority;
25. **(S1R2-B01)** a verb-like production reachable from `protocol` that belongs to no class, or to two classes, fails the partition check — including one introduced at frame level, where the R2 candidate could not detect it;
26. **(S1R2-B02)** a lattice value recorded with no attributable Actor, or an acceptance recorded against no `subject_digest`, is refused; removing `producer`, `executor`, `acceptor`, or the acceptance subject fails the attribution checks;
27. **(S1R3-B01)** a seventh bus verb introduced as a top-level alternative *inside* one of the six bus members — utterable on its own at statement position, inside the bus class — fails the bus class bound, as do the same shapes hidden in a nested group, behind an omittable leading element, or behind a reference to another production;
28. **(S1R3-B02)** a thirteenth `ledger_statement`, or a ledger terminal respelled so the count holds but membership does not, fails the ledger class bound; and a `Count` column in §3.3 that disagrees with the grammar fails `SPEC_CLASS_COUNTS_MATCH_GRAMMAR`.

## 12. Gate sequence

| Gate | Required result | Current state |
|---|---|---|
| M0 | Doctrine and citations reviewed | Complete |
| M1 | Fresh exact Mechanics Security + Spec `APPROVE` | Complete at R21 |
| **S1** | Consolidated spec, engineering doctrine, domain context, grammar, and declared candidate inputs pass independent exact Standards + Spec review | **Pending — twelve independently reviewed subjects, R1 through R16, have each returned at least one `REQUEST_CHANGES`; the R16 Standards axis returned `APPROVE` and its Spec axis `REQUEST_CHANGES`, which closes nothing; this bundle is the R17 repair of the R16 candidate and is unreviewed** |
| M2 | v0a tracer implemented/tested in IX over copied immutable evidence | Absent |
| M3 | Typed deterministic advisory artifact with units/nulls/provenance and out-of-sample baseline | Absent |
| M4 | Gaia exports/consumes through six verbs without automatic authority/spend/routing/new grammar | Absent |
| M5 | Agent Blackbox independently verifies fixed point, integrity, risks, and negative controls | Absent |
| M6 | Exact cross-repo Standards + Spec approvals and clean tests | Absent |
| M7 | Controlled GitHub heads/checks/mergeability | Absent |

The header milestone marker `M1 — NOT INTEGRATED` with **M2-M5 absent** is the required status statement. M6 and M7 are likewise `Absent` in the table above; the header enumerates the near gates and the table is the complete record. No gate row is advanced by this document.

The next action is **a fresh S1 independent review** over the exact bundle recorded in `gaia-s1-r2-bundle-manifest.md`, not implementation. That review SHALL explicitly test the anti-vibe negative controls, the deep-module seams, canonical terminology, the Design It Twice obligation, the vertical tracer boundary, the TDD evidence contract, the separation of review from acceptance, the six repairs S1-B1 through S1-B6, the two R3 repairs `S1R2-B01` and `S1R2-B02`, the two R4 repairs `S1R3-B01` and `S1R3-B02`, every repair from R5 through R15, and the two repairs of this revision, `S1R16-B01` and `S1R16-B02`. It SHALL treat the R17 repair's own claim of mechanical witnessability with the same suspicion the R2 through R16 reviews correctly applied to their predecessors: the honest test is a mutation the author did not write. **The previous revision of this paragraph predicted a ninth escape and predicted it in the wrong place, and that is recorded rather than quietly rewritten.** It said eight consecutive rounds had found the next escape in the §3.3 region reader and pointed the reviewer at that reader's two boundaries. The ninth escape was real and was found where the same sentence also said it would be — somewhere the author did **not** just look — but it was in the *citation* reader, not the region reader: a pattern deciding what counts as a line citation, which is the same defect shape one reader further out. The R16 Standards axis probed the region reader with twenty mutations and found nothing, which bounds that search rather than closing the family. So a reviewer should assume a **tenth** exists, should assume this paragraph is again pointing at the wrong reader, and should look first at whichever reader has no negative fixture of its own: the region reader's disclosed residuals — setext headings, fences inside block quotes and list items, HTML blocks, and a designator written in a form the pattern does not read — remain open, and so does every reader that decides what a *thing* is before another check decides whether it is true. If it approves, the decision-maker may separately authorize T1/M2 with an explicit budget and copied-evidence manifest.

## 13. Explicit non-adoptions

- Do not adopt the SSSF implementation: T0 found 0/14 required controls present, 1 unknown.
- Do not use placeholder `echo` quality gates, agent-writable ignored evidence, or `git add -A`.
- Do not install Citadel or adopt its coordinator; retain only tri-state health, digest-bound plans, drift checks, digest-only footprints, audit events, and per-artifact prototype locks.
- Do not run a second control-plane polling loop or let Claude fallback touch live panes.
- Do not authorize 7-day/30-day unattended campaigns; feasibility remains unknown/rejected.
- Do not auto-fanout across paid/model surfaces; accounting and vendor boundaries must be known.
- Do not operationalize information-theoretic or power-law metrics until their rejected research blockers close.
- Do not create a seventh bus verb, filler actor, filler lane, or approval-by-consensus path.
- Do not create a separate Wayfinder map for bounded implementation; continue the existing organization decision map only when genuine product fog remains.
- Do not declare a shared advisory seam over IX, TARS, and HARI before two real implementations demonstrate a common contract.
- Do not name any component, role, seam, or artifact "oracle".
- Do not treat an unrecognised refusal code as a pass, and do not discard it.
- Do not acquire PostgreSQL, FalkorDB, DuckDB, a graph engine, an embedding model, or a projection library on the strength of §9.6.

## 14. Provenance ledger

### 14.1 Inputs carried forward from v0.1

| Input | Bytes | SHA-256 |
|---|---:|---|
| `gaia-mechanics-agentics-r21-independent-review.md` | 71,126 | `1f89db4b83a16e29f1ce2241e93a113cd3bde8fb4d462363082aaa6f810c0c48` |
| `docs/research/2026-08-09-mechanical-tensors-agentic-systems.md` | 31,668 | `9df446ffa6b5ea2fc06d51eb29a5dbbe1bcc8732a73b45854bd57db6510183a9` |
| `gaia-interagent-g20-g21-final-r2.md` | 28,213 | `e10a600b6ca5cf89c11b3aa53e5523c22ad769d2bf32e827f6c2f805c4d66a2f` |
| `gaia-direct-agent-mode-video-research.md` | 12,061 | `e573477d102a6b2ea249b62bc32931d653fe94e4b014e60f360372fd95fac7c9` |
| `gaia-coordinator-fence-design-r1.md` | 12,190 | `1dcc180f1b13a53e6fa5be1b1d59e334d93720d44772f646169cfa3ad8bbb146` |
| `gaia-interagent-longhorizon-r4-security-review.md` | 70,793 | `cc9aafd742757d008ffc45973e22aaf84f5964822b6ffee0cb88b9e6debcf89e` |
| `gaia-interagent-multicli-r4-security-review.md` | 62,181 | `ccad7eae5c80844a31d5d65c50404d1d8ca6b9840aec5d95ffea22e5c96ae950` |
| `gaia-information-theory-r2-security-review.md` | 41,724 | `b3db3cb2722defdbfb26d1ed704d2c967cb4a71ad256ab2fdb5147a7043f8280` |
| `gaia-agentic-architecture-powerlaw-r2-security-review.md` | 35,708 | `6c9e2f06180803e65b678dbba857855add4424a623a1853e669a7fd58625b0ba` |
| `gaia-citadel-r1-independent-evaluation.md` | 61,523 | `4913a0cfce6743430dde07a0de741cd1fa34929cc5768fdd54fb8f3ae0caa173` |
| `gaia-workgraph-projection-r1-oracle.md` | 26,344 | `b4661ce197e8b3df9d2ec40362a17cd59581bb8c5e4ee07b56fcdad4d28dbea1` |
| `gaia-engineering-skills-r8-independent-review.md` | 43,856 | `2167ad385899803a1b6aa17fde94978b2bb3fee65d1bd5aff8958594200e0cc8` |
| `gaia-engineering-skills-r6-review/docs/gaia-engineering/UPSTREAM-MAP.md` | 15,343 | `22e7dfd7167fea72396ec08c95803018f3a8506f54f65be0a5570428328b9f7f` |
| `gaia-engineering-skills-r6-review/THIRD_PARTY_NOTICES.md` | 18,573 | `6277bcdf3615fe817a3ec744b14a38edc4bf11af96c29690094c2a4e181ecbe1` |
| `recursive-improvement-primary-sources-research.md` | 33,769 | `d37c1971e16e67e2a5c2b9b6b0f10bc9c80c5efbb577e38caf4b028dbf2b4bc2` |
| `gaia-working-software-factory-advanced-research.md` | 55,838 | `f3556e9079c0534e124098ab00fda695c0722853fe3c220b72b7821bbc6b94df` |
| `gaia-software-factory-t0-conformance-scorecard.md` | 5,059 | `dc670a788ae99e1bd8ee01a14e5a45546edfed07b7e9ee34a4a0f5043b8e565f` |
| `hari/README.md` | 10,398 | `ff62b0505ae16bd02dadbf9ce9e43131cc21ee123355f960f62cf3b02f84bf93` |

The file named `gaia-workgraph-projection-r1-oracle.md` retains its historical filename. It is cited as evidence only; its name does not license the term "oracle" in any Gaia component, role, seam, or artifact.

### 14.2 The exact S1 R1 subject and its verdict

| Input | Bytes | SHA-256 |
|---|---:|---|
| `gaia-s1-independent-review-r1.md` | 39,060 | `84bc1e0be2cb59837e1e0a78219a7cb4c0f150eae226db0f5a135ec2c4f6f6b2` |
| `gaia-consolidated-mission-room-factory-spec-v0.1.md` (superseded by this file) | 24,177 | `b9819c1abf56db80fd75211767f7db032c32c596601a50e778b78343f7be39aa` |
| `gaia-mission-room-protocol-v0.1.ebnf` (superseded by v0.2) | 4,888 | `62a0cc3e1e4526748e2f67bd0953d2dc19a7a1afc16902eef01bf52dc13ea4d9` |

The R1 aggregate subject digest was `f44ecac23f0c9d180f1bcdcbacb920163a89dc0d04cebbbe6825dbc556c32832`. R1's Standards `REQUEST_CHANGES` and Spec `REQUEST_CHANGES` remain the current, valid verdict over those bytes. This bundle does not mark them `SUPERSEDED`; only a fresh review binding the new exact digest may do that.

### 14.3 New candidate inputs admitted by §9.6

| Input | Bytes | SHA-256 |
|---|---:|---|
| `gaia-artifact-workgraph-context-v0.1.md` | 10,956 | `a5e3492c94feafd6eeb3a243aed1359ce121417c5069cb59be0e17f99aa0c57a` |
| `gaia-artifact-workgraph-staleness-design-v0.1.md` | 53,037 | `7b9d0cd64ad29e897f6b294b2a2c91e1a5bcb708a89a1a4871a14eba926fd219` |
| `gaia-uncertainty-grammar-v0.1.ebnf` | 10,322 | `ee9f61b6be06aed76cdc0493c95c1540c6b148c31d75801ef9b67729b7394517` |
| `gaia-multiaxis-uncertainty-math-primary-research.md` | 11,568 | `06da04bd64d8e092d71f3900ccc87806602909b1839ce46855910da79c6d8c2c` |
| `gaia-multiaxis-uncertainty-analysis-design-v0.1.md` | 10,999 | `65a32cb0ddd5a579efb17c2d5fd66c25ffa876faa9afe8dd3f192cc67ea8c6a7` |

### 14.4 Copyright and third-party provenance

The upstream engineering-skills provenance gate is unchanged and is not reopened by this revision: exact author **Matt Pocock**, MIT notice preserved, pinned upstream commit `84fdeffd12f2ee307994d1eb6feb48173b6e0502`, all eighteen source paths mapped one-to-one, an accurate retained-phrasing ledger, and no implied endorsement. The bundle-local restatement is `gaia-s1-r2-third-party-notices.md`; the normative record remains `gaia-engineering-doctrine-v0.1.md` §7 and the pinned artifacts hashed in §14.1.

Primary mechanics citations remain those pinned in the mechanical research document; this spec imports their doctrine, not new claims about physical equivalence. Primary uncertainty and geometry citations remain those pinned in `gaia-multiaxis-uncertainty-math-primary-research.md`; this spec imports their limits, not new mathematical claims.

## 15. Residual uncertainty

- Whether the v0a tracer has enough cells for any useful v0b remains unknown; current evidence says do not authorize v0b first.
- Real IX-to-HARI benchmark value is unknown.
- Whether a JEPA-style latent objective adds predictive value over simple trace baselines is unknown; V-JEPA 2 evidence is physical/video-domain evidence, not Gaia evidence.
- Multi-host coordinator failover is deferred until measured need; a replicated consensus coordinator is not part of T1.
- Long-horizon stability, vendor cost accounting, and installed-loader behavior remain outside this specification.
- The §9.6 candidate inputs enlarge the reviewable surface substantially. Whether that surface is now too large for one S1 review to cover exactly is itself an open question for the reviewer, who may split the axis.
- Five ledger statements (`lane`, `capsule`, `fence`, `artifact`, `transition`/`receipt`) are new grammar surface. They add no bus verb and no authority, but they are new bytes and have never been reviewed.
- The R3 repair itself is new grammar surface: the `frame_statement` classification production, the `executor` and `acceptor` attribution rules, and the added fields on `advisory`, `transition_receipt`, and `close`. It adds no bus verb, no lattice, no refusal code, and no authority — but these are new bytes and have never been reviewed. A reviewer should note in particular that `close` changed field order and arity, which is a wire-format change for any future consumer, and that this is disclosed rather than presented as a pure addition.
- **The R4 repair adds no grammar surface.** Every production of `gaia-mission-room-protocol-v0.2.ebnf` is byte-identical to the R3 candidate; only the normative comment text changed, to state the class bounds over derivations and to name the checks that now decide them. The change is confined to the validator, the specification, the change ledger and the manifest. **The R5 repair adds no grammar surface either, and does not touch the grammar file at all**: `gaia-mission-room-protocol-v0.2.ebnf` is byte-identical to the R4 candidate, including its comments. R5 repairs two mechanisms the R4 independent Standards and Spec reviews found weaker than the prose that named them — an unbalanced group at a class member's leading position that was read as a satisfied bound, and the §3.3 **Leading terminals** cells, which no check read — and it adds no verb, production, lattice, refusal code, authority, subsystem or runtime **The R6 repair does not touch the grammar file either**, and it adds no positive check: it repairs three mechanisms inside the two §3.3 checks R5 added, which the two R5 independent reviews found weaker than the prose that named them — a duplicate class row that shadowed the genuine one, a cell read only for its backticked names, and a set comparison where §3.3 claims a multiset. It too adds no verb, production, lattice, refusal code, authority, subsystem or runtime, and the six non-privileged bus verbs are exactly preserved.
- The v0.1 non-blocking corrections N2, N3, N5, and N7 are addressed in the R2 revision; N1, N4, and N6 are addressed in the header/§12, §2.2, and the grammar list-cardinality comment respectively. None of these was blocking, and a reviewer may disagree that any of them is closed.
- The R3 revision closes the R2 review's non-blocking observations `S1R2-N01`, `S1R2-N02`, `S1R2-N03`, `S1R2-N05`, `S1R2-N08`, and `S1R2-N09`. It deliberately leaves four open, and none of them is claimed closed:
  - `S1R2-N04` — `non_goal_set` is 1..n but `quoted_text` admits the empty string, so `[""]` satisfies the cardinality while declaring no excluded outcome. Closing it needs either a non-empty-text production or an admission-time binder rule. Both widen grammar or scope beyond the two blockers, so it is recorded rather than repaired.
  - `S1R2-N06` — `freshness_state` remains assertable by fiat in the context-free grammar; only the extra-grammatical semantic check bars a producer from writing `FRESH` directly. The asymmetry with S1-B3, which made `advisory … APPROVED` unparseable, is real and is disclosed rather than argued away.
  - `S1R2-N07` — the `REQUIRED_ENTITIES` set is author-selected. Its one protocol-referenced exception, Acceptance Decision, is now attributed through `acceptor` and bound to a subject digest, but it still carries no typed identity of its own.
  - `LEDGER_LINE_REFS_IN_RANGE` still tests only that a cited line number falls inside the file. The R3 revision adds `LEDGER_ANCHORS_RESOLVE`, which asserts that **forty-five** load-bearing citations resolve to a line *containing the claimed text* — sixteen at R3, nine added at R4 for the claims that revision rested on, one added at R13 binding the §3.3 scope clause at `spec-v0.2:103` that the region reader now depends on, and nineteen added at R17 binding the twenty citations that round corrected, one of which was already anchored — the content half the R2 review asked for, and a check that caught four citations this author got wrong while writing the R4 revision. But that anchor table is **author-selected and partial**: it covers the citations that carry the argument, not all **239** `spec-v0.2:`/`ebnf-v0.2:` citations the change ledger carries, a figure `LEDGER_LINE_REFS_IN_RANGE` prints on every run. **That census is now the ledger's content and not a pattern's yield, which it was not before R17.** The figure standing here was 203, which was the count of the spans one pattern happened to match; the ledger also carried five comma-list spans that pattern could not match, so twenty line numbers were read by no check and sixteen of them were wrong — the defect the R16 Spec review reported as `S1R16-B01`. Those spans are normalised, those numbers are corrected, and `LEDGER_CITATIONS_CANONICAL` now refuses any citation-shaped span the range reader would not read, so the two figures are the same set by construction rather than by coincidence. Both figures are the ones measured over **these** bytes and are re-derived each round rather than carried forward; the standing "twenty-five" and "seventy-odd" here were stale over the R14 subject, which already carried twenty-six and 167, and the R14 Spec review reported that as `S1R14-B01` on its axis. A future edit could still reintroduce the defect class outside the anchored set.
- **The class bounds are scoped to leading position, and this is a real boundary, not a formality.** `PROTOCOL_PARTITION_EXHAUSTIVE` classifies every production a conforming utterance can present *as a statement*, and the three `*_CLASS_LEADING_TERMINALS_BOUND` checks bound each class by the leading terminals of its derivations. None of them classifies a production reachable from `protocol` at *field* or *continuation* position. The author probed this deliberately: adding `escalation = "escalate" , task_id ;` as a trailing field of the `fence` ledger statement passes the full suite. **The counts previously stated here were stale and are corrected**: that probe was reported as passing "50/50 positive and 38/38 negative" when the R3 suite it ran against was 51 positive and 40 negative — an inaccuracy inside the paragraph claiming probing rigour, recorded by the R3 review as `S1R3-N05(a)`. It is replayed in this revision as boundary control `POS-FIELD-d` and passes at the counts stated below. That the boundary is defensible — such a production cannot be uttered on its own, so it is a field of `fence` and not a seventh verb — is unchanged, but a reviewer who considers field-level widening to be smuggled surface should treat this as an open gap and say so. It is disclosed here rather than discovered later, and the R3 reviewer, having tested it independently, accepted it as disclosed while recording that a `handoff`-shaped field carrying a bus-verb-like terminal deserves a binder rule at M2.
- **The boundary is now exercised, not only asserted.** Four disclosed-boundary positive controls (`POS-FIELD-a`..`POS-FIELD-d`) mutate at field or continuation position inside a bus, ledger and frame statement, and the run fails if any of them trips a class bound. A control that trips would mean the repair silently widened the disclosed gap. This makes the *scope* of the repair mechanically checkable in the same way its *detection* is; it does not close the gap and does not claim to. Those four are the **field-position family only**, not the whole control set: the shipped suite carries **thirteen** positive controls in four families — `POS-FIELD-a`..`-d` for the field-position boundary, `POS-PIPELESS-a` for the pipe-less GFM row rendering, `POS-FENCE-a`..`-d` for fenced-block context and the closing edge of the declared §3.3 scope, and `POS-HEAD-a`..`-d` for its opening edge and for the sibling section that must stay outside it. Earlier revisions of this paragraph's neighbour below reported the four as if they were the whole, which the R14 Spec review reported as `S1R14-B02`.
- The R4 repair's own mechanical claim is exactly as strong as the mutations that test it. Eleven new negative fixtures were each verified to pass the **pre-repair** suite in full — 51 positive checks and 40 fixtures, nothing detected, no fixture rendered inapplicable — and to be caught by their named check after it. That is the strongest form of this evidence the author can produce, and it is still author-written. The R2 review broke the R1 claim and the R3 review broke the R2 repair's successor; an independent reviewer should assume the same of this one and write a mutation the author did not anticipate. **And an R4 reviewer did exactly that, on both axes.** The R5 repair holds itself to the same standard: its five new negative fixtures were each verified to pass the **pre-R5** suite in full — 56 positive checks and 51 fixtures, nothing detected, no fixture rendered inapplicable — and to be caught by their named check after it. **And two independent R5 reviewers did it again, on both axes, and this repair is the answer to what they wrote.** The R6 repair holds itself to the same standard once more: each of its six new negative fixtures was first run against the **R5 bytes**, where it is a false green at exit 0, and is caught by its named check on these bytes; and four legitimate re-renderings of a §3.3 cell were checked to confirm the wider reading does not false-red the two §3.3 checks. **Every round from R7 through R17 has held itself to that same standard.** Each round from R7 through R15 found the next escape in the same §3.3 region reader; the R17 round is the first whose repaired defect was in a different reader altogether, the ledger citation reader, and its new negative fixture `NEG-L-e` was verified to be a false green at exit 0 over the R16 bytes before it was caught here. The shipped suite over **these** bytes is **58 positive checks, 95 negative fixtures and 13 disclosed-boundary positive controls**, in the four control families named above, and those three figures are the ones the captured run in `gaia-s1-r2-validator-output.txt` prints. They are re-derived every round rather than carried forward: the figure standing here was **57, 94 and 13**, the R16 suite, which this round moves by adding one check and one fixture and no control; two rounds before that the same sentence carried **57, 62 and 4**, the R6 suite, already three rounds stale over the R14 subject's own 57, 93 and 12 — the R14 Spec review reported that as `S1R14-B02`. The same warning applies again and with the same force: the next blind spot is where the author did not just look.
- **Nullability at statement position is bounded only where it makes a leading terminal undecidable.** `CLASS_LEADING_TERMINALS_RESOLVE` refuses an alternative every element of which is omittable, because such an alternative has no leading terminal. It does **not** otherwise assert that a class member is non-nullable, and nullability is not a leading-terminal property. This is recorded, not repaired: a non-nullability rule would widen the mechanism beyond the two blockers it was authorized to close.
- **`S1R3-N07` is closed for the fixture harness only.** A fixture or control whose text anchor no longer matches is now reported as `INAPP` and counted separately from "not caught", so anchor drift can no longer read as detection, and lost coverage still fails the run. The R3 review reached this finding through `M6b`/`M7b`, whose red runs were anchor drift rather than detection. The other six non-blocking R3 observations — `S1R3-N01` through `S1R3-N06` — are **recorded and left open**, deliberately and without any claim of closure, because repairing them would widen beyond `S1R3-B01` and `S1R3-B02`. In particular `S1R3-N04` stands: `COPYRIGHT_GATE_PRESERVED` does not bind the text of the preserved notice, the pinned artifact digests, or the eighteen upstream source identities, and the reviewer's `C1`–`C5` mutations are all still undetected in these bytes.
- A fresh S1 may find further contradictions or missing bindings. **This author does not self-approve, and this document grants no approval to itself, to the candidate inputs of §9.6, or to any next effect.**

`GAIA_CONSOLIDATED_MISSION_ROOM_FACTORY_SPEC_V0_2_COMPLETE`
