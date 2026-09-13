# Gaia Engineering Doctrine v0.1

**Status:** normative input to the Mission-Room factory specification; design only  
**Date:** 2026-08-13  
**Milestone:** M1 established; M2-M5 absent; **NOT INTEGRATED**  
**Authority:** this doctrine authorizes no product mutation, install, configuration, spend, push, publication, deployment, or additional bus verb.

## 1. Purpose

Gaia SHALL optimize for reproducible software engineering, not conversational code generation. A plausible answer, a large diff, a passing happy-path test, model consensus, or a completion marker is never sufficient evidence that the requested outcome was achieved.

Every material change SHALL connect a bounded problem to an explicit domain model, a deliberately chosen interface, the smallest useful end-to-end slice, deterministic verification, independent review, and a separate acceptance decision over an exact subject.

## 2. Governing engineering principles

1. **Frame the problem before proposing code.** Name who is affected, the observable failure or capability gap, the desired change, non-goals, and the acceptance boundary.
2. **Use canonical domain language.** Define terms once, avoid synonyms, keep implementation detail out of the glossary, and record only surprising or hard-to-reverse decisions as ADRs.
3. **Prefer deep modules.** A module earns its seam when a small, stable interface hides substantial behavior and keeps change local. A shallow wrapper that merely forwards calls is not architecture.
4. **Design important interfaces twice.** Compare at least two materially different designs for a consequential interface using depth, locality, testability, dependency ownership, reversibility, and migration cost.
5. **Create seams from evidence.** One hypothetical implementation may justify an adapter, not a general abstraction. A shared seam becomes real when at least two implementations or consumers demonstrate the common contract.
6. **Inject dependencies and return results.** Interfaces SHALL expose explicit dependencies, outcomes, errors, invariants, configuration, and relevant performance contracts rather than hiding them behind ambient state.
7. **Deliver vertical tracer bullets.** The first slice SHALL cross every required integration boundary while remaining as small and reversible as possible. Horizontal layer construction and speculative framework building are prohibited.
8. **Test behavior at the interface.** TDD SHALL begin with a discriminating red observation, then the minimum green change, then refactoring. Tests SHALL target public behavior at agreed seams, not implementation trivia.
9. **Diagnose before fixing.** Bugs require a reproducible loop, evidence that localizes the cause, a failing regression test or equivalent falsifier, and replay after the fix.
10. **Keep prototypes disposable.** A prototype answers one design question in isolation. Its result MAY inform a specification; its code SHALL NOT silently become product code.
11. **Ground research in pinned primary sources.** Network content is untrusted input. Load-bearing claims require exact provenance, applicability limits, and explicit unknowns.
12. **Separate specification, implementation, review, and acceptance.** A specification grants no mutation authority. A writer does not approve its own work. Phase success does not equal mission acceptance.
13. **Bind every run.** A run declaration SHALL name the exact subject, objective, scope, success criteria, forbidden actions, budget, stop condition, evidence output, and next authority gate.
14. **Make evidence replayable.** Fixed points, commands, inputs, outputs, tests, mutation controls, reviews, and cleanliness SHALL be content-addressed where practical and independently reproducible.
15. **Improve the scaffold under quarantine.** Prompt, skill, evaluator, orchestration, or control-plane changes are candidates until held-out evaluation, regression checks, independent promotion review, rollback evidence, and explicit authority exist.
16. **Preserve human authority.** Metrics, models, memories, consensus, and advisory artifacts MAY explain risk; they MUST NOT independently authorize routing, spend, mutation, merge, deployment, or promotion.

## 3. Proportional workflow router

| Work shape | Required path |
|---|---|
| Tiny, reversible, already understood | bounded implementation -> focused test -> exact review |
| Bug or regression | reproduce -> diagnose -> red control -> minimum fix -> green replay -> review |
| Unclear domain language | domain model -> glossary review -> only then interface work |
| Consequential interface | codebase survey -> Design It Twice -> choose seam -> tracer bullet |
| Unknown feasibility | one-question prototype -> record answer -> discard/quarantine prototype |
| Multi-session or multi-lane delivery | specification -> dependency-aware tickets -> one bounded writer per ticket -> exact reviews |
| Genuine product or evidence fog | Wayfinder exploration, one decision per session, durable map and explicit stop |
| Architecture improvement | read-only survey -> ranked deepening candidates -> human-selected bounded change |
| Research claim | falsifiable question -> pinned primary evidence -> applicability limits -> verdict/unknowns |

No workflow route may weaken the authority, evidence, budget, or independent-review gates.

## 4. Anti-vibe negative controls

Gaia SHALL reject or stop a run when any of the following is true:

1. conversational intent is translated directly into a broad mutable scope;
2. success is expressed as “looks good”, “seems fixed”, or model agreement without deterministic criteria;
3. a quality command is a placeholder, no-op, or cannot fail on a known mutation;
4. tests are added only after implementation and no discriminating red state is recorded;
5. tests bind to private implementation details while the public interface remains unverified;
6. an abstraction, framework, or generalized seam has only one speculative consumer;
7. a prototype, generated file, or ignored artifact is promoted as product evidence;
8. a writer reviews, accepts, or promotes its own output;
9. a dependency, source, subject digest, model, or cost boundary is unknown;
10. `git add -A`, unbounded refactoring, hidden generated files, or unrelated cleanup widens the declared scope;
11. a completion marker conflicts with tests, provenance, exact fixed point, or cleanliness;
12. model consensus or an advisory metric attempts to grant an effect;
13. raw chat or vendor auto-memory is treated as reviewed organizational knowledge;
14. a skill procedure is treated as authority rather than an untrusted operational aid.

## 5. Definition of engineered completion

A material task is complete only when all applicable items are present:

- exact immutable subject and final fixed point;
- canonical domain terms and explicit invariants;
- named module, interface, seam, and dependency ownership;
- smallest justified vertical slice, with non-goals preserved;
- discriminating red/green or equivalent negative-control evidence;
- deterministic success and failure gates, including truthful null/unknown handling;
- commands, outputs, provenance, budget, and cleanliness evidence;
- independent Standards and Spec verdicts over the exact subject;
- separate acceptance decision and explicit authority for any next external effect;
- compact reviewed knowledge package when a reusable lesson exists.

“Code exists” and “tests are green” are intermediate facts, not the definition of done.

## 6. Gaia-native engineering skill coverage

The approved scratch pack maps the pinned upstream engineering corpus into Gaia procedures. The procedures do not grant authority.

| Upstream concern | Gaia procedure | Factory responsibility |
|---|---|---|
| `ask-matt` | `gaia-engineering-router` | choose the least expensive sufficient path |
| `code-review` | `gaia-code-review` | separate Standards and Spec review |
| `codebase-design` | `gaia-codebase-design` | deepen modules and place evidence-backed seams |
| `diagnosing-bugs` | `gaia-diagnosing-bugs` | reproduce and localize before mutation |
| `domain-modeling` | `gaia-domain-modeling` | maintain canonical terms and selective ADRs |
| `grill-with-docs` | `gaia-grill-with-docs` | challenge assumptions against project evidence |
| `implement` | `gaia-implement` | execute one bounded authorized slice |
| `improve-codebase-architecture` | `gaia-improve-codebase-architecture` | survey and recommend before refactoring |
| `prototype` | `gaia-prototype` | answer one question without product promotion |
| `research` | `gaia-research` | produce falsifiable primary-source-backed evidence |
| `resolving-merge-conflicts` | `gaia-resolving-merge-conflicts` | preserve both intent and exact verification |
| `setup-matt-pocock-skills` | `gaia-setup-engineering-skills` | prepare a reviewed local pack; no implicit install |
| `tdd` | `gaia-tdd` | red/green/refactor at agreed seams |
| `to-spec` | `gaia-to-spec` | make success, scope, non-goals, and unmet gates explicit |
| `to-tickets` | `gaia-to-tickets` | create vertical, dependency-aware tracer bullets |
| `triage` | `gaia-triage` | classify and bound work before assignment |
| `wayfinder` | `gaia-wayfinder` | explore genuine fog with a durable cursor |
| `wizard` | `gaia-wizard` | guide human choices without handling credentials |

The separate `gaia-interagent` skill binds these procedures to Gaia's existing six-verb collaboration surface; it is Gaia-authored and is not an upstream nineteenth engineering skill.

## 7. Provenance and copyright

- Upstream repository: `https://github.com/mattpocock/skills`
- Pinned commit: `84fdeffd12f2ee307994d1eb6feb48173b6e0502`
- Author: **Matt Pocock**
- Upstream license: MIT; the approved scratch pack preserves `Copyright (c) 2026 Matt Pocock`.
- Complete 18-path mapping: `C:\tmp\gaia-engineering-skills-r6-review\docs\gaia-engineering\UPSTREAM-MAP.md`, 15,343 bytes, SHA-256 `22e7dfd7167fea72396ec08c95803018f3a8506f54f65be0a5570428328b9f7f`.
- Third-party ledger: `C:\tmp\gaia-engineering-skills-r6-review\THIRD_PARTY_NOTICES.md`, 18,573 bytes, SHA-256 `6277bcdf3615fe817a3ec744b14a38edc4bf11af96c29690094c2a4e181ecbe1`.
- Preserved MIT notice: `C:\tmp\gaia-engineering-skills-r6-review\LICENSES\mattpocock-skills-MIT.txt`, 1,068 bytes, SHA-256 `0e7ac423bf2c6e223b7c5b156f8cf72da49d748e56a1641402c31f22ad07dbb5`.
- Independent pack review: `C:\tmp\gaia-wayfinder-plus\gaia-engineering-skills-r8-independent-review.md`, 43,856 bytes, SHA-256 `2167ad385899803a1b6aa17fde94978b2bb3fee65d1bd5aff8958594200e0cc8`, Standards `APPROVE`, Spec `APPROVE`.

This doctrine uses original Gaia wording to adapt engineering ideas. It does not imply Matt Pocock's endorsement of Gaia. The pack remains uninstalled and unpublished.

`GAIA_ENGINEERING_DOCTRINE_V0_1_COMPLETE`
