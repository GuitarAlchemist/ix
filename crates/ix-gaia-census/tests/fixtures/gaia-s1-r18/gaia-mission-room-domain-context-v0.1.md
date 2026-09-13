# Gaia Mission-Room Domain Context v0.1

**Status:** normative glossary candidate; design only  
**Date:** 2026-08-13  
**Milestone:** M1 established; M2-M5 absent; **NOT INTEGRATED**  
**Rule:** definitions describe the domain, not an implementation. Use the canonical term in specifications, tickets, evidence, tests, and reviews.

## Identity and collaboration

### Agent Profile

The durable logical identity to which capability claims and reviewed Competency Evidence belong.

Avoid: “agent” when the text actually means a process, pane, tab, model, or vendor session.

### Agent Incarnation

One concrete process or session acting for an Agent Profile. It may disappear and be replaced without changing the profile.

Avoid: worker identity, surface identity.

### Competency Evidence

Content-addressed evidence that an Agent Profile performed a named bounded capability under declared conditions and review. It is not a reputation or authority score.

Avoid: skill score, trust score, Elo.

### Mission Room

A temporary collaboration boundary for one objective, membership set, evidence boundary, budget, and acceptance contract.

Avoid: team, group, channel, or swarm when Mission Room is meant.

### Role Assignment

A temporary responsibility held by an Agent Incarnation within one Mission Room and epoch. It does not imply global permission.

Avoid: persona, permanent role.

### Lane

A live execution context bound to one Agent Incarnation, Role Assignment, work scope, and Lease. A pane, tab, shell, or historical surface is not a Lane by itself.

Avoid: pane, tab, session when Lane is meant.

## Work and authority

### Task Capsule

The portable work contract containing immutable base, objective, success criteria, non-goals, write scope, dependencies, exact verification, forbidden effects, budget, and evidence references.

Avoid: prompt, ticket, handoff when the full contract is meant.

### Claim

The recorded ownership of one bounded work item by an Agent Incarnation for the duration of a Lease.

Avoid: lock when the concept includes ownership and evidence scope.

### Lease

The time- and epoch-bounded validity of a Claim or authority assignment. Expiration makes later effects inadmissible.

Avoid: timeout when validity, not waiting behavior, is meant.

### Coordinator Fence

The mutation capability binding coordinator lineage, epoch, evidence pre-image, preconditions, intent, and bounded effect. A stale or divergent fence must fail closed.

Avoid: coordinator token, leader flag.

### Acceptance Decision

The explicit decision that an exact Evidence Bundle satisfies the Mission Room's acceptance contract. It is separate from phase success, review completion, and process exit.

Avoid: done, complete, green when acceptance is meant.

## Evidence and knowledge

### Evidence Bundle

The content-addressed inputs, commands, outputs, tests, controls, reviews, provenance, costs, and cleanliness facts for one Claim.

Avoid: log, report, marker when the complete evidentiary set is meant.

### Advisory Artifact

A deterministic, provenance-bearing result that may inform a decision but cannot grant authority or produce an external effect by itself.

Avoid: verdict when the artifact is non-authoritative.

### Knowledge Package

A compact, reviewed, versioned lesson derived from Evidence Bundles and suitable for later retrieval. Raw chat and vendor memory are not Knowledge Packages.

Avoid: memory, transcript.

## Software design

### Module

A behavior-bearing boundary with an interface and a hidden implementation.

Avoid: file or class when the boundary spans more or less than either.

### Interface

The complete contract a consumer must understand, including operations, data shapes, invariants, dependencies, errors, configuration, and relevant performance behavior.

Avoid: method signature when the broader contract is meant.

### Seam

A deliberate substitution or verification boundary justified by multiple implementations, multiple consumers, or a named external ownership boundary.

Avoid: abstraction point when no evidence justifies the abstraction.

### Adapter

A local translation at a boundary that isolates an external or replaceable dependency without pretending a general shared abstraction already exists.

Avoid: integration layer when a narrow adapter is meant.

## Working rules

1. Specifications introduce or reuse canonical terms from this glossary.
2. Tickets and Task Capsules link each material noun to its canonical term.
3. Tests use the same terms for observable states and errors.
4. Reviews identify synonym drift as a Spec defect when it changes meaning.
5. Implementation details belong in architecture documents, not in this context.
6. A new term requires a distinct domain concept; a new synonym is not a contribution.

`GAIA_MISSION_ROOM_DOMAIN_CONTEXT_V0_1_COMPLETE`
