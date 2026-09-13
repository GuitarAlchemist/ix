# Gaia S1 R2 — Third-party notice and source ledger

**Status:** bundle-local restatement of an already-accepted provenance gate
**Date:** 2026-08-13
**Revision:** R3 repair of the R2 candidate
**Milestone:** M1 established; M2-M5 absent; **NOT INTEGRATED**

This file exists so that an independent reviewer of the bundle can check the copyright gate without leaving the bundle. It **reopens nothing**. The S1 R1 review returned `PASS` on the copyright and provenance axis with no blocking finding, and the independent review of the R2 candidate returned `PASS` on the same axis, recording the author attribution, the MIT notice, the pinned commit, the eighteen-path mapping, the retained-phrasing ledger, and the absence of endorsement as **exact**. The constraints below are carried forward unchanged and unnegotiated.

The normative record remains `gaia-engineering-doctrine-v0.1.md` §7 (copied into this bundle byte-exact) together with the two pinned upstream-mapping artifacts hashed in §3 below.

## 1. Upstream work and licence

- **Upstream repository:** `https://github.com/mattpocock/skills`
- **Exact author:** **Matt Pocock**
- **Pinned upstream commit:** `84fdeffd12f2ee307994d1eb6feb48173b6e0502`
- **Upstream licence:** MIT
- **Preserved notice:** `Copyright (c) 2026 Matt Pocock`, retained verbatim in `LICENSES/mattpocock-skills-MIT.txt` (1,068 bytes, SHA-256 `0e7ac423bf2c6e223b7c5b156f8cf72da49d748e56a1641402c31f22ad07dbb5`), recorded as a byte-exact copy of the upstream `LICENSE`.
- **Licence compatibility:** MIT to MIT.
- **Endorsement:** **none claimed or implied.** No affiliation, sponsorship, endorsement, or review by Matt Pocock is asserted anywhere in this bundle. The Gaia doctrine states this directly at `gaia-engineering-doctrine-v0.1.md:123`.
- **Distribution state:** the derived engineering-skills pack remains **uninstalled and unpublished**. Nothing in this bundle installs, registers, publishes, or executes it.

## 2. The eighteen upstream sources

Eighteen upstream sources map one-to-one onto eighteen Gaia procedures. The mapping is reproduced here from `gaia-engineering-doctrine-v0.1.md` §6, which is in this bundle byte-exact.

| # | Upstream source | Gaia procedure |
|---:|---|---|
| 1 | `ask-matt` | `gaia-engineering-router` |
| 2 | `code-review` | `gaia-code-review` |
| 3 | `codebase-design` | `gaia-codebase-design` |
| 4 | `diagnosing-bugs` | `gaia-diagnosing-bugs` |
| 5 | `domain-modeling` | `gaia-domain-modeling` |
| 6 | `grill-with-docs` | `gaia-grill-with-docs` |
| 7 | `implement` | `gaia-implement` |
| 8 | `improve-codebase-architecture` | `gaia-improve-codebase-architecture` |
| 9 | `prototype` | `gaia-prototype` |
| 10 | `research` | `gaia-research` |
| 11 | `resolving-merge-conflicts` | `gaia-resolving-merge-conflicts` |
| 12 | `setup-matt-pocock-skills` | `gaia-setup-engineering-skills` |
| 13 | `tdd` | `gaia-tdd` |
| 14 | `to-spec` | `gaia-to-spec` |
| 15 | `to-tickets` | `gaia-to-tickets` |
| 16 | `triage` | `gaia-triage` |
| 17 | `wayfinder` | `gaia-wayfinder` |
| 18 | `wizard` | `gaia-wizard` |

**Count: exactly 18. No upstream folder is unmapped, and no nineteenth upstream skill is claimed.**

The exact repository-relative path of each of the eighteen sources at the pinned commit is enumerated in `UPSTREAM-MAP.md`, hashed in §3. That file was **not** among this lane's trusted inputs, so this ledger reproduces the identifiers and the one-to-one relation from the doctrine rather than restating path strings it has not read. A reviewer verifying paths should verify `UPSTREAM-MAP.md` at its pinned digest directly.

`gaia-interagent` is Gaia-authored. It binds these procedures to Gaia's existing six-verb collaboration surface and is **not** an upstream nineteenth engineering skill.

## 3. Pinned provenance artifacts

These artifacts are referenced by exact digest. They are not copied into this bundle because they were not among this lane's trusted read-only inputs. Their digests are carried forward unchanged from `gaia-engineering-doctrine-v0.1.md` §7 and were independently reproduced byte-exact by the S1 R1 lane.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `UPSTREAM-MAP.md` (complete 18-path mapping) | 15,343 | `22e7dfd7167fea72396ec08c95803018f3a8506f54f65be0a5570428328b9f7f` |
| `THIRD_PARTY_NOTICES.md` (retained-phrasing ledger) | 18,573 | `6277bcdf3615fe817a3ec744b14a38edc4bf11af96c29690094c2a4e181ecbe1` |
| `LICENSES/mattpocock-skills-MIT.txt` (preserved MIT notice) | 1,068 | `0e7ac423bf2c6e223b7c5b156f8cf72da49d748e56a1641402c31f22ad07dbb5` |
| `gaia-engineering-skills-r8-independent-review.md` (Standards `APPROVE`, Spec `APPROVE`) | 43,856 | `2167ad385899803a1b6aa17fde94978b2bb3fee65d1bd5aff8958594200e0cc8` |

## 4. Retained-phrasing ledger

The retained-phrasing disclosure is **accurate and unchanged**. Ten passages of derivative reworking are disclosed in `THIRD_PARTY_NOTICES.md` §1.1/§1.3, each quoted against its upstream source, each attributed to Matt Pocock with the pinned commit and the MIT notice. Their measured run lengths, in words, are:

`32, 28, 25, 21, 21, 20, 19, 16, 15, 15`

The longest retained run is 32 words, including punctuation and bold emphasis. The ledger applies its own minimality rule to itself and records its own measurement limits.

**Verification status, stated exactly.** This lane did **not** re-measure these figures: `THIRD_PARTY_NOTICES.md` was not among its trusted inputs. The figures are carried forward from two independent verifications already on the record:

- the independent R8 pack review confirmed them accurate at the pinned commit, **delta 0 across all 28 measured figures**, which also closes the earlier caveat that runs had been measured against cached releases rather than the pinned commit;
- the independent S1 R1 lane recomputed the digests of all four artifacts in §3 and reproduced them byte-exact, and returned `PASS` on this axis.

A reviewer who wants a third independent measurement must read `THIRD_PARTY_NOTICES.md` at the digest in §3. This ledger does not assert a measurement it did not perform.

**This remains true at R3.** `THIRD_PARTY_NOTICES.md` was not among the R3 lane's trusted inputs either, so the run lengths above are still carried forward rather than re-measured, and are still recorded as *accurate as declared and correctly scoped* rather than as independently re-measured here. The independent review of the R2 candidate reached the same conclusion and, correctly, also declined to re-measure bytes outside its subject. Three lanes have now declined to assert this measurement; none of them has made it. That is the honest state of the claim.

## 5. What this bundle adds, and what it does not

This bundle's own prose is original Gaia wording. It adapts engineering ideas; it does not copy upstream text.

The S1 R2 repairs (S1-B1 through S1-B6) touch the consolidated specification and the protocol grammar only. The R3 repairs (`S1R2-B01` and `S1R2-B02`) touch the consolidated specification, the protocol grammar, the change ledger, and the validator only. **No repair modifies, weakens, reinterprets, or reopens any element of this provenance gate:** not the author attribution, not the MIT notice, not the pinned commit, not the eighteen-path mapping, not the retained-phrasing ledger, and not the absence of endorsement.

Stated concretely for the R3 repair, because "nothing changed" is a claim a reviewer should be able to check rather than accept: classifying `mission` and `close` as frame statements, bounding the frame class, adding `producer` to `advisory`, `executor` to `transition_receipt`, and `acceptor` plus `subject_digest` to `close`, and adding a second EBNF reader to the validator, are all changes to Gaia-authored protocol and tooling. None of them reads, derives from, restates, or redistributes upstream text. The seven copied inputs — including the byte-exact doctrine that carries the normative provenance record at §7 — are **unmodified**, and the validator re-verifies all seven against their declared digests on every run via `TRUSTED_COPIES_BYTE_EXACT`.

One naming change is recorded for completeness because it touches a filename that appears in the provenance ledger. The repair for S1-B2 removes the term "oracle" from every Gaia component, role, seam, and artifact. The historical evidence file `gaia-workgraph-projection-r1-oracle.md` **keeps its filename**, because renaming a content-addressed evidence artifact would break its digest and its provenance. It is cited as evidence; its name licenses no Gaia component name.

## 6. Open item disclosed

`gaia-engineering-doctrine-v0.1.md` §7 cites the R8 review as Standards `APPROVE` / Spec `APPROVE` — independently confirmed accurate — but does not surface that R8 §6 records four residual findings, `F-R8-1` through `F-R8-4`, which remain **open**. Nor does it note that R8 supersedes the earlier measurement caveat described in §4 above.

This is disclosed here rather than by editing the doctrine, because the doctrine is carried into this bundle **byte-exact** and altering it would break the digest an independent reviewer must check. The disclosure discharges `gaia-engineering-doctrine-v0.1.md` §2 principle 11, which requires explicit unknowns on load-bearing claims. It was a non-blocking correction in S1 R1 and remains non-blocking.

## 7. Authority

This file grants nothing. It is a provenance restatement, not a licence grant, not an acceptance decision, and not an approval of the bundle that contains it.

`GAIA_S1_R2_THIRD_PARTY_NOTICES_COMPLETE`
