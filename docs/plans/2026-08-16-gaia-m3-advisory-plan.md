# Gaia M3 — `ix-gaia-advisory`, S0–S11 freeze record

**Document class:** successor freeze artifact for the approved M3 preregistration. It records the filled
frozen-input block, the reversibility analysis, and every disclosure the implementation owes.
**Lane:** W — writer/implementer, slices **S0–S11 only**. Lane W stops at S11.
**Date:** 2026-08-16
**Authority:** none. This document records measurements and a freeze. It approves nothing, accepts nothing,
and authorizes nothing. A marker is evidence only, never approval.

> **M2 verified** — by fresh independent dual `APPROVE` over the immutable 26-file subject.
> **M3 pending** — S0–S11 are implemented and frozen here; S12 and S13 are **not executed**.
> **M4 (export/consume) and M5 (Agent Blackbox verification) absent. Gaia NOT INTEGRATED.**

---

## 1. Inputs, by exact declared identity

| Input | Bytes | SHA-256 |
|---|---:|---|
| Approved preregistration `gaia-m3-advisory-preregistration-r4.md` | 180,720 | `8406bfa503e74e8e8c16813742e5bd43ac222fa4ed3acd5f5787748dcad13758` |
| Independent Spec review (`APPROVE`) `gaia-m3-preregistration-r4-spec-review.md` | 31,459 | `82543e54181d22ad1fd39fda099f2e3115d17aea0154b9f1c213506137d61e23` |

**The preregistration's SHA-256 at freeze time is `8406bfa503e74e8e8c16813742e5bd43ac222fa4ed3acd5f5787748dcad13758`.**
It is recorded here rather than inside the preregistration's own block, because a field that records a
document's hash cannot live inside the document whose hash it records.

**Verified M2 subject:** `C:\tmp\gaia-m2-ix-v0a-r2-review-a3e68957-20260815T2023Z` — 26 files / 747,205 bytes /
`IX-AGG-1` `a3e6895759d007c45d7311c9b05387befdf22212cc64671c6a94cd731dc4c235`. Re-measured before the first
edit and again after the last test run; **unchanged both times**.

---

## 2. Frozen-input block, filled

```
# --- known now, re-measured by this lane before the first edit ---
subject_aggregate_ix_agg_1            = a3e6895759d007c45d7311c9b05387befdf22212cc64671c6a94cd731dc4c235
subject_files / subject_bytes         = 26 / 747205
evidence_aggregate_ix_agg_1           = 217815fc65ef08c6d72dbe7211129ee60cc73531f2050c6e59788e1a40324f9b
declared_bundle_aggregate_gaia_agg_1  = e8a7d0927b03e9b9379892fb00bdb9ac132c760a39b0218fd7423d5b7122598e
evidence_files / evidence_bytes       = 14 / 663567
manifest_sha256                       = 275e5b85ede1a97e9d301c850f1731cdb3d2a24d11f6bb303111dfcb6f010fae
spec_sha256 / bytes / lf / cr         = 93b836e7…67660 / 77673 / 579 / 0     # F-RESTAMP target
ledger_sha256 / bytes / lf / cr       = 8a641eae…29416 / 206581 / 913 / 0    # F-STALE-FIGURE target
enumerated_cases                      = 26
predicted_admitted / predicted_rejected = 25 / 1                            # falsifier X2
predicted_truth_balance               = 1 : 24                              # falsifier X3
f_crlf_sites                          = 14  @ one per target file
f_debris_sites                        = 1   @ the instance manifest L101 identifies
f_debris_staged_file                  = "gaia-s1-r2-validator.py.stdout"
f_restamp_sites                       = 7   @ bytes 585, 2257, 5897, 54292, 55604, 68932, 73412
f_restamp_head_visible_w1024 / w4096  = 1 of 7 / 2 of 7
f_restamp_tail_visible_w1024 / w4096  = 0 of 7 / 0 of 7
f_stale_figure_sites                  = 3   @ ledger lines 552, 556, 646
f_stale_figure_token_offsets          = 117421, 119504, 145193, 145448      # 4 tokens, 3 sites
f_stale_figure_window_visible         = 0 of 3 at every swept w
f_pristine_sites                      = 1   @ the unmodified root
swept_window_bytes                    = [0, 1024, 4096]
wrt_fully_covered_files_w1024 / w4096 = 0 / 1
case_id_salt                          = "gaia-m3-case-v4"
holdout_fingerprint_budget_K          = 1
oos_primary_reason                    = NoEligibleUnseenPopulation

# --- filled at S11 by Lane W, before any Class-Holdout byte exists ---
rule_source_digest             = 668cc15761dc7c3924c425ce55d31be0fc73855d0d9f62a50b8e4eec6173a1dc
candidate_fingerprint  w=0     = 24a0c60e4a189ad12c50eb7db729ec619d125c1bf156d7d1732906c324fcb712
candidate_fingerprint  w=1024  = af886b20232ec000ca7ac4c4aacd15a905253ffb20d70dcdb361cb563d0c880c
candidate_fingerprint  w=4096  = e9273462d2a41d20c29abf8d1a8f7eceab7bc6c02737b8e58859cffa4e729f25
window_reference_digest w=0    = c2065b703d7313f4326b6a4234ed3c46430d435ea4b7058142150f892b420097
window_reference_digest w=1024 = 18e22ee37987de91ea1ebe7580b7c01f2e764c84f6f34d06cce5c1868ae43d93
window_reference_digest w=4096 = 92c574372a98196d45a7fc9ca7bee8e08bb60d7425db902429616473e86ac09c
corpus_digest_ix_agg_1         = 2304719d01c4ae698109154fe56bd1cd394d6422afcce8a8110f60a1677f95a6
admitted_cases                 = 25          rejected_cases = 1
truth_label_balance            = 1 : 24
this_document_prereg_sha256    = 8406bfa503e74e8e8c16813742e5bd43ac222fa4ed3acd5f5787748dcad13758

# --- would be filled only if the eligibility trigger fires; NOT part of M3-as-selected ---
holdout_corpus_digest_ix_agg_1 = <NOT APPLICABLE — no eligible Class-Holdout population exists>
ledger_digest_ix_agg_1         = <NOT APPLICABLE — ledger is empty>
```

**Order in which the S11 values were written.** `rule_source_digest` first, because both the candidate
fingerprints and nothing else depend on it; then the three `candidate_fingerprint` values in ascending width;
then the three `window_reference_digest` values in ascending width; then `corpus_digest_ix_agg_1`; then
`admitted_cases` / `rejected_cases`; then `truth_label_balance`. Every one is a function of bytes on disk,
computed in a single test run, and reproducible by re-running `cargo test -p ix-gaia-advisory --test s11_freeze
-- --nocapture`.

**The block was re-derived once, after `cargo fmt` was applied to the new crate.** Formatting changes the
rule's source text, so `rule_source_digest` and the three `candidate_fingerprint` values derived from it moved;
the three `window_reference_digest` values and `corpus_digest_ix_agg_1` did **not**, because they are functions
of the approved evidence bytes and of generator behaviour rather than of source layout, and neither did
`25 / 1` or `1 : 24`. That the change was exactly this bounded, and was predicted before it was measured, is
the evidence that the fingerprint is bound to the rule and to nothing else. The superseded values were
`rule_source_digest = 8b72282b…0565` with fingerprints `3fd1e7ed…298a` / `679159e6…24e0` / `735f6c6c…9e94`;
they are recorded here so the move is auditable rather than silent.

**The two preregistered predictions were checked, not assumed.** `X2` predicted **25 admitted / 1 rejected**
and the generator produced exactly that; `X3` predicted a truth balance of **1 : 24** and the oracle produced
exactly that, with `F-PRISTINE` the only truth-`T` case. The single rejection is the measured one: `F-CRLF` on
`gaia-s1-r2-validator-output.txt`, `NotApplicable { NoBareLf }`, over a file re-measured here at CR 214 /
LF 214 / **bare LF 0**.

---

## 3. Reversibility — a two-way door

- **Rollback:** delete `crates/ix-gaia-advisory/`, revert one `[workspace] members` line, revert one
  `crate-maturity.toml` row, delete this document. Nothing else is touched, and the corpus lives outside the
  repository so there is nothing to clean inside it.
- **No schema is frozen for a consumer.** `SCHEMA_VERSION = 1`, tier `experimental`, and there is no consumer
  anywhere: the artifact is read by this crate's own tests or by a human.
- **The corpus is never committed.** It is generated into a scratch directory outside the worktree
  (`%TEMP%\ix-gaia-m3-scratch`, overridable with `IX_GAIA_M3_SCRATCH_DIR`). Committing it would put derived
  copies of Gaia S1 evidence into git history — the one thing that would make this partly one-way — so only
  `corpus_digest_ix_agg_1` and the generator source are committed.
- **Revisit triggers:** an M4 authorization; the decision-maker's ruling on the out-of-sample reading, or on
  whether an unapproved lineage may ever ground an M3 claim; or the arrival of a population that is
  simultaneously present, carrying genuine declared-versus-measured disagreements, entitled, and unexposed.
- **One-way elements requiring sign-off before they occur:** committing the corpus (declined above);
  publishing the crate; emitting an artifact to any consumer outside this crate's tests; consuming a
  Class-`Holdout` population. **None of them has occurred.**

---

## 4. Disclosures the implementation owes

Each is a place where the preregistration underdetermined the implementation, or where the implementation had
to choose. Every one is recorded so a reviewer can attack it directly.

1. **`CostAccount.provenance_bytes_read` is a schema addition.** The two provenance aggregates are *measured
   over the evidence root the run actually read*, which costs one full pass. Folding that pass into a rule's
   `bytes_read` would make every rule look identical, so it is accounted separately — on exactly the footing
   the Window Reference Table's construction cost already had. A rule's byte figure therefore describes the
   rule, not the process.
2. **`AdvisoryRow.corroborated_bytes` is a schema addition.** Without a field for the second claim site,
   "both values present" in the contradiction control is not observable. It carries a unit (bytes) and an
   explicit null rule (`null` when the document states no second figure).
3. **`AdvisoryRequest` carries `window_reference_root` and `expected_provenance`.** A windowed rule that
   derived its reference table from the root under test would certify that root against itself; and a
   provenance control needs something to pin against. Both are `Option`, and a windowed rule at a non-zero
   width with no reference root is a refusal, not a degraded run.
4. **A declared row whose file the root does not hold is reported, not refused.** The row measures `null`,
   the name appears in `missing_names`, and the structural name-set rule binds it — this is what keeps the
   null semantics and the root findings live. The *binder* still refuses before emit when the request's own
   declared field cannot be resolved: an unreadable root, or a manifest that is absent, unparseable,
   self-duplicating, or declares an unsafe name.
5. **`F-DEBRIS` binds to the validator *program*, not to the captured-output row.** The citation's act is
   redirecting the validator's stdout, and stdout belongs to the program that writes it; the `…-output.txt`
   row is the product of redirecting *outside* the directory, which the same citation endorses. The choice is
   count-neutral. A reviewer preferring the other binding changes the staged file name and
   `corpus_digest_ix_agg_1`, and nothing else.
6. **The label oracle's ground truth is the frozen fourteen-file bundle**, the manifest's own bytes included.
   A perturbation of the manifest is therefore labelled `F` even though it leaves the thirteen declared rows
   reconciling. Without this, a `F-CRLF` case on the manifest would be labelled `T` and the frozen `1 : 24`
   balance would not hold.
7. **The structural ablation's per-rule outcomes follow the rule definitions, not the slice shorthand.** An
   extra file is caught by the name-set rule and **missed** by the length rule, because the length rule
   compares declared lengths and cannot see a surplus name. The slice plan's "F from all three" would require
   the length rule to detect a name-set change; the assertions written instead discriminate strictly more.
8. **`Contradictory` and `Unknown` both count as "not bound"** when a case is folded into a count vector.
   There is no seventh count. `Contradictory` says two claim sites disagree, which is not a binding of the
   root; the corpus produces none, and the property is exercised directly at artifact level instead.
9. **The reviewed M2 crate tree is pinned at `IX-AGG-1` `cf4c1b0e32047743f8dfe278689369d7ff8ef660be1a5c42cc701df122fac16b`**
   (23 files / 728,251 bytes). The full 26-file fixed point cannot be asserted from inside the worktree
   because two of the 26 — the workspace manifest and the maturity table — carry this crate's single
   authorized registration line each. It is re-measured out of band against the immutable snapshot instead,
   and it matched.
10. **The Streeling catalog was not regenerated.** Doing so writes `state/streeling/catalog.jsonl`, outside the
    authorized diff scope of "the new crate directory, one workspace members line, one maturity row". It is
    left for the decision-maker rather than taken unilaterally.
11. **`cargo fmt` is clean for `ix-gaia-advisory` and left untouched everywhere else.** It is advisory in this
    repository and `verify.ps1` warns rather than blocks on it, so the pre-existing workspace-wide skew —
    `ix-gaia-census` included, which C8 forbids touching — is preserved exactly as found.
12. **The `window_reference_digest` a *report* carries is a sweep digest** — SHA-256 over the
    LF-joined `width|digest` rows, ascending by width, no trailing newline — because a report spans three
    widths and therefore three tables. Each cell additionally carries its own table's digest.
13. **NC-20 is enforced as its stated purpose, not as a source-text proxy** *(post-freeze, blocker B2)*. The
    preregistration states NC-20's purpose as **"no unearned `Measured`"** and its check as a grep for
    `OutOfSampleStatus::Measured`. S0–S11 implemented the grep as *zero* occurrences anywhere in crate source,
    which is strictly stronger than the purpose: it forbade the **earned** construction along with the
    unearned one, so `OutOfSampleStatus::Measured` was unreachable on every input — including a future
    Class-`Holdout` run — and `emit()` would have returned `Unknown { reason: NoModelFitted }` on that path,
    violating NC-25. The control is restated to enforce the purpose exactly: **exactly one** construction
    site, inside the named private function `measured_from_earned_holdout`; a second site, or that site
    moving out of the function, fails. Because a restatement is also what someone would write while weakening
    a control, the restatement **adds** enforcement rather than removing it — see 14.

    This is a divergence from §7.2's sentence *"NC-20 asserts that no code path in M3 constructs it"* in
    letter, not in effect: no path M3-as-selected can reach constructs the variant, because `advise` binds no
    admission and `evaluate` still refuses Class-`Holdout` outright. What changed is that the variant is now
    *constructible under conditions*, as the preregistration always said it should be, instead of
    constructible under **no** conditions.
14. **Four behavioural controls now cover what the grep never could** *(post-freeze, blocker B2)*. The grep
    could only see *where* the variant is written, never *under what conditions* it is reached. Added:
    **NC-20a** (earned-only — Class-`Holdout`, a §11.3 ledger receipt naming this corpus and this candidate
    fingerprint, and a corpus digest this crate measured itself rather than one a caller supplied);
    **NC-20b** (class firewall — no `Development`, `Selection`, or class-less run can be admitted);
    **NC-20c** (no self-certification — the receipt's `report_digest` must equal the digest recomputed from
    the emitted report, pinned to the canonical relative path `corpus/report/evaluation-report.json`; a
    mismatch is `ReplayDivergence` and **no artifact**, a non-canonical path is a refusal);
    **NC-25a** (`NoModelFitted` is never the emitted primary reason on **any** path, holdout included — it is
    permanently true under C6 and is carried as a disclosure, never as a bar). Each condition carries its own
    negative witness. **Control-surface count: one grep replaced by one grep plus four behavioural controls.**
    **Test count: 70 → 85** — 69 inherited unchanged, **1 inherited restated** (NC-20, per 13), all 70 green;
    **10** colocated boundary unit tests in `src/lib.rs`, which exist there because the behaviour is
    deliberately unreachable through the closed public Holdout seam; **5** public-seam tests in
    `tests/b2_oos_emission.rs`. *(R2 raises this figure again — see 16–17.)*
15. **The repair moves no frozen value.** `rule_source_digest` is `SHA-256(src/rules/window_probe.rs)` and the
    three `candidate_fingerprint` values derive from it alone; the repair touches `src/lib.rs`,
    `src/digest.rs`, `src/evaluation.rs` and `tests/`, and not the rule. The §2 block above re-measures
    unchanged, and both S11 deliverables still emit byte-identically (`advisory-artifact.json` 3,734 B
    `f0736565…`, `evaluation-report.json` 31,744 B `571e6aaa…`). **No holdout was read, consumed or created,
    no ledger exists or was written, and the `K = 1` budget is untouched.**
16. **The earned proof tokens are sealed by the compiler, not by a doc comment** *(post-freeze, R1 Spec
    blocker B-2)*. Disclosure 13's restatement moved the chokepoint from `OutOfSampleStatus::Measured` to
    `EarnedHoldout`, and the R1 implementation left that new chokepoint unguarded: Rust field privacy is
    **module**-scoped, so a struct literal written anywhere in `src/lib.rs` — the very file S12 will edit —
    produced an `EarnedHoldout` without calling `admit` and reached `{"status":"Measured",…}` carrying
    attacker-chosen digests, with all 85 tests green and no control firing. Reproduced before repair, and a
    second, equally cheap bypass with it: forging the two *measured-digest* newtypes by tuple literal and
    passing them **through** `admit` gave the identical result. In that one dimension the restatement had
    reduced enforcement, which §10.3 licensed it only on condition of not doing.

    `MeasuredCorpusDigest`, `MeasuredReportDigest` and `EarnedHoldout` now live in a **private child module**
    `mod earned` whose fields carry no visibility qualifier at all. Both bypasses are now **compile errors**
    (`E0451` on the `EarnedHoldout` literal, `E0603` on the newtype constructors), not merely detected ones —
    for the parent module, every sibling, the test module and any future S12 edit alike. The parent-facing
    surface is the minimum needed: three constructors that measure, and four read-only accessors that cannot
    construct. `LedgerReceipt` is deliberately **not** sealed — it models evidence originating outside this
    crate, and a crate that mints its own receipts is the opposite of what §11.3 asks; that a receipt genuinely
    comes from the ledger stays S12's obligation and its reviewer's check.

    **New control NC-20d** pins the source invariant the compiler's guarantee rests on: `mod earned` exists
    once and is not `pub`; all three tokens are declared inside it; every `pub` inside it is exactly
    `pub(super)` and appears only on `fn` and `struct` items, never on a field; exactly one `EarnedHoldout`
    construction exists, inside `admit`; and nothing outside the module constructs a sealed token.
    **The public API surface is unchanged: 135 externally-`pub` items before, 135 after, zero differences.**
17. **`candidate_fingerprint` is pinned to its three frozen §17 values** *(post-freeze, R1 Spec residual R-2)*.
    It was pinned by nothing: `tests/s11_freeze.rs` reimplements §17's recipe locally and so pins its own copy,
    and NC-20a's width control builds its receipts with the very function it tests, so both sides moved
    together. Decoupling the function from `RULE_SOURCE_TEXT` entirely failed **no test in any of the fourteen
    binaries** — the receipt named *this candidate* only up to **width**, never up to **rule identity**. A
    colocated control now asserts the frozen rule source digest `668cc157…`, the three frozen fingerprints
    (`24a0c60e…`, `af886b20…`, `e9273462…`) against the crate function, and those same three re-derived from
    §17's recipe using the frozen digest **constant** rather than the crate's own source text — so the rule,
    the recipe and the values cannot drift together. **No frozen value moved; all three were already correct.**

    **Test count after R2: 85 → 87** (NC-20d, and the §17 fingerprint pin). Every one of the 85 is still
    present and green. Both S11 deliverables still emit byte-identically, and `Measured` remains dormant in
    production: `advise` binds no admission and `evaluate` still refuses Class-`Holdout` before any read.

---

## 5. What this records, and what it does not

**Records.** That `ix-gaia-advisory` exists as a workspace member at tier `experimental`; that its two public
verbs behave as preregistered across twelve vertical slices; that the frozen values above were measured rather
than asserted; that both named digest recipes reproduce the values the approved evidence declares, after the
whole suite has run; that no Class-`Holdout` byte was created, opened, or evaluated. **And, post-freeze, that
blocker B2 was repaired** — disclosures 13–15 — so that `OutOfSampleStatus::Measured` is constructible along
exactly one earned path instead of along none, without moving any frozen value and without opening the public
Class-`Holdout` seam; **and that the R1 Spec review's blocker B-2 and residual R-2 were then closed** —
disclosures 16–17 — so that the earned path's proof tokens are sealed by the compiler rather than by prose,
and the §17 candidate fingerprint is pinned to its frozen values. Neither correction moved a frozen value,
opened the seam, or made `Measured` reachable in production.

**Does not record.** That M3 is authorized or accepted — it is not. That any rule beats any baseline; no such
claim is computed anywhere, and the trivial floor's `false_agreement` of exactly `1.0` is reported precisely so
that no rule can be praised for beating nothing. That anything transfers to real Gaia missions — that stays
`UNKNOWN`. That an out-of-sample number exists — the typed answer is `Unknown { NoEligibleUnseenPopulation,
[SyntheticOrExposedPopulationOnly, NoRealLabelPopulation, NoModelFitted] }`. That the workspace test suite
passes in full — see the writer handoff for what was run and what was not. That anything may be committed,
pushed, published, merged, or exported.
