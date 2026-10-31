# #489 evidence: what the Jev routing spike already guarantees, offline

Research note for [#489](https://github.com/GuitarAlchemist/ix/issues/489) (map
[#483](https://github.com/GuitarAlchemist/ix/issues/483)). Local, not published. It records
evidence and a proposal; the operator decides #489. No network call, no key read, `run-live.ps1`
not executed.

Status words: **source-read**, **tested** (a command ran here), **indeterminate**.

## Pinned sources (tested)

- `crates/ix-router-spike/src/bin/jev_router.rs` at blob `0d8a6adcf1a13ee7182f53647eee85c3c2aabef8`
  and `state/router-spike/jev/run-live.ps1` at blob `a8a3702ef20a4bc4aac46d95215fcdba8ed703da`,
  both the blobs #489 names (`git rev-parse HEAD:<path>` on origin/main 99f2f869).
- The crate is standalone (empty `[workspace]` table): CI's workspace build never compiles it. Since
  #490 a dedicated CI step (ubuntu, stable) runs its tests.
- Command: `cargo test --offline --manifest-path crates/ix-router-spike/Cargo.toml --bin jev-router`.
  Before this branch: 17 passed. With the seven pins below: 24 passed.

## #489's contract cases against the current code

The seven new tests pin the **current** behaviour, gaps included. They were written green against
unchanged code: none was seen red, so each guards against a regression, not against a bug.

| #489 case | Current behaviour | Evidence | Gap |
|---|---|---|---|
| wrong model | `Invalid::WrongModel`, verdict KILL | existing `test_validate_rejects_contract_breaks`, `test_verdict_bands_follow_preregistration` | — |
| option mismatch | unknown choice or key set ≠ options → `Contract` | existing `test_validate_rejects_contract_breaks` | — |
| malformed / non-finite scores | null, negative, > 1, string → `Contract`; JSON cannot carry NaN (`json!(f64::NAN)` is null) | new `test_non_finite_or_out_of_range_scores_are_invalid` | — |
| ambiguous evidence | gold `ambiguous` is routed, never scored (existing test); an argmax **tie** is accepted whichever tied option is chosen, unflagged | new `test_argmax_tie_is_accepted_and_not_flagged` | yes |
| timeout | `run-live.ps1` writes `error`/`http_status`; `load_receipt` keeps only `response`, so a timeout is `Contract("model missing")` | new `test_timeout_and_provider_failure_collapse_into_one_contract_failure` | yes |
| provider failure | same `Contract("model missing")` as a timeout: the two are indistinguishable after loading | same test | yes |
| missing usage | per answer: invalid, never zero cost (existing); in the reported total: counted as 0 tokens | new `test_usage_total_counts_a_missing_usage_as_zero` | yes |
| digest mismatch | `Invalid::DigestMismatch`; an unanswered prompt is `Invalid::Missing` | new `test_digest_mismatch_and_missing_answer_are_invalid` | — |
| duplicate return | the whole scoring aborts (panic), never last-wins; an unplanned id aborts too | new `test_duplicate_receipt_id_aborts_scoring`, `test_unplanned_receipt_id_aborts_scoring` | — (fail-closed, but all-or-nothing) |

Every failure is fail-closed: none becomes a correct decline (existing
`test_invalid_response_is_never_a_correct_decline`). What is missing is the **kind** of failure:
#489 asks for explicit timeout and provider-unavailable outcomes, and the spike has one `Contract`
bucket for both.

## What leaves the machine (source-read)

`request()` sends exactly: `model` (`jev-1.13.0`), `state.user_message` (the prompt text), and one
`questions.intent` with `type: "choice"`, the options file's `instructions` and its `criteria` (option
id → description). Nothing else: no history, no file, no identity. The digest of these bytes is
pinned in `plan.json` and rechecked by `run-live.ps1` before sending.

## The live step (source-read, not run)

`run-live.ps1`: key from the environment, a second approval variable, one attempt per request, no
redirect, 30 s timeout, refuses to append to an old receipt, stops on 401/403/429 and on a 200
without usage. The budget stop is checked **after** each response (as #489 says, not a spending
cap). It does not stop on repeated 5xx or timeouts: each is recorded and the loop goes on.

## Recorded results on the frozen corpora (source-read, `jev-eval.json` files)

| arm | verdict | in-scope | OOS declined | invalid |
|---|---|---|---|---|
| base | COMPETITIVE_WITH_HEAD | 106/110 | 16/16 | 0 |
| reversed | ROBUST | 106/110 | 16/16 | 1 |
| fr | **KILL** | 100/110 | 15/16 | 4 |
| es | ROBUST | 103/110 | 16/16 | 1 |
| fresh | ROBUST | 110/112 | 16/16 | 1 |
| full | ROBUST | 178/190 | 31/31 | 1 |
| musicse | ROBUST | 1/1 | 81/86 | 0 |

The `fr` arm's counts clear its ROBUST band (99, 13); it is KILL on four invalid responses alone
(limit 2). Format failures, not routing errors, are what killed an arm: the contract's invalid path
matters as much as accuracy. All seven ran against `jev-1.13.0`; nothing here says anything about
1.14 (indeterminate: its documentation was not read).

## IXQL seams that exist (source-read)

- `ix-ixql` `Capability::call(CallArgs) -> Result<Produced, String>` (`capability.rs:77`), registered
  through `Executor::capabilities()`. `Produced` may carry a `Verdict`, the only way a verdict enters a
  run, read by `→ when T >= 0.7:`.
- `ix-baml` `BamlOperation::invoke(&Value) -> Result<Value, BamlError>` (`lib.rs:75`), reached by a
  `baml.<name>` step (`eval.rs:708`); `BamlError` has three variants, none for timeout or abstention.
- Neither seam knows the spike: no crate depends on `ix-router-spike`.

## Proposal for the operator (not a decision)

A Jev decision node as an IXQL `Capability` (the seam #489 prefers), reusing the spike's
`request`/`encode`/`validate`, whose `Produced.value` is a typed outcome:
`choice | abstain | invalid | timeout | provider_unavailable | missing`, each with provenance
(model, options version, request SHA-256) and usage when reported. `abstain` is `__none__` only;
`Err(String)` stays for adapter bugs. It attaches **no** `Verdict` derived from the probabilities:
a numeric score is not a calibrated truth value, and a verdict would let `→ when` act on it.

Smallest next slice, offline: an offline recorded-response adapter over the existing receipts that
returns those six outcomes, with the three gaps above as its first red tests (timeout ≠ provider
failure, tie flagged, missing usage counted as unknown cost in totals). Whether a duplicate or
unparsable receipt line (source-read: `load_receipt` panics on both) should void one item or the
whole run is a choice for the operator. A live adapter, 1.14 support and any spending stay separately
admitted.

French: [`docs/fr/research/agent-composition/2026-10-30-489-jev-decision-node-evidence.md`](../../fr/research/agent-composition/2026-10-30-489-jev-decision-node-evidence.md).
