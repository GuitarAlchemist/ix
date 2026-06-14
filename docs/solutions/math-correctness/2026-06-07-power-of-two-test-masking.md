---
title: "Power-of-two-only test inputs hide padding/truncation bugs in signal & transform code"
category: math-correctness
date: 2026-06-07
tags: [signal-processing, fft, wavelet, testing, adversarial-review, edge-cases, catalog-skills]
symptom: "New spectrogram + wavelet_denoise skills passed all unit tests and ran end-to-end, but an adversarial review found they silently produce wrong results for non-power-of-two inputs. Every test happened to use a power-of-two size."
root_cause: "ix-signal's rfft zero-pads each frame to the next power of two, and the Haar DWT halves length per level (dropping the last sample of an odd block). Both behaviors are invisible when every test input is already a power of two / divisible by 2^levels."
---

# Power-of-two test inputs mask padding/truncation bugs

## Problem

Wrapping `ix-signal` algorithms as catalog skills (`spectrogram`,
`wavelet_denoise`), the unit tests all passed and a live pipeline run produced
plausible output. An adversarial review (numerical-correctness lens) probed
inputs the tests didn't and found two real bugs:

1. **`spectrogram` — bin frequency miscalibration for non-pow2 `window_size`.**
   `ix_signal::fft::rfft` pads its input to `len.next_power_of_two()` before
   transforming. So a frame of length `window_size = 24` returns a **32-point**
   spectrum. The handler took `window_size/2 + 1 = 13` bins and reported them as
   if bin `k` = frequency `k·fs/24` — but they're actually `k·fs/32`. Every
   returned bin was mislabeled, and the bin *count* was wrong (the true
   one-sided count is `32/2+1 = 17`). A pure tone did **not** land where a caller
   computes `round(f·window_size)`.

2. **`wavelet_denoise` — silent length loss for non-divisible lengths.**
   `haar_forward` uses `n = signal.len() / 2` (integer division, dropping the
   last sample of an odd-length block); multi-level decomposition compounds it.
   A length-15 signal at `levels=1` came back length **14**; length-13 at
   `levels=3` came back length **8**. The skill's `@ai:invariant` and output
   schema both claimed "same length as the input." A caller doing element-wise
   comparison to the original signal gets a silently misaligned array.

**Why the tests missed it:** the `spectrogram` test used `window_size = 16` and
the `wavelet` test used a length-16 signal — both powers of two, both on the
"happy path" where padding is a no-op and the DWT round-trips exactly. The
defect lives entirely on the *other* branch.

## Solution

Two parts: reject the inputs the wrapped library can't handle correctly, and
add tests that encode the reviewer's exact counterexamples.

- **`spectrogram`**: require `window_size.is_power_of_two()` (the STFT is a
  radix-2 FFT; non-pow2 silently zero-pads). Reject otherwise with a clear error.
- **`wavelet_denoise`**: require `signal.len() % 2^levels == 0` so the Haar DWT
  round-trips to the original length; reject otherwise (honest boundary — the
  caller pads/trims to fit). Narrow the `@ai:invariant` to the *enforced*
  precondition rather than claiming unconditional length preservation.
- **Tests**: `spectrogram_rejects_non_power_of_two_window` (`window_size=24`),
  `wavelet_denoise_rejects_indivisible_length` (`length-15, levels=1`). These ARE
  the adversarial counterexamples, now executable.

## Lessons (grep-worthy)

- **For FFT/DWT/transform code, a passing test on a power-of-two size proves
  almost nothing about non-pow2 inputs.** The padding/truncation behavior is on
  a branch your nice round test never takes. Always add an explicit non-pow2 /
  odd-length / not-divisible probe.
- **Check what the underlying primitive does to length.** `rfft` here pads up;
  Haar `dwt` truncates down. A wrapper that reports `window_size/2+1` bins or
  promises "same length" is only correct on the divisible subset — surface that
  as a validated precondition, not an implicit assumption (CLAUDE.md
  `certainty := strength of live binding`).
- **An `@ai:invariant` bound to a power-of-two-only test is non-discriminating
  for the general claim.** The wavelet "preserves length" invariant would still
  pass if length-preservation broke for odd inputs — the test couldn't fail.
  Either narrow the claim to what the test covers, or add the edge-case test.
- **This is exactly what adversarial review buys over green CI.** Three lenses
  (correctness / @ai-binding / integration) on a 6-skill batch that was already
  "all tests pass, runs end-to-end" surfaced 4 P1s. The executable oracle (the
  new boundary tests) is the final arbiter, but the review is what *pointed* at
  the branch to test.

## Related

- `state/thinking-machine/dogfood-2026-06-07-findings.md` — the signal/SVD/GMM
  batch + the catalog-gap audit that prompted it.
- `[[feedback_green_but_dead]]` — green tests on the happy path ≠ correct.
- `[[feedback_rust_ci_invocation_parity]]` — run the real edge cases, not the
  convenient ones.
