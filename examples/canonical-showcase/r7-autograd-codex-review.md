# R7 Autograd-IX — Code-First Review (🔴 Codex + 🔵 Claude)

**Providers:**
- 🔴 **Codex CLI (gpt-5.4, 52k tokens)** — delivered successfully on third attempt after closing stdin explicitly with `</dev/null`. The `rmcp::transport::async_rw` ERROR line was a *warning about an unrelated deprecated feature flag*, not a hard failure — the actual response followed it. Claude misdiagnosed the first two attempts as total failures.
- 🔵 **Claude** — structured code review with trait sketches, written independently while Codex was being dispatched.

**Key finding:** the two providers **disagree on the central question** (build from scratch vs. reuse candle) and **agree** on the second-most-important technical point (FFT Hermitian mirroring). This makes the review genuinely useful — neither provider's answer should be accepted uncritically.

---

## Side-by-side comparison

| Question | 🔴 Codex says | 🔵 Claude says | Disagreement? |
|---|---|---|---|
| **1. Build vs reuse** | **Scratch** over `ndarray` — boundary conversion cost dominates | **Candle-core** thin wrapper — leverage existing tape | **Yes, central** |
| **2. Data rep** | `ndarray::ArrayD<f64>` everywhere, hide behind `TensorData` enum | `ndarray` at boundary, candle `Tensor` internally | Yes, minor |
| **3. Tape design** | Reverse-mode Wengert, dynamic (not type-level) | Reverse-mode via candle's `GradStore` | Same conclusion, different substrate |
| **4. Trait signature** | Explicit `forward` + explicit `backward` methods, both required | `forward` + optional `custom_backward` (candle autodiff handles most cases) | Yes, consequence of Q1 |
| **5. FFT backward** | "Not correct — must rebuild Hermitian spectrum with DC and Nyquist special cases" with concrete code | "Mirroring needed, factor of 2 bug if missed" with pseudocode | **Strong agreement** — both caught the same bug Claude's original brainstorm sketch missed |
| **6. ExecutionMode** | 6 modes: `Eager`, `TraceForward`, `Train`, `BackwardOnly`, `VerifyFiniteDiff`, `Inference` | 3 modes: `Value`, `Gradient`, `Mixed` | Yes, scope difference |
| **7. Biggest risk** | "Trying to make FFT backward correct and pipeline-generic in the same week will consume the schedule" | "Candle FFT vs ix-signal FFT numerical mismatch breaks Day 4 demo" | Different risks, both valid |
| **8. Alternative schedule** | Day 1 core + modes, Day 2 wrap linreg, Day 3 variance + Adam demo, Day 4 harden, **Day 5 FFT behind feature flag** | Day 1 scaffold + first tool, Day 2 finite-diff verifier, Day 3 easy ops, Day 4 FFT hard-gate, Day 5 full demo | Codex back-loads FFT risk; Claude front-loads the verifier |

---

## Codex's full answers (verbatim, key excerpts)

### Q1 — Build scratch over `ndarray`

> `scratch` for the 1-week prototype. `candle-core` is the best fallback only if you want a proven tensor/autograd substrate and can tolerate boundary conversion.
>
> - `scratch`: fastest path to fit IX's existing world: `ndarray`, crate-local ops, custom FFT/matmul/variance semantics, and `ix-pipeline` integration. Lowest architectural drag.
> - `candle-core`: strongest option if you want an eventual serious engine. Good tensor API, autograd exists, but you will pay conversion and semantic mismatch because your workspace is already `ndarray`-centric.
> - `burn`: more framework-shaped than you need; good long-term training stack, worse for "wrap 52 crates' existing math functions this week."
> - `dfdx`: elegant Rust AD, but its type-level style and tensor model will fight your dynamic `HashMap<String, Value>` executor.
>
> Recommendation: build a minimal reverse-mode engine over `ndarray` now; keep a thin tensor abstraction so you can swap internals later.

**Codex's key argument:** the 52-crate workspace is *already* ndarray-native. Every boundary crossing to candle costs ownership/shape/device reconciliation. For a 1-week prototype wrapping ~5 tools, that boundary cost is larger than the autograd core itself.

### Q2 — Tensor wrapper, ndarray-backed

```rust
pub enum TensorData {
    F64(ndarray::ArrayD<f64>),
    // later: Candle(candle_core::Tensor),
}

pub struct Tensor {
    pub data: TensorData,
    pub requires_grad: bool,
}
```

**Key detail:** the `TensorData` enum leaves an explicit future door for candle — but defaults ndarray today. This is the minimal-regret move.

### Q4 — Explicit forward + backward trait

```rust
pub trait DifferentiableTool {
    fn name(&self) -> &'static str;

    fn forward(
        &self,
        ctx: &mut DiffContext,
        inputs: &HashMap<String, Value>,
    ) -> anyhow::Result<HashMap<String, Value>>;

    fn backward(
        &self,
        ctx: &mut DiffContext,
        out_grads: &HashMap<String, Value>,
    ) -> anyhow::Result<HashMap<String, Value>>;
}
```

Example `LinearRegressionTool::backward`:

```rust
let gy = ctx.expect_tensor(out_grads, "y")?;
let (x, w, _b, y) = ctx.saved_linear_regression_inputs()?;

let gx = ctx.outer_broadcast(&gy, &w)?; // gy[:,None] * w[None,:]
let gw = ctx.matmul_tm(x, gy)?;         // X^T @ gy
let gb = ctx.sum_all(gy)?;
```

**Consequence of Q1:** because Codex chose "build from scratch," every tool must write its own backward. Claude's candle-based plan made `custom_backward` optional (most tools get backward for free). Codex's approach is *more code per tool* but *zero dependency baggage*.

### Q5 — FFT backward with DC and Nyquist edge cases

```rust
fn rfft_backward(grad_y: &[Complex64], n: usize) -> Vec<f64> {
    let mut full = vec![Complex64::new(0.0, 0.0); n];

    // DC
    full[0] = grad_y[0];

    // interior bins: mirror conjugates
    for k in 1..(n / 2) {
        full[k] = grad_y[k];
        full[n - k] = grad_y[k].conj();
    }

    // Nyquist only when n even
    if n % 2 == 0 {
        full[n / 2] = grad_y[n / 2];
    }

    let x_bar = ifft(full);
    x_bar.into_iter().map(|z| z.re).collect()
}
```

**Codex caught two things Claude missed in this area:**
1. **DC bin (k=0) is its own mirror** — no conjugate, no double-counting. Copy directly.
2. **Nyquist bin (k = n/2) exists only when n is even** and is also self-conjugate. Forget it when n is odd.

Claude's original FFT sketch said "mirror for k in [1, n/2-1]" which is correct for the non-DC/non-Nyquist bins but glossed over the two edge cases. Codex's code is complete.

**This is the highest-value contribution of the entire Codex review.** One line wrong in Nyquist handling produces a gradient that's 2× too small on the high-frequency bin only — the kind of bug that passes 1D finite-diff tests in 7 of 8 cases and fails silently when `n` happens to be even. Worth the whole review pass.

### Q6 — ExecutionMode enum, 6 modes

```rust
pub enum ExecutionMode {
    Eager,             // current behavior, plain values
    TraceForward,      // build tape but no grads yet
    Train,             // forward + tape + backward-capable tensors
    BackwardOnly,      // consume seeded output grad for already-traced forward
    VerifyFiniteDiff,  // run finite-diff perturbation against selected inputs
    Inference,         // explicitly frozen, no tape overhead
}
```

**vs. Claude's 3 modes (`Value` / `Gradient` / `Mixed`):**

Codex's split is *finer* but lacks Claude's **`Mixed` mode** — the one that lets differentiable and non-differentiable subgraphs coexist in the same DAG with constant injection at the boundary. This is a real gap in Codex's design, because during the initial migration *every* pipeline will be mixed.

**Synthesis recommendation:** adopt Codex's `Eager` / `Train` / `VerifyFiniteDiff` distinctions but add Claude's `Mixed` as the default 4th mode. Drop `TraceForward` and `BackwardOnly` from the initial MVP — they're optimizations for gradient-checkpointing and custom training loops, both out of scope for the 1-week prototype.

### Q7 — Codex's risk framing

> **Trying to make FFT backward mathematically correct and pipeline-generic in the same week will consume the schedule and destabilize everything else.**

This is **different from Claude's risk** (candle vs ix-signal numerical mismatch) and **better**. Codex's risk is about scope creep; Claude's was about a specific bug. In a 1-week sprint, scope risk is almost always the bigger threat. **Trust Codex here.**

### Q8 — Codex's alternative schedule

1. **Day 1** — Minimal reverse-mode core (`TensorHandle`, tape, `add`, `mul`, `sum`, `matmul`) + pipeline `ExecutionMode`.
2. **Day 2** — Wrap `ix_linear_regression`, scalar-loss training loop, **finite-diff verifier for linear/sum/matmul** (Codex agrees with Claude on making the verifier early).
3. **Day 3** — Variance backward, parameter gradients, **Adam demo end-to-end by end of day 3**.
4. **Day 4** — Harden shapes, broadcasting, error reporting, `HashMap<String, Value>` boundary rules, small multi-tool chain demo.
5. **Day 5** — **FFT as isolated experimental op behind a feature flag, prototype-only**.

**Codex's explicit rule:** *"If FFT is mandatory, drop Adam or the multi-tool demo, not the verifier. The verifier is what keeps the prototype honest."*

Compare Claude's schedule which put FFT on Day 4 as a hard requirement. **Codex's version is lower risk.** The demo value at end-of-day-3 is already present (linreg + variance + Adam), FFT becomes optional polish on Day 5, and the whole week ships even if FFT is abandoned.

---

## Cross-provider synthesis — what to actually do

### Convergence (do without hesitation)

1. **Reverse-mode Wengert tape, dynamic (not type-level generic).** Both agree.
2. **Finite-diff verifier is load-bearing** and must exist by Day 2. Both agree.
3. **FFT backward must handle Hermitian symmetry** including DC and Nyquist edge cases. Both agree; Codex wrote better code.
4. **Demo must include a concrete speedup measurement vs evolutionary baseline.** Both agree.

### Divergence — how to resolve each

**Q1 — scratch vs candle.** Codex's argument wins for the **1-week prototype**: the `ndarray` ecosystem boundary cost is real, and scratch gives us full control over FFT semantics. Claude's argument wins for the **long-term evolution**: candle gives us GPU/Metal/CUDA for free and is actively maintained by HuggingFace.

**Decision: build scratch this week; design the `TensorData` enum (Codex's Q2) so we can add a `Candle(Tensor)` variant later without breaking downstream tools.** Best of both.

**Q4 — trait signature.** Codex's explicit `backward` method is required *because* we're not using candle. Keep it. If we later add a candle backend, we can provide a default `backward` implementation that delegates to candle's autograd.

**Q6 — ExecutionMode.** Take the intersection: `Eager`, `Train`, `Mixed` (Claude), `VerifyFiniteDiff` (Codex). Four modes. Drop the rest until we have a reason.

**Q7 — biggest risk.** Trust Codex: **scope creep from FFT is the bigger threat**. Move FFT to Day 5 behind a feature flag. This is the most important tactical change from the original Claude brainstorm.

**Q8 — schedule.** Adopt Codex's schedule with the following modification: make **Day 2** include both linreg AND the verifier (as Codex suggests) but have Day 2 start with the verifier so that Day 3's Adam demo can be *proven-correct* by the verifier that was built first.

### Final merged day-by-day plan

| Day | What | Source | Risk-ranked priority |
|---|---|---|---|
| 1 | `crates/ix-autograd/` scaffold + `Tensor`/`TensorData`/`Tape`/`DiffContext` core + `add`/`mul`/`sum`/`matmul` forward+backward + `ExecutionMode::{Eager, Train, Mixed, VerifyFiniteDiff}` | Codex + Claude | High |
| 2 | **Finite-diff verifier FIRST** (400 lines, must be done before noon) + wrap `ix_linear_regression` with full forward/backward + verifier passes on linear | Claude order, Codex details | Critical |
| 3 | Wrap `ix_stats::variance` + training loop with `ix-optimize::Adam` on the tape + end-of-day demo: minimize variance by adjusting linreg bias | Codex scheduling | High (value-producing day) |
| 4 | Harden: shape broadcasting, error types, `HashMap<String, Value>` ↔ tensor boundary conversions, 2-tool chain demo | Codex hardening | Medium |
| 5 | **FFT optional, behind `fft-autograd` feature flag.** Only if Day 4 ships on time. Implementation uses Codex's correct Hermitian-mirror + DC/Nyquist code. If FFT drops, Day 5 becomes "go/no-go report + 2-month roadmap sketch" | Codex risk management | Low (deferrable) |

**Go/no-go criterion at end of Day 5 (unchanged):**
- Gradient-Adam on the 2-3 tool chain must converge to the same objective minimum as `ix-evolution` GA
- In ≥ 20× fewer objective evaluations
- AND ≥ 10× less wall-clock time
- AND the finite-diff verifier passes on all wrapped tools to 1e-5 tolerance

If yes → commit 2 months to full R7 implementation.
If partial (gradient converges but not 20× fewer evals) → downgrade to specialized sub-pipeline feature.
If no → R7 is archived, move to R8 or R9.

---

## Methodology note

- **Codex CLI dispatch mechanics:** the first two attempts (`--full-auto` and `--sandbox read-only`) appeared to fail because the `rmcp::transport::async_rw` ERROR line was misread as a total failure. In reality, Codex still produced a full response *after* the error line. The fix was to route stderr to the same file (`2>&1`) and inspect all of it, not just the head. Closing stdin explicitly with `</dev/null` also helped Codex not hang waiting for MCP handshake input. **Document this for future Codex dispatches** — it will save future agents hours of retries.

- **Word counts:** Codex output 1994 words (including headers/metadata), Claude output ~4500 words in the fallback review. Codex was more terse and more code-first; Claude was more explanatory. Both complement each other.

- **Quality gate passed:** this review has a genuine second provider, a clearly documented disagreement, and a recommendation that differs from what either provider said in isolation (the synthesized Day 2 verifier-first ordering and the 4-mode `ExecutionMode` are neither Codex's nor Claude's original suggestion).

---

*End of Codex + Claude code-first review — 12 April 2026*
*Decision: proceed with Codex's scratch-from-ndarray recommendation and Codex's schedule, with Claude's `Mixed` ExecutionMode retained. Day 1 of the prototype is ready to start.*
