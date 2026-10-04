# ADR-0007: The ComfyUI extension lives in the ix repo and reaches IX only through `ix-mcp` over stdio

Status: proposed (2026-10-04)

## Context

Three ComfyUI nodes had been proven outside the repo, in a throwaway tracer
directory: `IXSpectrogram` (audio → `ix_spectrogram` summary) and
`IXBraidControl` (a braid word → `ix_braid` strand paths → ControlNet lineart
and depth). The next work — a catalogue of rope knots, then origami — needs a
third node, `IXKnotControl`, and many more after it. A tracer folder has no
tests in CI, no review, and no history; the nodes needed a home.

Two questions had to be settled together:

1. **Where the pack lives.** A separate repository, a ComfyUI Manager
   registry entry, or a folder inside `ix`.
2. **How Python reaches IX.** Re-implementing the maths in Python, binding
   Rust through PyO3, keeping a long-running IX server, or starting `ix-mcp`
   once per node execution.

The geometry is where the value is, and it is subtle: a knot drawing is
checked against the knot its closure must be (Jones polynomial, writhe,
component count) and the rope is lifted only where it passes in front. That
logic is tested in Rust; a Python copy would drift from it the first time
either side changed.

## Decision

**1. The pack is `integrations/comfyui/ix_comfyui/`, in this repo.** It ships
with the tools it calls: a change to `ix_knot`'s output and the node that
reads it land in the same pull request and are tested together. It replaces
the tracer copies of `IXSpectrogram` and `IXBraidControl` and adds
`IXKnotControl`.

**2. Python holds no IX logic.** Each node checks its own inputs (sizes,
character sets, counts), calls one IX tool, checks the shape of what comes
back, and rasterizes it. The invariants, the layout and the over/under
decisions are IX's. The renderers draw y upward so the picture is the knot IX
returned and not its mirror; a test pins that.

**3. One `ix-mcp` process per call, over stdio JSON-RPC.** `bridge.call_tool`
sends `initialize` + `tools/call`, waits at most 30 s, and kills the process
on timeout. The child gets a minimal environment (`SYSTEMROOT`, `WINDIR`,
`TEMP`, `TMP` — no API keys) and runs in the pack's own `run/` folder. Start-up
costs tens of milliseconds, which is nothing next to a diffusion step.

**4. The binary is the pack's own, and no node input names a path, an
executable or a tool.** `install.py --binary <build>` copies an `ix-mcp` build
into `ix_comfyui/bin/` and records its SHA-256 beside it; both are gitignored.
Every call re-hashes the binary and refuses on mismatch. `TOOLS` is an
allowlist (`ix_spectrogram`, `ix_braid`, `ix_knot`); anything else is refused
before a process starts. The hash catches a binary swapped by accident (a
rebuild without the tool a node needs). **It is an integrity check, not a
security boundary**: whoever can write to the pack can edit `bridge.py` too.

**5. The pack is tested in CI without ComfyUI.** A `comfyui-pack` job builds
`ix-mcp`, installs it into the pack, and runs `integrations/comfyui/tests`
with numpy and Pillow only. The torch-side tests (ComfyUI's `AUDIO` in and
`IMAGE` out) skip by name where torch is absent and run in ComfyUI's own
Python.

**Rejected:**

- *A separate repository* — every tool change would become a cross-repo
  release, and the pack's tests would run against whatever `ix-mcp` happened
  to be installed.
- *PyO3 bindings* — a compiled extension per Python version and platform,
  built against ComfyUI's bundled interpreter; and a crash in Rust would take
  ComfyUI down with it. The process boundary costs little and contains that.
- *A long-running IX server* — lifecycle, port and restart questions for a
  node that runs a few times per generation.

## Reversibility

**Two-way door.** Nothing outside the pack depends on its layout: moving it to
its own repository is a copy plus a release step, and the bridge is one file.
The node class names (`IXSpectrogram`, `IXBraidControl`, `IXKnotControl`) are
what saved ComfyUI workflows reference; renaming one breaks those workflows,
so treat the names as the pack's public surface.

## Revisit triggers

- **The pack is to be installed by people who do not build `ix`.** ComfyUI
  Manager distribution needs a prebuilt binary per platform and a release
  process; that is when a separate repository, or release assets from this
  one, becomes worth it.
- **A node needs IX state across calls** (a loaded index, a model, a session).
  One process per call cannot hold it; a long-running server with an explicit
  lifecycle would.
- **Process start-up shows in a profile.** Not expected below hundreds of
  calls per generation.
- **A node needs a tool outside `TOOLS`.** Add it in the same pull request as
  the node, with a shape check in the bridge — never a pass-through.
