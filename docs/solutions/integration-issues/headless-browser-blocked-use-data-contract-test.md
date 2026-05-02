---
title: "Headless Browser Can't Capture a Vite SPA — Use a Data-Layer Contract Test Instead"
category: integration-issues
date: 2026-05-02
tags: [contract-testing, headless-browser, vite-spa, cross-repo, smoke-test, mcp, prime-radiant]
severity: medium
components: [ix-agent, ix-voicings, ga-react-components, prime-radiant]
resolution_time: "~30min"
---

# Headless Browser Can't Capture a Vite SPA — Use a Data-Layer Contract Test Instead

## Symptom

Phase 4 of the ix → ga voicings-in-Prime-Radiant integration needed an end-to-end smoke proving the 688K-point voicing cloud renders at scene offset `[200, 0, 0]`. Three independent headless-browser tools — Chrome `--headless=new`, Edge `--headless=new`, and Vercel's `agent-browser` CLI — all silently failed against the GA Vite SPA on Windows 11. Chrome/Edge exited cleanly (exit code 0) without writing the requested PNG and without producing stderr; `agent-browser` failed with `Chrome exited early without writing DevToolsActivePort`. The same toolchain captured a standalone static HTML page (the `/3d` route on `serve_viz`) without issue, so the failure was Vite-SPA-specific, not a generic Chromium-on-Windows problem.

## Investigation

- Started with Chrome `--headless=new --screenshot=out.png --virtual-time-budget=60000 http://localhost:5176/?voicings=1`. Process returned 0; no `out.png` was written; nothing on stderr.
- Switched to Edge with the same flags, on the assumption Chrome's profile state was stale. Identical silent-success behaviour, no PNG.
- Tried Vercel's `agent-browser` CLI (which manages its own Chromium instance and a CDP session). Hard fail: `Chrome exited early without writing DevToolsActivePort`.
- Bumped `--virtual-time-budget` from 30s to 60s to 120s. No change — exit code stays 0, the file never appears.
- Added `--no-sandbox --disable-gpu --disable-dev-shm-usage`. Still silent. No useful diagnostic ever surfaced.
- Pointed each tool at `http://localhost:8765/3d` (a static HTML page served by `serve_viz`). All three captured it correctly — confirming the failure is gated on the Vite dev server, not on headless Chromium per se.
- Concluded that any further investment in headless browser plumbing was Sentinel's-Void territory (governance over nothing) — the real contract to verify was the producer→consumer data path, not the pixel.

## Root Cause

Vite dev mode resolves and serves hundreds of ES modules on demand as the SPA bootstraps. The `--virtual-time-budget` flag fast-forwards Chromium's *virtual* clock so timer-based work completes early, but it does not wait for *real* network round-trips or module-graph resolution to settle. The screenshot fires on the virtual-clock deadline, before the SPA has actually mounted, and Chromium then treats the page as "done" and exits — with no PNG written because nothing visible was ever painted, and with exit code 0 because, from Chromium's point of view, nothing went wrong.

## Solution

Replace the visual smoke with a data-layer contract test that exercises the same producer→consumer path the visual would have proven. The test lives at `crates/ix-agent/tests/voicings_payload_contract.rs` (ix commit `b492882`, extended in `52a27ed`).

It has two halves with deliberately different gating.

### 1. Handler-shape assertion (always runs)

This proves the `ix_voicings_payload` MCP tool emits a payload conforming to `voicings.payload.v1`. It runs in any environment — no on-disk corpus required — so CI catches contract drift even when no one has built the OPTIC-K data yet.

```rust
#[test]
fn payload_handler_matches_voicings_payload_v1_schema() {
    let reg = ToolRegistry::new();
    let payload = reg
        .call("ix_voicings_payload", json!({}))
        .expect("ix_voicings_payload reachable via registry");

    assert_eq!(payload["schema"], "voicings.payload.v1");
    assert_eq!(payload["format"], "binary-positions+meta");

    for url_field in ["positions_url", "meta_url"] {
        let url = payload[url_field]
            .as_str()
            .unwrap_or_else(|| panic!("{url_field} must be a string"));
        assert!(url.starts_with("http"), "{url_field} should be an http URL, got {url}");
        assert!(!url.ends_with('/'), "{url_field} should not end with a slash");
    }

    let offset = payload["scene_offset"]
        .as_array()
        .expect("scene_offset is an array");
    assert_eq!(offset.len(), 3, "scene_offset must be [x,y,z]");
    assert_eq!(offset[0], 200.0, "default x offset clears the governance graph");

    assert!(payload["default_spread"].as_f64().unwrap() >= 0.0);
    assert!(payload["default_point_size"].as_f64().unwrap() > 0.0);
}
```

The `scene_offset[0] == 200.0` check is load-bearing: that's the magic number that places the voicing cloud beside (not on top of) the governance-graph render in Prime Radiant. If a future refactor of `voicings_payload` in `crates/ix-agent/src/handlers.rs` flips it back to the origin, this test fails before anyone notices the cloud is hiding inside the graph.

### 2. Data-integrity assertion (gated on file presence)

The binary buffer test asserts the on-disk artefacts honour the contract the payload promises — but skips silently if the OPTIC-K corpus hasn't been materialised locally. The gating matters: CI runners and fresh clones don't have `state/viz/voicing-positions.bin` (it's derived by running `serve_viz` against the 313K-voicing corpus), and we don't want CI red just because a developer hasn't run a one-time data-prep step.

```rust
#[test]
fn binary_buffer_layout_matches_meta_when_present() {
    let bin_path = project_root().join("state/viz/voicing-positions.bin");
    let meta_path = project_root().join("state/viz/voicing-positions.meta.json");

    if !bin_path.exists() || !meta_path.exists() {
        eprintln!("skipping: voicing-positions.{{bin,meta.json}} not present (run serve_viz first)");
        return;
    }

    let meta_bytes = std::fs::read(&meta_path).expect("read meta");
    let meta: Value = serde_json::from_slice(&meta_bytes).expect("meta is JSON");
    let total = meta["total"].as_u64().expect("meta.total is integer") as usize;

    let bin_size = std::fs::metadata(&bin_path).expect("stat bin").len() as usize;
    let expected = total * 3 * 4;
    assert_eq!(
        bin_size, expected,
        "bin size ({bin_size}) must equal total ({total}) * 3 floats * 4 bytes ({expected})"
    );

    // Blocks must tile the buffer with no gaps or overlap.
    let blocks = meta["instruments"].as_array().expect("instruments array");
    let mut running = 0usize;
    for block in blocks {
        let name = block["name"].as_str().expect("name");
        let offset = block["offset"].as_u64().expect("offset") as usize;
        let count = block["count"].as_u64().expect("count") as usize;
        assert_eq!(
            offset, running,
            "{name} block offset ({offset}) must equal running cursor ({running}) — gap or overlap"
        );
        running += count;
    }
    assert_eq!(running, total, "sum of per-instrument counts ({running}) must equal total ({total})");
}
```

### The contract being asserted

Two invariants that the GA-side `createVoicingCloud` consumer relies on:

1. **Buffer size matches metadata exactly:** `bin_size_bytes == meta.total * 3 floats * 4 bytes`. Three.js reads the buffer as a `Float32BufferAttribute` of length `total * 3`; any mismatch silently truncates the cloud or reads out-of-bounds garbage.
2. **Per-instrument blocks tile the buffer with zero gaps or overlap:** for each block in `meta.instruments`, `block.offset == sum(prior_block.count)`, and the final running cursor equals `meta.total`. The consumer slices the buffer per instrument to apply colour-per-vertex; a one-voicing gap shifts every subsequent instrument's colour by one and produces an off-by-one rainbow that's invisible at point-cloud density but corrupts click-pick lookup.

Both invariants are produced by `ensure_position_buffer` in `crates/ix-voicings/src/bin/serve_viz.rs`, which buckets `voicing-layout.json` rows by instrument (preserving source order) and writes each block contiguously. The test pins that producer's output shape so a future refactor that, say, switches to a `HashMap` (non-deterministic order) or interleaves instruments would fail loudly.

## Live Verification

Numbers from the live run after this test landed:

- `meta.total`: **688,351** voicings
- `voicing-positions.bin` size: **8,260,212 bytes** = `688,351 * 3 * 4` exactly
- Per-instrument blocks tile cleanly:
  - `guitar`: offset `0`, count `667,125` → cursor `667,125`
  - `bass`: offset `667,125`, count `12,614` → cursor `679,739`
  - `ukulele`: offset `679,739`, count `8,612` → cursor `688,351`
- Final cursor `688,351` matches `meta.total`. No gaps, no overlap.

Both `payload_handler_matches_voicings_payload_v1_schema` and `binary_buffer_layout_matches_meta_when_present` pass. The `corpus_lazy_lookup_resolves_when_present` companion test (Phase 5) also passes against the locally materialised `state/voicings/bass-corpus.json`. The data-layer half of the smoke is now a regression gate; the Three.js render half is documented as manual verification in `docs/plans/2026-05-02-voicings-in-prime-radiant.md` until a Vite-friendly headless approach (Playwright with real wall-clock waits, or hitting the production `vite build` output instead of the dev server) is wired up.

## Prevention

### The pattern, generalized

When visual verification is blocked — headless browsers crash, the SPA won't render under automation, the screenshot tool times out — do not declare the integration "untested." Write a contract test at the data layer that exercises the same producer to consumer interface the visual would have proved. The load-bearing principle: a render is downstream of a contract; if the contract holds (schema, sizes, offsets, URL shapes), the render's failure modes collapse to "Three.js / DOM / CSS" — a much smaller, more localized search space. The visual smoke becomes complementary confirmation, not a prerequisite for shipping.

### The honest-disclosure rule

A contract test verifies the contract, not the rendering. Say so explicitly. The PR description, the plan doc, and the test's own module docstring must spell out what was machine-verified versus what is deferred to manual smoke. The shipped example does this in its header — "This is the data-layer half of the smoke test. The Three.js render half ... is left to manual browser verification — see the plan doc." Mirror that in `docs/plans/2026-05-02-voicings-in-prime-radiant.md`: "X verified by `voicings_payload_contract.rs`; Y deferred to manual browser smoke." Hiding the gap is worse than acknowledging it — the gap is real either way, and an unacknowledged gap rots into a false sense of coverage.

### The CI-friendly file-gating idiom

Tests that depend on optional large data (OPTIC-K corpora, generated viz buffers, anything multi-hundred-MB) must skip silently when the data is absent so CI without the fixtures still passes. The Rust idiom in `crates/ix-agent/tests/voicings_payload_contract.rs`:

```rust
if !bin_path.exists() || !meta_path.exists() {
    eprintln!("skipping: voicing-positions.{bin,meta.json} not present (run serve_viz first)");
    return;
}
```

Two rules: (1) `eprintln!` the reason and the command that would produce the data — future-you will forget; (2) split the schema-shape assertions (always run) from the data-integrity assertions (file-gated), so the always-on half still catches handler regressions on a fresh clone.

### Test cases to add proactively

When shipping a cross-repo schema — anything landing under `governance/demerzel/schemas/` or referenced by an MCP tool name — the producer side must ship a contract test covering, at minimum:

- [ ] `schema` literal matches the published name and version (`voicings.payload.v1`, not `v2` or empty)
- [ ] every required field per the schema is present and non-null
- [ ] enum / format discriminants match (e.g. `format == "binary-positions+meta"`)
- [ ] defaults are sane (non-negative spreads, positive sizes, well-formed `[x,y,z]` offsets)
- [ ] URL-shaped fields actually look like URLs (`starts_with("http")`, no trailing slash, no `null`)
- [ ] malformed-input fallback: passing junk arguments returns a structured error, not a panic
- [ ] on-disk artifacts (when present) honour declared sizes and tile without gaps or overlap

### One thing NOT to do

Do not pull in a JSON Schema validator crate (`jsonschema`, `valico`, `boon`) just to power one contract test. Targeted key-by-key assertions catch the same regressions — wrong literal, missing field, malformed URL, off-by-one buffer — at a fraction of the dependency, build-time, and audit-surface cost. The IX workspace is intentionally minimal (pure Rust, `wgpu` only for GPU); a transitive chain pulled in to validate one payload shape is exactly the kind of creep the constitution exists to prevent. If three or more crates eventually need schema validation, revisit; until then, write the asserts.

## Related Documentation

### Sibling solution docs

No prior solution doc covers headless-browser failures or contract-test fallbacks. Closest tangents:

- `docs/solutions/integration-issues/cross-pollination-4-repo-ecosystem.md` — ix/tars/ga/Demerzel federation overview; shares the cross-repo MCP context but says nothing about verification.
- `docs/solutions/build-errors/windows-app-control-blocks-cargo-test-binaries.md` — different Windows-specific failure mode (WDAC blocking test binaries) but same lesson: surface the platform constraint instead of pretending it isn't there.

### Related plans

- `docs/plans/2026-05-02-voicings-in-prime-radiant.md` — **parent plan**. Phase 4 row already records "Visual smoke (Three.js render in scene) deferred to manual browser verification — headless Chrome/Edge can't capture the GA Vite SPA reliably." This solution doc is the postmortem of why that became a contract test instead.
- `docs/plans/2026-05-02-stable-tier-promotion.md` — sibling-day plan, same audit discipline.

### User auto-memory entries (cross-machine recall for future sessions)

- `feedback_browser_verification_no_excuses.md` — 2026-05-02 pushback against "I couldn't open a browser to verify." Allows the "genuinely blocked" carve-out **only** if you say specifically why. The contract-test fallback is the honest version of that carve-out.
- `reference_ga_remote_ui_control.md` — GA Prime Radiant is remote-controllable via MCP→SignalR; suggested as the QA path instead of Playwright. Worth noting as the alternative path that could supersede the contract-only verification.
- `reference_ga_two_react_apps.md` — disambiguates which Vite SPA the headless browser was failing against (5176 canonical, 5173 usually off, GaApi 5232).
- `feedback_harness_engineering_gap.md` — "stop-loop after 3x same symptom"; same diagnosis discipline applied here.
- `feedback_completion_bias.md` — 80% built + small change to finish ≠ done. Honest framing of "data layer verified, render layer deferred" is the antidote.

### Code + schema cross-references

- Test: `crates/ix-agent/tests/voicings_payload_contract.rs`
- Producer: `ix_voicings_payload` handler in `crates/ix-agent/src/handlers.rs`; registration in `crates/ix-agent/src/tools.rs`
- Binary buffer producer: `crates/ix-voicings/src/bin/serve_viz.rs` (`ensure_position_buffer`)
- Browser-side consumer: `crates/ix-voicings/web/3d.html` (the static page that *does* headless-screenshot fine — useful baseline)
- Pre-derivation source: `crates/ix-voicings/src/viz_precompute.rs` (`DETAIL_SAMPLE_CAP` at line 337)

### Caveat — schema not at current submodule HEAD

The `voicings.payload.v1` schema file ships in Demerzel commit `cb78692` on branch `r3-registry-check-ci`, but the current ix submodule pointer (Demerzel `4af1b8e` on `master`) does not yet include it. This is the same `master` ↔ `r3-registry-check-ci` divergence captured in `project_demerzel_branch_divergence.md`. The contract test in this doc still works correctly because the *Rust producer* (`ix_voicings_payload` handler) and the *Rust consumer* (the test) both live in ix and don't depend on the demerzel schema file at runtime — the schema is currently a documentation-only artifact. Closing the gap is a follow-up: cherry-pick `cb78692`'s schema onto demerzel master OR merge the two branches.
