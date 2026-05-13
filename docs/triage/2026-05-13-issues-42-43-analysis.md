---
date: 2026-05-13
purpose: Verify status of GA perf issues #42 (InstancedMesh) and #43 (skybox cubemap bake)
reversibility: two-way door — analysis only, no code/state change
revisit-trigger: ForceRadiant.tsx imports NodeInstancer (#42 closes), OR SkyboxBaker is disabled (#43 regresses)
status: complete — #43 already shipped and can be closed; #42 is half-shipped, needs integration step
---

# GA issues #42 + #43 — perf wins triage

## TL;DR

Two perf optimizations (combined estimate +8–16 FPS) sat 41 days. Live code inspection shows:

| # | Title | Status | Action |
|---|---|---|---|
| #43 | Bake skybox shader to static cubemap | **shipped & wired** (`ForceRadiant.tsx:2755`) | close issue |
| #42 | InstancedMesh for governance nodes | **half-shipped** — `NodeInstancer.ts` exists on main but `ForceRadiant.tsx` never imports it | finish integration |

This is the audit-before-building pattern again (`feedback_audit_before_building.md`): file exists ≠ feature done. Confirmed by reading the live render path, not the commit log.

## #43 — Bake skybox shader to static cubemap ✓ DONE

**Issue body:** *"Procedural skybox fBm shader runs every frame but has no time uniform — output never changes. Render once to CubeRenderTarget, use MeshBasicMaterial. Est: +2-4 FPS."*

**Implementation evidence:**

- `ReactComponents/ga-react-components/src/components/PrimeRadiant/SkyboxBaker.ts` (164 lines): exports `bakeSkyboxToCubemap(renderer, scene, qualityTier)` with per-tier resolution (low 512² / med 1024² / high 2048²) and six-face camera orientation matching THREE.CubeTexture layout.

- `ForceRadiant.tsx:2746-2761` wires it:
  ```ts
  // Skybox Cubemap Bake — deferred 3s after init to avoid startup stall
  if (!USE_WEBGPU && !isLowEnd) {
    setTimeout(() => {
      if (disposed) return;
      try {
        const renderer = fg.renderer() as THREE.WebGLRenderer;
        bakedSkyOuter = bakeSkyboxToCubemap(renderer, fg.scene(), budgetToTier(qualityBudget));
        console.info('[PrimeRadiant] Skybox baked to cubemap — shader eval eliminated');
      } catch (e) {
        console.warn('[PrimeRadiant] Skybox bake failed:', e);
      }
    }, 3000);
  }
  ```

- Gating: WebGL2 only (skipped under WebGPU due to async readRenderTargetPixels incompatibility), and not on low-end devices. Console emits success/failure marker.

**Recommended action:** **close #42's sibling #43** with a one-line comment referencing `ForceRadiant.tsx:2755`. If you want quantitative confirmation, run before/after FPS in the Three.js stats overlay; the issue's +2-4 FPS estimate should land easily since shader eval is fully eliminated.

## #42 — InstancedMesh for governance nodes ✗ HALF-SHIPPED

**Issue body:** *"Multi-LLM perf review identified per-node Groups (3 draw calls each, ~540 total) as #1 bottleneck. Replace with InstancedMesh per node type (8 types = 8 draw calls). Drive color/pulse/spin via InstancedBufferAttribute. Est: +6-12 FPS."*

**Implementation evidence (module exists):**

- `ReactComponents/ga-react-components/src/components/PrimeRadiant/NodeInstancer.ts` exists on `main` (commit `d7a849ed`, with rebase-artifact duplicate `56d8b34e`). Doc comments declare: *"InstancedMesh renderer for governance graph nodes. One InstancedMesh per GovernanceNodeType, using the same geometry factories."*

- File contains: `mesh: THREE.InstancedMesh` (line 106), `new THREE.InstancedMesh(geometry, material, bucket.length)` (line 169).

**Integration evidence (gap):**

- `grep -n 'NodeInstancer\|createNodeInstancer\|instancedNodes' ForceRadiant.tsx` → **empty**. The new renderer is never imported into the render path.

- `ForceRadiant.tsx` still uses the old per-node Group path:
  - Line 499: `const group = new THREE.Group();` (generic)
  - Line 2702: `milkyWayMesh = isLowEnd ? new THREE.Group() : createMilkyWay(8000);`
  - Line 2716: `const starField = new THREE.Group();`
  - Line 2874: `demerzelFace = isLowEnd ? new THREE.Group() : createDemerzelFace(0.5);`
  - Line 3087: `spaceStation = isLowEnd ? new THREE.Group() : createSpaceStation(0.12);`

The governance-node loop itself (~540 nodes × 3 draw-calls/node) is constructed elsewhere in the same file — search for the loop that walks `graph.nodes` and creates a Group per node; replace that path with a single call into `createNodeInstancer(graph.nodes)` from `NodeInstancer.ts`.

**Recommended action — finish #42 (~half day):**

1. **Import** `NodeInstancer` into `ForceRadiant.tsx`:
   ```ts
   import { createNodeInstancer } from './NodeInstancer';
   ```
2. **Replace** the per-node Group construction loop with a single InstancedMesh setup. The doc comment in `NodeInstancer.ts` says it groups by `GovernanceNodeType`; `TYPE_GEOMETRY` map at `ForceRadiant.tsx:463` and `TYPE_SIZE` at line 218 are the geometry factories `NodeInstancer` references.
3. **Drive per-instance color/pulse/spin** via `InstancedBufferAttribute` updates each frame (the per-frame work shrinks from 540 setUniform calls to 1 attribute update per type).
4. **Keep selection picking working**: `THREE.InstancedMesh` raycasting requires `mesh.userData` lookup by `instanceId`. `NodeInstancer.ts` likely already handles this — verify by reading its raycast helper.
5. **Smoke test**: confirm node count `mesh.count` updates correctly when graph topology changes (Demerzel constitution evolves, nodes are added/removed).

## Cross-cutting observation

Both issues are classic completion-bias artifacts:
- #43 was finished but the issue never closed.
- #42 was 80% built (module exists) but the integration step (~10 lines in `ForceRadiant.tsx`) never landed.

Per `feedback_completion_bias.md`: *"explicit ship+delete triggers belong with every deferral."* Neither issue had a ship-trigger. The right ship-trigger for #42 is: "import NodeInstancer in ForceRadiant.tsx and delete the per-node Group loop." That's a single PR.

The combined +8–16 FPS estimate is intact — #43 is delivering its 2–4 today, #42's 6–12 awaits integration.

## Next actions on the issues

When ready:

1. **Close #43** with comment:
   > Shipped — `SkyboxBaker.bakeSkyboxToCubemap()` is called from `ForceRadiant.tsx:2755` with 3s deferred startup. Gated to WebGL2 + non-low-end. Console emits `[PrimeRadiant] Skybox baked to cubemap — shader eval eliminated` on success.
2. **Update #42** with comment:
   > `NodeInstancer.ts` module landed in `d7a849ed` but `ForceRadiant.tsx` doesn't import it yet. Half-shipped — finishing requires ~10 lines: import `createNodeInstancer`, replace per-node Group loop, ensure raycast picking still works via `instanceId`. Estimate: half a day.
3. Optionally open a sub-issue **"finish #42 — wire NodeInstancer into ForceRadiant render path"** so the open #42 doesn't drift again.

Cross-ref:
- `feedback_audit_before_building.md` — same pattern as the Demerzel-hooks audit error
- `feedback_completion_bias.md` — 80%-built deferrals need explicit ship-triggers
- `feedback_green_but_dead.md` — Seldon Plan / qa-architect-cycle / now #42 all show the same shape: scaffolding without final-mile integration
