# Splat reveal effects (design, option A)

Status: design, agreed 2026-10-30; not implemented. Inspired by Spark's
[splat-reveal-effects](https://github.com/sparkjsdev/spark/blob/main/examples/splat-reveal-effects/index.html)
example, implemented inside our existing Cesium renderer.

## Behaviour

1. **Coarse levels render as a point cloud.** Splats in non-leaf (merged parent) tiles are drawn as small round
   points of their own colour, not as blurry Gaussians.
2. **The finest level (LOD0, leaf tiles) is unchanged.** It is the final tile a user sees. When its reveal has
   finished, it renders exactly as today: same shape, colour and opacity. Any reveal effect must end on this state.
3. **Two-stage reveal**, played per tile as it arrives:
   - **Stage 1 (points):** a coarse tile's points appear behind a sweep that spreads from the block centre, with a
     brief glowing edge at the sweep front.
   - **Stage 2 (splats):** when a leaf tile arrives, its splats start as points and grow into their full Gaussian
     shape, with colour easing from the point tint to the true colour.
   Under REPLACE refinement Cesium swaps a parent for its children in one frame, so stage 2 starting from points
   makes that swap look continuous.

Effects change size, opacity and colour only, not splat positions (see *Performance*). Spark's spiral, rain and
unroll effects move splats and are out of scope.

## Why Cesium must be patched

Findings from the Cesium 1.146 source (`@cesium/engine/Source/Scene/GaussianSplatPrimitive.js` and
`Shaders/PrimitiveGaussianSplatVS.glsl`/`FS.glsl`):

- **No effect hook.** The splat shader is fixed. `CustomShader`, available for ordinary models, is not applied to
  Gaussian splats.
- **One merged batch per tileset.** `GaussianSplatPrimitive` concatenates the splats of all selected tiles, in
  `tileset._selectedTiles` order, into one attribute texture, sorts them and issues one instanced draw. Once merged,
  the GPU does not know a splat's tile, level or arrival time.
- **Reachable from the app.** `GaussianSplatPrimitive` is a public export. The update loop calls
  `GaussianSplatPrimitive.transformTile(tile)` once per selected tile, in merge order, and
  `GaussianSplatPrimitive.buildGSplatDrawCommand(primitive, frameState)` after each merge. Both are looked up on the
  class at call time, so the app can wrap them at runtime without rebuilding Cesium.

## Design

**Per-splat reveal data.** A second texture alongside Cesium's splat attribute texture, using the same row layout
(`u_splatRowMask`, `u_splatRowShift`), with per splat:
- `arrival` — the tile's first-visible time in seconds (shared app clock);
- `leaf` — 1 for leaf tiles, 0 for merged parents.

The order is recorded by wrapping `transformTile` (grouped per tileset and per merge), then expanded with each
tile's `content.pointsLength`. It is truncated exactly like Cesium's own data when the splat budget trims
`numSplats`. Arrival times live in a `WeakMap<Cesium3DTile, number>`, set the first time a tile appears in a merge,
so a tile that reappears is not revealed again.

**Draw command.** Wrap `buildGSplatDrawCommand`: after the original builds the command, swap in a shader program
compiled from the original vertex/fragment sources with our code injected, and extend its `uniformMap` with
`u_revealTexture`, `u_revealNow` (app clock) and `u_revealOrigin` (block centre in the primitive's ENU frame).
Programs come from Cesium's shader cache, so they are compiled once.

**Vertex shader changes** (around `calcCovVectors`), with progress
`p = clamp((now − arrival − delay(distance from origin)) / duration, 0, 1)`:
- non-leaf: replace the covariance axes with a fixed screen radius (about 2 px, scaled by `p` for the sweep), bypass
  the "too small" discard for these points, and add the sweep-edge glow to the colour;
- leaf: `mix(point axes, true axes, ease(p))` and `mix(point tint, true colour, ease(p))`;
- **when `p == 1` the leaf path skips every change**, so final output equals the current shader's.

**Render loop.** The app uses request-render mode. While any reveal is active it requests a frame each animation
frame, then stops when every tile has `p == 1`.

**Settings.** A "Reveal effects" toggle in the panel, on by default; reveal duration around 1.2 s. Effects are
skipped when the user prefers reduced motion (`prefers-reduced-motion`) or the toggle is off.

## Performance

- Steady state: no visible change. One extra texture read and a few arithmetic operations per splat corner; extra
  memory about 4–8 bytes per splat (≈ 4–8 MB at 1 M splats).
- The point stage is cheaper than today: small points cover far fewer pixels than blurred coarse splats, and pixel
  blending was the measured bottleneck on integrated GPUs.
- During a reveal the app renders continuously for about 1–2 s per tile wave. That is easy on the RTX 3060 and may
  stutter on weak integrated GPUs, like map movement does today.
- Splats are sorted by their unmodified positions. Effects that moved splats would blend in the wrong order while
  animating, which is why effects only change size, opacity and colour.

## Risks

- **Coupled to Cesium internals** (`_selectedTiles`, snapshot timing, shader text). Cesium is pinned at 1.146.0; a
  startup self-check should disable the effects (and log once) if the expected functions or shader markers are
  missing, so the viewer falls back to today's rendering instead of breaking.
- **Asynchronous merges:** Cesium builds merged textures across frames. The reveal texture must follow the merge
  that is actually drawn, not the newest selection.

## Plan

1. **Spike (½ day).** Wrap the two functions, inject a no-op shader edit and a constant-colour point mode. Confirm
   the merge order matches and the program swap works in a dev build.
2. **Implement** the reveal texture, the shader changes, render-loop requests, the toggle, reduced motion and the
   self-check.
3. **Tests:**
   - unit tests for progress, easing and merge-order expansion, including budget truncation;
   - a browser test that the finished state matches today's: render a leaf tile with effects off and after the
     reveal finishes, and require a pixel-identical (or within 1/255) screenshot;
   - a browser test that coarse tiles draw as points and that the render loop goes idle after reveals;
   - screenshots of both stages.
4. **Profile** on the RTX 3060 and the Intel GPU (Chrome traces, as in `apps/web/README.md`).

Out of scope: drawing splats with Spark/three.js over Cesium (option B), and effects that move splats.
