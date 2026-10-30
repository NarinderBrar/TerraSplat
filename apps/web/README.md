# TerraSplat web app

React, Vite and CesiumJS 1.146.0 globe for the [release API](../../services/api/README.md).
Requires Node 22.12+ or 24+, the API environment and published `artifacts/releases`.

Run from the repository root:

```bash
npm ci
npm --prefix apps/web ci
npm run dev:app
# http://127.0.0.1:5173
npm run build:app
npm run test:app
```

`dev:app` starts the API on port 8000 and web app on 5173; Ctrl+C stops both.

To test from a phone on the same network, run `npm run dev:app:lan` and open
`http://<laptop-LAN-IP>:5173` (`hostname -I` shows it). The web server then
accepts LAN connections; the API stays on loopback behind the Vite proxy. A
host firewall must allow TCP 5173 (for ufw: `sudo ufw allow 5173/tcp`).
The existing `dev`, `build`, `test` and pilot browser commands remain separate.
The app build is in `apps/web/dist`; serve it with `/api` and `/releases` routed to the API,
or run `npm --prefix apps/web exec vite -- preview --host 127.0.0.1 --port 4173` locally.
Cesium assets are copied locally by the prepare step and excluded from Git.

### Running Chrome on the NVIDIA GPU (Linux hybrid-graphics laptops)

On laptops with Intel + NVIDIA graphics in PRIME "on-demand" mode, Chrome draws on the integrated Intel GPU.
Splat rendering is GPU-bound: Chrome traces on 2026-10-30 showed our code using only 2–16% of the main
thread while the Intel UHD (TGL GT1) was 68–95% busy, giving about 10 fps while moving the map. On the
RTX 3060 the same views were about 10× faster. Start a separate Chrome on the NVIDIA GPU with:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only google-chrome \
  --user-data-dir=/tmp/chrome-nvidia --use-angle=vulkan \
  --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan http://127.0.0.1:5173/
```

Why each part is needed:

- `__NV_PRIME_RENDER_OFFLOAD=1` and `__VK_LAYER_NV_optimus=NVIDIA_only` ask the driver to render this
  process on the NVIDIA GPU and expose only that GPU to Vulkan.
- `--use-angle=vulkan` and the `Vulkan…` features make Chrome's WebGL use Vulkan. The usual GLX/EGL
  offload variables do not work here: Chrome's window format comes from the Intel driver and the NVIDIA
  EGL driver rejects it (`eglInitialize: Invalid visual ID requested`), so WebGL falls back to software
  (SwiftShader) or fails.
- `--user-data-dir` forces a new Chrome process. Without it, an already-running Chrome simply opens a new
  window and ignores all of the settings above.

Check `chrome://gpu`: `GL_RENDERER` should name the NVIDIA GPU. With this setup Chrome also refuses WebGL
contexts that request `powerPreference: "high-performance"` (Cesium's default), so the app probes for that
and falls back to the browser default; see `powerPreference()` in `src/globe.ts`.

## Configuration

Vite variables are set before dev/build, for example:

```bash
VITE_REMOTE_IMAGERY=false npm run dev:app
VITE_API_BASE=https://releases.example.org npm run build:app
```

- `VITE_API_BASE`: API origin, default same origin. Used for metadata, release tiles and evidence.
  Cross-origin deployments must configure the API's CORS allowlist.
- `VITE_REMOTE_IMAGERY`: `false` disables Esri completely; default enabled.
  Bundled Natural Earth II is always local. Esri World Imagery appears below 1,000 km camera height;
  its source attribution stays in Cesium's visible credits. Review Esri terms before commercial launch.
- `TERRASPLAT_API_PROXY`: dev/preview API target, default `http://127.0.0.1:8000`.
- `VITE_TEST_BRIDGE=true`: exposes `window.terra` for browser diagnostics; leave unset for production.

No ion token, ion imagery, geocoder, Bing or terrain service is used.

## Controls

| Input | Action |
| --- | --- |
| Left drag / one finger | Grab and pan the globe |
| Wheel / pinch | Smooth zoom toward the cursor / pinch centre |
| Right drag / Ctrl + left drag | Rotate and tilt |
| Two fingers | Rotate, tilt and pinch zoom |
| Double click | Zoom one step toward the cursor |
| Compass / top-down button | Reset heading north / look straight down |
| + / − | Zoom toward the viewport centre |
| 3D Footprints | Toggle coverage outlines and orbit-scale markers |

Hover coverage for dates and lidar MAE; click the marker/outline or its keyboard-accessible list
button to frame the block. Regional indexes load automatically at startup; Cesium streams visible
content as the camera moves. The release panel can substitute historical releases without an
extra load button. Other blocks in the region retain their active releases. Tile diagnostics show
loaded/visible counts, bounding volumes and Cesium FPS.

## Rendering and limits

The pilot ground is roughly 17 m below the ellipsoid. Data stays unchanged. As in the pilot's optional
imagery overlay (alpha 0.55), globe translucency allows those splats to remain visible. Here the front
alpha is 0.12 within 800 m and rises to 1 at 8 km, retaining an opaque Earth in orbit. No terrain is
loaded; disabling terrain depth tests alone would not expose splats beneath an opaque globe.
See [Cesium globe translucency](https://cesium.com/learn/cesiumjs/ref-doc/GlobeTranslucency.html).

The controller uses Cesium's cursor-based zoom and 3D globe grab via `rotateEventTypes`;
`translateEventTypes` applies only to 2D/Columbus view ([controller reference](https://cesium.com/learn/cesiumjs/ref-doc/ScreenSpaceCameraController.html)).
Camera height is bounded to 35 m–30,000 km. Tiles use SSE 8, 128 MiB cache plus 32 MiB overflow,
movement request culling, foveated SSE, fog and request-render mode. REPLACE splats keep skip-LOD off.
Worker completion explicitly requests a frame so decoding/sorting can finish after camera movement stops.
These are initial budgets, not measured GPU performance claims.

Cesium 1.146 chooses its splat decoder using only the outer tileset's glTF declarations. The app loads
each dataset's release tileset directly, and the publisher also repeats the releases' declarations on the
regional indexes for other viewers.

## Validation

`test:app` runs Vitest and Playwright against the real API on port 8001 and Vite on 5174.
Browser tests skip when the local release catalogue is absent. Chromium defaults to
`/usr/bin/google-chrome`; override `CHROME_PATH` if needed. Remote imagery is disabled and external
requests blocked and asserted absent. Screenshots and pixel measurements go to
`apps/web/test-results/`. The test checks whole-globe startup, map hover/click, camera placement,
splat fragments inside the projected block, historical release switching and narrow layout.
Software rendering verifies correctness; it does not establish the 30 FPS target.

Follow-ups: real GPU FPS measurements on a fixed camera route (30 FPS target), jobs endpoint,
semantic layers/queries, lidar DSM comparison when a delivery layer exists.

Reveal effects: newly shown tiles sweep in as glowing points, then grow into their unchanged splats; refinement
swaps instantly. Toggle "Reveal effects" in the panel; off for reduced motion. [Design](../../docs/REVEAL_EFFECTS.md).
