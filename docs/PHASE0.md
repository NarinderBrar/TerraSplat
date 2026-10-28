# Phase 0: codec and placement compatibility

Scope: pipeline tooling and a static diagnostic viewer. The product web app,
job API, catalogue, terrain/imagery integration and training are deferred.

## Run

Requires Node.js 22.12+ (or 24+) and Python 3.10+ for the optional inventory.

```bash
npm ci
npm run build
npm test
npm run test:browser
npm run dev
```

The browser test uses `/usr/bin/google-chrome`. Set `CHROME_PATH` to another
Chromium executable if needed. It uses software WebGL for reproducible local
compatibility checks; this does not measure laptop GPU performance.

The viewer runs on localhost. Cesium workers and WASM are copied from the locked
npm package, so the fixture requires no ion token, remote basemap or credentials.
`npm run build` regenerates the fixture and copies Cesium runtime assets.
`public/cesium/`, build outputs and browser reports are ignored by Git.

## Format contract

- Runtime: CesiumJS **1.146.0**, with transitive packages fixed by `package-lock.json`.
- Encoder: **@adobe/spz 0.2.3**, explicitly requesting **SPZ header v2** with gzip.
- Wrapper: `scripts/generate-fixture.mjs`, versioned with this repository.
- Extension identifiers: `KHR_gaussian_splatting` with nested
  `KHR_gaussian_splatting_compression_spz_2`, both required.
- The tileset must declare both glTF extensions through `3DTILES_content_gltf`;
  otherwise Cesium selects its ordinary model content path. The tileset-level
  geometric error must be nonzero so traversal reaches the zero-error leaf.
- Runtime profile reference: [Cesium glTF fork at 511dbc90](https://github.com/CesiumGS/glTF/tree/511dbc90beacddd0085f7b6022a36da332a3b836).
- This fixture follows Cesium's compressed attribute path: position, RGBA,
  anisotropic scale and XYZW quaternion. It is not a claim of full conformance
  to every revision of the ratified base extension.
- SH degree 0 only; no view-dependent SH conversion is established here.
- Ellipse kernel, sRGB display color; SPZ inputs use log scales, logit opacity,
  and SH DC coefficients. Encoder conversion is disabled explicitly.
- Storage axes are glTF `(east, up, -north)`. Cesium applies glTF Y-up to
  tile Z-up; the root transform then places metric ENU in ECEF.

The synthetic origin uses the JAX_068 baseline longitude/latitude with an
**invented 100 m ellipsoidal height**. It establishes a coordinate test, not
registration of the original Skyfall scene. All colored axis centres lie
10 metres from the origin. Amber has a 45° quaternion rotation and 0.5 opacity.

`public/phase0/compatibility.json` records the codec header, dependency pins,
quantization errors, asset SHA-256 hashes and Cesium decoder WASM hash.
`fixture.json` is the independently readable expected geometry.

## Validation

The generator encodes and decodes SPZ and bounds quantization error in position,
log scale, color, logit opacity and quaternion orientation. Structural tests check
GLB chunk lengths, embedded codec version, attribute counts, spatial bounds,
asset hashes and the WGS84 placement matrix.

The browser gate must inspect rendered fragments, including coloured centres,
ellipse direction and opacity, rather than rely on a successful HTTP request or
an all-tiles-loaded event. Its JSON measurements and screenshot are written to
`test-results/`. A real fused scene, higher SH degrees, LOD requests and seam
continuity belong to subsequent phases.

Initial rendered check: all four colored ellipses were within 0.73 pixels of
their expected centres, and within 0.015 radians of their expected projected
orientations. This is a synthetic compatibility result, not a real-scene accuracy
measurement.

## Local input inventory

```bash
npm run inventory -- --output artifacts/phase0/inventory.json
# Override sibling directories when running elsewhere:
npm run inventory -- --reference-root /path/to/Ref --terrascope /path/to/TerraScope
```

The command only reads reference projects. It hashes the JAX_068 camera files,
records the Skyfall commit and submodule state, looks for ENU observer metadata,
and reports the available manual registration and GPU build tools. Missing
inputs are reported explicitly; the fixture is independent of them.

On the initial environment:

- Skyfall commit: `73194842a404910c5bbff006359eedfc56a98d60`.
- Prepared JAX_068 camera files are present, but its ENU observer metadata is absent.
- TerraScope has a visual registration and a representative flat ground datum;
  these do not establish surveyed ellipsoidal heights.
- Skyfall CUDA submodules are uninitialized; PyTorch and `nvcc` are absent from
  the active environment. The sandbox could not access the driver; a later host
  check during Phase 2 confirmed the RTX 3060 Laptop GPU (6 GB) is available.

Training setup cannot pass until the CUDA environment works. Reference projects
are left untouched. Do not run training or treat synthetic placement as evidence
of real-scene accuracy.
