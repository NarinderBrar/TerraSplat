# Segment mode

The viewer can colour splats by semantic class and highlight chosen classes ("Segment mode" in the side panel). All
buildings of any kind form one group, as do roads, trees, grass, ground and water.

## Data: one class byte per splat

A release may carry a segment layer next to its tiles:

    tiles/<id>.glb          SPZ splats (unchanged)
    segments/<id>.cls       uint8 class per splat, in the tile's splat order
    segments/segments.json  label, class codes and colours, per-tile splat counts and hashes

Codes are `pipeline/semantics.py` `CODES`: 0 unknown, 1 ground, 2 road, 3 building, 4 trees (vegetation), 5 water,
6 grass. 0 means unknown, so missing or unlabelled data never lights up as a class.

Classes are not a glTF attribute: Cesium decodes every attribute of an SPZ-compressed splat primitive through the SPZ
decoder, so an extra accessor would break loading. A sidecar keeps the tiles byte-identical, adds about 1 byte per
splat (the 19-view block: 132 KB of classes against 1.6 MB of tiles), is downloaded only when segment mode is on, and
lets a better classifier replace the layer without re-tiling.

Build a layer from a semantics run's `splat-classes.npz` (fused-PLY order) and the release's tile plan:

```bash
.venv-imagery/bin/python -m pipeline.segments --tiles artifacts/tiles/pilot-2015-19v-recipe-01-r1 \
  --export artifacts/stage1/pilot-2015-19v-recipe-01/export-r2 \
  --classes artifacts/semantics/jax068-2015-19v-r1-v2/splat-classes.npz \
  --label "Semantics v2 (experimental): …" --output artifacts/segments/<name>
```

Leaf tiles hold the original splats; each takes its own class, matched by position (the 19-view block matches within
6e-6 m). Coarse tiles hold merged splats; each takes the most common known class of its 16 nearest original splats.
A release config adds `"segments_dir"`; `pipeline/publish.py` checks the layer covers every tile, splat for splat,
copies it into the release and records `segments` in the manifest.

## Viewer

`apps/web/src/reveal.ts` already gives Cesium's merged splat batch a per-splat texture (tile arrival, mode, placeholder
time, leaf flag). The class is stored in the same texel above the leaf bit (`leaf + 2 * class`), so segment mode needs
no extra texture or draw pass. Class files are fetched per tile when segment mode is on (one `segments.json` probe per
release; releases without a layer are skipped) and the texture is re-uploaded when they arrive.

The side panel always lists the classes of a release with a segment layer (no mode switch). With none selected the
scene shows its real colours. Selected classes (one bit per code) become holograms in the shader, ported from
TerraScope's segmented view (`SegmentDrawAnimation.ts`): the class colour with a metric grid (4 m on buildings, 2 m
otherwise), an upward scan band and trails, and a pulse phased per class. Every other splat keeps its real colour and
alpha. Colour is set per splat from its centre, so with this pipeline's 1-3 m splats the grid reads as texture rather
than crisp lines. While a class is selected the scan and pulse redraw every frame; reduced motion and the reduced-quality
(weak GPU) profile keep the hologram still, so request-render mode draws only on change.

Buildings use TerraScope's segmented-view blue (shader RGB 0.1, 0.65, 1.0), building pulse phase, and line widths.
The JAX_068 authors' scene also uses TerraScope's 1.5 grid scale (6 m XY, 4.5 m vertical); other scenes use 4 m XY
and 3 m vertical. The Buildings swatch shows the highlight blue. Stored class codes and diagnostic raster palettes
remain the pipeline's palette.

The hologram blends with the original splat colour: its contribution pulses from 20% to 30%, retaining at least
70% of the base colour. Reduced motion and reduced quality use a steady 25% blend.

## Current layer

Segmentation is planned for the Skyfall-GS authors' scenes only (their models have the best image quality). They are
RGB only, so the classes will come from an RGB classifier (semantics v4/v5, docs/SEMANTICS.md): tuned on JAX_068, then
scored blind on the other seven DFC2019 Jacksonville scenes, and published as `skyfall-*-final-r3` layers on the same
tiles.

The first layer, `jax068-2015-19v-r2` (pipeline 19-view block, semantics v2 classes), proved the format and viewer and
was then withdrawn from the catalogue (`python -m pipeline.publish --withdraw-release ID`; files kept under
`artifacts/releases/.withdrawn/`).
