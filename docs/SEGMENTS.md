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

In the shader, with no class chosen every labelled splat is tinted its class colour and unknown splats turn grey; with
classes chosen (one bit per code), those are tinted and brightened and the rest fade to dim grey. The highlight is
static, so request-render mode still draws only on change.

## Current layer

`jax068-2015-19v-r2` (same tiles as r1) carries the semantics v2 classes: buildings are usable (IoU 0.70 against
DFC2019); trees, grass and ground are experimental (overall accuracy 0.68). See docs/SEMANTICS.md. A better classifier
(v4/v5) will be published as a new layer on the same tiles.
