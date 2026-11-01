# Next steps (as of 2026-11-01)

## Done: 1 km grid training (16 blocks)

Ran 21:58 on 2026-10-30 to 12:55 on 2026-10-31 (about 56 min per block); log `artifacts/grid-1km-gpu.log`. 15 blocks
passed first time. B3 trained but failed export: 30 splats 255-338 m up (4.5e-5 of opacity mass) broke the 250 m
vertical gate, which the floater removal did not cover. The removal rule was amended after that failure (recorded in
`FLOATER_POLICY_NOTE`, `pipeline/export.py`): it also removes centres more than 250 m above or below the anchor, under
the same 1% mass cap. B3 then reran from export with its training reused (31 splats removed, 8.3e-5 of mass). The other
15 blocks have no splats beyond 250 m, so their exports keep the same splats; their export stage reruns (code hash) the
next time each job runs.

### Stopping and resuming

Stop the loop first, then the running job (the job stops its training process group itself):

```bash
ps -ef | grep -E 'for c in a1|pipeline.job' | grep -v grep   # loop (bash -c ...) and job PIDs
kill <loop PID>     # first: otherwise the loop marks the block FAILED and starts the next one
kill <job PID>      # SIGTERM: the job terminates training and exits
nvidia-smi          # check no train.py is left
```

Resume by starting the same loop again. Every completed stage of every block is reused (`REUSE`), so finished blocks
take seconds. The stage that was interrupted restarts from its beginning: an interrupted training loses that block's
training so far (up to about 45 min); training checkpoints are not resumed.

Do not edit code that a stage hashes (its `-m` module, scripts in its command, `scripts/stage1-env.sh`) before
resuming, or completed stages count as changed and run again.

## 1. Grid: live (2026-10-31)

All 16 blocks are published and active in `artifacts/releases`; the central `jax068-2015` block is retired
(`pipeline.publish --retire-dataset`). The A2|A3 and A2|B2 seam failures are accepted as known issues for this
version (docs/phase3/GRID_1KM_RESULT.md). Deferred: diagnose and fix A2's southern edge; stop the viewer rendering
for up to 20 s after loading finishes.

## 2. Semantics: paused at v5 (2026-11-01)

Stopped here by decision: v5 is good enough to build on, and the remaining gaps are known. Full record in
docs/SEMANTICS.md (v1-v5).

- **Status**: v5 (OpenEarthMap SegFormer-B2, Kaggle run v2) failed its pre-registered targets on JAX_068 but is the
  best by far: accuracy 0.724, building IoU 0.690 (passes), tree IoU 0.366 against 0.80 / 0.60 / 0.50. Without the
  building height rule the same predictions score 0.857 / 0.806 / 0.366 (diagnostic, not a pass).
- **Known gaps**: the 2.5 m height rule turns low-placed roofs into ground; many tree crowns that are raised in our
  model are ground in DFC2019's labels, which caps tree IoU.
- **Assets**: model `artifacts/landcover/kaggle-v2/v5-oem-segformer-b2/best.safetensors` (sha256 in
  `configs/semantics-v5.json`; labels CC BY-NC-SA, non-commercial); JAX_068 run `artifacts/semantics/v5-jax068`;
  code `pipeline/landcover_*.py`, `pipeline/semantics_landcover.py`.
- **Grid classes (2026-11-01)**: v5 applied to all 16 grid blocks; lidar-covered cells score 0.596 / 0.602 / 0.334
  (docs/SEMANTICS.md). Segment releases `jax068-2015-grid-*-r2` are active in the review catalogue only, waiting
  for a visual check before going live (`pipeline.publish --config artifacts/semantics/v5-grid/<block>/release-r2.json
  --activate`). The authors' scenes are not getting layers.
- **When resumed**: (1) pre-register v6 without the building height rule; (2) decide whether the tree target can be
  met against DFC2019's tree labels or needs another reference; (3) blind test once on the other seven DFC2019 scenes
  (JAX_004, 164, 168, 175, 214, 260, 264; lidar in `artifacts/reference/dfc2019-jax/`, placement offsets measured
  first; do not inspect their labels before the method is fixed); (4) build segment layers (`pipeline.segments`) for
  the 12 authors' scenes (decided 2026-11-01: not for now).

## 3. Parked

- FlashSplat-style label lifting (per-pixel splat contributions instead of centres).
- Pléiades Neo for training: `pipeline.acquisitions` / `pipeline.block_inputs` read WorldView-3 only (the
  pre-purchase metadata check already supports it).
- True-neighbour experiment: waits for a qualified lidar reference (USGS 3DEP failed its checks).
- NYC empty-holdout change in `git stash`; the batch runner's shared DSM file.
