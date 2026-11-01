# Next steps (as of 2026-10-30)

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

## 2. Semantics and segment mode: authors' scenes only

Segment mode (docs/SEGMENTS.md) is built and verified; layers will be made for the 12 Skyfall-GS authors' scenes only.
Their data is RGB, so v2 (NDVI) does not apply.

1. **v4 failed** (2026-10-31): RemoteCLIP features scored 0.471 accuracy, building IoU 0.31, vegetation IoU 0.22
   against 0.80 / 0.60 / 0.50 (docs/SEMANTICS.md).
2. **v5 (in progress)**: a land-cover segmentation model for sub-metre imagery plus height, fused with the same
   verified multi-view projection. Tune on JAX_068. The model is SegFormer-B2 trained on OpenEarthMap (no
   Jacksonville, no DFC2019/US3D; labels mostly CC BY-NC-SA, so non-commercial).

   **Active run (started 2026-11-01): Kaggle notebook `npsbrar1986/terrasplat-landcover-v5`, version 2, T4 GPU**
   (https://www.kaggle.com/code/npsbrar1986/terrasplat-landcover-v5). Settings `configs/landcover-v5-kaggle.json`
   (batch 8, 40,000 steps, validation every 4,000). Inputs: the private dataset
   `npsbrar1986/terrasplat-landcover-bundle` (code, mit-b2 weights, split lists, checksums of our Zenodo download;
   built by `scripts/kaggle-landcover-bundle.py`) and the public copy `aletbm/global-land-cover-mapping-openearthmap`,
   whose 5,374 train/val files are checked against those checksums before training. Version 1 stopped at that check
   (the copy omits the 813 xBD labels that have no images; fixed). The notebook stops itself at 11 h and keeps its
   checkpoints. Check and collect it:

   ```bash
   env -u PYTHONPATH kaggle kernels status npsbrar1986/terrasplat-landcover-v5
   env -u PYTHONPATH kaggle kernels output npsbrar1986/terrasplat-landcover-v5 -p artifacts/landcover/kaggle-v2
   ```

   If it stopped at 11 h before step 40,000, attach that version's output to the notebook and push again: it resumes
   from `last.pt`. Then: write `configs/semantics-v5.json` (class mapping, height rules, targets as v3/v4) before any
   JAX_068 prediction, predict the 17 views with `pipeline.landcover_predict`, project and score.
3. **Blind test**: fix the method, then score once on the other seven DFC2019 scenes (JAX_004, 164, 168, 175, 214,
   260, 264). Their lidar DSM/CLS go in `artifacts/reference/dfc2019-jax/` (being downloaded); each scene first
   needs its placement offset measured like JAX_068. Do not inspect their labels before the method is fixed.
4. Build segment layers (`pipeline.segments`) for all 12 authors' scenes and publish `skyfall-*-final-r3` (same tiles
   plus layer). The four NYC scenes get layers without an accuracy number.

## 3. Parked

- FlashSplat-style label lifting (per-pixel splat contributions instead of centres).
- Pléiades Neo for training: `pipeline.acquisitions` / `pipeline.block_inputs` read WorldView-3 only (the
  pre-purchase metadata check already supports it).
- True-neighbour experiment: waits for a qualified lidar reference (USGS 3DEP failed its checks).
- NYC empty-holdout change in `git stash`; the batch runner's shared DSM file.
