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

## 1. Finish the grid (main track)

1. Check all 16 trainings completed (B3 fixed as above) (no memory-cap stops, exports and depth renders present).
2. Score A2, A3, B2, B3 and their seams with `pipeline.grid_seams` against the pre-registered
   `configs/grid-1km-decision.json`; report pass or fail as measured.
3. Publish all 16 to the review catalogue `artifacts/releases-grid-review` and check them in the browser together
   (streaming, seams, full-detail switching, performance).
4. Publish live; then retire the central `jax068-2015` block (needs a dataset-deactivation step; inactive releases
   can already be withdrawn with `pipeline.publish --withdraw-release`).

## 2. Semantics and segment mode: authors' scenes only

Segment mode (docs/SEGMENTS.md) is built and verified; layers will be made for the 12 Skyfall-GS authors' scenes only.
Their data is RGB, so v2 (NDVI) does not apply.

1. **v4 (pre-registered, `configs/semantics-v4.json`)**: RemoteCLIP features on the authors' JAX_068 model, after
   the grid run (about 15 min GPU). The weights (`LangSplat/ckpts/remoteclip/RemoteCLIP-ViT-L-14.pt`) are a PyTorch
   pickle; its pickle globals were checked (tensor data only), but loading it is the user's decision.

   ```bash
   L=/media/narinder/Data1/Projects-2/Geo-Spatial-Projects/Tools-For-References-Map/LangSplat
   env -u PYTHONPATH HF_HUB_OFFLINE=1 ../.venv-langsplat/bin/python scripts/clip-prototypes.py \
     configs/semantics-v4.json artifacts/semantics/v3-inputs/prototypes-v4.npz $L
   (cd $L && env -u PYTHONPATH HF_HUB_OFFLINE=1 ~/Documents/OrbitWorks/.venv-langsplat/bin/python -u preprocess.py \
     --dataset_path data/skyfall_jax068_authors_grid --resolution -1 --sam_type vit_b \
     --sam_ckpt_path ckpts/sam_vit_b_01ec64.pth --points_per_side 32 --points_per_batch 4 --crop_n_layers 1 \
     --clip_batch_size 4 --offload_sam --memory_safe --max_masks_per_level 512 \
     --clip_model ViT-L-14 --clip_pretrained ckpts/remoteclip/RemoteCLIP-ViT-L-14.pt --clip_dims 768)
   .venv-imagery/bin/python -m pipeline.semantics_clip --config configs/semantics-v4.json \
     --export artifacts/skyfall/JAX_068/final-placed --points artifacts/skyfall/JAX_068/final-depth/depth-points.npz \
     --truth artifacts/reference/dfc2019-jax068 --scene $L/data/skyfall_jax068_authors_grid \
     --features $L/data/skyfall_jax068_authors_grid/language_features \
     --prototypes artifacts/semantics/v3-inputs/prototypes-v4.npz \
     --registration artifacts/skyfall/JAX_068/lidar-authors/dsm-validation.json --output artifacts/semantics/v4-jax068
   ```
2. **v5 if v4 falls short**: a land-cover segmentation model for sub-metre imagery (e.g. trained on OpenEarthMap; not
   on DFC2019 or US3D Jacksonville) plus height, fused with the same verified multi-view projection. Tune on JAX_068.
3. **Blind test**: fix the method, then score once on the other seven DFC2019 scenes (JAX_004, 164, 168, 175, 214,
   260, 264). Their lidar DSM/CLS go in `artifacts/reference/dfc2019-jax/` (being downloaded); each scene first
   needs its placement offset measured like JAX_068. Do not inspect their labels before the method is fixed.
4. Build segment layers (`pipeline.segments`) for all 12 authors' scenes and publish `skyfall-*-final-r3` (same tiles
   plus layer). The four NYC scenes get layers without an accuracy number.
5. Withdraw the recoloured `jax068-2015-19v-classes-r1` once r3 layers exist (decision pending).

## 3. Parked

- FlashSplat-style label lifting (per-pixel splat contributions instead of centres).
- Pléiades Neo for training: `pipeline.acquisitions` / `pipeline.block_inputs` read WorldView-3 only (the
  pre-purchase metadata check already supports it).
- True-neighbour experiment: waits for a qualified lidar reference (USGS 3DEP failed its checks).
- NYC empty-holdout change in `git stash`; the batch runner's shared DSM file.
