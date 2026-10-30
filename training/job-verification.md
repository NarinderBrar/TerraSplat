# One-block job runner verification — 2026-10-30

The runner is `pipeline/job.py` (`python -m pipeline.job JOB.json`). Job format,
stage contracts, resume/reuse rules and publication behavior are in
[docs/JOBS.md](../docs/JOBS.md). The example jobs are
[central-19v.json](../configs/jobs/central-19v.json) and
[west-19v-job.json](../configs/jobs/west-19v-job.json).

## Recipe equivalence

The exact central plan is saved in
[job-central-19v-dry-run.txt](job-central-19v-dry-run.txt), generated with:

```bash
export PYTHONPATH=
.venv-sfm/bin/python -m pipeline.job configs/jobs/central-19v.json --dry-run
```

It uses the fixed recipe branch of `training/result-2015-19v.json:commands`.
The historical command list contains both basic and recipe experiments; one
job runs one training profile. `test_recipe_equivalence` compares all common
module/argument sequences, including loader check and GPU preflight. Required
additions are offline scan/verification, explicit cache/export/profile paths,
CPU preflight and the remaining delivery stages from Stage B. Publication adds
`--recover-identical` to resume a verified external publication after a crash.
The exact plan marks every GPU stage; the dry run creates no job outputs.

## Tests

```bash
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.venv-sfm/bin/python -m pytest \
  tests/python/test_job.py tests/sfm/test_publish.py \
  tests/python/test_phase2.py tests/sfm/test_reconstruct.py \
  tests/sfm/test_skyfall_inputs.py tests/sfm/test_export.py -q
```

**89 passed, 1 skipped**. The skip is an existing CPU tensor test in
`test_export.py`, because `.venv-sfm` has no PyTorch. A separate CPU-only attempt
in Stage 1 could not run because that environment has no pytest:

```bash
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
source scripts/stage1-env.sh
CUDA_VISIBLE_DEVICES= .venv-stage1/bin/python -m pytest tests/sfm/test_export.py::test_mask_gaussians_preserves_order_and_camera_appearance -q
# .venv-stage1/bin/python: No module named pytest
```

No dependencies were installed. Runner tests execute tiny real Python commands,
including a child that kills its runner after writing partial output. Coverage
includes full run/reuse, file/input invalidation, downstream staleness, CLI
boundaries, failures/logs, output preservation, unsafe cleanup and foreign-directory collision refusal,
interruption at directory rename, missing committed aliases, disk reserve,
optional-truth evidence, and publication activation/isolation. Publisher tests
also recover an identical release after catalogue interruption, reject changed
config/quality and preserve abandoned staging directories.

`git diff --check` passes.

## Real west CPU verification

CPU command (from the repository root):

```bash
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
source scripts/stage1-env.sh
.venv-sfm/bin/python -m pipeline.job configs/jobs/west-19v-job.json --no-gpu
```

All generated stage outputs and logs are under `artifacts/jobs/west-19v-job/`.
The raw imagery and metadata caches are offline inputs. The sourced environment
sets `OMP_NUM_THREADS=4`; COLMAP's existing matching thread count remains 3,
OpenBLAS remains 2, and its existing 900-second command timeout is unchanged.

An initial run without the sourced environment timed out in matching after
900.086 seconds (feature extraction 96.622 seconds). Its selected acquisitions,
PNG hashes, camera hashes and feature counts exactly matched the existing west
preparation. The failure report/logs were preserved under
`verification/exploratory-sfm/`, alongside the outer CLI log. An implementation
review interruption subsequently left reconstruction `running`, with its owned
partial directory recorded and the lock released; that snapshot and logs are
in `verification/review-interrupted-*`.

The full verification stopped at SfM. A separate bounded real check proves crop
interruption/resume and reuse through calibration without retrying SfM. Its
script, logs, state snapshots and result JSON live in
`artifacts/jobs/west-19v-job/verification/`.

**Full CPU verification is incomplete.** The frozen implementation stopped at
`reconstruct`: feature extraction passed in 160.63 seconds, then CPU exhaustive
matching hit the unchanged 900-second timeout (900.07 seconds measured). The
runner returned nonzero, recorded reconstruction as failed with its log path,
and left every downstream stage pending. No further SfM retry was run.

| Main job stage | Outcome | Seconds |
| --- | --- | ---: |
| select | done | 0.54 |
| sources | done | 21.22 |
| block_inputs | done | 73.66 |
| satellite_sfm | done | 0.73 |
| reconstruct | failed | 1072.62 |
| skyfall_inputs | not run | — |
| input_check | not run | — |
| cpu_preflight | not run | — |

Stage times include command/copy/output hashing time; input hashing precedes the
stage clock. Disk free space stayed above 20 GiB; the final measurement was
**59.21 GiB** on the artifacts volume.

The exact stopped stage command, from the repository root, was:

```bash
.venv-sfm/bin/python -m pipeline.reconstruct \
  --inputs /media/narinder/Data1/Projects-2/Geo-Spatial-Projects/artifacts/jobs/west-19v-job/attempts/block_inputs-b14ed581875b/inputs \
  --calibration /media/narinder/Data1/Projects-2/Geo-Spatial-Projects/artifacts/jobs/west-19v-job/attempts/satellite_sfm-3156df1c6211/calibration \
  --approved configs/jax068-west-19v.json \
  --output /media/narinder/Data1/Projects-2/Geo-Spatial-Projects/artifacts/jobs/west-19v-job/attempts/reconstruct-0128b543f1dd.partial/sfm
```

Its failing subprocess was:

```bash
/home/narinder/Documents/OrbitWorks/TerraSplat/.toolchains/sfm-cpu/bin/colmap exhaustive_matcher \
  --database_path /media/narinder/Data1/Projects-2/Geo-Spatial-Projects/artifacts/jobs/west-19v-job/attempts/reconstruct-0128b543f1dd.partial/sfm/database.db \
  --SiftMatching.use_gpu 0 \
  --SiftMatching.num_threads 3 \
  --SiftMatching.guided_matching 1 \
  --SiftMatching.max_error 3 \
  --SiftMatching.max_num_matches 16384
```

The failure report and inner logs were copied to
`verification/failed-full-reconstruction-report.json` and
`verification/failed-full-sfm-logs/`. The failed state snapshot is
`verification/failed-full-state.json`; the main `job-state.json` still records
that failure. The outer log is `/media/narinder/Data1/Projects-2/Geo-Spatial-Projects/artifacts/jobs/west-19v-job/reconstruct-0128b543f1dd.log`.

A bounded main-job rerun stops before SfM and verifies **all four completed
stages reuse**, in **28.53 seconds** wall time, with the
entire failed main state unchanged:

```bash
.venv-sfm/bin/python -m pipeline.job configs/jobs/west-19v-job.json --no-gpu --until satellite_sfm
```

```text
REUSE select
REUSE sources
REUSE block_inputs
REUSE satellite_sfm
```

For an independent real interruption check, a fresh job under
`verification/cpu-interruption-check/` uses the same inputs and stops after
`satellite_sfm`. Its config is `verification/cpu-interruption-job.json`.
The controller paused the runner during cropping, killed its crop subprocess
group and killed the runner. The exit code was -9; state remained `running`,
with exactly one owned `.partial` directory recorded. Resuming the identical
bounded command reused selection/sources, deleted that partial, completed
cropping (50.20 s) and calibration (0.37 s), and took
**76.97 seconds** wall time. A further identical rerun
reused all four stages. Resumed RGB hashes and the complete calibration report
match the main attempt exactly. No completed output was removed.

Evidence: `verification/bounded-results.json`, `bounded-interrupted-state.json`,
`bounded-resumed-state.json`, `bounded-main-reuse.log`, `bounded-resumed.log`,
`bounded-resume-reuse.log`, and `verify-west-bounded.py`.

The existing successful west result is **23,157 sparse points / 0.421855 px
median** (79,385 observations). The new attempt's selection hash, prepared PNG
hashes, camera hashes and feature counts all match that earlier result exactly.
**There is no new sparse model**, so point-count/reprojection equivalence,
select-through-CPU-preflight completion, reuse of all eight CPU stages and
full-SfM interruption/resume equivalence remain unverified. The runner's fake
full-flow tests cover those orchestration behaviors; they do not replace this
missing real evidence.


## Scope and files

Implemented: `pipeline/job.py`; two job JSONs; job tests; job/release docs;
this report and central dry-run listing; Phase 2 exit-gate updates in `tasks.md`.
Existing-module changes are limited to `pipeline/publish.py`: explicitly labelled
missing lidar metrics and opt-in identical-publication recovery. Its additional
tests are in `tests/sfm/test_publish.py`.

No GPU preflight, training, evaluation rendering, export, depth rendering, DSM
validation, tiling, encoding or real publication was executed. The full recipe
is tested through command equivalence and fake stage execution; GPU integration
remains unverified by this task. No imagery/lidar downloads, live release or
catalogue changes, existing experiment writes, or `apps/web/` edits were made by
this task. Completed job attempts are preserved. No commit, push, or Git index
mutation command was run by this task.
