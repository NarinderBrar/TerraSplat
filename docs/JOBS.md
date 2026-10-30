# One-block jobs

Run from the repository root with Python 3.12 (the `.venv-sfm` interpreter is
sufficient for the runner). No new Python dependencies are needed.

```bash
export PYTHONPATH=
.venv-sfm/bin/python -m pipeline.job configs/jobs/central-19v.json --dry-run
.venv-sfm/bin/python -m pipeline.job configs/jobs/west-19v-job.json --no-gpu
# Repeat exactly the same command to verify outputs and reuse completed stages.
.venv-sfm/bin/python -m pipeline.job configs/jobs/west-19v-job.json --no-gpu
```

A job is a JSON object with `version: 1` and these entries. Relative input paths
are relative to the repository root; run the CLI there. Copy a job file and choose
a fresh output root before a new experiment.

| Entry | Meaning |
| --- | --- |
| `output_root` | Dedicated job workspace, normally `artifacts/jobs/<name>` |
| `catalogue` | Existing imagery catalogue JSON |
| `block_config` | Block geometry JSON used for an offline acquisition scan |
| `approved_selection` | Fixed train/holdout acquisition IDs and selection policy |
| `source_cache` | Existing verified PAN/MSI cache (default `artifacts/imagery/raw`) |
| `training_profile` | Existing Stage 1 profile JSON |
| `tile_profile` | Existing tiling profile JSON |
| `ownership_grid`, `cell` | Optional shared ENU grid and owned cell; supply both |
| `lidar_truth` | Optional existing truth directory; never downloaded |
| `release` | Publisher config object with dataset/release identity, evidence and quality pointers |

The two example jobs use the approved central/west 19-view selections and the
fixed recipe profile. May 2 remains the independent holdout. No acquisition,
training setting or decision is selected from holdout or lidar scores.

`release.evidence` and `release.quality_sources` support stage placeholders such
as `{dsm_validation}/validation/dsm-validation.json` and
`{encoding}/tiles/tile-plan.json`. The runner fills `tiles_dir`, `stage1_run_dir`,
`ground_points`, capture dates, measured tile count and the approved holdout's
PSNR pointer from the new outputs. It never copies an earlier run's measured
scores into a job release. Without `lidar_truth`, DSM validation is omitted,
lidar quality pointers are removed, and lidar scores become null with the
explicit label `No independent lidar truth supplied`. The publisher accepts
that labelled absence and offers opt-in recovery of an identical interrupted
publication. These are the two required changes to an existing pipeline module;
other quality checks continue to apply.

The stage order is:

```text
select (offline scan + approved select)
 → sources (offline verification)
 → block_inputs → satellite_sfm → reconstruct → skyfall_inputs
 → input_check → cpu_preflight
 → gpu_preflight [GPU] → training [GPU] → evaluation [GPU]
 → export [GPU] → render_depth [GPU]
 → dsm_validation (if truth supplied) → tiling → encoding → publish
```

Each stage invokes the existing module or script in `.venv-imagery`, `.venv-sfm`,
`.venv-stage1`, or Node. Stage 1 processes source `scripts/stage1-env.sh` and all
children have empty `PYTHONPATH`. The full command argv, environment name,
GPU flag, start/end UTC times, elapsed seconds, log path, input hash, file-by-file
output hashes and output summary hash live in `job-state.json`. State updates
use fsync and atomic replacement. A job lock prevents concurrent writers.

A stage is reused only when it is done and both its input hash and all output
hashes match. Input hashes cover relevant job entries and files, command and
module code, publication preparation code, Stage 1 environment script, and upstream output
summaries. Selection also hashes the offline metadata cache; source verification
hashes the raw cache. These conservative checks can take time even when no stage
runs. The first invalid stage makes every subsequent stage pending/stale. Old
completed attempts remain on disk, including after `--from`.

Each attempt writes into `attempts/<stage>-<unique-id>.partial/`; successful
attempts are renamed into the corresponding directory without `.partial`.
Existing modules embed absolute output paths in reports, including manifests
whose hashes other reports record. The runner therefore retains a `.partial`
**symlink** to each committed attempt, preserving those paths without rewriting
hashed evidence. These committed aliases are never eligible for cleanup. Stages
that write inside their inputs (loader check, evaluation, encoding) receive
private copies. No completed upstream directory is modified.

On resume, unfinished stages are rerun. Cleanup is limited to direct attempt
containers ending in `.partial` recorded as owned after successful creation and before stage processing;
symlinks, completed attempts, unrecorded paths and paths outside the output root
are refused. Recovery also handles interruption after the success rename and
before recording completion. Do not manually edit state or remove committed
aliases. Preserve logs and inspect failures before retrying. Terminating the
runner normally terminates its active subprocess group; if forcibly killing it,
terminate that group too before resuming.

```bash
# Stop after a named stage (inclusive).
.venv-sfm/bin/python -m pipeline.job configs/jobs/west-19v-job.json --until reconstruct
# Force a new attempt for this stage and every later stage, retaining old outputs.
.venv-sfm/bin/python -m pipeline.job configs/jobs/west-19v-job.json --from block_inputs --no-gpu
# Print exact commands and run/reuse decisions; create no files.
.venv-sfm/bin/python -m pipeline.job configs/jobs/central-19v.json --dry-run
```

`--no-gpu` stops before the first GPU stage, including when that stage was
previously completed. Every real stage checks for at least 20 GiB free before
starting. A failure stops execution, records the failed stage and log, returns
nonzero and leaves downstream stages pending. GPU stages are implemented but
were not run during this task.

Publication defaults to an isolated catalogue inside the successful publish
attempt, exposed as `<output_root>/releases`. `--publish-to DIR` explicitly
chooses an external catalogue; only this option may write outside the job root.
The live `artifacts/releases` is never selected by default. No release is
activated without `--activate`. The publisher's immutable-ID rule still applies
when explicitly targeting an existing catalogue: a changed publication needs a
new release ID. The job invokes `pipeline.publish --recover-identical` so an
interruption after release rename can recover the catalogue after verifying
config and delivery hashes. Interrupted external staging directories are
preserved, and retries use unique staging paths. External release contents are
also hash-checked for reuse.

```bash
# This is a publication example, not part of CPU-only verification.
.venv-sfm/bin/python -m pipeline.job configs/jobs/central-19v.json --publish-to artifacts/releases-job-review
# Adding --activate would activate the new release in that chosen catalogue.
```

The central dry run follows the **recipe branch** of
`training/result-2015-19v.json:commands`. That historical list contains both basic
and recipe experiments; a single job trains one profile. Documented additions
are the offline scan, offline source/cache arguments, separate CPU preflight,
explicit export/depth paths for immutable stage isolation, explicit tile profile,
and the remaining validation/delivery stages from the Stage B scripts. CPU
verification evidence and exact dry-run output are recorded separately in
`training/job-verification.md`.

Tests:

```bash
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.venv-sfm/bin/python -m pytest tests/python/test_job.py tests/sfm/test_publish.py -q
```

Local verification: 89 tests pass, with one optional PyTorch skip. The fresh west
run reached CPU SfM matching and timed out at the existing 900-second limit.
Completed-stage reuse and bounded real crop interruption/resume passed; the full
CPU exit gate remains open. See [the measured report](../training/job-verification.md).
