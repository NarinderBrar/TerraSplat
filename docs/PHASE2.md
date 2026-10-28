# Phase 2: one-block reconstruction

Status: started. Prepared-input preflight, explicit coordinate-transform
preservation, isolated Stage 1 training, and raw RPC-aware crop tooling are
implemented. See [imagery preparation](../imagery/README.md) for I/O results. The raw-imagery-to-tiles pipeline has not run.
Phase 1's existing-scene conversions remain incomplete; Phase 0 is committed and
pushed. The user explicitly confirmed proceeding directly to Phase 2.

## Run the implemented checks

Python 3.10+ is sufficient for the preflight command; it uses the standard library.

```bash
python3 -m pipeline.phase2 configs/phase2-jax068.json --output artifacts/phase2/preflight.json
```

Exit code 0 means the prepared-input preflight passed. Exit code 2 means a
manifest/input problem or blocked prerequisite. A passed preflight does not
establish reconstruction quality, raw image suitability, GPU memory fitness,
DSM accuracy, or publication readiness. The command never starts training.

Paths in the manifest resolve relative to the manifest file, so the example
uses the sibling `Ref/Skyfall-GS` checkout. Override `python` with an absolute
path to the selected training environment's interpreter when needed.

The preflight checks:

- Nonempty training and held-out camera files, finite affine camera matrices,
  positive focal lengths, image presence, and finite COLMAP sparse point records.
- SHA-256 hashes of camera files, sparse points and image files; duplicate images
  within a split and overlapping image content between splits are rejected.
- The pinned Skyfall commit and changes to tracked training source files.
- A captured ENU anchor and normalization record bound to the camera/point hashes.
- CUDA availability and compiled PyTorch extensions in the selected interpreter,
  plus initialized submodules at their pinned commits.

It does not decode images or prove that different file encodings depict different
acquisitions. Raw-input acquisition IDs and whole-view holdouts still need to be
preserved by preprocessing.

## Capturing the coordinate contract

The manifest's `georeference` field points to a JSON file produced while preparing
and normalizing the actual block. It must contain:

```json
{
  "anchor": {
    "longitude": 0,
    "latitude": 0,
    "ellipsoidal_height_m": 0
  },
  "normalization": {
    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "translation": [0, 0, 0],
    "scale": 1,
    "z_shift": 0
  },
  "provenance": "Actual preprocessing run and ENU observer metadata reference",
  "input_sha256": {
    "transforms_train.json": "SHA-256 of actual camera file",
    "points3D.txt": "SHA-256 of actual sparse points"
  }
}
```

These values illustrate the shape only. They are not a valid Jacksonville anchor.
Use the actual ENU observer file and runtime normalization values; leave the
manifest field null when evidence is missing. The fixture's invented height and
TerraScope's flat visual datum must not fill this gap.

`pipeline/georeference.py` preserves the inspected Skyfall equation:

```text
model = scale * (R * enu - T) - [0, 0, z_shift]
```

It calculates its inverse and WGS84 ENU-to-ECEF placement, rejecting nonfinite
values, nonpositive scale, nonorthonormal rotations and reflected frames. All
reported matrices are **row-major**. The source must already be metric ENU;
this is not a UTM-to-ECEF approximation. The model remains Z-up. glTF axis,
covariance and SH conversions belong to the exporter and are not applied here.
Record the actual scale and z shift after Skyfall computes its point percentiles;
do not infer them from the scene name or dimensions.

## Initial local result

The supplied JAX_068 prepared-input audit found 17 training images, 2 held-out
images, and 69,027 sparse points. No image-file overlap was detected. It is an
existing full prepared scene, not a newly cropped 250 m CORE3D block.

The command correctly reports blocked because:

1. Captured ENU/normalization evidence is missing.
2. The audit manifest still selects the system `python3`, which has no training
   stack. A separate `.venv-stage1` environment now passes CUDA kernel checks and
   a bounded training diagnostic; see [Stage 1 setup](../training/README.md).
3. Skyfall's submodules are uninitialized. Its `train.py` imports FlowEdit and
   MoGe even when the generative stage is disabled.

A host check outside the sandbox confirms the RTX 3060 Laptop GPU has 6,144 MiB
VRAM (driver 595.99.02). The earlier `nvidia-smi` failure was sandbox access, not
a broken host driver. CUDA checks and eventual training must run with host GPU
access. The isolated Stage 1 environment is now available. The original audit manifest
still describes the shared unmodified checkout and system interpreter.

No reference checkout, images, CUDA installation, or system driver is modified.
Local reports are ignored by Git.

## Remaining Phase 2 work

Confirm the proposed raw CORE3D block, select suitable overlapping acquisitions
and matched MSI inputs, and implement SatelliteSfM preprocessing; retain the ENU observer and normalization at runtime;
run measured Stage 1 training within the memory budget; fuse and export the real
scene; compare registration and DSM heights against independent lidar evidence.
A working GPU environment and the actual input/validation datasets are required
before the end-to-end acceptance gate can pass.

## Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/python -q
```

Tests use isolated files and a local Git fixture. They exercise positive prepared
input validation, holdout leakage, stale normalization hashes, missing georeference,
modified backend source, malformed cameras, empty point clouds, transform inversion,
WGS84 placement and invalid/reflected normalizations. They do not mock GPU training.
