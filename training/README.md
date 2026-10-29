# Basic Skyfall Stage 1 environment

This setup creates an isolated Python environment, compiler toolchain and Skyfall
checkout inside TerraSplat. It does not modify the shared `../Ref/Skyfall-GS`
checkout or the system driver. The generated environment and model artifacts are
ignored by Git.

The [2015 pilot handoff](PILOT.md) now has deskewed images, source-derived masks,
explicit normalization and a verified three-train/one-holdout loader contract.
See [its verified result](pilot-verified.json). Full training on that block is next.

## Setup

The host needs `uv`, Git, GCC, an NVIDIA driver and Python 3.12+ for the toolchain
installer (the training environment itself uses Python 3.10). Downloads need
several GB of disk space. Building is bounded to two compiler jobs. On this
project's RTX 3060, the target architecture is explicitly SM 8.6.

```bash
bash scripts/setup-stage1.sh
source scripts/stage1-env.sh
python scripts/check-stage1.py
```

GPU commands must run with host GPU access; the restricted sandbox cannot see
the NVIDIA device. The check exercises nearest neighbours, fused SSIM gradients,
rasterizer gradients, and `train.py --help`. Its result is written to
`artifacts/stage1/environment.json`.

The toolchain installer downloads only NVCC, runtime development files and CCCL
from NVIDIA's [CUDA 12.8.1 redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json).
Archive SHA-256 values are pinned in the installer and checked before extraction.
It never invokes a system installer or modifies the GPU driver.

The Python environment uses Python 3.10, PyTorch 2.7.1 and torchvision 0.22.1
from the [official CUDA 12.8 wheel index](https://pytorch.org/get-started/previous-versions/#v271),
plus `training/requirements-stage1.txt`. The installed package snapshot is saved
as `training/stage1-installed.txt`; the three local extension sources are pinned
through the Skyfall commit and submodule revisions rather than portable wheel URLs.

## Source adaptations

Skyfall is pinned to `73194842a404910c5bbff006359eedfc56a98d60`. The preparation
script clones tracked source, not the reference project's datasets. It checks
the commit and refuses to overwrite unexpected changes to the patched training files.

Three patches are applied only to `.external/skyfall-gs`:

- `adapters/skyfall/stage1-only.patch` removes unconditional MoGe/FlowEdit imports
  and the global MoGe model load, rejects IDU and nonzero pseudo-depth loss, and
  resizes masks and depth to match downsampled images, and reports peak allocated
  GPU memory. Reconstruction, rasterization and gradients
  still use Skyfall's Stage 1 implementation.
- `adapters/skyfall/rasterizer-cstdint.patch` adds the rasterizer's missing
  standard integer header for the compiler used here.
- `adapters/skyfall/prepared-enu.patch` verifies prepared payload hashes,
  requires `--eval` to preserve the acquisition holdout, and consumes externally
  normalized cameras/points without a second normalization or random-point
  fallback. See [the handoff contract](PILOT.md).

This is a **basic Stage 1 configuration**, not the full published Stage 1 recipe:
monocular pseudo-depth regularization is disabled. Diffusion models and their
weights are not downloaded. Preserve Skyfall and extension license files; the
original code retains its upstream research-use terms.

## Bounded training diagnostic

```bash
source scripts/stage1-env.sh
python scripts/smoke-stage1.py
```

The command copies the selected JAX_068 camera, image, mask and sparse-point
files into a new timestamped artifact directory. It never trains against the
shared source directory: Skyfall rewrites `points3D.ply` during scene loading.
It hashes the copied inputs and records the source commit, adapter patch hash,
exact command, elapsed time and output checkpoint/PLY hashes in `report.json`.

The saved checkpoint iteration and finite PLY attributes are checked before the
diagnostic passes.

The diagnostic runs 20 iterations, resolution factor 4, SH degree 1, appearance
modeling enabled, CPU image storage, and no densification, depth loss or
pseudo-depth loss. It has a five-minute subprocess limit. It expects both the
iteration-20 checkpoint and raw PLY; failures preserve the log and report.
The GUI listener binds to localhost on an ephemeral port.

These outputs are **unregistered diagnostics**, not a trained production scene.
The published prepared JAX_068 inputs have no captured ENU origin. A full Phase 2
run still requires original imagery/RPC preparation, captured normalization and
geographic evidence, measured training, fused PLY export and independent accuracy
validation. Do not publish the smoke-test PLY as georeferenced output.

## Executed result

The initial bounded training diagnostic completed 20 iterations with 69,027
Gaussians and saved an iteration-20 checkpoint and raw PLY. PyTorch reported
267,636,224 bytes (about 255 MiB) peak allocated GPU tensor memory, excluding
CUDA context/driver overhead. This is not a projection of memory use during
full-resolution training or densification. All three CUDA kernel checks passed.
