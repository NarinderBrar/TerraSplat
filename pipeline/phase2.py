"""Audit a prepared one-block reconstruction input without starting GPU training."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from pipeline.georeference import placement


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=30)
        return {"ok": result.returncode == 0, "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "output": str(error)}


def preflight(manifest_path, check_environment=True):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported manifest version")

    def resolve(value):
        return (manifest_path.parent / value).resolve()

    scene, skyfall = resolve(manifest["prepared_scene"]), resolve(manifest["skyfall"])
    failures, evidence, splits = [], {}, {}
    for name in ("transforms_train.json", "transforms_test.json", "points3D.txt"):
        path = scene / name
        if not path.is_file():
            failures.append(f"Missing prepared input: {name}")
            continue
        evidence[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
        if not name.endswith(".json"):
            count = 0
            try:
                with path.open() as stream:
                    for line in stream:
                        if not line.strip() or line.lstrip().startswith("#"):
                            continue
                        fields = line.split()
                        if len(fields) < 8 or not all(math.isfinite(float(v)) for v in fields[1:4]):
                            raise ValueError("expected finite COLMAP XYZ point records")
                        count += 1
                if count == 0:
                    raise ValueError("no sparse points")
                evidence[name]["point_count"] = count
            except (UnicodeError, ValueError) as error:
                failures.append(f"Invalid sparse point input: {error}")
            continue
        try:
            camera = json.loads(path.read_text())
            frames = camera["frames"]
            if not frames:
                failures.append(f"Empty camera split: {name}")
            paths = []
            for frame in frames:
                matrix_key = "transform_matrix_rotated" if "R" in camera else "transform_matrix"
                matrix = frame[matrix_key]
                if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                    raise ValueError(f"{matrix_key} must be 4x4")
                if not all(math.isfinite(v) for row in matrix for v in row) or matrix[3] != [0, 0, 0, 1]:
                    raise ValueError("camera transform must be finite and affine")
                if any(not math.isfinite(frame[key]) or frame[key] <= 0 for key in ("fl_x", "fl_y")):
                    raise ValueError("camera focal lengths must be finite and positive")
                image = (scene / frame["file_path"]).resolve()
                if not image.is_relative_to(scene):
                    failures.append(f"Image escapes prepared scene: {frame['file_path']}")
                elif not image.is_file():
                    failures.append(f"Missing image: {frame['file_path']}")
                else:
                    # Hash pixels/files too: renaming the same acquisition must not bypass holdouts.
                    paths.append(digest(image))
            if len(paths) != len(set(paths)):
                failures.append(f"Duplicate image content in {name}")
            splits[name] = set(paths)
            evidence[name]["image_count"] = len(frames)
            evidence[name]["image_sha256"] = sorted(paths)
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"Invalid camera split {name}: {error}")
    if splits.get("transforms_train.json", set()) & splits.get("transforms_test.json", set()):
        failures.append("Training and held-out images overlap by content")
    georeference = None
    if not manifest.get("georeference"):
        failures.append("Missing captured ENU anchor and Skyfall normalization record")
    else:
        path = resolve(manifest["georeference"])
        try:
            record = json.loads(path.read_text())
            if not record.get("provenance"):
                raise ValueError("Georeference record needs provenance")
            for name in ("transforms_train.json", "points3D.txt"):
                if record["input_sha256"][name] != evidence.get(name, {}).get("sha256"):
                    raise ValueError(f"Georeference record does not match {name}")
            georeference = {
                "assumed_anchor": record.get("assumed_anchor", False),
                "anchor": record["anchor"],
                "normalization": record["normalization"],
                "provenance": record["provenance"],
                **placement(record["anchor"], record["normalization"]),
            }
            evidence["georeference"] = {"sha256": digest(path)}
        except (OSError, KeyError, TypeError, ValueError) as error:
            failures.append(f"Invalid georeference: {error}")
    commit = run(["git", "rev-parse", "HEAD"], cwd=skyfall)
    if not commit["ok"] or commit["output"] != manifest["expected_skyfall_commit"]:
        failures.append("Skyfall checkout does not match the pinned commit")
    dirty = run(
        ["git", "diff", "--name-only", "HEAD", "--", "train.py", "scene", "arguments", "gaussian_renderer", "utils"],
        cwd=skyfall,
    )
    profile = manifest.get("skyfall_patch_profile")
    if profile in ("basic-stage1-registered", "stage1-pseudo-depth"):
        approved_paths = {"train.py", "utils/camera_utils.py", "scene/dataset_readers.py"}
        if not dirty["ok"] or set(dirty["output"].splitlines()) != approved_paths:
            failures.append("Skyfall source differs from the approved Stage 1 adapter paths")
        patch_root = Path(__file__).resolve().parents[1] / "adapters/skyfall"
        stage_patch = "stage1-pseudo-depth.patch" if profile == "stage1-pseudo-depth" else "stage1-only.patch"
        for name, paths in (
            (stage_patch, ["train.py", "utils/camera_utils.py"]),
            ("prepared-enu.patch", ["scene/dataset_readers.py"]),
        ):
            result = run(["git", "diff", "HEAD", "--", *paths], cwd=skyfall)
            patch = patch_root / name
            if not result["ok"] or result["output"] != patch.read_text().strip():
                failures.append(f"Skyfall source does not match recorded adapter: {name}")
            evidence[name] = {"sha256": digest(patch)}
    elif profile is not None:
        failures.append("Unsupported Skyfall patch profile")
    elif not dirty["ok"] or dirty["output"]:
        failures.append("Skyfall training source differs from the pinned revision")
    environment = None
    if check_environment:
        # Import in the selected training interpreter, not the CLI's environment.
        probe = (
            "import torch, diff_gauss, simple_knn._C, fused_ssim; assert "
            'torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, '
            "torch.cuda.get_device_name(0))"
        )
        environment = run([manifest.get("python", sys.executable), "-c", probe], cwd=skyfall)
        if not environment["ok"]:
            failures.append("Training interpreter lacks a working CUDA/PyTorch extension stack")
        module_args = ["git", "submodule", "status"]
        if profile in ("basic-stage1-registered", "stage1-pseudo-depth"):
            module_args += [
                "--",
                "submodules/diff-gaussian-rasterization-depth",
                "submodules/simple-knn",
                "submodules/fused-ssim",
            ]
        modules = run(module_args, cwd=skyfall)
        if not modules["ok"] or any(line.startswith(("-", "+", "U")) for line in modules["output"].splitlines()):
            failures.append("Required Skyfall submodules are absent or do not match pinned revisions")
    return {
        "version": 1,
        "id": manifest["id"],
        "status": "blocked" if failures else "passed",
        "scope": (
            "prepared-input and placement audit; does not execute training or validate independent geographic accuracy"
        ),
        "manifest_sha256": digest(manifest_path),
        "inputs": evidence,
        "skyfall_commit": commit,
        "environment_checked": check_environment,
        "environment": environment,
        "georeference": georeference,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = preflight(args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Invalid manifest: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}: {args.output}")
    for failure in report["failures"]:
        print(f"- {failure}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
