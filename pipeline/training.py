"""Measured, resumable Stage 1 training of an already verified pilot scene."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from pipeline.phase2 import digest, preflight

ROOT = Path(__file__).resolve().parents[1]


def validate_profile(profile):
    if profile.get("version") != 1:
        raise ValueError("Unsupported training profile")
    for key in (
        "iterations",
        "warmup_iterations",
        "resolution",
        "sh_degree",
        "max_gaussians",
        "max_allocated_bytes",
        "timeout_seconds",
        "densification_interval",
    ):
        if type(profile[key]) is not int or profile[key] <= 0:
            raise ValueError(f"Positive integer required: {key}")
    if not 0 < profile["warmup_iterations"] < profile["iterations"]:
        raise ValueError("Warmup must precede final iteration")
    if (
        not 0
        <= profile["densify_from_iter"]
        < profile["densify_until_iter"]
        < profile["iterations"]
    ):
        raise ValueError("Densification must stop before final iteration")
    if profile["max_allocated_bytes"] > 5 * 1024**3:
        raise ValueError("Training profile must leave headroom on the 6 GB GPU")


def training_command(python, data, model, metrics, profile, target, checkpoint=None):
    validate_profile(profile)
    args = [
        str(python),
        "train.py",
        "-s",
        str(data),
        "-m",
        str(model),
        "--eval",
        "--iterations",
        str(target),
        "--resolution",
        str(profile["resolution"]),
        "--sh_degree",
        str(profile["sh_degree"]),
        "--data_device",
        "cpu",
        "--port",
        "0",
        "--kernel_size",
        "0.1",
        "--lambda_depth",
        "0",
        "--lambda_opacity",
        "0",
        "--lambda_pseudo_depth",
        "0",
        "--position_lr_max_steps",
        str(profile["iterations"]),
        "--densify_from_iter",
        str(profile["densify_from_iter"]),
        "--densify_until_iter",
        str(profile["densify_until_iter"]),
        "--densification_interval",
        str(profile["densification_interval"]),
        "--test_iterations",
        "-1",
        "--terrasplat_metrics",
        str(metrics),
        "--terrasplat_max_gaussians",
        str(profile["max_gaussians"]),
        "--terrasplat_max_allocated_bytes",
        str(profile["max_allocated_bytes"]),
    ]
    milestones = [i for i in (10000, 20000, profile["iterations"]) if i <= target]
    if target not in milestones:
        milestones.append(target)
    for option in ("--checkpoint_iterations", "--save_iterations"):
        args += [option, *map(str, sorted(set(milestones)))]
    if checkpoint is not None:
        args += ["--start_checkpoint", str(checkpoint)]
    if profile.get("detect_anomaly"):
        args += ["--detect_anomaly"]
    if profile.get("convert_sh_python"):
        args += ["--convert_SHs_python"]
    return args


def copy_scene(scene, target):
    contract_path = scene / "terrasplat-scene.json"
    contract = json.loads(contract_path.read_text())
    for relative, expected in contract["input_sha256"].items():
        src = (scene / relative).resolve()
        if not src.is_relative_to(scene) or digest(src) != expected:
            raise ValueError(f"Prepared payload changed: {relative}")
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    shutil.copyfile(contract_path, target / contract_path.name)
    return contract


def validate_outputs(model, iteration):
    import numpy as np
    import torch
    from plyfile import PlyData

    checkpoint = model / f"chkpnt{iteration}.pth"
    auxiliary = model / f"chkpnt{iteration}.pth.training-state.pth"
    ply = model / f"point_cloud/iteration_{iteration}/point_cloud.ply"
    # These artifacts were produced by this invocation in its new run directory.
    params, saved_iteration = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    if saved_iteration != iteration or not len(params[1]):
        raise ValueError("Checkpoint iteration or point count is invalid")

    def finite(value):
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint tensor")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)

    finite(params)
    vertices = PlyData.read(ply)["vertex"]
    if len(vertices) != len(params[1]):
        raise ValueError("Checkpoint and PLY point counts disagree")
    for name in vertices.data.dtype.names:
        if not np.isfinite(vertices[name]).all():
            raise ValueError(f"Nonfinite PLY attribute: {name}")
    return {
        "iteration": iteration,
        "splats": len(vertices),
        "sha256": {
            str(path.relative_to(model)): digest(path)
            for path in (checkpoint, auxiliary, ply)
        },
    }


def train(scene, output, profile_path):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    profile = json.loads(Path(profile_path).read_text())
    validate_profile(profile)
    if output.exists():
        raise ValueError("Use a new training output directory")
    audit = preflight(scene / "phase2-manifest.json", check_environment=True)
    if audit["status"] != "passed":
        raise ValueError(audit["failures"])
    output.mkdir(parents=True)
    data, model = output / "input", output / "model"
    contract = copy_scene(scene, data)
    model.mkdir()
    shutil.copyfile(data / "georeference.json", model / "georeference.json")
    (output / "preflight-report.json").write_text(json.dumps(audit, indent=2) + "\n")
    report = {
        "status": "running",
        "profile": profile,
        "profile_sha256": digest(Path(profile_path)),
        "runner_sha256": digest(Path(__file__)),
        "input_manifest_sha256": digest(data / "terrasplat-scene.json"),
        "train_catalog_ids": contract["train_catalog_ids"],
        "holdout_catalog_ids": contract["holdout_catalog_ids"],
        "input_scene": str(data),
        "model": str(model),
        "phases": [],
        "georeference_sha256": digest(model / "georeference.json"),
        "adapter_sha256": audit["inputs"]["stage1-only.patch"]["sha256"],
        "independent_accuracy_validated": False,
        "holdout_used_for_training_or_selection": False,
    }
    import torch

    report["environment"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    started = time.monotonic()
    checkpoint = None
    metrics = output / "metrics.jsonl"
    try:
        for target in (profile["warmup_iterations"], profile["iterations"]):
            if checkpoint is not None:
                for name, expected in report["phases"][-1]["outputs"]["sha256"].items():
                    if digest(model / name) != expected:
                        raise ValueError(
                            "Locally produced checkpoint changed before resume"
                        )
            command = training_command(
                sys.executable, data, model, metrics, profile, target, checkpoint
            )
            entry = {"target": target, "command": command}
            report["phases"].append(entry)
            (output / "training-report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            timeout = profile["timeout_seconds"] - (time.monotonic() - started)
            if timeout <= 0:
                raise TimeoutError("Training wall-time budget exhausted")
            print(f"Training to iteration {target}: {output}", flush=True)
            with (output / f"training-{target}.log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT / ".external/skyfall-gs",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            entry["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"Training phase failed: {target}; inspect its log")
            entry["outputs"] = validate_outputs(model, target)
            entry["schedule"] = json.loads(
                (model / "training-schedule.json").read_text()
            )
            checkpoint = model / f"chkpnt{target}.pth"
        samples = [json.loads(line) for line in metrics.read_text().splitlines()]
        report.update(
            status="trained",
            final_splats=report["phases"][-1]["outputs"]["splats"],
            peak_allocated_bytes=max(s["peak_allocated_bytes"] for s in samples),
            peak_reserved_bytes=max(s["peak_reserved_bytes"] for s in samples),
            sampled_device_used_bytes=max(
                s["sampled_device_used_bytes"] for s in samples
            ),
            checkpoint_resume_verified=report["phases"][1]["schedule"][
                "start_iteration"
            ]
            == profile["warmup_iterations"],
            metrics_sha256=digest(metrics),
            initial_ply_sha256=digest(
                model / "point_cloud/iteration_0/point_cloud.ply"
            ),
        )
        if (
            report["peak_allocated_bytes"] > profile["max_allocated_bytes"]
            or not report["checkpoint_resume_verified"]
        ):
            raise ValueError("Training exceeded its memory budget or failed to resume")
        for relative, expected in contract["input_sha256"].items():
            if digest(data / relative) != expected:
                raise ValueError("Training altered a prepared payload file")
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "training-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="artifacts/imagery/skyfall-2015-02")
    parser.add_argument("--profile", default="configs/train-2015-pilot.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.scene, args.output, args.profile), indent=2))


if __name__ == "__main__":
    main()
