"""Run a bounded, unregistered Stage 1 diagnostic on copied JAX_068 inputs."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / ".external/skyfall-gs"
    original = ROOT.parent / "Ref/Skyfall-GS/data/datasets_JAX/JAX_068"
    work = ROOT / "artifacts/stage1" / time.strftime("smoke-%Y%m%d-%H%M%S")
    work.mkdir(parents=True, exist_ok=False)
    prepared = work / "input"
    prepared.mkdir()
    hashes = {}
    def copy(relative):
        src = original / relative
        dst = prepared / relative
        if not src.resolve().is_relative_to(original.resolve()):
            raise ValueError(f"Input escapes scene: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[str(relative)] = hashlib.sha256(src.read_bytes()).hexdigest()
    for name in ("transforms_train.json", "transforms_test.json", "points3D.txt"):
        copy(name)
    for name in ("transforms_train.json", "transforms_test.json"):
        for frame in json.loads((original / name).read_text())["frames"]:
            copy(frame["file_path"])
            mask = Path("masks") / (Path(frame["file_path"]).stem + ".npy")
            if (original / mask).is_file():
                copy(mask)
    model = work / "model"
    command = [sys.executable, "train.py", "-s", str(prepared), "-m", str(model),
               "--eval", "--iterations", "20", "--resolution", "4", "--sh_degree", "1",
               "--appearance_enabled", "--data_device", "cpu", "--port", "0", "--kernel_size", "0.1",
               "--lambda_depth", "0", "--lambda_opacity", "0", "--lambda_pseudo_depth", "0",
               "--densify_until_iter", "0", "--test_iterations", "20", "--save_iterations", "20",
               "--checkpoint_iterations", "20"]
    report = {"status": "running", "purpose": "20-iteration environment diagnostic, not reconstruction quality validation",
              "georeference": "unregistered: original ENU anchor is missing", "command": command,
              "input_sha256": hashes, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
              "patch_sha256": hashlib.sha256((ROOT / "adapters/skyfall/stage1-only.patch").read_bytes()).hexdigest()}
    output = work / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    print(f"Training diagnostic: {work}", flush=True)
    with (work / "training.log").open("w") as log:
        try:
            result = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            report["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            report["returncode"] = None
            report["failure"] = "300-second diagnostic limit exceeded"
    text = (work / "training.log").read_text()
    ply = model / "point_cloud/iteration_20/point_cloud.ply"
    checkpoint = model / "chkpnt20.pth"
    success = report["returncode"] == 0 and ply.is_file() and checkpoint.is_file()
    report.update(status="passed" if success else "failed", elapsed_seconds=time.monotonic() - started)
    peak = re.search(r"TERRASPLAT_GPU_PEAK_BYTES=(\d+)", text)
    if peak:
        report["peak_cuda_allocated_bytes"] = int(peak.group(1))
    if success:
        try:
            import numpy as np
            import torch
            from plyfile import PlyData
            sys.path.insert(0, str(source))
            # Only deserialize this subprocess's own newly generated checkpoint.
            _, iteration = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert iteration == 20, "Checkpoint has the wrong iteration"
            vertices = PlyData.read(ply)["vertex"]
            for name in vertices.data.dtype.names:
                assert np.isfinite(vertices[name]).all(), f"Nonfinite Gaussian attribute: {name}"
            report["splat_count"] = len(vertices)
            report["validation"] = "checkpoint iteration 20 and finite PLY attributes"
        except (OSError, ValueError, AssertionError, RuntimeError, ImportError) as error:
            success = False
            report.update(status="failed", failure=str(error))
    if success:
        report["outputs"] = {str(p.relative_to(work)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (ply, checkpoint)}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}: {output}")
    if not success:
        print(text[-5000:])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
