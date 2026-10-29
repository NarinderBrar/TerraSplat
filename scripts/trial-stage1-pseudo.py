"""Run the authors' first 1,500 steps and retain peak memory even on CUDA OOM."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.phase2 import digest, preflight  # noqa: E402
from pipeline.training import copy_scene, training_command  # noqa: E402


def main():
    output = ROOT / "artifacts/stage1/ref-jax068-recipe-trial-01"
    scene = ROOT / "artifacts/imagery/ref-jax068/scene-recipe"
    profile_path = ROOT / "configs/train-ref-jax068-recipe.json"
    profile = json.loads(profile_path.read_text())
    audit = preflight(scene / "phase2-manifest.json")
    if audit["status"] != "passed":
        raise ValueError(audit["failures"])
    output.mkdir(parents=True, exist_ok=False)
    copy_scene(scene, output / "input")
    command = training_command(
        sys.executable, output / "input", output / "model", output / "metrics.jsonl", profile, 1500
    )
    # The wrapper records allocator counters after unwinding a failed training call.
    wrapper = """import json, runpy, sys, torch, traceback
from pathlib import Path
report = {"status": "running"}
try:
    sys.argv = sys.argv[2:]
    runpy.run_path("train.py", run_name="__main__")
    report["status"] = "passed"
except BaseException as exc:
    report.update(status="failed", error=str(exc), exception=type(exc).__name__)
    traceback.print_exc()
finally:
    report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), torch=torch.__version__)
    Path(REPORT).write_text(json.dumps(report, indent=2)+"\\n")
""".replace("REPORT", repr(str(output / "memory.json")))
    wrapped = [sys.executable, "-c", wrapper, "trial-wrapper", *command[1:]]
    report = {
        "assumed_anchor": True,
        "command": command,
        "wrapped_command": wrapped,
        "profile_sha256": digest(profile_path),
        "preflight": audit,
        "purpose": "Memory feasibility only; no holdout or lidar evaluation; crosses pseudo-depth start 1000",
    }
    started = time.monotonic()
    with (output / "training.log").open("w") as log:
        result = subprocess.run(
            wrapped, cwd=ROOT / profile["skyfall"], stdout=log, stderr=subprocess.STDOUT, timeout=1200, check=False
        )
    report.update(
        returncode=result.returncode,
        elapsed_seconds=time.monotonic() - started,
        memory=json.loads((output / "memory.json").read_text()),
        log_sha256=digest(output / "training.log"),
    )
    (output / "trial-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
