"""Read-only Phase 0 inventory. No models, data, or credentials are uploaded."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"ok": result.returncode == 0, "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "output": str(error)}


def file_record(path):
    if not path.is_file():
        return {"present": False}
    return {"present": True, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inventory(reference_root, terrascope, scene):
    skyfall = reference_root / "Skyfall-GS"
    data = skyfall / "data" / "datasets_JAX" / scene
    registration = terrascope / "docs" / "splat-registration" / f"{scene}.json"
    observers = sorted(data.rglob("enu_observer_latlonalt.json")) if data.exists() else []
    registration_data = json.loads(registration.read_text()) if registration.is_file() else {}
    return {
        "scene": scene,
        "skyfall": {"present": skyfall.is_dir(), "commit": command(["git", "-C", str(skyfall), "rev-parse", "HEAD"]),
                    "submodules": command(["git", "-C", str(skyfall), "submodule", "status"])},
        "inputs": {name: file_record(data / name) for name in ["transforms_train.json", "transforms_test.json", "points3D.ply"]},
        "georeference": {
            "observerFiles": [str(p.relative_to(data)) for p in observers],
            "automaticRegistration": "unverified" if observers else "missing ENU observer metadata in prepared scene",
            "terrascopeRegistration": file_record(registration),
            "manualBaseline": registration_data.get("georeference"),
            "verticalLimitation": registration_data.get("verticalAlignment", {}).get("limitation"),
            "note": "A visual registration and flat ground datum do not establish surveyed ECEF placement.",
        },
        "trainingEnvironment": {
            "nvcc": shutil.which("nvcc"), "torchImportable": importlib.util.find_spec("torch") is not None,
            "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        },
        "status": "Inventory only. Training, fused model provenance, and real-scene geographic accuracy remain separate gates.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    siblings = Path(__file__).resolve().parents[3]
    parser.add_argument("--reference-root", type=Path, default=siblings / "Ref")
    parser.add_argument("--terrascope", type=Path, default=siblings / "TerraScope")
    parser.add_argument("--scene", default="JAX_068")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(inventory(args.reference_root, args.terrascope, args.scene), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")
