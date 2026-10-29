"""Generate static MoGe camera-z priors for training acquisitions only."""

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from pipeline.phase2 import digest, preflight

MOGE_COMMIT = "aca893a40faf6905cb722d6614cc37cb87ea06f7"
MODEL_ID = "Ruicheng/moge-vitl"


def write_exr(path, depth):
    """Write the float32 Y channel consumed by Skyfall's read_exr."""
    import Imath
    import OpenEXR

    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    header = OpenEXR.Header(width, height)
    header["channels"] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    output = OpenEXR.OutputFile(str(path), header)
    try:
        output.writePixels({"Y": depth.tobytes()})
    finally:
        output.close()


def generate(scene, output, predict, revision, *, fov_x_used=True):
    """Predict from full RGB arrays; predict returns an H×W camera-z array."""
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(scene):
        raise ValueError("Use a new scene directory outside the input scene")
    audit = preflight(scene / "phase2-manifest.json", check_environment=False)
    if audit["status"] != "passed":
        raise ValueError(audit["failures"])
    contract = json.loads((scene / "terrasplat-scene.json").read_text())
    for relative, expected in contract["input_sha256"].items():
        path = (scene / relative).resolve()
        if not path.is_relative_to(scene) or digest(path) != expected:
            raise ValueError(f"Prepared payload changed: {relative}")
    if (scene / "depths_moge").exists():
        raise ValueError("Input scene already contains depth priors")
    frames = json.loads((scene / "transforms_train.json").read_text())["frames"]
    names = [Path(frame["file_path"]).stem for frame in frames]
    if (len(names) != len(set(names)) or set(names) != set(contract["train_catalog_ids"])
            or set(names) & set(contract["holdout_catalog_ids"])):
        raise ValueError("Training transforms disagree with acquisition splits")
    shutil.copytree(scene, output)
    manifest_path = output / "phase2-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["skyfall"] = os.path.relpath((scene / manifest["skyfall"]).resolve(), output)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    contract["input_sha256"][manifest_path.name] = digest(manifest_path)
    depths = output / "depths_moge"
    depths.mkdir()
    report = {
        "version": 1,
        "status": "generated",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "moge_commit": MOGE_COMMIT,
        "generator_sha256": digest(Path(__file__)),
        "source_manifest_sha256": digest(scene / "terrasplat-scene.json"),
        "georeference_sha256": digest(scene / "georeference.json"),
        "fov_x_used": fov_x_used,
        "depth_convention": "camera z; relative scale and offset; float32 EXR Y",
        "inference": {"resolution_level": 9, "apply_mask": True, "precision": "float32"},
        "holdout_used": False,
        "images": [],
    }
    for frame, name in zip(frames, names):
        print(f"MoGe training prior: {name}", flush=True)
        with Image.open(scene / frame["file_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        mask = np.load(scene / "masks" / f"{name}.npy")
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Mask dimensions disagree: {name}")
        fov = math.degrees(2 * math.atan(rgb.shape[1] / (2 * frame["fl_x"])))
        depth = np.array(predict(rgb, fov if fov_x_used else None), dtype=np.float32, copy=True)
        if depth.shape != mask.shape:
            raise ValueError(f"Depth dimensions disagree: {name}")
        valid = (mask > 0) & np.isfinite(depth)
        if not valid.any() or np.ptp(depth[valid]) == 0:
            raise ValueError(f"Empty or constant depth prior: {name}")
        depth[~valid] = np.nan
        path = depths / f"{name}.exr"
        write_exr(path, depth)
        relative = str(path.relative_to(output))
        contract["input_sha256"][relative] = digest(path)
        report["images"].append({
            "catalog_id": name,
            "shape": list(depth.shape),
            "fov_x_degrees": fov,
            "valid_fraction": float(valid.mean()),
            "depth_min": float(depth[valid].min()),
            "depth_max": float(depth[valid].max()),
            "image_sha256": digest(scene / frame["file_path"]),
            "mask_sha256": digest(scene / "masks" / f"{name}.npy"),
            "depth_sha256": digest(path),
        })
    report_path = output / "depth-priors-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    contract["input_sha256"][report_path.name] = digest(report_path)
    (output / "terrasplat-scene.json").write_text(json.dumps(contract, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="artifacts/imagery/pilot-8w/skyfall")
    parser.add_argument("--output", default="artifacts/imagery/pilot-8w/skyfall-moge")
    parser.add_argument("--revision", help="HF revision; default resolves the current model commit once")
    args = parser.parse_args()
    import torch
    from huggingface_hub import model_info
    from moge.model import MoGeModel
    import moge.model

    source = Path(moge.model.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--", "moge", "utils3d"], cwd=source)
    if commit != MOGE_COMMIT or dirty:
        raise ValueError("MoGe runtime must match the recorded source pin")
    revision = model_info(MODEL_ID, revision=args.revision).sha
    model = MoGeModel.from_pretrained(MODEL_ID, revision=revision).to("cuda").eval()
    uses_fov = "fov_x" in inspect.signature(model.infer).parameters

    def predict(rgb, fov):
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to("cuda") / 255
        with torch.inference_mode():
            kwargs = {"fov_x": fov} if uses_fov else {}
            return model.infer(tensor, **kwargs)["depth"].cpu().numpy()

    report = generate(args.scene, args.output, predict, revision, fov_x_used=uses_fov)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
