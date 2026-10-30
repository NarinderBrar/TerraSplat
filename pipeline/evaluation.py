"""Render initial/final pilot models and report image errors on valid pixels."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from pipeline.phase2 import digest

ROOT = Path(__file__).resolve().parents[1]


def image_metrics(reference, rendered, valid):
    if reference.shape != rendered.shape or reference.shape[:2] != valid.shape:
        raise ValueError("Image and validity dimensions disagree")
    if not np.isfinite(rendered).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite image")
    valid = valid.astype(bool)
    if not valid.any():
        raise ValueError("No valid evaluation pixels")
    difference = rendered[valid].astype(np.float64) - reference[valid].astype(np.float64)
    mse = float(np.mean(difference**2))
    return {
        "valid_pixels": int(valid.sum()),
        "mse": mse,
        "mae": float(np.mean(np.abs(difference))),
        "psnr_db": -10 * math.log10(max(mse, 1e-12)),
    }


def load_ply_model(gaussians, path):
    import torch
    from plyfile import PlyData

    data = PlyData.read(path)["vertex"].data
    count = len(data)

    def columns(prefix):
        names = sorted(
            (name for name in data.dtype.names if name.startswith(prefix)),
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )
        return torch.tensor(
            np.stack([np.asarray(data[name], dtype=np.float32) for name in names], axis=1),
            device="cuda",
        )

    def named(name):
        return torch.tensor(np.asarray(data[name], dtype=np.float32)[:, None], device="cuda")

    gaussians._xyz = torch.nn.Parameter(
        torch.tensor(
            np.stack(
                [np.asarray(data[axis], dtype=np.float32) for axis in ("x", "y", "z")],
                axis=1,
            ),
            device="cuda",
        )
    )
    gaussians._features_dc = torch.nn.Parameter(columns("f_dc_").reshape(count, 3, 1).transpose(1, 2))
    rest = columns("f_rest_").reshape(count, 3, -1)
    gaussians._features_rest = torch.nn.Parameter(rest.transpose(1, 2))
    gaussians._opacity = torch.nn.Parameter(named("opacity"))
    gaussians._scaling = torch.nn.Parameter(columns("scale_"))
    gaussians._rotation = torch.nn.Parameter(columns("rot_"))
    if "filter_3D" in data.dtype.names:
        gaussians.filter_3D = named("filter_3D")
    else:
        gaussians.filter_3D = torch.zeros(count, 1, device="cuda")
    gaussians.active_sh_degree = gaussians.max_sh_degree


def render_context(training, model):
    sys.path.insert(0, str(training.get("skyfall", ROOT / ".external/skyfall-gs")))
    from arguments import ModelParams, PipelineParams
    from scene import Scene, GaussianModel

    parser = argparse.ArgumentParser()
    mp, pp = ModelParams(parser), PipelineParams(parser)
    profile = training["profile"]
    arguments = [
        "-s",
        str(training["input_scene"]),
        "-m",
        str(model),
        "--eval",
        "--resolution",
        str(profile["resolution"]),
        "--sh_degree",
        str(profile["sh_degree"]),
        "--data_device",
        "cpu",
    ]
    if profile.get("convert_sh_python"):
        arguments += ["--convert_SHs_python"]
    args = parser.parse_args(arguments)
    gaussians = GaussianModel(profile["sh_degree"], profile.get("appearance_enabled", False), 4, 32)
    scene = Scene(mp.extract(args), gaussians, load_iteration=profile["iterations"])
    if profile.get("appearance_enabled"):
        import torch

        params, _ = torch.load(model / f"chkpnt{profile['iterations']}.pth", weights_only=False)
        gaussians.load_from_checkpoints(params)
    pipeline = pp.extract(args)
    views = [("train", c) for c in scene.getTrainCameras()] + [("holdout", c) for c in scene.getTestCameras()]
    return gaussians, views, pipeline


def render_ply(gaussians, views, pipeline, ply, fixed_appearance=False, keep_mask=None):
    import torch
    from gaussian_renderer import render

    appearance = gaussians.appearance_enabled
    # Fused colours already contain the MLP mapping; initial geometry has no learned embeddings.
    if "fused" in Path(ply).name or "iteration_0" in str(ply):
        gaussians.appearance_enabled = False
    load_ply_model(gaussians, ply)
    if keep_mask is not None:
        from pipeline.export import mask_gaussians

        mask_gaussians(gaussians, keep_mask)
    renders = {}
    with torch.no_grad():
        for role, camera in views:
            result = render(
                camera,
                gaussians,
                pipeline,
                torch.zeros(3, device="cuda"),
                0.1,
                testing=fixed_appearance or role == "holdout",
            )
            rgb = result["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            alpha = result["render_alpha"].squeeze().cpu().numpy()
            if not np.isfinite(alpha).all():
                raise ValueError("Nonfinite rendered alpha")
            renders[(role, camera.image_name)] = {
                "role": role,
                "catalog_id": camera.image_name,
                "rgb": rgb,
                "alpha": alpha,
                "reference": camera.original_image.permute(1, 2, 0).cpu().numpy(),
                "valid": camera.original_mask.squeeze().cpu().numpy().astype(bool),
            }
    gaussians.appearance_enabled = appearance
    return renders


def evaluate(run):
    run = Path(run).resolve()
    report_path = run / "training-report.json"
    training = json.loads(report_path.read_text())
    if training["status"] != "trained":
        raise ValueError("A completed, verified training run is required")
    model = Path(training["model"])
    for relative, expected in training["phases"][-1]["outputs"]["sha256"].items():
        if digest(model / relative) != expected:
            raise ValueError("Trained model changed before evaluation")
    initial = model / "point_cloud/iteration_0/point_cloud.ply"
    if (
        digest(initial) != training["initial_ply_sha256"]
        or digest(model / "georeference.json") != training["georeference_sha256"]
    ):
        raise ValueError("Initial model or georeference changed")
    gaussians, views, pipeline = render_context(training, model)
    profile = training["profile"]
    output = run / "evaluation"
    output.mkdir(exist_ok=False)
    records = {"initial": [], "final": []}
    previews = {}
    for stage, ply in (
        ("initial", initial),
        (
            "final",
            model / f"point_cloud/iteration_{profile['iterations']}/point_cloud.ply",
        ),
    ):
        renders = render_ply(gaussians, views, pipeline, ply)
        for role, camera in views:
            rendered = renders[(role, camera.image_name)]
            rgb, alpha = rendered["rgb"], rendered["alpha"]
            reference, valid = rendered["reference"], rendered["valid"]
            record = {
                "catalog_id": camera.image_name,
                "role": role,
                **image_metrics(reference, rgb, valid),
                "alpha_coverage_above_0_01": float(np.mean(alpha[valid] > 0.01)),
                "mean_alpha": float(np.mean(alpha[valid])),
            }
            records[stage].append(record)
            png = output / f"{stage}-{camera.image_name}.png"
            Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(png)
            previews[(stage, camera.image_name)] = Image.open(png).copy()
            if stage == "initial":
                previews[("reference", camera.image_name)] = Image.fromarray(np.rint(reference * 255).astype(np.uint8))
    canvas = Image.new("RGB", (1200, 440 * len(views)), "#20252b")
    draw = ImageDraw.Draw(canvas)
    for row, (role, camera) in enumerate(views):
        for column, stage in enumerate(("reference", "initial", "final")):
            tile = previews[(stage, camera.image_name)]
            tile.thumbnail((390, 400))
            canvas.paste(tile, (column * 400 + 5, row * 440 + 30))
            draw.text(
                (column * 400 + 5, row * 440 + 6),
                f"{role} {camera.image_name} | {stage}",
                fill="white",
            )
    canvas.save(output / "comparison.png")
    result = {
        "status": "evaluated",
        "assumed_anchor": training.get("assumed_anchor", False),
        "training_report_sha256": digest(report_path),
        "evaluator_sha256": digest(Path(__file__)),
        "resolution_factor": profile["resolution"],
        "metric_definition": (
            "MSE/MAE over RGB samples at valid mask pixels; PSNR uses unit-range colors and a 1e-12 MSE floor"
        ),
        "records": records,
        "holdout_used_for_training_or_selection": False,
        "independent_geographic_accuracy_validated": False,
        "outputs": {p.name: digest(p) for p in sorted(output.glob("*.png"))},
    }
    (output / "evaluation-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run), indent=2))


if __name__ == "__main__":
    main()
