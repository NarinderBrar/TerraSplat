"""Render the fused export from a synthetic nadir camera and back-project its depth.

The camera sits high above the block with a narrow field of view, so the
render is close to orthographic. Expected depth (alpha-weighted depth divided
by alpha) is back-projected to ENU points for `pipeline.dsm_validation --points`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.phase2 import digest

ROOT = Path(__file__).resolve().parents[1]


def nadir_camera(centre_xy, altitude, half_width, pixels):
    """World-to-camera pose and intrinsics for a camera looking straight down (model Z-up).

    COLMAP camera axes: x right (east), y down (south), z forward (down).
    """
    rotation = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # camera-to-world columns
    centre = np.array([centre_xy[0], centre_xy[1], altitude], dtype=np.float64)
    translation = -rotation.T @ centre
    fov = 2 * np.arctan(half_width / altitude)
    focal = pixels / (2 * np.tan(fov / 2))
    return {"rotation": rotation, "translation": translation, "centre": centre,
            "fov": float(fov), "focal": float(focal), "pixels": int(pixels)}


def back_project(depth, camera):
    """Pixel-centre rays (3DGS convention: principal point at (W/2 - 0.5, H/2 - 0.5))."""
    rows, columns = np.indices(depth.shape)
    principal = camera["pixels"] / 2 - 0.5
    x = (columns - principal) / camera["focal"] * depth
    y = (rows - principal) / camera["focal"] * depth
    points = np.stack([x, y, depth], -1).reshape(-1, 3)
    return points @ camera["rotation"].T + camera["centre"]


def render(run, output, pixel_m=0.25, altitude_m=17000.0, half_extent_m=145.0, min_alpha=0.5):
    import torch

    sys_path = str(ROOT / ".external/skyfall-gs")
    import sys
    sys.path.insert(0, sys_path)
    from scene.cameras import Camera
    from gaussian_renderer import render as skyfall_render
    from pipeline.evaluation import load_ply_model, render_context

    run, output = Path(run).resolve(), Path(output).resolve()
    training = json.loads((run / "training-report.json").read_text())
    manifest = json.loads((run / "export/export-manifest.json").read_text())
    fused = Path(manifest["fused_ply"]["path"])
    if digest(fused) != manifest["fused_ply"]["sha256"]:
        raise ValueError("Fused PLY changed after export")
    model_to_enu = np.asarray(manifest["georeference"]["model_to_enu"], dtype=np.float64)
    scale = float(np.cbrt(np.linalg.det(model_to_enu[:3, :3])))
    enu_to_model = np.linalg.inv(model_to_enu)

    gaussians, _, pipeline = render_context(training, Path(training["model"]))
    load_ply_model(gaussians, fused)
    centre = enu_to_model[:3, :3] @ np.zeros(3) + enu_to_model[:3, 3]
    pixels = int(round(2 * half_extent_m / pixel_m))
    camera = nadir_camera(centre[:2], altitude_m / scale, half_extent_m / scale, pixels)
    view = Camera(colmap_id=0, R=camera["rotation"], T=camera["translation"],
                  FoVx=camera["fov"], FoVy=camera["fov"], cx=0.0, cy=0.0,
                  image=torch.zeros(3, pixels, pixels), gt_alpha_mask=None,
                  image_name="nadir", uid=0)
    with torch.no_grad():
        result = skyfall_render(view, gaussians, pipeline, torch.zeros(3, device="cuda"), 0.1, testing=True)
    weighted = result["render_depth"].squeeze().double().cpu().numpy()
    alpha = result["render_alpha"].squeeze().double().cpu().numpy()
    if not (np.isfinite(weighted).all() and np.isfinite(alpha).all()):
        raise ValueError("Nonfinite rendered depth or alpha")
    valid = alpha >= min_alpha
    depth = np.where(valid, weighted / np.maximum(alpha, 1e-6), np.nan)
    model_points = back_project(np.nan_to_num(depth), camera)[valid.ravel()]
    enu = model_points @ model_to_enu[:3, :3].T + model_to_enu[:3, 3]

    if output.exists():
        raise ValueError("Use a new depth directory; remove the previous one first")
    output.mkdir(parents=True)
    np.savez_compressed(output / "depth-points.npz", enu=enu.astype(np.float64))
    report = {
        "schema_version": 1,
        "fused_ply_sha256": manifest["fused_ply"]["sha256"],
        "camera": {"type": "synthetic nadir, near-orthographic", "altitude_m": altitude_m,
                   "half_extent_m": half_extent_m, "pixel_m": pixel_m, "pixels": pixels,
                   "fov_deg": float(np.degrees(camera["fov"])),
                   "max_lean_m_for_30m_at_edge": float(30 * half_extent_m / altitude_m)},
        "depth": "alpha-weighted view depth / alpha (Skyfall rasterizer), alpha >= %.2f" % min_alpha,
        "points": int(len(enu)),
        "alpha_coverage": float(valid.mean()),
        "enu_height_range_m": [float(np.min(enu[:, 2])), float(np.max(enu[:, 2]))],
    }
    (output / "depth-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
