"""Fuse a verified Stage 1 model and record its georeferenced export contract."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from pipeline.phase2 import digest

ROOT = Path(__file__).resolve().parents[1]
SKYFALL = ROOT / ".external/skyfall-gs"
DEFAULT_SELECTION = ROOT / "artifacts/imagery/repro-approved-pilot/selection.json"

BLOCK_SIDE_M = 250.0
HALO_M = 20.0
BLOCK_HALF_EXTENT_M = BLOCK_SIDE_M / 2 + HALO_M
FLOATER_ENVELOPE_M = 350.0
VERTICAL_ENVELOPE_M = 250.0
MAX_REMOVED_OPACITY_MASS_FRACTION = 0.01
FLOATER_POLICY_NOTE = (
    "Fixed before inspecting 19-view removal statistics: allow at most 1% of total "
    "activated raw opacity mass to be removed, bounding removal to a small share "
    "of model opacity. Mass is sum(sigmoid(raw opacity)), not splat count or volume. "
    "Remove only ENU horizontal centre radius >350 m before fusion; existing "
    "extent, vertical and transform gates remain unchanged."
)


def floater_removal(attributes, georeference):
    centres = np.stack([attributes[k] for k in ("x", "y", "z")], axis=1)
    enu = apply_transform(georeference["model_to_enu"], centres)
    logits = attributes["opacity"]
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite opacity before floater removal")
    opacity = np.exp(-np.logaddexp(0, -logits))
    radius = np.hypot(enu[:, 0], enu[:, 1])
    keep = radius <= FLOATER_ENVELOPE_M
    removed = radius[~keep]
    total = float(opacity.sum())
    if total <= 0:
        raise ValueError("No activated opacity mass")
    fraction = float(opacity[~keep].sum() / total)
    return keep, {
        "envelope_m": FLOATER_ENVELOPE_M,
        "input_count": len(radius),
        "removed_count": int((~keep).sum()),
        "retained_count": int(keep.sum()),
        "removed_opacity_mass_fraction": fraction,
        "removed_radius_range_m": [float(removed.min()), float(removed.max())] if len(removed) else None,
        "max_removed_opacity_mass_fraction": MAX_REMOVED_OPACITY_MASS_FRACTION,
        "passed": fraction <= MAX_REMOVED_OPACITY_MASS_FRACTION and bool(keep.any()),
        "note": FLOATER_POLICY_NOTE,
    }


def mask_gaussians(gaussians, keep):
    """Apply the same ordered subset to geometry, filter and learned per-splat embeddings."""
    import torch

    mask = torch.as_tensor(keep, dtype=torch.bool, device=gaussians._xyz.device)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_embeddings"):
        value = getattr(gaussians, name, None)
        if value is not None:
            setattr(gaussians, name, torch.nn.Parameter(value[mask]))
    gaussians.filter_3D = gaussians.filter_3D[mask]


TILER_RULES = [
    "REPLACE refinement only; add no other refinement mode.",
    "Child geometricError is non-zero and strictly below its parent's.",
    "Each tile bounds contains every owned splat centre plus about 3 sigma extent, in the tile and every ancestor.",
    "Place the tileset with root.transform, not CESIUM_RTC.",
    "Use the same ENU anchor and model frame for visual and semantic tiles.",
]


def matrix4(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=0, rtol=0):
        raise ValueError(f"{name} must be an affine matrix")
    return matrix


def apply_transform(matrix, points):
    matrix = matrix4(matrix, "transform")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite Nx3 array")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def column_major(matrix):
    matrix = matrix4(matrix, "transform")
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


def verify_run(run):
    run = Path(run).resolve()
    training = json.loads((run / "training-report.json").read_text())
    if training.get("status") != "trained":
        raise ValueError("A completed, verified training run is required")
    model = Path(training["model"]).resolve()
    phase = training["phases"][-1]
    expected = phase["outputs"]["sha256"]
    final_ply = model / f"point_cloud/iteration_{phase['target']}/point_cloud.ply"
    checks = {
        "final_ply": (final_ply, expected[str(final_ply.relative_to(model))]),
        "georeference": (
            model / "georeference.json",
            training["georeference_sha256"],
        ),
        "initial_ply": (
            model / "point_cloud/iteration_0/point_cloud.ply",
            training["initial_ply_sha256"],
        ),
    }
    for name, (path, sha256) in checks.items():
        if not path.is_file() or digest(path) != sha256:
            raise ValueError(f"Trained {name} changed before export")
    for relative, sha256 in expected.items():
        if digest(model / relative) != sha256:
            raise ValueError(f"Trained output changed before export: {relative}")
    if training["final_splats"] != phase["outputs"]["splats"]:
        raise ValueError("Training report splat counts disagree")
    georeference = json.loads((model / "georeference.json").read_text())
    if georeference.get("matrix_layout") != "row-major":
        raise ValueError("Georeference matrices must be row-major")
    for key in ("anchor", "model_to_enu", "enu_to_ecef", "model_to_ecef"):
        if key not in georeference:
            raise ValueError(f"Georeference is missing {key}")
    return run, training, model, final_ply, georeference


def create_fused_ply(model, output, iteration, skyfall=SKYFALL, keep_mask=None):
    script = Path(skyfall) / "create_fused_ply.py"
    if not script.is_file():
        raise ValueError("Skyfall create_fused_ply.py is unavailable")
    shim = "\n".join(
        [
            "import runpy, sys, torch",
            "_load = torch.load",
            "def load(*args, **kwargs):",
            "    kwargs.setdefault('weights_only', False)",
            "    return _load(*args, **kwargs)",
            "torch.load = load",
            *([
                f"sys.path.insert(0, {str(ROOT)!r})",
                "import numpy as np",
                "from pipeline.export import mask_gaussians",
                "from scene.gaussian_model import GaussianModel",
                "_save = GaussianModel.save_fused_ply",
                "def save(self, *args, **kwargs):",
                f"    mask_gaussians(self, np.load({str(keep_mask)!r}))",
                "    return _save(self, *args, **kwargs)",
                "GaussianModel.save_fused_ply = save",
            ] if keep_mask is not None else []),
            "sys.argv = sys.argv[1:]",
            "runpy.run_path(sys.argv[0], run_name='__main__')",
        ]
    )
    command = [
        sys.executable,
        "-c",
        shim,
        str(script),
        "-m",
        str(model),
        "--output_ply",
        str(output),
        "--iteration",
        str(iteration),
        "--load_from_checkpoints",
        "--quiet",
    ]
    result = subprocess.run(
        command, cwd=skyfall, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            "Skyfall fused PLY creation failed:\n"
            + (result.stdout + result.stderr).strip()
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("Skyfall fused PLY creation produced no file")


def read_attributes(path):
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"]
    return {
        name: np.asarray(vertex.data[name], dtype=np.float64)
        for name in vertex.data.dtype.names
    }


def validate_attributes(attributes, expected_count):
    if not attributes:
        raise ValueError("PLY has no attributes")
    count = len(next(iter(attributes.values())))
    if count != expected_count:
        raise ValueError(
            f"Fused splat count {count} differs from trained count {expected_count}"
        )
    for name, values in attributes.items():
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"Nonfinite or malformed PLY attribute: {name}")
    for name in ("x", "y", "z", "opacity"):
        if name not in attributes:
            raise ValueError(f"PLY is missing {name}")
    scales = sorted(name for name in attributes if name.startswith("scale_"))
    rotations = sorted(name for name in attributes if name.startswith("rot_"))
    if len(scales) != 3 or len(rotations) != 4:
        raise ValueError("PLY requires three scales and four rotation values")
    opacity = 1 / (1 + np.exp(-attributes["opacity"]))
    scale = np.exp(np.stack([attributes[name] for name in scales], axis=1))
    rotation = np.stack([attributes[name] for name in rotations], axis=1)
    rotation_norm = np.linalg.norm(rotation, axis=1)
    if not ((opacity > 0) & (opacity < 1)).all():
        raise ValueError("Opacity is not strictly inside (0, 1) after activation")
    if not ((scale > 0) & np.isfinite(scale)).all():
        raise ValueError("Scales are not positive and finite after activation")
    return {
        "splat_count": count,
        "matches_trained_count": count == expected_count,
        "attributes_finite": True,
        "opacity_after_activation": {
            "min": float(opacity.min()),
            "max": float(opacity.max()),
            "strictly_inside_unit_interval": True,
        },
        "scales_after_activation": {
            "min": float(scale.min()),
            "max": float(scale.max()),
            "all_positive": True,
        },
        "rotation_norm": {
            "min": float(rotation_norm.min()),
            "max": float(rotation_norm.max()),
        },
    }


def enu_bbox_splat_centres(attributes, georeference):
    centres = np.stack(
        [attributes["x"], attributes["y"], attributes["z"]], axis=1
    )
    enu = apply_transform(georeference["model_to_enu"], centres)
    radius = np.hypot(enu[:, 0], enu[:, 1])
    opacity = 1 / (1 + np.exp(-attributes["opacity"]))
    order = np.argsort(radius)
    mass = np.cumsum(opacity[order]) / opacity.sum()
    weighted = {
        f"p{int(q * 100)}_radius_m": float(
            radius[order][min(np.searchsorted(mass, q), len(radius) - 1)]
        )
        for q in (0.5, 0.9, 0.95, 0.99)
    }
    minimum = enu.min(axis=0)
    maximum = enu.max(axis=0)
    plausible = bool(
        radius.max() <= FLOATER_ENVELOPE_M
        and abs(minimum[2]) <= VERTICAL_ENVELOPE_M
        and abs(maximum[2]) <= VERTICAL_ENVELOPE_M
    )
    return {
        "frame": "metric ENU at the recorded anchor",
        "min_m": minimum.tolist(),
        "max_m": maximum.tolist(),
        "span_m": (maximum - minimum).tolist(),
        "centre_m": ((minimum + maximum) / 2).tolist(),
        "nominal_block_half_extent_m": BLOCK_HALF_EXTENT_M,
        "fraction_within_nominal_half_extent": float(
            np.mean(radius <= BLOCK_HALF_EXTENT_M)
        ),
        "radius_percentiles_m": {
            "p50": float(np.percentile(radius, 50)),
            "p95": float(np.percentile(radius, 95)),
            "p99": float(np.percentile(radius, 99)),
            "max": float(radius.max()),
        },
        "opacity_weighted_radius": weighted,
        "envelope_m": FLOATER_ENVELOPE_M,
        "vertical_envelope_m": VERTICAL_ENVELOPE_M,
        "plausible": plausible,
        "note": (
            "Coarse sanity envelope only. Border floaters extend the full bbox "
            "beyond the nominal 250 m block plus 20 m halo; this is not an "
            "accuracy measurement."
        ),
    }


def round_trip(attributes, georeference, gsd_m):
    centres = np.stack(
        [attributes["x"], attributes["y"], attributes["z"]], axis=1
    )
    model_to_enu = matrix4(georeference["model_to_enu"], "model_to_enu")
    enu_to_ecef = matrix4(georeference["enu_to_ecef"], "enu_to_ecef")
    enu = apply_transform(model_to_enu, centres)
    ecef = apply_transform(enu_to_ecef, enu)
    enu_again = apply_transform(np.linalg.inv(enu_to_ecef), ecef)
    model_again = apply_transform(np.linalg.inv(model_to_enu), enu_again)
    error_m = np.linalg.norm(enu_again - enu, axis=1)
    error_model = np.linalg.norm(model_again - centres, axis=1)
    # root.transform uses model_to_ecef directly, so it must equal the chain.
    direct = apply_transform(georeference["model_to_ecef"], centres)
    composition_m = np.linalg.norm(direct - ecef, axis=1)
    gate = 0.1 * gsd_m
    return {
        "chain": "model -> ENU -> ECEF -> ENU -> model",
        "max_error_m": float(error_m.max()),
        "max_error_model_units": float(error_model.max()),
        "max_model_to_ecef_composition_error_m": float(composition_m.max()),
        "gsd_m": float(gsd_m),
        "gate_m": float(gate),
        "passed": bool(error_m.max() < gate and composition_m.max() < gate),
        "note": (
            "The round trip proves the recorded transform chain is numerically "
            "consistent; it does not establish independent placement or height "
            "accuracy."
        ),
    }


def gsd_from_selection(selection, catalog_ids):
    records = {
        record["catalog_id"]: record
        for group in ("train", "holdout")
        for record in selection.get(group, [])
    }
    missing = [catalog_id for catalog_id in catalog_ids if catalog_id not in records]
    if missing:
        raise ValueError(f"Selection is missing GSD records: {missing}")
    gsds = {catalog_id: float(records[catalog_id]["gsd_m"]) for catalog_id in catalog_ids}
    if any(not math.isfinite(value) or value <= 0 for value in gsds.values()):
        raise ValueError("Selection GSD values must be finite and positive")
    return {"gsd_m": min(gsds.values()), "by_catalog_id": gsds}


def load_gsd(path, catalog_ids):
    path = Path(path)
    return {**gsd_from_selection(json.loads(path.read_text()), catalog_ids), "source": str(path)}


def render_equivalence(training, model, fused_ply, final_ply, keep_mask=None):
    from pipeline.evaluation import image_metrics, render_context, render_ply

    gaussians, views, pipeline = render_context(training, model)
    raw = render_ply(gaussians, views, pipeline, final_ply, fixed_appearance=True, keep_mask=keep_mask)
    fused = render_ply(gaussians, views, pipeline, fused_ply)
    records = []
    for key, item in raw.items():
        metrics = image_metrics(item["rgb"], fused[key]["rgb"], item["valid"])
        records.append(
            {
                "role": item["role"],
                "catalog_id": item["catalog_id"],
                **metrics,
            }
        )
    return {
        "resolution_factor": training["profile"]["resolution"],
        "method": (
            "Render the raw final model and the fused PLY with the same Skyfall "
            "renderer and cameras; PSNR is between the two renders over valid "
            "mask pixels at the evaluation resolution."
            + (" Both renders use the identical ENU floater subset, including "
               "per-splat appearance embeddings." if keep_mask is not None else "")
        ),
        "views": records,
        "minimum_psnr_db": min(record["psnr_db"] for record in records),
    }


def export(run, selection=None, gsd=None, output=None):
    run, training, model, final_ply, georeference = verify_run(run)
    profile = training["profile"]
    catalog_ids = training["train_catalog_ids"] + training["holdout_catalog_ids"]
    if gsd is None:
        gsd_record = load_gsd(selection or DEFAULT_SELECTION, catalog_ids)
    else:
        gsd_record = {"gsd_m": float(gsd), "source": "command line"}
    output = Path(output).resolve() if output is not None else run / "export"
    fused = output / f"{run.name}-fused.ply"
    if output.exists():
        raise ValueError("Use a new export directory; existing exports are immutable")
    output.mkdir(parents=True)
    keep, removal = floater_removal(read_attributes(final_ply), georeference)
    if not removal["passed"]:
        (output / "export-manifest.json").write_text(json.dumps({
            "status": "failed", "floater_removal": removal,
            "validation": {"failures": ["Removed opacity-mass fraction exceeds 1% or no splats remain"]},
        }, indent=2) + "\n")
        raise ValueError("Removed opacity-mass fraction exceeds 1% or no splats remain")
    mask_path = None
    if removal["removed_count"]:
        mask_path = output / "retained-splats.npy"
        np.save(mask_path, keep)
    create_fused_ply(model, fused, profile["iterations"], training.get("skyfall", SKYFALL), mask_path)
    attributes = read_attributes(fused)
    validation = validate_attributes(attributes, training["final_splats"] - removal["removed_count"])
    bbox = enu_bbox_splat_centres(attributes, georeference)
    trip = round_trip(attributes, georeference, gsd_record["gsd_m"])
    failures = []
    if not bbox["plausible"] and not georeference.get("assumed_anchor", False):
        failures.append("ENU splat-centre bbox exceeds the coarse plausibility envelope")
    if not trip["passed"]:
        failures.append("Transform round-trip error exceeds 0.1 GSD")
    manifest = {
        "schema_version": 1,
        "assumed_anchor": georeference.get("assumed_anchor", False),
        "status": "validated" if not failures else "failed",
        "run_directory": str(run),
        "training_report_sha256": digest(run / "training-report.json"),
        "source": {
            "final_ply": {
                "path": str(final_ply),
                "sha256": digest(final_ply),
            },
            "georeference": {
                "path": str(model / "georeference.json"),
                "sha256": digest(model / "georeference.json"),
            },
            "checkpoint": {
                "path": str(model / f"chkpnt{profile['iterations']}.pth"),
                "sha256": training["phases"][-1]["outputs"]["sha256"][
                    f"chkpnt{profile['iterations']}.pth"
                ],
            },
        },
        "fused_ply": {
            "path": str(fused),
            "sha256": digest(fused),
            "bytes": fused.stat().st_size,
        },
        "splat_count": validation["splat_count"],
        "floater_removal": removal,
        "sh_degree": profile["sh_degree"],
        "fusion": {
            "source": "Skyfall create_fused_ply.py with --load_from_checkpoints",
            "attributes": [
                "xyz",
                "normals",
                "f_dc",
                "f_rest",
                "opacity",
                "scale",
                "rotation",
            ],
            "filter_3D_folded_into_opacity_and_scale": True,
            "filter_3D_attribute_written": False,
            "appearance_colour_mapping": profile.get("appearance_enabled", False),
            "splats_removed": training["final_splats"] - validation["splat_count"],
            "torch_load_compat": "weights_only=False shim inside the subprocess",
        },
        "georeference": {
            "assumed_anchor": georeference.get("assumed_anchor", False),
            "matrix_layout": georeference["matrix_layout"],
            "source_frame": georeference.get("source_frame"),
            "model_frame": georeference.get("model_frame"),
            "anchor": georeference["anchor"],
            "model_to_enu": georeference["model_to_enu"],
            "enu_to_ecef": georeference["enu_to_ecef"],
            "model_to_ecef": georeference["model_to_ecef"],
        },
        "tileset": {
            "root_transform": column_major(georeference["model_to_ecef"]),
            "root_transform_layout": "column-major 4x4, as 3D Tiles expects",
            "root_transform_equation": "root.transform = model_to_ecef",
            "source_matrix_layout": "row-major 4x4 in georeference.json",
            "frame_note": (
                "Splats remain in the Skyfall Z-up model frame. glTF is Y-up; "
                "any axis conversion belongs to the tiler and must not be "
                "applied twice."
            ),
            "rules": TILER_RULES,
        },
        "validation": {
            **validation,
            "enu_bbox": bbox,
            "extent_gate_required": not georeference.get("assumed_anchor", False),
            "extent_note": (
                "Assumed-anchor reference: full cloud extent is diagnostic; authors initialization "
                "covers a wider area and contains outliers. DSM uses the fixed 250 m AOI."
                if georeference.get("assumed_anchor", False) else "Pilot extent gate enforced"
            ),
            "failures": failures,
        },
        "round_trip": trip,
        "notes": [
            "Fused PLY is for renderers that expect standalone Gaussian attributes.",
            "Independent placement and height accuracy remain unverified.",
            *TILER_RULES,
        ],
    }
    if failures:
        (output / "export-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        raise ValueError("; ".join(failures))
    manifest["render_equivalence"] = render_equivalence(
        training, model, fused, final_ply, keep if removal["removed_count"] else None
    )
    manifest["status"] = "exported"
    (output / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "schema_version": 1,
        "assumed_anchor": georeference.get("assumed_anchor", False),
        "status": "exported",
        "run_directory": str(run),
        "source_final_ply_sha256": manifest["source"]["final_ply"]["sha256"],
        "source_georeference_sha256": manifest["source"]["georeference"]["sha256"],
        "fused_ply": manifest["fused_ply"],
        "splat_count": manifest["splat_count"],
        "sh_degree": manifest["sh_degree"],
        "enu_anchor": georeference["anchor"],
        "enu_bbox_m": {
            "min": bbox["min_m"],
            "max": bbox["max_m"],
            "span": bbox["span_m"],
            "fraction_within_nominal_half_extent": bbox[
                "fraction_within_nominal_half_extent"
            ],
        },
        "round_trip": {
            "max_error_m": trip["max_error_m"],
            "gsd_m": trip["gsd_m"],
            "gate_m": trip["gate_m"],
            "passed": trip["passed"],
        },
        "render_equivalence": manifest["render_equivalence"],
        "root_transform": manifest["tileset"]["root_transform"],
        "limitations": [
            "The round trip only proves transform-chain consistency.",
            "Independent placement and height accuracy remain unverified.",
            "The full ENU bbox includes border floaters beyond the nominal block.",
        ],
    }
    (output / "export-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--gsd", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.selection, args.gsd, args.output), indent=2))


if __name__ == "__main__":
    main()
