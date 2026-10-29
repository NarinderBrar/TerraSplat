"""Convert read-only TerraScope visual registrations to metric ENU placements.

Placed exports are derived manifests: training inputs and original exports stay immutable.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from pipeline.export import apply_transform, column_major, round_trip, read_attributes
from pipeline.georeference import placement
from pipeline.phase2 import digest
from pipeline.reference_scene import legacy_normalization

ROOT = Path(__file__).resolve().parents[1]
LABEL = ("Skyfall-GS Stage 1 trained by TerraSplat on the authors' prepared scene; "
         "visually registered by TerraScope (about 1.3-3.8 m); vertical approximate; "
         "not produced from TerraSplat's raw-imagery pipeline")
IDENTITY = {"rotation": np.eye(3).tolist(), "translation": [0, 0, 0], "scale": 1, "z_shift": 0}


def ground_ellipsoid(lon, lat, city):
    """Approximate city street altitude, with local EGM96 when installed; never fit to lidar."""
    orthometric = 12.0 if city == "JAX" else 5.0
    try:
        from pyproj import Transformer, datadir

        grid = next(p for p in (Path(datadir.get_data_dir()) / "egm96_15.gtx",
                                  Path("/usr/share/proj/egm96_15.gtx")) if p.is_file())
        transform = Transformer.from_pipeline(
            "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
            f"+step +proj=vgridshift +grids={grid} +multiplier=1 "
            "+step +proj=unitconvert +xy_in=rad +xy_out=deg")
        height = transform.transform(lon, lat, orthometric, errcheck=True)[2]
        return height, f"EGM96 egm96_15.gtx via pyproj; approximate city orthometric street {orthometric:g} m"
    except (ImportError, RuntimeError, StopIteration):
        undulation = -29.4 if city == "JAX" else -32.7
        return orthometric + undulation, f"city fallback geoid undulation {undulation:g} m; street {orthometric:g} m"


def registration_placement(registration, ground_height_m, core_model_xy=(0, 0)):
    affine = np.asarray(registration["modelRasterToReferencePixels"], dtype=float)
    raster, origin = registration["modelRaster"], registration["tileOrigin"]
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        raise ValueError("Expected finite 2x3 registration")
    unit = raster["unitsPerPixel"]
    pixel_at_zero = affine @ [(core_model_xy[0] - raster["left"]) / unit,
                              (raster["top"] - core_model_xy[1]) / unit, 1]
    world = pixel_at_zero + np.array([origin["x0"], origin["y0"]]) * 256
    pixels = 256 * 2 ** origin["z"]
    lon = world[0] / pixels * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * world[1] / pixels))))
    metres_per_pixel = 2 * math.pi * 6378137 / pixels * math.cos(math.radians(lat))
    horizontal = np.diag([metres_per_pixel, -metres_per_pixel]) @ affine[:, :2] @ np.diag([1 / unit, -1 / unit])
    scale = math.sqrt(np.linalg.det(horizontal))
    rotation = horizontal / scale
    if not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-8):
        raise ValueError("Registration must be a similarity without shear or reflection")
    anchor = {"longitude": float(lon), "latitude": float(lat), "ellipsoidal_height_m": ground_height_m}
    geo = placement(anchor, IDENTITY)
    matrix = np.eye(4)
    matrix[:2, :2] = horizontal
    matrix[:2, 3] = -horizontal @ np.asarray(core_model_xy)
    matrix[2, 2] = scale
    ground_z = registration.get("georeference", {}).get("groundZ", 0)
    matrix[2, 3] = -scale * ground_z
    geo.update(anchor=anchor, assumed_anchor=False, visually_registered=True, vertical_approximate=True,
               model_to_enu=matrix.tolist(),
               model_to_ecef=(np.asarray(geo["enu_to_ecef"]) @ matrix).tolist(),
               core_model_xy=list(core_model_xy), ground_model_z=ground_z, metres_per_model_unit=scale,
               rotation_about_up_deg=math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])),
               label=LABEL)
    return geo


def verify_frame(source, registration):
    source = Path(source)
    transforms = json.loads((source / "transforms_train.json").read_text())
    xyz = np.array([line.split()[1:4] for line in (source / "points3D.txt").read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")], dtype=float)
    norm = legacy_normalization(xyz, transforms)
    model = xyz @ np.asarray(norm["rotation"]).T - norm["translation"]
    model *= norm["scale"]
    model[:, 2] -= norm["z_shift"]
    raster = registration["modelRaster"]
    # Registrations project a 1200x1200 model raster; reject incompatible input frames.
    low, high = np.percentile(model, [1, 99], axis=0)
    if (low[:2] < [raster["left"], raster["top"] - 1200 * raster["unitsPerPixel"]]).any() or (
            high[:2] > [raster["left"] + 1200 * raster["unitsPerPixel"], raster["top"]]).any():
        raise ValueError("Prepared model frame exceeds registration raster")
    return {"normalization": norm, "sparse_points": len(xyz), "p01_model": low.tolist(),
            "p99_model": high.tolist(), "registration_raster_contains_sparse_core": True,
            "limitation": "Frame compatibility only; final-model landmarks are not independent Stage 1 accuracy"}


def place(run, source, registration_path, output, depth_points=None, core_model_xy=(0, 0)):
    run, output = Path(run), Path(output)
    if output.exists():
        raise ValueError("Use a new placed export directory")
    registration = json.loads(Path(registration_path).read_text())
    frame = verify_frame(source, registration)
    preliminary = registration_placement(registration, 0, core_model_xy)
    anchor = preliminary["anchor"]
    height, method = ground_ellipsoid(anchor["longitude"], anchor["latitude"], registration["scene"][:3])
    geo = registration_placement(registration, height, core_model_xy)
    geo["vertical_method"] = "TerraScope calibrated groundZ mapped to " + method
    original = run / "export/export-manifest.json"
    manifest = json.loads(original.read_text())
    if manifest["status"] != "exported":
        raise ValueError("Completed Stage 1 export required")
    fused = Path(manifest["fused_ply"]["path"])
    if digest(fused) != manifest["fused_ply"]["sha256"]:
        raise ValueError("Fused model changed")
    trip = round_trip(read_attributes(fused), geo, manifest["round_trip"]["gsd_m"])
    if not trip["passed"]:
        raise ValueError("Placed transform round trip failed")
    output.mkdir(parents=True)
    if depth_points:
        points = np.load(depth_points)["enu"]
        old = np.asarray(manifest["georeference"]["model_to_enu"])
        placed = apply_transform(np.asarray(geo["model_to_enu"]) @ np.linalg.inv(old), points)
        np.savez_compressed(output / "depth-points.npz", enu=placed)
    manifest.update(georeference=geo, assumed_anchor=False, round_trip=trip,
                    placement={"registration_sha256": digest(Path(registration_path)),
                               "source_export_sha256": digest(original), "frame_check": frame,
                               "label": LABEL, "vertical_method": geo["vertical_method"]})
    manifest["tileset"]["root_transform"] = column_major(geo["model_to_ecef"])
    (output / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "source", "registration", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--depth-points")
    args = parser.parse_args()
    result = place(args.run, args.source, args.registration, args.output, args.depth_points)
    print(json.dumps(result["placement"], indent=2))


if __name__ == "__main__":
    main()
