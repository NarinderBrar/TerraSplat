"""Compare a fused export with an independent lidar DSM (DFC2019 Track 3 truth).

The splat DSM is placed only through the recorded transform chain
(model -> ENU -> ECEF -> UTM, ellipsoidal heights), so the unregistered
comparison measures absolute placement and height. Skyfall's `dsmr` then finds
the best horizontal shift and vertical offset; that shift is the placement
error, and the residual after it measures shape.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# DFC2019 Track 3 CLS codes.
CLASSES = {"ground": 2, "vegetation": 5, "building": 6, "water": 9, "bridge": 17}
PLY_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}


def read_ply(path):
    """Minimal binary little-endian vertex reader (no plyfile in this environment)."""
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("Not a PLY file")
        fields, count = [], None
        while True:
            line = handle.readline().decode("ascii").split()
            if line[0] == "format" and line[1] != "binary_little_endian":
                raise ValueError("Only binary little-endian PLY is supported")
            if line[0] == "element":
                if line[1] != "vertex" or count is not None:
                    raise ValueError("Only a single vertex element is supported")
                count = int(line[2])
            elif line[0] == "property":
                fields.append((line[2], PLY_TYPES[line[1]]))
            elif line[0] == "end_header":
                break
        return np.fromfile(handle, dtype=np.dtype(fields), count=count)


def read_truth(directory, aoi):
    import rasterio

    directory = Path(directory)
    easting, northing, size, resolution = np.loadtxt(directory / f"{aoi}_DSM.txt")
    with rasterio.open(directory / f"{aoi}_DSM.tif") as source:
        dsm = source.read(1).astype(np.float64)
    with rasterio.open(directory / f"{aoi}_CLS.tif") as source:
        classes = source.read(1)
    size = int(size)
    if dsm.shape != (size, size) or classes.shape != dsm.shape:
        raise ValueError("Truth DSM, CLS and metadata disagree on size")
    # Sat-NeRF convention: the metadata origin is the lower-left corner; row 0 is north.
    return {"dsm": dsm, "classes": classes, "west": float(easting),
            "north": float(northing) + size * resolution, "resolution": float(resolution)}


def enu_to_utm(enu, georeference, utm_epsg):
    from pyproj import Transformer

    enu_to_ecef = np.asarray(georeference["enu_to_ecef"], dtype=np.float64)
    ecef = enu @ enu_to_ecef[:3, :3].T + enu_to_ecef[:3, 3]
    to_geodetic = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    lon, lat, height = to_geodetic.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    easting, northing = to_utm.transform(lon, lat)
    return np.stack([easting, northing, height], 1)


def points_dsm(points, grid):
    """Max height per truth-grid cell of dense surface points (Skyfall/plyflatten style, radius 0)."""
    size = grid["dsm"].shape[0]
    resolution = grid["resolution"]
    columns = np.floor((points[:, 0] - grid["west"]) / resolution).astype(np.int64)
    rows = np.floor((grid["north"] - points[:, 1]) / resolution).astype(np.int64)
    inside = (columns >= 0) & (columns < size) & (rows >= 0) & (rows < size)
    dsm = np.full(size * size, -np.inf)
    np.maximum.at(dsm, rows[inside] * size + columns[inside], points[inside, 2])
    dsm[np.isinf(dsm)] = np.nan
    return dsm.reshape(size, size)


def splat_dsm(points, radii, grid):
    """Max-height raster of splat footprints (discs of the given radii) on the truth grid."""
    size = grid["dsm"].shape[0]
    resolution = grid["resolution"]
    dsm = np.full((size, size), -np.inf)
    columns = (points[:, 0] - grid["west"]) / resolution - 0.5
    rows = (grid["north"] - points[:, 1]) / resolution - 0.5
    for column, row, height, radius in zip(columns, rows, points[:, 2], radii / resolution):
        reach = int(np.ceil(radius))
        c0, c1 = max(int(round(column)) - reach, 0), min(int(round(column)) + reach + 1, size)
        r0, r1 = max(int(round(row)) - reach, 0), min(int(round(row)) + reach + 1, size)
        if c0 >= c1 or r0 >= r1:
            continue
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        inside = (cc - column) ** 2 + (rr - row) ** 2 <= max(radius, 0.5) ** 2
        window = dsm[r0:r1, c0:c1]
        window[inside] = np.maximum(window[inside], height)
    dsm[np.isinf(dsm)] = np.nan
    return dsm


def error_stats(predicted, truth, mask):
    valid = mask & np.isfinite(predicted) & np.isfinite(truth)
    difference = predicted[valid] - truth[valid]
    if not len(difference):
        return {"pixels": 0}
    return {
        "pixels": int(valid.sum()),
        "completeness": float(valid.sum() / max(int((mask & np.isfinite(truth)).sum()), 1)),
        "bias_m": float(np.mean(difference)),
        "median_m": float(np.median(difference)),
        "mae_m": float(np.mean(np.abs(difference))),
        "rmse_m": float(np.sqrt(np.mean(difference**2))),
        "nmad_m": float(1.4826 * np.median(np.abs(difference - np.median(difference)))),
    }


def by_class(predicted, truth, classes, extent):
    result = {"all_labelled": error_stats(predicted, truth, extent & (classes != 65) & (classes != CLASSES["water"]))}
    for name in ("ground", "building", "vegetation"):
        result[name] = error_stats(predicted, truth, extent & (classes == CLASSES[name]))
    return result


def write_raster(path, array):
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", width=array.shape[1], height=array.shape[0],
                       count=1, dtype="float32", nodata=np.nan) as target:
        target.write(array.astype(np.float32), 1)


def register(truth_path, predicted_path, registered_path):
    """Skyfall/Sat-NeRF dsmr: integer-then-subpixel NCC shift plus a vertical offset."""
    sys.path.insert(0, str(ROOT / ".external/skyfall-gs"))
    import dsmr

    dx, dy, _, offset = dsmr.compute_shift(str(truth_path), str(predicted_path), scaling=False)
    dsmr.apply_shift(str(predicted_path), str(registered_path), dx, dy, 1, offset)
    return float(dx), float(dy), float(offset)


def offset_from_truth(dx, dy, z_offset, resolution):
    """Where the predicted surface sits relative to the truth, in metres.

    dsmr reports the predicted content's displacement in (column, row) pixels,
    rows growing southward, and the z offset to add to the predicted heights.
    """
    return {"east": dx * resolution, "north": -dy * resolution, "up": -z_offset,
            "horizontal": float(np.hypot(dx, dy) * resolution)}


def preview(path, truth, predicted, registered):
    import rasterio

    low, high = np.nanpercentile(truth, [1, 99])
    def grey(a):
        v = np.clip((a - low) / (high - low), 0, 1)
        out = np.rint(np.nan_to_num(v, nan=0) * 255).astype(np.uint8)
        return np.stack([out] * 3, -1)
    def signed(a):
        v = np.clip(a / 5, -1, 1)
        rgb = np.full(a.shape + (3,), 255, np.uint8)
        rgb[..., 0] = np.rint(255 * (1 - np.clip(-v, 0, 1))).astype(np.uint8)
        rgb[..., 1] = np.rint(255 * (1 - np.abs(v))).astype(np.uint8)
        rgb[..., 2] = np.rint(255 * (1 - np.clip(v, 0, 1))).astype(np.uint8)
        rgb[~np.isfinite(a)] = 0
        return rgb
    panels = [grey(truth), grey(predicted), signed(predicted - truth), signed(registered - truth)]
    image = np.concatenate(panels, 1).transpose(2, 0, 1)
    with rasterio.open(path, "w", driver="PNG", width=image.shape[2], height=image.shape[1],
                       count=3, dtype="uint8") as target:
        target.write(image)


def validate(export_dir, truth_dir, output, aoi="JAX_068", utm_epsg=32617,
             min_opacity=0.5, max_radius_m=3.0, depth_points=None):
    from pipeline.phase2 import digest

    export_dir, output = Path(export_dir).resolve(), Path(output).resolve()
    manifest = json.loads((export_dir / "export-manifest.json").read_text())
    fused = Path(manifest["fused_ply"]["path"])
    if digest(fused) != manifest["fused_ply"]["sha256"]:
        raise ValueError("Fused PLY changed after export")
    georeference = manifest["georeference"]
    truth = read_truth(truth_dir, aoi)

    if depth_points is not None:
        enu_points = np.load(depth_points)["enu"]
        predicted = points_dsm(enu_to_utm(enu_points, georeference, utm_epsg), truth)
        source = {
            "dsm": "max height per cell of back-projected rendered depth (see depth-report.json)",
            "depth_points": int(len(enu_points)),
            "depth_points_sha256": digest(Path(depth_points)),
            "deviation": "Same rendered-depth principle as Skyfall, from one synthetic nadir camera instead of its camera path",
        }
    else:
        predicted, source = footprint_dsm(fused, georeference, truth, utm_epsg, min_opacity, max_radius_m)
    return finish(export_dir, output, manifest, georeference, truth, truth_dir, aoi, utm_epsg, predicted, source)


def footprint_dsm(fused, georeference, truth, utm_epsg, min_opacity, max_radius_m):
    vertices = read_ply(fused)
    centres = np.stack([vertices[a].astype(np.float64) for a in "xyz"], 1)
    model_to_enu = np.asarray(georeference["model_to_enu"], dtype=np.float64)
    enu = centres @ model_to_enu[:3, :3].T + model_to_enu[:3, 3]
    scale = float(np.cbrt(np.linalg.det(model_to_enu[:3, :3])))
    opacity = 1 / (1 + np.exp(-vertices["opacity"].astype(np.float64)))
    log_scales = np.stack([vertices[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)
    # Footprint radius: the two largest axes' geometric mean (1 sigma), capped against needles.
    largest = np.sort(np.exp(log_scales) * scale, 1)[:, 1:]
    radii = np.minimum(np.sqrt(largest.prod(1)), max_radius_m)
    keep = opacity >= min_opacity
    points = enu_to_utm(enu[keep], georeference, utm_epsg)
    predicted = splat_dsm(points, radii[keep], truth)
    return predicted, {
        "dsm": f"max centre height over each splat's footprint disc (1 sigma, capped at {max_radius_m} m), opacity >= {min_opacity}",
        "splats_used": int(keep.sum()),
        "deviation": "Skyfall builds its DSM from rendered depth; this uses splat footprints, so registered MAE is not directly comparable to published numbers",
    }


def finish(export_dir, output, manifest, georeference, truth, truth_dir, aoi, utm_epsg, predicted, source):
    from pipeline.phase2 import digest

    half = 125.0
    block_utm = enu_to_utm(np.array([[-half, -half, 0], [half, half, 0]]), georeference, utm_epsg)
    size, resolution = truth["dsm"].shape[0], truth["resolution"]
    east = truth["west"] + (np.arange(size) + 0.5) * resolution
    north = truth["north"] - (np.arange(size) + 0.5) * resolution
    extent = ((east[None, :] >= block_utm[0, 0]) & (east[None, :] <= block_utm[1, 0])
              & (north[:, None] >= block_utm[0, 1]) & (north[:, None] <= block_utm[1, 1]))

    if output.exists():
        raise ValueError("Use a new validation directory; remove the previous one first")
    output.mkdir(parents=True)
    truth_path, predicted_path, registered_path = (
        output / "truth-dsm.tif", output / "splat-dsm.tif", output / "splat-dsm-registered.tif")
    write_raster(truth_path, truth["dsm"])
    write_raster(predicted_path, predicted)
    dx, dy, offset = register(truth_path, predicted_path, registered_path)
    import rasterio
    with rasterio.open(registered_path) as dataset:
        registered = dataset.read(1).astype(np.float64)
    preview(output / "dsm-comparison.png", truth["dsm"], predicted, registered)

    classes = truth["classes"]
    report = {
        "schema_version": 1,
        "status": "measured",
        "export_manifest_sha256": digest(export_dir / "export-manifest.json"),
        "fused_ply_sha256": manifest["fused_ply"]["sha256"],
        "reference": {
            "aoi": aoi,
            "source": "DFC2019 Track3-Truth (HSIP airborne lidar, ANPS about 0.8 m), via the Sat-NeRF EarthVision2022 dataset.zip",
            "files_sha256": {p.name: digest(p) for p in sorted(Path(truth_dir).iterdir()) if p.is_file()},
            "grid": {"utm_epsg": utm_epsg, "west": truth["west"], "north": truth["north"],
                     "resolution_m": resolution, "size": size},
            "heights": "WGS84 ellipsoidal (ground median about -17 m; orthometric would be about +12 m here)",
            "acquisition_date": "unknown (HSIP U.S. Cities lidar; not stated in the DFC2019/US3D papers)",
        },
        "method": {
            "placement": "fused export placed by model_to_enu then enu_to_ecef, converted to UTM with ellipsoidal heights",
            **source,
            "registration": "Skyfall-GS/Sat-NeRF dsmr: recursive NCC shift, z offset, no z scaling",
            "evaluated_extent": "250 m block (ENU square) inside the truth grid; CLS 65 (unlabelled) and water excluded from all_labelled",
        },
        "unregistered": by_class(predicted, truth["dsm"], classes, extent),
        "registration": {
            "dx_px": dx, "dy_px": dy, "z_offset_m": offset,
            "splats_offset_from_lidar_m": offset_from_truth(dx, dy, offset, resolution),
        },
        "registered": by_class(registered, truth["dsm"], classes, extent),
        "outputs": {"comparison_png": "dsm-comparison.png (truth | splat DSM | unregistered error | registered error; error scale +/-5 m, red = splats too high)"},
    }
    (output / "dsm-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--truth", required=True, help="Directory with <AOI>_DSM.tif/.txt and <AOI>_CLS.tif")
    parser.add_argument("--output", required=True)
    parser.add_argument("--aoi", default="JAX_068")
    parser.add_argument("--points", help="depth-points.npz from pipeline.render_depth (default: splat footprints)")
    args = parser.parse_args()
    report = validate(args.export, args.truth, args.output, args.aoi, depth_points=args.points)
    print(json.dumps({k: report[k] for k in ("unregistered", "registration", "registered")}, indent=2))


if __name__ == "__main__":
    main()
