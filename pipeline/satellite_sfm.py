"""Use pinned SatelliteSfM code to initialize ENU cameras from selected RPCs."""

import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

from pipeline.imagery import block_perimeter, shifted_rpc

COMMIT = "908c61937f6a16d6e1d50e98898db997e5e257bd"
CONTAINER = "ghcr.io/jayin92/satellitesfm@sha256:3c64997c9b2b4de5cdcad6d24972768423845fd7691015af1e4771c84fade9a5"


def source_modules(source):
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if actual != COMMIT:
        raise ValueError(f"SatelliteSfM must be pinned to {COMMIT}")
    dirty = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "preprocess"], cwd=source, text=True
    )
    if dirty:
        raise ValueError("SatelliteSfM calibration code has unrecorded changes")
    sys.path.insert(0, str(Path(source).resolve()))
    from preprocess.approximate_rpc_locally import approximate_rpc_locally
    from preprocess.rpc_model import RPCModel
    from preprocess.coordinate_system import latlonalt_to_enu

    return approximate_rpc_locally, RPCModel, latlonalt_to_enu


def metadata(record):
    from rasterio.rpc import RPC
    from rasterio.windows import Window

    window = Window(**record["coverage"]["window"])
    rpc = shifted_rpc(RPC.from_gdal(record["rpc"]), window)
    names = {
        "lonOff": "long_off",
        "lonScale": "long_scale",
        "latOff": "lat_off",
        "latScale": "lat_scale",
        "altOff": "height_off",
        "altScale": "height_scale",
        "rowOff": "line_off",
        "rowScale": "line_scale",
        "colOff": "samp_off",
        "colScale": "samp_scale",
        "rowNum": "line_num_coeff",
        "rowDen": "line_den_coeff",
        "colNum": "samp_num_coeff",
        "colDen": "samp_den_coeff",
    }
    date = datetime.fromisoformat(record["capture_time"].replace("Z", "+00:00"))
    return {
        "width": int(window.width),
        "height": int(window.height),
        "capture_date": [
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second,
        ],
        "rpc": {name: getattr(rpc, attribute) for name, attribute in names.items()},
    }


def independent_error(meta, K, W2C, bounds, observer, rpc_class, to_enu):
    import numpy as np

    # Offset samples do not coincide with the fitting grid (15x15x7).
    samples = np.array(
        list(
            itertools.product(
                *[
                    np.linspace(low, high, count + 2)[1:-1]
                    for (low, high), count in zip(bounds, (16, 16, 8))
                ]
            )
        )
    )
    lat, lon, alt = samples.T
    target_col, target_row = rpc_class(meta).projection(lat, lon, alt)
    e, n, u = to_enu(lat, lon, alt, *observer)
    xyz = np.column_stack((e, n, u, np.ones_like(e)))
    projected = xyz @ (K @ W2C)[:3, :].T
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0):
        raise ValueError("Initialized camera has nonfinite or behind-camera samples")
    predicted = projected[:, :2] / projected[:, 2:3]
    valid = (
        (target_col >= 0)
        & (target_row >= 0)
        & (target_col < meta["width"])
        & (target_row < meta["height"])
    )
    if valid.mean() < 0.95:
        raise ValueError(
            "Fewer than 95% of independent samples lie inside the selected crop"
        )
    errors = np.linalg.norm(
        predicted[valid] - np.column_stack((target_col, target_row))[valid], axis=1
    )
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite RPC approximation error")
    return {
        "sample_count": int(valid.sum()),
        "max_pixels": float(errors.max()),
        "p95_pixels": float(np.percentile(errors, 95)),
        "median_pixels": float(np.median(errors)),
    }


def geographic_frame(block):
    lons, lats, _ = block_perimeter(block)
    low, high = block["ellipsoidal_height_range_m"]
    bounds = ([min(lats), max(lats)], [min(lons), max(lons)], [low, high])
    observer = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2, low - 20]
    return bounds, observer


def calibrate(selection, source, output, max_error_pixels=1.0):
    import numpy as np

    if selection.get("status") != "candidate":
        raise ValueError("A qualifying candidate selection is required")
    if max_error_pixels <= 0 or not np.isfinite(max_error_pixels):
        raise ValueError("Calibration tolerance must be positive and finite")
    if Path(output).exists():
        raise ValueError("Calibration output already exists")
    fit, rpc_class, to_enu = source_modules(source)
    block = selection["block"]
    bounds, observer = geographic_frame(block)
    output = Path(output)
    for name in ("cameras", "metas"):
        (output / name).mkdir(parents=True)
    (output / "enu_observer_latlonalt.json").write_text(json.dumps(observer) + "\n")
    bbox = dict(zip(("lat_minmax", "lon_minmax", "alt_minmax"), bounds))
    (output / "latlonalt_bbx.json").write_text(json.dumps(bbox, indent=2) + "\n")
    corners = np.array(list(itertools.product(*bounds))).T
    enu = to_enu(*corners, *observer)
    enu_bbox = {
        f"{name}_minmax": [
            float(np.min(axis)) - (10 if name == "u" else 0),
            float(np.max(axis)) + (10 if name == "u" else 0),
        ]
        for name, axis in zip(("e", "n", "u"), enu)
    }
    (output / "enu_bbx.json").write_text(json.dumps(enu_bbox, indent=2) + "\n")
    report = {
        "stage": "rpc_camera_initialization_only",
        "source_commit": COMMIT,
        "container": CONTAINER,
        "selection_sha256": hashlib.sha256(
            json.dumps(selection, sort_keys=True).encode()
        ).hexdigest(),
        "observer_latlonalt": observer,
        "observer_note": "Exact numerical ENU origin defined from the crop bounds; altitude is lower bracket minus 20 m, not a ground-height measurement.",
        "max_allowed_error_pixels": max_error_pixels,
        "views": [],
        "pending": [
            "local selected images and block cloud review",
            "matched MSI and validated RGB alignment",
            "feature extraction and bundle adjustment",
            "skew correction and Skyfall conversion",
        ],
    }
    for role in ("train", "holdout"):
        for record in selection[role]:
            name = record["catalog_id"]
            if Path(name).name != name:
                raise ValueError("Invalid catalogue identifier")
            meta = metadata(record)
            K, W2C = fit(meta, *bounds, *observer, lat_N=15, lon_N=15, alt_N=7)
            error = independent_error(meta, K, W2C, bounds, observer, rpc_class, to_enu)
            camera = {
                "K": K.reshape(-1).tolist(),
                "W2C": W2C.reshape(-1).tolist(),
                "img_size": [meta["width"], meta["height"]],
            }
            (output / "cameras" / f"{name}.json").write_text(
                json.dumps(camera, indent=2) + "\n"
            )
            (output / "metas" / f"{name}.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )
            report["views"].append(
                {
                    "catalog_id": name,
                    "role": role,
                    "rpc_approximation": error,
                    "skew_pixels": float(K[0, 1]),
                    "metadata_sha256": record["metadata_sha256"],
                }
            )
    report["status"] = (
        "passed"
        if all(
            v["rpc_approximation"]["max_pixels"] <= max_error_pixels
            for v in report["views"]
        )
        else "blocked"
    )
    (output / "calibration-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--source", type=Path, default=Path(".external/satellitesfm"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = calibrate(json.loads(args.selection.read_text()), args.source, args.output)
    print(f"{report['status']}: {args.output / 'calibration-report.json'}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
