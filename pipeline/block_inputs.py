"""Prepare selected crops and split RGB inputs without mixing the held-out view."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from pipeline.imagery import crop, shifted_rpc
from pipeline.rgb import prepare_rgb
from pipeline.sources import file_hash
from pipeline.satellite_sfm import metadata, geographic_frame


def prepare(selection, sources, output):
    import rasterio
    from rasterio.rpc import RPC
    from rasterio.windows import Window
    import numpy as np

    if selection.get("status") != "candidate" or sources.get("status") != "downloaded":
        raise ValueError("Selected and downloaded source manifests are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selection_hash = hashlib.sha256(
        json.dumps(selection, sort_keys=True).encode()
    ).hexdigest()
    index = {p["uri"]: p for p in sources["products"]}
    report_path = output / "inputs-report.json"
    if (
        report_path.exists()
        and json.loads(report_path.read_text())["selection_sha256"] != selection_hash
    ):
        raise ValueError("Output belongs to a different acquisition selection")
    report = {
        "status": "preparing",
        "selection_sha256": selection_hash,
        "views": [],
        "note": "RGB handoff and candidate QC stage; no SfM or independent geographic accuracy validation yet.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for role in ("train", "holdout"):
        for view in selection[role]:
            paths = {}
            for band in ("pan", "msi"):
                source = index[view[band]["uri"]]
                if any(source[k] != view[band][k] for k in ("etag", "bytes")):
                    raise ValueError(
                        "Downloaded source identity changed from selection"
                    )
                path = output / "crops" / f"{view['catalog_id']}-{band}.tif"
                evidence = path.with_suffix(".json")
                if path.exists() and evidence.exists():
                    saved = json.loads(evidence.read_text())
                    if (
                        saved["block"] != selection["block"]
                        or saved["source_identity"]["sha256"] != source["sha256"]
                        or file_hash(path) != saved["sha256"]
                    ):
                        raise ValueError(
                            "Cached crop evidence does not match selected input"
                        )
                else:
                    saved = crop(source["path"], path, selection["block"])
                    if saved["source_identity"]["sha256"] != source["sha256"]:
                        raise ValueError("Source file changed after download")
                paths[band] = path
            target = output / role / "images" / f"{view['catalog_id']}.tif"
            if target.exists() and target.with_suffix(".json").exists():
                rgb = json.loads(target.with_suffix(".json").read_text())
                expected = {str(path): file_hash(path) for path in paths.values()}
                if (
                    rgb["source_sha256"] != expected
                    or file_hash(target) != rgb["sha256"]
                ):
                    raise ValueError("Cached RGB differs from source crops")
            else:
                rgb = prepare_rgb(
                    paths["pan"],
                    paths["msi"],
                    target,
                    selection["block"]["ellipsoidal_height_range_m"],
                )
            expected = metadata(view)
            capture_time = datetime.fromisoformat(
                view["capture_time"].replace("Z", "+00:00")
            )
            if rgb["capture_time"] != capture_time.strftime("%Y%m%d%H%M%S"):
                raise ValueError(
                    "Actual sensor timestamp differs from its selected sidecar"
                )
            with rasterio.open(target) as result:
                actual = result.rpcs.to_dict()
                # The virtual calibration crop must match the actual camera crop, not just its dimensions.
                with rasterio.open(paths["pan"]) as sensor:
                    actual_pan = sensor.rpcs.to_dict()
                wanted = shifted_rpc(
                    RPC.from_gdal(view["rpc"]), Window(**view["coverage"]["window"])
                ).to_dict()
                if any(
                    not np.allclose(actual_pan[key], value, atol=1e-10, rtol=1e-10)
                    for key, value in wanted.items()
                    if value is not None
                ):
                    raise ValueError(
                        "Selected sidecar RPCs differ from the actual sensor camera"
                    )
                if (
                    actual != actual_pan
                    or result.width != expected["width"]
                    or result.height != expected["height"]
                ):
                    raise ValueError("RGB and initialized camera frames differ")
                if (
                    json.loads(paths["pan"].with_suffix(".json").read_text())["window"]
                    != view["coverage"]["window"]
                ):
                    raise ValueError(
                        "Actual and virtual RPC crop windows differ; recalibrate before continuing"
                    )
            report["views"].append(
                {
                    "catalog_id": view["catalog_id"],
                    "role": role,
                    "rgb_path": str(target),
                    "rgb_sha256": rgb["sha256"],
                    "valid_fraction": rgb["valid_fraction"],
                    "height_parallax_msi_pixels": rgb["max_height_parallax_msi_pixels"],
                }
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Prepared {role} {view['catalog_id']}", flush=True)
    bounds, observer = geographic_frame(selection["block"])
    for role in ("train", "holdout"):
        folder = output / role
        (folder / "latlonalt_bbx.json").write_text(
            json.dumps(
                dict(zip(("lat_minmax", "lon_minmax", "alt_minmax"), bounds)), indent=2
            )
            + "\n"
        )
        (folder / "enu_observer_latlonalt.json").write_text(json.dumps(observer) + "\n")
    report["status"] = "prepared_for_review"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(
        json.loads(args.selection.read_text()),
        json.loads(args.sources.read_text()),
        args.output,
    )
    print(report["status"])


if __name__ == "__main__":
    main()
