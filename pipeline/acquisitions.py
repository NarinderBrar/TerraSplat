"""Inspect bounded CORE3D sidecars and choose a deterministic multi-view epoch."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import tarfile
import urllib.request

from pipeline.imagery import rpc_window

MAX_ARCHIVE_BYTES = 4 * 1024**2


def assignments(text):
    # IMD/RPB group markers lack semicolons; exclude them before parsing values.
    text = re.sub(r"(?m)^\s*(BEGIN_GROUP|END_GROUP)\s*=.*$", "", text)
    return {
        key: value.strip().strip('"')
        for key, value in re.findall(r"(?m)^\s*(\w+)\s*=\s*([^;]+);", text)
    }


def parse_sidecars(archive):
    from rasterio.rpc import RPC

    with tarfile.open(fileobj=io.BytesIO(archive)) as package:
        selected = {}
        for extension in (".IMD", ".RPB"):
            matches = [
                m
                for m in package.getmembers()
                if m.isfile() and m.name.endswith(extension) and "_PAN/" in m.name
            ]
            if len(matches) != 1 or matches[0].size > 128 * 1024:
                raise ValueError(f"Expected one small PAN {extension} sidecar")
            selected[extension] = assignments(
                package.extractfile(matches[0]).read().decode("utf-8")
            )
    imd, rpb = selected[".IMD"], selected[".RPB"]
    if imd["satId"] != "WV03" or imd["bandId"] != "P" or rpb["SpecId"] != "RPC00B":
        raise ValueError("Unexpected sensor, product or RPC coefficient order")
    names = {
        "height_off": "heightOffset",
        "height_scale": "heightScale",
        "lat_off": "latOffset",
        "lat_scale": "latScale",
        "long_off": "longOffset",
        "long_scale": "longScale",
        "line_off": "lineOffset",
        "line_scale": "lineScale",
        "samp_off": "sampOffset",
        "samp_scale": "sampScale",
        "err_bias": "errBias",
        "err_rand": "errRand",
    }
    values = {key: float(rpb[value]) for key, value in names.items()}
    for key, name in (
        ("line_num_coeff", "lineNumCoef"),
        ("line_den_coeff", "lineDenCoef"),
        ("samp_num_coeff", "sampNumCoef"),
        ("samp_den_coeff", "sampDenCoef"),
    ):
        values[key] = [float(v) for v in rpb[name].strip("()\n\t ").split(",")]
        if len(values[key]) != 20:
            raise ValueError("RPC polynomial requires 20 coefficients")
    if not all(
        math.isfinite(v)
        for value in values.values()
        for v in (value if isinstance(value, list) else [value])
    ):
        raise ValueError("RPC values must be finite")
    if any(
        values[name] <= 0
        for name in (
            "height_scale",
            "lat_scale",
            "long_scale",
            "line_scale",
            "samp_scale",
        )
    ):
        raise ValueError("RPC normalization scales must be positive")
    record = {
        "catalog_id": imd["CatId"],
        "product_id": imd["productOrderId"],
        "capture_time": imd["firstLineTime"],
        "width": int(imd["numColumns"]),
        "height": int(imd["numRows"]),
        "cloud_cover": float(imd["cloudCover"]),
        "sat_azimuth_deg": float(imd["meanSatAz"]),
        "sat_elevation_deg": float(imd["meanSatEl"]),
        "off_nadir_deg": float(imd["meanOffNadirViewAngle"]),
        "gsd_m": float(imd["meanProductGSD"]),
        "sun_elevation_deg": float(imd["meanSunEl"]),
        "rpc": RPC(**values).to_gdal(),
    }
    numeric = [v for v in record.values() if isinstance(v, (float, int))]
    if (
        not all(math.isfinite(v) for v in numeric)
        or not 0 <= record["cloud_cover"] <= 1
    ):
        raise ValueError("Invalid image quality metadata")
    datetime.fromisoformat(record["capture_time"].replace("Z", "+00:00"))
    return record


def inspect_product(product, all_products, cache, block):
    from rasterio.rpc import RPC

    uri = product["uri"].rsplit(".", 1)[0] + ".tar"
    archive = cache / (hashlib.sha256(uri.encode()).hexdigest() + ".tar")
    if archive.exists():
        data = archive.read_bytes()
    else:
        with urllib.request.urlopen(uri, timeout=30) as response:
            data = response.read(MAX_ARCHIVE_BYTES + 1)
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ValueError("Metadata archive exceeds the 4 MiB limit")
        archive.write_bytes(data)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Cached metadata archive exceeds the 4 MiB limit")
    record = parse_sidecars(data)
    if record["product_id"] not in product["key"]:
        raise ValueError("Sidecar product ID does not match its NITF")
    token = Path(product["key"]).name.split("-P1BS-")[0]
    matches = [
        p
        for p in all_products
        if p["sensor_product"] == "MSI"
        and Path(p["key"]).name.split("-M1BS-")[0] == token
        and record["product_id"] in p["key"]
    ]
    record.update(
        pan=product,
        msi=matches[0] if len(matches) == 1 else None,
        metadata_uri=uri,
        metadata_sha256=hashlib.sha256(data).hexdigest(),
    )
    try:
        window = rpc_window(
            RPC.from_gdal(record["rpc"]), record["width"], record["height"], block
        )
        record["coverage"] = {
            "status": "covered",
            "window": dict(
                col_off=window.col_off,
                row_off=window.row_off,
                width=window.width,
                height=window.height,
            ),
        }
    except ValueError as error:
        record["coverage"] = {"status": "rejected", "reason": str(error)}
    return record


def scan(catalogue, cache, block):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    products = catalogue["objects"]

    def inspect(product):
        try:
            return inspect_product(product, products, cache, block)
        except (OSError, ValueError, KeyError, tarfile.TarError) as error:
            return {"pan": product, "error": str(error)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(
            pool.map(inspect, [p for p in products if p["sensor_product"] == "PAN"])
        )
    return {
        "block": block,
        "records": records,
        "policy_note": "Whole-image cloud metadata is a screening hint, not block-level cloud validation.",
    }


def convergence(a, b):
    def vector(item):
        az, el = (
            math.radians(item["sat_azimuth_deg"]),
            math.radians(item["sat_elevation_deg"]),
        )
        return (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))

    dot = sum(x * y for x, y in zip(vector(a), vector(b)))
    return math.degrees(math.acos(max(-1, min(1, dot))))


def select(
    scan_report,
    max_span_days=7,
    max_cloud_cover=0.1,
    min_views=4,
    min_convergence_deg=10,
):
    if (
        not all(
            math.isfinite(v)
            for v in (max_span_days, max_cloud_cover, min_convergence_deg)
        )
        or min_views < 4
        or not 0 <= max_cloud_cover <= 1
        or max_span_days < 0
        or not 0 < min_convergence_deg <= 90
    ):
        raise ValueError("Invalid multi-view selection policy")
    usable = []
    for record in scan_report["records"]:
        if (
            "error" in record
            or record["coverage"]["status"] != "covered"
            or record["msi"] is None
        ):
            continue
        if (
            record["cloud_cover"] > max_cloud_cover
            or record["off_nadir_deg"] > 40
            or record["sun_elevation_deg"] < 15
        ):
            continue
        usable.append(record)
    usable.sort(key=lambda x: (x["capture_time"], x["catalog_id"]))
    choices = []
    for group in itertools.combinations(usable, min_views):
        if len({x["catalog_id"] for x in group}) != len(group):
            continue
        times = [
            datetime.fromisoformat(x["capture_time"].replace("Z", "+00:00"))
            for x in group
        ]
        span = (max(times) - min(times)).total_seconds() / 86400
        if span > max_span_days:
            continue
        # Leave one entire acquisition out before SfM/appearance fitting.
        train = list(group[:-1])
        holdout = group[-1]
        angle = max(convergence(a, b) for a, b in itertools.combinations(train, 2))
        if angle >= min_convergence_deg:
            choices.append(
                (span, -angle, tuple(x["catalog_id"] for x in group), train, holdout)
            )
    policy = dict(
        max_span_days=max_span_days,
        max_cloud_cover=max_cloud_cover,
        min_views=min_views,
        min_training_convergence_deg=min_convergence_deg,
    )
    if not choices:
        return {
            "status": "blocked",
            "policy": policy,
            "reason": "No group satisfies temporal, coverage, pair, cloud and training-angle gates",
            "usable_views": len(usable),
        }
    span, negative_angle, _, train, heldout = min(choices, key=lambda x: x[:3])
    return {
        "status": "candidate",
        "block": scan_report["block"],
        "policy": policy,
        "span_days": span,
        "training_max_convergence_deg": -negative_angle,
        "train": train,
        "holdout": [heldout],
        "limitations": "Candidate only: block cloud review, local imagery and RGB alignment remain required. Holdout is excluded from training SfM.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scanning = commands.add_parser("scan")
    scanning.add_argument("catalogue", type=Path)
    scanning.add_argument("--block", type=Path, required=True)
    scanning.add_argument("--cache", type=Path, default=Path(".cache/core3d-metadata"))
    choosing = commands.add_parser("select")
    choosing.add_argument("scan", type=Path)
    choosing.add_argument("--max-span-days", type=float, default=7)
    for command in (scanning, choosing):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "scan":
        report = scan(
            json.loads(args.catalogue.read_text()),
            args.cache,
            json.loads(args.block.read_text()),
        )
    else:
        report = select(
            json.loads(args.scan.read_text()), max_span_days=args.max_span_days
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report.get('status', 'scanned')}: {args.output}")
    return 2 if report.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
