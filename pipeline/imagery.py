"""Catalogue CORE3D imagery and preserve RPC camera metadata in pixel crops."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://spacenet-dataset.s3.amazonaws.com/"
PREFIX = "Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/Jacksonville/WV3/"


def catalogue():
    objects = []
    token = None
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        query = {"list-type": "2", "prefix": PREFIX}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(query), timeout=30) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s:Contents", ns):
            key = item.findtext("s:Key", namespaces=ns)
            if not key.endswith(".NTF"):
                continue
            objects.append({"key": key, "uri": BUCKET + urllib.parse.quote(key, safe="/"),
                            "bytes": int(item.findtext("s:Size", namespaces=ns)),
                            "etag": item.findtext("s:ETag", namespaces=ns).strip('"'),
                            "last_modified": item.findtext("s:LastModified", namespaces=ns),
                            "sensor_product": "PAN" if "/PAN/" in key else "MSI"})
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=ns)
        if not token:
            raise ValueError("Truncated S3 listing has no continuation token")
    return {"source": BUCKET, "prefix": PREFIX, "objects": objects,
            "note": "Candidate source catalogue only; acquisition dates, cloud masks and epoch grouping still require preflight."}


def shifted_rpc(rpc, window):
    """Pixel-only crop: coefficients/scales stay fixed, line/sample origins move."""
    return type(rpc)(**{**rpc.to_dict(), "line_off": rpc.line_off - window.row_off,
                        "samp_off": rpc.samp_off - window.col_off})


def block_perimeter(config):
    from rasterio.warp import transform
    lon, lat = config["longitude"], config["latitude"]
    side, halo = config["side_m"], config["halo_m"]
    if not all(math.isfinite(v) for v in (lon, lat, side, halo)) or not (-180 <= lon < 180 and -80 <= lat <= 84):
        raise ValueError("Block must have finite coordinates in the UTM latitude range")
    if side <= 0 or halo < 0 or side + 2 * halo > 2000:
        raise ValueError("Block must be positive and no wider than 2 km including halo")
    zone = int((lon + 180) // 6) + 1
    crs = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    x, y = transform("EPSG:4326", crs, [lon], [lat])
    half = side / 2 + halo
    # Densify each edge instead of assuming nonlinear RPC projections preserve straight edges.
    points = []
    for step in range(17):
        t = -half + 2 * half * step / 16
        points.extend([(x[0] + t, y[0] - half), (x[0] + t, y[0] + half),
                       (x[0] - half, y[0] + t), (x[0] + half, y[0] + t)])
    lons, lats = transform(crs, "EPSG:4326", *zip(*points))
    return lons, lats, crs


def rpc_window(rpc, width, height, block):
    from rasterio.transform import RPCTransformer
    from rasterio.windows import Window
    low, high = block["ellipsoidal_height_range_m"]
    if not all(math.isfinite(v) for v in (low, high)) or low >= high:
        raise ValueError("An increasing finite ellipsoidal-height range is required")
    if low < rpc.height_off - rpc.height_scale or high > rpc.height_off + rpc.height_scale:
        raise ValueError("Requested height bracket exceeds the RPC normalization range")
    lons, lats, _ = block_perimeter(block)
    rows, cols = [], []
    with RPCTransformer(rpc) as transformer:
        for level in (low, (low + high) / 2, high):
            r, c = transformer.rowcol(lons, lats, zs=[level] * len(lons), op=lambda v: v)
            rows.extend(r); cols.extend(c)
    if not all(math.isfinite(float(v)) for v in rows + cols):
        raise ValueError("RPC projection produced nonfinite image coordinates")
    padding = 8
    left, top = math.floor(min(cols)) - padding, math.floor(min(rows)) - padding
    right, bottom = math.ceil(max(cols)) + padding, math.ceil(max(rows)) + padding
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError("The image does not fully cover the requested block and height bracket")
    return Window(left, top, right - left, bottom - top)


def crop(source, output, block, max_bytes=64 * 1024**2):
    import numpy as np
    import rasterio
    from rasterio.transform import RPCTransformer
    started = time.monotonic()
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Output already exists; choose a new crop path")
    remote = source.startswith("https://")
    source_identity = None
    if remote:
        def identity():
            with urllib.request.urlopen(urllib.request.Request(source, method="HEAD"), timeout=30) as response:
                return {key: response.headers.get(key) for key in ("ETag", "Last-Modified", "Content-Length")}
        source_identity = identity()
    else:
        path = Path(source)
        with path.open("rb") as stream:
            source_identity = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30",
               GDAL_HTTP_CONNECTTIMEOUT="10", GDAL_HTTP_MAX_RETRY="0")
    source_path = "/vsicurl/" + source if remote else source
    partial = output.with_suffix(".partial.tif")
    if partial.exists():
        raise ValueError("A partial crop already exists; inspect it before retrying")
    with rasterio.Env(**env), rasterio.open(source_path) as src:
        if src.rpcs is None:
            raise ValueError("Source has no RPC camera metadata")
        window = rpc_window(src.rpcs, src.width, src.height, block)
        touched_bytes = 0
        for dtype, (bh, bw) in zip(src.dtypes, src.block_shapes):
            blocks_x = math.ceil((window.col_off + window.width) / bw) - math.floor(window.col_off / bw)
            blocks_y = math.ceil((window.row_off + window.height) / bh) - math.floor(window.row_off / bh)
            touched_bytes += blocks_x * blocks_y * bw * bh * np.dtype(dtype).itemsize
        if touched_bytes > max_bytes:
            raise ValueError(f"Crop touches {touched_bytes} uncompressed block bytes, above the {max_bytes} limit")
        data = src.read(window=window)
        rpc = shifted_rpc(src.rpcs, window)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Sensor pixels retain the source's approximate affine plus their authoritative RPC model.
        profile = {"driver": "GTiff", "width": int(window.width), "height": int(window.height),
                   "count": src.count, "dtype": src.dtypes[0], "transform": src.window_transform(window),
                   "crs": src.crs, "compress": "deflate", "tiled": True, "nodata": src.nodata}
        try:
            with rasterio.open(partial, "w", **profile) as dst:
                dst.write(data)
                dst.update_tags(ns="RPC", **rpc.to_gdal())
            with rasterio.open(partial) as reread:
                if not np.array_equal(reread.read(), data):
                    raise ValueError("Written crop pixels differ from source window")
                if reread.rpcs is None:
                    raise ValueError("Written crop lost RPC metadata")
                errors = []
                lons, lats, _ = block_perimeter(block)
                with RPCTransformer(src.rpcs) as original, RPCTransformer(reread.rpcs) as cropped:
                    for level in block["ellipsoidal_height_range_m"]:
                        r1, c1 = original.rowcol(lons, lats, zs=[level] * len(lons), op=lambda v: v)
                        r2, c2 = cropped.rowcol(lons, lats, zs=[level] * len(lons), op=lambda v: v)
                        errors.extend(abs(float(a) - float(b) - window.row_off) for a, b in zip(r1, r2))
                        errors.extend(abs(float(a) - float(b) - window.col_off) for a, b in zip(c1, c2))
                if max(errors) > 1e-6:
                    raise ValueError("RPC crop coordinate continuity check failed")
            if remote and identity() != source_identity:
                raise ValueError("Remote source changed while cropping")
            with partial.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            report = {"source": source, "source_identity": source_identity, "block": block,
                      "source_shape": [src.height, src.width], "output_shape": list(data.shape),
                      "window": {"col_off": window.col_off, "row_off": window.row_off, "width": window.width, "height": window.height},
                      "estimated_uncompressed_block_bytes": touched_bytes,
                      "rpc_pixel_error_max": max(errors), "rpc": rpc.to_gdal(), "sha256": checksum,
                      "elapsed_seconds": time.monotonic() - started,
                      "source_metadata": {key: value for key, value in src.tags().items() if key in ("NITF_IDATIM", "NITF_IID1", "NITF_ICAT", "NITF_IREP")},
                      "gdal": rasterio.__gdal_version__, "rasterio": rasterio.__version__,
                      "georeference_note": "Sensor crop with preserved RPCs, not orthorectified or reconstructed; affine is approximate.",
                      "limits_note": "Block-byte bound estimates decompressed raster work; it is not a measured HTTP transfer cap."}
            output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            partial.rename(output)
            return report
        finally:
            partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("catalogue")
    listing.add_argument("--output", required=True, type=Path)
    clipping = commands.add_parser("crop")
    clipping.add_argument("source")
    clipping.add_argument("--timeout-seconds", type=float, default=120)
    clipping.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    clipping.add_argument("--block", required=True, type=Path)
    clipping.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "catalogue":
        report = catalogue()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Catalogued {len(report['objects'])} NITF images: {args.output}")
    elif not args._worker:
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            parser.error("timeout-seconds must be positive and finite")
        try:
            result = subprocess.run([sys.executable, "-u", "-m", "pipeline.imagery", *sys.argv[1:], "--_worker"], timeout=args.timeout_seconds)
            return result.returncode
        except subprocess.TimeoutExpired:
            parser.exit(2, f"Crop exceeded {args.timeout_seconds:g} seconds; source was not accepted. Inspect any partial output before retrying.\n")
    else:
        report = crop(args.source, args.output, json.loads(args.block.read_text()))
        print(f"Cropped {report['output_shape']}; RPC error {report['rpc_pixel_error_max']:.2g} px: {args.output}")


if __name__ == "__main__":
    sys.exit(main())
