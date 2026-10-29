"""Fetch only a selected CORE3D view set with explicit size and identity bounds."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(product, destination, timeout_seconds=300):
    destination = Path(destination)
    path = destination / Path(product["key"]).name
    evidence = path.with_suffix(".json")
    if path.exists():
        record = json.loads(evidence.read_text())
        if (
            any(record[k] != product[k] for k in ("uri", "bytes", "etag"))
            or file_hash(path) != record["sha256"]
        ):
            raise ValueError(f"Cached source does not match catalogue: {path}")
        return {
            **record,
            **{key: product[key] for key in ("role", "catalog_id") if key in product},
            "path": str(path),
            "reused": True,
        }
    if evidence.exists():
        raise ValueError(f"Source evidence exists without its image: {evidence}")
    partial = path.with_suffix(".partial.NTF")
    if partial.exists():
        raise ValueError(f"Inspect incomplete source before retrying: {partial}")
    started = time.monotonic()
    checksum = hashlib.sha256()
    count = 0
    with (
        urllib.request.urlopen(product["uri"], timeout=30) as response,
        partial.open("xb") as output,
    ):
        if (
            int(response.headers["Content-Length"]) != product["bytes"]
            or response.headers.get("ETag", "").strip('"') != product["etag"]
        ):
            raise ValueError("Remote object identity changed after selection")
        while chunk := response.read(1024**2):
            count += len(chunk)
            if count > product["bytes"] or time.monotonic() - started > timeout_seconds:
                raise ValueError(
                    "Source exceeded the declared size or download time budget"
                )
            checksum.update(chunk)
            output.write(chunk)
    if count != product["bytes"]:
        raise ValueError("Source download is incomplete")
    record = {
        **product,
        "sha256": checksum.hexdigest(),
        "downloaded_bytes": count,
        "elapsed_seconds": time.monotonic() - started,
    }
    evidence.write_text(json.dumps(record, indent=2) + "\n")
    partial.rename(path)
    print(f"Verified {path.name}: {count / 1024**2:.0f} MiB", flush=True)
    return {**record, "path": str(path), "reused": False}


def fetch_selection(selection, destination, max_total_bytes=8 * 1024**3):
    if selection.get("status") != "candidate":
        raise ValueError("A candidate acquisition selection is required")
    products = []
    for role in ("train", "holdout"):
        for view in selection[role]:
            for band in ("pan", "msi"):
                products.append(
                    {**view[band], "role": role, "catalog_id": view["catalog_id"]}
                )
    if len({p["uri"] for p in products}) != len(products):
        raise ValueError("Source set contains duplicate products")
    if not products or any(
        not isinstance(p["bytes"], int) or p["bytes"] <= 0 for p in products
    ):
        raise ValueError("Source sizes must be positive integers")
    if sum(p["bytes"] for p in products) > max_total_bytes or any(
        p["bytes"] > 1024**3 for p in products
    ):
        raise ValueError(
            f"Selected sources exceed the explicit {max_total_bytes / 1024**3:g} GiB total or 1 GiB per-image budget"
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    remaining = sum(
        p["bytes"] for p in products if not (destination / Path(p["key"]).name).exists()
    )
    if shutil.disk_usage(destination).free < remaining + 5 * 1024**3:
        raise ValueError("Insufficient disk space plus 5 GiB reserve")
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda p: fetch(p, destination), products))
    return {
        "status": "downloaded",
        "products": records,
        "total_source_bytes": sum(p["bytes"] for p in products),
        "note": "Source snapshots verified against selected ETags/sizes and local SHA-256; no claims about cloud or RGB quality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument(
        "--destination", type=Path, default=Path("artifacts/imagery/raw")
    )
    parser.add_argument("--max-total-gib", type=float, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fetch_selection(
        json.loads(args.selection.read_text()),
        args.destination,
        max_total_bytes=int(args.max_total_gib * 1024**3),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(report['products'])} sources: {args.output}")


if __name__ == "__main__":
    main()
