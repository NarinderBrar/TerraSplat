"""Align MSI color by RPC to the native PAN grid and inject measured PAN detail."""

import hashlib
import json
from pathlib import Path


def prepare_rgb(pan_path, msi_path, output, height_range):
    import numpy as np
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import RPCTransformer
    from scipy.ndimage import map_coordinates

    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("RGB output already exists")
    with rasterio.open(pan_path) as pan, rasterio.open(msi_path) as msi:
        if pan.count != 1 or pan.rpcs is None or msi.rpcs is None:
            raise ValueError("PAN and MSI RPC models are required")
        indices = [
            msi.colorinterp.index(color) + 1
            for color in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        ]
        # Compare full acquisition timestamps, not merely matching calendar dates.
        timestamp = pan.tags().get("NITF_IDATIM")
        if not timestamp or timestamp != msi.tags().get("NITF_IDATIM"):
            raise ValueError("PAN/MSI capture timestamps must match")
        height = (pan.rpcs.height_off + msi.rpcs.height_off) / 2
        low, high = height_range
        if not low <= height <= high:
            raise ValueError(
                "RPC alignment reference lies outside the crop height bracket"
            )
        rows = np.linspace(0, pan.height - 1, 7)
        cols = np.linspace(0, pan.width - 1, 7)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr = rr.ravel()
        cc = cc.ravel()
        with RPCTransformer(pan.rpcs) as p, RPCTransformer(msi.rpcs) as m:
            projected = []
            for level in (low, height, high):
                lon, lat = p.xy(rr, cc, zs=np.full(rr.shape, level))
                r, c = m.rowcol(lon, lat, zs=np.full(rr.shape, level), op=lambda v: v)
                projected.append(np.column_stack((r, c)))
            parallax = max(
                float(np.linalg.norm(x - projected[1], axis=1).max()) for x in projected
            )
            if parallax > 0.5:
                raise ValueError(
                    f"Height-dependent color misalignment exceeds 0.5 MSI pixels: {parallax}"
                )
            rgb = np.empty((3, pan.height, pan.width), dtype=np.float32)
            source = msi.read(indices).astype(np.float32)
            for top in range(0, pan.height, 128):
                bottom = min(pan.height, top + 128)
                r, c = np.meshgrid(
                    np.arange(top, bottom), np.arange(pan.width), indexing="ij"
                )
                lon, lat = p.xy(r.ravel(), c.ravel(), zs=np.full(r.size, height))
                mr, mc = m.rowcol(lon, lat, zs=np.full(r.size, height), op=lambda v: v)
                coordinates = np.vstack((np.asarray(mr) - 0.5, np.asarray(mc) - 0.5))
                for band in range(3):
                    rgb[band, top:bottom] = map_coordinates(
                        source[band], coordinates, order=1, mode="constant", cval=np.nan
                    ).reshape(r.shape)
        pan_data = pan.read(1).astype(np.float32)
        valid = np.isfinite(rgb).all(axis=0) & (rgb > 0).all(axis=0) & (pan_data > 0)
        fraction = float(valid.mean())
        if fraction < 0.9:
            raise ValueError(
                f"Less than 90% of the PAN crop has valid aligned color: {fraction}"
            )
        intensity = np.mean(rgb, axis=0)
        pan_std = float(pan_data[valid].std())
        color_std = float(intensity[valid].std())
        if pan_std <= 0 or color_std <= 0:
            raise ValueError("Color/PAN dynamic range is empty")
        matched = (pan_data - float(pan_data[valid].mean())) * (
            color_std / pan_std
        ) + float(intensity[valid].mean())
        ratio = np.clip(matched / np.maximum(intensity, 1), 0, 4)
        injected = rgb * ratio[None]
        lo, hi = np.percentile(injected[:, valid], [1, 99])
        if hi <= lo:
            raise ValueError("RGB stretch has no dynamic range")
        byte = np.clip(np.nan_to_num((injected - lo) * 255 / (hi - lo)), 0, 255).astype(
            "uint8"
        )
        byte[:, ~valid] = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            output,
            "w",
            driver="GTiff",
            width=pan.width,
            height=pan.height,
            count=3,
            dtype="uint8",
            transform=pan.transform,
            crs=pan.crs,
            compress="deflate",
            tiled=True,
        ) as dst:
            dst.write(byte)
            dst.write_mask(valid.astype("uint8") * 255)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            dst.update_tags(NITF_IDATIM=timestamp)
            dst.update_tags(ns="RPC", **pan.rpcs.to_gdal())
        with rasterio.open(output) as result:
            if result.rpcs.to_gdal() != pan.rpcs.to_gdal() or not np.array_equal(
                result.read(), byte
            ):
                raise ValueError("RGB write changed pixels or PAN RPC geometry")
        report = {
            "status": "prepared",
            "capture_time": timestamp,
            "rgb_band_indices": indices,
            "alignment_reference_ellipsoidal_height_m": height,
            "height_range_m": height_range,
            "max_height_parallax_msi_pixels": parallax,
            "valid_fraction": fraction,
            "method": "RPC-aligned MSI RGB, mean/std PAN intensity match, ratio injection clipped to [0,4], shared 1–99 percentile byte stretch",
            "stretch_bounds": [float(lo), float(hi)],
            "shape": list(byte.shape),
            "source_sha256": {
                str(pan_path): hashlib.sha256(Path(pan_path).read_bytes()).hexdigest(),
                str(msi_path): hashlib.sha256(Path(msi_path).read_bytes()).hexdigest(),
            },
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "limitations": "Relative display color, not calibrated radiance or a validated color-accuracy product. RPC consistency does not establish image-based registration accuracy; inspect the crops before SfM.",
        }
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        return report
