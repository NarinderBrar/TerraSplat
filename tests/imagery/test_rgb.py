import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.rpc import RPC
from rasterio.transform import from_origin

from pipeline.rgb import prepare_rgb


@pytest.fixture
def pair(tmp_path):
    poly = [0.0] * 20
    den = poly.copy()
    den[0] = 1
    row = poly.copy()
    row[2] = -1
    col = poly.copy()
    col[1] = 1
    paths = []
    for band, size in (("pan", 96), ("msi", 48)):
        rr, cc = np.indices((size, size))
        field = 100 + rr + cc
        pixels = (
            field[None] * 2
            if band == "pan"
            else np.stack(
                [field, field, field * 2, field, field * 3, field, field, field]
            )
        )
        rpc = RPC(
            height_off=0,
            height_scale=500,
            lat_off=30,
            lat_scale=0.01,
            long_off=-81,
            long_scale=0.01,
            line_off=(size - 1) / 2,
            line_scale=size / 2,
            samp_off=(size - 1) / 2,
            samp_scale=size / 2,
            line_num_coeff=row,
            line_den_coeff=den,
            samp_num_coeff=col,
            samp_den_coeff=den,
        )
        path = tmp_path / f"{band}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=pixels.shape[0],
            dtype="uint16",
            transform=from_origin(-81.01, 30.01, 0.02 / size, 0.02 / size),
            crs="EPSG:4326",
        ) as ds:
            ds.write(pixels.astype("uint16"))
            ds.update_tags(ns="RPC", **rpc.to_gdal())
            ds.update_tags(NITF_IDATIM="20150101120000")
            if band == "msi":
                ds.colorinterp = (
                    ColorInterp.undefined,
                    ColorInterp.blue,
                    ColorInterp.green,
                    ColorInterp.undefined,
                    ColorInterp.red,
                    ColorInterp.undefined,
                    ColorInterp.undefined,
                    ColorInterp.undefined,
                )
        paths.append(path)
    return paths


def test_rgb_uses_explicit_band_mapping_and_preserves_pan_camera(pair, tmp_path):
    pan, msi = pair
    output = tmp_path / "rgb.tif"
    report = prepare_rgb(pan, msi, output, [-50, 300])
    assert report["rgb_band_indices"] == [5, 3, 2]
    assert report["max_height_parallax_msi_pixels"] < 1e-6
    with rasterio.open(output) as dst, rasterio.open(pan) as original:
        data = dst.read()
        mask = dst.dataset_mask() > 0
        assert dst.count == 3 and dst.dtypes == ("uint8",) * 3
        assert dst.rpcs.to_gdal() == original.rpcs.to_gdal()
        assert data[0, mask].mean() > data[1, mask].mean() > data[2, mask].mean()


def test_rgb_rejects_an_unmatched_acquisition(pair, tmp_path):
    pan, msi = pair
    with rasterio.open(msi, "r+") as ds:
        ds.update_tags(NITF_IDATIM="20150102120000")
    with pytest.raises(ValueError, match="timestamps must match"):
        prepare_rgb(pan, msi, tmp_path / "bad.tif", [-50, 300])


def test_rgb_rejects_height_reference_outside_crop_bracket(pair, tmp_path):
    with pytest.raises(ValueError, match="reference lies outside"):
        prepare_rgb(*pair, tmp_path / "bad.tif", [10, 300])
