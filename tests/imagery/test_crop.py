import json
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.rpc import RPC
from rasterio.transform import from_origin
from rasterio.windows import Window

from pipeline.imagery import crop, rpc_window


@pytest.fixture
def sensor(tmp_path):
    coeff = [0.0] * 20
    denominator = coeff.copy(); denominator[0] = 1
    line = coeff.copy(); line[2] = -1; line[3] = 0.1
    sample = coeff.copy(); sample[1] = 1; sample[3] = -0.05
    rpc = RPC(height_off=0, height_scale=500, lat_off=30.3488, lat_scale=0.01,
              long_off=-81.6638, long_scale=0.01, line_off=255, samp_off=255,
              line_scale=255, samp_scale=255, line_num_coeff=line, line_den_coeff=denominator,
              samp_num_coeff=sample, samp_den_coeff=denominator)
    pixels = np.arange(512 * 512, dtype='uint16').reshape(1, 512, 512)
    path = tmp_path / 'sensor.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=512, width=512, count=1, dtype='uint16',
                       transform=from_origin(-81.67, 30.35, 0.00001, 0.00001), crs='EPSG:4326', tiled=True) as ds:
        ds.write(pixels)
        ds.update_tags(ns='RPC', **rpc.to_gdal())
    block = dict(longitude=-81.6638, latitude=30.3488, side_m=250, halo_m=20, ellipsoidal_height_range_m=[-50, 300])
    return path, rpc, block


def test_crop_preserves_pixels_and_height_dependent_rpc_geometry(sensor, tmp_path):
    path, rpc, block = sensor
    output = tmp_path / 'crop.tif'
    report = crop(str(path), output, block)
    with rasterio.open(path) as source, rasterio.open(output) as cropped:
        window = Window(**report['window'])
        assert np.array_equal(cropped.read(), source.read(window=window))
        assert cropped.rpcs.line_off == rpc.line_off - window.row_off
        assert cropped.rpcs.samp_off == rpc.samp_off - window.col_off
        assert cropped.rpcs.line_num_coeff == rpc.line_num_coeff
        assert cropped.rpcs.height_scale == rpc.height_scale
    assert report['rpc_pixel_error_max'] < 1e-6
    assert json.loads(output.with_suffix('.json').read_text())['sha256'] == report['sha256']


def test_oversized_read_is_rejected_before_output(sensor, tmp_path):
    path, _, block = sensor
    output = tmp_path / 'crop.tif'
    with pytest.raises(ValueError, match='above the'):
        crop(str(path), output, block, max_bytes=1)
    assert not output.exists()


def test_uncovered_block_is_not_silently_clipped(sensor):
    _, rpc, block = sensor
    block['longitude'] += 0.1
    with pytest.raises(ValueError, match='does not fully cover'):
        rpc_window(rpc, 512, 512, block)


def test_height_extrapolation_is_rejected(sensor):
    _, rpc, block = sensor
    block['ellipsoidal_height_range_m'] = [-501, 300]
    with pytest.raises(ValueError, match='normalization range'):
        rpc_window(rpc, 512, 512, block)


def test_existing_output_is_preserved(sensor, tmp_path):
    path, _, block = sensor
    output = tmp_path / 'crop.tif'; output.write_bytes(b'keep me')
    with pytest.raises(ValueError, match='already exists'):
        crop(str(path), output, block)
    assert output.read_bytes() == b'keep me'


def test_missing_rpc_is_rejected(sensor, tmp_path):
    path, _, block = sensor
    bare = tmp_path / 'bare.tif'
    with rasterio.open(path) as src, rasterio.open(bare, 'w', **src.profile) as dst:
        dst.write(src.read())
    with pytest.raises(ValueError, match='no RPC'):
        crop(str(bare), tmp_path / 'crop.tif', block)


def test_cli_terminates_worker_when_crop_deadline_expires(sensor, tmp_path):
    path, _, block = sensor
    config = tmp_path / "block.json"; config.write_text(json.dumps(block))
    output = tmp_path / "deadline.tif"
    result = subprocess.run([sys.executable, "-m", "pipeline.imagery", "crop", str(path),
                             "--block", str(config), "--output", str(output), "--timeout-seconds", "0.000001"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "source was not accepted" in result.stderr
    assert not output.exists()
