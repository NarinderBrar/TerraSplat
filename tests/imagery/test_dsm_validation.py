import numpy as np
import pytest

from pipeline.dsm_validation import (
    error_stats,
    read_ply,
    offset_from_truth,
    register,
    splat_dsm,
    write_raster,
)


def scene(size=128, seed=3):
    rng = np.random.default_rng(seed)
    dsm = np.zeros((size, size))
    for _ in range(25):
        r, c, h, w = rng.integers(0, size - 20, 2).tolist() + rng.integers(4, 18, 2).tolist()
        dsm[r:r + h, c:c + w] = rng.uniform(3, 30)
    return dsm + rng.normal(0, 0.05, dsm.shape)


def test_offset_reports_where_predicted_content_sits_relative_to_truth(tmp_path):
    truth = scene()
    # Predicted content sits 3 px east and 2 px north of the truth, and 1.5 m too high.
    predicted = np.full_like(truth, np.nan)
    predicted[:-2, 3:] = truth[2:, :-3] + 1.5
    write_raster(tmp_path / "truth.tif", truth)
    write_raster(tmp_path / "pred.tif", predicted)
    dx, dy, offset = register(tmp_path / "truth.tif", tmp_path / "pred.tif", tmp_path / "reg.tif")
    shift = offset_from_truth(dx, dy, offset, 0.5)
    assert shift["east"] == pytest.approx(1.5, abs=0.26)
    assert shift["north"] == pytest.approx(1.0, abs=0.26)
    assert shift["up"] == pytest.approx(1.5, abs=0.1)


def test_splat_dsm_keeps_the_highest_footprint():
    grid = {"dsm": np.zeros((10, 10)), "west": 0.0, "north": 5.0, "resolution": 0.5}
    points = np.array([[1.25, 3.75, 2.0], [1.25, 3.75, 7.0], [4.0, 1.0, 1.0]])
    dsm = splat_dsm(points, np.array([0.3, 0.3, 0.6]), grid)
    assert dsm[2, 2] == 7.0  # row (5-3.75)/0.5-0.5 = 2, column 1.25/0.5-0.5 = 2
    assert np.isfinite(dsm).sum() >= 2
    assert np.isnan(dsm[0, 9])


def test_error_stats_report_bias_and_spread():
    truth = np.zeros((4, 4))
    predicted = np.full((4, 4), 1.0)
    predicted[0, 0] = np.nan
    stats = error_stats(predicted, truth, np.ones((4, 4), bool))
    assert stats["pixels"] == 15
    assert stats["bias_m"] == pytest.approx(1.0)
    assert stats["mae_m"] == pytest.approx(1.0)
    assert stats["nmad_m"] == pytest.approx(0.0)
    assert stats["completeness"] == pytest.approx(15 / 16)


def test_read_ply_parses_binary_vertices(tmp_path):
    data = np.array([(1.0, 2.0, 3.0, 0.5)], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4")])
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n" + b"".join(
        f"property float {n}\n".encode() for n in data.dtype.names) + b"end_header\n"
    (tmp_path / "a.ply").write_bytes(header + data.tobytes())
    vertices = read_ply(tmp_path / "a.ply")
    assert vertices["z"][0] == 3.0 and vertices["opacity"][0] == 0.5


def test_points_dsm_takes_the_cell_maximum():
    from pipeline.dsm_validation import points_dsm

    grid = {"dsm": np.zeros((4, 4)), "west": 0.0, "north": 2.0, "resolution": 0.5}
    points = np.array([[0.1, 1.9, 1.0], [0.4, 1.6, 4.0], [1.9, 0.1, 2.0], [5.0, 5.0, 9.0]])
    dsm = points_dsm(points, grid)
    assert dsm[0, 0] == 4.0
    assert dsm[3, 3] == 2.0
    assert np.isnan(dsm).sum() == 14


def test_half_overlap_registration_preserves_counts_and_coverage(tmp_path):
    from pipeline.dsm_validation import coverage_stats
    import rasterio

    truth = scene()
    predicted = np.full_like(truth, np.nan)
    predicted[:, :64] = truth[:, :64] + 2
    write_raster(tmp_path/'truth.tif', truth)
    write_raster(tmp_path/'pred.tif', predicted)
    dx, dy, offset = register(tmp_path/'truth.tif', tmp_path/'pred.tif', tmp_path/'reg.tif')
    assert (dx,dy,offset) == pytest.approx((0,0,-2), abs=0.01)
    with rasterio.open(tmp_path/'reg.tif') as source:
        registered = source.read(1)
    extent = np.zeros(truth.shape,bool)
    extent[:,:64] = True
    grid = {'dsm':truth,'classes':np.full(truth.shape,2),'resolution':0.5}
    counts = coverage_stats(registered,grid,extent,side_m=64)
    assert counts['evaluated_pixels'] == 8192
    assert counts['truth_grid_covered_fraction'] == 0.5
    assert counts['block_area_covered_fraction'] == 0.5
    assert counts['labelled_coverage_fraction'] == 1
    assert error_stats(registered,truth,extent)['mae_m'] < 1e-5


def test_actual_split_truth_coverage_matches_half_blocks():
    import json
    from pathlib import Path
    from pipeline.dsm_validation import evaluation_extent, read_truth

    root = Path(__file__).resolve().parents[2]
    directory = root/'artifacts/reference/dfc2019-jax068'
    if not directory.exists():
        pytest.skip('Local truth fixture absent')
    grid = json.loads((root/'configs/jax068-grid.json').read_text())
    truth = read_truth(directory,'JAX_068')
    masks = [evaluation_extent(truth,{},grid=grid,cell=n) for n in ('west','east')]
    assert not (masks[0] & masks[1]).any()
    assert [int(m.sum()) for m in masks] == [127148,128768]
