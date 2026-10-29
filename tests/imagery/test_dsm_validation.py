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
