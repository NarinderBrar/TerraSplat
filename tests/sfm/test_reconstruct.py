import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from pipeline.reconstruct import (
    check_adjusted_camera,
    digest,
    prepare,
    reprojection_stats,
    run_command,
)


@pytest.fixture
def pilot(tmp_path):
    inputs, calibration = tmp_path / "inputs", tmp_path / "calibration"
    inputs.mkdir()
    (calibration / "cameras").mkdir(parents=True)
    approved = {"train_catalog_ids": ["a", "b", "c"], "holdout_catalog_ids": ["d"]}
    views = []
    for name in "abcd":
        image = inputs / f"{name}.tif"
        Image.new("RGB", (8, 6), (23, 65, 190)).save(image)
        K = np.eye(4)
        K[0, 1] = 37.25
        camera = {
            "K": K.ravel().tolist(),
            "W2C": np.eye(4).ravel().tolist(),
            "img_size": [8, 6],
        }
        (calibration / "cameras" / f"{name}.json").write_text(json.dumps(camera))
        views.append(
            {
                "catalog_id": name,
                "role": "holdout" if name == "d" else "train",
                "rgb_path": str(image),
                "rgb_sha256": digest(image),
            }
        )
    (inputs / "inputs-report.json").write_text(
        json.dumps({"selection_sha256": "selection", "views": views})
    )
    (calibration / "calibration-report.json").write_text(
        json.dumps(
            {"status": "passed", "selection_sha256": "selection", "views": views}
        )
    )
    for name in ["enu_observer_latlonalt.json", "enu_bbx.json", "latlonalt_bbx.json"]:
        (calibration / name).write_text("[30,-81,-70]")
    return inputs, calibration, approved


def test_preparation_preserves_pixels_skew_and_separates_holdout(pilot, tmp_path):
    inputs, calibration, approved = pilot
    output = tmp_path / "run"
    prepare(inputs, calibration, approved, output)
    assert {p.stem for p in (output / "images").iterdir()} == {"a", "b", "c"}
    assert {p.stem for p in (output / "holdout/images").iterdir()} == {"d"}
    np.testing.assert_array_equal(
        np.asarray(Image.open(inputs / "a.tif")),
        np.asarray(Image.open(output / "images/a.png")),
    )
    assert json.loads((output / "cameras/a.json").read_text())["K"][1] == 37.25
    assert (output / "enu_observer_latlonalt.json").read_bytes() == (
        calibration / "enu_observer_latlonalt.json"
    ).read_bytes()


def test_changed_rgb_is_rejected(pilot, tmp_path):
    inputs, calibration, approved = pilot
    Image.new("RGB", (8, 6)).save(inputs / "a.tif")
    with pytest.raises(ValueError, match="Changed RGB"):
        prepare(inputs, calibration, approved, tmp_path / "run")


def test_holdout_cannot_be_reassigned_to_training(pilot, tmp_path):
    inputs, calibration, approved = pilot
    p = inputs / "inputs-report.json"
    record = json.loads(p.read_text())
    record["views"][-1]["role"] = "train"
    p.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="roles differ"):
        prepare(inputs, calibration, approved, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_colmap_failure_is_not_hidden_by_logging(tmp_path):
    (tmp_path / "logs").mkdir()
    report = {"commands": []}
    with pytest.raises(subprocess.CalledProcessError):
        run_command(
            [sys.executable, "-c", "print('failed'); raise SystemExit(7)"],
            tmp_path,
            "failed",
            report,
        )
    assert report["commands"][0]["returncode"] == 7
    assert "failed" in (tmp_path / "logs/failed.log").read_text()


def test_residuals_use_full_skew_camera_at_satellite_distance():
    K = np.eye(4)
    K[:3, :3] = [[4e6, 70000, 12000], [0, 4e6, -3400], [0, 0, 1]]
    W2C = np.eye(4)
    W2C[2, 3] = 600000
    xyz = np.array([1.2, 3.4, 5.6], dtype=np.float64)
    p = (K @ W2C)[:3] @ np.append(xyz, 1)
    camera = {"K": K.ravel().tolist(), "W2C": W2C.ravel().tolist()}
    image = SimpleNamespace(name="a", xys=np.array([p[:2] / p[2]]))
    point = SimpleNamespace(xyz=xyz, image_ids=[1], point2D_idxs=[0])
    assert reprojection_stats({"a": camera}, {1: image}, {1: point})["max"] == 0
    camera["K"][1] = 0
    assert reprojection_stats({"a": camera}, {1: image}, {1: point})["max"] > 0.3


def test_adjustment_cannot_shift_the_enu_frame_or_drop_skew():
    original = {"K": np.eye(4).ravel().tolist(), "W2C": np.eye(4).ravel().tolist()}
    original["K"][1] = 70000
    adjusted = json.loads(json.dumps(original))
    adjusted["K"][2] += 4.5
    check_adjusted_camera(original, adjusted)
    adjusted["W2C"][3] += 0.001
    with pytest.raises(ValueError, match="extrinsics changed"):
        check_adjusted_camera(original, adjusted)
    adjusted["W2C"][3] = 0
    adjusted["K"][1] = 0
    with pytest.raises(ValueError, match="skew changed"):
        check_adjusted_camera(original, adjusted)
