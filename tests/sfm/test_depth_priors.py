import json
import math
import subprocess

import Imath
import numpy as np
import OpenEXR
from PIL import Image
import pytest

from pipeline.depth_priors import generate
from pipeline.phase2 import digest, preflight
from pipeline.training import copy_scene


@pytest.fixture
def scene(tmp_path):
    source = tmp_path / "scene"
    source.mkdir()
    (source / "images").mkdir()
    (source / "masks").mkdir()
    matrix = np.eye(4).tolist()
    for role, name, color in (("train", "april", 40), ("test", "may2", 90)):
        Image.new("RGB", (3, 2), (color, 0, 0)).save(source / f"images/{name}.png")
        np.save(source / f"masks/{name}.npy", [[1, 0, 1], [1, 1, 1]])
        (source / f"transforms_{role}.json").write_text(json.dumps({"frames": [{
            "file_path": f"images/{name}.png", "fl_x": 100, "fl_y": 100,
            "transform_matrix": matrix,
        }]}))
    (source / "points3D.txt").write_text("1 0 0 0 255 255 255 0\n")
    georef = {"anchor": {"longitude": 0, "latitude": 0, "ellipsoidal_height_m": 0},
              "normalization": {"rotation": np.eye(3).tolist(), "translation": [0, 0, 0],
                                "scale": 1, "z_shift": 0},
              "provenance": "synthetic test only",
              "input_sha256": {name: digest(source / name)
                               for name in ("transforms_train.json", "points3D.txt")}}
    (source / "georeference.json").write_text(json.dumps(georef))
    skyfall = tmp_path / "skyfall"
    skyfall.mkdir()
    subprocess.run(["git", "init", "-q", str(skyfall)], check=True)
    (skyfall / "train.py").write_text("# synthetic reference\n")
    subprocess.run(["git", "-C", str(skyfall), "add", "train.py"], check=True)
    subprocess.run(["git", "-C", str(skyfall), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(skyfall), "rev-parse", "HEAD"], text=True).strip()
    (source / "phase2-manifest.json").write_text(json.dumps({
        "version": 1, "id": "test", "prepared_scene": ".", "skyfall": "../skyfall",
        "expected_skyfall_commit": commit, "georeference": "georeference.json",
    }))
    contract = {"train_catalog_ids": ["april"], "holdout_catalog_ids": ["may2"],
                "input_sha256": {str(p.relative_to(source)): digest(p)
                                 for p in source.rglob("*") if p.is_file()}}
    (source / "terrasplat-scene.json").write_text(json.dumps(contract))
    return source


def test_training_only_masked_exr_and_manifest_roundtrip(scene, tmp_path):
    calls = []

    def fake_model(rgb, fov):
        calls.append((rgb, fov))
        return np.array([[1, 2, np.inf], [np.nan, 5, 6]])

    output = tmp_path / "nested" / "with-depth"
    report = generate(scene, output, fake_model, "fake-revision")
    assert len(calls) == 1
    assert calls[0][0].shape == (2, 3, 3)
    assert calls[0][0][0, 0, 0] == 40  # April RGB, never the May 2 holdout.
    assert calls[0][1] == pytest.approx(math.degrees(2 * math.atan(3 / 200)))
    assert sorted(p.name for p in (output / "depths_moge").iterdir()) == ["april.exr"]
    exr = OpenEXR.InputFile(str(output / "depths_moge/april.exr"))
    try:
        assert set(exr.header()["channels"]) == {"Y"}
        depth = np.frombuffer(exr.channel("Y", Imath.PixelType(Imath.PixelType.FLOAT)),
                              dtype=np.float32).reshape(2, 3)
    finally:
        exr.close()
    np.testing.assert_array_equal(depth, [[1, np.nan, np.nan], [np.nan, 5, 6]])
    assert report["images"][0]["valid_fraction"] == 0.5
    assert report["images"][0]["depth_min"] == 1
    assert report["images"][0]["depth_max"] == 6
    assert (output / "georeference.json").read_bytes() == (scene / "georeference.json").read_bytes()
    contract = json.loads((output / "terrasplat-scene.json").read_text())
    for relative, expected in contract["input_sha256"].items():
        assert digest(output / relative) == expected
    copy_scene(output, tmp_path / "training-copy")
    assert (tmp_path / "training-copy/depths_moge/april.exr").is_file()
    assert preflight(output / "phase2-manifest.json", False)["status"] == "passed"
    assert not (scene / "depths_moge").exists()


def test_changed_source_payload_is_rejected_before_prediction(scene, tmp_path):
    (scene / "images/april.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload changed"):
        generate(scene, tmp_path / "output", None, "fake")


def test_holdout_in_training_contract_is_rejected(scene, tmp_path):
    path = scene / "terrasplat-scene.json"
    contract = json.loads(path.read_text())
    contract["holdout_catalog_ids"].append("april")
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="acquisition splits"):
        generate(scene, tmp_path / "output", None, "fake")
