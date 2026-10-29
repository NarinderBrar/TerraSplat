import hashlib
import json
import subprocess

import pytest

from pipeline.georeference import multiply, placement
from pipeline.phase2 import preflight


IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@pytest.fixture
def block(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "train.png").write_bytes(b"distinct training image")
    (scene / "test.png").write_bytes(b"held-out image")
    matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    for split in ("train", "test"):
        (scene / f"transforms_{split}.json").write_text(json.dumps({"frames": [{"file_path": f"{split}.png", "transform_matrix": matrix, "fl_x": 100, "fl_y": 100}]}))
    (scene / "points3D.txt").write_text("# POINT3D_ID X Y Z R G B ERROR TRACK\n1 0 0 0 255 255 255 0\n")
    repo = tmp_path / "skyfall"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "train.py").write_text("# reference\n")
    subprocess.run(["git", "-C", str(repo), "add", "train.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    record = {
        "anchor": {"longitude": 0, "latitude": 0, "ellipsoidal_height_m": 20},
        "normalization": {"rotation": IDENTITY, "translation": [2, -3, 4], "scale": 2, "z_shift": 5},
        "provenance": "Synthetic preprocessing evidence for tests only",
        "input_sha256": {name: hashlib.sha256((scene / name).read_bytes()).hexdigest() for name in ("transforms_train.json", "points3D.txt")},
    }
    (tmp_path / "georeference.json").write_text(json.dumps(record))
    manifest = {"version": 1, "id": "test-block", "prepared_scene": "scene", "skyfall": "skyfall",
                "expected_skyfall_commit": commit, "georeference": "georeference.json"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_prepared_block_keeps_explicit_registration_and_input_hashes(block):
    report = preflight(block, check_environment=False)
    assert report["status"] == "passed"
    assert report["environment_checked"] is False
    assert report["inputs"]["transforms_train.json"]["image_count"] == 1
    assert report["georeference"]["anchor"]["ellipsoidal_height_m"] == 20


def test_renamed_holdout_content_is_rejected(block):
    scene = block.parent / "scene"
    (scene / "test.png").write_bytes((scene / "train.png").read_bytes())
    report = preflight(block, check_environment=False)
    assert report["status"] == "blocked"
    assert "Training and held-out images overlap by content" in report["failures"]


def test_stale_normalization_after_source_change_is_rejected(block):
    with (block.parent / "scene" / "points3D.txt").open("a") as stream:
        stream.write("2 10 20 30 255 0 0 0\n")
    report = preflight(block, check_environment=False)
    assert report["status"] == "blocked"
    assert any("does not match points3D.txt" in message for message in report["failures"])


def test_missing_georeference_is_never_replaced_by_a_synthetic_anchor(block):
    manifest = json.loads(block.read_text())
    manifest["georeference"] = None
    block.write_text(json.dumps(manifest))
    report = preflight(block, check_environment=False)
    assert report["status"] == "blocked"
    assert report["georeference"] is None


def test_modified_backend_is_rejected(block):
    (block.parent / "skyfall" / "train.py").write_text("# changed backend\n")
    report = preflight(block, check_environment=False)
    assert any("training source differs" in message for message in report["failures"])


def test_claiming_adapter_profile_does_not_allow_arbitrary_source_changes(block):
    manifest = json.loads(block.read_text())
    manifest["skyfall_patch_profile"] = "basic-stage1-registered"
    block.write_text(json.dumps(manifest))
    (block.parent / "skyfall/train.py").write_text("# unrecorded edit\n")
    assert any("recorded adapter" in message for message in preflight(block, False)["failures"])


def test_unknown_adapter_profile_is_rejected(block):
    manifest = json.loads(block.read_text())
    manifest["skyfall_patch_profile"] = "allow-anything"
    block.write_text(json.dumps(manifest))
    assert "Unsupported Skyfall patch profile" in preflight(block, False)["failures"]


@pytest.mark.parametrize("rotation", [IDENTITY, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]])
def test_normalization_inverse_restores_metric_enu(rotation):
    result = placement({"longitude": -81.66, "latitude": 30.34, "ellipsoidal_height_m": 11},
                       {"rotation": rotation, "translation": [5, -7, 9], "scale": 3.7, "z_shift": -4.2})
    product = multiply(result["model_to_enu"], result["source_to_model"])
    for i in range(4):
        assert product[i] == pytest.approx([int(i == j) for j in range(4)], abs=1e-10)


def test_wgs84_at_equator_places_axes_and_height_independently():
    result = placement({"longitude": 0, "latitude": 0, "ellipsoidal_height_m": 20},
                       {"rotation": IDENTITY, "translation": [0, 0, 0], "scale": 1, "z_shift": 0})
    assert result["enu_to_ecef"] == [[0, 0, 1, 6378157], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]


@pytest.mark.parametrize("scale,rotation", [(0, IDENTITY), (-1, IDENTITY), (1, [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), (float("nan"), IDENTITY)])
def test_invalid_or_reflected_normalization_is_rejected(scale, rotation):
    with pytest.raises(ValueError):
        placement({"longitude": 0, "latitude": 0, "ellipsoidal_height_m": 0},
                  {"rotation": rotation, "translation": [0, 0, 0], "scale": scale, "z_shift": 0})


def test_nonfinite_camera_is_rejected(block):
    path = block.parent / "scene" / "transforms_test.json"
    data = json.loads(path.read_text())
    data["frames"][0]["transform_matrix"][0][0] = float("nan")
    path.write_text(json.dumps(data))
    assert any("finite and affine" in message for message in preflight(block, False)["failures"])


def test_empty_sparse_cloud_is_rejected(block):
    (block.parent / "scene" / "points3D.txt").write_text("# no points\n")
    assert any("no sparse points" in message for message in preflight(block, False)["failures"])
