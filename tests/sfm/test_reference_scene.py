import json

import numpy as np
import pytest

from pipeline.georeference import placement
from pipeline.reference_scene import assumed_anchor, legacy_normalization, prepare


def test_legacy_without_rotation_is_identity():
    xyz = np.array([[10.0, 20, -3], [0, 5, 8], [9000, 5, -100]])
    norm = legacy_normalization(xyz, {})
    assert (
        placement({"latitude": 30.0, "longitude": -81.0, "ellipsoidal_height_m": -70.0}, norm)["model_to_enu"]
        == np.eye(4).tolist()
    )


def test_legacy_percentiles_and_inverse():
    xyz = np.array([[1.0, 2, 3], [4, 6, 20], [8, 9, 50], [1e4, -1e4, 100]])
    transforms = {"R": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "T": [3.0, 4, 5]}
    norm = legacy_normalization(xyz, transforms)
    rotated = xyz @ np.array(transforms["R"]).T - transforms["T"]
    expected = rotated * (256 / np.percentile(np.linalg.norm(rotated, axis=1), 99))
    expected[:, 2] -= np.percentile(expected[:, 2], 1)
    geo = placement({"latitude": 30.0, "longitude": -81.0, "ellipsoidal_height_m": -70.0}, norm)
    matrix = np.array(geo["source_to_model"])
    np.testing.assert_allclose(xyz @ matrix[:3, :3].T + matrix[:3, 3], expected)
    inverse = np.array(geo["model_to_enu"])
    np.testing.assert_allclose(expected @ inverse[:3, :3].T + inverse[:3, 3], xyz)


def test_assumed_anchor_uses_metadata_centre_and_low_bracket(tmp_path):
    pytest.importorskip("pyproj")
    path = tmp_path / "aoi.txt"
    path.write_text("436078.995946\n3357495.999923\n512\n0.5\n")
    anchor = assumed_anchor(path)
    assert anchor["assumed_anchor"] is True
    assert anchor["ellipsoidal_height_m"] == -70
    assert anchor["latitude"] == pytest.approx(30.3488, abs=0.0001)
    assert anchor["longitude"] == pytest.approx(-81.6637, abs=0.0001)


def test_prepare_preserves_payload_splits_and_reference(tmp_path):
    pytest.importorskip("pyproj")
    source = tmp_path / "reference"
    source.mkdir()
    for folder in ("images", "masks", "cameras"):
        (source / folder).mkdir()
    for role, name in [("train", "a"), ("test", "b")]:
        frame = {
            "file_path": f"images/{name}.png",
            "transform_matrix": np.eye(4).tolist(),
            "fl_x": 100,
            "fl_y": 100,
            "cx": 4,
            "cy": 4,
        }
        (source / f"transforms_{role}.json").write_text(json.dumps({"frames": [frame]}))
        (source / f"images/{name}.png").write_bytes(name.encode())
        np.save(source / f"masks/{name}.npy", np.ones((8, 8)))
        (source / f"cameras/{name}.json").write_text("{}")
    (source / "points3D.txt").write_text("1 10 20 30 255 0 0 0\n")
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    metadata = tmp_path / "aoi.txt"
    metadata.write_text("436078\n3357496\n512\n0.5\n")
    target = tmp_path / "prepared"
    geo = prepare(source, target, metadata)
    contract = json.loads((target / "terrasplat-scene.json").read_text())
    assert contract["train_catalog_ids"] == ["a"]
    assert contract["holdout_catalog_ids"] == ["b"]
    assert geo["assumed_anchor"] is True
    np.testing.assert_allclose(geo["model_to_enu"], np.eye(4))
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="new directory"):
        prepare(source, target, metadata)


def test_nyc_without_masks_or_cameras_matches_legacy_black_pixel_mask(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    source = tmp_path / 'NYC_004'
    (source / 'images').mkdir(parents=True)
    for role, name in [('train', 'a'), ('test', 'b')]:
        pixels = np.array([[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]], dtype=np.uint8)
        Image.fromarray(pixels).save(source / f'images/{name}.png')
        frame = {'file_path': f'./images/{name}.png', 'transform_matrix': np.eye(4).tolist()}
        (source / f'transforms_{role}.json').write_text(json.dumps({'frames': [frame]}))
    (source / 'transforms.json').write_text('{}')
    (source / 'points3D.txt').write_text('1 10 20 30 255 0 0 0\n')
    geo = {'anchor': {'latitude': 40, 'longitude': -74, 'ellipsoidal_height_m': 0},
           'assumed_anchor': True, **placement(
               {'latitude': 40, 'longitude': -74, 'ellipsoidal_height_m': 0},
               legacy_normalization(np.zeros((1, 3)), {}))}
    output = tmp_path / 'prepared'
    prepare(source, output, georeference=geo)
    np.testing.assert_array_equal(np.load(output / 'masks/a.npy'), [[0, 1], [1, 1]])
    assert not (source / 'masks').exists()
    assert not (output / 'cameras').exists()
    contract = json.loads((output / 'terrasplat-scene.json').read_text())
    assert contract['train_catalog_ids'] == ['a']
    assert contract['holdout_catalog_ids'] == ['b']
