"""Verify a prepared scene with the real Skyfall loader, without GPU training."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.phase2 import digest, preflight
from pipeline.skyfall_inputs import project


def rejected(reader, path, message, eval=True):
    try:
        reader(str(path), False, eval)
    except ValueError as exc:
        if message not in str(exc):
            raise
    else:
        raise AssertionError(f"Loader accepted invalid scene: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    args = parser.parse_args()
    scene = args.scene.resolve()
    audit = preflight(scene / "phase2-manifest.json", check_environment=False)
    if audit["status"] != "passed":
        raise ValueError(audit["failures"])
    sys.path.insert(0, str(ROOT / ".external/skyfall-gs"))
    from scene.dataset_readers import readSatelliteInfo

    contract_path = scene / "terrasplat-scene.json"
    contract = json.loads(contract_path.read_text())
    checks = {
        "preflight": "passed",
        "payload_manifest_sha256": digest(contract_path),
        "checker_sha256": digest(Path(__file__)),
        "adapter_sha256": audit["inputs"]["prepared-enu.patch"]["sha256"],
    }
    with tempfile.TemporaryDirectory(prefix="terrasplat-loader-") as temporary:
        copied = Path(temporary) / "input"
        shutil.copytree(scene, copied)
        rejected(readSatelliteInfo, copied, "require --eval", eval=False)
        loaded = readSatelliteInfo(str(copied), False, True)
        assert {c.image_name for c in loaded.train_cameras} == set(
            contract["train_catalog_ids"]
        )
        assert {c.image_name for c in loaded.test_cameras} == set(
            contract["holdout_catalog_ids"]
        )
        xyz = np.loadtxt(scene / "points3D.txt", usecols=(1, 2, 3), ndmin=2)
        assert len(loaded.point_cloud.points) == contract["point_count"]
        # The loader stores points as float32, so far outliers round by about |x| * 6e-8. Allow that rounding (with a
        # 4x margin) above the 1e-4 floor; a changed normalization would still be off by whole units.
        cloud_excess = np.abs(loaded.point_cloud.points - xyz) - (1e-4 + 2.4e-7 * np.abs(xyz))
        cloud_error = float(np.max(np.abs(loaded.point_cloud.points - xyz)))
        assert float(np.max(cloud_excess)) < 0, "Loader changed normalization"
        residuals = {}
        for camera in loaded.train_cameras + loaded.test_cameras:
            expected = json.loads(
                (scene / "cameras" / (camera.image_name + ".json")).read_text()
            )
            assert (camera.width, camera.height) == tuple(expected["img_size"])
            assert camera.mask.shape == (camera.height, camera.width)
            assert set(np.unique(camera.mask)) <= {0, 1}
            assert camera.mask.any()
            K = np.eye(4)
            K[0, 0] = camera.width / (2 * np.tan(camera.FovX / 2))
            K[1, 1] = camera.height / (2 * np.tan(camera.FovY / 2))
            K[0, 2] = (camera.cx + 1) * camera.width / 2
            K[1, 2] = (camera.cy + 1) * camera.height / 2
            W2C = np.eye(4)
            W2C[:3, :3] = camera.R.T
            W2C[:3, 3] = camera.T
            actual = {"K": K.ravel().tolist(), "W2C": W2C.ravel().tolist()}
            error = float(
                np.max(
                    np.linalg.norm(
                        project(actual, xyz) - project(expected, xyz), axis=1
                    )
                )
            )
            assert error < 1e-5, "Loader camera projection differs from prepared camera"
            residuals[camera.image_name] = error
            camera.image.close()
        original = (copied / "points3D.txt").read_bytes()
        with (copied / "points3D.txt").open("a") as stream:
            stream.write("# changed after conversion\n")
        rejected(readSatelliteInfo, copied, "checksum changed")
        (copied / "points3D.txt").write_bytes(original)
        transforms_path = copied / "transforms_train.json"
        transforms = json.loads(transforms_path.read_text())
        transforms.update(R=np.eye(3).tolist(), T=[0, 0, 0])
        for frame in transforms["frames"]:
            frame["transform_matrix_rotated"] = frame["transform_matrix"]
        transforms_path.write_text(json.dumps(transforms))
        changed = json.loads((copied / "terrasplat-scene.json").read_text())
        changed["input_sha256"]["transforms_train.json"] = digest(transforms_path)
        (copied / "terrasplat-scene.json").write_text(json.dumps(changed))
        rejected(readSatelliteInfo, copied, "second rotation")
    checks.update(
        status="passed",
        train_cameras=len(loaded.train_cameras),
        holdout_cameras=len(loaded.test_cameras),
        sparse_points=len(xyz),
        float32_ply_max_position_error_model_units=cloud_error,
        loaded_camera_projection_error_pixels=residuals,
        rejected=["missing --eval", "changed input hash", "second normalization"],
        training_executed=False,
        independent_accuracy_validated=False,
    )
    output = scene / "loader-report.json"
    output.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
