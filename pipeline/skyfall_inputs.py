"""Deskew a verified sparse reconstruction and preserve explicit Skyfall splits."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform

from pipeline.georeference import placement
from pipeline.reconstruct import check_source, digest
from pipeline.satellite_sfm import COMMIT

PROFILE = "terrasplat-prepared-enu-v1"


def deskew(K, size):
    """Map pixel centers into an expanded canvas without clipping the shear."""
    K = np.asarray(K, dtype=np.float64).reshape(4, 4)
    width, height = size
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    H = np.eye(3)
    H[0, 1] = -K[0, 1] / K[1, 1]
    corners = np.array(
        [[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1]]
    ).T
    warped = H @ corners
    lower = np.floor(warped[:2].min(axis=1))
    upper = np.ceil(warped[:2].max(axis=1))
    H[:2, 2] = -lower
    new_size = (upper - lower + 1).astype(int).tolist()
    corrected = K.copy()
    corrected[:3, :3] = H @ K[:3, :3]
    corrected[0, 1] = 0.0
    return H, corrected, new_size


def warp_rgb(pixels, H, size, valid):
    """SciPy uses integer array-index centers, matching the camera convention."""
    inverse = np.linalg.inv(H)
    matrix = inverse[:2, :2][::-1, ::-1]
    offset = inverse[:2, 2][::-1]
    shape = (size[1], size[0])
    channels = [
        affine_transform(
            pixels[:, :, c],
            matrix,
            offset,
            output_shape=shape,
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        for c in range(3)
    ]
    if valid.shape != pixels.shape[:2]:
        raise ValueError("Source validity mask and RGB dimensions disagree")
    coverage = affine_transform(
        valid.astype(np.float32),
        matrix,
        offset,
        output_shape=shape,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    coverage = (coverage >= 1 - 1e-6).astype(np.uint8)
    rgb = np.stack(channels, axis=-1)
    rgb[coverage == 0] = 0
    return rgb, coverage


def normalization(xyz):
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError("A finite training sparse cloud is required")
    center = np.median(xyz, axis=0)
    radius = np.percentile(np.linalg.norm(xyz - center, axis=1), 99)
    if radius <= 0:
        raise ValueError("Sparse cloud has no usable extent")
    scale = 256.0 / radius
    shift = float(np.percentile((xyz[:, 2] - center[2]) * scale, 1))
    return {
        "rotation": np.eye(3).tolist(),
        "translation": center.tolist(),
        "scale": float(scale),
        "z_shift": shift,
    }


def transform_points(xyz, matrix):
    return (
        np.column_stack((xyz, np.ones(len(xyz))))
        @ np.asarray(matrix, dtype=np.float64)[:3].T
    )


def model_camera(camera, K, size, source_to_model, rotation):
    c2w = np.linalg.inv(np.asarray(camera["W2C"], dtype=np.float64).reshape(4, 4))
    c2w[:3, 3] = transform_points(c2w[:3, 3][None], source_to_model)[0]
    c2w[:3, :3] = np.asarray(rotation) @ c2w[:3, :3]
    return {
        "K": K.ravel().tolist(),
        "W2C": np.linalg.inv(c2w).ravel().tolist(),
        "img_size": size,
    }, c2w


def project(camera, xyz):
    K = np.asarray(camera["K"], dtype=np.float64).reshape(4, 4)
    W2C = np.asarray(camera["W2C"], dtype=np.float64).reshape(4, 4)
    pixels = np.column_stack((xyz, np.ones(len(xyz)))) @ (K @ W2C)[:3].T
    if not np.isfinite(pixels).all() or np.any(pixels[:, 2] <= 0):
        raise ValueError("Nonfinite or behind-camera projection")
    return pixels[:, :2] / pixels[:, 2:3]


def convert(reconstruction, output, approved, source, inputs):
    import rasterio

    reconstruction, output = Path(reconstruction).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new conversion output directory")
    report_path = reconstruction / "reconstruction-report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "reconstructed" or report["source_commit"] != COMMIT:
        raise ValueError("A verified sparse reconstruction is required")
    input_report_path = Path(inputs) / "inputs-report.json"
    input_report = json.loads(input_report_path.read_text())
    if input_report["selection_sha256"] != report["selection_sha256"]:
        raise ValueError(
            "Source validity metadata and reconstruction selections disagree"
        )
    source_views = {v["catalog_id"]: v for v in input_report["views"]}
    expected = {
        role: set(approved[f"{role}_catalog_ids"]) for role in ("train", "holdout")
    }
    if (
        len(expected["train"]) < 3
        or not expected["holdout"]
        or expected["train"] & expected["holdout"]
    ):
        raise ValueError(
            "Need at least three training acquisitions and an independent holdout"
        )
    for role in expected:
        identifiers = [v["catalog_id"] for v in report["views"] if v["role"] == role]
        if (
            len(identifiers) != len(set(identifiers))
            or set(identifiers) != expected[role]
        ):
            raise ValueError("Reconstruction roles differ from approved acquisitions")
    if len(report["views"]) != sum(map(len, expected.values())):
        raise ValueError("Unexpected acquisition role")
    for relative, checksum in report["output_sha256"].items():
        path = (reconstruction / relative).resolve()
        if not path.is_relative_to(reconstruction) or digest(path) != checksum:
            raise ValueError("Sparse reconstruction changed after verification")
    model = reconstruction / "adjusted-2"
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        if f"adjusted-2/{name}" not in report["output_sha256"]:
            raise ValueError("Sparse model lacks recorded hashes")
    check_source(source, COMMIT)
    sys.path.insert(0, str(Path(source).resolve()))
    from preprocess_sfm.colmap_sfm_utils import extract_camera_dict
    from preprocess_sfm.colmap.read_model import read_model

    cameras = extract_camera_dict(str(model))
    _, _, points = read_model(str(model), ".txt")
    if set(cameras) != {name + ".png" for name in expected["train"]}:
        raise ValueError("Sparse model cameras differ from training acquisitions")
    xyz = np.array([p.xyz for p in points.values()], dtype=np.float64)
    norm = normalization(xyz)
    observer_path = reconstruction / "enu_observer_latlonalt.json"
    observer = json.loads(observer_path.read_text())
    if observer != report["enu_observer_latlonalt"]:
        raise ValueError("ENU observer differs from the reconstruction report")
    anchor = dict(zip(("latitude", "longitude", "ellipsoidal_height_m"), observer))
    geographic = placement(anchor, norm)
    normalized = transform_points(xyz, geographic["source_to_model"])
    restored = transform_points(normalized, geographic["model_to_enu"])
    roundtrip = float(np.max(np.abs(restored - xyz)))
    if roundtrip > 1e-8:
        raise ValueError("Normalization does not restore metric ENU")
    for name in ("images", "masks", "cameras"):
        (output / name).mkdir(parents=True, exist_ok=True)
    for name in ("enu_observer_latlonalt.json", "enu_bbx.json", "latlonalt_bbx.json"):
        shutil.copyfile(reconstruction / name, output / name)
    records = []
    splits = {
        role: {"camera_model": "PINHOLE", "normalization_applied": True, "frames": []}
        for role in expected
    }
    for view in report["views"]:
        name, role = view["catalog_id"], view["role"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid acquisition identifier")
        parent = reconstruction if role == "train" else reconstruction / "holdout"
        image_path = parent / "images" / (name + ".png")
        original_camera_path = parent / "cameras" / (name + ".json")
        if (
            digest(image_path) != view["png_sha256"]
            or digest(original_camera_path) != view["camera_sha256"]
        ):
            raise ValueError("Input image or original camera changed after SfM")
        camera = (
            cameras[name + ".png"]
            if role == "train"
            else json.loads(original_camera_path.read_text())
        )
        raw = source_views[name]
        raw_path = Path(raw["rgb_path"])
        if raw["role"] != role or digest(raw_path) != view["source_rgb_sha256"]:
            raise ValueError("Source RGB validity metadata differs from SfM inputs")
        with rasterio.open(raw_path) as dataset:
            valid = (dataset.dataset_mask() > 0).astype(np.uint8)
            raw_rgb = np.moveaxis(dataset.read(), 0, -1)
        if abs(float(valid.mean()) - raw["valid_fraction"]) > 1e-8:
            raise ValueError(
                "Source validity mask differs from RGB preparation evidence"
            )
        with Image.open(image_path) as image:
            if image.mode != "RGB" or list(image.size) != camera["img_size"]:
                raise ValueError("Camera and RGB dimensions disagree")
            if not np.array_equal(np.asarray(image), raw_rgb):
                raise ValueError("Source RGB and SfM pixels disagree")
            H, K, size = deskew(camera["K"], camera["img_size"])
            rgb, mask = warp_rgb(np.asarray(image), H, size, valid)
        Image.fromarray(rgb).save(output / "images" / (name + ".png"))
        np.save(output / "masks" / (name + ".npy"), mask)
        corrected, c2w = model_camera(
            camera, K, size, geographic["source_to_model"], norm["rotation"]
        )
        (output / "cameras" / (name + ".json")).write_text(
            json.dumps(corrected, indent=2) + "\n"
        )
        old_pixels = project(camera, xyz)
        predicted = np.column_stack((old_pixels, np.ones(len(xyz)))) @ H.T
        error = float(
            np.max(
                np.linalg.norm(
                    predicted[:, :2] - project(corrected, normalized), axis=1
                )
            )
        )
        if error > 1e-5:
            raise ValueError("Warp/normalized-camera projection mismatch")
        splits[role]["frames"].append(
            {
                "file_path": f"./images/{name}.png",
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "w": size[0],
                "h": size[1],
                "transform_matrix": c2w.tolist(),
            }
        )
        records.append(
            {
                "catalog_id": name,
                "role": role,
                "original_size": camera["img_size"],
                "corrected_size": size,
                "pixel_transform": H.tolist(),
                "inverse_pixel_transform": np.linalg.inv(H).tolist(),
                "mask_valid_fraction": float(mask.mean()),
                "source_valid_fraction": float(valid.mean()),
                "source_mask_sha256": hashlib.sha256(valid.tobytes()).hexdigest(),
                "projection_equivalence_max_pixels": error,
                "camera_source": "bundle-adjusted training model"
                if role == "train"
                else "RPC-only holdout; no fitted correction",
            }
        )
    for role, filename in (
        ("train", "transforms_train.json"),
        ("holdout", "transforms_test.json"),
    ):
        (output / filename).write_text(json.dumps(splits[role], indent=2) + "\n")
    with (output / "points3D.txt").open("w") as stream:
        stream.write(
            "# Normalized Skyfall initialization points. Source SfM ERROR retained; tracks omitted.\n"
        )
        for (point_id, point), position in zip(points.items(), normalized):
            stream.write(
                " ".join(
                    [
                        str(point_id),
                        *(format(float(v), ".17g") for v in position),
                        *(str(int(v)) for v in point.rgb),
                        format(float(point.error), ".17g"),
                    ]
                )
                + "\n"
            )
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "version": 1,
        "id": "jax068-2015-historical-pilot",
        "prepared_scene": ".",
        "skyfall": os.path.relpath(root / ".external/skyfall-gs", output),
        "expected_skyfall_commit": "73194842a404910c5bbff006359eedfc56a98d60",
        "python": str(root / ".venv-stage1/bin/python"),
        "georeference": "georeference.json",
        "skyfall_patch_profile": "basic-stage1-registered",
    }
    (output / "phase2-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = {
        str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    georeference = {
        "anchor": anchor,
        "normalization": norm,
        **geographic,
        "input_sha256": hashes.copy(),
        "normalization_applied": True,
        "provenance": {
            "reconstruction_report_sha256": digest(report_path),
            "source_commit": COMMIT,
            "selection_sha256": report["selection_sha256"],
            "fit_inputs": "training sparse points only",
        },
    }
    (output / "georeference.json").write_text(json.dumps(georeference, indent=2) + "\n")
    hashes["georeference.json"] = digest(output / "georeference.json")
    contract = {
        "version": 1,
        "profile": PROFILE,
        "normalization_applied": True,
        "requires_eval": True,
        "point_count": len(points),
        "train_catalog_ids": sorted(expected["train"]),
        "holdout_catalog_ids": sorted(expected["holdout"]),
        "input_sha256": hashes,
    }
    (output / "terrasplat-scene.json").write_text(json.dumps(contract, indent=2) + "\n")
    result = {
        "status": "prepared",
        "profile": PROFILE,
        "reconstruction_report_sha256": digest(report_path),
        "converter_sha256": digest(Path(__file__)),
        "source_inputs_report_sha256": digest(input_report_path),
        "point_count": len(points),
        "views": records,
        "normalization": norm,
        "enu_roundtrip_max_m": roundtrip,
        "independent_accuracy_validated": False,
        "payload_manifest_sha256": digest(output / "terrasplat-scene.json"),
        "pending": [
            "Skyfall loader verification",
            "measured Stage 1 training",
            "independent geographic validation",
        ],
    }
    (output / "conversion-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", default="artifacts/imagery/sfm-2015-02")
    parser.add_argument("--source", default=".external/satellitesfm")
    parser.add_argument("--inputs", default="artifacts/imagery/block-14d")
    parser.add_argument("--approved", default="configs/jax068-pilot.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = convert(
        args.reconstruction,
        args.output,
        json.loads(Path(args.approved).read_text()),
        args.source,
        args.inputs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
