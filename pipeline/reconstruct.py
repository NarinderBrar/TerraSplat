"""Run CPU ColmapForVisSat on the approved, RPC-initialized pilot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from pipeline.satellite_sfm import COMMIT

COLMAP_COMMIT = "79a6d150714d0eb350a1b10e17bedf91877568a1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reprojection_stats(cameras, images, points):
    """Recompute residuals after the final BA, including the retained skew."""
    residuals = []
    for point in points.values():
        for image_id, point_index in zip(point.image_ids, point.point2D_idxs):
            image = images[image_id]
            camera = cameras[image.name]
            K = np.asarray(camera["K"], dtype=np.float64).reshape(4, 4)
            W2C = np.asarray(camera["W2C"], dtype=np.float64).reshape(4, 4)
            projected = (K @ W2C)[:3] @ np.append(point.xyz, 1.0)
            if not np.isfinite(projected).all() or projected[2] <= 0:
                raise ValueError("Nonfinite or behind-camera sparse point")
            residuals.append(
                float(
                    np.linalg.norm(
                        projected[:2] / projected[2] - image.xys[point_index]
                    )
                )
            )
    if not residuals or not np.isfinite(residuals).all():
        raise ValueError("No finite sparse observations")
    return {
        "observations": len(residuals),
        "median": float(np.median(residuals)),
        "p95": float(np.percentile(residuals, 95)),
        "max": float(max(residuals)),
    }


def check_source(path, commit):
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise ValueError(f"Unexpected source revision at {path}")
    subprocess.run(["git", "-C", str(path), "diff", "--exit-code", "HEAD"], check=True)


def check_adjusted_camera(initial, adjusted):
    before = np.asarray(initial["W2C"], dtype=np.float64)
    after = np.asarray(adjusted["W2C"], dtype=np.float64)
    if not np.allclose(before, after, rtol=0, atol=1e-9):
        raise ValueError("Fixed camera extrinsics changed; ENU frame requires review")
    expected = np.asarray(initial["K"], dtype=np.float64).copy()
    expected[[2, 6]] = np.asarray(adjusted["K"])[[2, 6]]
    if not np.allclose(expected, adjusted["K"], rtol=0, atol=1e-9):
        raise ValueError("Fixed intrinsics or camera skew changed")


def prepare(inputs, calibration, approved, output):
    """Copy verified pixels and full cameras, separating acquisition roles."""
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new reconstruction output directory")
    evidence = json.loads((Path(inputs) / "inputs-report.json").read_text())
    initial = json.loads((Path(calibration) / "calibration-report.json").read_text())
    if (
        initial.get("status") != "passed"
        or evidence["selection_sha256"] != initial["selection_sha256"]
    ):
        raise ValueError("Inputs and passed calibration must share the same selection")
    expected = {
        role: set(approved[f"{role}_catalog_ids"]) for role in ("train", "holdout")
    }
    if expected["train"] & expected["holdout"]:
        raise ValueError("Training and holdout acquisitions overlap")
    for records in (evidence["views"], initial["views"]):
        for role in expected:
            actual = [v["catalog_id"] for v in records if v["role"] == role]
            if len(actual) != len(set(actual)) or set(actual) != expected[role]:
                raise ValueError("Acquisition roles differ from the approved pilot")
        if len(records) != sum(map(len, expected.values())):
            raise ValueError("Unexpected acquisition role")
    for folder in ("images", "cameras", "holdout/images", "holdout/cameras", "logs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for name in ("enu_observer_latlonalt.json", "enu_bbx.json", "latlonalt_bbx.json"):
        shutil.copyfile(Path(calibration) / name, output / name)
    report = {
        "status": "prepared",
        "selection_sha256": evidence["selection_sha256"],
        "source_commit": COMMIT,
        "colmap_commit": COLMAP_COMMIT,
        "views": [],
        "commands": [],
        "independent_accuracy_validated": False,
    }
    for view in evidence["views"]:
        identifier, role = view["catalog_id"], view["role"]
        if Path(identifier).name != identifier:
            raise ValueError("Invalid acquisition identifier")
        source = Path(view["rgb_path"])
        if digest(source) != view["rgb_sha256"]:
            raise ValueError(f"Changed RGB input: {identifier}")
        camera = json.loads(
            (Path(calibration) / "cameras" / f"{identifier}.json").read_text()
        )
        K = np.asarray(camera["K"], dtype=np.float64).reshape(4, 4)
        W2C = np.asarray(camera["W2C"], dtype=np.float64).reshape(4, 4)
        if not np.isfinite(K).all() or not np.isfinite(W2C).all():
            raise ValueError("Nonfinite camera")
        target = output if role == "train" else output / "holdout"
        with Image.open(source) as image:
            if list(image.size) != camera["img_size"] or image.mode != "RGB":
                raise ValueError("RGB pixels and camera dimensions disagree")
            image.save(target / "images" / f"{identifier}.png")
        (target / "cameras" / f"{identifier}.json").write_text(
            json.dumps(camera, indent=2) + "\n"
        )
        report["views"].append(
            {
                "catalog_id": identifier,
                "role": role,
                "source_rgb_sha256": view["rgb_sha256"],
                "png_sha256": digest(target / "images" / f"{identifier}.png"),
                "camera_sha256": digest(target / "cameras" / f"{identifier}.json"),
                "pixel_transform": np.eye(3).tolist(),
                "skew_pixels": float(K[0, 1]),
            }
        )
    return report


def run_command(args, output, name, report, timeout=900):
    log = Path(output) / "logs" / f"{name}.log"
    started = time.monotonic()
    entry = {"name": name, "argv": list(map(str, args)), "log": str(log)}
    report["commands"].append(entry)
    print(name, flush=True)
    try:
        with log.open("w") as stream:
            result = subprocess.run(
                entry["argv"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(
                    os.environ, QT_QPA_PLATFORM="offscreen", OPENBLAS_NUM_THREADS="2"
                ),
                timeout=timeout,
                check=False,
            )
        entry["returncode"] = result.returncode
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, entry["argv"])
    finally:
        entry["elapsed_seconds"] = time.monotonic() - started


def reconstruct(inputs, calibration, approved, output, source, colmap):
    check_source(source, COMMIT)
    check_source(Path(".external/colmap-for-vissat"), COLMAP_COMMIT)
    output = Path(output).resolve()
    colmap = Path(colmap).resolve()
    runtime = json.loads((colmap.parent.parent / "runtime.json").read_text())
    if runtime["colmap_commit"] != COLMAP_COMMIT or runtime[
        "executable_sha256"
    ] != digest(colmap):
        raise ValueError("SfM executable differs from the recorded CPU build")
    report = prepare(inputs, calibration, approved, output)
    report["executable_sha256"] = digest(colmap)
    report["runtime"] = runtime
    report["runner_sha256"] = digest(Path(__file__))
    report["enu_observer_latlonalt"] = json.loads(
        (output / "enu_observer_latlonalt.json").read_text()
    )
    sys.path.insert(0, str(Path(source).resolve()))
    from preprocess_sfm.colmap_sfm_utils import init_posed_sfm, extract_camera_dict
    from preprocess_sfm.colmap.read_model import read_model

    db = output / "database.db"
    images = output / "images"
    try:
        run_command(
            [
                colmap,
                "feature_extractor",
                "--database_path",
                db,
                "--image_path",
                images,
                "--ImageReader.camera_model",
                "PERSPECTIVE",
                "--SiftExtraction.use_gpu",
                "0",
                "--SiftExtraction.num_threads",
                # The pinned initializer requires image_id == camera_id;
                # parallel extraction can finish images out of camera order.
                "1",
                "--SiftExtraction.max_image_size",
                "10000",
                "--SiftExtraction.domain_size_pooling",
                "1",
                "--SiftExtraction.max_num_features",
                "8192",
            ],
            output,
            "features",
            report,
        )
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT image_id, name, camera_id FROM images"
            ).fetchall()
            expected = {
                v["catalog_id"] + ".png"
                for v in report["views"]
                if v["role"] == "train"
            }
            if {r[1] for r in rows} != expected or any(r[0] != r[2] for r in rows):
                raise ValueError(
                    "COLMAP database image/camera mapping differs from training inputs"
                )
            report["features"] = dict(
                conn.execute(
                    "SELECT images.name, keypoints.rows FROM images "
                    "JOIN keypoints ON images.image_id=keypoints.image_id"
                )
            )
        run_command(
            [
                colmap,
                "exhaustive_matcher",
                "--database_path",
                db,
                "--SiftMatching.use_gpu",
                "0",
                "--SiftMatching.num_threads",
                "3",
                "--SiftMatching.guided_matching",
                "1",
                "--SiftMatching.max_error",
                "3",
                "--SiftMatching.max_num_matches",
                "16384",
            ],
            output,
            "matching",
            report,
            # Exhaustive matching of 20 views takes about 15 minutes on the 3 threads above, right at the default
            # limit, and longer when the machine is busy. Only the safety timeout changes; matching settings do not.
            timeout=3600,
        )
        with sqlite3.connect(db) as conn:
            report["verified_pairs"] = [
                dict(pair_id=r[0], inliers=r[1])
                for r in conn.execute("SELECT pair_id, rows FROM two_view_geometries")
            ]
        cameras = {
            p.stem + ".png": json.loads(p.read_text())
            for p in (output / "cameras").glob("*.json")
        }
        adjusted = output / "cam_dict_adjusted.json"
        adjusted.write_text(json.dumps(cameras, indent=2) + "\n")
        for threshold in (32, 2):
            tri = output / f"triangulated-{threshold}"
            ba = output / f"adjusted-{threshold}"
            tri.mkdir()
            ba.mkdir()
            init_posed_sfm(str(db), str(adjusted), str(tri))
            run_command(
                [
                    colmap,
                    "point_triangulator",
                    "--database_path",
                    db,
                    "--image_path",
                    images,
                    "--input_path",
                    tri,
                    "--output_path",
                    tri,
                    "--Mapper.ba_refine_principal_point",
                    "0",
                    "--Mapper.filter_min_tri_angle",
                    "4.99",
                    "--Mapper.init_max_forward_motion",
                    "1e20",
                    "--Mapper.tri_min_angle",
                    "5",
                    "--Mapper.tri_merge_max_reproj_error",
                    str(threshold),
                    "--Mapper.tri_complete_max_reproj_error",
                    str(threshold),
                    "--Mapper.filter_max_reproj_error",
                    str(threshold),
                    "--Mapper.extract_colors",
                    "1",
                    "--Mapper.ba_refine_focal_length",
                    "0",
                    "--Mapper.ba_refine_extra_params",
                    "0",
                    "--Mapper.max_extra_param",
                    "1e20",
                    "--Mapper.ba_local_num_images",
                    "6",
                    "--Mapper.ba_local_max_num_iterations",
                    "100",
                    "--Mapper.ba_global_images_ratio",
                    "1.0000001",
                    "--Mapper.ba_global_max_num_iterations",
                    "100",
                    "--Mapper.tri_ignore_two_view_tracks",
                    "0",
                ],
                output,
                f"triangulation-{threshold}",
                report,
            )
            run_command(
                [
                    colmap,
                    "bundle_adjuster",
                    "--input_path",
                    tri,
                    "--output_path",
                    ba,
                    "--BundleAdjustment.max_num_iterations",
                    "5000",
                    "--BundleAdjustment.refine_focal_length",
                    "0",
                    "--BundleAdjustment.refine_principal_point",
                    "1",
                    "--BundleAdjustment.refine_extra_params",
                    "0",
                    "--BundleAdjustment.refine_extrinsics",
                    "0",
                    "--BundleAdjustment.function_tolerance",
                    "0",
                    "--BundleAdjustment.gradient_tolerance",
                    "0",
                    "--BundleAdjustment.parameter_tolerance",
                    "1e-10",
                    "--BundleAdjustment.constrain_points",
                    "1",
                    "--BundleAdjustment.constrain_points_loss_weight",
                    "0.01",
                ],
                output,
                f"bundle-adjustment-{threshold}",
                report,
            )
            cameras = extract_camera_dict(str(ba))
            adjusted.write_text(json.dumps(cameras, indent=2) + "\n")
        _, model_images, points = read_model(str(ba), ".txt")
        if set(cameras) != expected or len(model_images) != len(expected) or not points:
            raise ValueError(
                "Sparse reconstruction is empty or missing training cameras"
            )
        xyz = np.array([p.xyz for p in points.values()], dtype=np.float64)
        errors = np.array([p.error for p in points.values()], dtype=np.float64)
        if not np.isfinite(xyz).all() or not np.isfinite(errors).all():
            raise ValueError("Nonfinite sparse reconstruction")
        target = output / "cameras_adjusted"
        target.mkdir()
        for name, camera in cameras.items():
            check_adjusted_camera(
                json.loads(
                    (output / "cameras" / (Path(name).stem + ".json")).read_text()
                ),
                camera,
            )
            (target / (Path(name).stem + ".json")).write_text(
                json.dumps(camera, indent=2) + "\n"
            )
        observations = {
            image.name: int((image.point3D_ids != -1).sum())
            for image in model_images.values()
        }
        if any(count == 0 for count in observations.values()):
            raise ValueError("A training camera has no triangulated observations")
        report.update(
            status="reconstructed",
            sparse_points=len(points),
            reprojection_error_pixels=reprojection_stats(cameras, model_images, points),
            enu_percentiles_2_98=np.percentile(xyz, [2, 98], axis=0).tolist(),
            observations_by_image=observations,
            track_lengths={
                str(n): sum(len(p.image_ids) == n for p in points.values())
                for n in range(2, len(expected) + 1)
            },
            output_sha256={
                str(p.relative_to(output)): digest(p) for p in sorted(ba.glob("*.txt"))
            },
            principal_point_adjustments={
                name: [
                    cameras[name]["K"][i]
                    - json.loads(
                        (output / "cameras" / (Path(name).stem + ".json")).read_text()
                    )["K"][i]
                    for i in (2, 6)
                ]
                for name in cameras
            },
            pending=[
                "skew correction and recorded pixel transform",
                "Skyfall conversion and normalization capture",
                "independent geographic validation",
                "training",
            ],
        )
        return report
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "reconstruction-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="artifacts/imagery/block-14d")
    parser.add_argument(
        "--calibration", default="artifacts/imagery/calibration-14d-validated"
    )
    parser.add_argument("--approved", default="configs/jax068-pilot.json")
    parser.add_argument("--source", default=".external/satellitesfm")
    parser.add_argument("--colmap", default=".toolchains/sfm-cpu/bin/colmap")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = reconstruct(
        args.inputs,
        args.calibration,
        json.loads(Path(args.approved).read_text()),
        args.output,
        args.source,
        args.colmap,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
