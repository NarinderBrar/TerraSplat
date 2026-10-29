"""Copy the authors' JAX_068 scene into the existing normalized input contract."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from pipeline.georeference import placement
from pipeline.phase2 import digest

ROOT = Path(__file__).resolve().parents[1]


def legacy_normalization(xyz, transforms):
    if "R" not in transforms:
        return {"rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0], "scale": 1.0, "z_shift": 0.0}
    rotation = np.asarray(transforms["R"], dtype=np.float64)[:3, :3]
    centre = np.asarray(transforms["T"], dtype=np.float64)
    rotated = xyz @ rotation.T - centre
    radius = np.percentile(np.linalg.norm(rotated, axis=1), 99)
    if not np.isfinite(rotated).all() or radius <= 0:
        raise ValueError("Invalid legacy sparse cloud")
    scale = 256 / radius
    shift = np.percentile((rotated * scale)[:, 2], 1)
    return {
        "rotation": rotation.tolist(),
        "translation": centre.tolist(),
        "scale": float(scale),
        "z_shift": float(shift),
    }


def assumed_anchor(metadata, low=-50):
    from pyproj import Transformer

    east, north, size, resolution = np.loadtxt(metadata)
    # Match SatelliteSfM: centre of the geographic AOI bounds, low bracket minus 20 m.
    corners = [
        (east + x * size * resolution, north + y * size * resolution) for x, y in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]
    transformer = Transformer.from_crs(32617, 4326, always_xy=True)
    lon, lat = transformer.transform(*zip(*corners))
    return {
        "latitude": (min(lat) + max(lat)) / 2,
        "longitude": (min(lon) + max(lon)) / 2,
        "ellipsoidal_height_m": low - 20,
        "assumed_anchor": True,
    }


def prepare(source, output, metadata, patch_profile="basic-stage1-registered"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Use a new directory outside the reference source")
    source_hashes = {}

    def read(relative):
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Reference input escapes scene")
        source_hashes[str(relative)] = digest(path)
        return path

    transforms = {role: json.loads(read(f"transforms_{role}.json").read_text()) for role in ("train", "test")}
    points_path = read("points3D.txt")
    records = [
        line.split()
        for line in points_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    xyz = np.array([row[1:4] for row in records], dtype=np.float64)
    norm = legacy_normalization(xyz, transforms["train"])
    anchor = assumed_anchor(metadata)
    geographic = placement(anchor, norm)
    normalized = xyz @ np.asarray(norm["rotation"]).T - norm["translation"]
    normalized *= norm["scale"]
    normalized[:, 2] -= norm["z_shift"]
    output.mkdir(parents=True)
    names = {}
    for role, original in transforms.items():
        frames = []
        names[role] = []
        for frame in original["frames"]:
            frame = frame.copy()
            name = Path(frame["file_path"]).stem
            names[role].append(name)
            for relative in (frame["file_path"], f"masks/{name}.npy", f"cameras/{name}.json"):
                src = read(relative)
                dst = output / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            c2w = np.array(frame.pop("transform_matrix_rotated", frame["transform_matrix"]), dtype=np.float64)
            c2w[:3, 3] *= norm["scale"]
            c2w[2, 3] -= norm["z_shift"]
            frame["transform_matrix"] = c2w.tolist()
            frames.append(frame)
        (output / f"transforms_{role}.json").write_text(json.dumps({"frames": frames}, indent=2) + "\n")
    if set(names["train"]) & set(names["test"]):
        raise ValueError("Reference splits overlap")
    with (output / "points3D.txt").open("w") as stream:
        for row, point in zip(records, normalized):
            row[1:4] = [format(v, ".17g") for v in point]
            stream.write(" ".join(row) + "\n")
    manifest = {
        "version": 1,
        "id": "ref-jax068",
        "prepared_scene": ".",
        "skyfall": os.path.relpath(
            ROOT
            / (".external/skyfall-gs-pseudo" if patch_profile == "stage1-pseudo-depth" else ".external/skyfall-gs"),
            output,
        ),
        "expected_skyfall_commit": "73194842a404910c5bbff006359eedfc56a98d60",
        "python": str(ROOT / ".venv-stage1/bin/python"),
        "georeference": "georeference.json",
        "skyfall_patch_profile": patch_profile,
    }
    (output / "phase2-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    geo = {
        "anchor": anchor,
        "assumed_anchor": True,
        "normalization": norm,
        **geographic,
        "input_sha256": hashes.copy(),
        "normalization_applied": True,
        "provenance": {
            "source": str(source),
            "source_sha256": source_hashes,
            "aoi_metadata_sha256": digest(Path(metadata)),
            "anchor": "ASSUMED: AOI geographic bounds centre, pilot low bracket -50 minus 20 m; true anchor absent",
            "normalization": (
                "Legacy branch: R/T absent => identity (centre 0, scale 1, z-shift 0); otherwise p99 radius and p1 z"
            ),
        },
    }
    (output / "georeference.json").write_text(json.dumps(geo, indent=2) + "\n")
    hashes["georeference.json"] = digest(output / "georeference.json")
    contract = {
        "version": 1,
        "profile": "terrasplat-prepared-enu-v1",
        "normalization_applied": True,
        "requires_eval": True,
        "point_count": len(xyz),
        "train_catalog_ids": names["train"],
        "holdout_catalog_ids": names["test"],
        "input_sha256": hashes,
        "assumed_anchor": True,
    }
    (output / "terrasplat-scene.json").write_text(json.dumps(contract, indent=2) + "\n")
    # Verify on a disposable copy: the legacy loader writes points3D.ply.
    return geo


def verify_legacy(source, prepared):
    import tempfile

    sys.path.insert(0, str(ROOT / ".external/skyfall-gs"))
    from scene.dataset_readers import readSatelliteInfo

    with tempfile.TemporaryDirectory(prefix="jax068-legacy-") as temporary:
        copied = Path(temporary) / "scene"
        shutil.copytree(source, copied)
        legacy = readSatelliteInfo(str(copied), False, True)
        actual = readSatelliteInfo(str(Path(prepared).resolve()), False, True)
        point_error = float(np.max(np.abs(legacy.point_cloud.points - actual.point_cloud.points)))
        if (len(legacy.train_cameras), len(legacy.test_cameras)) != (
            len(actual.train_cameras), len(actual.test_cameras)
        ):
            raise ValueError("Prepared camera split counts differ from legacy")
        errors = []
        for left, right in zip(legacy.train_cameras + legacy.test_cameras, actual.train_cameras + actual.test_cameras):
            assert left.image_name == right.image_name
            errors.append(float(max(np.max(np.abs(left.R - right.R)), np.max(np.abs(left.T - right.T)))))
        if point_error > 1e-4 or max(errors) > 1e-6:
            raise ValueError("Prepared scene differs from legacy loader")
    report = {
        "status": "passed",
        "assumed_anchor": True,
        "point_max_error": point_error,
        "camera_max_error": max(errors),
        "train": len(actual.train_cameras),
        "test": len(actual.test_cameras),
    }
    (Path(prepared) / "legacy-loader-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT.parent / "Ref/Skyfall-GS/data/datasets_JAX/JAX_068"))
    parser.add_argument("--output", default="artifacts/imagery/ref-jax068/scene")
    parser.add_argument("--metadata", default="artifacts/reference/dfc2019-jax068/JAX_068_DSM.txt")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--patch-profile", default="basic-stage1-registered", choices=["basic-stage1-registered", "stage1-pseudo-depth"]
    )
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_legacy(args.source, args.output), indent=2))
    else:
        print(json.dumps(prepare(args.source, args.output, args.metadata, args.patch_profile), indent=2))


if __name__ == "__main__":
    main()
