"""Plan georeferenced 3D Tiles for a fused Stage 1 export.

Python owns geometry: model->ENU, the block crop, the quadtree, parent LOD
merges, bounds and geometric error. `scripts/encode-tiles.mjs` then encodes
each planned tile as an SPZ glb with the Phase 0 profile and writes tileset.json.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from pipeline.export import apply_transform, column_major, matrix4
from pipeline.phase2 import digest

ROOT = Path(__file__).resolve().parents[1]

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
# Floats per splat in a tile payload, already in glTF axes:
# position(3), log scale(3), quaternion xyzw(4), opacity logit(1), SH DC rgb(3).
RECORD = 14
# glTF is right-handed Y-up: (east, north, up) -> (east, up, -north),
# a -90 degree rotation about east.
ENU_TO_GLTF = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
ENU_TO_GLTF_QUAT = np.array([math.cos(math.pi / 4), -math.sin(math.pi / 4), 0, 0])


def validate_profile(profile):
    if profile.get("version") != 1:
        raise ValueError("Unsupported tiling profile")
    for key in ("block_half_extent_m", "extent_sigma", "max_opacity"):
        if not (isinstance(profile[key], (int, float)) and profile[key] > 0):
            raise ValueError(f"Positive number required: {key}")
    if type(profile["leaf_max_splats"]) is not int or profile["leaf_max_splats"] < 64:
        raise ValueError("leaf_max_splats must be an integer of at least 64")
    if not profile["max_opacity"] < 1:
        raise ValueError("max_opacity must be below 1")
    return profile


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def logit(values):
    return np.log(values / (1 - values))


def quaternion_matrices(quaternions):
    """Rotation matrices for (w, x, y, z) quaternions, normalized first."""
    q = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
        ],
        1,
    )


def matrix_quaternions(matrices):
    """(w, x, y, z) quaternions for proper rotation matrices (Shepperd's method)."""
    m = matrices
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    candidates = np.stack(
        [trace, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], 1
    )
    choice = np.argmax(candidates, axis=1)
    q = np.empty((len(m), 4))
    for case in range(4):
        s = choice == case
        if not s.any():
            continue
        a = m[s]
        if case == 0:
            r = np.sqrt(1 + trace[s]) * 2
            q[s] = np.stack([r / 4, (a[:, 2, 1] - a[:, 1, 2]) / r,
                             (a[:, 0, 2] - a[:, 2, 0]) / r, (a[:, 1, 0] - a[:, 0, 1]) / r], 1)
        else:
            i = case - 1
            j, k = (i + 1) % 3, (i + 2) % 3
            r = np.sqrt(1 + a[:, i, i] - a[:, j, j] - a[:, k, k]) * 2
            values = np.empty((len(a), 4))
            values[:, 0] = (a[:, k, j] - a[:, j, k]) / r
            values[:, 1 + i] = r / 4
            values[:, 1 + j] = (a[:, j, i] + a[:, i, j]) / r
            values[:, 1 + k] = (a[:, k, i] + a[:, i, k]) / r
            q[s] = values
    q *= np.where(q[:, :1] < 0, -1, 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def quaternion_multiply(a, b):
    """Hamilton product of (w, x, y, z) quaternions; `a` may be a single quaternion."""
    a = np.broadcast_to(a, b.shape)
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        1,
    )


def covariances(scales, quaternions):
    rotation = quaternion_matrices(quaternions)
    return rotation @ (scales[:, :, None] ** 2 * np.swapaxes(rotation, 1, 2))


def sh_colour(dc, rest, directions):
    """3DGS degree<=2 SH colour (before the +0.5 shift) for unit view directions."""
    x, y, z = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]
    colour = SH_C0 * dc
    if rest.shape[1] >= 3:
        colour = colour - SH_C1 * y * rest[:, 0] + SH_C1 * z * rest[:, 1] - SH_C1 * x * rest[:, 2]
    if rest.shape[1] >= 8:
        colour = (
            colour
            + SH_C2[0] * x * y * rest[:, 3]
            + SH_C2[1] * y * z * rest[:, 4]
            + SH_C2[2] * (2 * z * z - x * x - y * y) * rest[:, 5]
            + SH_C2[3] * x * z * rest[:, 6]
            + SH_C2[4] * (x * x - y * y) * rest[:, 7]
        )
    return colour


def bake_view_colour(centres, dc, rest, camera_centres):
    """Fold view-dependent SH into DC: the mean rendered colour over the cameras."""
    colours = []
    for camera in camera_centres:
        directions = centres - camera
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        colours.append(np.maximum(sh_colour(dc, rest, directions) + 0.5, 0))
    return (np.mean(colours, axis=0) - 0.5) / SH_C0


def read_fused_ply(path):
    from plyfile import PlyData

    data = PlyData.read(path)["vertex"].data
    names = data.dtype.names

    def columns(prefix):
        selected = sorted(
            (n for n in names if n.startswith(prefix)), key=lambda n: int(n.rsplit("_", 1)[1])
        )
        return np.stack([np.asarray(data[n], dtype=np.float64) for n in selected], 1)

    rest = columns("f_rest_") if any(n.startswith("f_rest_") for n in names) else np.zeros((len(data), 0))
    coefficients = rest.shape[1] // 3
    return {
        "centres": np.stack([np.asarray(data[a], dtype=np.float64) for a in "xyz"], 1),
        "dc": columns("f_dc_"),
        # PLY stores f_rest channel-major; reshape to (splat, coefficient, rgb).
        "rest": rest.reshape(len(data), 3, coefficients).transpose(0, 2, 1),
        "opacity_logit": np.asarray(data["opacity"], dtype=np.float64),
        "log_scales": columns("scale_"),
        "quaternions": columns("rot_"),
    }


def to_enu(splats, model_to_enu):
    """Bake a positive uniform scale and rotation about Up into positions and covariances."""
    matrix = matrix4(model_to_enu, "model_to_enu")
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if scale <= 0:
        raise ValueError("model_to_enu must have positive uniform scale")
    rotation = linear / scale
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
            or not np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-9)):
        raise ValueError("model_to_enu must be a uniform scale with rotation about Up only")
    quaternion = matrix_quaternions(rotation[None])[0]
    return {
        **splats,
        "centres": apply_transform(matrix, splats["centres"]),
        "log_scales": splats["log_scales"] + math.log(scale),
        "quaternions": quaternion_multiply(quaternion, splats["quaternions"]),
    }


def take(splats, mask):
    return {key: value[mask] for key, value in splats.items()}


def merge_voxels(splats, voxel_m, max_opacity):
    """Moment-matched merge of splats sharing a voxel."""
    keys = np.floor(splats["centres"] / voxel_m).astype(np.int64)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    group = group.ravel()
    count = group.max() + 1
    scales = np.exp(splats["log_scales"])
    opacity = sigmoid(splats["opacity_logit"])
    weight = opacity * np.prod(scales, axis=1) ** (2 / 3)
    weight = np.maximum(weight, 1e-12)

    def total(values):
        out = np.zeros((count,) + values.shape[1:])
        np.add.at(out, group, values)
        return out

    weight_sum = total(weight)
    mean = total(weight[:, None] * splats["centres"]) / weight_sum[:, None]
    offset = splats["centres"] - mean[group]
    spread = covariances(scales, splats["quaternions"]) + offset[:, :, None] * offset[:, None, :]
    covariance = total(weight[:, None, None] * spread) / weight_sum[:, None, None]
    values, vectors = np.linalg.eigh(covariance)
    vectors[np.linalg.det(vectors) < 0, :, 0] *= -1
    merged_scales = np.sqrt(np.maximum(values, 1e-12))
    sorted_scales = np.sort(scales, axis=1)
    area = sorted_scales[:, 1] * sorted_scales[:, 2]
    merged_sorted = np.sort(merged_scales, axis=1)
    merged_area = merged_sorted[:, 1] * merged_sorted[:, 2]
    merged_opacity = np.clip(total(opacity * area) / merged_area, 1e-6, max_opacity)
    return {
        "centres": mean,
        "dc": total(weight[:, None] * splats["dc"]) / weight_sum[:, None],
        "rest": np.zeros((count, 0, 3)),
        "opacity_logit": logit(merged_opacity),
        "log_scales": np.log(merged_scales),
        "quaternions": matrix_quaternions(vectors),
    }


def extents(splats, sigma):
    """Axis-aligned half-extent of each splat's sigma ellipsoid."""
    covariance = covariances(np.exp(splats["log_scales"]), splats["quaternions"])
    return sigma * np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))


def bounds(splats, sigma):
    if not len(splats["centres"]):
        return None
    reach = extents(splats, sigma)
    return np.min(splats["centres"] - reach, 0), np.max(splats["centres"] + reach, 0)


def union(*boxes):
    boxes = [b for b in boxes if b is not None]
    return np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)


def build_tree(splats, profile, leaf_error_m, square, tile_id="r"):
    """Quadtree over (east, north). Leaves keep full-resolution splats."""
    (west, south), side = square
    if len(splats["centres"]) <= profile["leaf_max_splats"]:
        return {"id": tile_id, "splats": splats, "geometric_error": leaf_error_m,
                "voxel_m": 0.0, "children": []}
    half = side / 2
    east_mid, north_mid = west + half, south + half
    e, n = splats["centres"][:, 0], splats["centres"][:, 1]
    quadrants = [(e < east_mid) & (n < north_mid), (e >= east_mid) & (n < north_mid),
                 (e < east_mid) & (n >= north_mid), (e >= east_mid) & (n >= north_mid)]
    corners = [(west, south), (east_mid, south), (west, north_mid), (east_mid, north_mid)]
    children = [
        build_tree(take(splats, mask), profile, leaf_error_m, (corner, half), f"{tile_id}{index}")
        for index, (mask, corner) in enumerate(zip(quadrants, corners))
        if mask.any()
    ]
    # Parent voxels at least double the children's, so error strictly decreases downward.
    voxel = max([2 * c["voxel_m"] for c in children] + [side / 256, leaf_error_m])
    while True:
        merged = merge_voxels(splats, voxel, profile["max_opacity"])
        if len(merged["centres"]) <= profile["leaf_max_splats"]:
            break
        voxel *= 2
    return {"id": tile_id, "splats": merged, "geometric_error": voxel * math.sqrt(3),
            "voxel_m": voxel, "children": children}


def assign_bounds(tile, sigma):
    for child in tile["children"]:
        assign_bounds(child, sigma)
    tile["bounds"] = union(bounds(tile["splats"], sigma), *(c["bounds"] for c in tile["children"]))


def walk(tile, ancestors=()):
    yield tile, ancestors
    for child in tile["children"]:
        yield from walk(child, ancestors + (tile,))


def validate_tree(root, cropped_count, sigma):
    """Independent checks of the tiler rules; raises on any violation."""
    leaves = 0
    for tile, ancestors in walk(root):
        splats = tile["splats"]
        for key, value in splats.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {key} in tile {tile['id']}")
        if not tile["geometric_error"] > 0:
            raise ValueError(f"Tile {tile['id']} has zero geometric error")
        for parent in ancestors[-1:]:
            if not tile["geometric_error"] < parent["geometric_error"]:
                raise ValueError(f"Tile {tile['id']} error does not decrease below its parent")
        if not tile["children"]:
            leaves += len(splats["centres"])
        reach = extents(splats, sigma)
        low, high = splats["centres"] - reach, splats["centres"] + reach
        for owner in (tile, *ancestors):
            box_low, box_high = owner["bounds"]
            if (low < box_low - 1e-6).any() or (high > box_high + 1e-6).any():
                raise ValueError(f"Tile {tile['id']} splats exceed bounds of {owner['id']}")
    if leaves != cropped_count:
        raise ValueError(f"Leaves hold {leaves} splats, expected {cropped_count}")


def gltf_payload(splats):
    """Float32 records in glTF axes; the encoder writes them to SPZ unchanged."""
    centres = splats["centres"] @ ENU_TO_GLTF.T
    quaternions = quaternion_multiply(ENU_TO_GLTF_QUAT, splats["quaternions"])
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    records = np.concatenate(
        [centres, splats["log_scales"], quaternions[:, [1, 2, 3, 0]],
         splats["opacity_logit"][:, None], splats["dc"]],
        1,
    )
    return records.astype("<f4")


def box_volume(low, high):
    centre, half = (low + high) / 2, (high - low) / 2
    return [*centre, half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]]


def camera_centres(scene):
    frames = json.loads((Path(scene) / "transforms_train.json").read_text())["frames"]
    return np.array([np.asarray(f["transform_matrix"], dtype=np.float64)[:3, 3] for f in frames])


def plan(export_dir, output, profile, grid=None, cell=None):
    profile = validate_profile(profile)
    export_dir, output = Path(export_dir).resolve(), Path(output).resolve()
    manifest_path = export_dir / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "exported":
        raise ValueError("A completed export is required")
    fused = Path(manifest["fused_ply"]["path"])
    if digest(fused) != manifest["fused_ply"]["sha256"]:
        raise ValueError("Fused PLY changed after export")
    # Models trained elsewhere have no run directory; their manifest names the scene they were trained on.
    scene = manifest.get("camera_scene") or json.loads(
        (Path(manifest["run_directory"]) / "training-report.json").read_text())["input_scene"]
    georeference = manifest["georeference"]

    splats = read_fused_ply(fused)
    if len(splats["centres"]) != manifest["splat_count"]:
        raise ValueError("Fused PLY splat count differs from the export manifest")
    cameras = camera_centres(scene)
    splats["dc"] = bake_view_colour(splats["centres"], splats["dc"], splats["rest"], cameras)
    splats["rest"] = np.zeros((len(splats["centres"]), 0, 3))
    splats = to_enu(splats, georeference["model_to_enu"])

    half = profile["block_half_extent_m"]
    owned = (np.abs(splats["centres"][:, 0]) <= half) & (np.abs(splats["centres"][:, 1]) <= half)
    ownership = {}
    if grid is not None:
        from pipeline.ownership import owned_mask, cell_corners, grid_partition_check
        if cell not in grid['cells']:
            raise ValueError(f'Unknown ownership cell: {cell}')
        if not grid_partition_check(grid)['passed']:
            raise ValueError('Ownership grid has a gap or overlap')
        owned = owned_mask(splats['centres'], georeference, grid, cell)
        ownership = {'frame': 'half-open cell in the shared grid ENU frame',
                     'ownership_grid': grid, 'ownership_cell': cell,
                     'polygon_enu_m': cell_corners(grid, cell, georeference['enu_to_ecef']).tolist()}
    cropped = take(splats, owned)
    if not owned.any():
        raise ValueError('No splats inside the owned crop')
    leaf_error = float(manifest["round_trip"]["gsd_m"])
    low = np.minimum(cropped['centres'][:, :2].min(0), [-half, -half])
    high = np.maximum(cropped['centres'][:, :2].max(0), [half, half])
    root = build_tree(cropped, profile, leaf_error, (low, float((high-low).max())))
    assign_bounds(root, profile["extent_sigma"])
    validate_tree(root, int(owned.sum()), profile["extent_sigma"])

    if output.exists():
        raise ValueError("Use a new tiles directory; remove the previous one first")
    (output / "payload").mkdir(parents=True)
    tiles = []
    for tile, ancestors in walk(root):
        payload = gltf_payload(tile["splats"])
        path = output / "payload" / f"{tile['id']}.bin"
        path.write_bytes(payload.tobytes())
        low, high = tile["bounds"]
        tiles.append({
            "id": tile["id"],
            "parent": ancestors[-1]["id"] if ancestors else None,
            "splats": len(payload),
            "payload": f"payload/{tile['id']}.bin",
            "payload_sha256": digest(path),
            "geometric_error": tile["geometric_error"],
            "merge_voxel_m": tile["voxel_m"],
            "box": box_volume(low, high),
            "bounds_enu_m": {"min": low.tolist(), "max": high.tolist()},
        })
    reach = extents(cropped, profile["extent_sigma"])
    result = {
        "schema_version": 1,
        "status": "planned",
        "profile": profile,
        "export_manifest_sha256": digest(manifest_path),
        "fused_ply_sha256": manifest["fused_ply"]["sha256"],
        "record_layout": "float32 LE x14: gltf position(3), log scale(3), "
                         "quaternion xyzw(4), opacity logit(1), SH DC rgb(3)",
        "axes": "payload in glTF Y-up (east, up, -north); bounding boxes in tile Z-up ENU",
        "root_transform": column_major(georeference["enu_to_ecef"]),
        "root_transform_equation": "root.transform = enu_to_ecef; model_to_enu is baked into the splats",
        "enu_anchor": georeference["anchor"],
        "colour": {
            "sh_degree": 0,
            "method": "Mean 3DGS SH colour over the training camera directions, folded into DC",
            "training_cameras": len(cameras),
        },
        "crop": {
            "frame": "metric ENU square centred on the anchor",
            **ownership,
            "half_extent_m": half,
            "input_splats": len(owned),
            "owned_splats": int(owned.sum()),
            "removed_splats": int((~owned).sum()),
            "owned_z_range_m": [float(cropped["centres"][:, 2].min()),
                                float(cropped["centres"][:, 2].max())],
            "max_extent_m": float(reach.max()),
            "p99_extent_m": float(np.percentile(reach.max(1), 99)),
        },
        "levels": max(len(a) for _, a in walk(root)) + 1,
        "tiles": tiles,
        "rules": manifest["tileset"]["rules"],
        "notes": [
            "The block is a square in ENU; the source block is a UTM square, about 0.3 degrees apart here.",
            "View-dependent SH is baked at the training directions; SH degree >= 1 export awaits a "
            "verified Cesium SH convention.",
            "Independent placement and height accuracy remain unverified.",
        ],
    }
    (output / "tile-plan.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="Export directory with export-manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default=str(ROOT / "configs/tiles-2015-pilot.json"))
    parser.add_argument("--grid", help="Shared ENU ownership grid")
    parser.add_argument("--cell", help="Cell name in the ownership grid")
    args = parser.parse_args()
    if bool(args.grid) != bool(args.cell):
        parser.error('--grid and --cell must be provided together')
    result = plan(args.export, args.output, json.loads(Path(args.profile).read_text()),
                  json.loads(Path(args.grid).read_text()) if args.grid else None, args.cell)
    print(json.dumps({k: result[k] for k in ("status", "crop", "levels")}, indent=2))
    for tile in result["tiles"]:
        print(tile["id"], tile["splats"], round(tile["geometric_error"], 3))


if __name__ == "__main__":
    main()
