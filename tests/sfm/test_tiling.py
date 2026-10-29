import numpy as np
import pytest

from pipeline.tiling import (
    ENU_TO_GLTF,
    SH_C0,
    SH_C1,
    RECORD,
    assign_bounds,
    bake_view_colour,
    build_tree,
    covariances,
    extents,
    gltf_payload,
    matrix_quaternions,
    merge_voxels,
    quaternion_matrices,
    sigmoid,
    to_enu,
    validate_profile,
    validate_tree,
    walk,
)

PROFILE = {"version": 1, "block_half_extent_m": 125, "leaf_max_splats": 64,
           "extent_sigma": 3, "max_opacity": 0.99}


def random_splats(count, seed=0, spread=100.0):
    rng = np.random.default_rng(seed)
    quaternions = rng.normal(size=(count, 4))
    return {
        "centres": rng.uniform(-spread, spread, size=(count, 3)),
        "dc": rng.normal(size=(count, 3)),
        "rest": np.zeros((count, 0, 3)),
        "opacity_logit": rng.normal(size=count),
        "log_scales": rng.uniform(-2, 0.5, size=(count, 3)),
        "quaternions": quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True),
    }


def test_quaternion_matrix_round_trip():
    splats = random_splats(200)
    recovered = matrix_quaternions(quaternion_matrices(splats["quaternions"]))
    same = np.abs(np.sum(recovered * splats["quaternions"], axis=1))
    assert np.allclose(same, 1, atol=1e-9)


def test_gltf_payload_rotates_positions_and_covariances_into_y_up():
    splats = random_splats(50)
    records = gltf_payload(splats).astype(np.float64)
    assert records.shape == (50, RECORD)
    east, north, up = splats["centres"].T
    assert np.allclose(records[:, :3], np.stack([east, up, -north], 1), atol=1e-4)
    xyzw = records[:, 6:10]
    gltf_covariance = covariances(np.exp(records[:, 3:6]), xyzw[:, [3, 0, 1, 2]])
    enu_covariance = covariances(np.exp(splats["log_scales"]), splats["quaternions"])
    expected = ENU_TO_GLTF @ enu_covariance @ ENU_TO_GLTF.T
    assert np.allclose(gltf_covariance, expected, rtol=1e-4, atol=1e-6)


def test_singleton_voxels_keep_their_splat():
    splats = random_splats(20, spread=500)
    merged = merge_voxels(splats, 0.01, 0.99)
    order = np.lexsort(splats["centres"].T[::-1])
    merged_order = np.lexsort(merged["centres"].T[::-1])
    assert np.allclose(merged["centres"][merged_order], splats["centres"][order])
    assert np.allclose(np.sort(merged["log_scales"][merged_order], 1),
                       np.sort(splats["log_scales"][order], 1), atol=1e-6)
    assert np.allclose(merged["opacity_logit"][merged_order], splats["opacity_logit"][order], atol=1e-6)


def test_merge_matches_weighted_mean_and_spread():
    splats = random_splats(2)
    splats["centres"] = np.array([[0.1, 0.1, 0.1], [0.5, 0.2, 0.3]])
    merged = merge_voxels(splats, 1.0, 0.99)
    assert len(merged["centres"]) == 1
    scales = np.exp(splats["log_scales"])
    weight = sigmoid(splats["opacity_logit"]) * np.prod(scales, 1) ** (2 / 3)
    mean = weight @ splats["centres"] / weight.sum()
    assert np.allclose(merged["centres"][0], mean)
    covariance = covariances(np.exp(merged["log_scales"]), merged["quaternions"])[0]
    # Merged covariance must contain the member spread along the separation axis.
    axis = splats["centres"][1] - splats["centres"][0]
    axis /= np.linalg.norm(axis)
    assert axis @ covariance @ axis > 0


def test_view_colour_bake_is_identity_without_rest_and_uses_directions_with_it():
    splats = random_splats(10)
    cameras = np.array([[0, 0, 1e6], [1e6, 0, 1e6]])
    dc = np.abs(splats["dc"]) * 0.1
    assert np.allclose(bake_view_colour(splats["centres"], dc, np.zeros((10, 0, 3)), cameras), dc)
    rest = np.zeros((10, 8, 3))
    rest[:, 1] = 0.1  # z linear term; 3DGS view directions run camera -> splat, so z = -1 from above
    baked = bake_view_colour(splats["centres"], dc, rest, cameras[:1])
    assert np.allclose(baked, dc - SH_C1 * 0.1 / SH_C0, atol=1e-6)


def test_to_enu_scales_positions_and_log_scales_and_rejects_tilt():
    splats = random_splats(5)
    matrix = np.diag([2.0, 2.0, 2.0, 1.0])
    matrix[:3, 3] = [1, 2, 3]
    enu = to_enu(splats, matrix.tolist())
    assert np.allclose(enu["centres"], 2 * splats["centres"] + [1, 2, 3])
    assert np.allclose(enu["log_scales"], splats["log_scales"] + np.log(2))
    rotated = matrix.copy()
    rotated[:3, :3] = [[2, 0, 0], [0, 0, -2], [0, 2, 0]]
    with pytest.raises(ValueError, match="rotation"):
        to_enu(splats, rotated.tolist())


def tree(count=600):
    splats = random_splats(count, spread=120)
    root = build_tree(splats, PROFILE, 0.335, ((-125, -125), 250))
    assign_bounds(root, PROFILE["extent_sigma"])
    return root, splats


def test_tree_follows_the_tiler_rules():
    root, splats = tree()
    validate_tree(root, len(splats["centres"]), PROFILE["extent_sigma"])
    tiles = list(walk(root))
    assert len(tiles) > 1
    for tile, ancestors in tiles:
        assert len(tile["splats"]["centres"]) <= PROFILE["leaf_max_splats"]
        if ancestors:
            assert tile["geometric_error"] < ancestors[-1]["geometric_error"]
    leaf_total = sum(len(t["splats"]["centres"]) for t, _ in tiles if not t["children"])
    assert leaf_total == len(splats["centres"])


def test_tree_validation_rejects_bounds_that_miss_a_splat():
    root, splats = tree()
    leaf = next(t for t, _ in walk(root) if not t["children"])
    low, high = leaf["bounds"]
    reach = extents(leaf["splats"], PROFILE["extent_sigma"])
    leaf["bounds"] = (low, np.max(leaf["splats"]["centres"] + reach, 0) - 0.5)
    with pytest.raises(ValueError, match="exceed bounds"):
        validate_tree(root, len(splats["centres"]), PROFILE["extent_sigma"])


def test_profile_validation():
    assert validate_profile(dict(PROFILE)) == PROFILE
    with pytest.raises(ValueError):
        validate_profile({**PROFILE, "max_opacity": 1.0})
    with pytest.raises(ValueError):
        validate_profile({**PROFILE, "leaf_max_splats": 10})
