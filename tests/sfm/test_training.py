import json
from pathlib import Path

import pytest

from pipeline.phase2 import digest
from pipeline.training import copy_scene, training_command, training_manifest, validate_profile


@pytest.fixture
def profile():
    return json.loads(Path("configs/train-2015-pilot.json").read_text())


def test_resume_keeps_full_learning_rate_and_densification_schedule(profile):
    command = training_command("python", "input", "model", "metrics", profile, 30000, "owned.pth")
    assert "--eval" in command
    assert "--appearance_enabled" not in command
    assert "--convert_SHs_python" in command
    assert command[command.index("--position_lr_max_steps") + 1] == "30000"
    assert command[command.index("--densify_until_iter") + 1] == "20000"
    assert command[command.index("--start_checkpoint") + 1] == "owned.pth"
    assert command[command.index("--test_iterations") + 1] == "-1"
    assert command[command.index("--lambda_pseudo_depth") + 1] == "0"


def test_memory_profile_leaves_device_headroom(profile):
    profile["max_allocated_bytes"] = 6 * 1024**3
    with pytest.raises(ValueError, match="headroom"):
        validate_profile(profile)


def test_copied_training_scene_keeps_georeference_and_rejects_changed_payload(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    georef = source / "georeference.json"
    georef.write_text('{"anchor":"captured"}')
    (source / "terrasplat-scene.json").write_text(json.dumps({"input_sha256": {"georeference.json": digest(georef)}}))
    copy_scene(source, tmp_path / "copy")
    assert (tmp_path / "copy/georeference.json").read_bytes() == georef.read_bytes()
    georef.write_text('{"anchor":"changed"}')
    with pytest.raises(ValueError, match="payload changed"):
        copy_scene(source, tmp_path / "failed")


def test_depth_weight_defaults_to_zero_and_passes_static_weight(profile):
    command = training_command("python", "input", "model", "metrics", profile, 1000)
    assert command[command.index("--lambda_depth") + 1] == "0"
    profile["lambda_depth"] = 0.5
    command = training_command("python", "input", "model", "metrics", profile, 1000)
    assert command[command.index("--lambda_depth") + 1] == "0.5"
    assert command[command.index("--lambda_pseudo_depth") + 1] == "0"


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf"), "0.5", True, None])
def test_depth_weight_must_be_finite_and_nonnegative(profile, weight):
    profile["lambda_depth"] = weight
    with pytest.raises(ValueError, match="lambda_depth"):
        validate_profile(profile)


def test_authors_recipe_keeps_training_only_and_full_pseudo_schedule():
    recipe = json.loads(Path("configs/train-ref-jax068-recipe.json").read_text())
    command = training_command("python", "input", "model", "metrics", recipe, 1500)
    assert "--eval" in command
    assert "--appearance_enabled" in command
    for flag, value in [
        ("lambda_depth", "0"),
        ("lambda_pseudo_depth", "0.5"),
        ("lambda_opacity", "10"),
        ("start_sample_pseudo", "1000"),
        ("end_sample_pseudo", "21000"),
        ("sh_degree", "1"),
        ("densify_grad_threshold", "0.0001"),
    ]:
        assert command[command.index("--" + flag) + 1] == value
    assert command[command.index("--test_iterations") + 1] == "-1"


@pytest.mark.parametrize('key', ['lambda_pseudo_depth', 'lambda_opacity'])
def test_optional_loss_weights_must_be_finite(profile, key):
    profile[key] = float('nan')
    with pytest.raises(ValueError, match=key):
        validate_profile(profile)


def test_recipe_adapter_selection_preserves_prepared_scene(tmp_path):
    original = {
        "prepared_scene": ".", "georeference": "georeference.json",
        "skyfall": "../../.external/skyfall-gs",
        "skyfall_patch_profile": "basic-stage1-registered",
        "expected_skyfall_commit": "pinned",
    }
    path = tmp_path / "phase2-manifest.json"
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    recipe = json.loads(Path("configs/train-2015-pilot-8w-recipe.json").read_text())
    selected = training_manifest(tmp_path, recipe)
    assert selected["prepared_scene"] == str(tmp_path)
    assert selected["georeference"] == str(tmp_path / "georeference.json")
    assert selected["skyfall"].endswith("/.external/skyfall-gs-pseudo")
    assert selected["skyfall_patch_profile"] == "stage1-pseudo-depth"
    assert selected["expected_skyfall_commit"] == "pinned"
    assert path.read_bytes() == before
    assert training_manifest(tmp_path, {})["skyfall"] == str((tmp_path / original["skyfall"]).resolve())
