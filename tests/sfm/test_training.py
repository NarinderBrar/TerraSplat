import json
from pathlib import Path

import pytest

from pipeline.phase2 import digest
from pipeline.training import copy_scene, training_command, validate_profile


@pytest.fixture
def profile():
    return json.loads(Path("configs/train-2015-pilot.json").read_text())


def test_resume_keeps_full_learning_rate_and_densification_schedule(profile):
    command = training_command(
        "python", "input", "model", "metrics", profile, 30000, "owned.pth"
    )
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
    (source / "terrasplat-scene.json").write_text(
        json.dumps({"input_sha256": {"georeference.json": digest(georef)}})
    )
    copy_scene(source, tmp_path / "copy")
    assert (tmp_path / "copy/georeference.json").read_bytes() == georef.read_bytes()
    georef.write_text('{"anchor":"changed"}')
    with pytest.raises(ValueError, match="payload changed"):
        copy_scene(source, tmp_path / "failed")
