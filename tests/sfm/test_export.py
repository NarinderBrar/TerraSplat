import json

import numpy as np
import pytest

from pipeline.export import (
    column_major,
    enu_bbox_splat_centres,
    gsd_from_selection,
    round_trip,
    validate_attributes,
    verify_run,
)
from pipeline.georeference import placement
from pipeline.phase2 import digest


def attributes(points, opacity=None, scale=0.0):
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    result = {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "f_dc_0": np.zeros(count),
        "f_dc_1": np.zeros(count),
        "f_dc_2": np.zeros(count),
        "opacity": np.full(count, opacity) if opacity is not None else np.zeros(count),
        "scale_0": np.full(count, scale),
        "scale_1": np.full(count, scale),
        "scale_2": np.full(count, scale),
        "rot_0": np.ones(count),
        "rot_1": np.zeros(count),
        "rot_2": np.zeros(count),
        "rot_3": np.zeros(count),
    }
    return result


def georeference():
    return placement(
        {"longitude": -81.66376056134547, "latitude": 30.348819714393777,
         "ellipsoidal_height_m": -70},
        {
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [16.3, 11.9, 49.3],
            "scale": 1.1745735791433705,
            "z_shift": -8.072884706269226,
        },
    )


def test_column_major_matches_3d_tiles_layout():
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 1]]
    assert column_major(matrix) == [1, 5, 9, 0, 2, 6, 10, 0, 3, 7, 11, 0, 4, 8, 12, 1]


def test_round_trip_uses_the_recorded_transform_chain():
    points = np.array([[-100, -80, 10], [0, 0, 50], [90, 70, 120]], dtype=float)
    result = round_trip(attributes(points), georeference(), 0.335)
    assert result["gate_m"] == pytest.approx(0.0335)
    assert result["max_error_m"] < 1e-6
    assert result["max_model_to_ecef_composition_error_m"] < 1e-6
    assert result["passed"]


def test_round_trip_rejects_model_to_ecef_that_disagrees_with_the_chain():
    points = np.array([[0, 0, 50]], dtype=float)
    record = georeference()
    shifted = np.array(record["model_to_ecef"], dtype=float)
    shifted[0, 3] += 1.0
    record["model_to_ecef"] = shifted.tolist()
    result = round_trip(attributes(points), record, 0.335)
    assert result["max_model_to_ecef_composition_error_m"] == pytest.approx(1.0)
    assert not result["passed"]


def test_attribute_validation_activates_opacity_and_scales():
    result = validate_attributes(attributes([[0, 0, 0], [1, 2, 3]]), 2)
    assert result["splat_count"] == 2
    assert result["opacity_after_activation"]["strictly_inside_unit_interval"]
    assert result["scales_after_activation"]["all_positive"]
    assert result["rotation_norm"]["min"] == pytest.approx(1.0)


def test_attribute_validation_rejects_count_nonfinite_and_saturated_opacity():
    with pytest.raises(ValueError, match="splat count"):
        validate_attributes(attributes([[0, 0, 0]]), 2)
    nonfinite = attributes([[0, 0, 0]])
    nonfinite["x"][0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_attributes(nonfinite, 1)
    with pytest.raises(ValueError, match="Opacity"):
        validate_attributes(attributes([[0, 0, 0]], opacity=100.0), 1)


def test_enu_bbox_is_reported_and_checked_against_the_envelope():
    geo = georeference()
    geo["model_to_enu"] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    inside = enu_bbox_splat_centres(
        attributes([[-100, -100, 0], [100, 100, 50]], opacity=0.0), geo
    )
    assert inside["min_m"] == [-100.0, -100.0, 0.0]
    assert inside["max_m"] == [100.0, 100.0, 50.0]
    assert inside["plausible"]
    outside = enu_bbox_splat_centres(
        attributes([[-100, -100, 0], [400, 100, 50]], opacity=0.0), geo
    )
    assert not outside["plausible"]


def test_gsd_is_the_finest_selected_native_value():
    selection = {
        "train": [{"catalog_id": "a", "gsd_m": 0.335}, {"catalog_id": "b", "gsd_m": 0.361}],
        "holdout": [{"catalog_id": "c", "gsd_m": 0.337}],
    }
    result = gsd_from_selection(selection, ["a", "b", "c"])
    assert result["gsd_m"] == pytest.approx(0.335)
    with pytest.raises(ValueError, match="missing GSD"):
        gsd_from_selection(selection, ["d"])


def test_verify_run_checks_final_model_and_georeference_hashes(tmp_path):
    model = tmp_path / "model"
    (model / "point_cloud/iteration_10").mkdir(parents=True)
    (model / "point_cloud/iteration_0").mkdir()
    final = model / "point_cloud/iteration_10/point_cloud.ply"
    final.write_bytes(b"final")
    initial = model / "point_cloud/iteration_0/point_cloud.ply"
    initial.write_bytes(b"initial")
    georef = model / "georeference.json"
    georef.write_text(json.dumps({"matrix_layout": "row-major", "anchor": {},
                                 "model_to_enu": [], "enu_to_ecef": [],
                                 "model_to_ecef": []}))
    report = {
        "status": "trained",
        "model": str(model),
        "input_scene": str(tmp_path / "input"),
        "georeference_sha256": digest(georef),
        "initial_ply_sha256": digest(initial),
        "final_splats": 1,
        "phases": [{"target": 10, "outputs": {
            "splats": 1,
            "sha256": {"point_cloud/iteration_10/point_cloud.ply": digest(final)},
        }}],
    }
    (tmp_path / "training-report.json").write_text(json.dumps(report))
    verify_run(tmp_path)
    georef.write_text("{}")
    with pytest.raises(ValueError, match="georeference changed"):
        verify_run(tmp_path)


def test_floater_removal_keeps_boundary():
    from pipeline.export import floater_removal

    geo = {"model_to_enu": np.eye(4)}
    data = attributes([[0, 0, 0], [350, 0, 250], [0, 0, -250]])
    keep, report = floater_removal(data, geo)
    assert keep.tolist() == [True, True, True]
    assert report["removed_count"] == 0
    assert report["removed_opacity_mass_fraction"] == 0
    assert report["removed_radius_range_m"] is None
    assert report["removed_up_range_m"] is None
    assert report["passed"]
    assert enu_bbox_splat_centres(data, geo)["plausible"]


def test_floater_removal_removes_splats_beyond_vertical_envelope():
    from pipeline.export import floater_removal

    data = attributes([[0, 0, 0]] * 98 + [[0, 0, 251], [10, 0, -260]])
    keep, report = floater_removal(data, {"model_to_enu": np.eye(4)})
    assert keep.tolist() == [True] * 98 + [False, False]
    assert report["removed_up_range_m"] == [-260, 251]
    assert report["removed_opacity_mass_fraction"] == pytest.approx(0.02)
    assert not report["passed"]


def test_floater_removal_uses_enu_radius_and_opacity_mass():
    from pipeline.export import floater_removal

    transform = np.eye(4)
    transform[0, 3] = 10
    data = attributes([[0, 0, 0], [341, 0, 0], [0, 351, 0]])
    data["opacity"] = np.array([0., -8., -9.])
    keep, report = floater_removal(data, {"model_to_enu": transform})
    assert keep.tolist() == [True, False, False]
    mass = 1 / (1 + np.exp(-data["opacity"]))
    assert report["removed_count"] == 2
    assert report["removed_opacity_mass_fraction"] == pytest.approx(mass[1:].sum() / mass.sum())
    assert report["removed_radius_range_m"] == pytest.approx([351, np.hypot(10, 351)])
    assert report["passed"]


def test_export_rejects_more_than_one_percent_removed_mass_before_fusion(tmp_path, monkeypatch):
    import pipeline.export as exporter

    data = attributes([[0, 0, 0], [351, 0, 0]])
    training = {"profile": {}, "train_catalog_ids": [], "holdout_catalog_ids": []}
    monkeypatch.setattr(exporter, "verify_run", lambda run: (
        tmp_path, training, tmp_path, tmp_path / "raw.ply", {"model_to_enu": np.eye(4)},
    ))
    monkeypatch.setattr(exporter, "read_attributes", lambda path: data)
    with pytest.raises(ValueError, match="opacity-mass fraction exceeds 1%"):
        exporter.export(tmp_path, gsd=0.3)
    report = json.loads((tmp_path / "export/export-manifest.json").read_text())
    assert report["status"] == "failed"
    assert report["floater_removal"]["removed_opacity_mass_fraction"] == 0.5
    assert not report["floater_removal"]["passed"]
    assert not list((tmp_path / "export").glob("*.ply"))


def test_floater_removal_allows_exactly_one_percent_mass():
    from pipeline.export import floater_removal

    data = attributes([[0, 0, 0]] * 99 + [[351, 0, 0]])
    _, report = floater_removal(data, {"model_to_enu": np.eye(4)})
    assert report["removed_opacity_mass_fraction"] == 0.01
    assert report["passed"]


def test_mask_gaussians_preserves_order_and_camera_appearance():
    from types import SimpleNamespace
    from pipeline.export import mask_gaussians

    torch = pytest.importorskip("torch")
    per_splat = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_embeddings")
    model = SimpleNamespace(**{name: torch.nn.Parameter(torch.arange(6.).reshape(3, 2)) for name in per_splat})
    model.filter_3D = torch.arange(3.).reshape(3, 1)
    camera_embeddings = torch.arange(8.).reshape(4, 2)
    model.appearance_embeddings = camera_embeddings
    mask_gaussians(model, np.array([True, False, True]))
    for name in per_splat:
        assert torch.equal(getattr(model, name), torch.tensor([[0., 1.], [4., 5.]]))
    assert torch.equal(model.filter_3D, torch.tensor([[0.], [2.]]))
    assert model.appearance_embeddings is camera_embeddings
