import io
import tarfile

import pytest

from pipeline.acquisitions import parse_sidecars, select


def sidecar_archive():
    fields = dict(
        satId='"WV03"',
        bandId='"P"',
        CatId='"VIEW_1"',
        productOrderId='"PRODUCT_1"',
        firstLineTime="2015-01-01T12:00:00Z",
        numColumns="512",
        numRows="512",
        cloudCover="0.05",
        meanSatAz="90",
        meanSatEl="65",
        meanOffNadirViewAngle="25",
        meanProductGSD="0.35",
        meanSunEl="60",
    )
    imd = "\n".join(f"{key} = {value};" for key, value in fields.items())
    rpb = 'SpecId = "RPC00B";\nBEGIN_GROUP = IMAGE\n'
    for key in ("heightOffset", "latOffset", "longOffset", "lineOffset", "sampOffset"):
        rpb += f"{key} = 0;\n"
    for key in ("heightScale", "latScale", "longScale", "lineScale", "sampScale"):
        rpb += f"{key} = 1;\n"
    for key in ("errBias", "errRand"):
        rpb += f"{key} = 1;\n"
    for key in ("lineNumCoef", "lineDenCoef", "sampNumCoef", "sampDenCoef"):
        rpb += f"{key} = (" + ",".join(["1"] + ["0"] * 19) + ");\n"
    rpb += "END_GROUP = IMAGE\nEND;"
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, text in (("scene_PAN/view.IMD", imd), ("scene_PAN/view.RPB", rpb)):
            data = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def records():
    return [
        dict(
            catalog_id=f"view-{i}",
            capture_time=f"2015-01-0{i + 1}T12:00:00Z",
            cloud_cover=0,
            sat_azimuth_deg=i * 40,
            sat_elevation_deg=65,
            off_nadir_deg=25,
            sun_elevation_deg=60,
            coverage={"status": "covered"},
            msi={"uri": f"msi-{i}"},
        )
        for i in range(4)
    ]


def test_parse_real_archive_structure_retains_rpc_order_and_quality_fields():
    result = parse_sidecars(sidecar_archive())
    assert result["capture_time"] == "2015-01-01T12:00:00Z"
    assert result["cloud_cover"] == 0.05
    assert result["rpc"]["LINE_DEN_COEFF"].split() == ["1.0"] + ["0.0"] * 19
    assert result["width"] == 512


def test_selection_is_deterministic_and_reserves_a_whole_view():
    source = records()
    result = select({"block": {}, "records": source})
    reversed_result = select({"block": {}, "records": list(reversed(source))})
    assert result == reversed_result
    assert result["status"] == "candidate"
    assert len(result["train"]) == 3 and len(result["holdout"]) == 1
    assert {r["catalog_id"] for r in result["train"]}.isdisjoint(
        r["catalog_id"] for r in result["holdout"]
    )
    assert result["span_days"] == 3


def test_selection_does_not_silently_widen_time_window():
    source = records()
    source[-1]["capture_time"] = "2015-01-20T12:00:00Z"
    result = select({"block": {}, "records": source})
    assert result["status"] == "blocked"
    assert result["policy"]["max_span_days"] == 7


@pytest.mark.parametrize(
    "change",
    [
        {"cloud_cover": 0.3},
        {"coverage": {"status": "rejected"}},
        {"msi": None},
        {"catalog_id": "view-0"},
    ],
)
def test_one_unusable_or_duplicate_acquisition_cannot_satisfy_view_count(change):
    source = records()
    source[-1].update(change)
    assert select({"block": {}, "records": source})["status"] == "blocked"


def test_training_angles_are_checked_after_holdout_removal():
    source = records()
    for r in source[:3]:
        r["sat_azimuth_deg"] = 90
    assert select({"block": {}, "records": source})["status"] == "blocked"


def test_invalid_epoch_policy_is_rejected():
    with pytest.raises(ValueError):
        select({"block": {}, "records": records()}, max_span_days=float("nan"))
