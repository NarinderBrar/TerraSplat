from pipeline.satellite_sfm import metadata, geographic_frame


def test_upstream_metadata_keeps_capture_date_and_crop_offsets():
    rpc = {
        "HEIGHT_OFF": "0",
        "HEIGHT_SCALE": "500",
        "LAT_OFF": "30",
        "LAT_SCALE": ".01",
        "LONG_OFF": "-81",
        "LONG_SCALE": ".01",
        "LINE_OFF": "100",
        "LINE_SCALE": "100",
        "SAMP_OFF": "200",
        "SAMP_SCALE": "200",
        "LINE_NUM_COEFF": " ".join(["0"] * 20),
        "LINE_DEN_COEFF": " ".join(["1"] + ["0"] * 19),
        "SAMP_NUM_COEFF": " ".join(["0"] * 20),
        "SAMP_DEN_COEFF": " ".join(["1"] + ["0"] * 19),
    }
    record = {
        "rpc": rpc,
        "coverage": {
            "window": {"row_off": 10, "col_off": 20, "width": 50, "height": 60}
        },
        "capture_time": "2015-01-01T12:34:56.123Z",
    }
    result = metadata(record)
    assert result["capture_date"] == [2015, 1, 1, 12, 34, 56]
    assert result["rpc"]["rowOff"] == 90 and result["rpc"]["colOff"] == 180
    assert result["rpc"]["latOff"] == 30 and result["rpc"]["lonOff"] == -81
    assert result["width"] == 50 and result["height"] == 60


def test_enu_origin_is_explicit_numerical_reference_not_inferred_ground_height():
    block = dict(
        longitude=-81.66,
        latitude=30.34,
        side_m=250,
        halo_m=20,
        ellipsoidal_height_range_m=[-50, 300],
    )
    bounds, observer = geographic_frame(block)
    assert bounds[0][0] < 30.34 < bounds[0][1]
    assert bounds[1][0] < -81.66 < bounds[1][1]
    assert observer[2] == -70 and bounds[2] == [-50, 300]
