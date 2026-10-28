"""Explicit Skyfall normalization and WGS84 placement, in row-major matrices."""
import math


def multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def placement(anchor, normalization):
    lon, lat, height = (anchor[k] for k in ("longitude", "latitude", "ellipsoidal_height_m"))
    if not all(math.isfinite(v) for v in (lon, lat, height)) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError("Anchor must have finite WGS84 longitude, latitude and ellipsoidal height")
    rotation = normalization["rotation"]
    translation = normalization["translation"]
    scale, shift = normalization["scale"], normalization["z_shift"]
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation) or len(translation) != 3:
        raise ValueError("Normalization requires a 3x3 rotation and three translation values")
    values = [v for row in rotation for v in row] + translation + [scale, shift]
    if not all(math.isfinite(v) for v in values) or scale <= 0:
        raise ValueError("Normalization values must be finite and scale must be positive")
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
            if abs(dot - (1 if i == j else 0)) > 1e-8:
                raise ValueError("Normalization rotation must be orthonormal")
    r = rotation
    determinant = sum(r[0][i] * (r[1][(i + 1) % 3] * r[2][(i + 2) % 3] - r[1][(i + 2) % 3] * r[2][(i + 1) % 3]) for i in range(3))
    if abs(determinant - 1) > 1e-8:
        raise ValueError("Normalization rotation must preserve handedness")
    source_to_model = [[scale * r[i][j] for j in range(3)] + [-scale * translation[i] - (shift if i == 2 else 0)] for i in range(3)] + [[0, 0, 0, 1]]
    offset = [translation[0], translation[1], translation[2] + shift / scale]
    model_to_enu = [[r[j][i] / scale for j in range(3)] + [sum(r[j][i] * offset[j] for j in range(3))] for i in range(3)] + [[0, 0, 0, 1]]
    lon, lat = math.radians(lon), math.radians(lat)
    slon, clon, slat, clat = math.sin(lon), math.cos(lon), math.sin(lat), math.cos(lat)
    radius = 6378137 / math.sqrt(1 - 6.6943799901413165e-3 * slat * slat)
    ecef = [(radius + height) * clat * clon, (radius + height) * clat * slon,
            (radius * (1 - 6.6943799901413165e-3) + height) * slat]
    enu_to_ecef = [[-slon, -slat * clon, clat * clon, ecef[0]],
                   [clon, -slat * slon, clat * slon, ecef[1]],
                   [0, clat, slat, ecef[2]], [0, 0, 0, 1]]
    return {"matrix_layout": "row-major", "source_frame": "metric ENU", "model_frame": "Skyfall Z-up",
            "normalization_equation": "model = scale * (R * enu - T) - [0, 0, z_shift]",
            "source_to_model": source_to_model, "model_to_enu": model_to_enu,
            "enu_to_ecef": enu_to_ecef, "model_to_ecef": multiply(enu_to_ecef, model_to_enu)}
