# CORE3D imagery preparation

Planning update: the 2015 CORE3D inputs below are historical validation fixtures only. For a current-city model, qualify recent imagery and date-compatible validation before full training; see [the revised plan](../docs/PLAN_V2.md).

This implements the first raw-imagery step of Phase 2: catalogue the public
Jacksonville WorldView-3 sources and extract sensor-pixel crops while preserving
RPC camera geometry. It does not run SfM or generate calibrated RGB training data.

## Environment

Keep GDAL/Rasterio separate from the verified CUDA training environment.
Python 3.12+ and uv are used here.

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv venv --python /usr/bin/python3 .venv-imagery
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv-imagery/bin/python -r imagery/requirements.txt -c imagery/installed.txt
.venv-imagery/bin/python -m pipeline.imagery catalogue --output artifacts/imagery/catalogue.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-imagery/bin/python -m pytest tests/imagery -q
```

The pytest setting excludes unrelated host plugins, including ROS plugins that
may be exposed through the host's Python path. The tests use real local raster
files and RPC transformers; they make no network calls.

The anonymous S3 listing records each NITF URI, size, ETag and modification date.
ETags identify the selected remote object and are not assumed to be SHA-256
checksums. The initial listing has 26 PAN and 26 MSI NITFs. It does not download
the image collection or silently group acquisitions from different dates.

## Crop a selected image

Select a source URI from the catalogue, then run:

```bash
.venv-imagery/bin/python -m pipeline.imagery crop SOURCE_URI --block configs/jax068-block.json --output artifacts/imagery/selected-crop.tif
```

Replace `SOURCE_URI` with the actual HTTPS URI. Each output path must be new.
The corresponding `.json` records the source identity, source and crop sizes,
pixel window, RPCs, software versions, output SHA-256 and continuity error.

`configs/jax068-block.json` selects a 250 m square with a 20 m halo around
TerraScope's JAX_068 visual-baseline centre. It defines the square in local UTM,
then projects a densified perimeter through each image's RPCs at the endpoints
and midpoint of an explicit ellipsoidal-height bracket. The bracket is a crop
hypothesis, not a measured elevation or a substitute for the missing ENU origin.
Images that do not fully cover the requested window are rejected.

The crop keeps native pixel values and resolution. It subtracts only the crop's
row/column origins from `LINE_OFF`/`SAMP_OFF`; RPC coefficients and all scale
parameters remain unchanged. The output carries both RPC tags and the shifted
source affine. The affine remains an approximate sensor-image location and must
not be used as proof of orthorectification.

The writer verifies all output pixels against the source window and uses GDAL
RPC transformers to check geographic-to-pixel continuity at both height-bracket
endpoints. Remote object identity is checked before and after the read. The
source NITF remains untouched.

A 64 MiB default limit applies to estimated *uncompressed source blocks touched*,
across all bands. It bounds raster work, not measured HTTP transfer bytes.
Rasterio/GDAL may issue multiple range requests and fetch additional metadata.
The CLI enforces a 120-second overall worker deadline in addition to HTTP
timeouts; use `--timeout-seconds` to change it. A timeout rejects the source and
may leave an unaccepted `.partial.tif` for inspection. Whole-scene fallback
downloads are a separate explicit action, not an automatic behavior.

## Executed source test

The first May 1, 2015 PAN NITF opened remotely with RPC metadata in about 2.6
seconds, but the subsequent remote pixel crop was slow and was terminated.
The companion `_lv1.tif` read quickly but was already warped and had no RPC
metadata, so it was rejected as raw reconstruction input.

Following the planned fallback, one 866,900,574-byte PAN NITF was downloaded
explicitly under a 1 GiB source limit and hashed. Cropping it locally succeeded:

- Output: one uint16 band, 908 rows by 1,242 columns.
- Estimated source blocks touched: 8 MiB uncompressed.
- All saved pixels equal the source window; observed range 128–2047.
- Maximum RPC offset-continuity error: 0 pixels.
- Source acquisition timestamp: `20150501160357`.

The source and output checksums are recorded in `imagery/verified-result.json`.
The source NITF and derived GeoTIFF stay in ignored `artifacts/imagery/`.
This verifies one PAN crop, not multi-view coverage, usable color, a successful
SfM reconstruction, or georeferenced splats. Neither matched MSI crops nor a
full imagery set have been prepared yet.

## Multi-view continuation

The approved 13-day pilot now has three training views and one holdout, with
matched PAN/MSI crops, RGB inputs and RPC-derived initial cameras prepared. See
[SatelliteSfM handoff](../sfm/README.md) for the executed result and reproduction.

## Next gates

Validate acquisition epochs, overlap, cloud coverage and angular diversity before
choosing multiple views. Preserve source metadata and acquisition IDs through
RPC-aware preprocessing. Pan-sharpen the matched MSI/PAN data with a validated
color/band convention, then integrate SatelliteSfM and retain its ENU observer
metadata. Existing prepared JAX_068 images remain useful for diagnostics but do
not replace this raw-input provenance.

## References

- [CORE3D baseline and data provenance](https://github.com/pubgeo/core3d-open)
- [Rasterio RPC transforms and height handling](https://rasterio.readthedocs.io/en/stable/topics/transforms.html)
- [Rasterio window I/O and source block behavior](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html)
