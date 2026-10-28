# Free imagery candidate request brief

Prepared locally for review; not sent to a provider or programme.

TerraSplat needs a small multiview satellite imagery dataset for a georeferenced Gaussian-splat reconstruction pilot. The pilot location is flexible; prefer an urban block about 250 × 250 m with a 20 m halo. The initial Jacksonville AOI is supplied in [GeoJSON](pilot-aoi.geojson), but another location is acceptable.

Please establish all of the following before selecting a dataset:

1. Free access applicable to an independent developer without a research affiliation, and the exact licence governing reconstruction, derived models and redistribution. State any restrictions on public demos or commercial use; do not presume permission.
2. Capture timestamps, source acquisition identifiers and footprints for at least four overlapping source views: three training views and one whole-acquisition holdout. Prefer capture in the preceding 12 months and a group within seven days. A single stereo pair or three-view collection is insufficient for this split; adjacent tiles from the same acquisition do not add views.
3. Actual high-resolution sensor imagery, preferably approximately 0.3–0.5 m PAN with matched multispectral color, with delivered RPC coefficients in embedded tags or documented sidecars. State native resolution separately from resampled resolution. Provide a tiny metadata/sample package first.
4. Viewing angles/azimuths, overlap, clouds, shadows and any substantial scene changes. Incidence magnitude alone does not establish convergence or camera diversity.
5. Geometric/radiometric processing level, band definitions, bit depth, nodata and calibration information. Map mosaics or orthorectified display images alone do not satisfy the existing sensor-camera contract.
6. Independent lidar/DSM/control with acquisition dates, CRS, vertical datum and accuracy, or verifiably unchanged checkpoints with documented validation limits. A DSM reconstructed from these same images is not independent evidence.
7. Download sizes, access steps and whether an adapter is needed. The current crop path was verified on WorldView-3 NITF/RPC inputs; compatibility with another product must be tested on the sample.

Record the evidence, then reserve holdouts before calibration or appearance fitting. Do not create an approved training manifest until the input and validation gates pass.
