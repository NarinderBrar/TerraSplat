# Segmentation model and downloads

[Project home](../README.md) · [All documentation](README.md)

The trained checkpoint, image-to-splat class workflow, and evaluation status. Run commands from the repository root.

## How segmentation works

Segment mode labels splats as ground, road, building, trees, grass or water using a sidecar class layer (one byte per
splat), so geometry tiles stay byte-identical. With no classes selected, the scene keeps its original colours;
selected classes receive a holographic blend. The v5 model, projection/fusion pipeline, tile layers, and viewer are
implemented, but the classifier has not met all accuracy targets ([docs/SEMANTICS.md](SEMANTICS.md)):

| Version | Method | Outcome |
| --- | --- | --- |
| v1 | NDVI (WorldView-3 multispectral), OpenStreetMap and height | failed 2 of 3 (accuracy 0.52, building IoU 0.70, tree IoU 0.19) |
| v2 | v1 with corrected class rules (diagnostic) | accuracy 0.68, building 0.70, tree 0.29; needs multispectral imagery, so not usable for RGB-only scenes |
| v3 | LangSplat SAM + CLIP features | failed (accuracy 0.25) |
| v4 | SAM + RemoteCLIP | failed (accuracy 0.47, building 0.31, tree 0.22) |
| **v5** | SegFormer-B2 trained on [OpenEarthMap](https://open-earth-map.org/) plus height | trained; JAX_068 accuracy 0.724, building IoU 0.690, tree IoU 0.366; building target passes, overall and tree targets fail |

The v5 model was trained using the Kaggle workflow, then applied to the training views of each scene. Pixel
probabilities are projected through the cameras with visibility checks, fused by median, combined with height rules,
and assigned to splats. The authors' scene layers cover the central 700 m square. Disk-backed, chunked fusion keeps
the 60-view NYC scenes from requiring the full probability stack in RAM; saved predictions can be fused on CPU.

All 16 grid blocks and 12 authors' scenes have v5 layers in `artifacts/releases-grid-review`. JAX_068 is the tuning
scene. A blind accuracy evaluation on the other seven DFC2019 Jacksonville scenes remains outstanding; generating
their layers is not an independent accuracy evaluation.

### Download the trained segmentation model

The training script and saved outputs are public on
[Kaggle: TerraSplat landcover v5](https://www.kaggle.com/code/npsbrar1986/terrasplat-landcover-v5).
Open the **Output** tab and select version **2**. The inference files are
`v5-oem-segformer-b2/best.safetensors` (about 105 MiB) and `v5-oem-segformer-b2/config.json`;
`report.json` records training and validation results. The best checkpoint was selected at step 36,000:
OpenEarthMap validation mIoU **0.680**, pixel accuracy **0.811**. These are image-space segmentation metrics, separate
from the projected 3D scene scores above.

With the Kaggle CLI installed and configured, download the version used by TerraSplat:

```bash
kaggle kernels output npsbrar1986/terrasplat-landcover-v5/2 \
  --file-pattern '^v5-oem-segformer-b2/(best\.safetensors|config\.json|report\.json)$' \
  -p artifacts/landcover/kaggle-v2
```

Checkpoint SHA-256: `c4967d6c7a71a42679c04ad164cb36b150450b30d1b9277cffb3cfc97fc0b15a`.
Use the downloaded directory with `pipeline.landcover_predict --model artifacts/landcover/kaggle-v2/v5-oem-segformer-b2`.
Weights stay outside Git. The model is documented for
non-commercial research/evaluation under its training-data terms; the repository's MIT code licence does not
relicense the weights or datasets.
