#!/usr/bin/env bash
# Semantics v5 class layers for the Skyfall authors' scenes (central 700 m), published to the review catalogue.
# Usage: scripts/authors-segments.sh NYC_004 NYC_010 ...   (skips depth and predictions that already exist)
cd ~/Documents/OrbitWorks/TerraSplat
for s in "$@"; do
  id=skyfall-$(echo $s | tr 'A-Z_' 'a-z-'); d=artifacts/skyfall/$s; O=artifacts/semantics/v5-authors/$s
  base=$(ls configs/releases/$id-final-r*.json | sort | tail -1); n=${base##*-r}; n=${n%.json}; suffix=r$((n + 1))
  case $s in
    JAX_068) note="DFC2019 lidar on the central 290 m: accuracy 0.72, building IoU 0.69, tree IoU 0.37 (v5)";;
    JAX_*) note="no accuracy score for this scene (its DFC2019 labels are kept for a blind test)";;
    *) note="no reference data for this scene, accuracy unmeasured";;
  esac
  label="Semantics v5 (experimental): OpenEarthMap SegFormer-B2 land cover + height, central 700 m; $note"
  echo "== $s ($id, $suffix) $(date +%T)"
  mkdir -p $O
  [ -d $d/final-depth-350 ] || (source scripts/stage1-env.sh >/dev/null && env -u PYTHONPATH .venv-stage1/bin/python -m pipeline.render_depth \
      --run artifacts/stage1/ref-jax068-recipe-01 --export $d/final-placed --half-extent 350 --output $d/final-depth-350) > $O/depth.log 2>&1 || { echo "$s depth FAILED"; continue; }
  [ -d $O/probs ] || env -u PYTHONPATH .venv-landcover/bin/python -m pipeline.landcover_predict \
      --model artifacts/landcover/kaggle-v2/v5-oem-segformer-b2 --images $d/scene/images --output $O/probs > $O/predict.log 2>&1 || { echo "$s predict FAILED"; continue; }
  env -u PYTHONPATH .venv-imagery/bin/python -m pipeline.semantics_grid scene --scene $d/scene --export $d/final-placed \
      --points $d/final-depth-350/depth-points.npz --probabilities $O/probs --output $O/classes-350 > $O/classes.log 2>&1 || { echo "$s classes FAILED"; tail -3 $O/classes.log; continue; }
  env -u PYTHONPATH .venv-imagery/bin/python -m pipeline.segments --tiles artifacts/tiles/$id-final-r1 --export $d/final-placed \
      --classes $O/classes-350/splat-classes.npz --label "$label" --output artifacts/segments/$id-v5-700m 2>&1 | tail -1 || { echo "$s segments FAILED"; continue; }
  env -u PYTHONPATH .venv-imagery/bin/python -m pipeline.semantics_grid release --job $base --segments artifacts/segments/$id-v5-700m \
      --suffix $suffix --output $O/release-$suffix.json > /dev/null || { echo "$s release FAILED"; continue; }
  env -u PYTHONPATH .venv-sfm/bin/python -m pipeline.publish --config $O/release-$suffix.json --releases artifacts/releases-grid-review --activate 2>&1 | tail -1 | cut -c1-100
  python3 -c "import json;d=json.load(open('$O/classes-350/evaluation.json'));print('  cells classified %.2f' % (d['cells']['owned_with_evidence']/d['grid']['owned_cells']))"
done
echo "== done $(date +%T)"
