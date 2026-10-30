"""Per-tile segment layers: one class byte per tile splat, for the viewer's segment mode. Run with .venv-imagery.

    python -m pipeline.segments --tiles artifacts/tiles/<plan> --export <export dir> \
        --classes artifacts/semantics/<run>/splat-classes.npz --label "semantics v2 (experimental)" --output artifacts/segments/<name>

Writes <tile id>.cls (uint8 per splat, in the tile's payload order, which SPZ and Cesium keep) and segments.json.
Leaf tiles hold the original splats, so each takes its own class (matched by position). Coarse tiles hold merged
splats; each takes the most common known class among the nearest original splats.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.export import apply_transform
from pipeline.phase2 import digest
from pipeline.semantics import CODES, COLOURS
from pipeline.tiling import ENU_TO_GLTF, read_fused_ply

RECORD = 14
COARSE_NEIGHBOURS = 16
LEAF_MATCH_M = 1e-3


def majority(labels):
    """Most common known class per row (0 = unknown only when every neighbour is unknown)."""
    counts = np.stack([(labels == code).sum(axis=1) for code in range(max(CODES.values()) + 1)], axis=1)
    counts[:, CODES['unknown']] = 0
    best = counts.argmax(axis=1)
    return np.where(counts.max(axis=1) > 0, best, CODES['unknown']).astype(np.uint8)


def tile_classes(positions_enu, leaf, tree, classes):
    if leaf:
        distance, index = tree.query(positions_enu)
        return classes[index], float(distance.max()) if len(distance) else 0.0
    _, index = tree.query(positions_enu, k=min(COARSE_NEIGHBOURS, len(classes)))
    return majority(classes[index.reshape(len(positions_enu), -1)]), None


def build(tiles_dir, export_dir, classes_path, label, output):
    from scipy.spatial import cKDTree

    tiles_dir, output = Path(tiles_dir), Path(output)
    if output.exists():
        raise ValueError('Use a new segments directory')
    plan = json.loads((tiles_dir / 'tile-plan.json').read_text())
    manifest_path = Path(export_dir) / 'export-manifest.json'
    if digest(manifest_path) != plan['export_manifest_sha256']:
        raise ValueError('The tile plan was made from a different export')
    export = json.loads(manifest_path.read_text())
    data = np.load(classes_path)
    classes = data['classes'].astype(np.uint8)
    if json.loads(str(data['legend'])) != CODES:
        raise ValueError('Splat classes use different codes')
    fused = read_fused_ply(Path(export['fused_ply']['path']))['centres']
    if len(fused) != len(classes):
        raise ValueError('Class count differs from the fused PLY')
    tree = cKDTree(apply_transform(np.asarray(export['georeference']['model_to_enu']), fused))
    parents = {t['parent'] for t in plan['tiles']}
    output.mkdir(parents=True)
    tiles, counts, worst_leaf = {}, np.zeros(max(CODES.values()) + 1, np.int64), 0.0
    for tile in plan['tiles']:
        records = np.fromfile(tiles_dir / tile['payload'], '<f4').reshape(-1, RECORD)
        positions = records[:, :3].astype(np.float64) @ ENU_TO_GLTF  # glTF (east, up, -north) back to ENU
        leaf = tile['id'] not in parents
        values, distance = tile_classes(positions, leaf, tree, classes)
        if leaf:
            if distance > LEAF_MATCH_M:
                raise ValueError(f"Leaf tile {tile['id']} splats do not match the fused PLY ({distance} m)")
            worst_leaf = max(worst_leaf, distance)
            counts += np.bincount(values, minlength=len(counts))
        path = output / f"{tile['id']}.cls"
        path.write_bytes(values.tobytes())
        tiles[tile['id']] = {'splats': len(values), 'leaf': leaf, 'sha256': digest(path)}
    names = {code: name for name, code in CODES.items()}
    document = {
        'schema_version': 1, 'label': label,
        'format': 'tiles/<id>.glb -> segments/<id>.cls: uint8 class per splat in tile payload order',
        'classes': [{'code': code, 'name': names[code], 'colour': list(COLOURS[code])} for code in sorted(names)],
        'source': {'splat_classes': str(classes_path), 'splat_classes_sha256': digest(Path(classes_path)),
                   'tile_plan_sha256': digest(tiles_dir / 'tile-plan.json')},
        'method': {'leaf': f'own class, matched by position (max distance {worst_leaf:.2e} m)',
                   'coarse': f'most common known class of the {COARSE_NEIGHBOURS} nearest original splats'},
        'leaf_splats_by_class': {names[c]: int(n) for c, n in enumerate(counts) if c in names},
        'tiles': tiles,
    }
    (output / 'segments.json').write_text(json.dumps(document, indent=2) + '\n')
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tiles', required=True)
    parser.add_argument('--export', required=True, help='Export directory the tiles were planned from')
    parser.add_argument('--classes', required=True, help='splat-classes.npz from pipeline.semantics(_clip)')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    document = build(args.tiles, args.export, args.classes, args.label, args.output)
    print(f"{len(document['tiles'])} tile layers written; leaf splats by class {document['leaf_splats_by_class']}")


if __name__ == '__main__':
    main()
