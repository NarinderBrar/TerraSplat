"""Derive a class-coloured copy of an export for viewing semantics as an ordinary release. Run with .venv-imagery.

Splat geometry, opacity and order are unchanged. Each splat's colour becomes its class colour (SH DC only, higher
SH bands zeroed); unknown splats keep their own colour in greyscale so the scene stays readable.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from pipeline.phase2 import digest
from pipeline.semantics import CODES, COLOURS

SH_C0 = 0.28209479177387814


def recolour(vertices, classes):
    names = vertices.dtype.names
    if len(classes) != len(vertices):
        raise ValueError('Class count differs from splat count')
    dc = np.stack([vertices[f'f_dc_{i}'] for i in range(3)], 1).astype(np.float64)
    grey = np.repeat((0.5 + SH_C0 * dc).mean(1, keepdims=True), 3, 1)
    rgb = np.clip(grey, 0, 1)
    for code, colour in COLOURS.items():
        if code != CODES['unknown']:
            rgb[classes == code] = np.asarray(colour) / 255
    out = vertices.copy()
    for i in range(3):
        out[f'f_dc_{i}'] = ((rgb[:, i] - 0.5) / SH_C0).astype(out.dtype[f'f_dc_{i}'])
    for name in names:
        if name.startswith('f_rest_'):
            out[name] = 0
    return out


def derive(export_dir, classes_path, output):
    from plyfile import PlyData, PlyElement

    export_dir, output = Path(export_dir), Path(output)
    if output.exists():
        raise ValueError('Use a new export directory')
    manifest = json.loads((export_dir / 'export-manifest.json').read_text())
    source = Path(manifest['fused_ply']['path'])
    if digest(source) != manifest['fused_ply']['sha256']:
        raise ValueError('Fused PLY changed after export')
    data = PlyData.read(str(source))
    classes = np.load(classes_path)['classes']
    coloured = recolour(data['vertex'].data, classes)
    output.mkdir(parents=True)
    fused = output / (source.stem + '-classes.ply')
    PlyData([PlyElement.describe(coloured, 'vertex')], text=False).write(str(fused))
    manifest['fused_ply'] = {'path': str(fused.resolve()), 'sha256': digest(fused), 'bytes': fused.stat().st_size}
    manifest['semantic_colouring'] = {
        'source_export_manifest_sha256': digest(export_dir / 'export-manifest.json'),
        'source_fused_ply_sha256': digest(source), 'classes': str(classes_path), 'classes_sha256': digest(Path(classes_path)),
        'legend': {name: COLOURS[code] for name, code in CODES.items()},
        'note': 'Colour encodes the semantic class; unknown splats keep their own colour in greyscale. Geometry is unchanged.',
    }
    (output / 'export-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    counts = {name: int((classes == code).sum()) for name, code in CODES.items()}
    return {'fused_ply': str(fused), 'splats': len(classes), 'by_class': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True)
    parser.add_argument('--classes', required=True, help='splat-classes.npz from pipeline.semantics')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.export, args.classes, args.output), indent=2))


if __name__ == '__main__':
    main()
