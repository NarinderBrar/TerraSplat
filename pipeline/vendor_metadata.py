"""Turn vendor image metadata into a pipeline.imagery_check offer (CSV). Run with any environment with the stdlib.

Supported:
- Maxar (WorldView, GeoEye): `.IMD` with `.RPB`, either as files in a delivery folder or inside `.tar` archives.
- Airbus Pléiades Neo (and Pléiades 1A/1B): DIMAP v2 `DIM_*.XML` with `RPC_*.XML`.

    python -m pipeline.vendor_metadata DELIVERY... --aoi WEST SOUTH EAST NORTH --output offer.csv
    python -m pipeline.imagery_check offer.csv

The DIMAP reader follows the documented DIMAP v2 structure and has only been checked against a synthetic example;
verify it on the first real delivery (see docs/IMAGERY_SPEC.md).
"""

import argparse
import csv
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ElementTree

from pipeline.imagery_check import FIELDS

# Vendor product levels and types, normalised to the terms pipeline.imagery_check expects.
LEVELS = {'LV1B': 'L1B', 'LV1A': 'L1A', 'BASIC': 'BASIC', 'SENSOR': 'SENSOR', 'PRIMARY': 'PRIMARY',
          'LV2A': 'STANDARD', 'STANDARD': 'STANDARD', 'LV3D': 'ORTHO', 'ORTHO': 'ORTHO', 'LV3X': 'ORTHO'}


def point_in_polygon(lon, lat, polygon):
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def covers(footprint, aoi):
    """True when all four AOI corners lie inside the image footprint (lon, lat ring)."""
    if aoi is None or not footprint:
        return False
    west, south, east, north = aoi
    return all(point_in_polygon(x, y, footprint) for x, y in ((west, south), (east, south), (east, north), (west, north)))


# ---- Maxar ----------------------------------------------------------------------------------------------------

def _value(text, key):
    match = re.search(rf'^\s*{key} = "?([^";]+)"?;', text, re.M)
    return match.group(1).strip() if match else None


def read_maxar_imd(text, has_rpc):
    """One record per Maxar band product (PAN or MSI) from an .IMD text."""
    band = _value(text, 'bandId')
    level = (_value(text, 'productType') or '').upper()
    if level not in LEVELS:
        level = (_value(text, 'productLevel') or '').upper()
    footprint = [(float(_value(text, f'{c}Lon')), float(_value(text, f'{c}Lat'))) for c in ('UL', 'UR', 'LR', 'LL')]
    return {
        'id': _value(text, 'CatId'),
        'capture_time': _value(text, 'firstLineTime'),
        'sat_azimuth_deg': float(_value(text, 'meanSatAz')),
        'sat_elevation_deg': float(_value(text, 'meanSatEl')),
        'sun_elevation_deg': float(_value(text, 'meanSunEl')),
        'gsd_m': float(_value(text, 'meanProductGSD') or _value(text, 'meanCollectedGSD')),
        'cloud_cover': float(_value(text, 'cloudCover')),
        'product_level': LEVELS.get(level, level or 'UNKNOWN'),
        'rpc': has_rpc,
        'band': 'pan' if band == 'P' else 'msi',
        'footprint': footprint,
    }


def maxar_records(path):
    path = Path(path)
    if path.is_dir():
        # A delivery folder or an archive cache: loose .IMD files and .tar archives alike.
        records = [r for archive in sorted(path.rglob('*.tar')) for r in maxar_records(archive)]
        return records + [r for imd in sorted(path.rglob('*.IMD')) for r in maxar_records(imd)]
    texts = []
    if path.suffix == '.tar':
        with tarfile.open(path) as package:
            names = package.getnames()
            for member in package.getmembers():
                if member.name.endswith('.IMD'):
                    rpc = member.name[:-4] + '.RPB' in names
                    texts.append((package.extractfile(member).read().decode(), rpc))
    else:
        texts.append((path.read_text(), path.with_suffix('.RPB').exists()))
    return [read_maxar_imd(text, rpc) for text, rpc in texts]


# ---- Airbus DIMAP v2 ------------------------------------------------------------------------------------------

def _find(root, *names):
    """First element matching a path of local names (namespaces ignored)."""
    nodes = [root]
    for name in names:
        nodes = [child for node in nodes for child in node.iter() if child.tag.rsplit('}', 1)[-1] == name]
        if not nodes:
            return None
    return nodes[0]


def _text(root, *names):
    node = _find(root, *names)
    return node.text.strip() if node is not None and node.text else None


def read_dimap(path):
    """One record from a Pléiades / Pléiades Neo DIMAP v2 DIM_*.XML (PAN and MS bands of one acquisition)."""
    path = Path(path)
    root = ElementTree.parse(path).getroot()
    centre = next((node for node in root.iter() if node.tag.rsplit('}', 1)[-1] == 'Located_Geometric_Values'
                   and _text(node, 'LOCATION_TYPE') in ('Center', 'center', 'CENTER')), None)
    if centre is None:
        raise ValueError(f'{path}: no centre Located_Geometric_Values')
    incidence = float(_text(centre, 'Acquisition_Angles', 'INCIDENCE_ANGLE'))
    gsd = [float(v) for v in (_text(centre, 'GSD_ACROSS_TRACK'), _text(centre, 'GSD_ALONG_TRACK')) if v]
    bands = {(_text(node, 'BAND_ID') or '').upper() for node in root.iter() if node.tag.rsplit('}', 1)[-1] == 'Band_Spectral_Range'}
    footprint = [(float(_text(v, 'LON')), float(_text(v, 'LAT'))) for v in root.iter()
                 if v.tag.rsplit('}', 1)[-1] == 'Vertex' and _text(v, 'LON') is not None]
    date, time = _text(root, 'IMAGING_DATE'), _text(root, 'IMAGING_TIME')
    level = (_text(root, 'PROCESSING_LEVEL') or 'UNKNOWN').upper()
    rpc = any(path.parent.glob('RPC_*.XML')) or _find(root, 'Rational_Function_Model') is not None
    return {
        'id': _text(root, 'SOURCE_ID') or _text(root, 'DATASET_NAME') or path.stem,
        'capture_time': f'{date}T{time}' if date and time else date,
        # DIMAP gives the viewing azimuth and the incidence angle at the ground; elevation is its complement.
        'sat_azimuth_deg': float(_text(centre, 'Acquisition_Angles', 'AZIMUTH_ANGLE')),
        'sat_elevation_deg': 90 - incidence,
        'sun_elevation_deg': float(_text(centre, 'Solar_Incidences', 'SUN_ELEVATION')),
        'gsd_m': max(gsd) if gsd else float('nan'),
        'cloud_cover': float(_text(root, 'CLOUD_COVERAGE') or 'nan') / 100,
        'product_level': LEVELS.get(level, level),
        'rpc': rpc,
        'bands': ['pan'] * ('P' in bands) + ['msi'] * bool(bands - {'P', ''}),
        'footprint': footprint,
    }


# ---- Offer -----------------------------------------------------------------------------------------------------

def offer(paths, aoi):
    """Vendor-neutral rows for pipeline.imagery_check; Maxar PAN and MSI products of one catalog id are merged."""
    rows = {}
    for path in map(Path, paths):
        dims = sorted(path.rglob('DIM_*.XML')) if path.is_dir() else ([path] if path.name.startswith('DIM_') else [])
        for dim in dims:
            record = read_dimap(dim)
            rows[record['id']] = record
        if not dims:
            for record in maxar_records(path):
                row = rows.setdefault(record['id'], record | {'bands': [], 'footprints': []})
                row['bands'] = sorted(set(row['bands']) | {record['band']})
                row['rpc'] = row['rpc'] and record['rpc']
                row['footprints'].append(record['footprint'])
                if record['band'] == 'pan':  # the PAN product sets GSD and geometry
                    row.update({k: record[k] for k in ('gsd_m', 'sat_azimuth_deg', 'sat_elevation_deg')})
    result = []
    for row in rows.values():
        footprints = row.pop('footprints', None) or [row['footprint']]
        result.append({**{k: row[k] for k in FIELDS if k in row}, 'bands': '+'.join(row['bands']),
                       'covers_aoi': all(covers(f, aoi) for f in footprints)})
    return sorted(result, key=lambda r: r['capture_time'] or '')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='Delivery folders, DIM_*.XML, .IMD or Maxar .tar archives')
    parser.add_argument('--aoi', type=float, nargs=4, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = offer(args.paths, args.aoi)
    with open(args.output, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} images written to {args.output}')


if __name__ == '__main__':
    main()
