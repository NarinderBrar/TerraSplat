"""Pre-purchase check of a commercial image set against docs/IMAGERY_SPEC.md.

Input is a vendor-neutral JSON or CSV list of offered images (catalogue metadata only, no pixels).
Tiers are anchored to the three image sets we measured against lidar; anything beyond them is unmeasured.
"""

import argparse
import csv
from datetime import datetime
import itertools
import json
import math
from pathlib import Path

# Per-image limits; the first three match pipeline.acquisitions.usable_records and select().
MAX_OFF_NADIR_DEG = 40
MIN_SUN_ELEVATION_DEG = 15
MAX_CLOUD_COVER = 0.1
MAX_GSD_M = 0.5
RAW_LEVELS = {'L1B', 'L1A', 'BASIC', 'PRIMARY', 'SENSOR'}
# Measured training sets (registered lidar MAE, JAX_068), from artifacts/imagery/*/selection.json:
# views, largest azimuth gap (also folded onto 0-180°) and widest pairwise convergence, rounded so each
# set meets its own tier.
MEASURED = [
    {'name': '19-view recipe', 'views': 19, 'max_azimuth_gap_deg': 96.6, 'max_folded_azimuth_gap_deg': 39.6,
     'max_convergence_deg': 56.4, 'mae_m': 1.36},
    {'name': '5-view', 'views': 5, 'max_azimuth_gap_deg': 185.2, 'max_folded_azimuth_gap_deg': 76.4,
     'max_convergence_deg': 49.6, 'mae_m': 2.82},
    {'name': '3-view', 'views': 3, 'max_azimuth_gap_deg': 185.2, 'max_folded_azimuth_gap_deg': 102.4,
     'max_convergence_deg': 49.6, 'mae_m': 3.32},
]
ONE_SIDE_GAP_DEG = 270
# Azimuths folded onto 0-180°: a single along-track pass (forward, nadir, backward) collapses to one direction.
ONE_PLANE_FOLDED_GAP_DEG = 150
MAX_SPAN_DAYS = 7
FIELDS = ['id', 'capture_time', 'sat_azimuth_deg', 'sat_elevation_deg', 'sun_elevation_deg', 'gsd_m',
          'cloud_cover', 'product_level', 'rpc', 'bands', 'covers_aoi']


def read_images(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
    else:
        value = json.loads(path.read_text())
        rows = value['images'] if isinstance(value, dict) else value
    images = []
    for row in rows:
        missing = [f for f in FIELDS if row.get(f) in (None, '')]
        if missing:
            raise ValueError(f"Image {row.get('id', '?')} is missing {', '.join(missing)}")
        flag = (lambda v: v if isinstance(v, bool) else str(v).strip().lower() in ('1', 'true', 'yes'))
        bands = row['bands'] if isinstance(row['bands'], list) else str(row['bands']).replace('+', ' ').split()
        images.append({
            'id': str(row['id']),
            'capture_time': datetime.fromisoformat(str(row['capture_time']).replace('Z', '+00:00')),
            'sat_azimuth_deg': float(row['sat_azimuth_deg']) % 360,
            'sat_elevation_deg': float(row['sat_elevation_deg']),
            'sun_elevation_deg': float(row['sun_elevation_deg']),
            'gsd_m': float(row['gsd_m']),
            'cloud_cover': float(row['cloud_cover']),
            'product_level': str(row['product_level']).strip().upper(),
            'rpc': flag(row['rpc']),
            'bands': sorted(b.strip().lower() for b in bands),
            'covers_aoi': flag(row['covers_aoi']),
        })
    if len({i['id'] for i in images}) != len(images):
        raise ValueError('Image ids must be unique')
    return images


def image_problems(image):
    off_nadir = 90 - image['sat_elevation_deg']
    checks = [
        (image['rpc'], 'no RPC camera model'),
        (image['product_level'] in RAW_LEVELS, f"product level {image['product_level']} is not raw sensor geometry"),
        ({'pan', 'msi'} <= set(image['bands']), 'needs both PAN and MSI bands'),
        (image['covers_aoi'], 'does not cover the AOI plus halo'),
        (image['gsd_m'] <= MAX_GSD_M, f"GSD {image['gsd_m']:.2f} m > {MAX_GSD_M} m"),
        (off_nadir <= MAX_OFF_NADIR_DEG, f'off-nadir {off_nadir:.1f}° > {MAX_OFF_NADIR_DEG}°'),
        (image['sun_elevation_deg'] >= MIN_SUN_ELEVATION_DEG,
         f"sun elevation {image['sun_elevation_deg']:.1f}° < {MIN_SUN_ELEVATION_DEG}°"),
        (image['cloud_cover'] <= MAX_CLOUD_COVER, f"cloud cover {image['cloud_cover']:.0%} > {MAX_CLOUD_COVER:.0%}"),
    ]
    return [message for ok, message in checks if not ok]


def direction(image):
    az, el = math.radians(image['sat_azimuth_deg']), math.radians(image['sat_elevation_deg'])
    return (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))


def convergence(a, b):
    dot = sum(x * y for x, y in zip(direction(a), direction(b)))
    return math.degrees(math.acos(max(-1, min(1, dot))))


def geometry(images):
    def largest_gap(values, circle):
        values = sorted({round(v % circle, 6) for v in values})
        if len(values) == 1:
            return float(circle)
        return max((values[(k + 1) % len(values)] - values[k]) % circle for k in range(len(values)))

    azimuths = [i['sat_azimuth_deg'] for i in images]
    angles = [convergence(a, b) for a, b in itertools.combinations(images, 2)]
    times = [i['capture_time'] for i in images]
    return {
        'views': len(images),
        'max_azimuth_gap_deg': largest_gap(azimuths, 360) if len(images) > 1 else 360.0,
        'max_folded_azimuth_gap_deg': largest_gap([a % 180 for a in azimuths], 180) if len(images) > 1 else 180.0,
        'max_convergence_deg': max(angles, default=0.0),
        'median_convergence_deg': sorted(angles)[len(angles) // 2] if angles else 0.0,
        'span_days': (max(times) - min(times)).total_seconds() / 86400,
        'distinct_days': len({t.date() for t in times}),
    }


def tier(training):
    """Most-measured set whose views and angle spread the training views meet or exceed."""
    for reference in MEASURED:
        if (training['views'] >= reference['views']
                and training['max_azimuth_gap_deg'] <= reference['max_azimuth_gap_deg']
                and training['max_folded_azimuth_gap_deg'] <= reference['max_folded_azimuth_gap_deg']
                and training['max_convergence_deg'] >= reference['max_convergence_deg']):
            return reference
    return None


def check(images, holdout=None):
    problems = {i['id']: image_problems(i) for i in images}
    usable = [i for i in images if not problems[i['id']]]
    if holdout is not None and holdout not in {i['id'] for i in usable}:
        raise ValueError(f'Holdout {holdout} is not a usable image')
    # One whole acquisition is held out for validation once there are enough views to spare one.
    if holdout is None and len(usable) >= 4:
        holdout = min(usable, key=lambda i: abs(90 - i['sat_elevation_deg']))['id']
    training = [i for i in usable if i['id'] != holdout]
    stats = geometry(training) if training else None
    reference = tier(stats) if stats else None
    warnings = []
    if stats and stats['views'] >= 2:
        if stats['max_folded_azimuth_gap_deg'] > ONE_PLANE_FOLDED_GAP_DEG:
            warnings.append('All training views lie in one viewing plane, typical of a single along-track pass; '
                            'building sides facing across the track are barely seen.')
        elif stats['max_azimuth_gap_deg'] > ONE_SIDE_GAP_DEG:
            warnings.append('All training views look from one side (largest azimuth gap '
                            f"{stats['max_azimuth_gap_deg']:.0f}°); building sides facing away are barely seen.")
        if stats['span_days'] > MAX_SPAN_DAYS:
            warnings.append(f"Views span {stats['span_days']:.0f} days; screen for construction and seasonal "
                            'change before training (spec prefers ≤7 days).')
    if stats and stats['views'] >= 2 and holdout is None:
        warnings.append('No holdout: buy at least 4 usable images so one can be kept out for validation.')
    if stats is None or stats['views'] < 2:
        verdict = 'reject'
        summary = 'Fewer than two usable images: heights cannot be reconstructed with this pipeline.'
    elif reference is None:
        verdict = 'unmeasured'
        summary = ('Usable, but fewer views or narrower angles than any set we measured '
                   '(2-view stereo is unmeasured; expect worse than 3.3 m).')
    else:
        verdict = 'expected'
        summary = (f"Meets or exceeds our measured {reference['name']} set "
                   f"(registered lidar MAE {reference['mae_m']:.2f} m in Jacksonville JAX_068).")
    return {
        'verdict': verdict, 'summary': summary,
        'measured_reference': reference, 'training_geometry': stats, 'holdout': holdout,
        'usable': [i['id'] for i in usable], 'rejected': {k: v for k, v in problems.items() if v},
        'warnings': warnings,
        'limits': {'max_off_nadir_deg': MAX_OFF_NADIR_DEG, 'min_sun_elevation_deg': MIN_SUN_ELEVATION_DEG,
                   'max_cloud_cover': MAX_CLOUD_COVER, 'max_gsd_m': MAX_GSD_M, 'raw_levels': sorted(RAW_LEVELS)},
        'note': ('Expected accuracy comes from one 250 m block in one city; it is not a guarantee. '
                 'Sensors other than WorldView-3 also need a metadata adapter before the pipeline can read them.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', help='JSON (list or {"images": [...]}) or CSV with the spec fields')
    parser.add_argument('--holdout', help='Image id to keep out for validation (default: the most nadir one)')
    parser.add_argument('--output', help='Write the full report as JSON')
    args = parser.parse_args()
    report = check(read_images(args.images), args.holdout)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(f"{report['verdict'].upper()}: {report['summary']}")
    if report['training_geometry']:
        g = report['training_geometry']
        print(f"Training views {g['views']} (holdout {report['holdout'] or 'none'}); largest azimuth gap "
              f"{g['max_azimuth_gap_deg']:.0f}°; widest convergence {g['max_convergence_deg']:.0f}°; "
              f"span {g['span_days']:.1f} days on {g['distinct_days']} day(s)")
    for image, problems in report['rejected'].items():
        print(f'Rejected {image}: ' + '; '.join(problems))
    for warning in report['warnings']:
        print('Warning: ' + warning)


if __name__ == '__main__':
    main()
