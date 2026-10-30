import json
from pathlib import Path

import pytest

from pipeline.vendor_metadata import covers, maxar_records, offer, read_dimap

ROOT = Path(__file__).resolve().parents[2]

DIMAP = """<?xml version="1.0" encoding="UTF-8"?>
<Dimap_Document xmlns="http://www.example.com/dimap">
  <Dataset_Identification><DATASET_NAME>DS_PNEO3_202405151612345_FR1_PX_W082N30</DATASET_NAME></Dataset_Identification>
  <Dataset_Content><CLOUD_COVERAGE>4.5</CLOUD_COVERAGE>
    <Dataset_Extent>
      <Vertex><LON>-81.70</LON><LAT>30.38</LAT></Vertex><Vertex><LON>-81.62</LON><LAT>30.38</LAT></Vertex>
      <Vertex><LON>-81.62</LON><LAT>30.31</LAT></Vertex><Vertex><LON>-81.70</LON><LAT>30.31</LAT></Vertex>
    </Dataset_Extent></Dataset_Content>
  <Processing_Information><Product_Settings><PROCESSING_LEVEL>SENSOR</PROCESSING_LEVEL></Product_Settings></Processing_Information>
  <Dataset_Sources><Source_Identification><SOURCE_ID>PNEO3_202405151612345</SOURCE_ID>
    <Strip_Source><IMAGING_DATE>2024-05-15</IMAGING_DATE><IMAGING_TIME>16:12:34.5Z</IMAGING_TIME></Strip_Source>
  </Source_Identification></Dataset_Sources>
  <Radiometric_Data><Radiometric_Settings>
    <Band_Spectral_Range><BAND_ID>P</BAND_ID></Band_Spectral_Range>
    <Band_Spectral_Range><BAND_ID>R</BAND_ID></Band_Spectral_Range>
    <Band_Spectral_Range><BAND_ID>NIR</BAND_ID></Band_Spectral_Range>
  </Radiometric_Settings></Radiometric_Data>
  <Geometric_Data><Use_Area>
    <Located_Geometric_Values><LOCATION_TYPE>Top Center</LOCATION_TYPE>
      <Acquisition_Angles><AZIMUTH_ANGLE>1</AZIMUTH_ANGLE><INCIDENCE_ANGLE>1</INCIDENCE_ANGLE></Acquisition_Angles></Located_Geometric_Values>
    <Located_Geometric_Values><LOCATION_TYPE>Center</LOCATION_TYPE>
      <Acquisition_Angles><AZIMUTH_ANGLE>123.4</AZIMUTH_ANGLE><INCIDENCE_ANGLE>14.2</INCIDENCE_ANGLE></Acquisition_Angles>
      <Solar_Incidences><SUN_AZIMUTH>110.0</SUN_AZIMUTH><SUN_ELEVATION>65.3</SUN_ELEVATION></Solar_Incidences>
      <Ground_Sample_Distance><GSD_ACROSS_TRACK>0.31</GSD_ACROSS_TRACK><GSD_ALONG_TRACK>0.33</GSD_ALONG_TRACK></Ground_Sample_Distance>
    </Located_Geometric_Values></Use_Area></Geometric_Data>
</Dimap_Document>
"""


def test_dimap_reader_on_a_synthetic_pleiades_neo_product(tmp_path):
    (tmp_path / 'DIM_PNEO3_1.XML').write_text(DIMAP)
    (tmp_path / 'RPC_PNEO3_1.XML').write_text('<rpc/>')
    record = read_dimap(tmp_path / 'DIM_PNEO3_1.XML')
    assert record['id'] == 'PNEO3_202405151612345'
    assert record['capture_time'] == '2024-05-15T16:12:34.5Z'
    assert record['sat_azimuth_deg'] == 123.4 and record['sat_elevation_deg'] == pytest.approx(75.8)
    assert record['sun_elevation_deg'] == 65.3 and record['gsd_m'] == 0.33 and record['cloud_cover'] == pytest.approx(0.045)
    assert record['product_level'] == 'SENSOR' and record['rpc'] and record['bands'] == ['pan', 'msi']
    rows = offer([tmp_path], (-81.669, 30.341, -81.658, 30.351))
    assert rows[0]['covers_aoi'] is True and rows[0]['bands'] == 'pan+msi'
    assert offer([tmp_path], (-81.75, 30.341, -81.658, 30.351))[0]['covers_aoi'] is False


def test_covers_requires_every_aoi_corner():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert covers(square, (1, 1, 9, 9)) and not covers(square, (1, 1, 11, 9)) and not covers([], (1, 1, 2, 2))


def test_maxar_reader_matches_the_recorded_19_view_metadata():
    cache = ROOT / '.cache/core3d-metadata'
    selection = ROOT / 'artifacts/imagery/pilot-19v/selection.json'
    if not cache.is_dir() or not selection.is_file():
        pytest.skip('Maxar metadata cache not available')
    expected = {v['catalog_id']: v for v in (lambda s: s['train'] + s['holdout'])(json.loads(selection.read_text()))}
    checked = 0
    for archive in sorted(cache.glob('*.tar')):
        for record in maxar_records(archive):
            if record['id'] in expected and record['band'] == 'pan':
                e = expected[record['id']]
                assert record['sat_azimuth_deg'] == pytest.approx(e['sat_azimuth_deg'])
                assert record['sat_elevation_deg'] == pytest.approx(e['sat_elevation_deg'])
                assert record['sun_elevation_deg'] == pytest.approx(e['sun_elevation_deg'])
                assert record['gsd_m'] == pytest.approx(e['gsd_m']) and record['cloud_cover'] == pytest.approx(e['cloud_cover'])
                assert record['product_level'] == 'BASIC' and record['rpc']
                checked += 1
    assert checked >= 15
    # A folder of archives reads the same records as the archives one by one.
    assert len(offer([cache], (-81.6693, 30.3418, -81.6582, 30.3511))) == len({r['id'] for a in cache.glob('*.tar') for r in maxar_records(a)})
