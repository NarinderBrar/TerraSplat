import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.ownership import cell_mask, in_frame, owned_mask, partition_check
from pipeline.seam_check import decide, height_step, seam_metrics

ROOT = Path(__file__).resolve().parents[2]


def grid():
    return {'enu_to_ecef': np.eye(4).tolist(), 'cells': {
        'west': {'east_m': [-250, 0], 'north_m': [-125, 125]},
        'east': {'east_m': [0, 250], 'north_m': [-125, 125]}}}


def test_half_open_partition_includes_the_boundary_exactly_once():
    g = grid()
    xyz = np.array([[-250,0,0], [-1e-9,0,0], [0,0,0], [1e-9,0,0], [249,0,0]])
    w, e = [cell_mask(xyz, g['cells'][n]) for n in ('west','east')]
    assert not (w & e).any()
    assert (w | e).all()
    assert w.tolist() == [True,True,False,False,False]
    assert partition_check(g)['passed']
    for delta in (-0.001, 0.001):
        bad = copy.deepcopy(g)
        bad['cells']['east']['east_m'][0] = delta
        assert not partition_check(bad)['passed']


def test_independent_anchors_and_height_tilt_use_one_shared_frame():
    g = grid()
    angle = 2e-5
    frame = np.eye(4)
    frame[:3,:3] = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    frame[:3,3] = [-125,0,-70]
    shared = np.array([[-0.001,0,300], [0.001,0,300]])
    local = in_frame(shared, g['enu_to_ecef'], frame)
    # A local x<=125 crop wrongly includes both points; the shared crop does not.
    assert (local[:,0] < 125).all()
    assert owned_mask(local, {'enu_to_ecef':frame}, g, 'west').tolist() == [True,False]
    assert in_frame(local, frame, g['enu_to_ecef']) == pytest.approx(shared, abs=1e-9)


def test_seam_metrics_distinguish_vertical_step_from_registered_shape_error():
    x = np.array([-20,-5,5,20.])
    shared = np.array([[e,n,0] for n in (1,0) for e in x])
    truth = {'dsm': np.zeros((2,4)), 'classes': np.array([[2,2,2,2],[6,6,6,6]])}
    raw = {'west': np.full((2,4),2.), 'east': np.full((2,4),3.)}
    registered = {'west': np.zeros((2,4)), 'east': np.zeros((2,4))}
    registered['east'][:,2] = 0.8
    result = seam_metrics(raw, registered, truth, shared, grid())
    assert result['height_continuity']['ground']['unregistered']['step_m'] == 1
    assert result['height_continuity']['all']['registered']['step_m'] == pytest.approx(0.8)
    assert result['lidar_error_by_distance']['east']['seam']['pixels'] == 2
    assert result['lidar_error_by_distance']['east']['interior']['mae_m'] == 0
    rules = json.loads((ROOT/'configs/phase3-split-decision.json').read_text())['rules']
    decision = decide(result, {'partition':partition_check(grid()),'doubled_centres':0}, rules, {'west':1.3,'east':1.3})
    assert not decision['passed']
    assert not decision['checks']['east_seam']
    assert not decision['checks']['height_ground']


def test_missing_height_pixels_are_inconclusive_not_zero_step():
    missing = np.full((2,2), np.nan)
    step = height_step(missing, missing, np.zeros((2,2)), np.ones((2,2),bool), np.ones((2,2),bool), np.ones((2,2),bool))
    assert step == {'west_pixels':0, 'east_pixels':0, 'step_m':None}


def test_profiles_and_acquisition_roles_are_unchanged():
    original = json.loads((ROOT/'configs/train-2015-pilot-19v-recipe.json').read_text())
    selection = json.loads((ROOT/'configs/jax068-pilot-19v.json').read_text())
    for name in ('west','east'):
        profile = json.loads((ROOT/f'configs/train-2015-{name}-19v-recipe.json').read_text())
        assert {k:v for k,v in profile.items() if k != 'id'} == {k:v for k,v in original.items() if k != 'id'}
        approved = json.loads((ROOT/f'configs/jax068-{name}-19v.json').read_text())
        assert approved['train_catalog_ids'] == selection['train_catalog_ids']
        assert approved['holdout_catalog_ids'] == selection['holdout_catalog_ids']


def test_tiler_crops_shared_cells_and_publisher_uses_shared_footprint(tmp_path):
    pytest.importorskip('plyfile')
    from plyfile import PlyData, PlyElement
    from pipeline.phase2 import digest
    from pipeline.tiling import plan
    from pipeline.publish import footprint
    from pipeline.georeference import placement

    g = json.loads((ROOT/'configs/jax068-grid.json').read_text())
    norm = dict(rotation=np.eye(3).tolist(), translation=[0,0,0], scale=1,z_shift=0)
    global_points = np.array([[-0.01,0,50],[0.01,0,50],[-200,0,50],[200,0,50]])
    rings = []
    for name in ('west','east'):
        cell = g['cells'][name]
        geo = placement(dict(longitude=cell['centre_longitude'], latitude=cell['centre_latitude'], ellipsoidal_height_m=-70), norm)
        geo['anchor'] = dict(longitude=cell['centre_longitude'],latitude=cell['centre_latitude'],ellipsoidal_height_m=-70)
        local = in_frame(global_points, g['enu_to_ecef'], geo['enu_to_ecef'])
        fields = list('xyz') + [f'f_dc_{i}' for i in range(3)] + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
        data = np.zeros(4,dtype=[(n,'<f4') for n in fields])
        for i,k in enumerate('xyz'):
            data[k] = local[:,i]
        data['rot_0'] = 1
        export = tmp_path/name
        export.mkdir()
        ply = export/'fused.ply'
        PlyData([PlyElement.describe(data,'vertex')],text=False).write(ply)
        (export/'transforms_train.json').write_text(json.dumps({'frames':[{'transform_matrix':np.eye(4).tolist()}]}))
        manifest = {'status':'exported','camera_scene':str(export),'fused_ply':{'path':str(ply),'sha256':digest(ply)},
                    'splat_count':4,'georeference':geo,'round_trip':{'gsd_m':0.3},'tileset':{'rules':[]}}
        (export/'export-manifest.json').write_text(json.dumps(manifest))
        profile = {'version':1,'block_half_extent_m':125,'leaf_max_splats':64,'extent_sigma':3,'max_opacity':0.99}
        result = plan(export,tmp_path/(name+'-tiles'),profile,g,name)
        assert result['crop']['owned_splats'] == 2
        polygon,_ = footprint(result)
        rings.append(polygon['coordinates'][0])
    # The exact shared-grid seam endpoints survive independent scene placement.
    assert rings[0][1] == pytest.approx(rings[1][0],abs=1e-10)
    assert rings[0][2] == pytest.approx(rings[1][3],abs=1e-10)


def test_export_depth_validation_and_seam_report_end_to_end(tmp_path):
    pytest.importorskip('plyfile')
    pytest.importorskip('pyproj')
    import rasterio
    from pyproj import Transformer
    from pipeline.dsm_validation import truth_enu, validate, read_truth
    from pipeline.georeference import placement
    from pipeline.phase2 import digest
    from pipeline.seam_check import check

    truth_dir = tmp_path/'truth'
    truth_dir.mkdir()
    size = 64
    rows, cols = np.indices((size,size))
    # Terrain with enough texture to identify an exact zero registration shift.
    terrain = (np.sin(rows/3) + np.cos(cols/4) + np.sin((cols+rows)/5)).astype('float32')
    terrain[20:30,20:30] += 10
    classes = np.full((size,size),2,dtype='uint8')
    for suffix, values in [('DSM',terrain),('CLS',classes)]:
        with rasterio.open(truth_dir/f'JAX_068_{suffix}.tif','w',driver='GTiff',width=size,height=size,count=1,dtype=values.dtype) as output:
            output.write(values,1)
    (truth_dir/'JAX_068_DSM.txt').write_text('436000\n3357000\n64\n0.5\n')
    lon,lat = Transformer.from_crs(32617,4326,always_xy=True).transform(436016,3357016)
    geo = placement(dict(longitude=lon,latitude=lat,ellipsoidal_height_m=0),
                    dict(rotation=np.eye(3).tolist(),translation=[0,0,0],scale=1,z_shift=0))
    g = grid()
    g['enu_to_ecef'] = geo['enu_to_ecef']
    grid_path = tmp_path/'grid.json'
    grid_path.write_text(json.dumps(g))
    truth = read_truth(truth_dir,'JAX_068')
    points = truth_enu(truth,geo['enu_to_ecef'])
    exports, depths, validations = {}, {}, {}
    for name in ('west','east'):
        exports[name] = tmp_path/name
        exports[name].mkdir()
        local = points[cell_mask(points,g['cells'][name])]
        ply = exports[name]/'fused.ply'
        data = np.zeros(len(local),dtype=[(n,'<f4') for n in 'xyz'])
        for i,n in enumerate('xyz'):
            data[n] = local[:,i]
        header = ('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(data))+'\n'
                  +'\n'.join(f'property float {n}' for n in 'xyz')+'\nend_header\n')
        ply.write_bytes(header.encode()+data.tobytes())
        manifest = {'status':'exported','fused_ply':{'path':str(ply),'sha256':digest(ply)},'georeference':geo}
        (exports[name]/'export-manifest.json').write_text(json.dumps(manifest))
        depths[name] = tmp_path/(name+'-depth.npz')
        # Use the full rendered halo to make registration independent of ownership cropping.
        np.savez(depths[name],enu=points)
        validations[name] = tmp_path/(name+'-validation')
        result = validate(exports[name],truth_dir,validations[name],depth_points=depths[name],grid=g,cell=name)
        assert result['registered']['all_labelled']['mae_m'] < 1e-5
    result = check(grid_path,exports,depths,validations,truth_dir,ROOT/'configs/phase3-split-decision.json')
    assert result['ownership']['doubled_centres'] == 0
    assert result['ownership']['counts']['west']['owned_within_1m'] > 0
    assert result['lidar_error_by_distance']['east']['interior']['pixels'] > 0
    assert result['decision']['checks']['east_seam']
    assert result['decision']['checks']['west_mae']
    raster = validations['west']/'splat-dsm.tif'
    with raster.open('ab') as output:
        output.write(b'changed')
    with pytest.raises(ValueError,match='raster changed'):
        check(grid_path,exports,depths,validations,truth_dir,ROOT/'configs/phase3-split-decision.json')


def test_publication_evidence_uses_actual_holdout_and_refuses_overwrite(tmp_path, monkeypatch):
    import runpy
    collect = runpy.run_path(str(ROOT/'scripts/phase3-split-evidence.py'))['collect']
    monkeypatch.chdir(tmp_path)
    def save(path, value):
        p = Path(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value))
    run = 'artifacts/stage1/west-2015-19v-recipe-01'
    tile = 'artifacts/tiles/west-2015-19v-recipe-01-r1'
    stats = {'mae_m':1.2, 'rmse_m':2.1, 'completeness':0.98}
    save('artifacts/validation/west-2015-19v-recipe-01-lidar/dsm-validation.json', {
        'registered':{k:stats for k in ('all_labelled','ground','building','vegetation')},
        'registration':{'splats_offset_from_lidar_m':{'horizontal':1.5}}})
    save(f'{run}/evaluation/evaluation-report.json',{'records':{'final':[
        {'catalog_id':'May2','role':'holdout','psnr_db':15.5},
        {'catalog_id':'train','role':'train','psnr_db':100}]}})
    save('artifacts/imagery/west-19v/selection.json',{'holdout':[{'catalog_id':'May2'}]})
    save('artifacts/validation/phase3-split-seam-check.json',{'decision':{'passed':True}})
    save(f'{tile}/tile-plan.json',{'crop':{'owned_splats':123}})
    save(f'{tile}/tiles-report.json',{'tiles':{'r':{},'r0':{}},'totals':{'glbBytes':5678}})
    Path('training').mkdir()
    collect('west')
    result = json.loads(Path('training/phase3-west-result.json').read_text())
    assert result['quality']['holdout_psnr_db'] == 15.5
    assert result['quality']['tiles'] == 2
    assert result['quality']['splats'] == 123
    assert len(result['inputs_sha256']) == 5
    with pytest.raises(FileExistsError):
        collect('west')


def test_general_partition_accepts_the_1km_grid_and_finds_gaps_and_overlaps():
    import json
    from pathlib import Path
    from pipeline.ownership import grid_partition_check
    g = json.loads((Path(__file__).resolve().parents[2] / 'configs/jax068-grid-1km.json').read_text())
    assert grid_partition_check(g) == {'cells': 16, 'lattice_gaps': 0, 'lattice_overlaps': 0, 'passed': True}
    gap = copy.deepcopy(g); gap['cells']['B2']['east_m'] = [-250, -10]
    assert grid_partition_check(gap)['lattice_gaps'] > 0
    overlap = copy.deepcopy(g); overlap['cells']['B2']['east_m'] = [-250, 10]
    assert grid_partition_check(overlap)['lattice_overlaps'] > 0
