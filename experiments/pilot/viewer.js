import '../phase0/style.css';
const C = window.Cesium;
const status = document.querySelector('#status');
const base = '/tiles/pilot-2015-04/';
try {
  const plan = await fetch(`${base}tile-plan.json`).then(r => {
    if (!r.ok) throw new Error(`Tile plan HTTP ${r.status}`);
    return r.json();
  });
  const viewer = new C.Viewer('viewer', {
    baseLayer: false, baseLayerPicker: false, geocoder: false, homeButton: false,
    sceneModePicker: false, navigationHelpButton: false, animation: false, timeline: false,
    fullscreenButton: false, infoBox: false, selectionIndicator: false,
    skyBox: false, skyAtmosphere: false,
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  const { scene } = viewer;
  scene.globe.show = false;
  scene.backgroundColor = C.Color.fromCssColorString('#0b0f14');
  scene.highDynamicRange = false;
  viewer.resolutionScale = 1;
  const anchor = plan.enu_anchor;
  const frame = C.Transforms.eastNorthUpToFixedFrame(
    C.Cartesian3.fromDegrees(anchor.longitude, anchor.latitude, anchor.ellipsoidal_height_m));
  const tileset = await C.Cesium3DTileset.fromUrl(`${base}tileset.json`);
  scene.primitives.add(tileset);
  // Median ground height of the owned splats, metres above the ENU anchor.
  const ground = 53;
  const views = {
    oblique: () => viewer.camera.lookAtTransform(frame, new C.Cartesian3(0, -320, 260)),
    top: () => viewer.camera.lookAtTransform(frame, new C.Cartesian3(0, -1, 420)),
  };
  views.oblique();
  document.querySelector('#reset').addEventListener('click', views.oblique);
  document.querySelector('#top').addEventListener('click', views.top);
  document.querySelector('#bounds').addEventListener('change', event => { tileset.debugShowBoundingVolume = event.target.checked; });
  let osm;
  document.querySelector('#osm').addEventListener('change', event => {
    scene.globe.show = event.target.checked;
    scene.globe.translucency.enabled = true;
    scene.globe.translucency.frontFaceAlpha = 0.55;
    if (event.target.checked && !osm) {
      osm = scene.imageryLayers.addImageryProvider(new C.OpenStreetMapImageryProvider({ url: 'https://tile.openstreetmap.org/' }));
    }
  });
  function fail(error) { status.textContent = `Failed: ${error.message ?? error}`; }
  tileset.tileFailed.addEventListener(fail);
  scene.renderError.addEventListener((_scene, error) => fail(error));
  const visible = new Set();
  tileset.tileVisible.addEventListener(tile => visible.add(tile.content.url.split('/').pop().split('?')[0]));
  window.pilot = {
    viewer, tileset, plan, frame, ground, views,
    project(enu) {
      const world = C.Matrix4.multiplyByPoint(frame, new C.Cartesian3(...enu), new C.Cartesian3());
      const p = C.SceneTransforms.worldToWindowCoordinates(scene, world);
      const rect = viewer.canvas.getBoundingClientRect();
      return p && { x: p.x + rect.left, y: p.y + rect.top };
    },
    // Tiles drawn in the most recent frame, by glb name.
    visibleTiles() { visible.clear(); viewer.render(); return [...visible].sort(); },
  };
  const splats = plan.tiles.filter(t => !plan.tiles.some(c => c.parent === t.id)).reduce((n, t) => n + t.splats, 0);
  document.querySelector('#details').textContent =
    `${plan.tiles.length} tiles · ${plan.levels} levels · ${splats.toLocaleString()} full-resolution splats`;
  tileset.allTilesLoaded.addEventListener(() => {
    status.textContent = `Pilot tiles loaded · CesiumJS ${C.VERSION}`;
  });
} catch (error) {
  status.textContent = `Failed: ${error.message}`;
  console.error(error);
}
