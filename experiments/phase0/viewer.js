import './style.css';
const C = window.Cesium;
const status = document.querySelector('#status');
try {
  const fixture = await fetch('/phase0/fixture.json').then(r => {
    if (!r.ok) throw new Error(`Fixture HTTP ${r.status}`);
    return r.json();
  });
  const viewer = new C.Viewer('viewer', {
    baseLayer: false, baseLayerPicker: false, geocoder: false, homeButton: false,
    sceneModePicker: false, navigationHelpButton: false, animation: false, timeline: false,
    fullscreenButton: false, infoBox: false, selectionIndicator: false,
    skyBox: false, skyAtmosphere: false,
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  viewer.scene.globe.show = false;
  viewer.scene.backgroundColor = C.Color.fromCssColorString('#0b0f14');
  viewer.scene.highDynamicRange = false;
  viewer.resolutionScale = 1;
  const anchor = fixture.anchor;
  const frame = C.Transforms.eastNorthUpToFixedFrame(C.Cartesian3.fromDegrees(anchor.longitude, anchor.latitude, anchor.ellipsoidalHeight));
  const tileset = await C.Cesium3DTileset.fromUrl('/phase0/tileset.json');
  viewer.scene.primitives.add(tileset);
  function resetCamera() {
    viewer.camera.lookAtTransform(frame, new C.Cartesian3(27, -40, 30));
  }
  resetCamera();
  document.querySelector('#reset').addEventListener('click', resetCamera);
  document.querySelector('#globe').addEventListener('change', event => { viewer.scene.globe.show = event.target.checked; });
  document.querySelector('#bounds').addEventListener('change', event => { tileset.debugShowBoundingVolume = event.target.checked; });
  function fail(error) { status.textContent = `Failed: ${error.message ?? error}`; }
  tileset.tileFailed.addEventListener(fail);
  viewer.scene.renderError.addEventListener((_scene, error) => fail(error));
  // Expose geometry, not a ready flag, so browser checks must inspect rendered pixels.
  window.phase0 = {
    viewer, tileset, fixture,
    project(enu) {
      const world = C.Matrix4.multiplyByPoint(frame, new C.Cartesian3(...enu), new C.Cartesian3());
      const p = C.SceneTransforms.worldToWindowCoordinates(viewer.scene, world);
      const rect = viewer.canvas.getBoundingClientRect();
      return { x: p.x + rect.left, y: p.y + rect.top };
    },
    world(enu) { return C.Matrix4.multiplyByPoint(frame, new C.Cartesian3(...enu), new C.Cartesian3()); },
  };
  tileset.allTilesLoaded.addEventListener(() => {
    status.textContent = `Fixture loaded · CesiumJS ${C.VERSION} · SPZ v2 · 5 splats`;
  });
} catch (error) {
  status.textContent = `Failed: ${error.message}`;
  console.error(error);
}
