import type * as Cesium from 'cesium';
import { apiUrl, getJson, type Dataset, type Footprint, type Release, type Manifest } from './api';
import { footprintCentre, footprintTooltip } from './format';
import { controlMapping } from './controls';
import { regionalIndexUrl } from './regionalIndex';
import { groundPatches, heightmap } from './ground';

declare global { interface Window { Cesium: typeof Cesium; terra?: Globe } }
const C = window.Cesium;
const PIN_NEAR_M = 4000, PIN_HIT_PX = 12;

function insideRing(lon: number, lat: number, ring: number[][]) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > lat) !== (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
export interface MapStats { loaded: number; visible: number; heading: number; height: number }
const tileOptions = {
  maximumScreenSpaceError: 12, cacheBytes: 128 * 1024 * 1024,
  maximumCacheOverflowBytes: 32 * 1024 * 1024,
  cullRequestsWhileMoving: true, foveatedScreenSpaceError: true, skipLevelOfDetail: false,
};
export class Globe {
  readonly viewer: Cesium.Viewer;
  private tileUrls = new WeakMap<Cesium.Cesium3DTileset, string>();
  private loadedTiles = new WeakMap<Cesium.Cesium3DTileset, Set<Cesium.Cesium3DTile>>();
  private indexes = new Map<string, Cesium.Cesium3DTileset>();
  private overrides = new Map<string, Cesium.Cesium3DTileset[]>();
  private footprints = new C.CustomDataSource('footprints');
  private features = new Map<string, Footprint>();
  private handler: Cesium.ScreenSpaceEventHandler;
  private visible = new Set<Cesium.Cesium3DTile>();
  private debug = false;
  private generation = 0;
  private disposed = false;
  private removeListeners: (() => void)[] = [];
  private hovered?: string;
  private lastStats = '';
  loadedUrls: string[] = [];
  constructor(container: HTMLElement, private onStats: (stats: MapStats) => void,
    private onHover: (text: string, x: number, y: number) => void,
    private onSelect: (feature: Footprint) => void, private onError: (error: unknown) => void) {
    this.viewer = new C.Viewer(container, {
      baseLayer: false, baseLayerPicker: false, geocoder: false, homeButton: false,
      sceneModePicker: false, navigationHelpButton: false, animation: false, timeline: false,
      fullscreenButton: false, infoBox: false, selectionIndicator: false,
      skyBox: false, skyAtmosphere: false, requestRenderMode: true, maximumRenderTimeChange: Infinity,
      contextOptions: { webgl: { preserveDrawingBuffer: import.meta.env.VITE_TEST_BRIDGE === 'true' } },
    });
    const { scene, camera } = this.viewer;
    scene.highDynamicRange = false;
    scene.backgroundColor = C.Color.fromCssColorString('#101a24');
    scene.globe.depthTestAgainstTerrain = false;
    // Fallback for releases without a recorded ground height: their splats lie below the ellipsoid, so fade the
    // globe near the surface to keep them visible. setFootprints replaces this with ground terrain when it can.
    scene.globe.translucency.enabled = true;
    scene.globe.translucency.frontFaceAlphaByDistance = new C.NearFarScalar(800, 0.12, 8000, 1);
    scene.globe.showGroundAtmosphere = false;
    scene.fog.enabled = true;
    Object.assign(scene.screenSpaceCameraController, controlMapping(C));
    camera.setView({ destination: C.Cartesian3.fromDegrees(-70, 22, 22_000_000) });
    camera.percentageChanged = 0.01;
    this.viewer.screenSpaceEventHandler.removeInputAction(C.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
    this.handler = new C.ScreenSpaceEventHandler(this.viewer.canvas);
    this.handler.setInputAction((e: { position: Cesium.Cartesian2 }) => this.zoom(0.5, e.position), C.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
    this.handler.setInputAction((e: { position: Cesium.Cartesian2 }) => {
      const f = this.picked(e.position); if (f) this.onSelect(f);
    }, C.ScreenSpaceEventType.LEFT_CLICK);
    // Hover runs at most once per animation frame and never renders: scene.pick() re-rendered the whole
    // scene (splats included) into a pick buffer on every mouse move, ~110 ms each in a profiled session.
    let pending: Cesium.Cartesian2 | undefined;
    this.handler.setInputAction((e: { endPosition: Cesium.Cartesian2 }) => {
      if (!pending) requestAnimationFrame(() => {
        const position = pending!; pending = undefined;
        if (this.disposed) return;
        const f = this.picked(position);
        this.highlight(f?.properties.release_id);
        this.onHover(f ? footprintTooltip(f) : '', position.x, position.y);
      });
      pending = C.Cartesian2.clone(e.endPosition);
    }, C.ScreenSpaceEventType.MOUSE_MOVE);
    // Splat decoding/sorting finishes on workers after camera movement has stopped.
    // Pinned 1.146 exposes this event at runtime but omits it from its declarations.
    const tasks = C.TaskProcessor as typeof C.TaskProcessor & { taskCompletedEvent: Cesium.Event };
    this.removeListeners.push(tasks.taskCompletedEvent.addEventListener(() => scene.requestRender()));
    this.removeListeners.push(scene.preRender.addEventListener(() => this.visible.clear()));
    this.removeListeners.push(scene.postRender.addEventListener(() => {
      const stats = this.stats();
      const key = JSON.stringify(stats);
      if (key !== this.lastStats) { this.lastStats = key; this.onStats(stats); }
    }));
    this.removeListeners.push(camera.changed.addEventListener(() => {
      scene.requestRender();
    }));
    this.removeListeners.push(camera.moveEnd.addEventListener(() => {
      if (camera.positionCartographic.height > 20_000) return;
      const rect = camera.computeViewRectangle();
      const f = [...this.features.values()].find(feature => {
        const [lon, lat] = footprintCentre(feature);
        return rect && C.Rectangle.contains(rect, C.Cartographic.fromDegrees(lon, lat));
      });
      if (f) this.onSelectInView?.(f);
    }));
    this.removeListeners.push(scene.renderError.addEventListener((_scene, error) => this.onError(error)));
    void this.imagery().catch(onError);
    void this.viewer.dataSources.add(this.footprints);
    if (import.meta.env.VITE_TEST_BRIDGE === 'true') window.terra = this;
  }
  onSelectInView?: (feature: Footprint) => void;
  private async imagery() {
    const provider = await C.TileMapServiceImageryProvider.fromUrl('/cesium/Assets/Textures/NaturalEarthII');
    if (this.disposed) return;
    this.viewer.imageryLayers.addImageryProvider(provider);
    if (import.meta.env.VITE_REMOTE_IMAGERY !== 'false') {
      // Shown at every height so the globe looks the same zoomed out and in; Natural Earth stays underneath as the instant first frame.
      this.viewer.imageryLayers.addImageryProvider(new C.UrlTemplateImageryProvider({
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        maximumLevel: 19, credit: new C.Credit('Imagery © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community', true),
      }));
    }
    this.viewer.scene.requestRender();
  }
  /** Geometric hit test: pins by screen distance, outlines by the ground point under the cursor. No pick render. */
  private picked(position: Cesium.Cartesian2) {
    if (!this.footprints.show || !this.features.size) return undefined;
    const { scene, camera } = this.viewer;
    const occluder = new C.EllipsoidalOccluder(C.Ellipsoid.WGS84, camera.positionWC);
    for (const f of this.features.values()) {
      const [lon, lat] = footprintCentre(f);
      const pin = C.Cartesian3.fromDegrees(lon, lat);
      // Pins are only drawn beyond PIN_NEAR_M (see setFootprints).
      if (C.Cartesian3.distance(camera.positionWC, pin) < PIN_NEAR_M || !occluder.isPointVisible(pin)) continue;
      const screen = C.SceneTransforms.worldToWindowCoordinates(scene, pin);
      if (screen && C.Cartesian2.distance(screen, position) <= PIN_HIT_PX) return f;
    }
    const ray = camera.getPickRay(position);
    const ground = (ray && scene.globe.pick(ray, scene)) ?? camera.pickEllipsoid(position);
    if (!ground) return undefined;
    const c = C.Cartographic.fromCartesian(ground);
    const lon = C.Math.toDegrees(c.longitude), lat = C.Math.toDegrees(c.latitude);
    return [...this.features.values()].find(f => insideRing(lon, lat, f.geometry.coordinates[0]));
  }
  private highlight(id?: string) {
    if (this.hovered === id) return;
    this.hovered = id;
    for (const entity of this.footprints.entities.values) {
      const color = entity.id.startsWith(`${id}:`) ? C.Color.YELLOW : C.Color.fromCssColorString('#65f4bc');
      if (entity.polyline) entity.polyline.material = new C.ColorMaterialProperty(color);
      if (entity.point) entity.point.color = new C.ConstantProperty(color);
    }
    this.viewer.scene.requestRender();
  }
  setFootprints(features: Footprint[]) {
    for (const f of features) {
      const id = f.properties.release_id; this.features.set(id, f);
      this.footprints.entities.add({ id: `${id}:outline`, polyline: {
        positions: C.Cartesian3.fromDegreesArray(f.geometry.coordinates[0].flatMap(p => p.slice(0, 2))),
        clampToGround: true, width: 3, material: C.Color.fromCssColorString('#65f4bc'),
      } });
      const [lon, lat] = footprintCentre(f);
      this.footprints.entities.add({ id: `${id}:pin`, position: C.Cartesian3.fromDegrees(lon, lat), point: {
        pixelSize: 15, color: C.Color.fromCssColorString('#65f4bc'), outlineColor: C.Color.BLACK,
        outlineWidth: 3, distanceDisplayCondition: new C.DistanceDisplayCondition(PIN_NEAR_M, Infinity),
      } });
    }
    this.footprints.show = false;
    this.useGroundTerrain(features);
  }
  /** Lift the base imagery to each release's model ground so splats sit on an opaque, bright map. */
  private useGroundTerrain(features: Footprint[]) {
    const patches = groundPatches(features);
    if (!patches.length) return;
    const tiling = new C.GeographicTilingScheme(), size = 32;
    this.viewer.terrainProvider = new C.CustomHeightmapTerrainProvider({
      width: size, height: size, tilingScheme: tiling,
      callback: (x, y, level) => {
        const r = tiling.tileXYToRectangle(x, y, level);
        const deg = C.Math.toDegrees;
        return heightmap(deg(r.west), deg(r.south), deg(r.east), deg(r.north), size, patches);
      },
    });
    this.viewer.scene.globe.translucency.enabled = false;
    this.viewer.scene.requestRender();
  }
  showFootprints(show: boolean) { this.footprints.show = show; this.viewer.scene.requestRender(); }
  private async load(url: string, resourceUrl = apiUrl(url)) {
    const tileset = await C.Cesium3DTileset.fromUrl(resourceUrl, tileOptions);
    if (this.disposed) { tileset.destroy(); throw new Error('Viewer closed'); }
    this.tileUrls.set(tileset, url);
    const loaded = new Set<Cesium.Cesium3DTile>();
    this.loadedTiles.set(tileset, loaded);
    tileset.tileLoad.addEventListener(tile => { loaded.add(tile); this.viewer.scene.requestRender(); });
    tileset.tileUnload.addEventListener(tile => { loaded.delete(tile); this.viewer.scene.requestRender(); });
    tileset.debugShowBoundingVolume = this.debug;
    tileset.tileVisible.addEventListener(tile => this.visible.add(tile));
    tileset.tileFailed.addEventListener(error => this.onError(error));
    tileset.allTilesLoaded.addEventListener(() => this.viewer.scene.requestRender());
    this.viewer.scene.primitives.add(tileset);
    this.viewer.scene.requestRender();
    return tileset;
  }
  async loadIndexes(datasets: Dataset[], signal: AbortSignal) {
    await Promise.all([...new Set(datasets.map(d => d.region))].map(async region => {
      const link = await getJson(`/api/index/${encodeURIComponent(region)}`, signal) as { tileset_url: string };
      if (typeof link.tileset_url !== 'string') throw new Error('Invalid regional index');
      const json = await getJson(link.tileset_url, signal);
      const sourceUrl = new URL(apiUrl(link.tileset_url), window.location.href).href;
      this.indexes.set(region, await this.load(link.tileset_url, regionalIndexUrl(json, sourceUrl)));
    }));
    this.loadedUrls = [...this.indexes.keys()].map(region => `/releases/index/${region}/tileset.json`);
  }
  async switchRelease(dataset: Dataset, release: Release, datasets: Dataset[]) {
    const generation = ++this.generation;
    const index = this.indexes.get(dataset.region);
    if (!index) throw new Error('Regional index is still loading');
    if (release.release_id === dataset.active_release_id) {
      for (const t of this.overrides.get(dataset.region) ?? []) this.viewer.scene.primitives.remove(t);
      this.overrides.delete(dataset.region); index.show = true;
    } else {
      // Preserve other blocks in this region while substituting this dataset's release.
      const releases = datasets.filter(d => d.region === dataset.region)
        .map(d => d.dataset_id === dataset.dataset_id ? release : d.active_release).filter(r => r !== null);
      const loaded: Cesium.Cesium3DTileset[] = [];
      try {
        for (const r of releases) loaded.push(await this.load(r.tileset_url));
      } catch (error) { loaded.forEach(t => this.viewer.scene.primitives.remove(t)); throw error; }
      if (generation !== this.generation || this.disposed) { loaded.forEach(t => { if (!t.isDestroyed()) this.viewer.scene.primitives.remove(t); }); return; }
      for (const t of this.overrides.get(dataset.region) ?? []) this.viewer.scene.primitives.remove(t);
      this.overrides.set(dataset.region, loaded); index.show = false;
    }
    this.loadedUrls = [...this.indexes].flatMap(([region, t]) => t.show ? [`/releases/index/${region}/tileset.json`] :
      (this.overrides.get(region) ?? []).map(t => this.tileUrls.get(t)!));
    this.viewer.scene.requestRender();
  }
  flyTo(feature: Footprint): Promise<void> {
    const positions = feature.geometry.coordinates[0].map(p => C.Cartesian3.fromDegrees(p[0], p[1], -17));
    const sphere = C.BoundingSphere.fromPoints(positions);
    return new Promise(resolve => {
      this.viewer.camera.flyToBoundingSphere(sphere, {
        duration: 1.8, offset: new C.HeadingPitchRange(0, -0.65, Math.max(sphere.radius * 3.2, 400)),
        complete: resolve, cancel: resolve,
      });
      this.viewer.scene.requestRender();
    });
  }
  zoom(factor: number, position = new C.Cartesian2(this.viewer.canvas.clientWidth / 2, this.viewer.canvas.clientHeight / 2)) {
    const { camera } = this.viewer;
    const target = camera.pickEllipsoid(position, this.viewer.scene.globe.ellipsoid);
    if (!target) return;
    const destination = C.Cartesian3.lerp(target, camera.positionWC, factor, new C.Cartesian3());
    const height = C.Cartographic.fromCartesian(destination).height;
    if (height < 35 || height > 30_000_000) return;
    camera.flyTo({ destination, orientation: { heading: camera.heading, pitch: camera.pitch, roll: 0 }, duration: 0.35 });
    this.viewer.scene.requestRender();
  }
  north(top = false) {
    const camera = this.viewer.camera;
    const centre = new C.Cartesian2(this.viewer.canvas.clientWidth / 2, this.viewer.canvas.clientHeight / 2);
    const target = camera.pickEllipsoid(centre);
    if (target) camera.flyToBoundingSphere(new C.BoundingSphere(target, 1), {
      offset: new C.HeadingPitchRange(0, top ? -Math.PI / 2 : camera.pitch, C.Cartesian3.distance(camera.positionWC, target)), duration: 0.5,
    });
  }
  setDebug(value: boolean) {
    this.debug = value; this.viewer.scene.debugShowFramesPerSecond = value;
    for (const t of [...this.indexes.values(), ...[...this.overrides.values()].flat()]) t.debugShowBoundingVolume = value;
    this.viewer.scene.requestRender();
  }
  stats(): MapStats {
    const active = [...this.indexes.values(), ...[...this.overrides.values()].flat()].filter(t => t.show);
    return { loaded: active.reduce((n, t) => n + (this.loadedTiles.get(t)?.size ?? 0), 0), visible: this.visible.size,
      heading: Math.round(C.Math.toDegrees(this.viewer.camera.heading)), height: Math.round(this.viewer.camera.positionCartographic.height) };
  }
  projectFootprint(manifest: Manifest) {
    return manifest.footprint.coordinates[0].slice(0, -1).map(([lon, lat]) =>
      C.SceneTransforms.worldToWindowCoordinates(this.viewer.scene, C.Cartesian3.fromDegrees(lon, lat, -17)));
  }
  tilesReady() {
    return [...this.indexes.values(), ...[...this.overrides.values()].flat()].filter(t => t.show).every(t => t.tilesLoaded);
  }

  destroy() {
    this.disposed = true; ++this.generation;
    this.removeListeners.forEach(remove => remove()); this.handler.destroy(); this.viewer.destroy();
    if (window.terra === this) delete window.terra;
  }
}
