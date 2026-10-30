import type * as Cesium from 'cesium';
import { apiUrl, type Dataset, type Footprint, type Release, type Manifest } from './api';
import { footprintCentre, footprintTooltip } from './format';
import { controlMapping } from './controls';
import { groundPatches, heightmap } from './ground';

declare global { interface Window { Cesium: typeof Cesium; terra?: Globe } }
const C = window.Cesium;
const PIN_NEAR_M = 4000, PIN_HIT_PX = 12, STATS_INTERVAL_MS = 100;

function insideRing(lon: number, lat: number, ring: number[][]) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > lat) !== (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
export interface MapStats { loaded: number; visible: number; bytes: number; heading: number; height: number }
export interface Layers { pipeline: boolean; skyfall: boolean; overlap: 'skyfall' | 'pipeline' }
/** Cesium asks for the discrete GPU; some hybrid-GPU setups (Chrome + NVIDIA offload on Linux) refuse that request
 * outright, so fall back to the browser's default GPU instead of failing to start. */
function powerPreference(): WebGLPowerPreference {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2', { powerPreference: 'high-performance' });
  gl?.getExtension('WEBGL_lose_context')?.loseContext();
  return gl ? 'high-performance' : 'default';
}
const tileOptions = {
  // 12 stopped at 2-4 m merged parents at 900 m (cars smeared); 8 reaches full-resolution leaves there.
  maximumScreenSpaceError: 8, cacheBytes: 128 * 1024 * 1024,
  maximumCacheOverflowBytes: 32 * 1024 * 1024,
  cullRequestsWhileMoving: true, foveatedScreenSpaceError: true, skipLevelOfDetail: false,
};
export class Globe {
  readonly viewer: Cesium.Viewer;
  private tileUrls = new WeakMap<Cesium.Cesium3DTileset, string>();
  private loadedTiles = new WeakMap<Cesium.Cesium3DTileset, Set<Cesium.Cesium3DTile>>();
  private indexes = new Map<string, Cesium.Cesium3DTileset>();
  private layers: Layers = { pipeline: true, skyfall: true, overlap: 'skyfall' };
  private cacheBudget = 128 * 1024 * 1024;
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
  private lastStatsAt = 0;
  private statsTimer?: ReturnType<typeof setTimeout>;
  loadedUrls: string[] = [];
  constructor(container: HTMLElement, private onStats: (stats: MapStats) => void,
    private onHover: (text: string, x: number, y: number) => void,
    private onSelect: (feature: Footprint) => void, private onError: (error: unknown) => void) {
    this.viewer = new C.Viewer(container, {
      baseLayer: false, baseLayerPicker: false, geocoder: false, homeButton: false,
      sceneModePicker: false, navigationHelpButton: false, animation: false, timeline: false,
      fullscreenButton: false, infoBox: false, selectionIndicator: false,
      skyBox: false, skyAtmosphere: false, requestRenderMode: true, maximumRenderTimeChange: Infinity,
      contextOptions: { webgl: { preserveDrawingBuffer: import.meta.env.VITE_TEST_BRIDGE === 'true',
        powerPreference: powerPreference() } },
    });
    const { scene, camera } = this.viewer;
    scene.highDynamicRange = false;
    scene.backgroundColor = C.Color.fromCssColorString('#101a24');
    scene.globe.depthTestAgainstTerrain = false;
    // The camera controller reads scene depth back from the GPU (pickPositionWorldCoordinates -> readPixels) on every
    // drag and wheel event, stalling until the frame finishes: ~3 s in an 8 s profiled session. Without depth picking it
    // intersects the globe on the CPU instead; zoom then aims at the ground under the cursor rather than a roof.
    // Nothing else in the app uses pickPosition.
    Object.defineProperty(scene, 'pickPositionSupported', { get: () => false });
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
    // Stats feed React state; emitting after every frame re-rendered the whole panel each frame.
    // Emit at most every STATS_INTERVAL_MS, with a trailing update once movement stops.
    this.removeListeners.push(scene.postRender.addEventListener(() => this.queueStats()));
    this.removeListeners.push(camera.changed.addEventListener(() => {
      scene.requestRender();
    }));
    this.removeListeners.push(camera.moveEnd.addEventListener(() => {
      if (camera.positionCartographic.height > 20_000) return;
      const rect = camera.computeViewRectangle();
      const f = [...this.features.values()].find(feature => {
        if (!this.datasetVisible(feature.properties.dataset_id)) return false;
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
  private async load(url: string, show = true) {
    const tileset = await C.Cesium3DTileset.fromUrl(apiUrl(url), { ...tileOptions, cacheBytes: this.cacheBudget,
      maximumCacheOverflowBytes: this.cacheBudget / 4 });
    tileset.show = show;
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
  private datasetVisible(id: string) {
    const skyfall = id.startsWith('skyfall-');
    if (!(skyfall ? this.layers.skyfall : this.layers.pipeline)) return false;
    const overlap = this.features.size && [...this.features.values()].some(f => f.properties.dataset_id === 'skyfall-jax-068');
    if (!overlap || !this.layers.skyfall || !this.layers.pipeline) return true;
    if (id === 'jax068-2015') return this.layers.overlap === 'pipeline';
    if (id === 'skyfall-jax-068') return this.layers.overlap === 'skyfall';
    return true;
  }
  setLayers(layers: Layers) {
    this.layers = layers;
    for (const [id, tileset] of this.indexes) tileset.show = this.datasetVisible(id);
    this.loadedUrls = [...this.indexes].filter(([, t]) => t.show).map(([, t]) => this.tileUrls.get(t)!);
    this.useGroundTerrain([...this.features.values()].filter(f => this.datasetVisible(f.properties.dataset_id)));
    this.viewer.scene.requestRender();
  }
  async loadIndexes(datasets: Dataset[], signal?: AbortSignal) {
    // Dataset tilesets allow overlap suppression without loading an opaque regional index.
    // Only small JSON roots load here; Cesium requests content through its camera traversal.
    this.cacheBudget = Math.floor(256 * 1024 * 1024 / Math.max(1, datasets.length));
    await Promise.all(datasets.filter(d => d.active_release).map(async d => {
      const t = await this.load(d.active_release!.tileset_url, this.datasetVisible(d.dataset_id));
      if (signal?.aborted) { this.viewer.scene.primitives.remove(t); return; }
      this.indexes.set(d.dataset_id, t);
    }));
    this.setLayers(this.layers);
  }
  async switchRelease(dataset: Dataset, release: Release, _datasets: Dataset[]) {
    const generation = ++this.generation;
    const loaded = await this.load(release.tileset_url, false);
    if (generation !== this.generation || this.disposed) {
      if (!loaded.isDestroyed()) this.viewer.scene.primitives.remove(loaded);
      return;
    }
    const old = this.indexes.get(dataset.dataset_id);
    if (old) this.viewer.scene.primitives.remove(old);
    this.indexes.set(dataset.dataset_id, loaded);
    this.setLayers(this.layers);
  }
  flyTo(feature: Footprint): Promise<void> {
    const positions = feature.geometry.coordinates[0].map(p => C.Cartesian3.fromDegrees(p[0], p[1], feature.properties.ground_height_m ?? 0));
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
    for (const t of [...this.indexes.values()]) t.debugShowBoundingVolume = value;
    this.viewer.scene.requestRender();
  }
  private queueStats() {
    if (this.statsTimer || this.disposed) return;
    const wait = Math.max(0, STATS_INTERVAL_MS - (performance.now() - this.lastStatsAt));
    this.statsTimer = setTimeout(() => {
      this.statsTimer = undefined;
      if (this.disposed) return;
      this.lastStatsAt = performance.now();
      const stats = this.stats(), key = JSON.stringify(stats);
      if (key !== this.lastStats) { this.lastStats = key; this.onStats(stats); }
    }, wait);
  }
  stats(): MapStats {
    const active = [...this.indexes.values()].filter(t => t.show);
    return { loaded: active.reduce((n, t) => n + (this.loadedTiles.get(t)?.size ?? 0), 0), visible: this.visible.size,
      bytes: active.reduce((n, t) => n + t.totalMemoryUsageInBytes, 0),
      heading: Math.round(C.Math.toDegrees(this.viewer.camera.heading)), height: Math.round(this.viewer.camera.positionCartographic.height) };
  }
  projectFootprint(manifest: Manifest) {
    return manifest.footprint.coordinates[0].slice(0, -1).map(([lon, lat]) =>
      C.SceneTransforms.worldToWindowCoordinates(this.viewer.scene, C.Cartesian3.fromDegrees(lon, lat, -17)));
  }
  tilesReady() {
    return [...this.indexes.values()].filter(t => t.show).every(t => t.tilesLoaded);
  }

  destroy() {
    this.disposed = true; ++this.generation; clearTimeout(this.statsTimer);
    this.removeListeners.forEach(remove => remove()); this.handler.destroy(); this.viewer.destroy();
    if (window.terra === this) delete window.terra;
  }
}
