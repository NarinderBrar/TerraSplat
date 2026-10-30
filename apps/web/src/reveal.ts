import type * as Cesium from 'cesium';

/**
 * Splat reveal effects for Cesium 1.146 (docs/REVEAL_EFFECTS.md).
 *
 * Cesium merges every selected tile of a tileset into one splat batch, so the GPU no longer knows a splat's tile.
 * We record the merge order by wrapping GaussianSplatPrimitive.transformTile (called once per tile, in merge order,
 * after the data generation is bumped), then give the committed snapshot a per-splat texture of tile state and
 * inject the effect into the splat vertex shader. Per merge, each tile is:
 * - FULL: appearing where nothing related was shown: glowing points sweep in from the block centre, then grow;
 * - GROW: replacing a shown ancestor (zooming in): starts as points and grows into its splats;
 * - NONE: replacing shown descendants (zooming out), or already shown: unchanged;
 * - a placeholder, independently: shown only because its children are still loading. It draws as points, fading back
 *   to its splats after PLACEHOLDER_SECONDS if the children never arrive.
 * Once a reveal finishes and no placeholder remains, the shader skips every change.
 */

// Exact text from Cesium 1.146 PrimitiveGaussianSplatVS.glsl; injection is skipped if any marker is missing.
const MARKERS = {
  main: 'void main() {',
  cov: 'vec4 covVectors = calcCovVectors(splatViewPos.xyz, Vrk);',
  discard: 'if (dot(covVectors.xy, covVectors.xy) < 4.0 && dot(covVectors.zw, covVectors.zw) < 4.0) {',
  end: 'v_splitDirection = u_splitDirection;',
};

export const PLACEHOLDER_SECONDS = 5;
export enum Mode { None = 0, Full = 1, Grow = 2 }
export interface TileState { arrival: number; mode: Mode; placeholderSince: number }

const GLOBALS = /* glsl */ `
uniform highp sampler2D u_revealTexture;
uniform float u_revealNow;
uniform float u_revealDuration;
uniform float u_revealSpeed;
uniform float u_revealPointPx;
uniform float u_revealAnimate;

// x: progress 0..1 (1 when not animating); y: placeholder point weight 0..1; z: mode (0 none, 1 full, 2 grow).
vec3 revealState(uint index, vec3 position) {
  uint rowMask = uint(u_splatRowMask);
  vec4 data = texelFetch(u_revealTexture, ivec2(int(index & rowMask), int(index >> uint(u_splatRowShift))), 0);
  if (u_revealAnimate < 0.5) return vec3(1.0, 0.0, 0.0);
  float progress = 1.0;
  if (data.y > 0.5) {
    bool full = data.y < 1.5;
    float delay = full ? length(position.xy) / u_revealSpeed : 0.0;
    progress = clamp((u_revealNow - data.x - delay) / (full ? u_revealDuration : 0.55 * u_revealDuration), 0.0, 1.0);
  }
  float points = data.z >= 0.0 ? 1.0 - smoothstep(PLACEHOLDER_SECONDS - 1.0, PLACEHOLDER_SECONDS, u_revealNow - data.z) : 0.0;
  return vec3(progress, points, data.y);
}
`;

const COVARIANCE = /* glsl */ `
    vec3 revealData = revealState(texIdx, splatPosition.xyz);
    float reveal = revealData.x;
    bool revealing = reveal < 1.0 || revealData.y > 0.0;
    bool growOnly = revealData.z > 1.5;
    // FULL: points appear behind the sweep (0..0.45), then grow into splats (0.45..1). GROW: points grow (0..1).
    float revealPoints = growOnly ? 1.0 : smoothstep(0.0, 0.45, reveal);
    float revealGrow = min(growOnly ? smoothstep(0.0, 1.0, reveal) : smoothstep(0.45, 1.0, reveal), 1.0 - revealData.y);
    // Loading pulse for placeholder points: a ripple travelling out from the block centre about once a second,
    // offset per point so the cloud shimmers rather than flashing in unison.
    float revealJitter = fract(sin(dot(splatPosition.xyz, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
    float revealPulse = revealData.y * (0.5 + 0.5 * sin(6.2831853 * (u_revealNow - length(splatPosition.xy) / 120.0
        + 0.35 * revealJitter)));
    if (revealing) {
        if (revealPoints <= 0.0) {
            gl_Position = discardVec;
            return;
        }
        float pointPx = u_revealPointPx * revealPoints * (1.0 + 0.6 * revealPulse);
        vec4 revealAxes = vec4(pointPx, 0.0, 0.0, pointPx);
        if (revealData.y > 0.0) {
            // Placeholders are sparse coarse tiles: draw each splat as a shrunken copy of itself (25-45%, breathing
            // with the pulse) so the cloud stays visible, but never smaller than a point.
            vec4 shrunk = covVectors * (0.25 + 0.2 * revealPulse / revealData.y);
            if (dot(shrunk.xy, shrunk.xy) > pointPx * pointPx) revealAxes = mix(revealAxes, shrunk, revealData.y);
        }
        covVectors = mix(revealAxes, covVectors, revealGrow);
    }
`;

const COLOUR = /* glsl */ `
    if (revealing) {
        // Points stay visible regardless of splat opacity; a glow marks the sweep front and fades as splats grow.
        float glow = (reveal < 1.0 ? (growOnly ? 0.3 : 0.6) * exp(-10.0 * reveal) : 0.0) + 0.55 * revealPulse;
        v_splatColor.a = mix(max(v_splatColor.a, 0.6 + 0.35 * revealPulse + 0.25 * (1.0 - revealData.y)), v_splatColor.a, revealGrow);
        v_splatColor.rgb = min(v_splatColor.rgb + glow * vec3(0.55, 0.85, 1.0), vec3(1.0));
    }
`;

export function injectRevealShader(source: string): string | undefined {
  if (!Object.values(MARKERS).every(marker => source.split(marker).length === 2)) return undefined;
  return source
    .replace(MARKERS.main, `#define PLACEHOLDER_SECONDS ${PLACEHOLDER_SECONDS.toFixed(1)}\n` + GLOBALS + MARKERS.main)
    .replace(MARKERS.cov, MARKERS.cov + COVARIANCE)
    // Points are deliberately tiny, so Cesium's "negligible size" discard must not apply to revealing splats.
    .replace(MARKERS.discard, 'if (!revealing && dot(covVectors.xy, covVectors.xy) < 4.0 && dot(covVectors.zw, covVectors.zw) < 4.0) {')
    .replace(MARKERS.end, COLOUR + MARKERS.end);
}

/** Per-splat (arrival, mode, placeholder-since or -1, 0) in Cesium's row layout, truncated like Cesium's splat budget. */
export function revealTexels(tiles: (TileState & { count: number })[], numSplats: number, rowWidth: number) {
  const height = Math.max(1, Math.ceil(numSplats / rowWidth));
  const data = new Float32Array(rowWidth * height * 4);
  let offset = 0;
  for (const tile of tiles) {
    const end = Math.min(offset + tile.count, numSplats);
    for (let i = offset; i < end; i++) data.set([tile.arrival, tile.mode, tile.placeholderSince, 0], 4 * i);
    offset += tile.count;
    if (offset >= numSplats) break;
  }
  return { data, width: rowWidth, height };
}

/** How a tile entering a merge should appear, given the tiles of the previous merge. */
export function appearance<T extends { parent?: T }>(tile: T, previous: Set<T>): Mode {
  if (previous.has(tile)) return Mode.None;
  for (let a = tile.parent; a; a = a.parent) if (previous.has(a)) return Mode.Grow;
  for (const other of previous) for (let a = other.parent; a; a = a.parent) if (a === tile) return Mode.None;
  return Mode.Full;
}

/** Progress of a splat at `delay` seconds behind the sweep, as in the shader. */
export function revealProgress(now: number, arrival: number, delay: number, duration: number) {
  return Math.min(1, Math.max(0, (now - arrival - delay) / duration));
}

// Cesium 1.146 exports these renderer classes at runtime but marks them private in its declarations.
interface GpuObject { destroy(): void }
interface Program { _vertexShaderSource: { sources: string[]; defines: string[] }; _fragmentShaderSource: unknown;
  _attributeLocations: Record<string, number> }
interface DrawCommand { shaderProgram: Program; uniformMap: Record<string, () => unknown> }
interface Internals {
  GaussianSplatPrimitive?: Record<string, unknown>;
  ShaderSource: new (options: object) => unknown;
  ShaderProgram: { fromCache(options: object): Program };
  Texture: new (options: object) => GpuObject;
  Sampler: new (options: object) => unknown;
}
interface Primitive {
  _splatDataGeneration: number; _snapshot?: { generation: number };
  _numSplats: number; _splatRowMask: number; _drawCommand?: DrawCommand; isDestroyed(): boolean;
}
interface Tile {
  parent?: Tile; children: unknown[]; _screenSpaceError: number; content: { pointsLength: number };
  tileset: { gaussianSplatPrimitive?: Primitive; memoryAdjustedScreenSpaceError: number };
}
// Splat counts are captured at merge time: Cesium may unload a tile's content while its merge is still on screen.
interface Merge { generation: number; tiles: Tile[]; states: TileState[]; counts: number[] }

export class Reveal {
  /** False when Cesium's internals differ from 1.146; the viewer then renders exactly as before. */
  readonly installed: boolean;
  animate = !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  duration = 1.8;
  speed = 250; // metres per second for the sweep from the block centre
  pointPx = 1.6;
  private readonly start = performance.now() / 1000;
  private readonly states = new WeakMap<Tile, TileState>();
  private readonly merges = new WeakMap<Primitive, Map<number, Merge>>();
  private readonly textures = new WeakMap<Primitive, { generation: number; texture: GpuObject; context: unknown }>();
  private readonly committed = new Set<Primitive>();
  /** Counts by mode in the latest committed merge, for diagnostics and tests. */
  readonly stats = { merges: 0, full: 0, grow: 0, none: 0, placeholders: 0 };
  private busyUntil = -Infinity;

  constructor(private readonly C: typeof Cesium, private readonly scene: Cesium.Scene) {
    const P = (C as unknown as Internals).GaussianSplatPrimitive;
    const transformTile = P?.transformTile, build = P?.buildGSplatDrawCommand;
    this.installed = typeof transformTile === 'function' && typeof build === 'function';
    if (!this.installed || !P) { console.warn('Reveal effects disabled: unexpected Cesium splat internals'); return; }
    P.transformTile = (tile: Tile) => {
      this.guard(() => this.record(tile));
      return (transformTile as (t: Tile) => unknown)(tile);
    };
    P.buildGSplatDrawCommand = (primitive: Primitive, frameState: { context: unknown }) => {
      (build as (p: Primitive, f: unknown) => void)(primitive, frameState);
      this.guard(() => this.decorate(primitive, frameState.context));
    };
    // Zooming in keeps the same coarse tiles selected until their children load, so Cesium does not re-merge;
    // placeholder status must be re-checked every frame.
    scene.preRender.addEventListener(() => this.guard(() => this.refreshPlaceholders()));
  }

  private failed = false;
  /** Effects must never stop Cesium's render loop: on any error, switch them off and render as before. */
  private guard(action: () => void) {
    if (this.failed) return;
    try { action(); } catch (error) {
      this.failed = true;
      this.animate = false;
      console.warn('Reveal effects disabled after an error', error);
    }
  }

  private static isPlaceholder(tile: Tile) {
    // Wants more detail (Cesium would refine it) but is drawn because its children are not loaded yet.
    return tile.children.length > 0 && tile._screenSpaceError > tile.tileset.memoryAdjustedScreenSpaceError;
  }

  private refreshPlaceholders() {
    if (!this.animate) return;
    const now = this.now();
    for (const primitive of this.committed) {
      if (primitive.isDestroyed()) { this.committed.delete(primitive); continue; }
      const cached = this.textures.get(primitive), merge = cached && this.merges.get(primitive)?.get(cached.generation);
      if (!cached || !merge) continue;
      let changed = false;
      merge.tiles.forEach((tile, i) => {
        const state = merge.states[i], placeholder = Reveal.isPlaceholder(tile);
        if (placeholder !== state.placeholderSince >= 0) {
          state.placeholderSince = placeholder ? now : -1;
          changed = true;
          if (placeholder) this.busyUntil = Math.max(this.busyUntil, now + PLACEHOLDER_SECONDS);
        }
      });
      if (changed) this.upload(primitive, merge, cached.context);
    }
  }

  now() { return performance.now() / 1000 - this.start; }
  /** True while any tile's sweep or growth is still playing; the caller keeps requesting frames. */
  active() { return this.installed && this.animate && this.now() < this.busyUntil; }

  private record(tile: Tile) {
    const primitive = tile.tileset.gaussianSplatPrimitive;
    if (!primitive) return;
    const merges = this.merges.get(primitive) ?? new Map<number, Merge>();
    this.merges.set(primitive, merges);
    const generation = primitive._splatDataGeneration;
    let merge = merges.get(generation);
    if (!merge) { merge = { generation, tiles: [], states: [], counts: [] }; merges.set(generation, merge); }
    const previous = [...merges.values()].filter(m => m.generation < generation).at(-1);
    const now = this.now(), prior = this.states.get(tile);
    const mode = appearance(tile, new Set(previous?.tiles ?? []));
    // Kept tiles continue their animation; new appearances restart it.
    const state: TileState = mode === Mode.None && prior && previous?.tiles.includes(tile)
      ? { ...prior } : { arrival: now, mode, placeholderSince: -1 };
    const placeholder = Reveal.isPlaceholder(tile);
    state.placeholderSince = placeholder ? (prior && prior.placeholderSince >= 0 ? prior.placeholderSince : now) : -1;
    this.states.set(tile, state);
    merge.tiles.push(tile);
    merge.states.push(state);
    merge.counts.push(tile.content.pointsLength);
    if (state.mode !== Mode.None) this.busyUntil = Math.max(this.busyUntil, state.arrival + 375 / this.speed + this.duration);
    if (placeholder) this.busyUntil = Math.max(this.busyUntil, state.placeholderSince + PLACEHOLDER_SECONDS);
  }

  private decorate(primitive: Primitive, context: unknown) {
    const command = primitive._drawCommand;
    const generation = primitive._snapshot?.generation;
    const merges = this.merges.get(primitive);
    const merge = generation === undefined ? undefined : merges?.get(generation);
    if (!command || !merges || !merge || generation === undefined) return;
    const vs = command.shaderProgram._vertexShaderSource;
    const injected = injectRevealShader(vs.sources.join('\n'));
    if (!injected) return;
    // Keep the committed merge (the next merge compares against it) and newer ones still in flight.
    for (const key of [...merges.keys()]) if (key < generation) merges.delete(key);

    const R = this.C as unknown as Internals;
    if (this.textures.get(primitive)?.generation !== generation) this.upload(primitive, merge, context);
    this.committed.add(primitive);
    Object.assign(this.stats, {
      merges: this.stats.merges + 1,
      full: merge.states.filter(t => t.mode === Mode.Full).length,
      grow: merge.states.filter(t => t.mode === Mode.Grow).length,
      none: merge.states.filter(t => t.mode === Mode.None).length,
      placeholders: merge.states.filter(t => t.placeholderSince >= 0).length,
    });
    command.shaderProgram = R.ShaderProgram.fromCache({
      context,
      vertexShaderSource: new R.ShaderSource({ sources: [injected], defines: vs.defines }),
      fragmentShaderSource: command.shaderProgram._fragmentShaderSource,
      attributeLocations: command.shaderProgram._attributeLocations,
    });
    Object.assign(command.uniformMap, {
      // Read through the cache: placeholder refreshes replace the texture without rebuilding the command.
      u_revealTexture: () => this.textures.get(primitive)?.texture,
      u_revealNow: () => this.now(),
      u_revealDuration: () => this.duration,
      u_revealSpeed: () => this.speed,
      u_revealPointPx: () => this.pointPx,
      u_revealAnimate: () => (this.animate ? 1 : 0),
    });
    this.scene.requestRender();
  }

  private upload(primitive: Primitive, merge: Merge, context: unknown) {
    const C = this.C, R = C as unknown as Internals;
    const texels = revealTexels(merge.states.map((state, i) => ({ ...state, count: merge.counts[i] })),
      primitive._numSplats, primitive._splatRowMask + 1);
    this.textures.get(primitive)?.texture.destroy();
    const texture = new R.Texture({
      context, width: texels.width, height: texels.height,
      pixelFormat: C.PixelFormat.RGBA, pixelDatatype: C.PixelDatatype.FLOAT, flipY: false,
      source: { width: texels.width, height: texels.height, arrayBufferView: texels.data },
      sampler: new R.Sampler({ minificationFilter: C.TextureMinificationFilter.NEAREST,
        magnificationFilter: C.TextureMagnificationFilter.NEAREST }),
    });
    const cached = { generation: merge.generation, texture, context };
    this.textures.set(primitive, cached);
    this.scene.requestRender();
    return cached;
  }
}
