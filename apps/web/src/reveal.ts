import type * as Cesium from 'cesium';

/**
 * Splat reveal effects for Cesium 1.146 (docs/REVEAL_EFFECTS.md).
 *
 * Cesium merges every selected tile of a tileset into one splat batch, so the GPU no longer knows a splat's tile.
 * We record the merge order by wrapping GaussianSplatPrimitive.transformTile (called once per tile, in merge order,
 * after the data generation is bumped), then give the committed snapshot a per-splat texture of (arrival time,
 * animate) and inject the effect into the splat vertex shader. A tile appearing where nothing was shown before plays
 * two stages: glowing points sweep in from the block centre, then grow into its splats. Refinement (children
 * replacing a visible ancestor) swaps instantly as before. Once a reveal finishes the shader skips every change.
 */

// Exact text from Cesium 1.146 PrimitiveGaussianSplatVS.glsl; injection is skipped if any marker is missing.
const MARKERS = {
  main: 'void main() {',
  cov: 'vec4 covVectors = calcCovVectors(splatViewPos.xyz, Vrk);',
  discard: 'if (dot(covVectors.xy, covVectors.xy) < 4.0 && dot(covVectors.zw, covVectors.zw) < 4.0) {',
  end: 'v_splitDirection = u_splitDirection;',
};

const GLOBALS = /* glsl */ `
uniform highp sampler2D u_revealTexture;
uniform float u_revealNow;
uniform float u_revealDuration;
uniform float u_revealSpeed;
uniform float u_revealPointPx;
uniform float u_revealAnimate;

// Reveal progress 0..1 after a sweep delay from the block centre; 1 for tiles that do not animate.
float revealProgress(uint index, vec3 position) {
  uint rowMask = uint(u_splatRowMask);
  vec2 data = texelFetch(u_revealTexture, ivec2(int(index & rowMask), int(index >> uint(u_splatRowShift))), 0).rg;
  if (u_revealAnimate < 0.5 || data.y < 0.5) return 1.0;
  return clamp((u_revealNow - data.x - length(position.xy) / u_revealSpeed) / u_revealDuration, 0.0, 1.0);
}
`;

const COVARIANCE = /* glsl */ `
    float reveal = revealProgress(texIdx, splatPosition.xyz);
    bool revealing = reveal < 1.0;
    // Stage 1 (0..0.45): points appear behind the sweep. Stage 2 (0.45..1): points grow into the splats.
    float revealPoints = smoothstep(0.0, 0.45, reveal);
    float revealGrow = smoothstep(0.45, 1.0, reveal);
    if (revealing) {
        if (revealPoints <= 0.0) {
            gl_Position = discardVec;
            return;
        }
        float pointPx = u_revealPointPx * revealPoints;
        covVectors = mix(vec4(pointPx, 0.0, 0.0, pointPx), covVectors, revealGrow);
    }
`;

const COLOUR = /* glsl */ `
    if (revealing) {
        // Points stay visible regardless of splat opacity; a glow marks the sweep front and fades as splats grow.
        float glow = 0.6 * exp(-10.0 * reveal);
        v_splatColor.a = mix(max(v_splatColor.a, 0.85), v_splatColor.a, revealGrow);
        v_splatColor.rgb = min(v_splatColor.rgb + glow * vec3(0.55, 0.85, 1.0), vec3(1.0));
    }
`;

export function injectRevealShader(source: string): string | undefined {
  if (!Object.values(MARKERS).every(marker => source.split(marker).length === 2)) return undefined;
  return source
    .replace(MARKERS.main, GLOBALS + MARKERS.main)
    .replace(MARKERS.cov, MARKERS.cov + COVARIANCE)
    // Points are deliberately tiny, so Cesium's "negligible size" discard must not apply to revealing splats.
    .replace(MARKERS.discard, 'if (!revealing && dot(covVectors.xy, covVectors.xy) < 4.0 && dot(covVectors.zw, covVectors.zw) < 4.0) {')
    .replace(MARKERS.end, COLOUR + MARKERS.end);
}

/** Per-splat (arrival seconds, animate flag) in Cesium's row layout, truncated like Cesium's own splat budget. */
export function revealTexels(tiles: { count: number; arrival: number; animate: boolean }[], numSplats: number,
  rowWidth: number) {
  const height = Math.max(1, Math.ceil(numSplats / rowWidth));
  const data = new Float32Array(rowWidth * height * 2);
  let offset = 0;
  for (const tile of tiles) {
    const end = Math.min(offset + tile.count, numSplats);
    for (let i = offset; i < end; i++) { data[2 * i] = tile.arrival; data[2 * i + 1] = tile.animate ? 1 : 0; }
    offset += tile.count;
    if (offset >= numSplats) break;
  }
  return { data, width: rowWidth, height };
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
  _numSplats: number; _splatRowMask: number; _drawCommand?: DrawCommand;
}
interface Tile { parent?: Tile; content: { pointsLength: number }; tileset: { gaussianSplatPrimitive?: Primitive } }

export class Reveal {
  /** False when Cesium's internals differ from 1.146; the viewer then renders exactly as before. */
  readonly installed: boolean;
  animate = !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  duration = 1.8;
  speed = 250; // metres per second for the sweep from the block centre
  pointPx = 1.6;
  private readonly start = performance.now() / 1000;
  private readonly arrivals = new WeakMap<object, { time: number; animate: boolean }>();
  private readonly merges = new WeakMap<Primitive, Map<number, Tile[]>>();
  private readonly textures = new WeakMap<Primitive, { generation: number; texture: GpuObject }>();
  private lastArrival = -Infinity;

  constructor(private readonly C: typeof Cesium, private readonly scene: Cesium.Scene) {
    const P = (C as unknown as Internals).GaussianSplatPrimitive;
    const transformTile = P?.transformTile, build = P?.buildGSplatDrawCommand;
    this.installed = typeof transformTile === 'function' && typeof build === 'function';
    if (!this.installed || !P) { console.warn('Reveal effects disabled: unexpected Cesium splat internals'); return; }
    P.transformTile = (tile: Tile) => { this.record(tile); return (transformTile as (t: Tile) => unknown)(tile); };
    P.buildGSplatDrawCommand = (primitive: Primitive, frameState: { context: unknown }) => {
      (build as (p: Primitive, f: unknown) => void)(primitive, frameState);
      this.decorate(primitive, frameState.context);
    };
  }

  now() { return performance.now() / 1000 - this.start; }
  /** True while any tile's sweep or growth is still playing; the caller keeps requesting frames. */
  active() { return this.installed && this.animate && this.now() < this.lastArrival + 375 / this.speed + this.duration; }

  private record(tile: Tile) {
    const primitive = tile.tileset.gaussianSplatPrimitive;
    if (!primitive) return;
    const merges = this.merges.get(primitive) ?? new Map<number, Tile[]>();
    this.merges.set(primitive, merges);
    const generation = primitive._splatDataGeneration;
    if (!merges.has(generation)) merges.set(generation, []);
    merges.get(generation)!.push(tile);
    if (!this.arrivals.has(tile)) {
      // Refinement swaps instantly: replaying the point stage would turn a finished model back into dust.
      let ancestor = tile.parent, animate = true;
      for (; ancestor && animate; ancestor = ancestor.parent) animate = !this.arrivals.has(ancestor);
      const now = this.now();
      this.arrivals.set(tile, { time: now, animate });
      if (animate) this.lastArrival = Math.max(this.lastArrival, now);
    }
  }

  private decorate(primitive: Primitive, context: unknown) {
    const command = primitive._drawCommand;
    const generation = primitive._snapshot?.generation;
    const merges = this.merges.get(primitive);
    const tiles = generation === undefined ? undefined : merges?.get(generation);
    if (!command || !merges || !tiles || generation === undefined) return;
    const vs = command.shaderProgram._vertexShaderSource;
    const injected = injectRevealShader(vs.sources.join('\n'));
    if (!injected) return;
    // Drop merges older than the committed one; newer ones may still be in flight.
    for (const key of [...merges.keys()]) if (key < generation) merges.delete(key);

    const C = this.C, R = C as unknown as Internals;
    let cached = this.textures.get(primitive);
    if (!cached || cached.generation !== generation) {
      cached?.texture.destroy();
      const texels = revealTexels(tiles.map(tile => ({
        count: tile.content.pointsLength, arrival: this.arrivals.get(tile)?.time ?? -1e6,
        animate: this.arrivals.get(tile)?.animate ?? false,
      })), primitive._numSplats, primitive._splatRowMask + 1);
      const texture = new R.Texture({
        context, width: texels.width, height: texels.height,
        pixelFormat: C.PixelFormat.RG, pixelDatatype: C.PixelDatatype.FLOAT, flipY: false,
        source: { width: texels.width, height: texels.height, arrayBufferView: texels.data },
        sampler: new R.Sampler({ minificationFilter: C.TextureMinificationFilter.NEAREST,
          magnificationFilter: C.TextureMagnificationFilter.NEAREST }),
      });
      cached = { generation, texture };
      this.textures.set(primitive, cached);
    }
    command.shaderProgram = R.ShaderProgram.fromCache({
      context,
      vertexShaderSource: new R.ShaderSource({ sources: [injected], defines: vs.defines }),
      fragmentShaderSource: command.shaderProgram._fragmentShaderSource,
      attributeLocations: command.shaderProgram._attributeLocations,
    });
    const texture = cached.texture;
    Object.assign(command.uniformMap, {
      u_revealTexture: () => texture,
      u_revealNow: () => this.now(),
      u_revealDuration: () => this.duration,
      u_revealSpeed: () => this.speed,
      u_revealPointPx: () => this.pointPx,
      u_revealAnimate: () => (this.animate ? 1 : 0),
    });
    this.scene.requestRender();
  }
}
