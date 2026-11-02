import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { appearance, buildingHologramScale, BUILDING_HOLOGRAM_COLOUR, heldAncestors, injectRevealShader, Mode, revealProgress, revealTexels, SEGMENT_COLOURS, segmentUrls } from '../src/reveal';

const shader = readFileSync(new URL('../../../node_modules/@cesium/engine/Source/Shaders/PrimitiveGaussianSplatVS.glsl', import.meta.url), 'utf8');

describe('splat reveal effects', () => {
  it('injects into the pinned Cesium splat shader and refuses unknown shaders', () => {
    const injected = injectRevealShader(shader)!;
    expect(injected).toContain('uniform highp sampler2D u_revealTexture;');
    expect(injected).toContain('vec4 revealData = texelFetch(u_revealTexture,');
    expect(injected).toContain('#define PLACEHOLDER_SECONDS 20.0');
    expect(injected).toContain('if (!revealing && dot(covVectors.xy');
    // The effect sits between the covariance and the end of main, leaving Cesium's code otherwise intact.
    expect(injected.indexOf('vec4 revealData')).toBeGreaterThan(injected.indexOf('calcCovVectors(splatViewPos.xyz, Vrk)'));
    expect(injected.indexOf('if (revealing) {\n        // Appearing points')).toBeLessThan(injected.indexOf('v_splitDirection = u_splitDirection;'));
    expect(injectRevealShader(shader.replace('void main() {', 'void main(){'))).toBeUndefined();
  });

  it('lays out per-splat state in merge order and truncates like the splat budget', () => {
    const { data, width, height } = revealTexels([
      { count: 3, arrival: 1.5, mode: Mode.None, placeholderSince: 4, leaf: false },
      { count: 4, arrival: 2.5, mode: Mode.Grow, placeholderSince: -1, leaf: true }], 5, 4);
    expect([width, height]).toEqual([4, 2]);
    expect(Array.from(data.subarray(0, 4))).toEqual([1.5, 0, 4, 0]);
    expect(Array.from(data.subarray(12, 20))).toEqual([2.5, 2, -1, 1, 2.5, 2, -1, 1]);
    expect(Array.from(data.subarray(20))).toEqual(new Array(12).fill(0));
  });

  it('classifies appearances: new area, zoom in, zoom out, kept', () => {
    interface Node { name: string; parent?: Node }
    const root: Node = { name: 'root' }, child: Node = { name: 'child', parent: root };
    const grandchild: Node = { name: 'gc', parent: child };
    expect(appearance(root, new Set())).toBe(Mode.Full);
    expect(appearance(grandchild, new Set([root]))).toBe(Mode.Grow);
    expect(appearance(root, new Set([grandchild]))).toBe(Mode.None);
    expect(appearance(child, new Set([child]))).toBe(Mode.None);
    expect(appearance<Node>({ name: 'elsewhere' }, new Set([child]))).toBe(Mode.Full);
  });

  it('progress waits for the sweep, then reaches exactly 1', () => {
    expect(revealProgress(2, 1, 0.5, 1.2)).toBeCloseTo(0.5 / 1.2);
    expect(revealProgress(1.2, 1, 0.5, 1.2)).toBe(0);
    expect(revealProgress(9, 1, 0.5, 1.2)).toBe(1);
  });

  it('packs segment classes above the leaf bit and ignores class data of the wrong length', () => {
    const { data } = revealTexels([
      { count: 2, arrival: 0, mode: Mode.None, placeholderSince: -1, leaf: true, classes: new Uint8Array([3, 4]) },
      { count: 2, arrival: 0, mode: Mode.None, placeholderSince: -1, leaf: false, classes: new Uint8Array([2, 2]) },
      { count: 1, arrival: 0, mode: Mode.None, placeholderSince: -1, leaf: true, classes: new Uint8Array([5, 5]) }], 5, 8);
    expect([3, 7, 11, 15, 19].map(i => data[i])).toEqual([1 + 6, 1 + 8, 4, 4, 1]);
  });

  it('maps tile content to its segment file and release layer', () => {
    expect(segmentUrls('http://h/releases/jax-r2/tiles/r01.glb?v=1')).toEqual({
      layer: 'http://h/releases/jax-r2/segments/segments.json', tile: 'http://h/releases/jax-r2/segments/r01.cls' });
    expect(segmentUrls('http://h/releases/jax-r2/tileset.json')).toBeUndefined();
  });

  it('injects segment uniforms and uses the pipeline class colours', () => {
    const injected = injectRevealShader(shader)!;
    expect(injected).toContain('uniform float u_segmentMask;');
    expect(injected).toContain('int segmentClass = int(revealData.w + 0.5) / 2;');
    expect(injected).toContain('bool revealLeaf = mod(revealData.w, 2.0) > 0.5;');
    expect(injected).toContain('uniform float u_segmentMotion;');
    expect(injected).toContain('segmentHologram(segmentTint, splatPosition.xyz, segmentClass)');
    const python = readFileSync(new URL('../../../pipeline/semantics.py', import.meta.url), 'utf8');
    const colours = /COLOURS = \{([^}]*)\}/.exec(python)![1];
    const parsed = [...colours.matchAll(/(\d+): \((\d+), (\d+), (\d+)\)/g)].map(m => [Number(m[1]), [m[2], m[3], m[4]].map(Number)]);
    expect(Object.fromEntries(parsed)).toEqual(Object.fromEntries(SEGMENT_COLOURS.map((c, i) => [i, c])));
  });

  it('uses TerraScope building blue and its JAX_068 grid scale across release revisions', () => {
    expect(BUILDING_HOLOGRAM_COLOUR).toEqual([0.1, 0.65, 1]);
    expect(buildingHologramScale('http://h/releases/skyfall-jax-068-final-r3/tiles/r01.glb')).toBe(1.5);
    expect(buildingHologramScale('http://h/releases/skyfall-nyc-004-final-r2/tiles/r01.glb')).toBe(1);
    expect(buildingHologramScale('http://h/releases/jax068-2015-grid-a2-r2/tiles/r01.glb')).toBe(1);
  });
});

describe('held parents', () => {
  it('keeps the nearest loaded ancestor of growing tiles once, unless already selected', () => {
    type T = { name: string; parent?: T; contentReady?: boolean };
    const root: T = { name: 'root', contentReady: true }, mid: T = { name: 'mid', parent: root, contentReady: false };
    const a: T = { name: 'a', parent: mid, contentReady: true }, b: T = { name: 'b', parent: mid, contentReady: true };
    const other: T = { name: 'other', parent: root, contentReady: true };
    expect(heldAncestors([a, b], () => true)).toEqual([root]);
    expect(heldAncestors([a, b], t => t === b)).toEqual([root]);
    expect(heldAncestors([a, b], () => false)).toEqual([]);
    expect(heldAncestors([a, root], () => true)).toEqual([]);
    expect(heldAncestors([other], () => true)).toEqual([root]);
  });
});
