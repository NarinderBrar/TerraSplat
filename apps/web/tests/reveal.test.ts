import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { appearance, injectRevealShader, Mode, revealProgress, revealTexels } from '../src/reveal';

const shader = readFileSync(new URL('../../../node_modules/@cesium/engine/Source/Shaders/PrimitiveGaussianSplatVS.glsl', import.meta.url), 'utf8');

describe('splat reveal effects', () => {
  it('injects into the pinned Cesium splat shader and refuses unknown shaders', () => {
    const injected = injectRevealShader(shader)!;
    expect(injected).toContain('uniform highp sampler2D u_revealTexture;');
    expect(injected).toContain('vec3 revealData = revealState(texIdx, splatPosition.xyz);');
    expect(injected).toContain('#define PLACEHOLDER_SECONDS 5.0');
    expect(injected).toContain('if (!revealing && dot(covVectors.xy');
    // The effect sits between the covariance and the end of main, leaving Cesium's code otherwise intact.
    expect(injected.indexOf('revealState(texIdx')).toBeGreaterThan(injected.indexOf('calcCovVectors(splatViewPos.xyz, Vrk)'));
    expect(injected.indexOf('if (revealing) {\n        // Points')).toBeLessThan(injected.indexOf('v_splitDirection = u_splitDirection;'));
    expect(injectRevealShader(shader.replace('void main() {', 'void main(){'))).toBeUndefined();
  });

  it('lays out per-splat state in merge order and truncates like the splat budget', () => {
    const { data, width, height } = revealTexels([
      { count: 3, arrival: 1.5, mode: Mode.None, placeholderSince: 4, },
      { count: 4, arrival: 2.5, mode: Mode.Grow, placeholderSince: -1 }], 5, 4);
    expect([width, height]).toEqual([4, 2]);
    expect(Array.from(data.subarray(0, 4))).toEqual([1.5, 0, 4, 0]);
    expect(Array.from(data.subarray(12, 20))).toEqual([2.5, 2, -1, 0, 2.5, 2, -1, 0]);
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
});
