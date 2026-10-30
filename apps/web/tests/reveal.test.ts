import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { injectRevealShader, revealProgress, revealTexels } from '../src/reveal';

const shader = readFileSync(new URL('../../../node_modules/@cesium/engine/Source/Shaders/PrimitiveGaussianSplatVS.glsl', import.meta.url), 'utf8');

describe('splat reveal effects', () => {
  it('injects into the pinned Cesium splat shader and refuses unknown shaders', () => {
    const injected = injectRevealShader(shader)!;
    expect(injected).toContain('uniform highp sampler2D u_revealTexture;');
    expect(injected).toContain('float reveal = revealProgress(texIdx, splatPosition.xyz);');
    expect(injected).toContain('if (!revealing && dot(covVectors.xy');
    // The effect sits between the covariance and the end of main, leaving Cesium's code otherwise intact.
    expect(injected.indexOf('revealProgress(texIdx')).toBeGreaterThan(injected.indexOf('calcCovVectors(splatViewPos.xyz, Vrk)'));
    expect(injected.indexOf('if (revealing) {\n        // Points')).toBeLessThan(injected.indexOf('v_splitDirection = u_splitDirection;'));
    expect(injectRevealShader(shader.replace('void main() {', 'void main(){'))).toBeUndefined();
  });

  it('lays out per-splat arrival and leaf flags in merge order and truncates like the splat budget', () => {
    const { data, width, height } = revealTexels(
      [{ count: 3, arrival: 1.5, animate: false }, { count: 4, arrival: 2.5, animate: true }], 5, 4);
    expect([width, height]).toEqual([4, 2]);
    expect(Array.from(data.subarray(0, 10))).toEqual([1.5, 0, 1.5, 0, 1.5, 0, 2.5, 1, 2.5, 1]);
    expect(Array.from(data.subarray(10))).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it('progress waits for the sweep, then reaches exactly 1', () => {
    expect(revealProgress(2, 1, 0.5, 1.2)).toBeCloseTo(0.5 / 1.2);
    expect(revealProgress(1.2, 1, 0.5, 1.2)).toBe(0);
    expect(revealProgress(9, 1, 0.5, 1.2)).toBe(1);
  });
});
