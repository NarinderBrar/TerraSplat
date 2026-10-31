/** Rendering tiers. Splat rendering is fill-rate bound, so integrated and software GPUs get a lighter profile. */
export type Quality = 'full' | 'reduced';

const WEAK_RENDERERS = /Intel\(R\) (UHD|HD) Graphics|Intel.*(UHD|HD) Graphics|Mali|Adreno|PowerVR|SwiftShader|llvmpipe|Microsoft Basic Render/i;

export function qualityFor(renderer: string | undefined, override?: string | null): Quality {
  if (override === 'full' || override === 'reduced') return override;
  return renderer && WEAK_RENDERERS.test(renderer) ? 'reduced' : 'full';
}

export const QUALITY_PROFILES = {
  full: { maximumScreenSpaceError: 8, fullDetailBlocks: 16, movingResolutionScale: 1, loadingGlow: true },
  reduced: { maximumScreenSpaceError: 12, fullDetailBlocks: 1, movingResolutionScale: 0.6, loadingGlow: false },
} as const;

export const FULL_DETAIL_M = 2_000;
// A block keeps full detail until FULL_DETAIL_EXIT_M, so small camera moves near the limit do not flip it back and forth.
export const FULL_DETAIL_EXIT_M = 2_500;

/** Blocks to refine completely: the nearest within FULL_DETAIL_M, or FULL_DETAIL_EXIT_M for blocks already refined. */
export function fullDetailBlocks<T>(blocks: { tileset: T; show: boolean; distance: number }[], current: Set<T>, limit: number) {
  return new Set(blocks.filter(b => b.show && b.distance < (current.has(b.tileset) ? FULL_DETAIL_EXIT_M : FULL_DETAIL_M))
    .sort((a, b) => a.distance - b.distance).slice(0, limit).map(b => b.tileset));
}
/** The unmasked WebGL renderer string, when the browser exposes it. */
export function webglRenderer(): string | undefined {
  const gl = document.createElement('canvas').getContext('webgl2');
  const info = gl?.getExtension('WEBGL_debug_renderer_info');
  const renderer = info ? gl!.getParameter(info.UNMASKED_RENDERER_WEBGL) as string : undefined;
  gl?.getExtension('WEBGL_lose_context')?.loseContext();
  return renderer;
}
