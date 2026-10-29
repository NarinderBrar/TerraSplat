interface IndexTile {
  content?: { uri?: string; url?: string };
  children?: IndexTile[];
}
interface RegionalIndex {
  asset: { version: string };
  root: IndexTile;
  extensionsUsed?: string[];
  extensionsRequired?: string[];
  extensions?: Record<string, unknown>;
}
/** Cesium 1.146 selects its splat decoder from the outermost tileset only.
 * Existing regional indexes omit the external releases' glTF declarations.
 * Supply the pinned delivery profile in memory, preserving all bounds/transforms.
 */
export function regionalIndexUrl(value: unknown, sourceUrl: string) {
  const index = structuredClone(value) as RegionalIndex;
  if (!index?.root || index.asset?.version !== '1.1') throw new Error('Invalid regional tileset');
  const extension = '3DTILES_content_gltf';
  const splats = ['KHR_gaussian_splatting', 'KHR_gaussian_splatting_compression_spz_2'];
  index.extensionsUsed = [...new Set([...(index.extensionsUsed ?? []), extension])];
  index.extensionsRequired = [...new Set([...(index.extensionsRequired ?? []), extension])];
  index.extensions = { ...index.extensions, [extension]: { extensionsUsed: splats, extensionsRequired: splats } };
  function resolve(tile: IndexTile) {
    if (tile.content?.uri) tile.content.uri = new URL(tile.content.uri, sourceUrl).href;
    if (tile.content?.url) tile.content.url = new URL(tile.content.url, sourceUrl).href;
    tile.children?.forEach(resolve);
  }
  resolve(index.root);
  return `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(index))}`;
}
