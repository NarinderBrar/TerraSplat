// glb packing for the Cesium SPZ-v2 splat profile proven by the Phase 0 fixture.
export const C0 = 0.28209479177387814;
export const baseExtension = 'KHR_gaussian_splatting';
export const compressionExtension = 'KHR_gaussian_splatting_compression_spz_2';

export function splatGltf({ count, byteLength, min, max, generator }) {
  const types = [
    ['POSITION', 'VEC3', 5126], ['COLOR_0', 'VEC4', 5121],
    [`${baseExtension}:SCALE`, 'VEC3', 5126], [`${baseExtension}:ROTATION`, 'VEC4', 5126],
  ];
  return {
    asset: { version: '2.0', generator },
    extensionsUsed: [baseExtension, compressionExtension], extensionsRequired: [baseExtension, compressionExtension],
    scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    buffers: [{ byteLength }], bufferViews: [{ buffer: 0, byteOffset: 0, byteLength }],
    accessors: types.map(([name, type, componentType]) => ({
      componentType, count, type,
      ...(name === 'POSITION' ? { min, max } : {}),
      ...(name === 'COLOR_0' ? { normalized: true } : {}),
    })),
    meshes: [{ primitives: [{ mode: 0, attributes: Object.fromEntries(types.map(([name], i) => [name, i])),
      extensions: { [baseExtension]: { kernel: 'ellipse', colorSpace: 'srgb_rec709_display',
        projection: 'perspective', sortingMethod: 'cameraDistance',
        extensions: { [compressionExtension]: { bufferView: 0 } } } },
    }] }],
  };
}

export function packGlb(json, bin) {
  const text = Buffer.from(JSON.stringify(json));
  const jsonSize = Math.ceil(text.length / 4) * 4;
  const binSize = Math.ceil(bin.length / 4) * 4;
  const result = Buffer.alloc(28 + jsonSize + binSize);
  result.writeUInt32LE(0x46546c67, 0); result.writeUInt32LE(2, 4); result.writeUInt32LE(result.length, 8);
  result.writeUInt32LE(jsonSize, 12); result.writeUInt32LE(0x4e4f534a, 16);
  result.fill(0x20, 20, 20 + jsonSize); text.copy(result, 20);
  result.writeUInt32LE(binSize, 20 + jsonSize); result.writeUInt32LE(0x004e4942, 24 + jsonSize);
  result.set(bin, 28 + jsonSize);
  return result;
}

export const tilesetExtensions = {
  extensionsUsed: ['3DTILES_content_gltf'], extensionsRequired: ['3DTILES_content_gltf'],
  extensions: { '3DTILES_content_gltf': {
    extensionsUsed: [baseExtension, compressionExtension], extensionsRequired: [baseExtension, compressionExtension],
  } },
};
