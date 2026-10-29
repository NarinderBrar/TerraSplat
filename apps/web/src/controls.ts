import type * as Cesium from 'cesium';
export function controlMapping(C: typeof Cesium) {
  return {
    // In 3D, globe rotation grabs the ground; translateEventTypes applies only in 2D.
    rotateEventTypes: [C.CameraEventType.LEFT_DRAG],
    translateEventTypes: [C.CameraEventType.LEFT_DRAG],
    zoomEventTypes: [C.CameraEventType.WHEEL, C.CameraEventType.PINCH],
    tiltEventTypes: [C.CameraEventType.RIGHT_DRAG, C.CameraEventType.PINCH,
      { eventType: C.CameraEventType.LEFT_DRAG, modifier: C.KeyboardEventModifier.CTRL }],
    lookEventTypes: [],
    minimumZoomDistance: 35,
    maximumZoomDistance: 30_000_000,
    enableCollisionDetection: true,
    inertiaZoom: 0.8,
  };
}
