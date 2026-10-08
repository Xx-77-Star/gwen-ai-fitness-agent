// Pure geometry: no renderer, model-specific coordinates or asset dimensions.
export const FRAME_OCCUPANCY = 0.69;

export function boxCorners(box) {
  return [box.min.x, box.max.x].flatMap(x =>
    [box.min.y, box.max.y].flatMap(y =>
      [box.min.z, box.max.z].map(z => ({ x, y, z }))));
}

export function normalizationFor(box) {
  const size = Object.fromEntries(["x", "y", "z"].map(k => [k, box.max[k] - box.min[k]]));
  const extent = Math.max(size.x, size.y, size.z);
  if (!Number.isFinite(extent) || extent <= 0) throw new Error("模型没有有效的可见尺寸。");
  const center = Object.fromEntries(["x", "y", "z"].map(k => [k, (box.min[k] + box.max[k]) / 2]));
  return { size, center, scale: 1 / extent };
}

// Solve the perspective inequalities for EVERY corner, including depth.
// x / ((distance - z) * tan(hfov / 2)) <= occupancy; likewise for y.
export function fitDistance(box, center, basis, verticalFov, aspect, occupancy = FRAME_OCCUPANCY) {
  return fitPointsDistance(boxCorners(box), center, basis, verticalFov, aspect, occupancy);
}

export function fitPointsDistance(points, center, basis, verticalFov, aspect, occupancy = FRAME_OCCUPANCY) {
  if (!(aspect > 0 && verticalFov > 0 && verticalFov < Math.PI && occupancy > 0 && occupancy < 1)) {
    throw new Error("无效的视口或取景参数。");
  }
  const tanY = Math.tan(verticalFov / 2);
  const tanX = tanY * aspect;
  const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
  let distance = 0;
  for (const corner of points) {
    const p = { x: corner.x - center.x, y: corner.y - center.y, z: corner.z - center.z };
    const z = dot(p, basis.back);
    distance = Math.max(distance,
      z + Math.abs(dot(p, basis.right)) / (tanX * occupancy),
      z + Math.abs(dot(p, basis.up)) / (tanY * occupancy));
  }
  return distance;
}

// Find both the distance AND the lateral target that center the perspective
// silhouette. An AABB center alone is biased by deep/asymmetric hair geometry.
export function fitPerspectiveFrame(points, center, basis, verticalFov, aspect, occupancy = FRAME_OCCUPANCY) {
  if (!(aspect > 0 && verticalFov > 0 && verticalFov < Math.PI && occupancy > 0 && occupancy < 1 && points.length)) {
    throw new Error("无效的投影取景参数。");
  }
  const ky = Math.tan(verticalFov / 2) * occupancy;
  const kx = ky * aspect;
  const dot = (p, axis) => (p.x - center.x) * axis.x + (p.y - center.y) * axis.y + (p.z - center.z) * axis.z;
  let lowerX = -Infinity, upperX = Infinity, lowerY = -Infinity, upperY = Infinity, front = -Infinity;
  for (const point of points) {
    const x = dot(point, basis.right), y = dot(point, basis.up), z = dot(point, basis.back);
    lowerX = Math.max(lowerX, x + kx * z);
    upperX = Math.min(upperX, x - kx * z);
    lowerY = Math.max(lowerY, y + ky * z);
    upperY = Math.min(upperY, y - ky * z);
    front = Math.max(front, z);
  }
  return {
    distance: Math.max((lowerX - upperX) / (2 * kx), (lowerY - upperY) / (2 * ky), front + Number.EPSILON),
    offsetX: (lowerX + upperX) / 2,
    offsetY: (lowerY + upperY) / 2,
  };
}
