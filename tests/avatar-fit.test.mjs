import test from "node:test";
import assert from "node:assert/strict";
import { normalizationFor, fitDistance, fitPointsDistance, fitPerspectiveFrame, boxCorners, FRAME_OCCUPANCY } from "../frontend/avatar-fit.mjs";

const basis = { right: { x: 1, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 }, back: { x: 0, y: 0, z: 1 } };
const origin = { x: 0, y: 0, z: 0 };

test("normalization follows arbitrary asset scale and translation", () => {
  for (const factor of [0.0001, 1, 10_000]) {
    const box = { min: { x: 12 * factor, y: -7 * factor, z: 4 * factor }, max: { x: 14 * factor, y: -1 * factor, z: 8 * factor } };
    const result = normalizationFor(box);
    assert.ok(Math.abs(result.scale * factor - 1 / 6) < 1e-10);
    for (const k of ["x", "y", "z"]) {
      assert.ok(Math.abs((box.min[k] + box.max[k] - 2 * result.center[k]) * result.scale) < 1e-10);
    }
  }
});

test("perspective fit respects portrait, landscape and deep geometry", () => {
  const box = { min: { x: -0.5, y: -0.7, z: -1.8 }, max: { x: 0.5, y: 0.7, z: 1.8 } };
  for (const aspect of [0.35, 0.7, 1, 1.5, 3.5]) {
    const fov = 35 * Math.PI / 180;
    const distance = fitDistance(box, origin, basis, fov, aspect);
    let maxExtent = 0;
    for (const p of boxCorners(box)) {
      const y = Math.abs(p.y / ((distance - p.z) * Math.tan(fov / 2)));
      const x = Math.abs(p.x / ((distance - p.z) * Math.tan(fov / 2) * aspect));
      assert.ok(x <= FRAME_OCCUPANCY + 1e-12);
      assert.ok(y <= FRAME_OCCUPANCY + 1e-12);
      maxExtent = Math.max(maxExtent, x, y);
    }
    assert.ok(Math.abs(maxExtent - FRAME_OCCUPANCY) < 1e-12);
  }
});

test("point-based fit avoids empty AABB corners and preserves all projected points", () => {
  const points = [{ x: 0, y: 0, z: 2 }, { x: 0, y: 1, z: 0 }, { x: 0, y: -1, z: 0 }, { x: 1, y: 0, z: 0 }];
  const fov = Math.PI / 4;
  const distance = fitPointsDistance(points, origin, basis, fov, 1);
  const boxFit = fitDistance({ min: { x: 0, y: -1, z: 0 }, max: { x: 1, y: 1, z: 2 } }, origin, basis, fov, 1);
  assert.ok(distance < boxFit);
  for (const p of points) {
    assert.ok(Math.max(Math.abs(p.x), Math.abs(p.y)) / ((distance - p.z) * Math.tan(fov / 2)) <= FRAME_OCCUPANCY + 1e-12);
  }
});

test("invalid model bounds and viewport fail clearly", () => {
  assert.throws(() => normalizationFor({ min: origin, max: origin }));
  assert.throws(() => fitDistance({ min: origin, max: origin }, origin, basis, Math.PI / 4, 0));
});

test("asymmetric deep silhouettes are centered and fill narrow viewports", () => {
  const points = [
    { x: -1, y: -1, z: 0.2 }, { x: .7, y: 1, z: 0.3 },
    { x: 2, y: .5, z: -2 }, { x: -.6, y: 1.2, z: -.4 },
  ];
  const fov = 35 * Math.PI / 180;
  for (const aspect of [.4, .7, 1, 2.4]) {
    const fit = fitPerspectiveFrame(points, origin, basis, fov, aspect);
    const projected = points.map(p => ({
      x: (p.x - fit.offsetX) / ((fit.distance - p.z) * Math.tan(fov / 2) * aspect),
      y: (p.y - fit.offsetY) / ((fit.distance - p.z) * Math.tan(fov / 2)),
    }));
    for (const p of projected) {
      assert.ok(Math.abs(p.x) <= FRAME_OCCUPANCY + 1e-10);
      assert.ok(Math.abs(p.y) <= FRAME_OCCUPANCY + 1e-10);
    }
    const width = (Math.max(...projected.map(p => p.x)) - Math.min(...projected.map(p => p.x))) / 2;
    const height = (Math.max(...projected.map(p => p.y)) - Math.min(...projected.map(p => p.y))) / 2;
    assert.ok(Math.abs(Math.max(width, height) - FRAME_OCCUPANCY) < 1e-10);
  }
});
