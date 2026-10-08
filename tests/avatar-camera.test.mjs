import test from "node:test";
import assert from "node:assert/strict";
import { CameraMotion, DEFAULT_ELEVATION } from "../frontend/avatar-camera.mjs";

test("default art direction is an elevated front view", () => {
  assert.ok(DEFAULT_ELEVATION >= Math.PI / 9 && DEFAULT_ELEVATION <= Math.PI * 2 / 9);
});

test("changing action goals cannot teleport the current camera", () => {
  const motion = new CameraMotion();
  motion.snap([0, 0, 0], 2.4);
  motion.moveTo([.3, .6, .2], 3.2);
  assert.deepEqual(motion.value, [0, 0, 0, 2.4]);
  motion.update(1 / 60);
  assert.ok(motion.value[1] > 0 && motion.value[1] < .02);
  const position = [...motion.value], velocity = [...motion.velocity];
  motion.moveTo([0, 0, 0], 2.4);
  assert.deepEqual(motion.value, position);
  assert.deepEqual(motion.velocity, velocity);
});

test("camera motion settles without overshoot and is independent of frame rate", () => {
  const simulate = dt => {
    const motion = new CameraMotion();
    motion.snap([0, 0, 0], 2);
    motion.moveTo([.5, .8, .3], 3);
    for (let t = 0; t < 1 - 1e-8; t += dt) {
      motion.update(dt);
      assert.ok(motion.value[1] <= .8);
    }
    return motion;
  };
  const a = simulate(1 / 30), b = simulate(1 / 120);
  a.value.forEach((value, index) => assert.ok(Math.abs(value - b.value[index]) < 1e-10));
  for (let i = 0; i < 300; i++) a.update(1 / 60);
  assert.equal(a.moving, false);
  assert.deepEqual(a.value, [.5, .8, .3, 3]);
});
