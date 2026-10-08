import test from "node:test";
import assert from "node:assert/strict";
import { AnimationMixer, AnimationClip, NumberKeyframeTrack, Object3D } from "../frontend/vendor/three/build/three.module.js";
import { AvatarAnimator, AVATAR_CLIPS } from "../frontend/avatar-animation.mjs";

function setup() {
  const model = new Object3D();
  const mixer = new AnimationMixer(model);
  const clips = Object.values(AVATAR_CLIPS).map((name, index) =>
    new AnimationClip(name, 1, [new NumberKeyframeTrack(".position[x]", [0, 1], [index, index + 0.5])]));
  return { model, mixer, animator: new AvatarAnimator(mixer, clips) };
}
const advance = (animator, seconds) => {
  for (let t = 0; t < seconds; t += 0.01) animator.update(0.01);
};

test("only Idle_Base, Interact, Run and Cast_Cycle are mapped; idle loops by default", () => {
  assert.deepEqual(Object.values(AVATAR_CLIPS), ["Idle_Base", "Interact", "Run", "Cast_Cycle"]);
  const { animator } = setup();
  advance(animator, 2.5);
  assert.equal(animator.state, "idle");
  assert.ok(animator.currentAction.isRunning());
  animator.dispose();
});

test("Interact runs once, ignores repeated clicks and returns to Idle_Base", () => {
  const { animator } = setup();
  assert.equal(animator.interact(), true);
  advance(animator, 0.4);
  const time = animator.currentAction.time;
  assert.equal(animator.interact(), false);
  assert.equal(animator.currentAction.time, time);
  advance(animator, 0.9);
  assert.equal(animator.state, "idle");
  assert.equal(animator.actions.interact.isRunning(), false);
  animator.dispose();
});

test("busy state loops Run; an interaction returns to Run if work is still pending", () => {
  const { animator } = setup();
  animator.setBusy(true);
  advance(animator, 2.4);
  assert.equal(animator.state, "run");
  animator.interact();
  advance(animator, 1.3);
  assert.equal(animator.state, "run");
  animator.setBusy(false);
  assert.equal(animator.state, "idle");
  animator.dispose();
});

test("work finishing during Interact returns to Idle, without resurrecting stale Run", () => {
  const { animator } = setup();
  animator.setBusy(true);
  animator.interact();
  animator.setBusy(false);
  assert.equal(animator.state, "interact");
  advance(animator, 1.3);
  assert.equal(animator.state, "idle");
  assert.equal(animator.actions.run.isRunning(), false);
  animator.dispose();
});

test("Cast_Cycle is opt-in, plays once, and returns to current busy intent", () => {
  const { animator } = setup();
  assert.equal(animator.actions.cast.isRunning(), false);
  animator.setBusy(true);
  assert.equal(animator.triggerCastCycle(), true);
  assert.equal(animator.state, "cast");
  advance(animator, 1.3);
  assert.equal(animator.state, "run");
  animator.dispose();
});

test("rapid state changes clear stale actions; disposal removes finished listeners", () => {
  const { animator, mixer } = setup();
  for (let i = 0; i < 10; i++) {
    animator.setBusy(true);
    animator.update(.03);
    animator.setBusy(false);
    animator.update(.03);
  }
  advance(animator, .4);
  assert.equal(animator.actions.idle.getEffectiveWeight(), 1);
  assert.equal(animator.actions.run.isRunning(), false);
  animator.dispose();
  assert.equal(mixer.hasEventListener("finished", animator.finished), false);
});

test("return to idle overlaps the moving tail of a one-shot instead of freezing its last pose", () => {
  const { animator } = setup();
  animator.interact();
  advance(animator, .85);
  assert.equal(animator.state, "idle", "return blend should already have begun before the one-shot ends");
  assert.ok(animator.actions.interact.time < 1);
  assert.ok(animator.actions.interact.getEffectiveWeight() > 0);
  assert.ok(animator.actions.idle.getEffectiveWeight() > 0);
  animator.dispose();
});

test("rapid interrupted blends never expose the skeleton's unanimated bind pose", () => {
  const model = new Object3D();
  model.position.x = 1000;
  const clips = Object.values(AVATAR_CLIPS).map((name, index) =>
    new AnimationClip(name, 1, [new NumberKeyframeTrack(".position[x]", [0, 1], [index, index])]));
  const animator = new AvatarAnimator(new AnimationMixer(model), clips);
  for (let i = 0; i < 100; i++) {
    if (i % 3 === 0) animator.setBusy(true);
    if (i % 3 === 1) animator.setBusy(false);
    animator.update(1 / 60);
    assert.ok(model.position.x >= -1e-6 && model.position.x <= 3.000001, `bind-pose flash: x=${model.position.x}`);
  }
  animator.dispose();
});
