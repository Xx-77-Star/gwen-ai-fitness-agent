import { LoopOnce, LoopRepeat } from "./vendor/three/build/three.module.js";

export const AVATAR_CLIPS = Object.freeze({
  idle: "Idle_Base",
  interact: "Interact",
  run: "Run",
  cast: "Cast_Cycle",
});

// Product intent is independent of the active one-shot. Finishing Interact
// returns to Run while busy, or Idle when the pending work has completed.
export class AvatarAnimator {
  constructor(mixer, clips, { enabledClips = AVATAR_CLIPS, onChange = () => {}, onSettled = () => {} } = {}) {
    this.mixer = mixer;
    this.onChange = onChange;
    this.onSettled = onSettled;
    this.busy = false;
    this.state = null;
    this.actions = {};
    this.fadeDuration = 0.28;
    this.fadeRemaining = 0;
    this.pendingReturn = false;
    this.loopTimes = {};
    for (const [state, name] of Object.entries(enabledClips)) {
      const clip = clips.find(candidate => candidate.name === name);
      if (!clip) throw new Error(`模型缺少所需动作：${name}`);
      const action = mixer.clipAction(clip);
      const once = state === "interact" || state === "cast";
      action.setLoop(once ? LoopOnce : LoopRepeat, once ? 1 : Infinity);
      action.clampWhenFinished = once;
      this.actions[state] = action;
    }
    this.finished = event => {
      if (event.action === this.currentAction && this.isOneShot) {
        // Do not mutate the mixer's active actions inside its update loop.
        this.pendingReturn = true;
      }
    };
    mixer.addEventListener("finished", this.finished);
    this.transition("idle");
  }

  get isOneShot() { return this.state === "interact" || this.state === "cast"; }
  get currentAction() { return this.actions[this.state]; }
  get currentClip() { return this.currentAction.getClip(); }

  transition(state) {
    if (state === this.state || !this.actions[state]) return false;
    const previousState = this.state;
    const previous = this.currentAction;
    const next = this.actions[state];
    // Drop older fading actions to prevent accumulated weights on rapid input.
    for (const action of Object.values(this.actions)) {
      if (action !== previous) {
        const loopState = Object.keys(this.actions).find(key => this.actions[key] === action);
        if ((loopState === "idle" || loopState === "run") && action.isScheduled()) {
          this.loopTimes[loopState] = action.time;
        }
        action.stop();
      }
    }
    next.reset().setEffectiveTimeScale(1).setEffectiveWeight(1).play();
    if (state === "idle" || state === "run") next.time = this.loopTimes[state] ?? 0;
    if (previous) next.crossFadeFrom(previous, this.fadeDuration, false);
    this.state = state;
    this.fadeRemaining = previous ? this.fadeDuration : 0;
    this.pendingReturn = false;
    this.onChange(state, previousState);
    return true;
  }

  setBusy(busy) {
    this.busy = Boolean(busy);
    if (!this.isOneShot) this.transition(this.busy ? "run" : "idle");
  }

  interact() {
    return this.isOneShot ? false : this.transition("interact");
  }

  triggerCastCycle() {
    return this.isOneShot ? false : this.transition("cast");
  }

  idle() {
    this.busy = false;
    this.transition("idle");
  }

  update(delta) {
    // Blend back while the outgoing tail is STILL MOVING, before LoopOnce
    // clamps at its last pose. A hitch finishing a clip uses the deferred path.
    if (this.pendingReturn || (this.isOneShot
      && this.currentClip.duration - this.currentAction.time <= this.fadeDuration + delta)) {
      this.transition(this.busy ? "run" : "idle");
    }
    this.mixer.update(delta);
    if (this.fadeRemaining > 0) {
      this.fadeRemaining = Math.max(0, this.fadeRemaining - delta);
      if (this.fadeRemaining === 0) {
        for (const [state, action] of Object.entries(this.actions)) {
          if (action !== this.currentAction) {
            if ((state === "idle" || state === "run") && action.isScheduled()) this.loopTimes[state] = action.time;
            action.stop();
          }
        }
        this.onSettled(this.state);
      }
    }
  }

  dispose() {
    this.mixer.removeEventListener("finished", this.finished);
    this.mixer.stopAllAction();
  }
}
