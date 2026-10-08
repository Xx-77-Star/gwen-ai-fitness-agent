// Art direction only. Camera coordinates and distance still come from the GLB.
export const DEFAULT_ELEVATION = Math.PI / 6;

// Exact critically damped response: changing an animation updates the goal,
// never the current camera. Position and velocity remain continuous.
export class CameraMotion {
  constructor(smoothTime = 0.24) {
    this.smoothTime = smoothTime;
    this.value = [0, 0, 0, 0];
    this.goal = [...this.value];
    this.velocity = [0, 0, 0, 0];
    this.moving = false;
  }

  snap(target, distance) {
    this.value = [...target, distance];
    this.goal = [...this.value];
    this.velocity.fill(0);
    this.moving = false;
  }

  moveTo(target, distance) {
    this.goal = [...target, distance];
    this.moving = true;
  }

  update(delta) {
    if (!this.moving || delta <= 0) return false;
    const omega = 2 / this.smoothTime;
    const decay = Math.exp(-omega * delta);
    for (let i = 0; i < this.value.length; i++) {
      const error = this.value[i] - this.goal[i];
      const change = (this.velocity[i] + omega * error) * delta;
      this.value[i] = this.goal[i] + (error + change) * decay;
      this.velocity[i] = (this.velocity[i] - omega * change) * decay;
    }
    if (this.value.every((v, i) => Math.abs(v - this.goal[i]) < 1e-6 && Math.abs(this.velocity[i]) < 1e-5)) {
      this.value = [...this.goal];
      this.velocity.fill(0);
      this.moving = false;
    }
    return true;
  }
}
