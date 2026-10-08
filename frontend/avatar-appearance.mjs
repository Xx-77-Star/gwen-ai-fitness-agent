// Original model visibility events use 30 fps. Expressions are alternate
// skinned meshes, not morph targets or an invented texture replacement.
export class AvatarAppearance {
  constructor(root) {
    this.events = {};
    this.meshes = [];
    root.traverse(object => {
      if (object.userData.submeshVisibilityEvents) this.events = object.userData.submeshVisibilityEvents;
      if (object.isMesh) this.meshes.push(object);
    });
    this.usedHashes = new Set();
    for (const events of Object.values(this.events)) {
      for (const event of events) {
        for (const hash of event.showSubmeshList ?? []) this.usedHashes.add(hash);
      }
    }
  }

  apply(clipName, time) {
    const states = new Map(this.meshes.map(mesh => [mesh.material.userData.hash, mesh.material.userData.visible !== false]));
    const frame = time * 30;
    for (const event of this.events[clipName] ?? []) {
      if (frame < (event.startFrame ?? 0) || (event.endFrame != null && frame > event.endFrame)) continue;
      for (const hash of event.hideSubmeshList ?? []) states.set(hash, false);
      for (const hash of event.showSubmeshList ?? []) states.set(hash, true);
    }
    for (const mesh of this.meshes) {
      const visible = states.get(mesh.material.userData.hash) && mesh.material.opacity > 0;
      mesh.material.visible = Boolean(visible);
      mesh.visible = Boolean(visible);
    }
  }

  includeAllUsed() {
    for (const mesh of this.meshes) {
      mesh.material.visible = mesh.material.opacity > 0
        && (mesh.material.userData.visible !== false || this.usedHashes.has(mesh.material.userData.hash));
      mesh.visible = mesh.material.visible;
    }
  }
}
