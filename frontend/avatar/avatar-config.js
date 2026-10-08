export const HERO_AVATAR_CONFIG = Object.freeze({
  modelUrl: new URL("../assets/ai-coach.glb", import.meta.url).href,
  enabledClips: Object.freeze({
    idle: "Idle_Base",
    interact: "Interact",
    run: "Run",
  }),
  occupancy: 0.88,
  loadTimeoutMs: 30000,
});

// Compact Chat companion dock. Shares the same GLB and the same three clips
// (Idle_Base / Interact / Run) as the Hero. Framing shows the FULL character,
// centered by the engine's perspective silhouette fit, with breathing room so
// the head and feet never touch the tile edges during Run.
export const CHAT_AVATAR_CONFIG = Object.freeze({
  ...HERO_AVATAR_CONFIG,
  occupancy: 0.95,
  // Center the model's bounding box (not the asymmetric silhouette) so Gwen
  // reads as balanced in a small tile during Run, and never gets cropped.
  opticalCentering: false,
  renderPixelRatio: 2,
  loadTimeoutMs: 20000,
});