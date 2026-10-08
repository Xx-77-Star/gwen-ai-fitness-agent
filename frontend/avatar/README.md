# Gwen Hero Avatar

The homepage mounts Gwen inside `#avatar-stage-slot`. Only frontend files
participate; `/chat` request and response formats are unchanged.

## Structure

- `AvatarEngine.js`: extracted GLB renderer, material handling, visible-geometry
  sampling, camera fitting, animation, pointer/keyboard interaction and disposal.
  It queries no page-specific DOM and publishes no global variable.
- `AvatarController.js`: homepage adapter; tracks busy intent before/after
  readiness, handles local loading/failure/retry UI and reduced-motion controls.
  The engine is dynamically imported so its failure cannot stop Chat.
- `avatar-config.js`: model URL, the three enabled clips, framing occupancy and
  a 30-second initialization timeout.
- `hero-avatar.css`: isolated Hero canvas/fallback layout.
- `../avatar-viewer.js`: independent lab adapter, including its existing
  `window.avatarViewer` diagnostics and four-animation controls.

The geometry, appearance and camera helpers stay in their existing modules.
The GLB, textures and local Three.js dependencies are unchanged.

## Engine API

```javascript
import { createAvatarEngine } from "./AvatarEngine.js";
import { HERO_AVATAR_CONFIG } from "./avatar-config.js";

const engine = await createAvatarEngine({
  ...HERO_AVATAR_CONFIG,
  mount: document.querySelector("#avatar-stage-slot"),
  signal: abortController.signal,
  onReady(engine) {},
  onError(error, details) {},
  onStateChange({ state, clip, busy, paused }) {},
});
engine.setBusy(true);       // Equivalent to play("Run").
engine.setBusy(false);      // Return to Idle, after an active Interact finishes.
engine.interact();          // One-shot; no effect on the chat request.
engine.play("Cast_Cycle");  // false: unavailable with the Hero configuration.
engine.reset();
engine.dispose();
```

`createAvatarEngine` resolves to the ready engine. `onError` reports startup
failure or a recoverable GPU context loss. Restoration calls `onReady` again.
Optional `onProgress` and `onDiagnostics` callbacks support the lab.
`getDiagnostics()` returns framing, enabled clips, animation and material
measurements; `measureGeometry: false` avoids per-vertex measurements.

The homepage controller is created synchronously by `app.js`; Chat's existing
`setBusy` calls its state interface. The controller retains only the latest busy
intent, so a request that finishes before model readiness cannot revive Run.
During Interact, the animator retains busy intent and selects Idle or Run when
the one-shot ends. Cast_Cycle is not initialized or sampled on the homepage.

Initial reduced-motion preference pauses playback; Chat never unpauses it.
The user can explicitly play/pause or interact. Background/offscreen rendering
pauses while busy state continues to update.

Startup failure/timeout shows the local WebP portrait and retry button, with
Chat still usable. Retry cancels the previous attempt and disposes its canvas.
Late GLB completion is discarded and its resources released. GPU context loss
temporarily reveals the fallback; recovery synchronizes current busy intent.

## Layout and verification

Desktop Hero height is 720px with a 600px Stage. At a 1440px viewport the sampled
Idle silhouette measures approximately 402–442 CSS pixels tall. Mobile Stage
height is 400px; framing reserves room for the status labels and controls.
Explicit zoom intentionally permits close-up cropping; reset restores full framing.

Serve `frontend` over HTTP, then run from the repository root with Playwright
available (and optionally `AVATAR_BROWSER_CHANNEL=msedge`):

```powershell
$env:AVATAR_BASE_URL = "http://127.0.0.1:8083"
node frontend/avatar/tests/hero.smoke.cjs
node --test tests/avatar-fit.test.mjs tests/avatar-animation.test.mjs tests/avatar-camera.test.mjs
node tests/avatar-viewer.smoke.cjs
node tests/avatar-continuity.smoke.cjs
node tests/avatar-appearance.smoke.cjs
```

The Hero suite loads the actual GLB and uses WebGL. It intercepts `/chat` to
deterministically verify success/failure, busy transitions, Interact during work,
Trace rendering, new chat, slow GLB readiness, failed loading/retry, failed engine
import, reduced motion and 390/320px touch layouts. Screenshots and measurements
are saved under the ignored `logs/avatar-hero/` directory.
These are desktop Chromium and mobile-emulation checks, not physical-device tests.
