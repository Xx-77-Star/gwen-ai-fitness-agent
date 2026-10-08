# Independent Avatar Viewer

This page does not import or modify the FitLife homepage. Its UI and stylesheet
remain separate. `avatar-viewer.js` is now a lab adapter over the shared
`avatar/AvatarEngine.js`; the homepage uses `avatar/AvatarController.js`.
Rendering, materials, camera fitting and interaction use the same engine.

## Open

From the project root:

```powershell
python -m http.server 8080 --bind 127.0.0.1 --directory frontend
```

Open `http://127.0.0.1:8080/avatar-test.html`. The existing FastAPI static mount
can also serve `/avatar-test.html`. Use HTTP; browser restrictions prevent ES
modules and GLB fetches from working reliably through `file://`.

## Four animations

| Intent | GLB clip | Behavior |
| --- | --- | --- |
| Default / idle | `Idle_Base` | Loop |
| Click / tap the character or press Enter | `Interact` | Once, then return to idle or busy |
| Plan generation / loading | `Run` | Loop while busy |
| Reserved future scene action | `Cast_Cycle` | Explicit trigger only; preview once |

The actual names are `Idle_Base` (capital I) and `Cast_Cycle` (underscore).
Only these four clips remain in `assets/ai-coach.glb`; 22 clips and their unused
keyframe data were removed. Asset size: **6,775,496 → 2,662,584 bytes**.
Skeleton, textures and the four retained clips' keyframes are unchanged.
`tools/prune-avatar-animations.py` makes the reduction reproducible.

The original export omitted expression primitives and its visibility events
were initially ignored. `tools/restore-avatar-expressions.py` restores the
original `Eye_Playful`, `Mouth_Playful` and `Mouth_Smile` triangle indices from
the matching full source model, reusing the existing vertex pool and atlas.
The current repaired asset is 2,660,136 bytes (unused event metadata removed).
`avatar-appearance.mjs` now executes the original 30 fps visibility events:
Scissors are shown in all four clips; Interact frames 10–44 switch to the
original wink/open-mouth meshes; Cast_Cycle uses Mouth_Smile. Visibility resets
on returning to Idle. No additional animation clips were added.

The test toolbar can simulate each state. Dragging the model, clicking empty
space and pinching do not trigger Interact. Repeated clicks during a one-shot
are ignored. Transitions blend for 280 ms, and completed actions are stopped.
One-shots begin blending back during their moving tail, before the final pose
clamps. Idle/Run resume their saved phase. Finished callbacks defer state changes
until outside the mixer's update loop.
If loading finishes during Interact, completion returns to Idle rather than
starting a stale Run. Reduced-motion preference pauses playback initially.

Lab integration/diagnostic interface (the homepage uses its own controller):

```javascript
window.avatarViewer.setBusy(true);   // Start plan generation / loading.
window.avatarViewer.setBusy(false);  // Finish or fail that work.
window.avatarViewer.interact();      // Once; also triggered by character clicks.
window.avatarViewer.triggerCastCycle(); // Once; no automatic scene trigger.
window.avatarViewer.reset();         // Restore auto framing, preserve action.
window.avatarViewer.getDiagnostics();
window.avatarViewer.dispose();
```

The homepage enables only Idle_Base, Interact and Run. Cast_Cycle remains in
the asset and in this lab, but is neither initialized nor callable by the Hero
engine. See `avatar/README.md` for the homepage lifecycle.

The caller owns the busy boolean: if multiple operations overlap, aggregate
them before calling `setBusy`. For asynchronous operations, clear busy in
`finally`, including on errors.

## Bounds and framing

All dimensions below are GLB scene units, not an inferred real-world height.

Visible model in its original static pose:

- Size XYZ: `[1.018446733, 1.514416143, 0.782305592]`.
- Minimum: `[-0.509223280, -0.002226999, -0.508787552]`.
- Maximum: `[0.509223453, 1.512189144, 0.273518040]`.
- Center: `[0.000000087, 0.754981072, -0.117634756]`.

Idle_Base animation envelope:

- Minimum, including Scissors: `[-0.619361365, -0.032721529, -1.121998230]`.
- Maximum: `[0.921829728, 1.310918919, 0.250524513]`.
- Center: `[0.151234182, 0.639098695, -0.435736858]`.
- Uniform normalization scale: `0.648848805`, computed as
  `1 / max(envelope.width, envelope.height, envelope.depth)`.

The normalizing parent translates by `-envelope.center * scale`. The complete
skeleton is inside that parent, so animation tracks cannot overwrite the
normalization transform. Normalization remains constant across all actions.

The GLB shares POSITION data between primitives, including hidden effects.
Using the accessor bounds or `Box3.setFromObject(model, true)` counts unused
vertices and creates excessive whitespace. Instead, this viewer honors
exported material visibility and visits only indices of visible triangles.
`SkinnedMesh.getVertexPosition` includes current skinning and morph transforms.

Each retained clip is sampled at 60 Hz. The bounds cover 98 Idle poses,
112 Interact poses, 46 Run poses and 50 Cast poses. Per-pose convex hulls are
merged, then expanded by 3% for sampling tolerance. Sampling is a practical
bound for this asset, not a mathematical guarantee for arbitrary replacement
animations; replacements should be revalidated.

For a hull point in the camera's right/up/back basis `(x,y,z)`, occupancy `q`,
and half-angle tangents `kx = q*tan(horizontalFOV/2)`,
`ky = q*tan(verticalFOV/2)`:

```text
Lx = max(x + kx*z), Ux = min(x - kx*z)
Ly = max(y + ky*z), Uy = min(y - ky*z)
distance = max((Lx-Ux)/(2*kx), (Ly-Uy)/(2*ky), frontDepth)
targetOffsetX = (Lx+Ux)/2
targetOffsetY = (Ly+Uy)/2
camera = derivedTarget + currentViewDirection * distance
```

The intended envelope occupancy is 69%; the visible animated character is
slightly smaller. Camera distance and target are recomputed for resize,
rotation and action changes, preserving the user's zoom ratio. During a
crossfade the two action envelopes are unioned only for clipping-plane safety.
Camera target and distance move continuously toward the active action's frame,
using a critically damped response instead of two immediate reframing jumps.
The default direction is a 30-degree elevated front view matching the supplied
reference. Camera remains stable within each settled looping action. No model-specific
position, scale or camera coordinates are hardcoded. Mobile uses a square
stage to avoid a narrow, excessively tall display area.

Zooming in intentionally allows close-up views. Reset returns to complete
framing. Near/far planes and zoom limits are derived from the active bounds.

## Rendering

Three.js **0.180.0** and required addons are vendored with their MIT license,
so there are no runtime CDN requests. WebGL2, sRGB output, ACES tone mapping,
white key light, hemisphere light and subdued purple fill are configured.

The supplied GLB uses `KHR_materials_unlit` and baked color textures. These
materials retain their original colors and do not receive additional lighting.
The lights are ready for lit/PBR assets; they do not wash out this asset.

The exporter marked fully opaque head/hair/hat atlases as BLEND. Their alpha
channels are inspected once, and those materials use the opaque pass.
Genuinely transparent eye/mouth decals retain blending and disable depth writes.
This prevents black faces and transparent sorting artifacts.

DPR is capped at 2. Rendering pauses when hidden or offscreen. The page includes
load errors/retry, context-loss handling, resize observation, keyboard controls,
and disposal of controls, animation actions, geometries, textures and renderer.

## Validation

```powershell
node --test tests/avatar-fit.test.mjs tests/avatar-animation.test.mjs tests/avatar-camera.test.mjs

# Requires Playwright, a Chromium browser and the HTTP server above.
# Set AVATAR_BROWSER_CHANNEL=msedge to use an installed Microsoft Edge.
node tests/avatar-viewer.smoke.cjs
node tests/avatar-continuity.smoke.cjs
node tests/avatar-appearance.smoke.cjs
```

The browser suite checks projected visible vertices, stable Idle framing,
all four action transitions, click hit testing, rotation, zoom, touch/pinch,
resize, reduced motion and a failed GLB load. It writes screenshots and
measurements to `logs/avatar-viewer/`, which is ignored by Git.

Verified in Microsoft Edge (Chromium) with desktop and emulated mobile/touch:

- 16 geometry/animation/camera unit tests passed.
- 16 browser acceptance categories passed, including four-clip transitions.
- Idle visible height after the angle correction: 60.7–66.6% on desktop;
  66.6% at mobile initial pose.
- Action-return continuity regression passed: maximum sampled camera step
  reduced from 1.879 to 0.015 normalized units across the tested transitions.
- No clipping in sampled poses, 24 rotation views, or tested action transitions.
- No browser console errors; mouse, keyboard, touch rotation and pinch passed.
- Original and compacted assets were compared: retained animation keyframe
  bytes, geometry, inverse bind matrices and embedded image bytes match.

This is browser and mobile-emulation validation; physical iOS/Android devices
have not been tested. User-initiated close-up zoom can intentionally crop.
