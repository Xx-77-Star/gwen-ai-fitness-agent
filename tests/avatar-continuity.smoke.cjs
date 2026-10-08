// Regression for the reference viewing angle and action-to-idle camera jumps.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

(async () => {
  const output = path.join("logs", "avatar-continuity");
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.AVATAR_BROWSER_CHANNEL ? { channel: process.env.AVATAR_BROWSER_CHANNEL } : {}),
    args: ["--enable-unsafe-swiftshader"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    await page.goto(`${process.env.AVATAR_BASE_URL || "http://127.0.0.1:8080"}/avatar-test.html`);
    await page.waitForFunction(() => window.avatarViewer?.getDiagnostics().ready, { timeout: 60000 });
    const baseline = await page.evaluate(() => window.avatarViewer.getDiagnostics());
    const offset = baseline.camera.map((n, i) => n - baseline.target[i]);
    const elevation = Math.atan2(offset[1], Math.hypot(offset[0], offset[2])) * 180 / Math.PI;
    const results = { elevation, transitions: {} };
    for (const action of ["interact", "cast", "run"]) {
      const trace = await page.evaluate(async action => {
        const viewer = window.avatarViewer;
        if (action === "run") {
          viewer.setBusy(true);
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        const frames = [];
        const read = () => {
          const d = viewer.getDiagnostics({ measureGeometry: false });
          frames.push({ timestamp: performance.now(), camera: d.camera, target: d.target, state: d.animationState, animationTime: d.animationTime });
        };
        read();
        if (action === "interact") viewer.interact();
        if (action === "cast") viewer.triggerCastCycle();
        if (action === "run") viewer.setBusy(false);
        const start = performance.now();
        while (performance.now() - start < (action === "interact" ? 3000 : 1800)) {
          await new Promise(requestAnimationFrame);
          read();
        }
        return frames;
      }, action);
      let maxCameraStep = 0, maxTargetStep = 0;
      for (let i = 1; i < trace.length; i++) {
        maxCameraStep = Math.max(maxCameraStep, Math.hypot(...trace[i].camera.map((n, k) => n - trace[i - 1].camera[k])));
        maxTargetStep = Math.max(maxTargetStep, Math.hypot(...trace[i].target.map((n, k) => n - trace[i - 1].target[k])));
      }
      results.transitions[action] = { maxCameraStep, maxTargetStep, finalState: trace.at(-1).state, trace };
    }
    await page.screenshot({ path: path.join(output, "default-view.png"), fullPage: true });
    fs.writeFileSync(path.join(output, "continuity.json"), JSON.stringify(results, null, 2));
    console.log(JSON.stringify({ elevation, transitions: Object.fromEntries(Object.entries(results.transitions).map(([key, { trace, ...stats }]) => [key, stats])) }, null, 2));
    assert.ok(elevation >= 20 && elevation <= 40, `reference requires elevated view, got ${elevation.toFixed(2)}°`);
    for (const [name, result] of Object.entries(results.transitions)) {
      assert.equal(result.finalState, "idle", name);
      assert.ok(result.maxCameraStep < .04, `${name} camera teleports by ${result.maxCameraStep}`);
      assert.ok(result.maxTargetStep < .025, `${name} target teleports by ${result.maxTargetStep}`);
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
