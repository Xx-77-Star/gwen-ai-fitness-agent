// Requires Playwright and a running static server. No application API is used.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const BASE_URL = process.env.AVATAR_BASE_URL || "http://127.0.0.1:8080";
const OUTPUT = process.env.AVATAR_TEST_OUTPUT || path.join("logs", "avatar-viewer");
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const diagnostics = page => page.evaluate(() => window.avatarViewer.getDiagnostics());
const close = (a, b, epsilon = 1e-6) => Math.abs(a - b) <= epsilon;
function fullyVisible(d) {
  for (let axis = 0; axis < 3; axis++) {
    assert.ok(d.screenBounds.min[axis] > -1, `clipped min axis ${axis}: ${d.screenBounds.min}`);
    assert.ok(d.screenBounds.max[axis] < 1, `clipped max axis ${axis}: ${d.screenBounds.max}`);
  }
}

(async () => {
  fs.mkdirSync(OUTPUT, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.AVATAR_BROWSER_CHANNEL ? { channel: process.env.AVATAR_BROWSER_CHANNEL } : {}),
    args: ["--enable-unsafe-swiftshader"],
  });
  const errors = [];
  const results = {};
  const watch = page => {
    page.on("pageerror", error => errors.push(error.message));
    page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  };
  async function open(context) {
    const page = await context.newPage();
    watch(page);
    await page.goto(`${BASE_URL}/avatar-test.html`);
    await page.waitForFunction(() => window.avatarViewer?.getDiagnostics().ready, { timeout: 60000 });
    return page;
  }
  try {
    const desktop = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
    const page = await open(desktop);
    let initial = await diagnostics(page);
    assert.equal(initial.animation, "Idle_Base");
    assert.deepEqual(initial.clips, ["Idle_Base", "Interact", "Run", "Cast_Cycle"]);
    assert.equal(initial.visibleMeshes.length, 7);
    assert.ok(initial.visibleMeshes.includes("Scissors"));
    assert.equal(initial.materials.find(m => m.name === "Head").transparent, false);
    assert.equal(initial.materials.find(m => m.name === "Eye_Base").transparent, true);
    const heights = [], widths = [];
    const initialTime = initial.animationTime;
    for (let i = 0; i < 20; i++) {
      const d = await diagnostics(page);
      fullyVisible(d);
      heights.push(d.screenBounds.height);
      widths.push(d.screenBounds.width);
      assert.ok(close(d.distance, initial.distance), "camera must not pump with Idle");
      await sleep(90);
    }
    let animated = await diagnostics(page);
    assert.notEqual(animated.animationTime, initialTime);
    assert.ok(Math.min(...heights) >= 0.60 && Math.max(...heights) <= 0.70, `desktop height: ${heights}`);
    results.desktop = { ...initial, sampledHeight: [Math.min(...heights), Math.max(...heights)] };
    await page.screenshot({ path: path.join(OUTPUT, "desktop.png"), fullPage: true });

    await page.locator("#pause").click();
    const paused = await diagnostics(page);
    await sleep(180);
    assert.equal((await diagnostics(page)).animationTime, paused.animationTime);
    await page.locator("#avatar-canvas").scrollIntoViewIfNeeded();
    const stage = await page.locator("#avatar-canvas").boundingBox();
    await page.mouse.move(stage.x + stage.width / 2, stage.y + stage.height / 2);
    await page.mouse.down();
    await page.mouse.move(stage.x + stage.width * 0.8, stage.y + stage.height * 0.55, { steps: 12 });
    await page.mouse.up();
    await sleep(500);
    let rotated = await diagnostics(page);
    assert.ok(Math.abs(rotated.camera[0] - initial.camera[0]) > 0.1, "mouse rotation failed");
    fullyVisible(rotated);
    await page.mouse.wheel(0, -250);
    await sleep(500);
    const zoomed = await diagnostics(page);
    assert.ok(zoomed.distance < rotated.distance, "wheel zoom failed");
    await page.locator("#reset").click();
    await sleep(150);
    assert.ok(close((await diagnostics(page)).distance, initial.distance), "reset failed");
    for (let i = 0; i < 24; i++) {
      await page.locator("#rotate-right").click();
      fullyVisible(await diagnostics(page));
    }
    await page.locator("#reset").click();
    await page.locator("#avatar-canvas").focus();
    await page.keyboard.press("ArrowUp");
    assert.ok((await diagnostics(page)).camera[1] > 0.1, "keyboard rotation failed");
    await page.keyboard.press("r");

    // A drag and an empty-stage click must not be treated as a model click.
    await page.locator("#avatar-canvas").click({ position: { x: 10, y: 10 } });
    assert.equal((await diagnostics(page)).animationState, "idle");
    const clickArea = await page.locator("#avatar-canvas").boundingBox();
    await page.locator("#avatar-canvas").click({ position: { x: clickArea.width / 2, y: clickArea.height / 2 } });
    assert.equal((await diagnostics(page)).animationState, "interact", "model click failed");
    assert.equal(await page.evaluate(() => window.avatarViewer.interact()), false, "repeated interaction restarted");
    for (let i = 0; i < 22; i++) {
      fullyVisible(await diagnostics(page));
      await sleep(100);
    }
    await page.waitForFunction(() => window.avatarViewer.getDiagnostics().animationState === "idle");
    await page.locator("#action-busy").click();
    assert.equal((await diagnostics(page)).animationState, "run");
    await page.locator("#avatar-canvas").scrollIntoViewIfNeeded();
    for (let i = 0; i < 12; i++) { fullyVisible(await diagnostics(page)); await sleep(100); }
    await page.screenshot({ path: path.join(OUTPUT, "run.png"), fullPage: true });
    await page.evaluate(() => window.avatarViewer.interact());
    await page.waitForFunction(() => window.avatarViewer.getDiagnostics().animationState === "run");
    await page.evaluate(() => window.avatarViewer.setBusy(false));
    await page.waitForFunction(() => window.avatarViewer.getDiagnostics().animationState === "idle");
    await page.evaluate(() => window.avatarViewer.triggerCastCycle());
    assert.equal((await diagnostics(page)).animationState, "cast");
    for (let i = 0; i < 12; i++) {
      fullyVisible(await diagnostics(page));
      if (i === 3) await page.screenshot({ path: path.join(OUTPUT, "cast.png"), fullPage: true });
      await sleep(70);
    }
    await page.waitForFunction(() => window.avatarViewer.getDiagnostics().animationState === "idle");
    await page.locator("#pause").click();
    await page.locator("#reset").click();
    results.animations = ["model click → Interact → Idle_Base", "busy → Run", "busy + Interact → Run", "end busy → Idle_Base", "manual Cast_Cycle → Idle_Base"];

    const sizes = [{ width: 1024, height: 768 }, { width: 768, height: 1024 }, { width: 390, height: 844 }, { width: 844, height: 390 }];
    results.resize = [];
    for (const viewport of sizes) {
      await page.setViewportSize(viewport);
      await sleep(150);
      const d = await diagnostics(page);
      fullyVisible(d);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "horizontal overflow");
      results.resize.push({ viewport, canvas: d.viewport, screen: d.screenBounds });
    }
    await desktop.close();

    const mobile = await browser.newContext({
      viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true,
    });
    const phone = await open(mobile);
    const beforeTouch = await diagnostics(phone);
    assert.equal(beforeTouch.dpr, 2, "DPR cap failed");
    fullyVisible(beforeTouch);
    const mobileOccupancy = Math.max(beforeTouch.screenBounds.width, beforeTouch.screenBounds.height);
    assert.ok(mobileOccupancy >= .60 && mobileOccupancy <= .70, `mobile occupancy: ${mobileOccupancy}`);
    await phone.screenshot({ path: path.join(OUTPUT, "mobile.png"), fullPage: true });
    const cdp = await mobile.newCDPSession(phone);
    const area = await phone.locator("#avatar-canvas").boundingBox();
    const cx = area.x + area.width / 2, cy = area.y + area.height / 2;
    const touch = (id, x, y) => ({ id, x, y, radiusX: 4, radiusY: 4, force: 1 });
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [touch(1, cx, cy)] });
    for (let i = 1; i <= 8; i++) {
      await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [touch(1, cx + i * 10, cy)] });
    }
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await sleep(350);
    const afterSwipe = await diagnostics(phone);
    assert.ok(Math.abs(afterSwipe.camera[0] - beforeTouch.camera[0]) > .1, "single touch rotation failed");
    fullyVisible(afterSwipe);
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [touch(1, cx - 30, cy), touch(2, cx + 30, cy)] });
    for (let i = 1; i <= 8; i++) {
      await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [touch(1, cx - 30 - i * 5, cy), touch(2, cx + 30 + i * 5, cy)] });
    }
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await sleep(300);
    const afterPinch = await diagnostics(phone);
    assert.ok(afterPinch.distance < afterSwipe.distance, "two-finger pinch failed");
    results.mobile = { before: beforeTouch, swipeCamera: afterSwipe.camera, pinchDistance: afterPinch.distance };
    await mobile.close();

    const reduced = await browser.newContext({ reducedMotion: "reduce" });
    const reducedPage = await open(reduced);
    assert.equal((await diagnostics(reducedPage)).paused, true, "reduced motion not honored");
    await reduced.close();
    assert.deepEqual(errors, []);

    // A failed asset request must produce a visible, recoverable error.
    const failureContext = await browser.newContext();
    const failure = await failureContext.newPage();
    await failure.route("**/assets/ai-coach.glb", route => route.fulfill({ status: 404, body: "missing" }));
    await failure.goto(`${BASE_URL}/avatar-test.html`);
    await failure.locator("#retry").waitFor({ state: "visible" });
    assert.match(await failure.locator("#loading-detail").textContent(), /无法加载 Avatar/);
    await failureContext.close();
    results.passed = ["four-clip GLB", "animation + stable camera", "material alpha correction", "model click + one-shot return", "busy Run + interaction return", "manual Cast_Cycle", "mouse drag", "wheel zoom", "24 rotation views", "reset", "keyboard", "responsive resize", "single touch", "pinch", "reduced motion", "404 error state"];
    results.consoleErrors = errors;
    fs.writeFileSync(path.join(OUTPUT, "acceptance.json"), JSON.stringify(results, null, 2));
    console.log(JSON.stringify({ passed: results.passed, desktopHeight: results.desktop.sampledHeight, mobileScreen: results.mobile.before.screenBounds, output: OUTPUT }, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
