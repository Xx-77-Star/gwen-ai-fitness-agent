// Uses real GLB/WebGL rendering and intercepted /chat responses; no backend writes.
// Run against the frontend HTTP server with Playwright available in NODE_PATH.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const base = process.env.AVATAR_BASE_URL || "http://127.0.0.1:8083";
const output = process.env.AVATAR_HERO_OUTPUT || path.join("logs", "avatar-hero");
const slot = "#avatar-stage-slot";
const state = (page, name) => page.waitForFunction(
  name => document.querySelector("#avatar-stage-slot").dataset.avatarState === name, name);
const ready = page => page.waitForSelector(`${slot}[data-avatar-ready="true"]`, { timeout: 60000 });
const diagnostics = page => page.evaluate(async () =>
  (await import("/app.js")).avatar.engine.getDiagnostics());
const top = page => page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
const gate = () => {
  let release;
  const promise = new Promise(resolve => { release = resolve; });
  return { promise, release };
};
const response = {
  conversation_id: "avatar-browser-test",
  response: "今天先热身，再完成适合你的训练。",
  trace: {
    run: { run_id: "hero-run", status: "success", duration_ms: 15, stop_reason: "completed" },
    timeline: [{ sequence: 1, node_name: "response", status: "success", duration_ms: 15 }],
    evidence: { memory: { facts: [], turns: [] }, rag: { chunks: [] }, tools: { exchanges: [] }, sources: [] },
  },
};
async function holdChat(page, failure = false) {
  const received = gate(), finish = gate();
  await page.route("**/chat", async route => {
    received.release(route.request().postDataJSON());
    await finish.promise;
    await route.fulfill({
      status: failure ? 500 : 200, contentType: "application/json",
      body: JSON.stringify(failure ? { detail: "模拟服务暂时不可用" } : response),
    });
  });
  return { received: received.promise, finish: finish.release };
}
async function submit(page) {
  await page.locator("#message-input").fill("帮我制定今天的训练计划");
  await page.locator("#send-button").click();
  await top(page);
}
function full(d) {
  for (let i = 0; i < 3; i++) {
    assert.ok(d.screenBounds.min[i] > -1, `clipped min ${d.screenBounds.min}`);
    assert.ok(d.screenBounds.max[i] < 1, `clipped max ${d.screenBounds.max}`);
  }
}
async function clickGwen(page) {
  const box = await page.locator(`${slot} canvas`).boundingBox();
  await page.mouse.click(box.x + box.width * .5, box.y + box.height * .43);
  await state(page, "Interact");
}
async function clearLabels(page) {
  const d = await diagnostics(page);
  full(d);
  const canvas = await page.locator(`${slot} canvas`).boundingBox();
  const labels = await page.locator(".stage-status").boundingBox();
  const foot = canvas.y + (1 - d.screenBounds.min[1]) * canvas.height / 2;
  assert.ok(foot < labels.y, `Gwen overlaps status labels: ${foot} >= ${labels.y}`);
}

(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({
    channel: process.env.AVATAR_BROWSER_CHANNEL || "msedge",
    headless: true, args: ["--enable-unsafe-swiftshader"],
  });
  const errors = [], results = {};
  async function pageFor(options = {}) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, ...options });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(error.message));
    return { context, page };
  }
  try {
    const { context, page } = await pageFor();
    await page.goto(base);
    await ready(page);
    await state(page, "Idle_Base");
    assert.equal(await page.locator(`${slot} canvas`).count(), 1);
    const d = await diagnostics(page);
    assert.deepEqual(d.enabledClips, ["Idle_Base", "Interact", "Run"]);
    assert.deepEqual(Object.keys(d.animationProfiles), ["idle", "interact", "run"]);
    assert.equal(await page.evaluate(async () =>
      (await import("/app.js")).avatar.engine.play("Cast_Cycle")), false);
    const stage = await page.locator(".companion-stage").boundingBox();
    const copy = await page.locator(".hero-copy").boundingBox();
    const hero = await page.locator(".hero").boundingBox();
    assert.ok(stage.x >= copy.x + copy.width - 1, "Gwen must be right of hero copy");
    assert.equal(stage.height, 600);
    assert.equal(hero.height, 720);
    const heights = [];
    for (let i = 0; i < 24; i++) {
      const sample = await diagnostics(page);
      full(sample);
      heights.push(sample.screenBounds.height * sample.viewport[1]);
      await page.waitForTimeout(90);
    }
    assert.notEqual((await diagnostics(page)).animationTime, d.animationTime);
    assert.ok(Math.min(...heights) >= 400 && Math.max(...heights) <= 470, `Gwen height ${Math.min(...heights)}–${Math.max(...heights)}`);
    results.desktop = { heroHeight: hero.height, stageHeight: stage.height, gwenHeight: [Math.min(...heights), Math.max(...heights)] };
    await page.screenshot({ path: path.join(output, "desktop.png"), fullPage: true });
    await clickGwen(page);
    for (let i = 0; i < 22; i++) { full(await diagnostics(page)); await page.waitForTimeout(100); }
    await state(page, "Idle_Base");

    const chat = await holdChat(page);
    await submit(page);
    const request = await chat.received;
    assert.equal(request.message, "帮我制定今天的训练计划");
    await state(page, "Run");
    const run = await diagnostics(page);
    await page.waitForTimeout(160);
    assert.notEqual((await diagnostics(page)).animationTime, run.animationTime);
    await clickGwen(page);
    await state(page, "Run");
    chat.finish();
    await page.waitForSelector(".message.assistant");
    await top(page);
    await state(page, "Idle_Base");
    assert.equal(await page.locator("#send-button").isEnabled(), true);
    await page.locator("#open-trace").click();
    await page.waitForSelector(".trace-drawer.is-open");
    assert.ok((await page.locator("#run-summary").textContent()).includes("hero-run"));
    assert.equal(await page.locator(".timeline-item").count(), 1);
    await page.keyboard.press("Escape");
    await page.waitForFunction(() => !document.querySelector("#trace-drawer").classList.contains("is-open"));
    await page.unroute("**/chat");
    const failed = await holdChat(page, true);
    await submit(page);
    await failed.received;
    await state(page, "Run");
    await clickGwen(page);
    failed.finish();
    await page.waitForSelector(".message.error");
    await top(page);
    await state(page, "Idle_Base");
    await page.locator("#new-chat").click();
    assert.equal(await page.locator(".message").count(), 0);
    assert.equal(await page.locator("#chat-empty").isVisible(), true);
    results.chat = ["success", "error during Interact", "Run loop", "Interact → Run while busy", "Trace drawer", "new chat"];
    await context.close();

    for (const finishEarly of [false, true]) {
      const { context, page } = await pageFor();
      const model = gate();
      await page.route("**/ai-coach.glb", async route => { await model.promise; await route.continue(); });
      const chat = await holdChat(page);
      await page.goto(base);
      await submit(page);
      await chat.received;
      assert.equal(await page.locator(slot).getAttribute("data-avatar-busy"), "true");
      assert.equal(await page.locator("[data-avatar-fallback]").isVisible(), true);
      if (finishEarly) { chat.finish(); await page.waitForSelector(".message.assistant"); await top(page); }
      model.release();
      await ready(page);
      await state(page, finishEarly ? "Idle_Base" : "Run");
      if (!finishEarly) {
        chat.finish();
        await page.waitForSelector(".message.assistant");
        await top(page);
        await state(page, "Idle_Base");
      }
      await context.close();
    }
    results.slowLoad = ["request still pending at ready → Run", "request completed before ready → Idle"];

    {
      const { context, page } = await pageFor();
      await page.route("**/ai-coach.glb", route => route.fulfill({ status: 404, body: "missing" }));
      const chat = await holdChat(page);
      await page.goto(base);
      await page.waitForSelector(`${slot}[data-avatar-load="error"]`);
      assert.equal(await page.locator("[data-avatar-fallback] img").isVisible(), true);
      await page.screenshot({ path: path.join(output, "fallback.png"), fullPage: false });
      await submit(page);
      await chat.received;
      chat.finish();
      await page.waitForSelector(".message.assistant");
      await top(page);
      await page.unroute("**/ai-coach.glb");
      await page.locator("[data-avatar-retry]").click();
      await ready(page);
      await state(page, "Idle_Base");
      assert.equal(await page.locator(`${slot} canvas`).count(), 1, "retry must not duplicate canvas");
      results.fallback = ["GLB 404", "chat still works", "retry succeeds"];
      await context.close();
    }
    {
      const { context, page } = await pageFor();
      await page.route("**/avatar/AvatarEngine.js", route => route.abort());
      const chat = await holdChat(page);
      await page.goto(base);
      await page.waitForSelector(`${slot}[data-avatar-load="error"]`);
      await submit(page);
      await chat.received;
      chat.finish();
      await page.waitForSelector(".message.assistant");
      results.engineFailure = "module failure is isolated from Chat";
      await context.close();
    }
    {
      const { context, page } = await pageFor({ reducedMotion: "reduce" });
      const chat = await holdChat(page);
      await page.goto(base);
      await ready(page);
      assert.equal((await diagnostics(page)).paused, true);
      await submit(page);
      await chat.received;
      await state(page, "Run");
      assert.equal((await diagnostics(page)).paused, true, "Chat must not override reduced motion");
      chat.finish();
      await page.waitForSelector(".message.assistant");
      await top(page);
      await page.locator("[data-avatar-motion]").click();
      assert.equal((await diagnostics(page)).paused, false);
      results.reducedMotion = "paused through Chat; explicit play works";
      await context.close();
    }
    for (const width of [390, 320]) {
      const { context, page } = await pageFor({ viewport: { width, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
      await page.goto(base);
      await ready(page);
      await page.locator(".companion-stage").scrollIntoViewIfNeeded();
      await page.waitForTimeout(300);
      await clearLabels(page);
      assert.equal((await page.locator(".companion-stage").boundingBox()).height, 400);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      const bounds = await page.locator(`${slot} canvas`).boundingBox();
      await page.touchscreen.tap(bounds.x + bounds.width * .5, bounds.y + bounds.height * .43);
      await state(page, "Interact");
      for (let i = 0; i < 20; i++) { await clearLabels(page); await page.waitForTimeout(100); }
      await state(page, "Idle_Base");
      await page.locator(".companion-stage").screenshot({ path: path.join(output, `mobile-${width}.png`) });
      results[`mobile${width}`] = "400px stage; complete silhouette; touch Interact; no horizontal overflow";
      await context.close();
    }
    assert.deepEqual(errors, [], "uncaught browser errors");
    results.pageErrors = errors;
    fs.writeFileSync(path.join(output, "results.json"), JSON.stringify(results, null, 2));
    console.log(JSON.stringify(results, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
