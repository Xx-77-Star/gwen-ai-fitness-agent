const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");

(async () => {
  const out = "logs/avatar-repair";
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ channel: process.env.AVATAR_BROWSER_CHANNEL || "msedge", headless: true, args: ["--enable-unsafe-swiftshader"] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(`${process.env.AVATAR_BASE_URL || "http://127.0.0.1:8080"}/avatar-test.html`);
    await page.waitForFunction(() => window.avatarViewer?.getDiagnostics().ready);
    const read = () => page.evaluate(() => window.avatarViewer.getDiagnostics());
    const idle = await read();
    assert.ok(idle.visibleMeshes.includes("Scissors"));
    await page.locator("#stage").screenshot({ path: `${out}/idle-scissors.png` });
    await page.evaluate(() => window.avatarViewer.interact());
    await page.waitForFunction(() => {
      const d = window.avatarViewer.getDiagnostics({ measureGeometry: false });
      return d.animationState === "interact" && d.animationTime >= .7 && d.animationTime < 1.3;
    });
    await page.locator("#pause").click();
    const interact = await read();
    for (const name of ["Scissors", "Eye_Playful", "Mouth_Playful"]) assert.ok(interact.visibleMeshes.includes(name), name);
    assert.ok(!interact.visibleMeshes.includes("Eye_Base"));
    assert.ok(interact.screenBounds.min.every(n => n > -1));
    assert.ok(interact.screenBounds.max.every(n => n < 1));
    await page.locator("#stage").screenshot({ path: `${out}/interact-expression.png` });
    await page.locator("#pause").click();
    await page.waitForFunction(() => window.avatarViewer.getDiagnostics({ measureGeometry: false }).animationState === "idle");
    const returned = await read();
    assert.ok(returned.visibleMeshes.includes("Eye_Base"));
    assert.ok(!returned.visibleMeshes.includes("Eye_Playful"));
    await page.evaluate(() => window.avatarViewer.triggerCastCycle());
    await page.waitForFunction(() => window.avatarViewer.getDiagnostics({ measureGeometry: false }).visibleMeshes.includes("Mouth_Smile"));
    assert.deepEqual(errors, []);
    fs.writeFileSync(`${out}/appearance.json`, JSON.stringify({ idle, interact, returned, errors }, null, 2));
    console.log("PASS: original scissors; Interact wink/open mouth; expression reset; Cast mouth; no clipping or page errors.");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
