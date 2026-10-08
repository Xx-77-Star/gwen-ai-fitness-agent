import { createAvatarEngine } from "./avatar/AvatarEngine.js";

const $ = id => document.getElementById(id);
const vectorText = values => values.map(n => n.toFixed(6)).join(", ");
const boxText = box => `min [${vectorText(box.min)}]\nmax [${vectorText(box.max)}]`;

// The lab is an adapter, with no rendering or animation implementation of its own.
export async function startAvatarViewer() {
  const events = new AbortController();
  let viewer;
  const showReady = engine => {
    viewer ??= engine;
    $("loading").hidden = true;
    $("state-label").textContent = "READY · 自动取景";
    document.querySelectorAll(".toolbar button, .animation-controls button").forEach(button => { button.disabled = false; });
  };
  const engine = await createAvatarEngine({
    mount: $("stage"),
    canvas: $("avatar-canvas"),
    onReady: showReady,
    onProgress: message => { $("loading-detail").textContent = message; },
    onError: error => {
      $("loading").hidden = false;
      $("loading-title").textContent = "Avatar 显示中断";
      $("loading-detail").textContent = error.message;
      $("state-label").textContent = "加载失败";
      $("retry").hidden = false;
    },
    onDiagnostics: d => {
      $("model-size").textContent = `[${vectorText(d.sourceSize)}]`;
      $("model-bounds").textContent = boxText(d.sourceBounds);
      $("model-center").textContent = `[${vectorText(d.sourceCenter)}]`;
      $("model-scale").textContent = `${d.scale.toFixed(8)}\n1 / max(Idle width, height, depth)`;
      $("camera-position").textContent = `[${vectorText(d.camera)}]`;
      $("camera-target").textContent = `[${vectorText(d.target)}]`;
      $("camera-fit").textContent = `目标 ${(d.occupancy * 100).toFixed(0)}% · distance ${d.distance.toFixed(4)}\n${d.viewport.join(" × ")} · near ${d.near.toFixed(4)} / far ${d.far.toFixed(4)}`;
      $("animation-info").textContent = `${d.animation} · ${d.duration.toFixed(3)}s\n${d.paused ? "已暂停" : ["interact", "cast"].includes(d.animationState) ? "播放一次" : "循环播放"} · t=${d.animationTime.toFixed(2)}s\n加载状态 ${d.busy ? "进行中" : "空闲"} · ${d.enabledClips.length} 个动作`;
      $("animation-bounds").textContent = boxText(d.animationBounds);
      $("frame-center").textContent = `[${vectorText(d.animationBounds.min.map((v, i) => (v + d.animationBounds.max[i]) / 2))}]`;
      $("action-busy").setAttribute("aria-pressed", String(d.busy));
      $("action-busy").textContent = d.busy ? "结束计划生成 / 加载" : "模拟计划生成 / 加载";
      $("pause").textContent = d.paused ? "播放动画" : "暂停动画";
      $("pause").setAttribute("aria-pressed", String(d.paused));
      $("material-info").textContent = `${d.visibleMeshes.length} 可见网格\nHemisphere + 白色主光 + 紫色补光\n保留烘焙贴图，unlit 不受灯光增亮`;
      $("render-info").textContent = `WebGL2 · DPR ${d.dpr.toFixed(2)}\n${d.calls} draw calls · ${d.triangles.toLocaleString()} triangles`;
    },
  });
  const on = (id, handler) => $(id).addEventListener("click", handler, { signal: events.signal });
  on("reset", () => viewer.reset());
  on("rotate-left", () => viewer.rotate(-Math.PI / 12));
  on("rotate-right", () => viewer.rotate(Math.PI / 12));
  on("zoom-in", () => viewer.zoom(0.85));
  on("zoom-out", () => viewer.zoom(1 / 0.85));
  on("pause", () => viewer.togglePause());
  on("action-idle", () => viewer.idle());
  on("action-interact", () => viewer.interact());
  on("action-busy", () => viewer.setBusy(!viewer.getDiagnostics({ measureGeometry: false }).busy));
  on("action-cast", () => viewer.triggerCastCycle());
  on("bounds", () => $("bounds").setAttribute("aria-pressed", String(viewer.toggleBounds())));
  const dispose = () => {
    events.abort();
    engine.dispose();
    if (window.avatarViewer === viewer) delete window.avatarViewer;
  };
  viewer = Object.freeze({ ...engine, dispose });
  window.avatarViewer = viewer;
  window.addEventListener("pagehide", event => { if (!event.persisted) dispose(); }, { signal: events.signal });
  return viewer;
}
