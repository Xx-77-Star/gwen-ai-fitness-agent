import { HERO_AVATAR_CONFIG } from "./avatar-config.js";

// Owns product state, including requests that start/end before the GLB is ready.
// Three.js is imported separately so its failure cannot prevent chat startup.
export class AvatarController {
  constructor(mount, config = HERO_AVATAR_CONFIG, options = {}) {
    this.mount = mount;
    this.config = config;
    this.interactive = options.interactive !== false;
    // viewerControls=false -> Gwen is a companion: no rotate/zoom, click only.
    this.viewerControls = options.viewerControls !== false;
    this.readyHint = options.readyHint ?? "Gwen 在这里陪伴你。";
    this.idleHint = options.idleHint ?? "Gwen 在这里陪伴你。";
    this.loadingMessage = options.loadingMessage ?? "Gwen 正在准备中，你可以先开始对话。";
    this.recoveringMessage = options.recoveringMessage ?? "3D 显示正在恢复，对话仍可继续。";
    this.errorMessage = options.errorMessage ?? "暂时无法显示 3D Gwen，对话仍可正常使用。";
    this.busyStatus = options.busyStatus ?? "GWEN IS THINKING";
    this.readyStatus = options.readyStatus ?? "GWEN READY";
    this.lazy = options.lazy === true;
    // Product state machine (page-level, not the Three.js animator):
    //   idle     -> Idle_Base
    //   greeting -> Interact once, then auto-return to Idle_Base
    //   busy     -> Run (driven by setBusy)
    this.state = "idle";
    this.greetingDurationMs = options.greetingDurationMs ?? 3000;
    this.onGreeting = options.onGreeting ?? null;
    this.onGreetingEnd = options.onGreetingEnd ?? null;
    this.starting = false;
    this.busy = false;
    this.engine = null;
    this.disposed = false;
    this.attempt = null;
    this.events = new AbortController();
    const stage = mount.closest(".companion-stage") ?? mount;
    this.fallback = mount.querySelector("[data-avatar-fallback]");
    this.message = mount.querySelector("[data-avatar-message]");
    this.retry = mount.querySelector("[data-avatar-retry]");
    this.indicator = stage?.querySelector?.("[data-avatar-status]") ?? null;
    this.motion = stage?.querySelector?.("[data-avatar-motion]") ?? null;
    this.hint = stage?.querySelector?.("[data-avatar-hint]") ?? null;
    this.retry?.addEventListener("click", () => this.start(), { signal: this.events.signal });
    this.motion?.addEventListener("click", () => this.engine?.togglePause(), { signal: this.events.signal });
    window.addEventListener("pagehide", event => {
      if (!event.persisted) this.dispose();
    }, { signal: this.events.signal });
    this.ready = this.lazy ? Promise.resolve(null) : this.start();
  }

  // Explicit page-level state entry point.
  //   setState("greeting") -> play Interact once, show greeting UI, auto idle.
  setState(state) {
    if (this.disposed) return false;
    if (state !== "greeting") return false;
    if (this.state === "greeting") return true;
    if (!this.engine) {
      this.pendingGreeting = true;
      this.ready = this.preload();
      return true;
    }
    this.state = "greeting";
    this.pendingGreeting = false;
    this.onGreeting?.();
    this.engine.interact();
    // Greeting owns its own duration: it does NOT end when the Interact clip
    // happens to finish. Only this timer decides when the bubble closes.
    clearTimeout(this.greetingTimer);
    this.greetingTimer = setTimeout(() => this.endGreeting(), this.greetingDurationMs);
    return true;
  }

  // Ends the greeting on the independent duration timer (see setState).
  endGreeting() {
    if (this.state !== "greeting") return;
    clearTimeout(this.greetingTimer);
    this.state = this.busy ? "busy" : "idle";
    this.onGreetingEnd?.();
  }

  // Companion mode: the canvas stops taking pointer/keyboard input, so the
  // engine's OrbitControls (drag rotate / wheel zoom / arrow keys) can never
  // fire. Gwen responds to one thing only: a click on her silhouette.
  // The engine itself is untouched; this is pure adapter behaviour.
  applyCompanionMode() {
    const canvas = this.mount.querySelector("canvas");
    if (canvas) {
      canvas.style.pointerEvents = "none";
      canvas.tabIndex = -1;
      canvas.setAttribute("aria-hidden", "true");
    }
    this.mount.style.cursor = "pointer";
    this.companionClickHandler = event => {
      if (this.disposed || !this.engine) return;
      const rect = this.mount.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      // Gwen's projected silhouette bounds (NDC) from the engine diagnostics.
      const bounds = this.engine.getDiagnostics?.()?.screenBounds;
      if (!bounds) return;
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
      const inside = x >= bounds.min[0] && x <= bounds.max[0]
        && y >= bounds.min[1] && y <= bounds.max[1];
      if (inside) this.engine.interact();
    };
    this.mount.addEventListener("click", this.companionClickHandler, { signal: this.events.signal });
  }

  preload() {
    if (this.disposed || this.starting || this.engine || this.attempt) return this.ready;
    return (this.ready = this.start());
  }

  async start() {
    if (this.disposed) return null;
    this.starting = true;
    this.attempt?.abort();
    this.engine?.dispose();
    this.engine = null;
    const attempt = new AbortController();
    this.attempt = attempt;
    const current = () => !this.disposed && this.attempt === attempt && !attempt.signal.aborted;
    this.showFallback("loading");
    const timer = setTimeout(() => {
      if (!current()) return;
      attempt.abort();
      this.showFallback("error");
    }, this.config.loadTimeoutMs);
    try {
      const { createAvatarEngine } = await import("./AvatarEngine.js");
      if (!current()) return null;
      return await createAvatarEngine({
        ...this.config,
        mount: this.mount,
        interactive: this.interactive,
        signal: attempt.signal,
        onReady: engine => {
          if (!current()) return;
          this.engine = engine;
          engine.setBusy(this.busy);
          this.mount.dataset.avatarReady = "true";
          this.mount.dataset.avatarLoad = "ready";
          this.mount.setAttribute("aria-busy", "false");
          if (this.fallback) this.fallback.hidden = true;
          if (this.motion) this.motion.hidden = false;
          if (this.pendingGreeting) {
            this.pendingGreeting = false;
            this.setState("greeting");
          }
          if (!this.viewerControls) this.applyCompanionMode();
          if (this.hint) this.hint.textContent = this.readyHint;
          this.updateStatus();
        },
        onStateChange: state => {
          if (!current()) return;
          this.mount.dataset.avatarState = state.clip;

          if (this.motion) {
            this.motion.textContent = state.paused ? "播放动画" : "暂停动画";
            this.motion.setAttribute("aria-pressed", String(state.paused));
          }
        },
        onError: (error, details) => {
          if (current()) this.showFallback(details?.recoverable ? "recovering" : "error");
        },
      });
    } catch {
      if (current()) this.showFallback("error");
      return null;
    } finally {
      clearTimeout(timer);
      this.starting = false;
      if (!this.engine) this.attempt = null;
    }
  }

  setBusy(busy) {
    if (this.disposed) return;
    this.busy = Boolean(busy);
    if (this.busy && this.lazy) this.preload();
    this.engine?.setBusy(this.busy);
    this.updateStatus();
  }

  updateStatus() {
    this.mount.dataset.avatarBusy = String(this.busy);
    if (this.indicator) this.indicator.textContent = this.busy ? this.busyStatus : this.readyStatus;
  }

  showFallback(status) {
    this.mount.dataset.avatarReady = "false";
    this.mount.dataset.avatarLoad = status;
    this.mount.setAttribute("aria-busy", String(status === "loading"));
    if (this.fallback) this.fallback.hidden = false;
    if (this.retry) this.retry.hidden = status === "loading";
    if (this.motion) this.motion.hidden = true;
    if (this.message) {
      this.message.textContent = status === "loading" ? this.loadingMessage
        : status === "recovering" ? this.recoveringMessage
          : this.errorMessage;
    }
    if (this.hint) this.hint.textContent = this.idleHint;
    this.updateStatus();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    clearTimeout(this.greetingTimer);
    if (this.companionClickHandler) this.mount.removeEventListener("click", this.companionClickHandler);
    this.events.abort();
    this.attempt?.abort();
    this.engine?.dispose();
    this.engine = null;
  }
}