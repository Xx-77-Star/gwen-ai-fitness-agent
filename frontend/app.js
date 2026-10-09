import { AvatarController } from "./avatar/AvatarController.js";
import { CHAT_AVATAR_CONFIG } from "./avatar/avatar-config.js";
import { parseJsonResponse } from "./response-json.mjs";

const avatarMount = document.querySelector("#avatar-stage-slot");
const greetingBubble = document.querySelector("[data-greeting-bubble]");
// First visit per session: show the greeting once, then remember it.
const GREETING_KEY = "fitlife_gwen_greeted";
let greetingHideTimer = null;
// Speech-bubble pop-in / fade-out (class-based so the CSS transition runs).
const showGreeting = () => {
  if (!greetingBubble) return;
  clearTimeout(greetingHideTimer);
  greetingBubble.hidden = false;
  requestAnimationFrame(() => greetingBubble.classList.add("is-visible"));
};
const hideGreeting = () => {
  if (!greetingBubble) return;
  greetingBubble.classList.remove("is-visible");
  greetingHideTimer = setTimeout(() => { greetingBubble.hidden = true; }, 300);
};
export const avatar = avatarMount
  ? new AvatarController(avatarMount, undefined, {
      // Gwen is an AI Companion, not a 3D model viewer: no rotate/zoom,
      // click her silhouette only (plays Interact -> Idle_Base).
      viewerControls: false,
      readyHint: "Gwen 在这里陪伴你。",
      onGreeting: showGreeting,
      onGreetingEnd: hideGreeting,
    })
  : null;

// First-visit Greeting Flow: ready -> Interact + bubble -> ~3s -> Idle_Base.
// No artificial delay: Gwen greets as soon as she is ready to be seen.
if (avatar && sessionStorage.getItem(GREETING_KEY) !== "true") {
  Promise.resolve(avatar.ready).then(() => {
    sessionStorage.setItem(GREETING_KEY, "true");
    avatar.setState("greeting");
  }).catch(() => null);
}

// Chat Experience companion: a compact, passive Gwen beside the composer.
// It reuses the same AvatarEngine + GLB and only switches Idle_Base / Run with
// chat request state. The Hero Avatar is deliberately NOT wired to chat.
const chatAvatarMount = document.querySelector("#chat-avatar");
const chatAvatarStatus = document.querySelector("#chat-avatar-status");
export const chatAvatar = chatAvatarMount
  ? new AvatarController(chatAvatarMount, CHAT_AVATAR_CONFIG, {
      interactive: false,
      lazy: true,
      readyHint: "",
      idleHint: "",
      loadingMessage: "Gwen 正在整理上下文并准备建议…",
      recoveringMessage: "Gwen 的 3D 显示正在恢复，对话不受影响。",
      errorMessage: "Gwen 暂时无法显示 3D，对话仍可正常使用。",
      busyStatus: "",
      readyStatus: "",
    })
  : null;

// Warm the compact companion after the Hero settles so the first message
// already finds it ready. Failure is non-fatal: the poster stays visible.
if (chatAvatar) {
  Promise.resolve(avatar?.ready)
    .catch(() => null)
    .then(() => {
      const schedule = window.requestIdleCallback ?? (callback => setTimeout(callback, 400));
      schedule(() => chatAvatar.preload().catch(() => null));
    });
}

function setChatCompanionStatus(busy) {
  if (!chatAvatarStatus) return;
  chatAvatarStatus.textContent = busy
    ? "正在处理你的问题…"
    : "随时待命，帮你把想法变成今天的训练。";
}
let chatBusy = false;

function updateLoadingOverlay() {
  // The status card is the feedback while it is visible; the old overlay is
  // only a fallback when SSE is unavailable.
  const statusActive = agentStatus && !agentStatus.hidden;
  if (loadingState) loadingState.hidden = !chatBusy || statusActive;
}

// ---- Gwen Agent execution status (stage list driven by live node events) ----
const agentStatus = document.querySelector("#agent-status");
const agentStatusSteps = document.querySelector("#agent-status-steps");
const AGENT_STAGES = [
  ["memory_retrieve", "正在读取你的训练档案"],
  ["intent", "正在理解你的需求"],
  ["rag_retrieve", "正在检索专业训练知识"],
  ["response", "正在生成个性化方案"],
];
let agentEventSource = null;

function resetAgentStatus() {
  if (!agentStatus || !agentStatusSteps) return;
  const fragment = document.createDocumentFragment();
  AGENT_STAGES.forEach(([, label]) => {
    const item = document.createElement("li");
    item.className = "agent-status-step";
    const mark = document.createElement("span");
    mark.className = "step-mark";
    mark.setAttribute("aria-hidden", "true");
    mark.append(document.createElement("span"));
    item.append(mark, document.createTextNode(label));
    fragment.append(item);
  });
  agentStatusSteps.replaceChildren(fragment);
  agentStatus.hidden = false;
  updateLoadingOverlay();
}

function markAgentStage(nodeName, phase) {
  if (!agentStatusSteps) return;
  const index = AGENT_STAGES.findIndex(([name]) => name === nodeName);
  if (index < 0) return;
  const items = agentStatusSteps.querySelectorAll(".agent-status-step");
  items.forEach((item, itemIndex) => {
    if (itemIndex < index) item.className = "agent-status-step is-done";
    else if (itemIndex === index) {
      item.className = "agent-status-step " + (phase === "end" ? "is-done" : "is-active");
    }
  });
}

function hideAgentStatus() {
  if (agentStatus) agentStatus.hidden = true;
  updateLoadingOverlay();
}

function closeAgentStatusStream() {
  if (agentEventSource) {
    agentEventSource.close();
    agentEventSource = null;
  }
}

function openAgentStatusStream(runConversationId) {
  if (!agentStatus || typeof EventSource === "undefined") return;
  closeAgentStatusStream();
  resetAgentStatus();
  const source = new EventSource(
    "/chat/" + encodeURIComponent(runConversationId) + "/events?user_id=" + encodeURIComponent(userId)
  );
  agentEventSource = source;
  source.onmessage = event => {
    try {
      const payload = JSON.parse(event.data);
      markAgentStage(payload.node_name, payload.phase);
    } catch { /* ignore malformed status frames */ }
  };
  source.onerror = () => {
    source.close();
    if (agentEventSource === source) agentEventSource = null;
  };
}

const modeButtons = document.querySelectorAll(".mode-button");
const goalCards = document.querySelectorAll(".goal-card");
const promptButtons = document.querySelectorAll("[data-prompt]");
const traceDrawer = document.querySelector("#trace-drawer");
const drawerBackdrop = document.querySelector("#drawer-backdrop");
const openTraceButtons = document.querySelectorAll("#open-trace");
const closeTraceButton = document.querySelector("#close-trace");
const startChatButton = document.querySelector("#start-chat");
const conversationSection = document.querySelector(".conversation-section");
// ---- Gwen Memory Center (management UI only; the memory service is untouched) ----
const memoryBackdrop = document.querySelector("#memory-backdrop");
const memoryList = document.querySelector("#memory-list");
const openMemoryButton = document.querySelector("#open-memory-center");
const viewMemoryButton = document.querySelector("#view-memory");
const closeMemoryButton = document.querySelector("#close-memory-center");
const resetMemoryButton = document.querySelector("#reset-memory");
const confirmBackdrop = document.querySelector("#memory-confirm-backdrop");
const confirmResetButton = document.querySelector("#confirm-memory-reset");
const cancelResetButton = document.querySelector("#cancel-memory-reset");
const MEMORY_FIELDS = [
  ["fitness_goal", "训练目标", "Gwen 会根据你的目标调整训练建议"],
  ["training_preference", "训练方式", "Gwen 会按你偏好的训练方式来安排"],
  ["experience_level", "训练经验", "Gwen 会按你的经验程度把握强度"],
];

function setMode(mode) {
  modeButtons.forEach((item) => item.classList.toggle("is-active", item.dataset.mode === mode));
  document.body.dataset.mode = mode;
  setDrawer(mode === "developer");
}
const askGwenButton = document.querySelector("#ask-gwen");
function openChat() {
  conversationSection?.scrollIntoView({behavior: "smooth", block: "start"});
  input?.focus({preventScroll: true});
}
if (startChatButton && conversationSection) {
  startChatButton.addEventListener("click", () => {
    openChat();
  });
}
askGwenButton?.addEventListener("click", openChat);
modeButtons.forEach((button) => button.addEventListener("click", () => {
  setMode(button.dataset.mode === "developer" ? "developer" : "user");
}));
goalCards.forEach((card) => card.addEventListener("click", () => {
  goalCards.forEach((item) => item.classList.toggle("is-selected", item === card));
  const goal = card.dataset.goal;
  if (goal) input.value = `我的训练目标是${goal}，请给我今天的训练建议。`;
  input.focus();
}));
promptButtons.forEach((button) => button.addEventListener("click", () => {
  input.value = button.dataset.prompt || ""; input.focus();
}));
function setDrawer(open) {
  traceDrawer.classList.toggle("is-open", open);
  traceDrawer.setAttribute("aria-hidden", String(!open));
  drawerBackdrop.hidden = !open;
  if (open) closeTraceButton.focus();
}
openTraceButtons.forEach((button) => button.addEventListener("click", () => setDrawer(true)));
closeTraceButton.addEventListener("click", () => setDrawer(false));
drawerBackdrop.addEventListener("click", () => setDrawer(false));
document.addEventListener("keydown", (event) => { if (event.key === "Escape" && traceDrawer.classList.contains("is-open")) setDrawer(false); });
function memoryNode(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

let memoryOpener = null;
function openMemoryCenter(opener) {
  if (!memoryBackdrop) return;
  // Remember the control that opened the panel so focus returns to it.
  if (opener) memoryOpener = opener;
  memoryBackdrop.hidden = false;
  closeMemoryButton?.focus();
  renderMemoryCenter();
}
function closeMemoryCenter() {
  if (memoryBackdrop) memoryBackdrop.hidden = true;
  // preventScroll is essential: .focus() would otherwise scroll the page to
  // the header button ("回到主页面") when the panel was opened from below.
  (memoryOpener || openMemoryButton)?.focus({preventScroll: true});
}
function openResetConfirm() {
  if (confirmBackdrop) confirmBackdrop.hidden = false;
  cancelResetButton?.focus();
}
function closeResetConfirm() {
  if (confirmBackdrop) confirmBackdrop.hidden = true;
  resetMemoryButton?.focus({preventScroll: true});
}

function formatMemoryDate(value) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) return "暂无记录";
  return `${date.getFullYear()}年${String(date.getMonth() + 1).padStart(2, "0")}月${String(date.getDate()).padStart(2, "0")}日`;
}

function renderMemorySummary(memories) {
  const summary = document.querySelector("#memory-summary");
  if (!summary) return;
  const countNode = summary.querySelector("[data-memory-count]");
  const updatedNode = summary.querySelector("[data-memory-updated]");
  const latest = memories.reduce((max, item) => {
    const t = item.updated_at ? new Date(item.updated_at).getTime() : 0;
    return t > max ? t : max;
  }, 0);
  if (countNode) countNode.textContent = `Gwen 记住了关于你的 ${memories.length} 条长期记忆`;
  if (updatedNode) updatedNode.textContent = `最近更新：${formatMemoryDate(latest ? new Date(latest) : null)}`;
}

async function renderMemoryCenter() {
  if (!memoryList) return;
  memoryList.replaceChildren(memoryNode("p", "empty-copy", "正在读取 Gwen 的记忆…"));
  try {
    const response = await fetch(`/memory/${encodeURIComponent(userId)}`);
    const payload = await parseJsonResponse(response);
    if (!response.ok) throw new Error(payload.detail || "记忆读取失败");
    const memories = payload.memories || [];
    renderMemorySummary(memories);
    const saved = new Map(memories.map(item => [item.memory_key, item]));
    const fragment = document.createDocumentFragment();
    MEMORY_FIELDS.forEach(([key, label, note]) => {
      const item = saved.get(key);
      const value = (item ? item.memory_value : "").trim();
      const card = memoryNode("div", "memory-card");
      card.id = `memory-card-${key}`;
      card.dataset.empty = String(!value);

      const head = memoryNode("div", "memory-card-head");
      head.append(memoryNode("span", "memory-card-label", label));
      const valueNode = memoryNode("p", value ? "memory-card-value" : "memory-card-value is-empty", value || "暂未设置");
      const noteNode = memoryNode("p", "memory-card-note", note);
      const updatedNode = memoryNode(
        "p", "memory-card-updated", `最后更新：${item ? formatMemoryDate(item.updated_at) : "暂无记录"}`
      );

      const edit = memoryNode("div", "memory-card-edit");
      const field = memoryNode("div", "memory-field");
      const input = memoryNode("input");
      input.id = `memory-input-${key}`;
      input.type = "text";
      input.maxLength = 4000;
      input.placeholder = "填写后 Gwen 会记住";
      input.value = value;
      field.append(input);
      const save = memoryNode("button", "memory-save", value ? "保存修改" : "保存");
      save.type = "button";
      const status = memoryNode("p", "memory-card-status", "");
      edit.append(field, save);

      save.addEventListener("click", async () => {
        const next = input.value.trim();
        if (!next) { status.className = "memory-card-status bad"; status.textContent = "请先填写内容。"; return; }
        save.disabled = true;
        status.className = "memory-card-status"; status.textContent = "正在保存…";
        try {
          const res = await fetch(`/memory/${encodeURIComponent(userId)}/${key}`, {
            method: "PUT", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({memory_value: next}),
          });
          const body = await parseJsonResponse(res);
          if (!res.ok) throw new Error(body.detail || "保存失败");
          // Immediately refresh the visible card content from the saved value.
          card.dataset.empty = "false";
          valueNode.className = "memory-card-value";
          valueNode.textContent = body.memory_value;
          input.value = body.memory_value;
          save.textContent = "保存修改";
          updatedNode.textContent = `最后更新：${formatMemoryDate(body.updated_at)}`;
          saved.set(key, body);
          renderMemorySummary([...saved.values()]);
          status.className = "memory-card-status ok"; status.textContent = "已保存，Gwen 记住了。";
        } catch (error) {
          status.className = "memory-card-status bad";
          status.textContent = error instanceof Error ? error.message : "保存失败";
        } finally { save.disabled = false; }
      });

      card.append(head, valueNode, noteNode, updatedNode, edit, status);
      fragment.append(card);
    });
    memoryList.replaceChildren(fragment);
  } catch (error) {
    memoryList.replaceChildren(memoryNode("div", "memory-card", error instanceof Error ? error.message : "记忆读取失败"));
  }
}

async function resetMemoryCenter() {
  if (!memoryList || !confirmResetButton) return;
  confirmResetButton.disabled = true;
  try {
    const response = await fetch(`/memory/${encodeURIComponent(userId)}`, { method: "DELETE" });
    const payload = await parseJsonResponse(response);
    if (!response.ok) throw new Error(payload.detail || "重置失败");
    closeResetConfirm();
    await renderMemoryCenter();
  } catch (error) {
    closeResetConfirm();
    memoryList.replaceChildren(memoryNode("div", "memory-card", error instanceof Error ? error.message : "重置失败"));
  } finally { confirmResetButton.disabled = false; }
}

openMemoryButton?.addEventListener("click", () => openMemoryCenter(openMemoryButton));
viewMemoryButton?.addEventListener("click", () => openMemoryCenter(viewMemoryButton));
closeMemoryButton?.addEventListener("click", closeMemoryCenter);
resetMemoryButton?.addEventListener("click", openResetConfirm);
confirmResetButton?.addEventListener("click", resetMemoryCenter);
cancelResetButton?.addEventListener("click", closeResetConfirm);
memoryBackdrop?.addEventListener("click", event => { if (event.target === memoryBackdrop) closeMemoryCenter(); });
confirmBackdrop?.addEventListener("click", event => { if (event.target === confirmBackdrop) closeResetConfirm(); });
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  if (confirmBackdrop && !confirmBackdrop.hidden) { closeResetConfirm(); return; }
  if (memoryBackdrop && !memoryBackdrop.hidden) closeMemoryCenter();
});
// ---- Workout check-in journal (persisted through /workouts) ----
const workoutBackdrop = document.querySelector("#workout-backdrop");
const workoutForm = document.querySelector("#workout-form");
const workoutDate = document.querySelector("#workout-date");
const workoutStatus = document.querySelector("#workout-status");
const workoutHistoryList = document.querySelector("#workout-history-list");
const workoutHistoryCount = document.querySelector("#workout-history-count");
const openWorkoutButton = document.querySelector("#open-workout-checkin");
const closeWorkoutButton = document.querySelector("#close-workout-checkin");
const cityInput = document.querySelector("#user-city");
const citySaveButton = document.querySelector("#save-user-city");
const cityStatus = document.querySelector("#city-status");
async function loadUserCity() {
  if (!cityInput) return;
  try {
    const response = await fetch(`/profile/${encodeURIComponent(userId || localStorage.getItem("fitlife-user-id") || "")}`);
    const payload = await parseJsonResponse(response);
    if (response.ok) {
      cityInput.value = payload.city || "";
      if (payload.city && cityStatus) {
        cityStatus.className = "city-status ok";
        cityStatus.textContent = "已记住所在地";
      }
    }
  } catch { /* profile may not exist yet */ }
}
citySaveButton?.addEventListener("click", async () => {
  if (!cityInput || !cityStatus) return;
  const city = cityInput.value.trim();
  if (!city) { cityStatus.className = "city-status bad"; cityStatus.textContent = "请填写城市"; return; }
  citySaveButton.disabled = true;
  try {
    const response = await fetch(`/profile/${encodeURIComponent(userId)}/city`, {
      method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({city}),
    });
    const body = await parseJsonResponse(response);
    if (!response.ok) throw new Error(body.detail || "城市保存失败");
    cityStatus.className = "city-status ok"; cityStatus.textContent = "城市已保存，Gwen 会长期记住";
  } catch (error) {
    cityStatus.className = "city-status bad"; cityStatus.textContent = error instanceof Error ? error.message : "城市保存失败";
  } finally { citySaveButton.disabled = false; }
});
const workoutMUSCLE_GROUPS = ["胸", "背", "腿", "肩", "手臂", "核心"];
let workoutOpener = null;

function formatWorkoutDate(value) {
  const date = value ? new Date(value + "T00:00:00") : null;
  if (!date || Number.isNaN(date.getTime())) return "暂无记录";
  return `${date.getFullYear()}年${String(date.getMonth() + 1).padStart(2, "0")}月${String(date.getDate()).padStart(2, "0")}日`;
}
function todayLocalDate() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
}
function renderWorkoutHistory(records) {
  if (!workoutHistoryList) return;
  if (!records.length) {
    workoutHistoryList.replaceChildren(memoryNode("p", "empty-copy", "还没有训练打卡记录。"));
    if (workoutHistoryCount) workoutHistoryCount.textContent = "暂无记录";
    return;
  }
  if (workoutHistoryCount) workoutHistoryCount.textContent = `${records.length} 条记录`;
  const fragment = document.createDocumentFragment();
  records.forEach((record) => {
    const item = memoryNode("article", "workout-history-item");
    const head = memoryNode("div", "workout-history-head");
    head.append(
      memoryNode("strong", "", `${formatWorkoutDate(record.date)} · ${record.muscle_group}`),
      memoryNode("span", "workout-history-exercise", record.exercise),
    );
    item.append(
      head,
      memoryNode("p", "workout-history-meta", `${record.weight}kg · ${record.sets} 组 × ${record.reps} 次 · ${record.feeling}`),
    );
    if (record.note) item.append(memoryNode("p", "workout-history-note", record.note));
    fragment.append(item);
  });
  workoutHistoryList.replaceChildren(fragment);
}
async function loadWorkoutHistory() {
  if (!workoutHistoryList) return;
  workoutHistoryList.replaceChildren(memoryNode("p", "empty-copy", "正在读取训练记录…"));
  try {
    const response = await fetch(`/workouts/${encodeURIComponent(userId)}?limit=20`);
    const payload = await parseJsonResponse(response);
    if (!response.ok) throw new Error(payload.detail || "训练记录读取失败");
    renderWorkoutHistory(Array.isArray(payload) ? payload : []);
  } catch (error) {
    workoutHistoryList.replaceChildren(memoryNode("div", "workout-history-item", error instanceof Error ? error.message : "训练记录读取失败"));
  }
}
function openWorkoutCheckin(opener) {
  if (!workoutBackdrop) return;
  if (opener) workoutOpener = opener;
  if (workoutDate && !workoutDate.value) workoutDate.value = todayLocalDate();
  workoutBackdrop.hidden = false;
  document.querySelector("#workout-exercise")?.focus();
  loadWorkoutHistory();
}
function closeWorkoutCheckin() {
  if (workoutBackdrop) workoutBackdrop.hidden = true;
  (workoutOpener || openWorkoutButton)?.focus({preventScroll: true});
}
openWorkoutButton?.addEventListener("click", () => openWorkoutCheckin(openWorkoutButton));
closeWorkoutButton?.addEventListener("click", closeWorkoutCheckin);
workoutBackdrop?.addEventListener("click", (event) => { if (event.target === workoutBackdrop) closeWorkoutCheckin(); });
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && workoutBackdrop && !workoutBackdrop.hidden) closeWorkoutCheckin();
});
workoutForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const saveButton = document.querySelector("#save-workout");
  const payload = {
    user_id: userId,
    date: workoutDate?.value || todayLocalDate(),
    muscle_group: document.querySelector("#workout-muscle-group")?.value || "",
    exercise: document.querySelector("#workout-exercise")?.value.trim() || "",
    weight: Number(document.querySelector("#workout-weight")?.value || 0),
    sets: Number(document.querySelector("#workout-sets")?.value || 0),
    reps: Number(document.querySelector("#workout-reps")?.value || 0),
    feeling: document.querySelector("#workout-feeling")?.value.trim() || "",
    note: document.querySelector("#workout-note")?.value.trim() || "",
  };
  if (workoutStatus) { workoutStatus.className = "workout-status"; workoutStatus.textContent = "正在保存…"; }
  if (saveButton) saveButton.disabled = true;
  try {
    const response = await fetch("/workouts", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    const body = await parseJsonResponse(response);
    if (!response.ok) throw new Error(body.detail || "训练打卡保存失败");
    if (workoutStatus) { workoutStatus.className = "workout-status ok"; workoutStatus.textContent = "已保存，Gwen 会参考这条训练记录。"; }
    workoutForm.reset();
    if (workoutDate) workoutDate.value = todayLocalDate();
    await loadWorkoutHistory();
  } catch (error) {
    if (workoutStatus) { workoutStatus.className = "workout-status bad"; workoutStatus.textContent = error instanceof Error ? error.message : "训练打卡保存失败"; }
  } finally {
    if (saveButton) saveButton.disabled = false;
  }
});
const NODE_DISPLAY = {
  load_profile: ["读取用户画像", "profile"], memory_retrieve: ["读取记忆", "memory"],
  rag_retrieve: ["检索知识库", "retrieval"], intent: ["识别用户意图", "reasoning"],
  tool_decision: ["判断工具需求", "decision"], tool_execution: ["执行工具", "tools"],
  response: ["生成回答", "response"], memory_update: ["更新记忆", "memory"],
  conversation_persistence: ["保存会话", "persistence"]
};
const loadingState = document.querySelector("#loading-state");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messageList = document.querySelector("#message-list");
const emptyState = document.querySelector("#chat-empty");
const traceRun = document.querySelector("#trace-run");
const connectionState = document.querySelector("#connection-state");
const newChatButton = document.querySelector("#new-chat");
let conversationId = sessionStorage.getItem("fitlife-conversation-id");
let userId = localStorage.getItem("fitlife-user-id");
if (!userId) { userId = `web-${crypto.randomUUID()}`; localStorage.setItem("fitlife-user-id", userId); }
loadUserCity();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || sendButton.disabled) return;
  appendMessage("user", message); input.value = ""; setBusy(true);
  // The client picks the run id so it can subscribe to node lifecycle events
  // before the request starts - no fixed timers, real workflow stage feedback.
  const runConversationId = conversationId || crypto.randomUUID();
  openAgentStatusStream(runConversationId);
  try {
    const response = await fetch("/chat", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({user_id: userId, message, conversation_id: runConversationId})});
    const payload = await parseJsonResponse(response);
    if (!response.ok) throw new Error(payload.detail || "Gwen 暂时没有回应，请稍后重试");
    conversationId = payload.conversation_id; sessionStorage.setItem("fitlife-conversation-id", conversationId);
    appendMessage("assistant", payload.response); renderTrace(payload.trace || {}); if (connectionState) if (connectionState) connectionState.textContent = "Gwen 在线";
  } catch (error) {
    appendMessage("error", error instanceof Error ? error.message : "Gwen 暂时没有回应"); if (connectionState) if (connectionState) connectionState.textContent = "Gwen 暂时离线";
  } finally { closeAgentStatusStream(); hideAgentStatus(); setBusy(false); input.focus(); }
});

newChatButton.addEventListener("click", () => {
  closeAgentStatusStream();
  hideAgentStatus();
  conversationId = null; sessionStorage.removeItem("fitlife-conversation-id");
  messageList.querySelectorAll(".message").forEach((node) => node.remove()); emptyState.hidden = false;
  renderTrace({}); if (traceRun) traceRun.textContent = "等待请求";
  if (connectionState) connectionState.textContent = "Gwen 就绪"; input.focus();
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
});
function setBusy(busy) {
  chatBusy = Boolean(busy);
  chatAvatar?.setBusy(busy);
  setChatCompanionStatus(busy);
  sendButton.disabled = busy; sendButton.textContent = busy ? "生成中" : "发送";
  updateLoadingOverlay();
  if (connectionState) if (connectionState) connectionState.textContent = busy ? "处理中" : connectionState.textContent;
}
function appendMessage(role, content) {
  emptyState.hidden = true; const message = document.createElement("div");
  message.className = `message ${role}`; message.textContent = content;
  messageList.append(message); messageList.scrollTop = messageList.scrollHeight;
}

function normalizeTrace(raw) {
  const source = object(raw);
  const nodes = array(source.nodes);
  const evidence = object(source.evidence);
  const memory = object(source.memory);
  const rag = object(evidence.rag) || object(source.rag);
  const tools = object(evidence.tools) || object(source.tools);
  const exchanges = array(source.tool_exchanges);
  const chunks = array(rag.chunks);
  const timeline = array(source.timeline).length
    ? array(source.timeline).map((item, index) => normalizeTimeline(item, index))
    : nodes.map((item, index) => normalizeTimeline(item, index));
  const normalizedTools = exchanges.length
    ? exchanges.map(normalizeToolExchange)
    : mergeLegacyTools(array(tools.calls), array(tools.results));
  const summary = object(source.summary);
  const run = object(source.run);
  return {
    run: {
      runId: text(run.run_id, source.run_id) || "", status: text(run.status, source.status) || "unknown",
      durationMs: number(run.duration_ms, source.duration_ms), stopReason: text(run.stop_reason) || "unknown",
      error: run.error || source.error || null
    },
    summary: {
      nodeCount: summaryValue(summary.node_count, timeline.length),
      toolCallCount: summaryValue(summary.tool_call_count, normalizedTools.length),
      toolResultCount: summaryValue(summary.tool_result_count, normalizedTools.filter((item) => item.hasResult).length),
      ragChunkCount: summaryValue(summary.rag_chunk_count, chunks.length),
      memoryFactCount: summaryValue(summary.memory_fact_count, array(evidence.memory?.facts).length || array(memory.long_term).length),
      memoryTurnCount: summaryValue(summary.memory_turn_count, array(evidence.memory?.turns).length || array(memory.short_term).length),
      sourceCount: summaryValue(summary.source_count, uniqueSources([...array(evidence.sources), ...array(source.sources)]).length)
    },
    timeline,
    evidence: {
      memory: {
        facts: array(evidence.memory?.facts).length ? array(evidence.memory?.facts) : array(memory.long_term),
        turns: array(evidence.memory?.turns).length ? array(evidence.memory?.turns) : array(memory.short_term)
      },
      rag: {chunks}, tools: {exchanges: normalizedTools},
      sources: uniqueSources([...array(evidence.sources), ...array(source.sources)])
    }
  };
}
function normalizeTimeline(raw, index) {
  const item = object(raw); const name = text(item.node_name) || "unknown";
  const display = NODE_DISPLAY[name] || [text(item.label) || name, text(item.category) || "other"];
  return {sequence: summaryValue(item.sequence, index + 1), nodeName: name,
    label: text(item.label) || display[0], category: text(item.category) || display[1],
    status: text(item.status) || "unknown", durationMs: number(item.duration_ms), details: object(item.details)};
}
function normalizeToolExchange(raw) {
  const item = object(raw);
  return {callId: text(item.call_id) || "", toolName: text(item.tool_name) || "unknown tool",
    arguments: item.arguments || {}, status: text(item.status) || "pending", result: item.result ?? null,
    error: item.error ?? null, resultSummary: object(item.result_summary),
    hasResult: Boolean(item.result !== null || item.error !== null || item.result_summary || ["success","error"].includes(text(item.status)))};
}
function mergeLegacyTools(calls, results) {
  const exchanges = calls.map((call, index) => {
    const item = object(call);
    return normalizeToolExchange({call_id: item.call_id || `legacy-call-${index + 1}`, tool_name: item.tool_name,
      arguments: item.arguments || {}, status: "pending"});
  });
  results.forEach((raw) => {
    const item = object(raw); const id = text(item.call_id, item.tool_call_id); const name = text(item.tool_name);
    let match = exchanges.find((exchange) => !exchange.hasResult && id && exchange.callId === id);
    if (!match) match = exchanges.find((exchange) => !exchange.hasResult && !exchange.callId && exchange.toolName === name);
    const normalized = normalizeToolExchange({call_id: id, tool_name: name, arguments: item.arguments || {},
      status: item.status, result: item.result, error: item.error, result_summary: item.result_summary});
    if (match) Object.assign(match, normalized); else exchanges.push(normalized);
  });
  return exchanges;
}
function renderTrace(raw) {
  const trace = normalizeTrace(raw);
  if (traceRun) traceRun.textContent = trace.run.runId || "无 Trace ID";
  renderRunSummary(trace); renderTimeline(trace.timeline); renderEvidence(trace.evidence);
}

function renderRunSummary(trace) {
  const container = document.querySelector("#run-summary");
  if (!trace.run.runId && !trace.summary.nodeCount && !trace.timeline.length) {
    container.replaceChildren(emptyCopy("发送消息后显示 Agent 执行过程。")); return;
  }
  const grid = document.createElement("div"); grid.className = "run-summary-grid";
  grid.append(summaryItem("状态", statusLabel(trace.run.status), trace.run.status),
    summaryItem("总耗时", duration(trace.run.durationMs)), summaryItem("节点数量", String(trace.summary.nodeCount)),
    summaryItem("工具数量", String(trace.summary.toolCallCount)), summaryItem("知识来源", String(trace.summary.sourceCount)),
    summaryItem("停止原因", trace.run.stopReason));
  const meta = document.createElement("div"); meta.className = "run-meta";
  const id = document.createElement("span"); id.className = "run-id"; id.textContent = trace.run.runId || "无 Trace ID";
  const reason = document.createElement("span"); reason.textContent = `停止原因：${trace.run.stopReason}`;
  meta.append(id, reason); const nodes = [grid, meta];
  if (trace.run.error) {
    const error = document.createElement("div"); error.className = "run-error";
    error.textContent = `${textValue(trace.run.error.type) || "错误"}：${textValue(trace.run.error.message) || "未知错误"}`; nodes.push(error);
  }
  container.replaceChildren(...nodes);
}
function summaryItem(label, value, status) {
  const item = document.createElement("div"); item.className = "summary-item";
  const caption = document.createElement("span"); caption.className = "summary-label"; caption.textContent = label;
  const output = document.createElement("strong"); output.className = status ? `summary-value status-${statusClass(status)}` : "summary-value";
  output.textContent = value; item.append(caption, output); return item;
}
function renderTimeline(items) {
  const container = document.querySelector("#execution-timeline");
  if (!items.length) { container.replaceChildren(emptyCopy("暂无执行节点。")); return; }
  const list = document.createElement("ol"); list.className = "timeline-list";
  items.forEach((item, index) => {
    const row = document.createElement("li"); row.className = `timeline-item status-${statusClass(item.status)}`;
    const details = document.createElement("details"); details.className = "timeline-details";
    const summary = document.createElement("summary"); summary.className = "timeline-card";
    const markerColumn = document.createElement("div"); markerColumn.className = "timeline-marker-column";
    const marker = document.createElement("div"); marker.className = "timeline-marker"; marker.setAttribute("aria-hidden", "true");
    markerColumn.append(marker);
    if (index < items.length - 1) {
      const connector = document.createElement("div"); connector.className = "timeline-connector";
      connector.setAttribute("aria-hidden", "true"); markerColumn.append(connector);
    }
    const content = document.createElement("div"); content.className = "timeline-content";
    const titleLine = document.createElement("div"); titleLine.className = "timeline-title-line";
    const title = document.createElement("strong"); title.textContent = `${item.sequence}. ${item.label}`;
    const meta = document.createElement("div"); meta.className = "timeline-meta";
    meta.append(statusBadge(item.status), durationText(item.durationMs)); titleLine.append(title, meta);
    const name = document.createElement("span"); name.className = "timeline-node-name"; name.textContent = `${item.nodeName} · ${item.category}`;
    const hint = document.createElement("span"); hint.className = "timeline-hint"; hint.textContent = "查看详情";
    content.append(titleLine, name, hint); summary.append(markerColumn, content);
    const detail = document.createElement("div"); detail.className = "detail-panel"; appendJson(detail, "节点详情", item.details);
    details.append(summary, detail); row.append(details); list.append(row);
  });
  container.replaceChildren(list);
}
function renderEvidence(evidence) {
  const container = document.querySelector("#trace-evidence"); const sections = [];
  const memory = section("Memory", "记忆");
  if (!evidence.memory.facts.length && !evidence.memory.turns.length) memory.append(emptyCopy("暂无记忆内容。"));
  else { const count = document.createElement("p"); count.className = "evidence-summary";
    count.textContent = `使用 ${evidence.memory.facts.length} 条长期记忆，${evidence.memory.turns.length} 条短期记忆`;
    memory.append(count, renderMemory(evidence.memory)); }
  sections.push(memory);
  const rag = section("RAG Sources", "知识检索");
  if (!evidence.rag.chunks.length) rag.append(emptyCopy("暂无知识检索结果。"));
  else { const count = document.createElement("p"); count.className = "evidence-summary"; count.textContent = `检索到 ${evidence.rag.chunks.length} 个知识片段`;
    rag.append(count, renderRag(evidence.rag.chunks)); }
  sections.push(rag);
  sections.push(renderTools(evidence.tools.exchanges));
  const sources = section("Sources", "知识来源");
  if (!evidence.sources.length) sources.append(emptyCopy("暂无来源。")); else sources.append(renderSources(evidence.sources));
  sections.push(sources); container.replaceChildren(...sections);
}

function renderMemory(memory) {
  const stack = document.createElement("div"); stack.className = "memory-stack";
  memory.facts.forEach((fact, index) => {
    const item = object(fact); const card = document.createElement("article"); card.className = "memory-card";
    const key = document.createElement("strong"); key.textContent = text(item.memory_key) || `Memory ${index + 1}`;
    const value = document.createElement("p"); value.className = "memory-value"; value.textContent = textValue(item.memory_value);
    card.append(key, value); stack.append(card);
  });
  memory.turns.slice(-4).forEach((turn, index) => {
    const item = object(turn); const card = document.createElement("article"); card.className = "memory-card memory-turn";
    const role = document.createElement("strong"); role.textContent = text(item.role) || `turn-${index + 1}`;
    const content = document.createElement("p"); content.className = "memory-value"; content.textContent = textValue(item.content);
    card.append(role, content); stack.append(card);
  });
  return stack;
}
function renderRag(chunks) {
  const stack = document.createElement("div"); stack.className = "rag-stack";
  chunks.forEach((raw) => {
    const item = object(raw); const card = document.createElement("article"); card.className = "rag-card";
    const heading = document.createElement("div"); heading.className = "rag-card-heading";
    const source = document.createElement("strong"); source.textContent = text(item.source) || "unknown source";
    const score = document.createElement("span"); score.className = "source-score"; score.textContent = `score ${number(item.score).toFixed(3)}`;
    heading.append(source, score); const excerpt = document.createElement("p"); excerpt.className = "content-excerpt"; excerpt.textContent = textValue(item.content);
    card.append(heading, excerpt); stack.append(card);
  });
  return stack;
}
function renderTools(exchanges) {
  const sectionNode = section("Tool Exchange", "工具调用");
  if (!exchanges.length) { sectionNode.append(emptyCopy("暂无工具调用。")); return sectionNode; }
  const list = document.createElement("div"); list.className = "tool-exchange-list";
  exchanges.forEach((exchange) => {
    const card = document.createElement("article"); card.className = `tool-exchange status-${statusClass(exchange.status)}`;
    const heading = document.createElement("div"); heading.className = "tool-exchange-heading";
    const name = document.createElement("strong"); name.textContent = exchange.toolName;
    const badges = document.createElement("div"); badges.className = "tool-exchange-badges"; badges.append(statusBadge(exchange.status));
    if (exchange.callId) { const id = document.createElement("span"); id.className = "call-id"; id.textContent = exchange.callId; badges.append(id); }
    heading.append(name, badges);
    const details = document.createElement("details"); details.className = "tool-exchange-details";
    const summary = document.createElement("summary"); summary.textContent = "查看参数与结果";
    const body = document.createElement("div"); body.className = "detail-panel";
    appendJson(body, "Arguments", exchange.arguments, true);
    if (exchange.result !== null) appendJson(body, "Result", exchange.result, true);
    if (exchange.error !== null) appendJson(body, "Error", exchange.error, true);
    if (Object.keys(exchange.resultSummary).length) appendJson(body, "Result Summary", exchange.resultSummary, true);
    details.append(summary, body); card.append(heading, details); list.append(card);
  });
  sectionNode.append(list); return sectionNode;
}
function renderSources(sources) {
  const list = document.createElement("div"); list.className = "source-list";
  sources.forEach((source) => { const tag = document.createElement("span"); tag.className = "source-tag"; tag.textContent = source; list.append(tag); });
  return list;
}
function section(eyebrow, title) {
  const node = document.createElement("section"); node.className = "evidence-section";
  const heading = document.createElement("div"); heading.className = "evidence-heading";
  const label = document.createElement("p"); label.className = "section-label"; label.textContent = eyebrow;
  const name = document.createElement("h4"); name.textContent = title; heading.append(label, name); node.append(heading); return node;
}
function appendJson(parent, title, value, open = false) {
  const details = document.createElement("details"); details.className = "json-details"; details.open = open;
  const summary = document.createElement("summary"); summary.textContent = title;
  const pre = document.createElement("pre"); pre.className = "json-content"; pre.textContent = formatJson(value);
  details.append(summary, pre); parent.append(details);
}
function emptyCopy(text) { const node = document.createElement("p"); node.className = "empty-copy"; node.textContent = text; return node; }
function statusBadge(status) { const node = document.createElement("span"); node.className = `status-badge status-${statusClass(status)}`; node.textContent = statusLabel(status); return node; }
function durationText(value) { const node = document.createElement("span"); node.className = "duration"; node.textContent = duration(value); return node; }
function statusClass(status) { const value = String(status || "").toLowerCase(); return value === "success" ? "success" : value === "error" ? "error" : value === "running" ? "running" : "neutral"; }
function statusLabel(status) { const labels = {success: "成功", error: "失败", running: "运行中", pending: "等待中", completed: "已完成"}; return labels[String(status || "").toLowerCase()] || String(status || "未知"); }
function duration(value) { const n = number(value); return n < 1000 ? `${n.toFixed(n < 10 ? 1 : 0)} ms` : `${(n / 1000).toFixed(2)} s`; }
function formatJson(value) { if (typeof value === "string") return value; try { return JSON.stringify(value, null, 2); } catch { return String(value); } }
function object(value) { return value && typeof value === "object" && !Array.isArray(value) ? value : {}; }
function array(value) { return Array.isArray(value) ? value : []; }
function text(...values) { return values.find((value) => typeof value === "string" && value.trim()) || ""; }
function textValue(value) { return typeof value === "string" ? value : formatJson(value ?? ""); }
function number(value) { const n = Number(value); return Number.isFinite(n) ? n : 0; }
function summaryValue(value, fallback) { const n = Number(value); return Number.isFinite(n) ? n : fallback; }
function uniqueSources(values) { return [...new Set(array(values).filter((value) => typeof value === "string" && value))]; }
window.FitLifeTrace = {normalizeTrace, renderRunSummary, renderTimeline, renderToolExchanges: renderTools, renderEvidence};



document.querySelectorAll(".empty-action").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt || "";
    form.requestSubmit();
  });
});


