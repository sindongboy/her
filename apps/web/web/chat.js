// Center column: WebSocket chat with token streaming.
import { state, on, set } from "/static/state.js";

const messagesEl = document.getElementById("messages");
const composer = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const orb = document.getElementById("orb");
const statusText = document.getElementById("status-text");
const welcomeEl = document.getElementById("welcome");

let socket = null;
let currentAssistant = null; // DOM node accumulating tokens
let thinkingEl = null;       // inline "Her is thinking" placeholder
let pendingTurn = false;

function showWelcome() {
  if (welcomeEl) welcomeEl.classList.remove("hidden");
}

function hideWelcome() {
  if (welcomeEl) welcomeEl.classList.add("hidden");
}

function setStatus(label, mode) {
  statusText.textContent = label;
  orb.className = `orb ${mode || "idle"}`;
}

function showThinking(label) {
  hideWelcome();
  if (!thinkingEl) {
    thinkingEl = document.createElement("article");
    thinkingEl.className = "message thinking";
    thinkingEl.innerHTML = `
      <span class="thinking-dots" aria-hidden="true"><span></span><span></span><span></span></span>
      <span class="thinking-label"></span>
    `;
    messagesEl.append(thinkingEl);
  }
  thinkingEl.querySelector(".thinking-label").textContent = label;
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideThinking() {
  if (thinkingEl) {
    thinkingEl.remove();
    thinkingEl = null;
  }
}

function renderMemoryAdded(detail) {
  hideWelcome();
  const el = document.createElement("article");
  el.className = "message memory-added";
  const facts = detail.facts || [];
  const notes = detail.notes || [];
  const parts = [];
  if (facts.length) {
    parts.push(
      ...facts.map(
        (f) =>
          `<div class="ma-item"><span class="ma-kind fact">사실</span><span class="ma-text">${
            f.person_name ? `<b>${escapeHTML(f.person_name)}</b> — ` : ""
          }${escapeHTML(f.predicate)} = ${escapeHTML(f.object)}</span></div>`,
      ),
    );
  }
  if (notes.length) {
    parts.push(
      ...notes.map(
        (n) =>
          `<div class="ma-item"><span class="ma-kind note">메모</span><span class="ma-text">${escapeHTML(n.content)}</span></div>`,
      ),
    );
  }
  el.innerHTML = `
    <div class="ma-head">✓ 기억에 추가했어요</div>
    <div class="ma-list">${parts.join("")}</div>
  `;
  messagesEl.append(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHTML(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderMessage(role, content) {
  hideWelcome();
  const el = document.createElement("article");
  el.className = `message ${role}`;
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = content;
  el.append(body);
  messagesEl.append(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return body;
}

function renderError(message) {
  hideWelcome();
  const el = document.createElement("article");
  el.className = "message error";
  el.textContent = `⚠ ${message}`;
  messagesEl.append(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function clearMessages() {
  // Remove only message articles, preserve the welcome node.
  for (const node of messagesEl.querySelectorAll(".message")) node.remove();
}

async function loadSessionMessages(sessionId) {
  clearMessages();
  if (!sessionId) {
    showWelcome();
    return;
  }
  try {
    const r = await fetch(`/api/sessions/${sessionId}/messages`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const msgs = await r.json();
    if (msgs.length === 0) {
      showWelcome();
    } else {
      hideWelcome();
      for (const m of msgs) renderMessage(m.role, m.content);
    }
  } catch (err) {
    renderError(`이전 메시지 로드 실패: ${err.message}`);
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws/chat`);

  socket.addEventListener("open", () => setStatus("연결됨", "idle"));
  socket.addEventListener("close", () => {
    setStatus("연결 끊김 — 재연결 중", "error");
    setTimeout(connect, 1500);
  });
  socket.addEventListener("error", () => setStatus("WS 오류", "error"));

  socket.addEventListener("message", (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }

    switch (msg.type) {
      case "hello":
        // schema_version handshake; ignored for now.
        break;

      case "memory_added": {
        renderMemoryAdded(msg);
        document.dispatchEvent(
          new CustomEvent("her:memory-added", { detail: msg }),
        );
        break;
      }

      case "recall": {
        const counts =
          (msg.facts?.length || 0) +
          (msg.notes?.length || 0) +
          (msg.events?.length || 0) +
          (msg.sessions?.length || 0);
        showThinking(
          counts > 0
            ? `${counts}개 기억 떠올림 — 생각 중`
            : "생각 중",
        );
        setStatus("생각 중", "thinking");
        document.dispatchEvent(
          new CustomEvent("her:recall", { detail: msg }),
        );
        break;
      }

      case "token":
        if (msg.session_id && msg.session_id !== state.sessionId) {
          set("sessionId", msg.session_id);
        }
        if (!currentAssistant) {
          hideThinking();
          currentAssistant = renderMessage("assistant", "");
          setStatus("답하는 중", "thinking");
        }
        currentAssistant.textContent += msg.text || "";
        messagesEl.scrollTop = messagesEl.scrollHeight;
        break;

      case "done":
        hideThinking();
        currentAssistant = null;
        pendingTurn = false;
        sendBtn.disabled = false;
        setStatus("준비됨", "idle");
        document.dispatchEvent(new CustomEvent("her:turn-complete"));
        break;

      case "error":
        hideThinking();
        renderError(msg.message || "알 수 없는 오류");
        currentAssistant = null;
        pendingTurn = false;
        sendBtn.disabled = false;
        setStatus("오류", "error");
        break;
    }
  });
}

function sendMessage(content) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    renderError("WS 가 아직 연결되지 않았어요. 잠시 후 다시 시도하세요.");
    return;
  }
  pendingTurn = true;
  sendBtn.disabled = true;
  renderMessage("user", content);
  showThinking("기억을 살펴보는 중");
  setStatus("기억 검색 중", "thinking");
  socket.send(
    JSON.stringify({
      type: "message",
      content,
      session_id: state.sessionId ?? null,
    }),
  );
}

composer.addEventListener("submit", (ev) => {
  ev.preventDefault();
  if (pendingTurn) return;
  const value = input.value.trim();
  if (!value) return;
  input.value = "";
  input.style.height = "auto";
  sendMessage(value);
});

input.addEventListener("keydown", (ev) => {
  // Don't submit while an IME composition is in progress (Korean Hangul,
  // Japanese kana, etc.). Without this guard, Enter triggers submit AND the
  // IME's finalize, so the just-committed character gets re-inserted into
  // the cleared textarea. keyCode 229 covers older browsers that don't yet
  // populate isComposing on the keydown event.
  if (ev.isComposing || ev.keyCode === 229) return;
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    composer.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
});

// Suggestion chips on the welcome screen prefill the input.
if (welcomeEl) {
  welcomeEl.addEventListener("click", (ev) => {
    const chip = ev.target.closest(".suggest-chip");
    if (!chip) return;
    input.value = chip.dataset.prompt || chip.textContent.trim();
    input.dispatchEvent(new Event("input"));
    input.focus();
  });
}

on("sessionId", (sid) => {
  // Don't reload mid-turn. The first token of a brand-new session sets the
  // sessionId so subsequent turns reuse it; reloading at that moment would
  // race with _persist_turn and duplicate the streaming assistant bubble.
  if (pendingTurn) return;
  loadSessionMessages(sid);
});

connect();
