// Left column: session list + new-session button.
import { state, on, set } from "/static/state.js";

const listEl = document.getElementById("session-list");
const newBtn = document.getElementById("new-session");
const titleEl = document.getElementById("session-title");

let sessions = [];

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso.replace(" ", "T") + (iso.includes("Z") ? "" : "Z"));
  if (isNaN(d.valueOf())) return iso;
  return d.toLocaleString("ko-KR", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function render() {
  listEl.innerHTML = "";
  for (const s of sessions) {
    const li = document.createElement("li");
    li.className = "session-item" + (s.id === state.sessionId ? " active" : "");
    li.dataset.id = s.id;
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = s.title || s.summary || `세션 ${s.id}`;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = fmtTime(s.last_active_at);
    li.append(title, meta);
    li.addEventListener("click", () => set("sessionId", s.id));
    listEl.append(li);
  }
  const active = sessions.find((s) => s.id === state.sessionId);
  if (active) {
    titleEl.textContent = `— ${active.title || active.summary || `세션 ${active.id}`}`;
  } else {
    titleEl.textContent = "— 가족같이 곁에 있는 비서";
  }
}

async function refresh() {
  try {
    const r = await fetch("/api/sessions");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    sessions = await r.json();
    render();
  } catch (err) {
    console.warn("session refresh failed", err);
  }
}

async function createSession() {
  const r = await fetch("/api/sessions", { method: "POST" });
  if (!r.ok) {
    console.warn("session create failed", r.status);
    return;
  }
  const s = await r.json();
  sessions.unshift(s);
  set("sessionId", s.id);
  render();
}

newBtn.addEventListener("click", createSession);

on("sessionId", () => render());
document.addEventListener("her:turn-complete", refresh);

refresh();
