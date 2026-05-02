// Sessions widget — list of recent chat sessions; click loads a session.
import { register } from "/static/widgets/index.js";
import { state, on, set } from "/static/state.js";
import { iconHTML } from "/static/icons.js";

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

register({
  type: "sessions",
  title: "대화",
  icon: "message",
  description: "최근 채팅 세션 목록",

  mount(container) {
    container.innerHTML = `
      <ul class="session-list" role="list"></ul>
      <button type="button" class="session-new-btn"><span class="icon">${iconHTML("plus")}</span><span>새 대화</span></button>
    `;
    const listEl = container.querySelector(".session-list");
    const newBtn = container.querySelector(".session-new-btn");

    let sessions = [];

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
    }

    async function refresh() {
      try {
        const r = await fetch("/api/sessions");
        if (!r.ok) return;
        sessions = await r.json();
        render();
      } catch (err) {
        console.warn("sessions refresh failed", err);
      }
    }

    async function createSession() {
      try {
        const r = await fetch("/api/sessions", { method: "POST" });
        if (!r.ok) return;
        const s = await r.json();
        sessions.unshift(s);
        set("sessionId", s.id);
        render();
      } catch (err) {
        console.warn("session create failed", err);
      }
    }

    newBtn.addEventListener("click", createSession);
    const offSid = on("sessionId", render);
    document.addEventListener("her:turn-complete", refresh);

    refresh();

    return () => {
      offSid?.();
      document.removeEventListener("her:turn-complete", refresh);
    };
  },
});
