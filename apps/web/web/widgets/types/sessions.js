// Sessions widget — list of recent chat sessions; click loads a session.
// Hover any item to reveal ✏ rename and ✕ archive actions.
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

    function escape(s) {
      return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function render() {
      listEl.innerHTML = "";
      for (const s of sessions) {
        const li = document.createElement("li");
        li.className = "session-item" + (s.id === state.sessionId ? " active" : "");
        li.dataset.id = s.id;
        const titleText = s.title || s.summary || `세션 ${s.id}`;
        li.innerHTML = `
          <div class="session-main">
            <div class="title">${escape(titleText)}</div>
            <div class="meta">${escape(fmtTime(s.last_active_at))}</div>
          </div>
          <div class="session-actions">
            <button data-action="rename" title="제목 편집"><span class="icon">${iconHTML("pencil")}</span></button>
            <button data-action="archive" title="삭제 (아카이브)"><span class="icon">${iconHTML("x")}</span></button>
          </div>
        `;
        const main = li.querySelector(".session-main");
        main.addEventListener("click", () => set("sessionId", s.id));
        li.querySelector("[data-action='rename']").addEventListener("click", (ev) => {
          ev.stopPropagation();
          startRename(li, s);
        });
        li.querySelector("[data-action='archive']").addEventListener("click", (ev) => {
          ev.stopPropagation();
          archiveSession(s);
        });
        listEl.append(li);
      }
    }

    function startRename(li, s) {
      const main = li.querySelector(".session-main");
      const current = s.title || s.summary || `세션 ${s.id}`;
      main.innerHTML = `
        <input class="session-rename-input" value="${escape(current)}" />
      `;
      const input = main.querySelector("input");
      input.focus();
      input.select();
      const commit = async () => {
        const next = input.value.trim();
        if (next && next !== current) {
          try {
            const r = await fetch(`/api/sessions/${s.id}`, {
              method: "PATCH",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ title: next }),
            });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            s.title = next;
          } catch (err) {
            console.warn("rename failed", err);
          }
        }
        await refresh();
      };
      const cancel = () => { render(); };
      input.addEventListener("blur", commit);
      input.addEventListener("keydown", (ev) => {
        if (ev.isComposing || ev.keyCode === 229) return;
        if (ev.key === "Enter") { ev.preventDefault(); input.blur(); }
        else if (ev.key === "Escape") { ev.preventDefault(); cancel(); }
      });
    }

    async function archiveSession(s) {
      const ok = confirm(`"${s.title || s.summary || `세션 ${s.id}`}" 삭제하시겠어요?\n(아카이브로 이동 — DB 에는 남아있음)`);
      if (!ok) return;
      try {
        const r = await fetch(`/api/sessions/${s.id}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        if (state.sessionId === s.id) set("sessionId", null);
      } catch (err) {
        console.warn("archive failed", err);
      }
      await refresh();
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
