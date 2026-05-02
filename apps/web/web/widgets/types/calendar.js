// Calendar widget — upcoming events from macOS Calendar (and DB events).
// Per-instance setting: max_events (default 12).
import { register } from "/static/widgets/index.js";

function fmtWhen(iso) {
  if (!iso) return "";
  const d = new Date(iso.replace(" ", "T") + (iso.includes("Z") ? "" : "Z"));
  if (isNaN(d.valueOf())) return iso;
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) {
    return d.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleString("ko-KR", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const MAX_OPTIONS = [5, 10, 15, 20, 30, 50];
const DEFAULT_MAX = 12;

register({
  type: "calendar",
  title: "일정",
  icon: "calendar",
  description: "다가오는 이벤트",

  mount(container, ctx) {
    let maxEvents = Number(ctx.settings?.max_events) || DEFAULT_MAX;
    let cachedEvents = [];

    container.innerHTML = `
      <div class="cal-toolbar">
        <label>최대 <select class="cal-max-select"></select> 개 표시</label>
      </div>
      <ul class="cal-list"></ul>
    `;
    const sel = container.querySelector(".cal-max-select");
    const listEl = container.querySelector(".cal-list");

    // Populate select options (include the current setting even if it's
    // not in the standard list, so existing configs round-trip).
    const opts = new Set([...MAX_OPTIONS, maxEvents]);
    for (const n of [...opts].sort((a, b) => a - b)) {
      const o = document.createElement("option");
      o.value = String(n);
      o.textContent = String(n);
      if (n === maxEvents) o.selected = true;
      sel.append(o);
    }

    sel.addEventListener("change", () => {
      maxEvents = Number(sel.value) || DEFAULT_MAX;
      ctx.updateSettings({ ...(ctx.settings || {}), max_events: maxEvents });
      render();
    });

    function render() {
      listEl.innerHTML = "";
      if (!cachedEvents.length) {
        listEl.innerHTML = `<li class="empty">예정된 일정이 없어요</li>`;
        return;
      }
      for (const e of cachedEvents.slice(0, maxEvents)) {
        const li = document.createElement("li");
        li.className = "cal-item";
        const head = document.createElement("div");
        head.className = "cal-when";
        head.textContent = fmtWhen(e.when_at);
        const body = document.createElement("div");
        body.className = "cal-title";
        body.textContent = e.title;
        li.append(head, body);
        if (e.calendar) {
          const m = document.createElement("div");
          m.className = "cal-meta";
          m.textContent = e.calendar;
          li.append(m);
        }
        listEl.append(li);
      }
    }

    async function refresh() {
      try {
        const r = await fetch("/api/widgets/calendar");
        if (!r.ok) {
          listEl.innerHTML = `<li class="empty">캘린더에 접근할 수 없어요</li>`;
          return;
        }
        cachedEvents = await r.json();
        render();
      } catch (err) {
        console.warn("calendar refresh failed", err);
      }
    }

    refresh();
    const interval = setInterval(refresh, 5 * 60 * 1000);
    return () => clearInterval(interval);
  },
});
