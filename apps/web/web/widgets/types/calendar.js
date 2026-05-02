// Calendar widget — upcoming events from macOS Calendar (and DB events).
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

register({
  type: "calendar",
  title: "일정",
  icon: "calendar",
  description: "다가오는 이벤트",

  mount(container) {
    container.innerHTML = `<ul class="cal-list"></ul>`;
    const listEl = container.querySelector(".cal-list");

    async function refresh() {
      try {
        const r = await fetch("/api/widgets/calendar");
        if (!r.ok) {
          listEl.innerHTML = `<li class="empty">캘린더에 접근할 수 없어요</li>`;
          return;
        }
        const events = await r.json();
        listEl.innerHTML = "";
        if (!events.length) {
          listEl.innerHTML = `<li class="empty">예정된 일정이 없어요</li>`;
          return;
        }
        for (const e of events.slice(0, 12)) {
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
      } catch (err) {
        console.warn("calendar refresh failed", err);
      }
    }

    refresh();
    const interval = setInterval(refresh, 5 * 60 * 1000);
    return () => clearInterval(interval);
  },
});
