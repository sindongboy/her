// Notes widget — most recent active notes from the memory store.
import { register } from "/static/widgets/index.js";
import { on } from "/static/state.js";

register({
  type: "notes",
  title: "메모",
  icon: "note",
  description: "최근 메모 목록",

  mount(container) {
    container.innerHTML = `<ul class="probe-list"></ul>`;
    const listEl = container.querySelector(".probe-list");

    async function refresh() {
      try {
        const r = await fetch("/api/memory/notes");
        if (!r.ok) return;
        const notes = await r.json();
        listEl.innerHTML = "";
        if (notes.length === 0) {
          const li = document.createElement("li");
          li.className = "empty";
          li.textContent = "아직 메모가 없어요";
          listEl.append(li);
          return;
        }
        for (const n of notes.slice(0, 12)) {
          const li = document.createElement("li");
          li.className = "probe-item note";
          const main = document.createElement("div");
          main.textContent = n.content;
          li.append(main);
          if (Array.isArray(n.tags) && n.tags.length) {
            const tagWrap = document.createElement("div");
            tagWrap.className = "tags";
            for (const t of n.tags) {
              const tag = document.createElement("span");
              tag.className = "tag";
              tag.textContent = t;
              tagWrap.append(tag);
            }
            li.append(tagWrap);
          }
          listEl.append(li);
        }
      } catch (err) {
        console.warn("notes widget refresh failed", err);
      }
    }

    document.addEventListener("her:turn-complete", refresh);
    document.addEventListener("her:memory-added", refresh);
    const offSid = on("sessionId", refresh);
    refresh();

    return () => {
      document.removeEventListener("her:turn-complete", refresh);
      document.removeEventListener("her:memory-added", refresh);
      offSid?.();
    };
  },
});
