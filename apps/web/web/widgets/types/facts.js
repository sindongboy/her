// Facts widget — active facts from the memory store, grouped per person.
import { register } from "/static/widgets/index.js";
import { on } from "/static/state.js";

register({
  type: "facts",
  title: "사실",
  icon: "bulb",
  description: "사람에 묶인 안정적인 사실 목록",

  mount(container) {
    container.innerHTML = `<ul class="probe-list facts-list"></ul>`;
    const listEl = container.querySelector(".facts-list");

    async function refresh() {
      try {
        const r = await fetch("/api/memory/facts");
        if (!r.ok) return;
        const facts = await r.json();
        listEl.innerHTML = "";

        if (facts.length === 0) {
          const li = document.createElement("li");
          li.className = "empty";
          li.textContent = "등록된 사실이 없어요";
          listEl.append(li);
          return;
        }

        // Group by person
        const groups = new Map();
        for (const f of facts) {
          const key = f.person_name || "(이름 없음)";
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key).push(f);
        }

        for (const [name, group] of groups) {
          const head = document.createElement("li");
          head.className = "facts-person-head";
          head.textContent = name;
          listEl.append(head);

          for (const f of group.slice(0, 6)) {
            const li = document.createElement("li");
            li.className = "probe-item fact";
            const main = document.createElement("div");
            main.innerHTML = `<span class="pred">${escape(f.predicate)}</span> <span class="arrow">→</span> <span class="obj">${escape(f.object)}</span>`;
            li.append(main);
            if (f.confidence != null && f.confidence < 1.0) {
              const meta = document.createElement("div");
              meta.className = "meta";
              meta.textContent = `신뢰도 ${f.confidence.toFixed(2)}`;
              li.append(meta);
            }
            listEl.append(li);
          }
        }
      } catch (err) {
        console.warn("facts widget refresh failed", err);
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

function escape(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
