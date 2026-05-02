// People widget — known family members / relations.
import { register } from "/static/widgets/index.js";

register({
  type: "people",
  title: "사람",
  icon: "user",
  description: "기억하고 있는 사람들",

  mount(container) {
    container.innerHTML = `<ul class="probe-list"></ul>`;
    const listEl = container.querySelector(".probe-list");

    async function refresh() {
      try {
        const r = await fetch("/api/memory/people");
        if (!r.ok) return;
        const people = await r.json();
        listEl.innerHTML = "";
        if (people.length === 0) {
          const li = document.createElement("li");
          li.className = "empty";
          li.textContent = "등록된 사람이 없어요";
          listEl.append(li);
          return;
        }
        for (const p of people.slice(0, 16)) {
          const li = document.createElement("li");
          li.className = "probe-item person";
          const main = document.createElement("div");
          main.textContent = p.name;
          li.append(main);
          if (p.relation) {
            const meta = document.createElement("div");
            meta.className = "meta";
            meta.textContent = p.relation;
            li.append(meta);
          }
          listEl.append(li);
        }
      } catch (err) {
        console.warn("people widget refresh failed", err);
      }
    }

    document.addEventListener("her:turn-complete", refresh);
    document.addEventListener("her:memory-added", refresh);
    refresh();

    return () => {
      document.removeEventListener("her:turn-complete", refresh);
      document.removeEventListener("her:memory-added", refresh);
    };
  },
});
