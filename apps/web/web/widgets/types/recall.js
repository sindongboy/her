// Recall widget — shows what Her remembered for the current turn.
// Subscribes to the 'her:recall' event dispatched by chat.js on every turn.
import { register } from "/static/widgets/index.js";
import { iconHTML } from "/static/icons.js";

const KIND_META = {
  fact:    { label: "사실",     icon: "bulb" },
  note:    { label: "메모",     icon: "note" },
  event:   { label: "일정",     icon: "calendar" },
  session: { label: "관련 대화", icon: "history" },
};

function emptyP(text) {
  const p = document.createElement("p");
  p.className = "empty";
  p.textContent = text;
  return p;
}

function makeGroup(kind, items, render) {
  if (!items || items.length === 0) return null;
  const meta = KIND_META[kind];
  const wrap = document.createElement("div");
  wrap.className = "recall-group";
  wrap.dataset.kind = kind;
  wrap.innerHTML = `
    <div class="group-head">
      <span class="pill"><span class="icon">${iconHTML(meta.icon)}</span><span>${meta.label}</span></span>
      <span class="count">${items.length}</span>
    </div>
  `;
  for (const it of items) wrap.append(render(it));
  return wrap;
}

function makeCard(primary, secondary) {
  const c = document.createElement("div");
  c.className = "recall-card";
  const p = document.createElement("div");
  p.textContent = primary;
  c.append(p);
  if (secondary) {
    const s = document.createElement("div");
    s.className = "secondary";
    s.textContent = secondary;
    c.append(s);
  }
  return c;
}

register({
  type: "recall",
  title: "메모리 회상",
  icon: "sparkles",
  description: "이번 답변에 떠올린 기억",

  mount(container) {
    container.innerHTML = `<div class="probe-recall"></div>`;
    const root = container.querySelector(".probe-recall");
    root.append(emptyP("질문을 보내면 채워집니다"));

    function onRecall(ev) {
      const detail = ev.detail || {};
      root.innerHTML = "";
      const groups = [
        makeGroup("fact",    detail.facts,    (f) => makeCard(`${f.predicate} → ${f.object}`, f.person_name ? `· ${f.person_name}` : "")),
        makeGroup("note",    detail.notes,    (n) => makeCard(n.content, "")),
        makeGroup("event",   detail.events,   (e) => makeCard(e.title, e.when_at)),
        makeGroup("session", detail.sessions, (s) => makeCard(s.summary || `세션 ${s.id}`, typeof s.score === "number" ? `score ${s.score.toFixed(2)}` : "")),
      ].filter(Boolean);

      if (groups.length === 0) {
        root.append(emptyP("이번 답변엔 떠올린 기억이 없어요"));
        return;
      }
      for (const g of groups) root.append(g);
    }

    document.addEventListener("her:recall", onRecall);
    return () => document.removeEventListener("her:recall", onRecall);
  },
});
