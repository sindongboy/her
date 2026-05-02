// Right column: memory probe + recent notes/people.
import { on } from "/static/state.js";
import { iconHTML } from "/static/icons.js";

const probeRecallEl = document.getElementById("probe-recall");
const recentNotesEl = document.getElementById("recent-notes");
const recentPeopleEl = document.getElementById("recent-people");

const KIND_META = {
  fact:    { label: "사실",   icon: "bulb" },
  note:    { label: "메모",   icon: "note" },
  event:   { label: "일정",   icon: "calendar" },
  session: { label: "관련 대화", icon: "history" },
};

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

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

  const head = document.createElement("div");
  head.className = "group-head";

  const pill = document.createElement("span");
  pill.className = "pill";
  pill.innerHTML = `<span class="icon">${iconHTML(meta.icon)}</span><span>${meta.label}</span>`;

  const count = document.createElement("span");
  count.className = "count";
  count.textContent = items.length;

  head.append(pill, count);
  wrap.append(head);

  for (const item of items) wrap.append(render(item));
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

function renderRecall(detail) {
  clear(probeRecallEl);

  const groups = [
    makeGroup("fact", detail.facts, (f) =>
      makeCard(
        `${f.predicate} → ${f.object}`,
        f.person_name ? `· ${f.person_name}` : "",
      ),
    ),
    makeGroup("note", detail.notes, (n) => makeCard(n.content, "")),
    makeGroup("event", detail.events, (e) => makeCard(e.title, e.when_at)),
    makeGroup("session", detail.sessions, (s) =>
      makeCard(
        s.summary || `세션 ${s.id}`,
        typeof s.score === "number" ? `score ${s.score.toFixed(2)}` : "",
      ),
    ),
  ].filter(Boolean);

  if (groups.length === 0) {
    probeRecallEl.append(emptyP("이번 답변엔 떠올린 기억이 없어요"));
    return;
  }
  for (const g of groups) probeRecallEl.append(g);
}

async function refreshSidebar() {
  try {
    const [notesR, peopleR] = await Promise.all([
      fetch("/api/memory/notes"),
      fetch("/api/memory/people"),
    ]);

    if (notesR.ok) {
      const notes = await notesR.json();
      clear(recentNotesEl);
      if (notes.length === 0) {
        recentNotesEl.append(emptyP("아직 메모가 없어요"));
      } else {
        for (const n of notes.slice(0, 8)) {
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
          recentNotesEl.append(li);
        }
      }
    }

    if (peopleR.ok) {
      const people = await peopleR.json();
      clear(recentPeopleEl);
      if (people.length === 0) {
        recentPeopleEl.append(emptyP("등록된 사람이 없어요"));
      } else {
        for (const p of people.slice(0, 12)) {
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
          recentPeopleEl.append(li);
        }
      }
    }
  } catch (err) {
    console.warn("memory sidebar refresh failed", err);
  }
}

document.addEventListener("her:recall", (ev) => renderRecall(ev.detail));
document.addEventListener("her:turn-complete", refreshSidebar);
on("sessionId", refreshSidebar);

refreshSidebar();
