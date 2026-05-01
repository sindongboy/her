// Right column: memory probe + recent notes/people.
import { on } from "/static/state.js";

const probeRecallEl = document.getElementById("probe-recall");
const recentNotesEl = document.getElementById("recent-notes");
const recentPeopleEl = document.getElementById("recent-people");

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function emptyP(text) {
  const p = document.createElement("p");
  p.className = "empty";
  p.textContent = text;
  return p;
}

function block(label, items, render) {
  if (!items || items.length === 0) return null;
  const wrap = document.createElement("div");
  wrap.className = "recall-block";
  const lab = document.createElement("div");
  lab.className = "label";
  lab.textContent = label;
  wrap.append(lab);
  for (const item of items) wrap.append(render(item));
  return wrap;
}

function card(title, secondary) {
  const c = document.createElement("div");
  c.className = "recall-card";
  const t = document.createElement("div");
  t.className = "primary";
  t.textContent = title;
  c.append(t);
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

  const blocks = [
    block("Facts", detail.facts, (f) =>
      card(
        `${f.predicate} → ${f.object}`,
        f.person_name ? `인물: ${f.person_name}` : "",
      ),
    ),
    block("Notes", detail.notes, (n) => card(n.content, "")),
    block("Events", detail.events, (e) => card(e.title, e.when_at)),
    block("Sessions", detail.sessions, (s) =>
      card(s.summary || `세션 ${s.id}`, `score ${s.score?.toFixed?.(2) ?? ""}`),
    ),
  ].filter(Boolean);

  if (blocks.length === 0) {
    probeRecallEl.append(emptyP("이번 답변에선 떠올린 기억이 없습니다."));
    return;
  }
  for (const b of blocks) probeRecallEl.append(b);
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
        recentNotesEl.append(emptyP("아직 메모가 없어요."));
      } else {
        for (const n of notes.slice(0, 8)) {
          const li = document.createElement("li");
          li.className = "probe-item";
          li.textContent = n.content;
          recentNotesEl.append(li);
        }
      }
    }
    if (peopleR.ok) {
      const people = await peopleR.json();
      clear(recentPeopleEl);
      if (people.length === 0) {
        recentPeopleEl.append(emptyP("등록된 사람이 없어요."));
      } else {
        for (const p of people.slice(0, 12)) {
          const li = document.createElement("li");
          li.className = "probe-item";
          const main = document.createElement("div");
          main.textContent = p.name;
          const meta = document.createElement("div");
          meta.className = "meta";
          meta.textContent = p.relation || "";
          li.append(main, meta);
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
