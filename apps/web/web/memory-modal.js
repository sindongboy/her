// Memory browser modal — opens from the topbar 🧠 기억 button.
// Four tabs (사실 / 메모 / 사람 / 일정) with full CRUD + archived toggle.
import { iconHTML } from "/static/icons.js";

const TABS = [
  { id: "facts",  label: "사실", icon: "bulb" },
  { id: "notes",  label: "메모", icon: "note" },
  { id: "people", label: "사람", icon: "user" },
  { id: "events", label: "일정", icon: "calendar" },
];

let currentTab = "facts";
let showArchived = false;
let modalEl = null;
let peopleCache = []; // for facts subject lookup

// ── boot ───────────────────────────────────────────────────────────────

function boot() {
  const btn = document.getElementById("memory-btn");
  if (!btn) return;
  btn.addEventListener("click", openModal);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}

// ── open / close ───────────────────────────────────────────────────────

function openModal() {
  if (modalEl) return;
  modalEl = document.createElement("div");
  modalEl.className = "memory-modal";
  modalEl.innerHTML = `
    <div class="mm-overlay"></div>
    <div class="mm-panel" role="dialog" aria-label="기억 관리">
      <header class="mm-head">
        <span class="icon">${iconHTML("brain")}</span>
        <h2>기억 관리</h2>
        <div class="mm-archived-toggle">
          <label>
            <input type="checkbox" id="mm-archived"> 아카이브 포함
          </label>
        </div>
        <button class="mm-close" aria-label="닫기"><span class="icon">${iconHTML("x")}</span></button>
      </header>
      <nav class="mm-tabs">
        ${TABS.map(t => `
          <button data-tab="${t.id}" class="mm-tab ${t.id === currentTab ? "active" : ""}">
            <span class="icon">${iconHTML(t.icon)}</span>
            <span>${t.label}</span>
          </button>
        `).join("")}
      </nav>
      <div class="mm-body" id="mm-body"></div>
    </div>
  `;
  document.body.append(modalEl);
  document.body.classList.add("mm-open");

  modalEl.querySelector(".mm-overlay").addEventListener("click", closeModal);
  modalEl.querySelector(".mm-close").addEventListener("click", closeModal);
  modalEl.querySelector("#mm-archived").addEventListener("change", (ev) => {
    showArchived = ev.target.checked;
    renderTab();
  });
  modalEl.querySelectorAll(".mm-tab").forEach((b) => {
    b.addEventListener("click", () => {
      currentTab = b.dataset.tab;
      modalEl.querySelectorAll(".mm-tab").forEach((x) => x.classList.toggle("active", x === b));
      renderTab();
    });
  });

  document.addEventListener("keydown", onKeydown);

  renderTab();
}

function closeModal() {
  if (!modalEl) return;
  modalEl.remove();
  modalEl = null;
  document.body.classList.remove("mm-open");
  document.removeEventListener("keydown", onKeydown);
}

function onKeydown(ev) {
  if (ev.key === "Escape") closeModal();
}

// ── tab rendering ──────────────────────────────────────────────────────

async function renderTab() {
  const body = modalEl?.querySelector("#mm-body");
  if (!body) return;
  body.innerHTML = `<div class="mm-loading">불러오는 중...</div>`;
  try {
    if (currentTab === "facts")  await renderFacts(body);
    else if (currentTab === "notes")  await renderNotes(body);
    else if (currentTab === "people") await renderPeople(body);
    else if (currentTab === "events") await renderEvents(body);
  } catch (err) {
    body.innerHTML = `<div class="mm-error">${escapeHTML(String(err.message || err))}</div>`;
  }
}

async function refreshPeopleCache() {
  try {
    const r = await fetch("/api/memory/people");
    if (r.ok) peopleCache = await r.json();
  } catch {}
}

// ── facts ──────────────────────────────────────────────────────────────

async function renderFacts(body) {
  await refreshPeopleCache();
  const r = await fetch(`/api/memory/facts?include_archived=${showArchived}`);
  const facts = r.ok ? await r.json() : [];

  body.innerHTML = `
    <div class="mm-toolbar">
      <button class="mm-add-btn" type="button"><span class="icon">${iconHTML("plus")}</span><span>사실 추가</span></button>
    </div>
    <div class="mm-add-form" hidden>
      <select class="mm-input mm-fact-person"><option value="">사람 선택</option></select>
      <input class="mm-input mm-fact-pred" placeholder="술어 (예: 좋아한다)" />
      <input class="mm-input mm-fact-obj"  placeholder="대상 (예: 단호박 케이크)" />
      <input class="mm-input mm-fact-conf" type="number" min="0" max="1" step="0.05" value="1.0" placeholder="신뢰도 0..1" />
      <button class="mm-save" type="button">저장</button>
    </div>
    <div class="mm-list mm-facts"></div>
  `;
  const list = body.querySelector(".mm-facts");
  if (!facts.length) {
    list.innerHTML = `<div class="mm-empty">등록된 사실이 없어요</div>`;
  } else {
    list.innerHTML = facts.map((f) => factRowHTML(f)).join("");
    list.querySelectorAll("[data-action='archive']").forEach((b) =>
      b.addEventListener("click", () => archive("facts", b.dataset.id)));
    list.querySelectorAll("[data-action='restore']").forEach((b) =>
      b.addEventListener("click", () => restore("facts", b.dataset.id)));
    list.querySelectorAll("[data-action='edit']").forEach((b) =>
      b.addEventListener("click", () => editFact(b.closest(".mm-row"), b.dataset.id, facts)));
  }

  // wire add form
  const addBtn = body.querySelector(".mm-add-btn");
  const form = body.querySelector(".mm-add-form");
  const personSel = form.querySelector(".mm-fact-person");
  for (const p of peopleCache) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.name + (p.relation ? ` (${p.relation})` : "");
    personSel.append(opt);
  }
  addBtn.addEventListener("click", () => form.toggleAttribute("hidden"));
  form.querySelector(".mm-save").addEventListener("click", async () => {
    const payload = {
      subject_person_id: parseInt(personSel.value, 10),
      predicate: form.querySelector(".mm-fact-pred").value.trim(),
      object: form.querySelector(".mm-fact-obj").value.trim(),
      confidence: parseFloat(form.querySelector(".mm-fact-conf").value) || 1.0,
    };
    if (!payload.subject_person_id || !payload.predicate || !payload.object) {
      alert("사람·술어·대상 모두 입력해 주세요");
      return;
    }
    const r2 = await fetch("/api/memory/facts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r2.ok) { alert("저장 실패"); return; }
    await renderTab();
  });
}

function factRowHTML(f) {
  const isArchived = !!f.archived_at;
  return `
    <div class="mm-row mm-fact ${isArchived ? "archived" : ""}" data-id="${f.id}">
      <div class="mm-fact-main">
        <span class="person">${escapeHTML(f.person_name || "?")}</span>
        <span class="pred">${escapeHTML(f.predicate)}</span>
        <span class="arrow">→</span>
        <span class="obj">${escapeHTML(f.object)}</span>
      </div>
      <div class="mm-fact-meta">
        <span class="conf">신뢰도 ${(f.confidence ?? 0).toFixed(2)}</span>
        ${f.source_session_id ? `<span class="src">세션 #${f.source_session_id}</span>` : ""}
        ${isArchived ? `<span class="archived-tag">아카이브</span>` : ""}
      </div>
      <div class="mm-actions">
        ${isArchived
          ? `<button class="mm-icon-btn" data-action="restore" data-id="${f.id}" title="복원"><span class="icon">${iconHTML("refresh")}</span></button>`
          : `
            <button class="mm-icon-btn" data-action="edit" data-id="${f.id}" title="편집"><span class="icon">${iconHTML("pencil")}</span></button>
            <button class="mm-icon-btn" data-action="archive" data-id="${f.id}" title="아카이브"><span class="icon">${iconHTML("x")}</span></button>
          `}
      </div>
    </div>
  `;
}

function editFact(rowEl, id, facts) {
  const f = facts.find((x) => x.id === parseInt(id, 10));
  if (!f) return;
  rowEl.classList.add("editing");
  rowEl.querySelector(".mm-fact-main").innerHTML = `
    <input class="mm-input mm-edit-pred" value="${escapeHTML(f.predicate)}" />
    <span class="arrow">→</span>
    <input class="mm-input mm-edit-obj"  value="${escapeHTML(f.object)}" />
    <input class="mm-input mm-edit-conf" type="number" min="0" max="1" step="0.05" value="${f.confidence ?? 1.0}" style="width:80px" />
  `;
  rowEl.querySelector(".mm-actions").innerHTML = `
    <button class="mm-save-btn" data-id="${id}">저장</button>
    <button class="mm-cancel-btn">취소</button>
  `;
  rowEl.querySelector(".mm-save-btn").addEventListener("click", async () => {
    const payload = {
      predicate: rowEl.querySelector(".mm-edit-pred").value.trim(),
      object:    rowEl.querySelector(".mm-edit-obj").value.trim(),
      confidence: parseFloat(rowEl.querySelector(".mm-edit-conf").value) || 1.0,
    };
    const r = await fetch(`/api/memory/facts/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) { alert("저장 실패"); return; }
    await renderTab();
  });
  rowEl.querySelector(".mm-cancel-btn").addEventListener("click", renderTab);
}

// ── notes ──────────────────────────────────────────────────────────────

async function renderNotes(body) {
  const r = await fetch(`/api/memory/notes?include_archived=${showArchived}`);
  const notes = r.ok ? await r.json() : [];

  body.innerHTML = `
    <div class="mm-toolbar">
      <button class="mm-add-btn" type="button"><span class="icon">${iconHTML("plus")}</span><span>메모 추가</span></button>
    </div>
    <div class="mm-add-form" hidden>
      <input class="mm-input mm-note-content" placeholder="메모 내용" />
      <input class="mm-input mm-note-tags" placeholder="태그 (쉼표로 구분: routine, todo)" />
      <button class="mm-save" type="button">저장</button>
    </div>
    <div class="mm-list mm-notes"></div>
  `;
  const list = body.querySelector(".mm-notes");
  if (!notes.length) {
    list.innerHTML = `<div class="mm-empty">등록된 메모가 없어요</div>`;
  } else {
    list.innerHTML = notes.map((n) => noteRowHTML(n)).join("");
    list.querySelectorAll("[data-action='archive']").forEach((b) =>
      b.addEventListener("click", () => archive("notes", b.dataset.id)));
    list.querySelectorAll("[data-action='restore']").forEach((b) =>
      b.addEventListener("click", () => restore("notes", b.dataset.id)));
    list.querySelectorAll("[data-action='edit']").forEach((b) =>
      b.addEventListener("click", () => editNote(b.closest(".mm-row"), b.dataset.id, notes)));
  }

  const addBtn = body.querySelector(".mm-add-btn");
  const form = body.querySelector(".mm-add-form");
  addBtn.addEventListener("click", () => form.toggleAttribute("hidden"));
  form.querySelector(".mm-save").addEventListener("click", async () => {
    const content = form.querySelector(".mm-note-content").value.trim();
    if (!content) { alert("내용을 입력해 주세요"); return; }
    const tagsRaw = form.querySelector(".mm-note-tags").value.trim();
    const tags = tagsRaw ? tagsRaw.split(",").map((s) => s.trim()).filter(Boolean) : [];
    const r2 = await fetch("/api/memory/notes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, tags }),
    });
    if (!r2.ok) { alert("저장 실패"); return; }
    await renderTab();
  });
}

function noteRowHTML(n) {
  const archived = !!n.archived_at;
  return `
    <div class="mm-row mm-note ${archived ? "archived" : ""}" data-id="${n.id}">
      <div class="mm-note-main">
        <div class="content">${escapeHTML(n.content)}</div>
        ${(n.tags && n.tags.length) ? `<div class="tags">${n.tags.map((t) => `<span class="tag">${escapeHTML(t)}</span>`).join("")}</div>` : ""}
      </div>
      <div class="mm-actions">
        ${archived
          ? `<button class="mm-icon-btn" data-action="restore" data-id="${n.id}" title="복원"><span class="icon">${iconHTML("refresh")}</span></button>`
          : `
            <button class="mm-icon-btn" data-action="edit" data-id="${n.id}" title="편집"><span class="icon">${iconHTML("pencil")}</span></button>
            <button class="mm-icon-btn" data-action="archive" data-id="${n.id}" title="아카이브"><span class="icon">${iconHTML("x")}</span></button>
          `}
      </div>
    </div>
  `;
}

function editNote(rowEl, id, notes) {
  const n = notes.find((x) => x.id === parseInt(id, 10));
  if (!n) return;
  rowEl.classList.add("editing");
  rowEl.querySelector(".mm-note-main").innerHTML = `
    <input class="mm-input mm-edit-content" value="${escapeHTML(n.content)}" />
    <input class="mm-input mm-edit-tags" value="${escapeHTML((n.tags || []).join(", "))}" />
  `;
  rowEl.querySelector(".mm-actions").innerHTML = `
    <button class="mm-save-btn" data-id="${id}">저장</button>
    <button class="mm-cancel-btn">취소</button>
  `;
  rowEl.querySelector(".mm-save-btn").addEventListener("click", async () => {
    const tagsRaw = rowEl.querySelector(".mm-edit-tags").value.trim();
    const payload = {
      content: rowEl.querySelector(".mm-edit-content").value.trim(),
      tags: tagsRaw ? tagsRaw.split(",").map((s) => s.trim()).filter(Boolean) : [],
    };
    const r = await fetch(`/api/memory/notes/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) { alert("저장 실패"); return; }
    await renderTab();
  });
  rowEl.querySelector(".mm-cancel-btn").addEventListener("click", renderTab);
}

// ── people ─────────────────────────────────────────────────────────────

async function renderPeople(body) {
  const r = await fetch(`/api/memory/people?include_archived=${showArchived}`);
  const people = r.ok ? await r.json() : [];

  body.innerHTML = `
    <div class="mm-toolbar">
      <button class="mm-add-btn" type="button"><span class="icon">${iconHTML("plus")}</span><span>사람 추가</span></button>
    </div>
    <div class="mm-add-form" hidden>
      <input class="mm-input mm-person-name" placeholder="이름 (예: 어머니)" />
      <input class="mm-input mm-person-relation" placeholder="관계 (예: mother)" />
      <input class="mm-input mm-person-birthday" placeholder="생일 YYYY-MM-DD 또는 MM-DD" />
      <button class="mm-save" type="button">저장</button>
    </div>
    <div class="mm-list mm-people"></div>
  `;
  const list = body.querySelector(".mm-people");
  if (!people.length) {
    list.innerHTML = `<div class="mm-empty">등록된 사람이 없어요</div>`;
  } else {
    list.innerHTML = people.map((p) => personRowHTML(p)).join("");
    list.querySelectorAll("[data-action='archive']").forEach((b) =>
      b.addEventListener("click", () => archive("people", b.dataset.id)));
    list.querySelectorAll("[data-action='restore']").forEach((b) =>
      b.addEventListener("click", () => restore("people", b.dataset.id)));
    list.querySelectorAll("[data-action='edit']").forEach((b) =>
      b.addEventListener("click", () => editPerson(b.closest(".mm-row"), b.dataset.id, people)));
  }

  const addBtn = body.querySelector(".mm-add-btn");
  const form = body.querySelector(".mm-add-form");
  addBtn.addEventListener("click", () => form.toggleAttribute("hidden"));
  form.querySelector(".mm-save").addEventListener("click", async () => {
    const name = form.querySelector(".mm-person-name").value.trim();
    if (!name) { alert("이름을 입력해 주세요"); return; }
    const r2 = await fetch("/api/memory/people", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        relation: form.querySelector(".mm-person-relation").value.trim() || null,
        birthday: form.querySelector(".mm-person-birthday").value.trim() || null,
      }),
    });
    if (!r2.ok) { alert("저장 실패"); return; }
    await renderTab();
  });
}

function personRowHTML(p) {
  const archived = !!p.archived_at;
  return `
    <div class="mm-row mm-person ${archived ? "archived" : ""}" data-id="${p.id}">
      <div class="mm-person-main">
        <div class="name">${escapeHTML(p.name)}</div>
        <div class="meta">
          ${p.relation ? `<span class="rel">${escapeHTML(p.relation)}</span>` : ""}
          ${p.birthday ? `<span class="bday">🎂 ${escapeHTML(p.birthday)}</span>` : ""}
        </div>
      </div>
      <div class="mm-actions">
        ${archived
          ? `<button class="mm-icon-btn" data-action="restore" data-id="${p.id}" title="복원"><span class="icon">${iconHTML("refresh")}</span></button>`
          : `
            <button class="mm-icon-btn" data-action="edit" data-id="${p.id}" title="편집"><span class="icon">${iconHTML("pencil")}</span></button>
            <button class="mm-icon-btn" data-action="archive" data-id="${p.id}" title="아카이브"><span class="icon">${iconHTML("x")}</span></button>
          `}
      </div>
    </div>
  `;
}

function editPerson(rowEl, id, people) {
  const p = people.find((x) => x.id === parseInt(id, 10));
  if (!p) return;
  rowEl.classList.add("editing");
  rowEl.querySelector(".mm-person-main").innerHTML = `
    <input class="mm-input mm-edit-name" value="${escapeHTML(p.name)}" />
    <input class="mm-input mm-edit-relation" value="${escapeHTML(p.relation || "")}" placeholder="관계" />
    <input class="mm-input mm-edit-birthday" value="${escapeHTML(p.birthday || "")}" placeholder="생일" />
  `;
  rowEl.querySelector(".mm-actions").innerHTML = `
    <button class="mm-save-btn" data-id="${id}">저장</button>
    <button class="mm-cancel-btn">취소</button>
  `;
  rowEl.querySelector(".mm-save-btn").addEventListener("click", async () => {
    const payload = {
      name: rowEl.querySelector(".mm-edit-name").value.trim(),
      relation: rowEl.querySelector(".mm-edit-relation").value.trim() || null,
      birthday: rowEl.querySelector(".mm-edit-birthday").value.trim() || null,
    };
    const r = await fetch(`/api/memory/people/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) { alert("저장 실패"); return; }
    await renderTab();
  });
  rowEl.querySelector(".mm-cancel-btn").addEventListener("click", renderTab);
}

// ── events ─────────────────────────────────────────────────────────────

async function renderEvents(body) {
  await refreshPeopleCache();
  const r = await fetch(`/api/memory/events?include_archived=${showArchived}`);
  const events = r.ok ? await r.json() : [];

  body.innerHTML = `
    <div class="mm-toolbar">
      <button class="mm-add-btn" type="button"><span class="icon">${iconHTML("plus")}</span><span>일정 추가</span></button>
    </div>
    <div class="mm-add-form" hidden>
      <input class="mm-input mm-event-title" placeholder="제목" />
      <input class="mm-input mm-event-type" placeholder="종류 (예: birthday, appointment)" />
      <input class="mm-input mm-event-when" placeholder="언제 YYYY-MM-DDTHH:MM" />
      <select class="mm-input mm-event-person"><option value="">관련 사람 (선택)</option></select>
      <button class="mm-save" type="button">저장</button>
    </div>
    <div class="mm-list mm-events"></div>
  `;
  const personSel = body.querySelector(".mm-event-person");
  for (const p of peopleCache) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.name;
    personSel.append(opt);
  }

  const list = body.querySelector(".mm-events");
  if (!events.length) {
    list.innerHTML = `<div class="mm-empty">등록된 일정이 없어요</div>`;
  } else {
    list.innerHTML = events.map((e) => eventRowHTML(e)).join("");
    list.querySelectorAll("[data-action='archive']").forEach((b) =>
      b.addEventListener("click", () => archive("events", b.dataset.id)));
    list.querySelectorAll("[data-action='restore']").forEach((b) =>
      b.addEventListener("click", () => restore("events", b.dataset.id)));
  }

  const addBtn = body.querySelector(".mm-add-btn");
  const form = body.querySelector(".mm-add-form");
  addBtn.addEventListener("click", () => form.toggleAttribute("hidden"));
  form.querySelector(".mm-save").addEventListener("click", async () => {
    const payload = {
      title: form.querySelector(".mm-event-title").value.trim(),
      type:  form.querySelector(".mm-event-type").value.trim() || "general",
      when_at: form.querySelector(".mm-event-when").value.trim(),
      person_id: parseInt(personSel.value, 10) || null,
    };
    if (!payload.title || !payload.when_at) { alert("제목과 시각을 입력해 주세요"); return; }
    const r2 = await fetch("/api/memory/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r2.ok) { alert("저장 실패"); return; }
    await renderTab();
  });
}

function eventRowHTML(e) {
  const archived = !!e.archived_at;
  return `
    <div class="mm-row mm-event ${archived ? "archived" : ""}" data-id="${e.id}">
      <div class="mm-event-main">
        <div class="title">${escapeHTML(e.title)}</div>
        <div class="meta">
          <span class="when">${escapeHTML(e.when_at)}</span>
          ${e.type ? `<span class="type">${escapeHTML(e.type)}</span>` : ""}
          ${e.recurrence ? `<span class="recur">${escapeHTML(e.recurrence)}</span>` : ""}
        </div>
      </div>
      <div class="mm-actions">
        ${archived
          ? `<button class="mm-icon-btn" data-action="restore" data-id="${e.id}" title="복원"><span class="icon">${iconHTML("refresh")}</span></button>`
          : `<button class="mm-icon-btn" data-action="archive" data-id="${e.id}" title="아카이브"><span class="icon">${iconHTML("x")}</span></button>`}
      </div>
    </div>
  `;
}

// ── shared archive / restore ───────────────────────────────────────────

async function archive(kind, id) {
  if (!confirm("아카이브하시겠어요? (복원 가능)")) return;
  const r = await fetch(`/api/memory/${kind}/${id}`, { method: "DELETE" });
  if (!r.ok) { alert("실패"); return; }
  await renderTab();
}

async function restore(kind, id) {
  const r = await fetch(`/api/memory/${kind}/${id}/restore`, { method: "POST" });
  if (!r.ok) { alert("실패"); return; }
  await renderTab();
}

// ── helpers ────────────────────────────────────────────────────────────

function escapeHTML(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
