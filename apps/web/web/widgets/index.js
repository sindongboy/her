// Widget framework — registry, panel rendering, localStorage config.
// Each widget type registers a definition with `register({...})`. The
// framework owns add/remove/reorder UI and persists layout in
// localStorage 'her.widgets'.

import { iconHTML } from "/static/icons.js";

const REGISTRY = new Map();
const CONFIG_KEY = "her.widgets";

const DEFAULT_CONFIG = {
  schema: 1,
  panels: { left: ["sessions"], right: ["recall", "facts", "notes", "people"] },
  settings: {},
};

let mounted = []; // {type, panel, root, cleanup}

// ── Public API ─────────────────────────────────────────────────────────

export function register(def) {
  if (!def || !def.type) throw new Error("widget def needs a type");
  REGISTRY.set(def.type, def);
}

export function listDefs() {
  return [...REGISTRY.values()];
}

export function loadConfig() {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    if (!raw) return clone(DEFAULT_CONFIG);
    const cfg = JSON.parse(raw);
    if (cfg?.schema !== 1) return clone(DEFAULT_CONFIG);
    cfg.panels = cfg.panels || { left: [], right: [] };
    cfg.panels.left ||= [];
    cfg.panels.right ||= [];
    cfg.settings ||= {};
    return cfg;
  } catch {
    return clone(DEFAULT_CONFIG);
  }
}

export function saveConfig(cfg) {
  localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg));
}

export function getSettings(type) {
  const cfg = loadConfig();
  return cfg.settings?.[type] ?? {};
}

export function setSettings(type, settings) {
  const cfg = loadConfig();
  cfg.settings = { ...cfg.settings, [type]: settings };
  saveConfig(cfg);
}

export function boot() {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderRails, { once: true });
  } else {
    renderRails();
  }
}

// ── Internals ──────────────────────────────────────────────────────────

function clone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function renderRails() {
  // Cleanup existing mounts
  for (const m of mounted) {
    try { m.cleanup?.(); } catch (err) { console.warn(err); }
    m.root?.remove();
  }
  mounted = [];

  const cfg = loadConfig();
  for (const panel of ["left", "right"]) {
    const rail = document.getElementById(`${panel}-rail`);
    if (!rail) continue;
    const stack = rail.querySelector(".widget-stack");
    stack.innerHTML = "";

    const seen = new Set();
    for (const type of cfg.panels[panel]) {
      if (seen.has(type)) continue;
      seen.add(type);
      const def = REGISTRY.get(type);
      if (!def) continue;
      stack.append(buildCard(panel, type, def));
    }
    stack.append(buildAdder(panel));
  }
}

function buildCard(panel, type, def) {
  const card = document.createElement("article");
  card.className = `widget widget-${type}`;
  card.dataset.type = type;
  card.dataset.panel = panel;

  const head = document.createElement("header");
  head.className = "widget-head";
  head.innerHTML = `
    <span class="icon widget-icon">${iconHTML(def.icon || "")}</span>
    <span class="widget-title">${escapeHTML(def.title || type)}</span>
    <div class="widget-actions">
      <button data-action="up"     title="위로 이동" aria-label="위로 이동"><span class="icon">${iconHTML("chevronUp")}</span></button>
      <button data-action="down"   title="아래로 이동" aria-label="아래로 이동"><span class="icon">${iconHTML("chevronDown")}</span></button>
      <button data-action="remove" title="제거" aria-label="제거"><span class="icon">${iconHTML("x")}</span></button>
    </div>
  `;
  head.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    if (action === "remove")    removeWidget(panel, type);
    else if (action === "up")   moveWidget(panel, type, -1);
    else if (action === "down") moveWidget(panel, type, +1);
  });
  card.append(head);

  const body = document.createElement("div");
  body.className = "widget-body";
  card.append(body);

  let cleanup = null;
  try {
    const ctx = {
      settings: getSettings(type),
      updateSettings: (s) => setSettings(type, s),
    };
    cleanup = def.mount(body, ctx);
  } catch (err) {
    console.error(`widget ${type} mount failed`, err);
    body.innerHTML = `<div class="widget-error">위젯 로드 실패: ${escapeHTML(String(err.message || err))}</div>`;
  }

  mounted.push({ type, panel, root: card, cleanup });
  return card;
}

function buildAdder(panel) {
  const btn = document.createElement("button");
  btn.className = "widget-add-btn";
  btn.dataset.panel = panel;
  btn.type = "button";
  btn.innerHTML = `<span class="icon">${iconHTML("plus")}</span><span>위젯 추가</span>`;
  btn.addEventListener("click", () => openCatalog(panel, btn));
  return btn;
}

function addWidget(panel, type) {
  const cfg = loadConfig();
  // Remove from other panel if present, then append.
  for (const p of ["left", "right"]) {
    cfg.panels[p] = cfg.panels[p].filter((t) => t !== type);
  }
  cfg.panels[panel].push(type);
  saveConfig(cfg);
  renderRails();
}

function removeWidget(panel, type) {
  const cfg = loadConfig();
  cfg.panels[panel] = cfg.panels[panel].filter((t) => t !== type);
  saveConfig(cfg);
  renderRails();
}

function moveWidget(panel, type, dir) {
  const cfg = loadConfig();
  const arr = cfg.panels[panel];
  const i = arr.indexOf(type);
  if (i < 0) return;
  const j = i + dir;
  if (j < 0 || j >= arr.length) return;
  [arr[i], arr[j]] = [arr[j], arr[i]];
  saveConfig(cfg);
  renderRails();
}

function openCatalog(panel, anchor) {
  const cfg = loadConfig();
  const used = new Set([...cfg.panels.left, ...cfg.panels.right]);
  const available = listDefs().filter((d) => !used.has(d.type));

  const pop = document.createElement("div");
  pop.className = "widget-catalog";

  if (available.length === 0) {
    pop.innerHTML = `<p class="empty">추가할 수 있는 위젯이 없어요</p>`;
  } else {
    pop.innerHTML = available
      .map(
        (d) => `
        <button data-type="${escapeHTML(d.type)}" class="catalog-item" type="button">
          <span class="icon">${iconHTML(d.icon || "")}</span>
          <span class="title">${escapeHTML(d.title || d.type)}</span>
          <span class="desc">${escapeHTML(d.description || "")}</span>
        </button>`,
      )
      .join("");
  }

  document.body.append(pop);
  positionPopover(pop, anchor);

  pop.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-type]");
    if (!btn) return;
    addWidget(panel, btn.dataset.type);
    pop.remove();
  });

  setTimeout(() => {
    function close(ev) {
      if (!pop.contains(ev.target) && ev.target !== anchor) {
        pop.remove();
        document.removeEventListener("click", close, true);
      }
    }
    document.addEventListener("click", close, true);
  }, 0);
}

function positionPopover(pop, anchor) {
  const r = anchor.getBoundingClientRect();
  const pw = pop.offsetWidth || 220;
  const ph = pop.offsetHeight || 200;
  let left = r.left;
  let top = r.bottom + 4;
  if (left + pw > window.innerWidth - 8) left = window.innerWidth - pw - 8;
  if (top + ph > window.innerHeight - 8) top = r.top - ph - 4;
  pop.style.left = `${Math.max(8, left)}px`;
  pop.style.top = `${Math.max(8, top)}px`;
}

function escapeHTML(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
