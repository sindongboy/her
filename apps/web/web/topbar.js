// Topbar tagline — keeps "session-title" in sync with the active session.
// Independent of the sessions widget so the tagline still updates even
// when the user removes the sessions widget from the rail.

import { state, on } from "/static/state.js";

const titleEl = document.getElementById("session-title");

let sessionsCache = [];

async function refreshSessions() {
  try {
    const r = await fetch("/api/sessions");
    if (!r.ok) return;
    sessionsCache = await r.json();
    applyTitle();
  } catch {}
}

function applyTitle() {
  if (!titleEl) return;
  if (!state.sessionId) {
    titleEl.textContent = "새 대화";
    return;
  }
  const s = sessionsCache.find((s) => s.id === state.sessionId);
  if (s) {
    titleEl.textContent = s.title || s.summary || `세션 ${s.id}`;
  } else {
    // Cache miss — fetch and retry once.
    refreshSessions();
  }
}

on("sessionId", applyTitle);
document.addEventListener("her:turn-complete", refreshSessions);
refreshSessions();
