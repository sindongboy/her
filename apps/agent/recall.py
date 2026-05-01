"""Memory recall — sessions/notes/facts/events strategies.

Raw SQL by design — see CLAUDE.md §4 module boundaries. Recall directly
hits store.conn so this module stays decoupled from MemoryStore method
ordering.

Strategy summary:
1. Structured  — keyword match → SQL facts + upcoming events per person.
2. Semantic    — embed the message, KNN-search vec_messages → join up to
                 sessions; recency boost for sessions whose last_active_at
                 is within 7 days.
3. Recency fallback — most-recent 5 sessions when semantic returns nothing.
4. Notes       — keyword LIKE match against active notes.
5. Recent attachments — last N attachments for the active session.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from apps.agent.gemini import GeminiClient
    from apps.memory.store import MemoryStore

log = structlog.get_logger(__name__)

_RECENCY_DAYS = 7
_RECENCY_BOOST = 1.5

_STRUCTURED_FACTS_LIMIT = 10
_STRUCTURED_EVENTS_LIMIT = 5
_SEMANTIC_CANDIDATES = 20
_SEMANTIC_TOP_K = 5
_RECENCY_FALLBACK_K = 5
_NOTES_LIMIT = 5


@dataclass(slots=True, frozen=True)
class RecallContext:
    sessions: list[tuple[int, str, float]] = field(default_factory=list)
    # (session_id, summary_or_title, score)
    facts: list[tuple[int, str, str, str]] = field(default_factory=list)
    # (fact_id, person_name, predicate, object)
    upcoming_events: list[tuple[int, str, str]] = field(default_factory=list)
    # (event_id, title, when_at)
    notes: list[tuple[int, str]] = field(default_factory=list)
    # (note_id, content)
    attachments: list[tuple[int, str, str]] = field(default_factory=list)
    # (attachment_id, path, description)
    attachment_ids: list[int] = field(default_factory=list)


async def recall(
    store: "MemoryStore",
    message: str,
    gemini_client: "GeminiClient",
    *,
    session_id: int | None = None,
) -> RecallContext:
    """Run recall strategies and merge into a RecallContext."""
    log.debug("recall.start", msg_len=len(message), session_id=session_id)

    structured_facts, structured_events = _structured_recall(store, message)

    sessions: list[tuple[int, str, float]] = []
    try:
        def _do_embed() -> list[float]:
            return gemini_client.embed(message, task_type="RETRIEVAL_QUERY")

        query_vec = await asyncio.to_thread(_do_embed)
        sessions = _semantic_search(store, query_vec)
    except Exception as exc:
        log.warning("recall.semantic_failed", error=str(exc))
        sessions = []

    if not sessions:
        sessions = _recency_fallback(store)

    notes = _notes_recall(store, message)

    attachment_tuples: list[tuple[int, str, str]] = []
    if session_id is not None:
        attachment_tuples = _recent_attachments_for_session(store, session_id)

    attachment_ids = [a[0] for a in attachment_tuples]

    log.debug(
        "recall.done",
        sessions=len(sessions),
        facts=len(structured_facts),
        events=len(structured_events),
        notes=len(notes),
        attachments=len(attachment_tuples),
    )
    return RecallContext(
        sessions=sessions,
        facts=structured_facts,
        upcoming_events=structured_events,
        notes=notes,
        attachments=attachment_tuples,
        attachment_ids=attachment_ids,
    )


# ── Strategy 1: Structured ────────────────────────────────────────────────


def _structured_recall(
    store: "MemoryStore",
    message: str,
) -> tuple[list[tuple[int, str, str, str]], list[tuple[int, str, str]]]:
    people = store.list_people()
    matched_ids: list[int] = [p.id for p in people if p.name and p.name in message]
    if not matched_ids:
        return [], []

    facts: list[tuple[int, str, str, str]] = []
    events: list[tuple[int, str, str]] = []

    for pid in matched_ids:
        person_row = store.conn.execute(
            "SELECT name FROM people WHERE id = ?", (pid,)
        ).fetchone()
        if not person_row:
            continue
        person_name: str = person_row["name"]

        fact_rows = store.conn.execute(
            """
            SELECT id, predicate, object
            FROM   facts
            WHERE  subject_person_id = ?
              AND  archived_at IS NULL
            ORDER  BY valid_from DESC
            LIMIT  ?
            """,
            (pid, _STRUCTURED_FACTS_LIMIT),
        ).fetchall()
        for row in fact_rows:
            facts.append((row["id"], person_name, row["predicate"], row["object"]))

        event_rows = store.conn.execute(
            """
            SELECT id, title, when_at
            FROM   events
            WHERE  person_id = ?
              AND  status = 'pending'
              AND  when_at >= datetime('now')
            ORDER  BY when_at
            LIMIT  ?
            """,
            (pid, _STRUCTURED_EVENTS_LIMIT),
        ).fetchall()
        for row in event_rows:
            events.append((row["id"], row["title"], row["when_at"]))

    return facts, events


# ── Strategy 2: Semantic ──────────────────────────────────────────────────


def _semantic_search(
    store: "MemoryStore",
    query_vec: list[float],
) -> list[tuple[int, str, float]]:
    """KNN-search vec_messages and surface the parent session, with recency boost."""
    vec_bytes = _floats_to_bytes(query_vec)

    rows = store.conn.execute(
        """
        SELECT vm.message_id,
               m.session_id,
               m.content,
               s.summary,
               s.title,
               s.last_active_at,
               vm.distance
        FROM   vec_messages vm
        JOIN   messages m ON m.id = vm.message_id
        JOIN   sessions s ON s.id = m.session_id
        WHERE  vm.embedding MATCH ?
          AND  k = ?
        ORDER  BY vm.distance
        """,
        (vec_bytes, _SEMANTIC_CANDIDATES),
    ).fetchall()

    if not rows:
        return []

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=_RECENCY_DAYS)
    by_session: dict[int, tuple[str, float]] = {}

    for row in rows:
        session_id = int(row["session_id"])
        distance: float = float(row["distance"])
        score = 1.0 / (1.0 + distance)

        last_active_str: str | None = row["last_active_at"]
        if last_active_str:
            try:
                ts = datetime.fromisoformat(last_active_str).replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    score *= _RECENCY_BOOST
            except ValueError:
                pass

        # Prefer session summary > session title > matching message snippet.
        text = row["summary"] or row["title"] or (row["content"] or "")[:120]
        prev = by_session.get(session_id)
        if prev is None or score > prev[1]:
            by_session[session_id] = (text, round(score, 4))

    results = [(sid, text, score) for sid, (text, score) in by_session.items()]
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:_SEMANTIC_TOP_K]


# ── Strategy 3: Recency fallback ──────────────────────────────────────────


def _recency_fallback(store: "MemoryStore") -> list[tuple[int, str, float]]:
    rows = store.conn.execute(
        """
        SELECT id, summary, title
        FROM   sessions
        WHERE  archived_at IS NULL
        ORDER  BY last_active_at DESC
        LIMIT  ?
        """,
        (_RECENCY_FALLBACK_K,),
    ).fetchall()
    return [
        (int(row["id"]), row["summary"] or row["title"] or "", 1.0)
        for row in rows
    ]


# ── Strategy 4: Notes ─────────────────────────────────────────────────────


def _notes_recall(store: "MemoryStore", message: str) -> list[tuple[int, str]]:
    """Naïve LIKE search against active notes — embeddings come in a later PR."""
    text = message.strip()
    if len(text) < 2:
        return []
    pattern = f"%{text[:80]}%"
    rows = store.conn.execute(
        """
        SELECT id, content
        FROM   notes
        WHERE  archived_at IS NULL
          AND  content LIKE ?
        ORDER  BY updated_at DESC
        LIMIT  ?
        """,
        (pattern, _NOTES_LIMIT),
    ).fetchall()
    return [(int(row["id"]), row["content"] or "") for row in rows]


# ── Strategy 5: Recent attachments for session ────────────────────────────


def _recent_attachments_for_session(
    store: "MemoryStore",
    session_id: int,
    limit: int = 3,
) -> list[tuple[int, str, str]]:
    rows = store.conn.execute(
        """
        SELECT id, path, description
        FROM   attachments
        WHERE  session_id = ?
        ORDER  BY id DESC
        LIMIT  ?
        """,
        (session_id, limit),
    ).fetchall()
    return [
        (int(row["id"]), row["path"] or "", row["description"] or "")
        for row in rows
    ]


# ── helpers ───────────────────────────────────────────────────────────────


def _floats_to_bytes(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)
