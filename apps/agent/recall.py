"""Memory recall — §5.4 three-strategy parallel retrieval.

Raw SQL by design — see CLAUDE.md §4 module boundaries.
AgentCore must not depend on typed memory-eng methods that may be added
concurrently; instead we hit store.conn directly so this module stays
decoupled from memory-eng's delivery schedule.

Strategy summary:
1. Structured  — keyword match → SQL facts + upcoming events per person.
2. Semantic    — embed the message, cosine-search vec_episodes (network I/O
                 runs in asyncio.to_thread), apply recency boost ×1.5 for
                 episodes within 7 days.
3. Recency fallback — most-recent 5 episodes when semantic returns nothing
                 (empty index or embedding unavailable).

Trade-off: SQLite has a single connection (single-threaded). Strategies 1 & 3
are pure SQL (fast); strategy 2 has an external API call (network-bound).
We run the SQL strategies inline and wrap only the embed call in to_thread so
the event loop doesn't block.
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

# Recency boost: episodes within this window get score multiplied.
_RECENCY_DAYS = 7
_RECENCY_BOOST = 1.5

# Max results per strategy.
_STRUCTURED_FACTS_LIMIT = 10
_STRUCTURED_EVENTS_LIMIT = 5
_SEMANTIC_CANDIDATES = 20
_SEMANTIC_TOP_K = 5
_RECENCY_FALLBACK_K = 5


@dataclass(slots=True, frozen=True)
class RecallContext:
    episodes: list[tuple[int, str, float]] = field(default_factory=list)
    # (episode_id, summary, score)
    facts: list[tuple[int, str, str, str]] = field(default_factory=list)
    # (fact_id, person_name, predicate, object)
    upcoming_events: list[tuple[int, str, str]] = field(default_factory=list)
    # (event_id, title, when_at)
    attachments: list[tuple[int, str, str]] = field(default_factory=list)
    # (attachment_id, path, description) — only populated when episode_id given
    attachment_ids: list[int] = field(default_factory=list)
    # flat list of attachment IDs for AgentResponse.used_attachment_ids


async def recall(
    store: "MemoryStore",
    message: str,
    gemini_client: "GeminiClient",
    *,
    episode_id: int | None = None,
) -> RecallContext:
    """Run recall strategies and merge into a RecallContext.

    Strategies:
    1. Structured  — keyword match → SQL facts + upcoming events per person.
    2. Semantic    — embed the message, cosine-search vec_episodes.
    3. Recency fallback — most-recent 5 episodes when semantic returns nothing.
    4. Recent attachments — last N attachments for the current episode (only
                            when episode_id is explicitly provided).

    Trade-off note: SQLite uses a single connection created on the event-loop
    thread; we must NOT call store.conn from worker threads. Therefore:
    - Strategies 1, 3, 4 (pure SQL) run inline on the event-loop thread.
    - Strategy 2 splits the work: the embed() network call runs in
      asyncio.to_thread, then the SQLite KNN query runs inline.
    """
    log.debug("recall.start", msg_len=len(message), episode_id=episode_id)

    # Strategy 1: pure SQL — run inline.
    structured_facts, structured_events = _structured_recall(store, message)

    # Strategy 2: embed is network-bound → run in thread.
    # SQLite KNN search runs inline afterward (same thread as conn).
    episodes: list[tuple[int, str, float]] = []
    try:
        def _do_embed() -> list[float]:
            return gemini_client.embed(message, task_type="RETRIEVAL_QUERY")

        query_vec = await asyncio.to_thread(_do_embed)
        episodes = _semantic_search(store, query_vec)
    except Exception as exc:
        log.warning("recall.semantic_failed", error=str(exc))
        episodes = []

    # Strategy 3 fallback if semantic returns nothing.
    if not episodes:
        episodes = _recency_fallback(store)

    # Strategy 4: recent attachments — only when continuing an episode.
    attachment_tuples: list[tuple[int, str, str]] = []
    if episode_id is not None:
        attachment_tuples = _recent_attachments_for_episode(store, episode_id)

    attachment_ids = [a[0] for a in attachment_tuples]

    log.debug(
        "recall.done",
        episodes=len(episodes),
        facts=len(structured_facts),
        events=len(structured_events),
        attachments=len(attachment_tuples),
    )
    return RecallContext(
        episodes=episodes,
        facts=structured_facts,
        upcoming_events=structured_events,
        attachments=attachment_tuples,
        attachment_ids=attachment_ids,
    )


# ── Strategy 1: Structured ────────────────────────────────────────────────


def _structured_recall(
    store: "MemoryStore",
    message: str,
) -> tuple[list[tuple[int, str, str, str]], list[tuple[int, str, str]]]:
    """Match person names in message; pull facts + upcoming events via SQL."""
    people = store.list_people()
    matched_ids: list[int] = [p.id for p in people if p.name and p.name in message]
    if not matched_ids:
        return [], []

    facts: list[tuple[int, str, str, str]] = []
    events: list[tuple[int, str, str]] = []

    for pid in matched_ids:
        # Lookup person name for the fact tuple.
        person_row = store.conn.execute(
            "SELECT name FROM people WHERE id = ?", (pid,)
        ).fetchone()
        if not person_row:
            continue
        person_name: str = person_row["name"]

        # Active facts for this person.
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

        # Upcoming events for this person.
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
    """Search vec_episodes with a pre-computed query vector, apply recency boost.

    Runs on the event-loop thread (same thread as SQLite conn creation).
    The embed() call that produced query_vec runs in asyncio.to_thread —
    see recall() above for the split.
    """
    vec_bytes = _floats_to_bytes(query_vec)

    # sqlite-vec KNN query.
    rows = store.conn.execute(
        """
        SELECT ve.episode_id,
               e.summary,
               e.when_at,
               ve.distance
        FROM   vec_episodes ve
        JOIN   episodes e ON e.id = ve.episode_id
        WHERE  ve.embedding MATCH ?
          AND  k = ?
        ORDER  BY ve.distance
        """,
        (vec_bytes, _SEMANTIC_CANDIDATES),
    ).fetchall()

    if not rows:
        return []

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=_RECENCY_DAYS)
    results: list[tuple[int, str, float]] = []

    for row in rows:
        distance: float = float(row["distance"])
        # Convert distance to a pseudo-score (lower distance = higher score).
        score = 1.0 / (1.0 + distance)

        when_str: str | None = row["when_at"]
        if when_str:
            try:
                when_dt = datetime.fromisoformat(when_str).replace(tzinfo=timezone.utc)
                if when_dt >= cutoff:
                    score *= _RECENCY_BOOST
            except ValueError:
                pass

        summary: str = row["summary"] or ""
        results.append((int(row["episode_id"]), summary, round(score, 4)))

    # Sort by score descending, take top K.
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:_SEMANTIC_TOP_K]


# ── Strategy 3: Recency fallback ──────────────────────────────────────────


def _recency_fallback(store: "MemoryStore") -> list[tuple[int, str, float]]:
    """Return the 5 most-recent episodes by when_at."""
    rows = store.conn.execute(
        """
        SELECT id, summary, when_at
        FROM   episodes
        ORDER  BY when_at DESC
        LIMIT  ?
        """,
        (_RECENCY_FALLBACK_K,),
    ).fetchall()
    return [
        (int(row["id"]), row["summary"] or "", 1.0)
        for row in rows
    ]


# ── Strategy 4: Recent attachments for episode ────────────────────────────


def _recent_attachments_for_episode(
    store: "MemoryStore",
    episode_id: int,
    limit: int = 3,
) -> list[tuple[int, str, str]]:
    """Return the most-recent attachments for a given episode.

    Returns a list of (attachment_id, path, description) tuples.
    Only included when episode_id is explicitly provided — new episodes
    with no prior context return [].
    """
    rows = store.conn.execute(
        """
        SELECT id, path, description
        FROM   attachments
        WHERE  episode_id = ?
        ORDER  BY id DESC
        LIMIT  ?
        """,
        (episode_id, limit),
    ).fetchall()
    return [
        (int(row["id"]), row["path"] or "", row["description"] or "")
        for row in rows
    ]


# ── helpers ───────────────────────────────────────────────────────────────


def _floats_to_bytes(values: list[float]) -> bytes:
    """Pack float list to little-endian f32 bytes for sqlite-vec."""
    return struct.pack(f"{len(values)}f", *values)
