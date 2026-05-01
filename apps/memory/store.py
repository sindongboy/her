"""Memory layer: SQLite + sqlite-vec store.

Phase 0 scope: boots the schema and exposes People CRUD. Episodes, events,
facts, attachments, and embedding search are implemented as the agent path
needs them.

Per CLAUDE.md §2.3, this layer always operates on real names. Anonymization
is the Agent Core LLM-adapter's responsibility, not this layer's.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import sqlite_vec

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

EMBED_MODEL_ID = "gemini-embedding-001"
EMBED_DIM = 768

_VALID_CHANNELS = {"voice", "text", "mixed"}
_VALID_STATUSES = {"pending", "done", "cancelled"}


@dataclass(slots=True, frozen=True)
class Person:
    id: int
    name: str
    relation: str | None
    birthday: str | None
    preferences: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(slots=True, frozen=True)
class Episode:
    id: int
    when_at: str
    summary: str | None
    primary_channel: str  # 'voice' | 'text' | 'mixed'


@dataclass(slots=True, frozen=True)
class Event:
    id: int
    person_id: int | None
    type: str
    title: str
    when_at: str
    recurrence: str | None
    source: str | None
    status: str  # 'pending' | 'done' | 'cancelled'


@dataclass(slots=True, frozen=True)
class Fact:
    id: int
    subject_person_id: int
    predicate: str
    object: str
    confidence: float
    source_episode_id: int | None
    valid_from: str
    archived_at: str | None


@dataclass(slots=True, frozen=True)
class Preference:
    person_id: int
    domain: str
    value: str
    last_seen_at: str


@dataclass(slots=True, frozen=True)
class Attachment:
    id: int
    episode_id: int
    sha256: str
    mime: str | None
    ext: str | None
    byte_size: int | None
    path: str
    description: str | None
    ingested_at: str


@dataclass(slots=True, frozen=True)
class EpisodeMatch:
    episode: Episode
    score: float   # final score after recency weight
    distance: float  # raw vec0 distance


class MemoryStore:
    """Synchronous façade over SQLite + sqlite-vec.

    Async callers wrap with `asyncio.to_thread`.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        self._migrate()

    def _migrate(self) -> None:
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            self.conn.executescript(f.read())

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            self.conn.execute("BEGIN")
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.conn.close()

    # ── people ─────────────────────────────────────────────────────────

    def add_person(
        self,
        name: str,
        relation: str | None = None,
        birthday: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO people (name, relation, birthday, preferences_json) "
            "VALUES (?, ?, ?, ?)",
            (name, relation, birthday, json.dumps(preferences or {}, ensure_ascii=False)),
        )
        return int(cur.lastrowid or 0)

    def get_person(self, person_id: int) -> Person | None:
        row = self.conn.execute(
            "SELECT * FROM people WHERE id = ?", (person_id,)
        ).fetchone()
        return _row_to_person(row) if row else None

    def list_people(self) -> list[Person]:
        rows = self.conn.execute("SELECT * FROM people ORDER BY id").fetchall()
        return [_row_to_person(r) for r in rows]

    # ── episodes ───────────────────────────────────────────────────────

    def add_episode(
        self,
        summary: str | None = None,
        *,
        primary_channel: str = "text",
        when_at: str | None = None,
    ) -> int:
        if primary_channel not in _VALID_CHANNELS:
            raise ValueError(f"primary_channel must be one of {_VALID_CHANNELS}")
        if when_at is not None:
            cur = self.conn.execute(
                "INSERT INTO episodes (when_at, summary, primary_channel) VALUES (?, ?, ?)",
                (when_at, summary, primary_channel),
            )
        else:
            cur = self.conn.execute(
                "INSERT INTO episodes (summary, primary_channel) VALUES (?, ?)",
                (summary, primary_channel),
            )
        return int(cur.lastrowid or 0)

    def get_episode(self, episode_id: int) -> Episode | None:
        row = self.conn.execute(
            "SELECT id, when_at, summary, primary_channel FROM episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        return _row_to_episode(row) if row else None

    def list_recent_episodes(self, limit: int = 20) -> list[Episode]:
        rows = self.conn.execute(
            "SELECT id, when_at, summary, primary_channel FROM episodes "
            "ORDER BY when_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def set_episode_summary(self, episode_id: int, summary: str) -> None:
        self.conn.execute(
            "UPDATE episodes SET summary = ? WHERE id = ?",
            (summary, episode_id),
        )

    def upsert_episode_embedding(
        self,
        episode_id: int,
        vector: list[float],
        *,
        model_id: str,
        dim: int,
        task_type: str,
    ) -> None:
        if len(vector) != dim:
            raise ValueError(f"vector length {len(vector)} != dim {dim}")
        blob = struct.pack(f"{dim}f", *vector)
        # vec0 virtual tables don't support INSERT OR REPLACE; delete then insert.
        self.conn.execute(
            "DELETE FROM vec_episodes WHERE episode_id = ?", (episode_id,)
        )
        self.conn.execute(
            "INSERT INTO vec_episodes (episode_id, embedding) VALUES (?, ?)",
            (episode_id, blob),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO episode_embedding_meta "
            "(episode_id, model_id, dim, task_type) VALUES (?, ?, ?, ?)",
            (episode_id, model_id, dim, task_type),
        )

    def search_episodes_by_embedding(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        recency_days: int = 7,
        recency_weight: float = 1.5,
    ) -> list[EpisodeMatch]:
        dim = len(query_vector)
        blob = struct.pack(f"{dim}f", *query_vector)

        candidates = self.conn.execute(
            "SELECT episode_id, distance FROM vec_episodes "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (blob, limit * 4),
        ).fetchall()

        now = datetime.now(tz=timezone.utc)
        results: list[EpisodeMatch] = []
        for row in candidates:
            ep = self.get_episode(row["episode_id"])
            if ep is None:
                continue
            distance = float(row["distance"])
            base_score = 1.0 / (1.0 + distance)
            weight = _recency_weight(ep.when_at, now, recency_days, recency_weight)
            results.append(
                EpisodeMatch(episode=ep, score=base_score * weight, distance=distance)
            )

        results.sort(key=lambda m: m.score, reverse=True)
        return results[:limit]

    # ── events ─────────────────────────────────────────────────────────

    def add_event(
        self,
        type: str,
        title: str,
        when_at: str,
        *,
        person_id: int | None = None,
        recurrence: str | None = None,
        source: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO events (person_id, type, title, when_at, recurrence, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (person_id, type, title, when_at, recurrence, source),
        )
        return int(cur.lastrowid or 0)

    def get_event(self, event_id: int) -> Event | None:
        row = self.conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return _row_to_event(row) if row else None

    def list_upcoming_events(
        self,
        *,
        within_hours: int = 24,
        now_iso: str | None = None,
    ) -> list[Event]:
        if now_iso is not None:
            rows = self.conn.execute(
                "SELECT * FROM events "
                "WHERE status = 'pending' "
                "  AND when_at >= ? "
                "  AND when_at <= datetime(?, '+' || ? || ' hours') "
                "ORDER BY when_at",
                (now_iso, now_iso, str(within_hours)),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events "
                "WHERE status = 'pending' "
                "  AND when_at >= CURRENT_TIMESTAMP "
                "  AND when_at <= datetime('now', '+' || ? || ' hours') "
                "ORDER BY when_at",
                (str(within_hours),),
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def set_event_status(self, event_id: int, status: str) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {_VALID_STATUSES}")
        self.conn.execute(
            "UPDATE events SET status = ? WHERE id = ?",
            (status, event_id),
        )

    # ── facts ──────────────────────────────────────────────────────────

    def add_fact(
        self,
        subject_person_id: int,
        predicate: str,
        object: str,
        *,
        confidence: float,
        source_episode_id: int | None = None,
    ) -> int:
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        cur = self.conn.execute(
            "INSERT INTO facts (subject_person_id, predicate, object, confidence, source_episode_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (subject_person_id, predicate, object, confidence, source_episode_id),
        )
        return int(cur.lastrowid or 0)

    def list_active_facts(self, subject_person_id: int) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts "
            "WHERE subject_person_id = ? AND archived_at IS NULL "
            "ORDER BY valid_from DESC",
            (subject_person_id,),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def archive_fact(self, fact_id: int) -> None:
        self.conn.execute(
            "UPDATE facts SET archived_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND archived_at IS NULL",
            (fact_id,),
        )

    # ── preferences ────────────────────────────────────────────────────

    def upsert_preference(self, person_id: int, domain: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO preferences (person_id, domain, value) VALUES (?, ?, ?) "
            "ON CONFLICT(person_id, domain, value) DO UPDATE SET "
            "last_seen_at = CURRENT_TIMESTAMP",
            (person_id, domain, value),
        )

    def list_preferences(self, person_id: int) -> list[Preference]:
        rows = self.conn.execute(
            "SELECT * FROM preferences WHERE person_id = ? "
            "ORDER BY domain, value",
            (person_id,),
        ).fetchall()
        return [_row_to_preference(r) for r in rows]

    # ── attachments ────────────────────────────────────────────────────

    def add_attachment(
        self,
        episode_id: int,
        *,
        sha256: str,
        path: str,
        mime: str | None = None,
        ext: str | None = None,
        byte_size: int | None = None,
        description: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO attachments "
            "(episode_id, sha256, mime, ext, byte_size, path, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (episode_id, sha256, mime, ext, byte_size, path, description),
        )
        return int(cur.lastrowid or 0)

    def list_attachments(self, episode_id: int) -> list[Attachment]:
        rows = self.conn.execute(
            "SELECT * FROM attachments WHERE episode_id = ? ORDER BY id",
            (episode_id,),
        ).fetchall()
        return [_row_to_attachment(r) for r in rows]

    def find_attachment_by_sha256(
        self, episode_id: int, sha256: str
    ) -> Attachment | None:
        row = self.conn.execute(
            "SELECT * FROM attachments WHERE episode_id = ? AND sha256 = ?",
            (episode_id, sha256),
        ).fetchone()
        return _row_to_attachment(row) if row else None


# ── helpers ────────────────────────────────────────────────────────────────


def _row_to_person(r: sqlite3.Row) -> Person:
    return Person(
        id=r["id"],
        name=r["name"],
        relation=r["relation"],
        birthday=r["birthday"],
        preferences=json.loads(r["preferences_json"]),
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


def _row_to_episode(r: sqlite3.Row) -> Episode:
    return Episode(
        id=r["id"],
        when_at=r["when_at"],
        summary=r["summary"],
        primary_channel=r["primary_channel"],
    )


def _row_to_event(r: sqlite3.Row) -> Event:
    return Event(
        id=r["id"],
        person_id=r["person_id"],
        type=r["type"],
        title=r["title"],
        when_at=r["when_at"],
        recurrence=r["recurrence"],
        source=r["source"],
        status=r["status"],
    )


def _row_to_fact(r: sqlite3.Row) -> Fact:
    return Fact(
        id=r["id"],
        subject_person_id=r["subject_person_id"],
        predicate=r["predicate"],
        object=r["object"],
        confidence=r["confidence"],
        source_episode_id=r["source_episode_id"],
        valid_from=r["valid_from"],
        archived_at=r["archived_at"],
    )


def _row_to_preference(r: sqlite3.Row) -> Preference:
    return Preference(
        person_id=r["person_id"],
        domain=r["domain"],
        value=r["value"],
        last_seen_at=r["last_seen_at"],
    )


def _row_to_attachment(r: sqlite3.Row) -> Attachment:
    return Attachment(
        id=r["id"],
        episode_id=r["episode_id"],
        sha256=r["sha256"],
        mime=r["mime"],
        ext=r["ext"],
        byte_size=r["byte_size"],
        path=r["path"],
        description=r["description"],
        ingested_at=r["ingested_at"],
    )


def _recency_weight(
    when_at_iso: str,
    now: datetime,
    recency_days: int,
    recency_weight: float,
) -> float:
    """Return recency_weight if episode is within recency_days, else 1.0."""
    try:
        # Parse ISO string (with or without timezone)
        ts_str = when_at_iso.replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            # Fallback: strip trailing fractional seconds weirdness
            ts = datetime.fromisoformat(ts_str[:19])
        # Make both tz-aware or both naive for comparison
        if ts.tzinfo is None:
            now_cmp = now.replace(tzinfo=None)
        else:
            now_cmp = now
        delta_days = (now_cmp - ts).total_seconds() / 86400
        if delta_days <= recency_days:
            return recency_weight
    except (ValueError, TypeError):
        pass
    return 1.0
