"""Memory layer: SQLite + sqlite-vec store (schema v2).

Schema v2 — sessions/messages/notes replace v1's episodes-only model.
On boot, an existing v1 file is auto-backed-up and a v2 file is created.

Per CLAUDE.md §2.3, this layer always operates on real names. Anonymization
is the Agent Core LLM-adapter's responsibility.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import sqlite_vec
import structlog

log = structlog.get_logger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_VERSION = 2

EMBED_MODEL_ID = "gemini-embedding-001"
EMBED_DIM = 768

_VALID_ROLES = {"user", "assistant", "system"}
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
    archived_at: str | None


@dataclass(slots=True, frozen=True)
class Session:
    id: int
    started_at: str
    last_active_at: str
    title: str | None
    summary: str | None
    archived_at: str | None


@dataclass(slots=True, frozen=True)
class Message:
    id: int
    session_id: int
    role: str
    content: str
    ts: str


@dataclass(slots=True, frozen=True)
class Event:
    id: int
    person_id: int | None
    type: str
    title: str
    when_at: str
    recurrence: str | None
    source: str | None
    status: str
    archived_at: str | None


@dataclass(slots=True, frozen=True)
class Fact:
    id: int
    subject_person_id: int
    predicate: str
    object: str
    confidence: float
    source_session_id: int | None
    valid_from: str
    archived_at: str | None


@dataclass(slots=True, frozen=True)
class Preference:
    person_id: int | None
    domain: str
    value: str
    last_seen_at: str


@dataclass(slots=True, frozen=True)
class Note:
    id: int
    content: str
    tags: list[str]
    source_session_id: int | None
    created_at: str
    updated_at: str
    archived_at: str | None


@dataclass(slots=True, frozen=True)
class Attachment:
    id: int
    session_id: int
    sha256: str
    mime: str | None
    ext: str | None
    byte_size: int | None
    path: str
    description: str | None
    ingested_at: str


@dataclass(slots=True, frozen=True)
class SessionMatch:
    session: Session
    score: float
    distance: float


class MemoryStore:
    """Synchronous façade over SQLite + sqlite-vec.

    Async callers wrap calls with ``asyncio.to_thread``.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connect()
        self._migrate_or_rebuild()

    def _connect(self) -> None:
        # check_same_thread=False because FastAPI/Starlette run async routes in
        # a worker thread distinct from the one that built the store. Access is
        # serialised by the asyncio loop in production and by the test runner
        # in tests, so it is safe to share the connection across threads.
        self.conn = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def _migrate_or_rebuild(self) -> None:
        """Detect a v1 schema and, if found, back the file up before rebuilding.

        v1 is identified by the presence of an `episodes` table and the absence
        of `sessions`. Anything else is treated as either fresh or already v2,
        and we just (re-)apply the idempotent v2 schema.
        """
        has_episodes = bool(
            self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='episodes'"
            ).fetchone()
        )
        has_sessions = bool(
            self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
            ).fetchone()
        )

        if has_episodes and not has_sessions:
            self._backup_and_rebuild()
            return

        # Patch missing columns BEFORE running the schema, so any indexes
        # that reference those columns succeed when re-applied.
        self._patch_v2_columns()
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _patch_v2_columns(self) -> None:
        """Add columns introduced after v2 first shipped — backwards-compatible
        ALTER TABLE so existing v2 DBs pick up the new shape. No-op on fresh
        DBs where the table itself doesn't exist yet."""
        for table, column, ddl in [
            ("people", "archived_at", "TEXT"),
            ("events", "archived_at", "TEXT"),
        ]:
            exists = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                continue
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                log.info("memory.column_added", table=table, column=column)

    def _backup_and_rebuild(self) -> None:
        """Close, rename db file to a timestamped backup, reopen with v2 schema."""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = self.db_path.with_name(self.db_path.name + f".bak-{ts}")

        self.conn.close()

        if self.db_path.exists():
            self.db_path.rename(backup_path)
            sys.stderr.write(
                "\n⚠️  기존 메모리 DB(v1) 를 v2 스키마로 새로 생성합니다.\n"
                f"   원본은 {backup_path} 로 백업되었습니다.\n\n"
            )
            log.warning(
                "memory.v1_db_backed_up",
                db=str(self.db_path),
                backup=str(backup_path),
            )

        self._connect()
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

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

    def list_people(self, *, include_archived: bool = False) -> list[Person]:
        if include_archived:
            rows = self.conn.execute("SELECT * FROM people ORDER BY id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM people WHERE archived_at IS NULL ORDER BY id"
            ).fetchall()
        return [_row_to_person(r) for r in rows]

    def update_person(
        self,
        person_id: int,
        *,
        name: str | None = None,
        relation: str | None = None,
        birthday: str | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if name is not None:
            sets.append("name = ?"); params.append(name)
        if relation is not None:
            sets.append("relation = ?"); params.append(relation or None)
        if birthday is not None:
            sets.append("birthday = ?"); params.append(birthday or None)
        if not sets:
            return
        sets.append("updated_at = CURRENT_TIMESTAMP")
        params.append(person_id)
        self.conn.execute(
            f"UPDATE people SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )

    def archive_person(self, person_id: int) -> None:
        self.conn.execute(
            "UPDATE people SET archived_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND archived_at IS NULL",
            (person_id,),
        )

    def restore_person(self, person_id: int) -> None:
        self.conn.execute(
            "UPDATE people SET archived_at = NULL WHERE id = ?", (person_id,)
        )

    # ── sessions ───────────────────────────────────────────────────────

    def add_session(
        self,
        *,
        title: str | None = None,
        summary: str | None = None,
        started_at: str | None = None,
    ) -> int:
        if started_at is not None:
            cur = self.conn.execute(
                "INSERT INTO sessions (started_at, last_active_at, title, summary) "
                "VALUES (?, ?, ?, ?)",
                (started_at, started_at, title, summary),
            )
        else:
            cur = self.conn.execute(
                "INSERT INTO sessions (title, summary) VALUES (?, ?)",
                (title, summary),
            )
        return int(cur.lastrowid or 0)

    def get_session(self, session_id: int) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row) if row else None

    def list_recent_sessions(
        self, *, limit: int = 50, include_archived: bool = False
    ) -> list[Session]:
        if include_archived:
            rows = self.conn.execute(
                "SELECT * FROM sessions ORDER BY last_active_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE archived_at IS NULL "
                "ORDER BY last_active_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_session(r) for r in rows]

    def set_session_title(self, session_id: int, title: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET title = ?, last_active_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (title, session_id),
        )

    def set_session_summary(self, session_id: int, summary: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET summary = ?, last_active_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (summary, session_id),
        )

    def touch_session(self, session_id: int) -> None:
        self.conn.execute(
            "UPDATE sessions SET last_active_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )

    def archive_session(self, session_id: int) -> None:
        self.conn.execute(
            "UPDATE sessions SET archived_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND archived_at IS NULL",
            (session_id,),
        )

    def upsert_session_embedding(
        self,
        session_id: int,
        vector: list[float],
        *,
        model_id: str,
        dim: int,
        task_type: str,
    ) -> None:
        if len(vector) != dim:
            raise ValueError(f"vector length {len(vector)} != dim {dim}")
        blob = struct.pack(f"{dim}f", *vector)
        self.conn.execute("DELETE FROM vec_sessions WHERE session_id = ?", (session_id,))
        self.conn.execute(
            "INSERT INTO vec_sessions (session_id, embedding) VALUES (?, ?)",
            (session_id, blob),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO session_embedding_meta "
            "(session_id, model_id, dim, task_type) VALUES (?, ?, ?, ?)",
            (session_id, model_id, dim, task_type),
        )

    def search_sessions_by_embedding(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        recency_days: int = 7,
        recency_weight: float = 1.5,
    ) -> list[SessionMatch]:
        dim = len(query_vector)
        blob = struct.pack(f"{dim}f", *query_vector)

        candidates = self.conn.execute(
            "SELECT session_id, distance FROM vec_sessions "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (blob, limit * 4),
        ).fetchall()

        now = datetime.now(tz=timezone.utc)
        results: list[SessionMatch] = []
        for row in candidates:
            sess = self.get_session(row["session_id"])
            if sess is None:
                continue
            distance = float(row["distance"])
            base_score = 1.0 / (1.0 + distance)
            weight = _recency_weight(sess.last_active_at, now, recency_days, recency_weight)
            results.append(
                SessionMatch(session=sess, score=base_score * weight, distance=distance)
            )

        results.sort(key=lambda m: m.score, reverse=True)
        return results[:limit]

    # ── messages ───────────────────────────────────────────────────────

    def add_message(self, session_id: int, role: str, content: str) -> int:
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {_VALID_ROLES}")
        cur = self.conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        # Bump session.last_active_at on every message append.
        self.conn.execute(
            "UPDATE sessions SET last_active_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
        return int(cur.lastrowid or 0)

    def list_messages(self, session_id: int) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY ts, id",
            (session_id,),
        ).fetchall()
        return [_row_to_message(r) for r in rows]

    def recall_messages(self, session_id: int, *, limit: int = 20) -> list[Message]:
        """Return up to *limit* most-recent messages in chronological order."""
        rows = self.conn.execute(
            "SELECT * FROM ("
            "  SELECT * FROM messages WHERE session_id = ? ORDER BY ts DESC, id DESC LIMIT ?"
            ") ORDER BY ts, id",
            (session_id, limit),
        ).fetchall()
        return [_row_to_message(r) for r in rows]

    def upsert_message_embedding(
        self,
        message_id: int,
        vector: list[float],
        *,
        model_id: str,
        dim: int,
        task_type: str,
    ) -> None:
        if len(vector) != dim:
            raise ValueError(f"vector length {len(vector)} != dim {dim}")
        blob = struct.pack(f"{dim}f", *vector)
        self.conn.execute("DELETE FROM vec_messages WHERE message_id = ?", (message_id,))
        self.conn.execute(
            "INSERT INTO vec_messages (message_id, embedding) VALUES (?, ?)",
            (message_id, blob),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO message_embedding_meta "
            "(message_id, model_id, dim, task_type) VALUES (?, ?, ?, ?)",
            (message_id, model_id, dim, task_type),
        )

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
                "  AND archived_at IS NULL "
                "  AND when_at >= ? "
                "  AND when_at <= datetime(?, '+' || ? || ' hours') "
                "ORDER BY when_at",
                (now_iso, now_iso, str(within_hours)),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events "
                "WHERE status = 'pending' "
                "  AND archived_at IS NULL "
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

    def list_all_events(self, *, include_archived: bool = False) -> list[Event]:
        if include_archived:
            rows = self.conn.execute(
                "SELECT * FROM events ORDER BY when_at DESC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE archived_at IS NULL "
                "ORDER BY when_at DESC"
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def update_event(
        self,
        event_id: int,
        *,
        type: str | None = None,
        title: str | None = None,
        when_at: str | None = None,
        recurrence: str | None = None,
        person_id: int | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if type is not None:     sets.append("type = ?"); params.append(type)
        if title is not None:    sets.append("title = ?"); params.append(title)
        if when_at is not None:  sets.append("when_at = ?"); params.append(when_at)
        if recurrence is not None: sets.append("recurrence = ?"); params.append(recurrence or None)
        if person_id is not None: sets.append("person_id = ?"); params.append(person_id)
        if not sets:
            return
        params.append(event_id)
        self.conn.execute(
            f"UPDATE events SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )

    def archive_event(self, event_id: int) -> None:
        self.conn.execute(
            "UPDATE events SET archived_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND archived_at IS NULL",
            (event_id,),
        )

    def restore_event(self, event_id: int) -> None:
        self.conn.execute(
            "UPDATE events SET archived_at = NULL WHERE id = ?", (event_id,)
        )

    # ── facts ──────────────────────────────────────────────────────────

    def add_fact(
        self,
        subject_person_id: int,
        predicate: str,
        object: str,
        *,
        confidence: float,
        source_session_id: int | None = None,
    ) -> int:
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        cur = self.conn.execute(
            "INSERT INTO facts "
            "(subject_person_id, predicate, object, confidence, source_session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (subject_person_id, predicate, object, confidence, source_session_id),
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

    def list_all_facts(self, *, include_archived: bool = False) -> list[Fact]:
        if include_archived:
            rows = self.conn.execute(
                "SELECT * FROM facts ORDER BY valid_from DESC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM facts WHERE archived_at IS NULL "
                "ORDER BY valid_from DESC"
            ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def get_fact(self, fact_id: int) -> Fact | None:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return _row_to_fact(row) if row else None

    def update_fact(
        self,
        fact_id: int,
        *,
        predicate: str | None = None,
        object: str | None = None,
        confidence: float | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if predicate is not None:
            sets.append("predicate = ?"); params.append(predicate)
        if object is not None:
            sets.append("object = ?"); params.append(object)
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError("confidence must be 0..1")
            sets.append("confidence = ?"); params.append(confidence)
        if not sets:
            return
        params.append(fact_id)
        self.conn.execute(
            f"UPDATE facts SET {', '.join(sets)} WHERE id = ?", tuple(params),
        )

    def archive_fact(self, fact_id: int) -> None:
        self.conn.execute(
            "UPDATE facts SET archived_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND archived_at IS NULL",
            (fact_id,),
        )

    def restore_fact(self, fact_id: int) -> None:
        self.conn.execute(
            "UPDATE facts SET archived_at = NULL WHERE id = ?", (fact_id,)
        )

    # ── preferences ────────────────────────────────────────────────────

    def upsert_preference(
        self, person_id: int | None, domain: str, value: str
    ) -> None:
        # COALESCE in the unique index handles NULL person_id.
        self.conn.execute(
            "INSERT INTO preferences (person_id, domain, value) VALUES (?, ?, ?) "
            "ON CONFLICT(COALESCE(person_id, -1), domain, value) DO UPDATE SET "
            "last_seen_at = CURRENT_TIMESTAMP",
            (person_id, domain, value),
        )

    def list_preferences(self, person_id: int | None) -> list[Preference]:
        if person_id is None:
            rows = self.conn.execute(
                "SELECT * FROM preferences WHERE person_id IS NULL "
                "ORDER BY domain, value"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM preferences WHERE person_id = ? "
                "ORDER BY domain, value",
                (person_id,),
            ).fetchall()
        return [_row_to_preference(r) for r in rows]

    # ── notes ──────────────────────────────────────────────────────────

    def add_note(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        source_session_id: int | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO notes (content, tags, source_session_id) VALUES (?, ?, ?)",
            (content, json.dumps(tags or [], ensure_ascii=False), source_session_id),
        )
        return int(cur.lastrowid or 0)

    def update_note(
        self,
        note_id: int,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if content is not None:
            sets.append("content = ?")
            params.append(content)
        if tags is not None:
            sets.append("tags = ?")
            params.append(json.dumps(tags, ensure_ascii=False))
        if not sets:
            return
        sets.append("updated_at = CURRENT_TIMESTAMP")
        params.append(note_id)
        self.conn.execute(
            f"UPDATE notes SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )

    def list_notes(self, *, include_archived: bool = False) -> list[Note]:
        if include_archived:
            rows = self.conn.execute(
                "SELECT * FROM notes ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM notes WHERE archived_at IS NULL "
                "ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [_row_to_note(r) for r in rows]

    def archive_note(self, note_id: int) -> None:
        self.conn.execute(
            "UPDATE notes SET archived_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND archived_at IS NULL",
            (note_id,),
        )

    def search_notes_by_keyword(
        self, keyword: str, *, limit: int = 5
    ) -> list[Note]:
        """Naïve LIKE-based note search. Embeddings come in a later PR."""
        if not keyword.strip():
            return []
        pattern = f"%{keyword.strip()}%"
        rows = self.conn.execute(
            "SELECT * FROM notes "
            "WHERE archived_at IS NULL AND content LIKE ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (pattern, limit),
        ).fetchall()
        return [_row_to_note(r) for r in rows]

    # ── attachments ────────────────────────────────────────────────────

    def add_attachment(
        self,
        session_id: int,
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
            "(session_id, sha256, mime, ext, byte_size, path, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, sha256, mime, ext, byte_size, path, description),
        )
        return int(cur.lastrowid or 0)

    def list_attachments(self, session_id: int) -> list[Attachment]:
        rows = self.conn.execute(
            "SELECT * FROM attachments WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [_row_to_attachment(r) for r in rows]

    def find_attachment_by_sha256(
        self, session_id: int, sha256: str
    ) -> Attachment | None:
        row = self.conn.execute(
            "SELECT * FROM attachments WHERE session_id = ? AND sha256 = ?",
            (session_id, sha256),
        ).fetchone()
        return _row_to_attachment(row) if row else None


# ── helpers ────────────────────────────────────────────────────────────────


def _row_to_person(r: sqlite3.Row) -> Person:
    archived_at: str | None = None
    try:
        archived_at = r["archived_at"]
    except (IndexError, KeyError):
        pass
    return Person(
        id=r["id"],
        name=r["name"],
        relation=r["relation"],
        birthday=r["birthday"],
        preferences=json.loads(r["preferences_json"]),
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        archived_at=archived_at,
    )


def _row_to_session(r: sqlite3.Row) -> Session:
    return Session(
        id=r["id"],
        started_at=r["started_at"],
        last_active_at=r["last_active_at"],
        title=r["title"],
        summary=r["summary"],
        archived_at=r["archived_at"],
    )


def _row_to_message(r: sqlite3.Row) -> Message:
    return Message(
        id=r["id"],
        session_id=r["session_id"],
        role=r["role"],
        content=r["content"],
        ts=r["ts"],
    )


def _row_to_event(r: sqlite3.Row) -> Event:
    archived_at: str | None = None
    try:
        archived_at = r["archived_at"]
    except (IndexError, KeyError):
        pass
    return Event(
        id=r["id"],
        person_id=r["person_id"],
        type=r["type"],
        title=r["title"],
        when_at=r["when_at"],
        recurrence=r["recurrence"],
        source=r["source"],
        status=r["status"],
        archived_at=archived_at,
    )


def _row_to_fact(r: sqlite3.Row) -> Fact:
    return Fact(
        id=r["id"],
        subject_person_id=r["subject_person_id"],
        predicate=r["predicate"],
        object=r["object"],
        confidence=r["confidence"],
        source_session_id=r["source_session_id"],
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


def _row_to_note(r: sqlite3.Row) -> Note:
    try:
        tags = json.loads(r["tags"])
        if not isinstance(tags, list):
            tags = []
    except (json.JSONDecodeError, TypeError):
        tags = []
    return Note(
        id=r["id"],
        content=r["content"],
        tags=tags,
        source_session_id=r["source_session_id"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        archived_at=r["archived_at"],
    )


def _row_to_attachment(r: sqlite3.Row) -> Attachment:
    return Attachment(
        id=r["id"],
        session_id=r["session_id"],
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
    """Return recency_weight if *when_at_iso* is within recency_days, else 1.0."""
    try:
        ts_str = when_at_iso.replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            ts = datetime.fromisoformat(ts_str[:19])
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
