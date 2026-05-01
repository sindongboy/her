"""Schema-level sanity checks for the v2 layout — guards extension load,
vec0 virtual tables, FK pragma, message role check, and version stamp."""

from __future__ import annotations

import sqlite3
import struct

import pytest

from apps.memory import EMBED_DIM, SCHEMA_VERSION, MemoryStore


def test_schema_version_pragma(store: MemoryStore) -> None:
    v = store.conn.execute("PRAGMA user_version").fetchone()[0]
    assert v == SCHEMA_VERSION == 2


def test_foreign_keys_pragma_is_on(store: MemoryStore) -> None:
    fk = store.conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


def test_required_tables_present(store: MemoryStore) -> None:
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    names = {r["name"] for r in rows}
    for required in (
        "people",
        "sessions",
        "messages",
        "events",
        "facts",
        "preferences",
        "notes",
        "attachments",
        "session_embedding_meta",
        "message_embedding_meta",
    ):
        assert required in names, f"missing table: {required}"


def test_vec_sessions_virtual_table_present(store: MemoryStore) -> None:
    row = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE name='vec_sessions'"
    ).fetchone()
    assert row is not None


def test_vec_messages_virtual_table_present(store: MemoryStore) -> None:
    row = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE name='vec_messages'"
    ).fetchone()
    assert row is not None


def test_vec_sessions_insert_and_match(store: MemoryStore) -> None:
    sid = store.add_session(summary="fixture")
    vec = [0.1] * EMBED_DIM
    blob = struct.pack(f"{EMBED_DIM}f", *vec)
    store.conn.execute(
        "INSERT INTO vec_sessions (session_id, embedding) VALUES (?, ?)",
        (sid, blob),
    )
    rows = store.conn.execute(
        "SELECT session_id FROM vec_sessions WHERE embedding MATCH ? "
        "ORDER BY distance LIMIT 1",
        (blob,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["session_id"] == sid


def test_message_role_check_constraint(store: MemoryStore) -> None:
    sid = store.add_session()
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'ghost', 'x')",
            (sid,),
        )
