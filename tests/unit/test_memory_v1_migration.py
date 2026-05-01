"""Tests for v1 → v2 schema migration: existing episodes-based DB is
backed up and a fresh v2 store is created in its place."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

from apps.memory import MemoryStore, SCHEMA_VERSION


def _make_v1_db(path: Path) -> None:
    """Hand-craft a minimal v1 schema (just episodes table) so MemoryStore
    treats this file as v1 on open."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute(
        "CREATE TABLE episodes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "when_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
        "summary TEXT, "
        "primary_channel TEXT NOT NULL"
        ")"
    )
    conn.execute(
        "INSERT INTO episodes (summary, primary_channel) VALUES (?, ?)",
        ("legacy v1 row", "text"),
    )
    conn.close()


def test_v1_db_is_backed_up_on_first_open(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _make_v1_db(db)

    store = MemoryStore(db)
    try:
        # The fresh DB should have v2 tables and no leftover v1 data.
        assert (
            store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            ).fetchone()
            is not None
        )
        assert store.list_recent_sessions() == []

        # user_version stamped to v2.
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        store.close()

    # Backup file exists alongside the new DB.
    backups = list(tmp_path.glob("db.sqlite.bak-*"))
    assert len(backups) == 1, f"expected one backup file, got {backups}"

    # Backup retains the v1 row.
    bk = sqlite3.connect(backups[0])
    rows = bk.execute("SELECT summary FROM episodes").fetchall()
    bk.close()
    assert rows[0][0] == "legacy v1 row"


def test_v2_db_open_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"

    s1 = MemoryStore(db)
    sid = s1.add_session(title="first")
    s1.close()

    # Reopen — must not back up or recreate.
    s2 = MemoryStore(db)
    try:
        sessions = s2.list_recent_sessions()
        assert any(s.id == sid for s in sessions)
    finally:
        s2.close()

    backups = list(tmp_path.glob("db.sqlite.bak-*"))
    assert backups == []


def test_fresh_db_is_v2_immediately(tmp_path: Path) -> None:
    db = tmp_path / "fresh.sqlite"
    s = MemoryStore(db)
    try:
        assert s.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        s.close()
