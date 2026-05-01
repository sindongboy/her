"""Schema-level sanity checks. These guard against silent regressions in the
SQLite + sqlite-vec setup (extension load, vec0 virtual table, FK pragma)."""

from __future__ import annotations

import struct

from apps.memory import EMBED_DIM, MemoryStore


def test_foreign_keys_pragma_is_on(store: MemoryStore) -> None:
    fk = store.conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


def test_vec0_virtual_table_present(store: MemoryStore) -> None:
    row = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE name='vec_episodes'"
    ).fetchone()
    assert row is not None
    assert row["name"] == "vec_episodes"


def test_vec0_insert_and_match_round_trip(store: MemoryStore) -> None:
    store.conn.execute(
        "INSERT INTO episodes (when_at, summary, primary_channel) VALUES (?, ?, ?)",
        ("2026-04-30T00:00:00", "fixture", "text"),
    )
    eid = store.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    vec = [0.1] * EMBED_DIM
    blob = struct.pack(f"{EMBED_DIM}f", *vec)
    store.conn.execute(
        "INSERT INTO vec_episodes (episode_id, embedding) VALUES (?, ?)",
        (eid, blob),
    )

    rows = store.conn.execute(
        "SELECT episode_id FROM vec_episodes "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
        (blob,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["episode_id"] == eid


def test_episode_primary_channel_check_constraint(store: MemoryStore) -> None:
    import sqlite3

    try:
        store.conn.execute(
            "INSERT INTO episodes (when_at, summary, primary_channel) VALUES (?, ?, ?)",
            ("2026-04-30T00:00:00", "bad", "fax"),
        )
    except sqlite3.IntegrityError:
        return
    raise AssertionError("CHECK constraint did not reject invalid primary_channel")
