"""Tests for Fact CRUD in MemoryStore."""

from __future__ import annotations

import pytest

from apps.memory import Fact, MemoryStore


def test_add_and_list_active_facts(store: MemoryStore) -> None:
    pid = store.add_person("Alice")
    fid = store.add_fact(pid, "likes", "coffee", confidence=0.9)
    facts = store.list_active_facts(pid)
    assert len(facts) == 1
    assert facts[0].id == fid
    assert facts[0].predicate == "likes"
    assert facts[0].object == "coffee"
    assert facts[0].confidence == pytest.approx(0.9)
    assert facts[0].archived_at is None


def test_add_fact_with_source_session(store: MemoryStore) -> None:
    pid = store.add_person("Bob")
    sid = store.add_session(title="conversation")
    fid = store.add_fact(pid, "prefers", "tea", confidence=0.8, source_session_id=sid)
    facts = store.list_active_facts(pid)
    assert len(facts) == 1
    assert facts[0].source_session_id == sid


def test_archive_fact_removes_from_active(store: MemoryStore) -> None:
    pid = store.add_person("Carol")
    fid1 = store.add_fact(pid, "likes", "cats", confidence=0.95)
    fid2 = store.add_fact(pid, "dislikes", "dogs", confidence=0.7)

    store.archive_fact(fid1)

    active = store.list_active_facts(pid)
    active_ids = [f.id for f in active]
    assert fid1 not in active_ids
    assert fid2 in active_ids


def test_archive_fact_sets_archived_at(store: MemoryStore) -> None:
    pid = store.add_person("Dave")
    fid = store.add_fact(pid, "owns", "car", confidence=0.85)
    store.archive_fact(fid)

    row = store.conn.execute(
        "SELECT archived_at FROM facts WHERE id = ?", (fid,)
    ).fetchone()
    assert row is not None
    assert row["archived_at"] is not None


def test_archive_fact_idempotent(store: MemoryStore) -> None:
    pid = store.add_person("Eve")
    fid = store.add_fact(pid, "x", "y", confidence=0.5)
    store.archive_fact(fid)
    # Second archive should not raise
    store.archive_fact(fid)


def test_list_active_facts_ordered_by_valid_from_desc(store: MemoryStore) -> None:
    pid = store.add_person("Frank")
    # Insert with explicit different timestamps to guarantee ordering
    store.conn.execute(
        "INSERT INTO facts (subject_person_id, predicate, object, confidence, valid_from) "
        "VALUES (?, ?, ?, ?, ?)",
        (pid, "has", "cat", 0.8, "2026-01-01T00:00:00"),
    )
    fid1 = store.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    store.conn.execute(
        "INSERT INTO facts (subject_person_id, predicate, object, confidence, valid_from) "
        "VALUES (?, ?, ?, ?, ?)",
        (pid, "has", "dog", 0.9, "2026-04-30T00:00:00"),
    )
    fid2 = store.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    facts = store.list_active_facts(pid)
    # Most recent first
    assert facts[0].id == fid2
    assert facts[1].id == fid1


def test_confidence_boundary_zero(store: MemoryStore) -> None:
    pid = store.add_person("Grace")
    fid = store.add_fact(pid, "maybe", "nothing", confidence=0.0)
    facts = store.list_active_facts(pid)
    assert len(facts) == 1
    assert facts[0].confidence == pytest.approx(0.0)


def test_confidence_boundary_one(store: MemoryStore) -> None:
    pid = store.add_person("Hank")
    fid = store.add_fact(pid, "definitely", "something", confidence=1.0)
    facts = store.list_active_facts(pid)
    assert len(facts) == 1
    assert facts[0].confidence == pytest.approx(1.0)


def test_confidence_out_of_range_raises(store: MemoryStore) -> None:
    pid = store.add_person("Ivy")
    with pytest.raises(ValueError, match="confidence"):
        store.add_fact(pid, "x", "y", confidence=1.1)
    with pytest.raises(ValueError, match="confidence"):
        store.add_fact(pid, "x", "y", confidence=-0.1)


def test_list_active_facts_empty(store: MemoryStore) -> None:
    pid = store.add_person("Jack")
    assert store.list_active_facts(pid) == []


def test_fact_is_frozen_dataclass(store: MemoryStore) -> None:
    pid = store.add_person("Kate")
    fid = store.add_fact(pid, "p", "o", confidence=0.5)
    facts = store.list_active_facts(pid)
    assert isinstance(facts[0], Fact)
    with pytest.raises(Exception):
        facts[0].predicate = "mutated"  # type: ignore[misc]
